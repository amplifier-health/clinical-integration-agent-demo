# Clinical Integration Agent — Architecture & Flow Spec (v0)

Strawman for the Abridge-integration demo. Decisions marked **[REC]** are my
recommendation; **[OPEN]** needs a call. Built to be shared with the backend owner.

---

## 1. What we're building

A voice-biomarker **integration agent** that sits between a live ambient recording
(Abridge-style) and the clinician-facing UI. As the encounter is recorded, the agent
turns rolling audio into biomarker signals, reasons over them together with the
patient chart and live transcript, and pushes **typed events** to the UI that populate
Abridge's existing surfaces: *Suggested discussion topics* (mobile), the *Note*,
*Flowsheets*, and the *Abridge AI* chat panel. A separate **Reasoning tab** renders the
agent's internal work live.

The agent is a **client of** the Amplifier voice API — it is not part of it. That
boundary is deliberate: the voice API stays a clean product, the agent is the
integration layer.

---

## 2. The core design idea: one event bus, many views

There is exactly **one outbound stream** of typed events from the agent to the UI.
Every Abridge surface — and the Reasoning tab — is just a **filtered view** of that
stream. This is what makes the demo coherent and avoids "intermittent API calls."

```
                 ┌──────────────── Agent service (separate backend) ─────────────────┐
  audio frames   │  ingest → rolling 30s/15s buffer → Amplifier Voice API (per chunk) │
  ── & ──────────▶  ▲                                          │                       │
  transcript     │  │        signal buffer  ◀─────────────────┘                       │
  (WebSocket in) │  │             │                                                    │
                 │  │             ▼   every N chunks / threshold cross                 │
                 │  │      REASONING CYCLE (LLM + chart + transcript + trend)          │
                 │  │             │                                                    │
                 │  └──────  typed events  ──────────────────────────────────┐        │
                 └───────────────────────────────────────────────────────────┼────────┘
                                                                              │ (WebSocket out)
   ┌──────────────────────────────── UI subscribers ─────────────────────────▼─────────┐
   │ mobile: Suggested topics │ Note addenda │ Flowsheets │ Abridge AI chat │ Reasoning │
   └────────────────────────────────────────────────────────────────────────────────────┘
```

### Event types (agent → UI)
| event | payload | UI target |
|---|---|---|
| `biomarker.result` | chunk_id, t_start, signals[{name,score,tier}], audio_quality | Reasoning tab; trend store |
| `topic.suggested` | id, title, rationale, source(`voice`/`chart`), confidence, trend[] | mobile Suggested topics |
| `flowsheet.update` | section, row, value, provenance{quote,t} | Flowsheets panel |
| `reasoning.step` | phase, message, tool_call?, inputs, outputs | Reasoning tab |
| `note.addendum` | section, text, citations[] | Note (post-visit) |
| `followup.item` | text, why, discussed(bool) | Note / next-visit list |
| `trial.match` | nct_id, title, why_relevant, eligibility_hint | Abridge AI panel |

### Client → agent
`audio.frame` (binary, ~1–5s PCM) and `transcript.token` (text + t). The agent owns
buffering and the 30s/15s sliding window — **[REC]** so the "agent does everything"
thesis holds and the UI stays dumb.

**Transport [REC]:** a single **WebSocket** (bidirectional). Fallback: SSE (server→client)
+ chunked POST (client→server) if the backend stack prefers it.

---

## 3. Real-time loop (during recording)

1. **Ingest** audio frames + transcript tokens over WS.
2. **Buffer**: maintain a rolling 30s window, emit a chunk every 15s (50% overlap).
3. **Score**: POST each chunk to `POST /v2/models/{model}/analyze`; poll the job;
   append `{t, signals}` to the per-signal trend buffer. Emit `biomarker.result`.
4. **Trigger a reasoning cycle** when either: N chunks accumulated (**[REC]** N=4 ≈ 1 min)
   **or** a signal crosses a tier boundary (e.g., → `consider`/`moderate`).
5. **Reasoning cycle** (LLM): inputs = recent signal trend + running transcript + patient
   chart. Decides whether anything is worth surfacing. Guardrails:
   - **Dedupe** against topics already suggested and against the chart problem list.
   - **Suppress if already discussed** — scan the transcript for the topic.
   - **Confidence + tier floor** — only surface `consider`+ with a coherent rationale.
   - Emit `topic.suggested` (and optionally `flowsheet.update`) + `reasoning.step`s.

This is what fills the mobile **Suggested discussion topics** card live, alongside the
chart-derived items already there (inhaler usage, smoking cessation).

---

## 4. Post-visit pass (on "stop")

Heavier, mostly non-latency-critical reasoning:
1. **Aggregate** the full-visit biomarker trajectory (peak tier + mean prob per signal).
2. **Reconcile** signals × transcript × chart → three buckets:
   - **Confirmed & discussed** → `note.addendum` (structured findings, flowsheet rows).
   - **Discussed, add to follow-up** → `followup.item(discussed=true)`.
   - **Flagged by voice, NOT discussed** → `followup.item(discussed=false)` = **the gap list**
     (the differentiated output — "not the focus today, revisit next time").
3. **Clinical trials**: call the Clinical Trials MCP with the patient's conditions **plus**
   the gap-list signals → `trial.match` events into the Abridge AI panel. The point is
   trials for things that were *not* the visit's focus (what the doctor might miss).
4. **Note integration**: append structured addenda with **provenance** (voice quote +
   timestamp), mirroring Abridge's "Ambient Source" pattern.

---

## 5. Demo timeline (3 minutes — must be tight)

| t | on screen | live vs baked |
|---|---|---|
| 0:00–0:20 | Patient chart loads; pre-visit "watch-for" briefing | baked |
| 0:20–1:30 | Recording runs; transcript scrolls; 1–2 Suggested topics appear; a Flowsheet row auto-fills; Reasoning tab shows the loop | **replayed biomarkers**, live LLM reasoning |
| 1:30–1:40 | "Stop recording" | — |
| 1:40–2:40 | Post-visit: note addenda write in, gap list appears, follow-up items | live LLM over baked signals |
| 2:40–3:00 | Clinical-trial matches surface in Abridge AI panel | live MCP call |

**[REC] Pre-compute the biomarker results on the demo audio and replay them on a
timeline synced to playback.** Reason: each voice-API call is ~15–25s — you cannot wait
for live scoring inside a 3-minute window, and stage Wi-Fi/API flakiness is a needless
risk. Keep the **LLM reasoning and the MCP trial call live** (they're fast and are the
"wow"). The agent code path is identical; only the biomarker source is a replay adapter.

---

## 6. Integration ideas beyond the three surfaces

1. **Flowsheet auto-population** (screenshot 1): map biomarker outputs to structured
   flowsheet rows with the ambient-source waveform icon + the triggering quote. Highest
   visual payoff and it matches an existing Abridge element exactly.
2. **Provenance links**: every suggestion carries the audio moment + signal that caused
   it ("tap to see why") — mirrors the "Ambient Source" quote card.
3. **Explainable chat**: wire the reasoning into the **Abridge AI** panel so the clinician
   can ask "why did you flag airway obstruction?" and get the biomarker+chart rationale.
4. **Pre-visit briefing**: agent pre-computes "watch-for" items from the chart + last
   visit's voice-flagged-but-undiscussed gap list — closes the longitudinal loop.
5. **Order/action suggestions**: elevated-BP signal → BP recheck; airway-obstruction →
   spirometry/auscultation prompt. Actionable, not just informational.
6. **End-of-visit safety net**: if a high-tier signal never appears in the transcript by
   "stop," escalate to a "before you close, consider…" prompt.
7. **Longitudinal voice trend** as a chart element (reuse the trajectory dashboard).

---

## 7. Open decisions
- **[DECIDED] Biomarkers are replayed, not live.** Precomputed result objects are replayed
  on a time-compressed timeline; the UI/Reasoning-tab **visually mocks** the API call
  (a `reasoning.step` "POST /v2/models/{model}/analyze → 200 (replayed)"). The demo
  "speeds through" the encounter at an adjustable multiplier. The real agent swaps the
  replay adapter for live calls with zero UI change.
- **[DECIDED] Scenario/model — women's health (`aria`).** Default demo: a fictional 31F
  seen for an anxiety follow-up while the voice model's **elevated-androgens** signal rises
  across the visit and becomes the post-visit **PCOS gap** (workup + trials). A respiratory
  (`breath`) scenario ships as an alternate. Both synthetic.
- **[OPEN] Transcript** — pre-scripted timed transcript (recommended) vs live ASR.
- **[OPEN] One agent w/ tools vs multi-agent** (monitor / scribe / research). Multi-agent
  reads better in the Reasoning tab but is more to build.
- **[OPEN] Reasoning-tab fidelity** — full trace vs curated event log (recommend curated).
- **[OPEN] Chart source** — static JSON fixture for the demo patient (recommended).

---

## 8. Contract the two workstreams share

Backend and UI only need to agree on the **event schema in §2**. Everything else can move
independently: the backend can swap replay↔live biomarkers, single↔multi agent, without UI
changes; the UI can restyle any surface as long as it subscribes to the same event types.
Ship a `mock-agent` that emits the scripted event sequence over WS so the UI can be built
before the real agent is done.
