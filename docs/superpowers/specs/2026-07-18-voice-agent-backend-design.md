# Amplifier Voice Agent — Backend Design

**Date:** 2026-07-18
**Status:** Draft for review
**Context:** Abridge x Anthropic x Lightspeed hackathon ("The Future of Agentic AI in Healthcare"). 3-minute demo. Everything built must be open source.

## Goal

An agent system that adds vocal biomarker intelligence to the clinical visit workflow. It processes doctor–patient conversation audio through the Amplifier Health API in near real time, reasons over the results with Claude, and maintains a longitudinal patient record that surfaces conditions (depression, anxiety, sleep disorders, PCOS-related signals) months before they appear in transcripts or ICD-10 coding.

The demo story: a patient comes in for an ear infection and a wellness visit, says she's fine. Her voice says otherwise. Over successive visits, the system's early flags precede her actual PCOS coding (5 visits later) and depression/anxiety diagnosis (10 visits later).

**Positioning:** the pipeline mirrors Abridge's own three-stage encounter workflow (before / during / after visit). Abridge captures what the patient says; Amplifier captures how they say it.

| Abridge stage | Abridge does | Amplifier layer adds |
|---|---|---|
| Before visit | History, open care gaps | Vocal-trend summary from prior visits, voice-sourced "topics to discuss" |
| During visit | Ambient transcription, note capture | Vocal biomarker signals streaming from the same audio, discordance alerts |
| After visit | Drafted specialty note | Vocal findings note section, screener recommendations, chart update draft, next-visit topics |

## Principles

- **Patient-agnostic.** No story hardcoded. The app runs any patient directory dropped into the store. The demo patient is just data.
- **Real pipeline, no fakes in the architecture.** The agent genuinely chunks audio, transcribes, calls the Amplifier API, polls jobs, and reasons live. The stage demo differs only in inputs (pre-recorded audio, pre-warmed store), never in code path.
- **Backend first.** Frontend UX is deferred; the backend exposes a stable SSE event contract any UI can attach to.
- **Open source.** MIT license. Dependencies limited to open components (FastAPI, faster-whisper, Anthropic SDK, httpx).
- **Draft, never write.** Agents propose chart updates; a clinician approves. Nothing writes to a record autonomously.

## Architecture

### Visit lifecycle (the core loop)

```
                    ┌─────────────────────────────────────────────┐
                    │                PATIENT STORE                │
                    │  notes, vocal results, ICD-10 history,      │
                    │  topics-to-discuss, pre-visit briefs        │
                    └──────┬──────────────────────────▲───────────┘
                           │ read                     │ write (drafts)
                 ┌─────────▼─────────┐      ┌─────────┴─────────┐
                 │  1. PRE-VISIT     │      │  3. POST-VISIT    │
                 │     AGENT         │      │     AGENT         │
                 │  brief + topics   │      │  summary, note    │
                 └─────────┬─────────┘      │  section, chart   │
                           │ context        │  draft, topics    │
                 ┌─────────▼─────────┐      └─────────▲─────────┘
                 │  2. VISIT SESSION │               │
                 │  audio → chunks → │  results +    │
                 │  whisper + API →  │  observations │
                 │  reasoner         ├───────────────┘
                 └───────────────────┘
```

Stage 3's output is stage 1's input for the next visit. That closed loop is what produces the longitudinal early-detection story over successive appointments.

### Components

**1. Patient store** (`store/`)
Plain JSON files per patient — transparent, git-friendly, no database.

```
data/patients/<patient_id>/
  patient.json            # alias, age, sex (no real PHI)
  visits.json             # ordered index: date, reason, ICD-10 codes, status
  visits/<n>/
    pre_visit_brief.json  # brief + topics (written at end of visit n-1)
    transcript.json       # whisper output, per chunk, speaker-tagged where possible
    chunks/<k>.json       # Amplifier job result per chunk (real API response shape)
    observations.json     # reasoner outputs during the visit
    summary.json          # post-visit agent output: note section, chart draft, topics
```

**2. Pre-visit agent** (Claude, live)
Input: patient ID. Reads prior visits' summaries, vocal results, ICD-10 history from the store. Output: pre-visit brief — vocal trends across visits, discordance history, topics to discuss. Streams to the event bus and persists to the store.

**3. Visit session** (the during-visit pipeline)

- **AudioSource** — interface with two implementations: `FileReplaySource` (streams a pre-recorded file in near real time, with a speed multiplier for testing and demo pacing) and, later, `MicSource`. The rest of the pipeline cannot tell them apart.
- **Chunker** — cuts the stream at ~30–60 s boundaries (API floor is 15 s, ceiling 20 min; 32 MB max). Emits chunk files.
- **Transcriber** — faster-whisper (local, open source) per chunk. Model size configurable (`base`/`small` default for speed).
- **Amplifier client** — per chunk: `POST /v2/audio/uploads`, then `POST /v2/models/haven/analyze` (behavioral-health model: mood-disruption, anxiety, stress, hypervigilance, attention-dysregulation, fatigue). Auth via `X-Account-ID` + `X-API-Key`. **Async by design:** analyze is rate-limited at 5 req/min and jobs take real time, so the client keeps a job queue, polls `GET /v2/jobs/{id}`, respects rate limits, and delivers results out of order onto a results queue. The pipeline never blocks recording on results.
- **Visit reasoner** (Claude, live, streaming) — consumes each completed chunk result with the chunk transcript, cumulative signal state, and the pre-visit brief in context. Has one tool: `read_chart(patient_id)` for full history on demand. Emits 1–2 sentence observations per milestone — this is where discordance surfaces ("transcript: 'I'm okay, just tired'; voice: mood-disruption elevated for the third consecutive chunk; no mood history on chart").

**4. Post-visit agent** (Claude, live)
Input: pre-visit brief, all chunk results, full transcript, reasoner observations. Output (structured):
- visit summary with a **vocal findings** section (note-ready, Abridge-style)
- **chart update draft** (proposed problem-list/screener items, e.g. PHQ-9, GAD-7 — for clinician approval)
- **topics to discuss** for the next visit
Persists everything to the store.

**5. Longitudinal analyst** (Claude, live)
On demand (demo Act 2): reads every visit's vocal results and the ICD-10 timeline; produces the early-detection narrative plus structured deltas per condition — visit where voice first flagged vs. visit where first coded. This drives the timeline visualization later.

**6. Event bus + API** (FastAPI)
- `POST /patients/{id}/visits/start` — kicks off pre-visit agent, then visit session
- `POST /visits/{id}/end` — flush remaining jobs, run post-visit agent
- `POST /patients/{id}/longitudinal` — run longitudinal analyst
- `GET /events` — SSE stream. Event types: `pre_visit_brief`, `chunk_created`, `transcript`, `api_job_created`, `api_job_result`, `agent_token` (streamed Claude text), `observation`, `visit_summary`, `chart_draft`, `topics`, `longitudinal_delta`, `error`. This contract is the frontend's only dependency on the backend.

### Claude layer

- Anthropic Python SDK, streaming everywhere. Default model `claude-sonnet-5` for the per-chunk reasoner (fast, short outputs); post-visit and longitudinal agents configurable up to Opus. Consult current model guidance at build time.
- **Rehearsal cache:** every live Claude call is recorded keyed by (agent, visit, milestone). On stage, a call that errors or exceeds ~6 s to first token falls back to the recording silently. Live when the network cooperates (nearly always); never dead air when it doesn't.
- Mock mode for UI development (canned responses, zero tokens).

### Demo mode vs. real mode

Same binary, one config flag set:
- **Real:** mic or full-length file, real-time chunk cadence, live Amplifier jobs.
- **Demo:** shorter pre-recorded file at accelerated cadence; the patient store pre-warmed with prior visits' results from earlier *real* runs; Amplifier jobs for the live visit may use results cached from a rehearsal run of the same audio (config: `amplifier.cache = warm`), because analyze jobs take minutes and the demo has three. Claude stays live.

Honest answer for judges: "Vocal analysis results were computed by our production API ahead of time because jobs take minutes; every word of agent reasoning you watched was generated live, and here is the same pipeline running end-to-end for real."

## Testing

- **Fast mode:** whole visit lifecycle at high speed with a synthetic patient bundle and mock Claude — full-flow verification in seconds.
- **Synthetic patient generator:** produces a bundle in the exact store shape until real demo-patient data (audio + ICD-10 history) arrives.
- Unit coverage on the chunker (boundary/min-length rules), Amplifier client (rate limiting, out-of-order results, job retry), and store round-trips.

## Non-goals

- No EHR integration, no real PHI, no authentication story beyond API keys in env vars.
- No frontend polish in this phase — only the SSE contract.
- No autonomous chart writes, ever.

## Open items

- Demo patient data delivery: audio files + visit/ICD-10 history from Vijay (format: the store shape above; `onset_analysis/*.md` files provide the ICD-10 backbone).
- Chunk length final call (30 s vs 60 s) after measuring real job latency against the 5/min rate limit.
- Frontend design (Abridge-style plug-in look) — separate spec later.
