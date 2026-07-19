# Clinical Integration Agent — an ambient voice-biomarker layer for the clinical visit

Built for the **Abridge × Anthropic × Lightspeed hackathon** ("The Future of Agentic AI in Healthcare").

Ambient scribes capture **what** the patient says. This agent captures **how** they say it — it runs the visit audio through the [Amplifier Health voice-biomarker API](https://docs.amplifierhealth.com) and reasons over the signals with Claude, live, to surface conditions the words alone miss: depression, anxiety, PCOS/androgen changes, iron deficiency, fatigue, and more.

It is designed as a **plug-in layer for an existing ambient scribe**, not a competitor to one. The scribe already has the microphone and the transcript; this agent adds a parallel signal — the voice — and returns typed events the host UI can render on its existing surfaces (voice signals on the capture screen, the clinical note, an orders tab, a behind-the-scenes reasoning view).

> **Not a diagnostic device.** The agent never diagnoses. It describes vocal-biomarker signals, suggests validated screeners (PHQ-9, GAD-7), and drafts chart updates for a clinician to approve. The clinician is always the decision-maker.

---

## The idea in one picture

The agent works across the three stages of an encounter, and closes a longitudinal loop across visits:

| Stage | What the agent does |
|---|---|
| **Before visit** | Reads the patient's prior chart + prior vocal results and produces a "watch-for" brief — vocal trends, discordances, topics to raise. |
| **During visit** | Scores the audio in ~30s chunks through Amplifier, reasons over each result with the transcript, and streams observations — especially **discordance** (patient says "I'm fine"; the voice says otherwise). Alert-fatigue controls decide what's worth interrupting the visit for. |
| **After visit** | Drafts a SOAP note, a vocal-findings section, a chart-update draft (for approval), and next-visit topics. Opt-in extras (clinician-configurable): suggested screener **orders** (GAD-7/PHQ-9), matched **clinical trials** for a care gap, and — at "detailed" depth — **PubMed citations** for the main finding. |

**The differentiated output — longitudinal early detection.** Across successive visits, the agent computes, per condition, the gap between when the **voice first flagged** a signal and when the condition was **first ICD-10 coded**. On the real demo patient: the voice flagged anxiety ~68 days and PCOS ~119 days before the chart coded them, with mood disruption still rising and uncoded.

---

## Architecture

```
                        ┌──────────────────────────────────────────────┐
                        │                 PATIENT STORE (JSON)          │
                        │  chart · prior signals · notes · artifacts    │
                        └──────┬───────────────────────────▲───────────┘
                    read (prior visits only)         write (drafts)
                        ┌──────▼───────────┐      ┌──────────┴────────┐
                        │  1. PRE-VISIT    │      │  3. POST-VISIT    │
                        │     AGENT        │      │  summary · SOAP   │
                        │  brief + topics  │      │  note · gaps ·    │
                        └──────┬───────────┘      │  trials · draft   │
                               │ brief            └──────────▲────────┘
                        ┌──────▼──────────────────────────────┴────────┐
                        │  2. VISIT SESSION (one shared Claude memory)  │
                        │  audio/replay → chunk → Amplifier signals →   │
                        │  live reasoner (alert-gated) per chunk        │
                        └───────────────────────────────────────────────┘
                                          │
                                 typed event stream (SSE)
                                          ▼
        mobile voice-signals · clinical note (orders / literature) · reasoning view
```

Every stage emits **typed, versioned events** onto one bus, streamed over SSE. Each Abridge-style surface is just a filtered view of that one stream. The reasoner and the post-visit agent share **one Claude conversation** for the whole visit, so the final analysis is done by an agent that watched the entire encounter, and reasons over the *full* voice-biomarker object (condition signals + 18 wellness dimensions + speech prosody) — translating it into qualitative clinical language, never raw scores.

- **Backend:** Python, FastAPI, httpx, Anthropic SDK (Claude Opus 4.8), pydantic. faster-whisper + pydub for the live-audio path (ffmpeg required for non-WAV).
- **Frontend:** a self-contained Abridge-style clinician UI (`mock_agent/viewer.html`) that speaks the backend's event contract directly — no adapter.

---

## The output contract (what a plugin integrates against)

The event stream is a **first-class, versioned, enforced contract** — the thing another product builds against.

- Every event carries an envelope: `type`, `contract_version`, `phase` (`pre_visit`/`live`/`post_visit`/`longitudinal`/`telemetry`/`reasoning`/`lifecycle`), `session_id`, a monotonic `seq`, and `ts`. Payload fields sit alongside (flat wire format).
- Every payload is validated at the single emit choke point against a pydantic model (`clinical_agent/contract.py`). Validation is **fail-soft at runtime** (a drift is logged and flagged `contract_error`, never aborts a live visit) and **strict in tests**.
- The machine-readable spec is generated from the models to `docs/contract/contract.json` and served live at `GET /contract`. A test fails if the committed spec ever drifts from the code.

**Live vs. final are the same types, two access modes.** The stream delivers deltas (`finding`/observation updates as the visit unfolds); `GET /patients/{id}/visits/{n}/result` returns the **folded snapshot** — the same clinical outputs reduced to final state — for late joiners, reconnects, or EHR write-back.

Key clinical event types (full schema in `docs/contract/contract.json`):

| Event | Meaning |
|---|---|
| `pre_visit_brief` | Watch-for brief from prior visits |
| `observation` | Live per-chunk reasoning (alert-gated) |
| `api_job_result` | Amplifier signals for a chunk (`signals[].name/score/level/flagged`) |
| `visit_summary` | Vocal findings, discordance, screener recommendations |
| `visit_note` | SOAP note (chief complaint / subjective / objective / assessment / plan) |
| `chart_draft` | Chart-update draft for clinician approval |
| `topics` | Topics for the next visit |
| `screener_suggested` | Validated screener order (GAD-7/PHQ-9) for a flagged condition *(opt-in)* |
| `trial_match` | Recruiting clinical trial matched to a care gap *(opt-in)* |
| `literature_ref` | PubMed citation supporting the main finding *(only at "detailed" depth)* |
| `longitudinal_delta` | Per-condition: voice-flagged visit vs. first-coded visit |

`visit_analyzing` (lifecycle) marks the moment live reasoning ends and the note is being written, so the UI can leave the "recording" state.

### Clinician settings (configured from the UI, sent per visit)

Config rides `POST /visits/start` as `{config: {...}}`, bound per-visit via a contextvar so the agents read it without threading it through every call (`clinical_agent/clinician_config.py`):

- **Explanation depth** — `minimal` / `standard` / `detailed`: how much rationale each output carries. `detailed` additionally pulls real PubMed citations.
- **Enabled outputs** — which clinical event types surface. Suppression happens at the single emit gate; only clinical types are gateable (lifecycle/telemetry/reasoning always flow). Screeners and trials default **off** (opt-in).

---

## HTTP API

| Method / path | Purpose |
|---|---|
| `GET /patients` | List patients |
| `GET /patients/{id}/chart` | Patient chart (visits, codes, prior signals, notes) |
| `GET /patients/{id}/visits` | Visit list (for an appointment picker) |
| `POST /patients/{id}/visits/start` | Start a visit. Body `{visit?, audio_path?, config?}` — omit `audio_path` to replay the appointment from precomputed results. Supersedes any running visit. |
| `POST /patients/{id}/visits/stop` | Cancel the running visit (UI calls this when switching appointments) |
| `WS /patients/{id}/visits/stream` | **Real ingestion boundary** — the scribe streams PCM frames + `visit.start`/`visit.end`; we bucket into the 30s/15s pipeline. (The demo mocks this with replay.) |
| `GET /events` | SSE stream of typed events |
| `GET /patients/{id}/visits/{n}/audio` | The visit's recording (demo plays it back, sped up) |
| `GET /patients/{id}/visits/{n}/result` | Folded snapshot of a completed visit |
| `POST /patients/{id}/longitudinal` | Run the cross-visit early-detection analysis |
| `GET /contract` | The machine-readable output contract |

---

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest          # offline test suite
```

### Run the demo (real backend + UI, no live API calls)

Two servers — the API and the static viewer:

```bash
# API on :8000 — reads ANTHROPIC_API_KEY from .env; replays precomputed signals, live Claude reasoning
AMPLIFIER_OFFLINE=1 SPEED=8 .venv/bin/uvicorn clinical_agent.main:app --port 8000
# Viewer on :8080 (any static server)
python3 -m http.server 8080 --directory mock_agent
```

Open `http://localhost:8080/viewer.html?base=http://localhost:8000&pid=<patient>`, pick a visit with data, and tap record. The phone triggers `POST /visits/start`, which replays that appointment. (`:8000` is the API — opening it directly shows "not found"; the page lives on `:8080` and `base` points it at the API.)

### Run the demo fully offline (no backend, no keys)

```bash
python3 mock_agent/server.py    # dependency-free; replays a synthetic scenario, serves the viewer at /
```

### Verify against the real APIs (one-shot smoke test)

```bash
AMPLIFIER_ACCOUNT_ID=... AMPLIFIER_API_KEY=... ANTHROPIC_API_KEY=... \
  .venv/bin/python scripts/smoke_live.py path/to/audio.wav   # ≥15s of speech
```

### Configuration

Copy `.env.example` to `.env` and fill in keys. All env vars:

| Variable | Default | Purpose |
|---|---|---|
| `AMPLIFIER_ACCOUNT_ID` / `AMPLIFIER_API_KEY` | — | Amplifier Health API credentials |
| `AMPLIFIER_BASE_URL` | `https://api.amplifierhealth.com` | API base |
| `AMPLIFIER_USE_CASES` | `aria` | Comma-separated Amplifier models to score each chunk |
| `AMPLIFIER_OFFLINE` | off | Never call the live API — use precomputed/cached results only |
| `AMPLIFIER_CACHE` | `off` | `warm` = reuse cached results, `record` = live + cache |
| `ANTHROPIC_API_KEY` | — | Claude (agents run live by default) |
| `ANTHROPIC_MODEL` | `claude-opus-4-8` | Claude model |
| `MOCK_CLAUDE` / `MOCK_WHISPER` | off | Offline agent / transcription stubs |
| `SPEED` | `1.0` | Replay speed multiplier (demo pacing) |
| `DATA_DIR` | `data` | Patient store root |

Demo mode and real mode run the **same code path** — only config and data differ.

---

## How a demo visit runs (replay, not live audio)

A demo visit is replayed from precomputed data — no audio is streamed *into* the pipeline, Whisper is never called. `replay_visit` reads the appointment's stored per-chunk `aria` results and emits them on the same contract as a live visit, while the diarized transcript is pretend-streamed **time-aligned** to each tick (standing in for the live ASR a scribe would provide in production). The **agent reasoning is real Claude**; only the audio stream and the Amplifier call are mocked. Each appointment reasons over only the visits that precede it (`chart(before=N)`), so the early-detection reasoning is causal, not clairvoyant.

For presentation, the UI plays the visit's **real recording sped up** (so you can hear the encounter) and stops the "recording" state automatically when live reasoning finishes (`visit_analyzing`), while the note continues to fill. Selecting a different appointment cancels the running visit so its reasoning can't bleed into the next.

---

## Repository layout

```
clinical_agent/        backend package
  contract.py          the versioned output contract (pydantic models + registry + labels)
  clinician_config.py  per-visit clinician settings (explanation depth, enabled outputs)
  events.py            event bus: config-gated suppression, validates + envelopes every event
  session.py           run_visit (live) · replay_visit (precomputed) · run_streaming_visit (WebSocket)
  streaming.py         AudioBucketer — buckets streamed PCM into the 30s/15s pipeline
  agents/roles.py      pre-visit, reasoner, post-visit, SOAP note, longitudinal, screeners, trials, PubMed
  agents/base.py       Claude wrapper (streaming, tools, rehearsal-cache fallback, mock mode)
  amplifier.py         async Amplifier client (rate limits, 429 retry, result cache, offline mode)
  store.py             JSON patient store (chart, visits, per-visit artifacts)
  api.py / main.py     FastAPI app + entrypoint
mock_agent/            the clinician UI (viewer.html) + a dependency-free replay server
pipeline/  viz/        offline tools: chunk→score→aggregate; trajectory dashboard
scripts/               import_demo_patient, dump_contract, smoke_live, build_notes, prewarm_cache
docs/contract/         generated machine-readable contract spec
tests/                 offline test suite (contract, events, session, roles, amplifier, ...)
```

---

## Data & privacy

**No patient data is committed to this repository.** `data/` is gitignored; the bundled patient is fully synthetic (`clinical_agent/synthetic.py`). The real demo dataset (chart, per-appointment transcripts, precomputed `aria` results) lives only in a private GCS bucket. The importer fetches on demand with your `gcloud` credentials, so anyone without bucket access simply runs against the synthetic patient:

```bash
.venv/bin/python scripts/import_demo_patient.py gs://YOUR_BUCKET/demo-data \
  --patient-dir gs://YOUR_BUCKET/demo-data/patient_<ID> --add-live-visit
```

Voice is identifiable data; in production, biomarker inference requires patient consent distinct from recording consent, and the model's performance should be validated per subgroup.

---

## License

MIT.
