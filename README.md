# Clinical Integration Agent

A voice-biomarker layer for the clinical visit workflow, built for the Abridge x Anthropic x Lightspeed hackathon ("The Future of Agentic AI in Healthcare").

Ambient documentation tools capture **what** the patient says. This agent captures **how** they say it — running visit audio through the [Amplifier Health API](https://docs.amplifierhealth.com) (vocal biomarkers) and reasoning over the results with Claude, live, across three stages of the encounter:

| Stage | What the agent does |
|---|---|
| **Before visit** | Reads the patient's chart and prior vocal results; produces a brief with vocal trends and topics to discuss |
| **During visit** | Chunks the audio in near real time, transcribes with Whisper, analyzes each chunk through Amplifier's `haven` behavioral-health model, and streams clinical observations — especially discordance between the patient's words and their vocal signals |
| **After visit** | Drafts a visit summary with a vocal-findings section, screener recommendations (PHQ-9, GAD-7), a chart update draft for clinician approval, and topics for the next visit — which becomes the next visit's pre-visit input |

Over successive visits, the longitudinal analyst computes, per condition, the gap between when the **voice first flagged** a signal and when the condition was **first ICD-10 coded** — the early-detection story.

The agents never diagnose. They describe signals, suggest validated screeners, and draft — a clinician approves.

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │                PATIENT STORE (JSON)         │
                    └──────┬──────────────────────────▲───────────┘
                           │ read                     │ write (drafts)
                 ┌─────────▼─────────┐      ┌─────────┴─────────┐
                 │  1. PRE-VISIT     │      │  3. POST-VISIT    │
                 │     AGENT         │      │     AGENT         │
                 └─────────┬─────────┘      └─────────▲─────────┘
                           │ brief                    │
                 ┌─────────▼──────────────────────────┴─────────┐
                 │  2. VISIT SESSION                            │
                 │  audio → chunker → whisper + Amplifier API   │
                 │  (async job queue) → Claude reasoner         │
                 └──────────────────────────────────────────────┘
```

Everything emits to an event bus streamed over SSE (`GET /events`) — the contract a frontend attaches to.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                       # 21 tests, all offline

# Demo mode (no credentials needed): mock Claude, mock Whisper, cached Amplifier results
MOCK_CLAUDE=1 MOCK_WHISPER=1 AMPLIFIER_CACHE=warm SPEED=20 \
  .venv/bin/uvicorn clinical_agent.main:app

# In another terminal:
curl -N localhost:8000/events &        # watch the event stream
curl -s localhost:8000/patients | python3 -m json.tool
curl -s -X POST localhost:8000/patients/demo-synthetic/visits/start \
  -H 'content-type: application/json' -d '{"audio_path": "visit.wav"}'
curl -s -X POST localhost:8000/patients/demo-synthetic/longitudinal
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `AMPLIFIER_ACCOUNT_ID` / `AMPLIFIER_API_KEY` | — | Amplifier Health API credentials |
| `AMPLIFIER_BASE_URL` | `https://api.amplifierhealth.com` | API base |
| `ANTHROPIC_API_KEY` | — | Claude (agents run live by default) |
| `ANTHROPIC_MODEL` | `claude-opus-4-8` | Claude model |
| `MOCK_CLAUDE` / `MOCK_WHISPER` | off | Offline agent/transcription stubs |
| `AMPLIFIER_CACHE` | `off` | `warm` = reuse cached results, `record` = live + cache |
| `SPEED` | `1.0` | Audio replay speed multiplier (demo pacing) |
| `WHISPER_MODEL` | `base` | faster-whisper model size |
| `DATA_DIR` | `data` | Patient store root |

Demo mode and real mode run the same code path — only config and data differ. `ffmpeg` is required for non-WAV audio.

## SSE event contract

| Event | Payload fields |
|---|---|
| `visit_started` | `patient, visit, date, reason` |
| `pre_visit_brief` | `patient, brief, vocal_trends, topics_to_discuss` |
| `chunk_created` | `patient, chunk, start_s, end_s` |
| `transcript` | `patient, chunk, text` |
| `api_job_created` | `chunk, job_id` |
| `api_job_result` | `chunk, cached, signals, summary` |
| `agent_token` | `agent, text` (streamed Claude output) |
| `agent_tool_call` | `agent, tool, input` |
| `observation` | `patient, chunk, text` |
| `visit_summary` | `patient, visit, summary, vocal_findings, discordance, screener_recommendations` |
| `chart_draft` | `patient, visit, items` |
| `topics` | `patient, visit, items` |
| `longitudinal_delta` | `patient, condition, first_voice_flag_visit, first_coded_visit, visits_early` |
| `longitudinal_narrative` | `patient, text` |
| `error` | `patient, message` |

## Live smoke test

Verifies the real Amplifier wire format and a live Claude call:

```bash
AMPLIFIER_ACCOUNT_ID=... AMPLIFIER_API_KEY=... ANTHROPIC_API_KEY=... \
  .venv/bin/python scripts/smoke_live.py path/to/audio.wav
```

## Data

**This repository contains no real patient data.** The `data/` directory is gitignored; the bundled patient is fully synthetic (`clinical_agent/synthetic.py`). Any patient bundle dropped into `data/patients/<id>/` in the same shape works without code changes.

## License

MIT

---

## Frontend demo UI (this branch)

An Abridge-style clinician UI that speaks the **same SSE event contract as the backend**
(`observation`, `api_job_result`, `agent_token`, `visit_summary`, `chart_draft`, `topics`,
`trial_match`, `longitudinal_delta`, …) — no adapter layer.

- **`mock_agent/viewer.html`** — the UI: mobile recorder (tap the record button to start a
  visit), clinical Note with ambient addenda + chart-update drafts, an "Ambient AI" panel
  (care-gap follow-ups, matched clinical trials, longitudinal early-detection), and a
  behind-the-scenes agent-reasoning stream. Configurable via query params:
  `?base=http://localhost:8000` (backend URL), `?pid=<patient>`, `?audio=<path>`.
- **`mock_agent/server.py`** + `build_scenario_womenshealth.py` — a dependency-free mock that
  serves the same endpoints (`/patients/{id}/chart`, `/events`) and replays a synthetic,
  backend-contract scenario, so the UI can be developed and demoed without the backend.
- **`pipeline/`**, **`viz/`** — offline tools: chunk audio → score via the Amplifier API →
  aggregate; and an interactive per-visit voice-biomarker trajectory dashboard.

**Run against the real backend** (two servers — the API and the static viewer):

```bash
# API on :8000 (reads ANTHROPIC_API_KEY from .env; no audio, no Whisper, no live Amplifier calls)
SPEED=40 .venv/bin/uvicorn clinical_agent.main:app --port 8000
# viewer on :8080 (any static server)
python3 -m http.server 8080 --directory mock_agent
```

Then open `http://localhost:8080/viewer.html?base=http://localhost:8000&pid=demo-patient` and
pick a visit. `:8000` is the API — opening it directly shows "not found," which is expected; the
page lives on `:8080` and the `base` param points it at the API. Tap record and the phone triggers
`POST /patients/{id}/visits/start`, which **replays that appointment** (below). **Fully offline
demo (no backend, no key):** `python3 mock_agent/server.py` then open its URL.

## How a visit runs (replay, not live audio)

A demo visit is replayed from precomputed data — no audio is streamed and Whisper is never
called. `replay_visit` reads the appointment's stored per-chunk `aria` results and emits them on
the same SSE contract as a live visit, while the diarized transcript we already have is
pretend-streamed **time-aligned** to each tick (standing in for the live ASR we'd run on the
scribe's audio in production). The **agent reasoning is real Claude**; only the audio stream and
the Amplifier API are mocked. Each appointment reasons over only the visits that precede it
(`chart(before=N)`), so its early-detection reasoning is causal, not clairvoyant.

## Demo data lives in GCS, not this repo

No patient data is committed here. The precomputed demo dataset (chart, per-appointment
transcripts, precomputed `aria` results) lives only in a private bucket
`gs://YOUR_BUCKET/demo-data`. The importer accepts `gs://` URIs and fetches on demand with your
`gcloud` credentials — so anyone running the public repo without access to that bucket cannot
obtain the data (the fetch raises and the import aborts; the app run without an imported patient
just starts with the synthetic one). To set up the real patient:

```bash
# import chart/history + per-visit signals & transcripts + a planned live visit
.venv/bin/python scripts/import_demo_patient.py \
  gs://YOUR_BUCKET/demo-data \
  --patient-dir gs://YOUR_BUCKET/demo-data/patient_<PATIENT_ID> \
  --add-live-visit
```

The replay path needs only the small per-chunk results + transcripts (not the multi-GB audio), so
no cache-prewarm step is required. `scripts/build_notes.py --pid demo-patient` optionally
backfills a SOAP note for each historical visit.
