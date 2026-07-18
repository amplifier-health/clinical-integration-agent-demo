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

## Frontend demo UI & standalone components (this branch)

This branch adds the clinician-facing demo and offline tooling that sit around the backend above.

- **`mock_agent/viewer.html`** — an Abridge-style UI: mobile recorder (tap-to-record),
  clinical Note with an ambient flowsheet + voice-informed addenda, an "Ambient AI" panel
  (care-gap follow-ups + matched clinical trials), and a behind-the-scenes agent-reasoning stream.
- **`agent/`** — a standalone persistent-conversation Claude agent (low-effort live triage,
  high-effort post-visit analysis, real ClinicalTrials.gov lookup) + a dependency-free SSE server
  (`agent/run_agent.py --serve`) that drives the UI from a replayed scenario.
- **`mock_agent/`** — scenario builders + a stdlib SSE mock server for building the UI without the backend.
- **`pipeline/`** — chunk audio → score via the Amplifier API → aggregate per visit.
- **`viz/`** — an interactive per-visit voice-biomarker trajectory dashboard.

> **Integration note:** the backend (`clinical_agent/`, FastAPI on :8000) and this UI currently use
> **different SSE event names** — the backend emits `observation` / `api_job_result` / `agent_token` /
> `visit_summary`, while the UI consumes `topic.suggested` / `biomarker.result` / `reasoning.step` /
> `note.addendum`. Wiring the UI to the real backend means mapping one contract onto the other (or having
> the backend emit the UI's events). See `docs/AGENT_FLOW_SPEC.md` for the UI-side contract.
