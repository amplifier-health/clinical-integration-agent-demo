# Mock Agent

A dependency-free stand-in for the clinical-integration agent. It **replays a
scenario** (precomputed biomarker results + scripted reasoning) as a live,
time-compressed event stream, so the UI can be built against the real event
contract before the production agent exists. The API call is **visually mocked**
(a `reasoning.step` "POST /v2/models/{model}/analyze → 200 (replayed)") — no live
API calls, no credentials, nothing private.

## Run

```bash
python3 mock_agent/build_scenario_womenshealth.py   # (re)generate the default scenario
python3 mock_agent/server.py --port 8787            # serve (defaults to womens-health)
open http://localhost:8787/                         # built-in reference viewer
```

Scenarios (swap with `--scenario`): `scenario_womenshealth.json` (**default** — Aria
model; anxiety-focused visit with a quietly rising elevated-androgens signal that becomes
a PCOS gap) and `scenario_respiratory.json` (breath model; COPD). Both are synthetic.

Adjust playback speed with the field in the viewer, or `?speed=N` on the stream.
`speed=8` ≈ a 115s encounter in ~14s; `speed=1` is real-time.

## Event contract (SSE `event:` names)

The UI subscribes with `EventSource('/events?speed=8')` and routes each named event
to a surface. This is the ONLY thing the UI and the production backend must agree on
— see `docs/AGENT_FLOW_SPEC.md` §2.

| event | goes to |
|---|---|
| `meta` | stream header (scenario, speed) |
| `briefing` | pre-visit "watch-for" list |
| `transcript.token` | scrolling transcript |
| `reasoning.step` | Reasoning tab (incl. the mocked API call) |
| `biomarker.result` | trend store / Reasoning tab |
| `topic.suggested` | mobile Suggested discussion topics |
| `flowsheet.update` | Flowsheets panel |
| `note.addendum` | Note (post-visit) |
| `followup.item` | follow-up / gap list (`discussed:false` = the gap) |
| `trial.match` | Abridge AI panel |
| `end` | stream complete |

## Swapping scenarios / data

`scenario_*.json` = `{chart, encounter_seconds, events:[{t, type, ...}]}`. Author a new
one (see `build_scenario.py`) and pass `--scenario`. To drive it from *real* precomputed
biomarker objects, emit one `biomarker.result` per chunk with the model's signals; keep
the file out of this public repo if it contains real patient data.

## Production swap

Replace this SSE replay with the real agent (WebSocket, live API + MCP). If it emits the
same event names, **the UI does not change.**
