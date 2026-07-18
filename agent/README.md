# Real Clinical-Integration Agent

A genuine Claude agent (`claude-opus-4-8`, tool runner) that does the reasoning for
the Abridge-style demo. **Only two things are mocked** — audio is not streamed live,
and the voice-biomarker API is not polled live (results are replayed from precomputed
objects). Everything else is real: Claude decides what to surface, writes the note
addenda and gap list, and performs a **real ClinicalTrials.gov lookup**.

## How it works

`clinical_agent.py` replays a scenario's INPUT events (biomarker results + transcript)
and calls Claude at reasoning triggers:

- **Live, every ~4 conclusive chunks** — Claude sees the chart, running transcript, and
  voice-biomarker trend, and decides (via `suggest_topic` / `update_flowsheet` tools)
  whether anything new clears the bar. It dedupes against what's already suggested or
  already discussed, and prefers the *rising-but-not-the-focus* signal.
- **Post-visit** — Claude writes note addenda (`add_note_addendum`), the follow-up /
  **gap** list (`flag_followup`, `discussed=false` = voice-flagged but not addressed),
  and calls `search_clinical_trials` (real ClinicalTrials.gov v2 API) for the gap.

Every tool call becomes a UI event; the agent's narration and each decision become
`reasoning.step` events. Same event contract as the mock agent (see
`docs/AGENT_FLOW_SPEC.md` §2) — **the UI does not change** whether it's driven by the
scripted mock or this real agent.

## Setup

```bash
pip install -r agent/requirements.txt        # anthropic SDK
cp .env.example .env                          # add ANTHROPIC_API_KEY
```

## Run

Print the real agent's event stream as JSONL:

```bash
python agent/run_agent.py --scenario mock_agent/scenario_womenshealth.json --speed 8
```

Serve it live to the UI (real Claude reasoning as the "recording" plays):

```bash
python agent/run_agent.py --scenario mock_agent/local/scenario_womenshealth_real.json --serve --port 8788
open http://localhost:8788/
```

## Real-data scenario (local only)

`mock_agent/build_scenario_real.py` assembles a scenario from the **real** precomputed
Aria results + transcript for one patient/visit. Its output goes to `mock_agent/local/`
which is **gitignored** — real (de-identified) patient data must never enter this public
repo. The shipped `scenario_womenshealth.json` uses synthetic-but-realistic values so the
demo is safe to publish; point the real scenario at the agent locally for a demo with
genuine biomarker trajectories.

## Demo safety

For a 3-minute demo, run the agent once ahead of time and replay its JSONL (deterministic),
or run live if the network is reliable — the biomarker replay + mocked API call are the
same either way; only Claude's reasoning is live.
