# Clinical Integration Agent — Voice-Biomarker Trajectory Demo

A standalone, open-source demo of a clinical-integration pipeline that turns a
patient's longitudinal visit audio into **per-visit voice-biomarker trajectories**.
For each visit it splits the patient's speech into overlapping windows, scores each
window with a voice-biomarker model via the [Amplifier Health API](https://api.amplifierhealth.com),
aggregates the results per visit, and renders an interactive dashboard.

The motivating question: **can a voice model surface a condition before it is
discussed or coded in the clinic?** The included sample shows a patient whose
`anxiety` and `elevated-androgens` signals rise across "silent" early visits
(coded for unrelated complaints) and peak at the eventual anxiety and PCOS diagnoses.

> **Data note:** the bundled `sample_data/` is **synthetic and illustrative only** —
> it is not real patient data. Bring your own audio and API credentials to run the
> pipeline on real recordings.

## What's here

```
pipeline/
  chunk_audio.py      # split a recording into 30s / 15s-overlap windows (ffmpeg)
  run_aria_batch.py   # score each chunk via POST /v2/models/{model}/analyze, poll, store JSON
  aggregate.py        # roll per-chunk results up into per-visit trajectories
viz/
  build_viz.py        # render the interactive dashboard from an aggregate.json
  trajectories.html   # prebuilt dashboard from the synthetic sample (open in a browser)
sample_data/
  aggregate.json          # synthetic per-visit aggregate that drives the dashboard
  visits.example.json     # example per-visit metadata config
```

## Quick look (no setup)

Open `viz/trajectories.html` in any browser. It's a self-contained page (no network
calls) showing the condition-signal tier/probability heatmap, the anxiety-vs-androgens
trajectory, wellness metrics, and speech features for the synthetic sample.

## Run the pipeline on your own audio

Requirements: Python 3.9+ (standard library only) and `ffmpeg`/`ffprobe` on `PATH`.

1. Copy `.env.example` to `.env` and fill in your API credentials:
   ```
   AMPLIFIER_API_KEY=...
   AMPLIFIER_API_ID=...
   ```
2. Chunk each visit's mono audio into its own results namespace:
   ```bash
   python pipeline/chunk_audio.py visit01.wav chunks/v01
   python pipeline/run_aria_batch.py chunks/v01 results/v01 --model aria
   ```
   Repeat per visit (`v02`, `v03`, …).
3. Describe the visits in a `visits.json` (see `sample_data/visits.example.json`),
   aggregate, and render:
   ```bash
   python pipeline/aggregate.py results visits.json --out aggregate.json --patient MYCASE
   python viz/build_viz.py aggregate.json viz/trajectories.html
   ```

## Model & signals

`run_aria_batch.py` defaults to the `aria` model, whose signals include
`anxiety`, `mood-disruption`, `elevated-androgens`, `elevated-blood-pressure`,
`iron-deficiency`, `fatigue`, and `dehydration`. Any model name accepted by the API
works via `--model`. See the [API docs](https://api.amplifierhealth.com) for the
full model and sign catalog.

## License

MIT — see `LICENSE`.
