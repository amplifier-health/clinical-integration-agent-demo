#!/usr/bin/env python3
"""Assemble a LOCAL-ONLY real-data scenario from precomputed Aria results.

Reads the real per-chunk Aria result JSONs, the real diarized transcript, and the
real cross-visit aggregate for one patient/visit, and writes an INPUTS-ONLY
scenario (chart + transcript.token + biomarker.result timeline). The real agent
(agent/clinical_agent.py) produces the OUTPUT events (topics, flowsheet, gaps,
trials) — nothing is scripted here.

Output goes to mock_agent/local/ which is gitignored — this file may reference
real (de-identified) patient data and must never enter the public repo.

Usage:
    python mock_agent/build_scenario_real.py \
        --results  /path/to/scratchpad/results/aria \
        --txdir    /path/to/scratchpad/txfull \
        --aggregate /path/to/scratchpad/results/aggregate.json \
        --visit 16
"""
import argparse, glob, json, os, re

def load_chunks(results_dir, seq):
    rows = []
    for f in glob.glob(os.path.join(results_dir, f"v{seq:02d}_c*.json")):
        d = json.load(open(f))
        if d.get("status") != "done" or not d.get("signals"):
            continue
        ci = d.get("chunk", 0)
        sig = [{"name": n, "score": round(s.get("score") or 0, 4), "tier": s.get("level")}
               for n, s in d["signals"].items()]
        rows.append((ci, sig))
    rows.sort()
    return rows

def load_transcript(txdir, seq):
    pat = re.compile(r'^(\d+\.\d+)-(\d+\.\d+)\s+\[SPEAKER_\d+\]:\s*(.*)$')
    toks = []
    for role, who in (("patient", "PT"), ("doctor", "DR")):
        f = os.path.join(txdir, f"v{seq:02d}_{role}.txt")
        if not os.path.exists(f):
            continue
        for line in open(f):
            m = pat.match(line.strip())
            if m and m.group(3).strip():
                toks.append((float(m.group(1)), who, m.group(3).strip()))
    toks.sort()
    # thin to ~12 tokens so the demo transcript stays readable
    if len(toks) > 14:
        step = len(toks) / 14
        toks = [toks[int(i * step)] for i in range(14)]
    return toks

def history_from_aggregate(agg_path, seq):
    if not os.path.exists(agg_path):
        return []
    agg = json.load(open(agg_path))
    prior = [v for v in agg["visits"] if v["seq"] < seq]
    items = []
    for sig, label in [("elevated-androgens", "androgen-associated"), ("anxiety", "anxiety")]:
        tiers = [v["signals"].get(sig, {}).get("peak_tier") for v in prior if v["signals"].get(sig)]
        tiers = [t for t in tiers if t]
        if tiers:
            items.append({"text": f"Voice {label} signal across prior visits: {' → '.join(tiers[-4:])}",
                          "source": "voice-history"})
    return items

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--txdir", required=True)
    p.add_argument("--aggregate", default="")
    p.add_argument("--visit", type=int, default=16)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "local", "scenario_womenshealth_real.json"))
    a = p.parse_args()

    chunks = load_chunks(a.results, a.visit)
    toks = load_transcript(a.txdir, a.visit)
    if not chunks:
        raise SystemExit(f"no completed Aria chunks for visit {a.visit} in {a.results}")

    E = []
    def ev(t, type, **kw): E.append({"t": round(t, 1), "type": type, **kw})

    for it in history_from_aggregate(a.aggregate, a.visit):
        ev(-2, "briefing_item", **it)
    # real biomarker chunks every 15s (mocked API call annotated by the agent)
    for ci, sig in chunks:
        ev(ci * 15, "biomarker.result", chunk_id=ci, t_start=ci * 15, signals=sig)
    for tstart, who, text in toks:
        ev(min(tstart, chunks[-1][0] * 15), "transcript.token", who=who, text=text)
    E.sort(key=lambda e: e["t"])

    chart = {  # de-identified — local only
        "name": "Patient A (real, de-identified)", "mrn": "—", "age": 31, "sex": "F",
        "problems": ["Generalized anxiety disorder", "Recurrent depressive episodes", "Fatigue"],
        "meds": ["SSRI"], "chief_complaint": "Anxiety follow-up.",
        "prior_gap": "Menstrual irregularity noted previously; not worked up."}
    out = {"scenario": f"womens_health_REAL_v{a.visit}", "chart": chart,
           "encounter_seconds": chunks[-1][0] * 15 + 30, "events": E,
           "note": "REAL de-identified data — LOCAL ONLY, do not commit."}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(chunks)} real chunks, {len(toks)} transcript tokens, visit {a.visit}")

if __name__ == "__main__":
    main()
