#!/usr/bin/env python3
"""Aggregate per-chunk model results into per-visit signal trajectories.

Expects results laid out as one sub-directory per visit:
    RESULTS_DIR/<visit_id>/c000_0000s.json, c001_0015s.json, ...
and a visits.json describing each visit:
    [{"id": "v01", "date": "2025-02-02", "label": "Ear infection", "mark": "runway"}, ...]

Writes AGGREGATE_OUT (default aggregate.json) consumed by viz/build_viz.py.
Conclusive chunks only (overall_level != "inconclusive") feed wellness/speech means;
per-signal scores ignore chunks where that signal is inconclusive.

Usage:
    python pipeline/aggregate.py RESULTS_DIR visits.json [--out aggregate.json]
"""
import argparse, json, glob, os, collections, statistics as st

SEV = {"none": 0, "low": 1, "consider": 2, "moderate": 3, "elevated": 4, "inconclusive": -1}
SEVR = {v: k for k, v in SEV.items()}
SIGNAL_ORDER = ["anxiety", "mood-disruption", "elevated-androgens", "elevated-blood-pressure",
                "iron-deficiency", "fatigue", "dehydration"]

def aggregate_visit(files):
    chunks = [json.load(open(f)) for f in files]
    chunks = [c for c in chunks if c.get("status") == "done"]
    concl = [c for c in chunks if c.get("overall_level") not in ("inconclusive", None)]
    sig, peak = collections.defaultdict(list), collections.defaultdict(lambda: -1)
    for c in chunks:
        for name, s in (c.get("signals") or {}).items():
            if s.get("level") != "inconclusive" and s.get("score") is not None:
                sig[name].append(s["score"]); peak[name] = max(peak[name], SEV.get(s.get("level"), -1))
    signals = {k: {"mean": round(st.mean(v), 4), "max": round(max(v), 4), "n": len(v),
                   "peak_tier": SEVR.get(peak[k], "none")} for k, v in sig.items()}
    well, wlab, wanch = collections.defaultdict(list), {}, {}
    for c in concl:
        for m in (c.get("extended_metrics") or []):
            well[m["metric_id"]].append(m["score_mean"]); wlab[m["metric_id"]] = m["label"]
            wanch[m["metric_id"]] = [m.get("low_anchor"), m.get("high_anchor")]
    wellness = {k: {"label": wlab[k], "mean": round(st.mean(v), 4), "n": len(v), "anchors": wanch[k]}
                for k, v in well.items()}
    sp, slab, sunit = collections.defaultdict(list), {}, {}
    for c in concl:
        for vf in (c.get("vocal_features") or []):
            if vf.get("value") is not None:
                sp[vf["feature"]].append(vf["value"]); slab[vf["feature"]] = vf["label"]; sunit[vf["feature"]] = vf.get("unit")
    speech = {k: {"label": slab[k], "mean": round(st.mean(v), 4), "unit": sunit[k], "n": len(v)}
              for k, v in sp.items()}
    return len(chunks), len(concl), signals, wellness, speech

def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_dir"); p.add_argument("visits_json")
    p.add_argument("--out", default="aggregate.json"); p.add_argument("--patient", default="SAMPLE")
    a = p.parse_args()
    visits_meta = json.load(open(a.visits_json))
    out_visits = []
    for i, vm in enumerate(visits_meta, 1):
        files = glob.glob(os.path.join(a.results_dir, vm["id"], "*.json"))
        n, nc, signals, wellness, speech = aggregate_visit(files)
        out_visits.append({"seq": i, "date": vm.get("date"), "label": vm.get("label"),
                           "mark": vm.get("mark", ""), "n_chunks": n, "n_conclusive": nc,
                           "signals": signals, "wellness": wellness, "speech": speech})
    json.dump({"patient": a.patient, "visits": out_visits, "signal_order": SIGNAL_ORDER,
               "generated": f"{len(out_visits)} visits"}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {len(out_visits)} visits")

if __name__ == "__main__":
    main()
