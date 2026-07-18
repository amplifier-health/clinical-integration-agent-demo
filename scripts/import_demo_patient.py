"""Import the GCS demo dataset (precomputed aria results) into the patient store.

Usage: .venv/bin/python scripts/import_demo_patient.py /path/to/abridge-hackathon-demo-071826 \
           [--pid 16bbcdbe] [--data-dir data] [--add-live-visit]

Maps:
  aggregate.json + aria_results/*.json -> data/patients/<pid>/{patient,visits}.json
  per-visit artifacts: signals (store shape), chunks (per-chunk trace for the frontend)
ICD-10 codes are parsed from aggregate visit labels like "Anxiety Dx (F419)".
Visits absent from aggregate.json get dates interpolated between known neighbors.
"""
import argparse
import json
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical_agent.store import Icd10Code, PatientMeta, PatientStore, VisitMeta  # noqa: E402

CODE_DESCRIPTIONS = {
    "F41.9": "Anxiety disorder, unspecified",
    "N91.2": "Oligomenorrhea, unspecified",
    "E28.2": "Polycystic ovarian syndrome",
    "F32.9": "Major depressive disorder, single episode, unspecified",
}
FLAGGED_TIERS = {"consider", "moderate", "elevated", "high"}


def parse_codes(label: str) -> list[Icd10Code]:
    codes = []
    for raw in re.findall(r"\(([A-Z]\d{2,4})\)", label or ""):
        code = f"{raw[:3]}.{raw[3:]}" if len(raw) > 3 else raw
        codes.append(Icd10Code(code=code, description=CODE_DESCRIPTIONS.get(code, label)))
    return codes


def visit_signals(chunks: list[dict]) -> list[dict]:
    """Aggregate per-chunk aria signals into the store's per-visit signal list."""
    by_sign: dict[str, list[float]] = defaultdict(list)
    tiers: dict[str, str] = {}
    order = ["inconclusive", "none", "low", "consider", "moderate", "elevated", "high"]
    for c in chunks:
        if c.get("status") != "done" or c.get("overall_level") == "inconclusive":
            continue
        for name, s in (c.get("signals") or {}).items():
            if s.get("level") == "inconclusive":
                continue
            by_sign[name].append(s.get("score", 0.0))
            if order.index(s.get("level", "none")) > order.index(tiers.get(name, "inconclusive")):
                tiers[name] = s["level"]
    out = []
    for name, scores in sorted(by_sign.items()):
        level = tiers.get(name, "none")
        out.append({
            "name": name,
            "label": name.replace("-", " ").title(),
            "score": round(sum(scores) / len(scores), 4),
            "level": level,
            "flagged": level in FLAGGED_TIERS,
        })
    return out


def interpolate_dates(seqs: list[int], known: dict[int, str]) -> dict[int, str]:
    """Fill missing visit dates by linear interpolation between known neighbors."""
    dates = dict(known)
    anchors = sorted(known)
    for seq in seqs:
        if seq in dates:
            continue
        before = max((a for a in anchors if a < seq), default=None)
        after = min((a for a in anchors if a > seq), default=None)
        if before is not None and after is not None:
            d0, d1 = date.fromisoformat(known[before]), date.fromisoformat(known[after])
            frac = (seq - before) / (after - before)
            dates[seq] = (d0 + timedelta(days=int((d1 - d0).days * frac))).isoformat()
        elif before is not None:
            dates[seq] = (date.fromisoformat(known[before]) + timedelta(days=14 * (seq - before))).isoformat()
        else:
            dates[seq] = (date.fromisoformat(known[after]) - timedelta(days=14 * (after - seq))).isoformat()
    return dates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--pid", default="16bbcdbe")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--add-live-visit", action="store_true",
                    help="append a 'planned' visit at the end for the live-demo act")
    args = ap.parse_args()

    agg = json.loads((args.src / "aggregate.json").read_text())
    agg_by_seq = {v["seq"]: v for v in agg["visits"]}

    chunks_by_seq: dict[int, list[dict]] = defaultdict(list)
    for f in sorted((args.src / "aria_results").glob("*.json")):
        r = json.loads(f.read_text())
        chunks_by_seq[r["seq"]].append(r)

    seqs = sorted(chunks_by_seq)
    dates = interpolate_dates(seqs, {s: v["date"] for s, v in agg_by_seq.items()})

    store = PatientStore(args.data_dir)
    store.save_patient(PatientMeta(id=args.pid, alias="Demo Patient (de-identified)", age=33, sex="F"))

    visits = []
    for seq in seqs:
        av = agg_by_seq.get(seq, {})
        label = av.get("label", f"Visit {seq}")
        visits.append(VisitMeta(
            number=seq,
            date=dates[seq],
            reason=re.sub(r"\s*\([A-Z]\d{2,4}\)", "", label),
            icd10=parse_codes(label),
            has_audio=True,
            status="complete",
        ))
        store.write_artifact(args.pid, seq, "signals", visit_signals(chunks_by_seq[seq]))
        store.write_artifact(args.pid, seq, "chunks", [
            {"chunk": c["chunk"], "start": c.get("start"), "overall_level": c.get("overall_level"),
             "signals": {n: {"score": s.get("score"), "level": s.get("level")}
                         for n, s in (c.get("signals") or {}).items()}}
            for c in sorted(chunks_by_seq[seq], key=lambda c: c["chunk"])])
        if av:
            store.write_artifact(args.pid, seq, "wellness",
                                 {"wellness": av.get("wellness"), "speech": av.get("speech"),
                                  "mark": av.get("mark")})

    if args.add_live_visit:
        last = date.fromisoformat(dates[seqs[-1]])
        visits.append(VisitMeta(number=seqs[-1] + 1, date=(last + timedelta(days=21)).isoformat(),
                                reason="Follow-up (today)", has_audio=True, status="planned"))

    store.save_visits(args.pid, visits)
    n_art = sum(1 for _ in (args.data_dir / "patients" / args.pid / "visits").rglob("*.json"))
    print(f"imported patient {args.pid}: {len(visits)} visits, {n_art} artifacts")


if __name__ == "__main__":
    main()
