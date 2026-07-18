"""Backfill a basic SOAP visit note for each historical (completed) visit.

Reads the stored transcript + the ICD-10 codes coded for each visit and drafts a note
via the documentation agent, so the valid coded labels are surfaced in a real note rather
than dropped. Run offline with MOCK_CLAUDE=1, or live with ANTHROPIC_API_KEY set.

Usage:
    MOCK_CLAUDE=1 .venv/bin/python scripts/build_notes.py --pid demo-patient
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical_agent.agents import roles  # noqa: E402
from clinical_agent.config import Settings  # noqa: E402
from clinical_agent.events import EventBus  # noqa: E402
from clinical_agent.store import PatientStore  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default="demo-patient")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--only", type=int, default=None, help="only this visit number")
    args = ap.parse_args()

    store = PatientStore(args.data_dir)
    settings = Settings(data_dir=args.data_dir)
    bus = EventBus()

    for v in store.list_visits(args.pid):
        if v.status != "complete" or (args.only is not None and v.number != args.only):
            continue
        transcript = store.read_artifact(args.pid, v.number, "transcript")
        signals = store.read_artifact(args.pid, v.number, "signals")
        note = await roles.build_visit_note(settings, bus, store, args.pid, v.number,
                                            transcript, [c.model_dump() for c in v.icd10], signals)
        print(f"visit {v.number} ({v.date}): note written — {len(note.get('assessment', []))} coded, "
              f"{len(note.get('plan', []))} plan item(s)")


if __name__ == "__main__":
    asyncio.run(main())
