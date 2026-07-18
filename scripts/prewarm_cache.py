"""Fill the Amplifier cache from precomputed results — WITHOUT calling the API.

The cache is keyed by the SHA-256 of each chunk's audio bytes, so precomputed results
can't just be copied in — the keys must come from the backend's own chunker. This script
runs `chunk_file` over the visit audio and writes the precomputed aria result for each
chunk (index-aligned) into the cache. After this, run the backend with AMPLIFIER_OFFLINE=1
(or AMPLIFIER_CACHE=warm) and no live Amplifier calls are made.

Usage:
    .venv/bin/python scripts/prewarm_cache.py \
        --audio /path/to/visit.wav \
        --results '/path/to/demo-data/aria_results/v16_c*.json' \
        --data-dir data
"""
import argparse
import asyncio
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical_agent.amplifier import AmplifierClient  # noqa: E402
from clinical_agent.audio import chunk_file  # noqa: E402
from clinical_agent.config import Settings  # noqa: E402
from clinical_agent.events import EventBus  # noqa: E402
from clinical_agent.gcs import localize  # noqa: E402


def _to_api_result(precomp: dict) -> dict:
    """Convert a stored aria result (signals keyed by name) to the API result shape."""
    sig = precomp.get("signals") or {}
    return {
        "signals": [{"name": n, "score": v.get("score"), "level": v.get("level"),
                     "flagged": v.get("flagged")} for n, v in sig.items()],
        "summary": {"overall_level": precomp.get("overall_level")},
        "audio_quality": precomp.get("audio_quality"),
        "vocal_features": precomp.get("vocal_features"),
        "extended_metrics": precomp.get("extended_metrics"),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="visit audio — local path or gs:// URI")
    ap.add_argument("--results", required=True,
                    help="that visit's precomputed chunk JSONs — a local glob, or a gs:// wildcard "
                         "(e.g. 'gs://YOUR_BUCKET/demo-data/aria_results/v16_c*.json')")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--use-case", default="aria")
    args = ap.parse_args()

    audio = localize(args.audio)  # pulled from GCS on demand; never bundled in the repo
    if str(args.results).startswith("gs://"):
        rdir = localize(args.results)              # wildcard matches land flat in a temp dir
        files = glob.glob(str(rdir / "*.json"))
    else:
        files = glob.glob(args.results)
    def _chunk_idx(f: str) -> int:
        m = re.search(r"c(\d+)", Path(f).stem)
        if m is None:
            raise SystemExit(f"unexpected result filename (no cNNN index): {f}")
        return int(m.group(1))

    files = sorted(files, key=_chunk_idx)
    precomp = [json.loads(Path(f).read_text()) for f in files]
    precomp = [p for p in precomp if p.get("status") == "done"]
    if not precomp:
        raise SystemExit(f"no completed results matched {args.results}")

    settings = Settings(data_dir=args.data_dir, amplifier_use_cases=[args.use_case], amplifier_cache="warm")
    client = AmplifierClient(settings, EventBus())

    written = total = 0
    async for chunk in chunk_file(audio, speed=1e9):
        total += 1
        if chunk.index >= len(precomp):
            continue  # more backend chunks than precomputed results — leave uncached (warned below)
        client._cache_write(chunk, _to_api_result(precomp[chunk.index]))
        written += 1
    if total != len(precomp):
        print(f"WARNING: the backend chunker produced {total} chunks but there are {len(precomp)} "
              f"precomputed results — {max(0, total - written)} chunk(s) have NO cached result, so "
              f"offline mode will show empty signals for them. Re-generate the results with the same "
              f"chunking, or point --audio at the exact audio the results were computed from.")
    print(f"prewarmed {written} cache entries ({args.use_case}) into {settings.data_dir / 'amplifier_cache'} "
          f"— run with AMPLIFIER_OFFLINE=1 for zero live Amplifier calls")


if __name__ == "__main__":
    asyncio.run(main())
