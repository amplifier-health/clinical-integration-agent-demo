"""Localize gs:// paths on demand so demo data lives ONLY in the private bucket.

No patient data is stored in this repo. The importer and cache-prewarm read directly
from GCS at runtime, using the caller's `gcloud` credentials. Anyone running the public
repo without access to the bucket can't fetch the data — `localize()` raises SystemExit,
which aborts the one-shot import/prewarm step. The app itself never sees the data, so if
the demo patient was never imported, `main.py` just starts with the synthetic patient.
"""
import subprocess
import tempfile
from pathlib import Path


def localize(path: str | Path, dest: Path | None = None) -> Path:
    """Return a local path for `path`. If it's a gs:// URI (file, directory, or wildcard),
    download it with `gcloud storage` (falling back to `gsutil`) into a temp dir and return
    the local equivalent. Local paths are returned unchanged."""
    p = str(path)
    if not p.startswith("gs://"):
        return Path(path)
    dest = dest or Path(tempfile.mkdtemp(prefix="demo-gcs-"))
    dest.mkdir(parents=True, exist_ok=True)
    last_err = "no `gcloud` or `gsutil` found on PATH"
    for cmd in (["gcloud", "storage", "cp", "-r", p, str(dest)],
                ["gsutil", "-m", "cp", "-r", p, str(dest)]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            continue
        if r.returncode == 0:
            break
        last_err = (r.stderr or "").strip()
    else:
        raise SystemExit(
            f"Could not fetch {p} from GCS — the demo's data lives only in that private bucket, "
            f"so you need `gcloud` access to it.\n{last_err}")
    # `cp -r gs://.../name dest` lands at dest/name; a wildcard lands the matches in dest/
    name = p.rstrip("/").split("/")[-1]
    landed = dest / name
    return landed if landed.exists() else dest
