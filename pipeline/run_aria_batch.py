#!/usr/bin/env python3
"""Score audio chunks with an Amplifier Health model and store per-chunk results.

Reads AMPLIFIER_API_KEY / AMPLIFIER_API_ID from the environment (or a .env file),
submits each chunk to POST /v2/models/{model}/analyze as multipart audio, polls the
job, and writes one JSON result per chunk. Runs chunks concurrently.

Usage:
    python pipeline/run_aria_batch.py CHUNK_DIR RESULTS_DIR [--model aria] [--workers 10]

Nothing here is specific to a private backend: it calls the public API documented at
https://api.amplifierhealth.com and reads only local audio files you provide.
"""
import argparse, json, os, time, uuid, glob
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

API = os.environ.get("AMPLIFIER_API_BASE", "https://api.amplifierhealth.com/v2")

def load_env():
    """Populate os.environ from a local .env if the vars aren't already set."""
    if os.environ.get("AMPLIFIER_API_KEY") and os.environ.get("AMPLIFIER_API_ID"):
        return
    for path in (".env", os.path.join(os.path.dirname(__file__), "..", ".env")):
        if os.path.exists(path):
            for line in open(path):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def headers():
    key, acct = os.environ.get("AMPLIFIER_API_KEY"), os.environ.get("AMPLIFIER_API_ID")
    if not key or not acct:
        raise SystemExit("Set AMPLIFIER_API_KEY and AMPLIFIER_API_ID (see .env.example).")
    return {"X-API-Key": key, "X-Account-ID": acct}

def multipart(path):
    boundary = "----chunk" + uuid.uuid4().hex
    fn = os.path.basename(path)
    body = bytearray()
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="audio"; filename="{fn}"\r\n'.encode()
    body += b"Content-Type: audio/wav\r\n\r\n"
    body += open(path, "rb").read()
    body += f"\r\n--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"

def api(method, url, data=None, ctype=None):
    req = urllib.request.Request(url, data=data, method=method, headers=headers())
    if ctype:
        req.add_header("Content-Type", ctype)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(2 ** attempt); continue
            return {"_error": f"HTTP {e.code}: {e.read()[:200].decode('utf-8', 'ignore')}"}
        except Exception as e:
            if attempt < 4:
                time.sleep(2 ** attempt); continue
            return {"_error": str(e)}
    return {"_error": "retries exhausted"}

def process(path, results_dir, model, poll_timeout=180):
    out = os.path.join(results_dir, os.path.basename(path).replace(".wav", ".json"))
    if os.path.exists(out) and json.load(open(out)).get("status") == "done":
        return "skip"
    body, ctype = multipart(path)
    sub = api("POST", f"{API}/models/{model}/analyze", body, ctype)
    if "_error" in sub or not sub.get("job_id"):
        json.dump({"file": os.path.basename(path), "status": "error", "error": sub.get("_error")}, open(out, "w"))
        return "error"
    jid = sub["job_id"]; t0 = time.time()
    while time.time() - t0 < poll_timeout:
        time.sleep(6)
        j = api("GET", f"{API}/jobs/{jid}")
        if j.get("status") == "done":
            r = j.get("result") or {}
            json.dump({"file": os.path.basename(path), "job_id": jid, "status": "done",
                       "overall_level": (r.get("summary") or {}).get("overall_level"),
                       "signals": {s["name"]: {"score": s.get("score"), "level": s.get("level"),
                                               "flagged": s.get("flagged")} for s in r.get("signals", [])},
                       "audio_quality": r.get("audio_quality"),
                       "vocal_features": (r.get("summary") or {}).get("description", {}).get("vocal_features"),
                       "extended_metrics": r.get("extended_metrics")}, open(out, "w"))
            return "done"
        if "_error" in j or j.get("status") in ("error", "failed"):
            json.dump({"file": os.path.basename(path), "status": "error"}, open(out, "w"))
            return "error"
    json.dump({"file": os.path.basename(path), "status": "timeout"}, open(out, "w"))
    return "timeout"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("chunk_dir"); p.add_argument("results_dir")
    p.add_argument("--model", default="aria"); p.add_argument("--workers", type=int, default=10)
    a = p.parse_args()
    load_env(); os.makedirs(a.results_dir, exist_ok=True)
    chunks = sorted(glob.glob(os.path.join(a.chunk_dir, "*.wav")))
    print(f"scoring {len(chunks)} chunks with model '{a.model}' ({a.workers} workers)")
    counts = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(process, c, a.results_dir, a.model) for c in chunks]
        for n, f in enumerate(as_completed(futs), 1):
            s = f.result(); counts[s] = counts.get(s, 0) + 1
            if n % 10 == 0 or n == len(chunks):
                print(f"  {n}/{len(chunks)} {counts}")
    print("done", counts)

if __name__ == "__main__":
    main()
