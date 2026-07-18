import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx

from clinical_agent.audio import AudioChunk
from clinical_agent.config import Settings
from clinical_agent.events import EventBus

_DONE = {"completed", "done", "succeeded"}
_FAILED = {"failed", "error"}


class _RateLimiter:
    def __init__(self, max_calls: int, per_seconds: float, now=time.monotonic, sleep=asyncio.sleep):
        self.max_calls, self.per_seconds = max_calls, per_seconds
        self._now, self._sleep = now, sleep
        self._stamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                cutoff = self._now() - self.per_seconds
                self._stamps = [t for t in self._stamps if t > cutoff]
                if len(self._stamps) < self.max_calls:
                    self._stamps.append(self._now())
                    return
                await self._sleep(self._stamps[0] + self.per_seconds - self._now())


class AmplifierClient:
    """Async client for the Amplifier Health v2 API.

    Upload flow per chunk: presigned upload -> analyze (haven) -> poll job.
    Analyze calls respect the 5/min rate limit; results land whenever jobs
    finish, so callers must tolerate out-of-order completion.
    """

    def __init__(self, settings: Settings, bus: EventBus, cache_dir: Path | None = None):
        self.s = settings
        self.bus = bus
        self.cache_dir = cache_dir or settings.data_dir / "amplifier_cache"
        self.limiter = _RateLimiter(max_calls=5, per_seconds=60.0)       # analyze endpoints
        self.general_limiter = _RateLimiter(max_calls=8, per_seconds=60.0)  # uploads/jobs (10/min cap)
        self.poll_interval = 10.0
        self.max_429_retries = 6
        self._auth = {"X-Account-ID": settings.amplifier_account_id, "X-API-Key": settings.amplifier_api_key}

    async def _request(self, http: httpx.AsyncClient, method: str, url: str,
                       limiter: "_RateLimiter | None", **kw) -> httpx.Response:
        """Rate-limited request with retry on 429, honoring Retry-After."""
        for _ in range(self.max_429_retries):
            if limiter is not None:
                await limiter.acquire()
            r = await http.request(method, url, **kw)
            if r.status_code != 429:
                return r.raise_for_status()
            await asyncio.sleep(float(r.headers.get("retry-after", 15)))
        raise RuntimeError(f"still rate-limited after {self.max_429_retries} retries: {method} {url}")

    # -- cache ---------------------------------------------------------
    def _cache_path(self, chunk: AudioChunk) -> Path:
        # key on audio + the set of models scored, so changing use-cases doesn't collide
        models = "+".join(self.s.amplifier_use_cases)
        return self.cache_dir / f"{hashlib.sha256(chunk.wav_bytes).hexdigest()}-{models}.json"

    def _cache_read(self, chunk: AudioChunk):
        p = self._cache_path(chunk)
        return json.loads(p.read_text()) if p.exists() else None

    def _cache_write(self, chunk: AudioChunk, result: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(chunk).write_text(json.dumps(result, indent=2))

    # -- API flow ------------------------------------------------------
    @staticmethod
    def _merge(results: dict[str, dict]) -> dict:
        """One use-case → its result unchanged. Many → signals merged (tagged with model)."""
        if len(results) == 1:
            return next(iter(results.values()))
        signals = [{**s, "model": uc} for uc, res in results.items() for s in (res.get("signals") or [])]
        return {"signals": signals, "by_model": {uc: r.get("summary") for uc, r in results.items()}}

    async def _analyze_one(self, http: httpx.AsyncClient, upload_ref: str, model: str, chunk_index: int) -> dict:
        # NB: analyze expects form encoding, not JSON (verified against the live API)
        job = (await self._request(http, "POST", f"{self.s.amplifier_base_url}/v2/models/{model}/analyze",
                                   self.limiter, data={"audio_upload_ref": upload_ref},
                                   headers=self._auth)).json()
        job_id = job.get("id") or job.get("job_id")
        await self.bus.emit("api_job_created", chunk=chunk_index, model=model, job_id=job_id)
        while True:
            status = (await self._request(http, "GET", f"{self.s.amplifier_base_url}/v2/jobs/{job_id}",
                                          self.general_limiter, headers=self._auth)).json()
            state = str(status.get("status", "")).lower()
            if state in _DONE:
                return status["result"]
            if state in _FAILED:
                raise RuntimeError(f"Amplifier job {job_id} ({model}) failed: {status}")
            await asyncio.sleep(self.poll_interval)

    async def analyze(self, chunk: AudioChunk) -> dict:
        if self.s.amplifier_cache == "warm":
            cached = self._cache_read(chunk)
            if cached is not None:
                await self.bus.emit("api_job_result", chunk=chunk.index, cached=True,
                                    signals=cached.get("signals"), summary=cached.get("summary"))
                return cached
        async with httpx.AsyncClient(timeout=60) as http:
            up = (await self._request(http, "POST", f"{self.s.amplifier_base_url}/v2/audio/uploads",
                                      self.general_limiter, json={"content_type": "audio/wav"},
                                      headers=self._auth)).json()
            (await http.put(up["upload_url"], content=chunk.wav_bytes,
                            headers=up.get("required_headers", {}))).raise_for_status()
            results: dict[str, dict] = {}
            for model in self.s.amplifier_use_cases:  # score the same upload against each use-case
                res = await self._analyze_one(http, up["upload_ref"], model, chunk.index)
                results[model] = res
                await self.bus.emit("api_job_result", chunk=chunk.index, cached=False, model=model,
                                    signals=res.get("signals"), summary=res.get("summary"))
        merged = self._merge(results)
        if self.s.amplifier_cache in ("warm", "record"):
            self._cache_write(chunk, merged)
        return merged
