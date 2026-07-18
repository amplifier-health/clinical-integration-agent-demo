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
        self.limiter = _RateLimiter(max_calls=5, per_seconds=60.0)
        self.poll_interval = 5.0
        self._auth = {"X-Account-ID": settings.amplifier_account_id, "X-API-Key": settings.amplifier_api_key}

    # -- cache ---------------------------------------------------------
    def _cache_path(self, chunk: AudioChunk) -> Path:
        return self.cache_dir / f"{hashlib.sha256(chunk.wav_bytes).hexdigest()}.json"

    def _cache_read(self, chunk: AudioChunk):
        p = self._cache_path(chunk)
        return json.loads(p.read_text()) if p.exists() else None

    def _cache_write(self, chunk: AudioChunk, result: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(chunk).write_text(json.dumps(result, indent=2))

    # -- API flow ------------------------------------------------------
    async def analyze(self, chunk: AudioChunk) -> dict:
        if self.s.amplifier_cache == "warm":
            cached = self._cache_read(chunk)
            if cached is not None:
                await self.bus.emit("api_job_result", chunk=chunk.index, cached=True,
                                    signals=cached.get("signals"), summary=cached.get("summary"))
                return cached
        async with httpx.AsyncClient(timeout=60) as http:
            up = (await http.post(f"{self.s.amplifier_base_url}/v2/audio/uploads",
                                  json={"content_type": "audio/wav"},
                                  headers=self._auth)).raise_for_status().json()
            (await http.put(up["upload_url"], content=chunk.wav_bytes,
                            headers=up.get("required_headers", {}))).raise_for_status()
            await self.limiter.acquire()
            job = (await http.post(f"{self.s.amplifier_base_url}/v2/models/haven/analyze",
                                   json={"audio_upload_ref": up["upload_ref"]},
                                   headers=self._auth)).raise_for_status().json()
            job_id = job.get("id") or job.get("job_id")
            await self.bus.emit("api_job_created", chunk=chunk.index, job_id=job_id)
            while True:
                status = (await http.get(f"{self.s.amplifier_base_url}/v2/jobs/{job_id}",
                                         headers=self._auth)).raise_for_status().json()
                state = str(status.get("status", "")).lower()
                if state in _DONE:
                    result = status["result"]
                    break
                if state in _FAILED:
                    raise RuntimeError(f"Amplifier job {job_id} failed: {status}")
                await asyncio.sleep(self.poll_interval)
        if self.s.amplifier_cache in ("warm", "record"):
            self._cache_write(chunk, result)
        await self.bus.emit("api_job_result", chunk=chunk.index, cached=False,
                            signals=result.get("signals"), summary=result.get("summary"))
        return result
