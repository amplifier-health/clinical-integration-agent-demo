import asyncio
import io
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from pydub import AudioSegment


@dataclass
class AudioChunk:
    index: int
    wav_bytes: bytes
    start_s: float
    end_s: float


async def chunk_file(
    path: Path,
    *,
    chunk_seconds: float = 45.0,
    min_seconds: float = 15.0,
    speed: float = 1.0,
) -> AsyncIterator[AudioChunk]:
    """Slice an audio file into chunks, replaying at ~real time (scaled by speed).

    The trailing slice merges into the previous chunk when shorter than
    min_seconds (the Amplifier API floor is 15s).
    """
    audio = AudioSegment.from_file(path).set_frame_rate(16000).set_channels(1)
    total_ms = len(audio)
    step_ms = int(chunk_seconds * 1000)
    bounds: list[tuple[int, int]] = []
    pos = 0
    while pos < total_ms:
        end = min(pos + step_ms, total_ms)
        bounds.append((pos, end))
        pos = end
    if len(bounds) > 1 and (bounds[-1][1] - bounds[-1][0]) < min_seconds * 1000:
        last = bounds.pop()
        prev = bounds.pop()
        bounds.append((prev[0], last[1]))

    for i, (start, end) in enumerate(bounds, start=1):
        await asyncio.sleep((end - start) / 1000.0 / speed)
        buf = io.BytesIO()
        audio[start:end].export(buf, format="wav")
        yield AudioChunk(index=i, wav_bytes=buf.getvalue(), start_s=start / 1000, end_s=end / 1000)
