import io
import math
import struct
import wave
from pathlib import Path

import pytest


def sine_wav_bytes(seconds: float, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(seconds * rate)
        frames = b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(n)
        )
        w.writeframes(frames)
    return buf.getvalue()


@pytest.fixture
def wav_100s(tmp_path) -> Path:
    p = tmp_path / "visit.wav"
    p.write_bytes(sine_wav_bytes(100.0))
    return p
