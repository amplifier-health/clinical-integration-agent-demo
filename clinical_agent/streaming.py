"""Real-time audio ingestion — the production side of the scribe integration boundary.

The ambient scribe streams raw PCM frames to us over a WebSocket; we own the buffering.
`AudioBucketer` accumulates the frames and emits an overlapping window (`window_s` long)
every `hop_s` seconds — the 30s/15s cadence the Amplifier API is fed at. Each emitted
window is an `AudioChunk` identical in shape to the file chunker's output, so the exact
same downstream pipeline (transcribe → analyze → reason → typed SSE events) runs on it.

The demo does NOT use this path — it replays precomputed results (see `session.replay_visit`).
This module is the real transport that replaces that mock in production.
"""
import io

from pydub import AudioSegment

from clinical_agent.audio import AudioChunk


class AudioBucketer:
    """Buffer streamed 16-bit mono PCM and emit a rolling `window_s`-second window every
    `hop_s` seconds. With window 30s / hop 15s each window overlaps the previous by 15s,
    which keeps signals stable across window edges."""

    def __init__(self, sample_rate: int = 16000, window_s: float = 30.0,
                 hop_s: float = 15.0, min_s: float = 15.0):
        self.sr = sample_rate
        self.window_s = window_s
        self.hop_s = hop_s
        self.min_s = min_s
        self._buf = bytearray()   # raw int16 mono PCM for the whole visit
        self._hops = 0            # hop boundaries already emitted
        self._index = 0           # 1-based chunk index, matching the file chunker

    def _seconds(self) -> float:
        return len(self._buf) / 2 / self.sr  # 2 bytes per 16-bit sample

    def feed(self, pcm: bytes) -> None:
        self._buf.extend(pcm)

    def _window(self, t_start: float, t_end: float) -> AudioChunk:
        a, b = int(t_start * self.sr) * 2, int(t_end * self.sr) * 2
        seg = AudioSegment(data=bytes(self._buf[a:b]), sample_width=2,
                           frame_rate=self.sr, channels=1)
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        self._index += 1
        return AudioChunk(index=self._index, wav_bytes=buf.getvalue(),
                          start_s=round(t_start, 2), end_s=round(t_end, 2))

    def pop_ready(self) -> list[AudioChunk]:
        """Windows for every hop boundary reached since the last call (usually 0 or 1)."""
        out = []
        while self._seconds() >= (self._hops + 1) * self.hop_s:
            self._hops += 1
            t_end = self._hops * self.hop_s
            out.append(self._window(max(0.0, t_end - self.window_s), t_end))
        return out

    def flush(self) -> AudioChunk | None:
        """The final partial window on visit end — the audio since the last hop boundary,
        emitted only if it clears the Amplifier minimum (`min_s`)."""
        t_end = self._seconds()
        if t_end - self._hops * self.hop_s < self.min_s:
            return None  # too short to score; it already overlapped into the last full window
        return self._window(max(0.0, t_end - self.window_s), t_end)
