import asyncio
import tempfile


class Transcriber:
    def __init__(self, model_size: str = "base", mock: bool = False):
        self.model_size = model_size
        self.mock = mock
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, compute_type="int8")
        return self._model

    def _run(self, wav_bytes: bytes) -> str:
        model = self._load()
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(wav_bytes)
            f.flush()
            segments, _info = model.transcribe(f.name)
            return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(self, wav_bytes: bytes) -> str:
        if self.mock:
            return "[mock transcript]"
        return await asyncio.to_thread(self._run, wav_bytes)
