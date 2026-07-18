import io
import wave

from clinical_agent.audio import chunk_file


async def collect(path, **kw):
    return [c async for c in chunk_file(path, **kw)]


async def test_chunk_boundaries(wav_100s):
    chunks = await collect(wav_100s, chunk_seconds=45.0, min_seconds=15.0, speed=1e9)
    # 100s -> 45 + 45 + 10; trailing 10s < 15s floor merges into chunk 2 (45+55)
    assert len(chunks) == 2
    assert chunks[0].index == 1 and chunks[1].index == 2
    assert abs((chunks[0].end_s - chunks[0].start_s) - 45.0) < 0.1
    assert abs((chunks[1].end_s - chunks[1].start_s) - 55.0) < 0.1


async def test_chunks_are_valid_wav(wav_100s):
    chunks = await collect(wav_100s, speed=1e9)
    with wave.open(io.BytesIO(chunks[0].wav_bytes)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
