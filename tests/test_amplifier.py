

import httpx
import respx

from clinical_agent.amplifier import AmplifierClient, _RateLimiter
from clinical_agent.audio import AudioChunk
from clinical_agent.config import Settings
from clinical_agent.events import EventBus

BASE = "https://api.test"


def make_client(tmp_path, cache="off", use_cases=("haven",)):
    s = Settings(amplifier_base_url=BASE, amplifier_account_id="acct", amplifier_api_key="key",
                 amplifier_cache=cache, amplifier_use_cases=list(use_cases))
    return AmplifierClient(s, EventBus(), cache_dir=tmp_path / "cache")


def chunk(n=1):
    return AudioChunk(index=n, wav_bytes=b"RIFFfake" + bytes([n]), start_s=0, end_s=45)


RESULT = {"signals": [{"name": "mood-disruption", "score": 0.7, "level": "high", "flagged": True}],
          "summary": {"overall_level": "high"}, "audio_quality": {"voice_percentage": 92.0}}


@respx.mock
async def test_full_flow(tmp_path):
    respx.post(f"{BASE}/v2/audio/uploads").respond(200, json={
        "upload_url": "https://storage.test/put", "upload_ref": "ref-1",
        "required_headers": {"Content-Type": "audio/wav"}})
    respx.put("https://storage.test/put").respond(200)
    respx.post(f"{BASE}/v2/models/haven/analyze").respond(200, json={"id": "job-1", "status": "queued"})
    respx.get(f"{BASE}/v2/jobs/job-1").mock(side_effect=[
        httpx.Response(200, json={"id": "job-1", "status": "processing"}),
        httpx.Response(200, json={"id": "job-1", "status": "completed", "result": RESULT}),
    ])
    client = make_client(tmp_path)
    client.poll_interval = 0  # no real sleeping in tests
    result = await client.analyze(chunk())
    assert result["signals"][0]["name"] == "mood-disruption"
    analyze_call = respx.calls[2].request
    assert analyze_call.headers["X-Account-ID"] == "acct"
    # analyze is form-encoded, not JSON
    assert analyze_call.content == b"audio_upload_ref=ref-1"


@respx.mock
async def test_warm_cache_skips_network(tmp_path):
    client = make_client(tmp_path, cache="warm")
    client._cache_write(chunk(), RESULT)
    result = await client.analyze(chunk())  # respx would 404 any request
    assert result == RESULT and len(respx.calls) == 0


async def test_rate_limiter_spaces_calls():
    clock = [0.0]
    waits = []

    async def fake_sleep(s):
        waits.append(s)
        clock[0] += s

    rl = _RateLimiter(max_calls=2, per_seconds=60, now=lambda: clock[0], sleep=fake_sleep)
    await rl.acquire()
    await rl.acquire()
    await rl.acquire()
    assert waits and abs(sum(waits) - 60.0) < 0.01  # third call waited a full window


@respx.mock
async def test_offline_never_calls_api(tmp_path):
    # AMPLIFIER_OFFLINE: cache miss -> empty result, cache hit -> cached, and NEVER any network call
    s = Settings(amplifier_base_url=BASE, amplifier_account_id="a", amplifier_api_key="k",
                 amplifier_offline=True, amplifier_use_cases=["aria"])
    client = AmplifierClient(s, EventBus(), cache_dir=tmp_path / "cache")
    miss = await client.analyze(chunk(1))            # respx would 404 any request
    assert miss["signals"] == [] and len(respx.calls) == 0
    client._cache_write(chunk(2), RESULT)
    hit = await client.analyze(chunk(2))
    assert hit == RESULT and len(respx.calls) == 0
