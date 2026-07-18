import asyncio
import json
import time

from fastapi.testclient import TestClient

from clinical_agent.agents import base
from clinical_agent.amplifier import AmplifierClient
from clinical_agent.api import create_app
from clinical_agent.audio import chunk_file
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.store import PatientStore
from clinical_agent.synthetic import generate_synthetic_patient
from clinical_agent.transcribe import Transcriber


async def _seed_cache(amplifier, wav):
    result = {"signals": [], "summary": {}}
    async for c in chunk_file(wav, speed=1e9):
        amplifier._cache_write(c, result)


def test_visit_endpoint(tmp_path, wav_100s):
    settings = Settings(mock_claude=True, mock_whisper=True, amplifier_cache="warm",
                        speed=1e9, data_dir=tmp_path)
    bus, store = EventBus(), PatientStore(tmp_path)
    generate_synthetic_patient(store)
    for name, resp in [("previsit", {"brief": "b", "vocal_trends": [], "topics_to_discuss": []}),
                       ("postvisit", {"summary": "done", "vocal_findings": [], "transcript_findings": [],
                                      "discordance": "", "screener_recommendations": [],
                                      "chart_update_draft": [], "next_visit_topics": []}),
                       ("longitudinal", {"narrative": "n", "deltas": []})]:
        base.MOCK_RESPONSES[name] = json.dumps(resp)
    base.MOCK_RESPONSES["reasoner"] = "ok"
    amplifier = AmplifierClient(settings, bus, cache_dir=tmp_path / "cache")
    asyncio.run(_seed_cache(amplifier, wav_100s))

    app = create_app(settings, store, bus, Transcriber(mock=True), amplifier)
    with TestClient(app) as client:
        resp = client.post("/patients/demo-synthetic/visits/start",
                           json={"audio_path": str(wav_100s)})
        assert resp.status_code == 202
        deadline = time.time() + 10
        while time.time() < deadline:
            if store.read_artifact("demo-synthetic", 10, "summary"):
                break
            time.sleep(0.1)
        assert store.read_artifact("demo-synthetic", 10, "summary")["summary"] == "done"

        out = client.post("/patients/demo-synthetic/longitudinal").json()
        assert out["narrative"] == "n"
