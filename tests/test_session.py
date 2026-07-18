import json

from clinical_agent.agents import base
from clinical_agent.amplifier import AmplifierClient
from clinical_agent.audio import chunk_file
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.session import run_visit
from clinical_agent.store import PatientStore
from clinical_agent.synthetic import generate_synthetic_patient
from clinical_agent.transcribe import Transcriber


async def test_full_visit_fast_mode(tmp_path, wav_100s):
    settings = Settings(mock_claude=True, mock_whisper=True, amplifier_cache="warm",
                        speed=1e9, data_dir=tmp_path)
    bus = EventBus()
    q = bus.subscribe()
    store = PatientStore(tmp_path)
    generate_synthetic_patient(store)
    base.MOCK_RESPONSES["previsit"] = json.dumps(
        {"brief": "b", "vocal_trends": [], "topics_to_discuss": []})
    base.MOCK_RESPONSES["reasoner"] = "nothing notable"
    base.MOCK_RESPONSES["postvisit"] = json.dumps(
        {"summary": "done", "vocal_findings": [], "transcript_findings": [], "discordance": "none",
         "screener_recommendations": [], "chart_update_draft": [], "next_visit_topics": ["x"]})

    amplifier = AmplifierClient(settings, bus, cache_dir=tmp_path / "cache")
    # pre-seed the amplifier cache for both expected chunks (100s @45s -> 2 chunks)
    result = {"signals": [{"name": "mood-disruption", "score": 0.7, "level": "high", "flagged": True}],
              "summary": {"overall_level": "high"}}
    async for c in chunk_file(wav_100s, speed=1e9):
        amplifier._cache_write(c, result)

    summary = await run_visit(settings, bus, store, Transcriber(mock=True), amplifier,
                              "demo-synthetic", wav_100s)
    assert summary["summary"] == "done"
    # visit marked complete and artifacts persisted
    visits = store.list_visits("demo-synthetic")
    assert visits[-1].status == "complete"
    assert store.read_artifact("demo-synthetic", 10, "summary")["summary"] == "done"
    assert len(store.read_artifact("demo-synthetic", 10, "transcript")) == 2
    types = [q.get_nowait()["type"] for _ in range(q.qsize())]
    for expected in ("visit_started", "pre_visit_brief", "chunk_created", "transcript",
                     "api_job_result", "observation", "visit_summary", "visit_complete"):
        assert expected in types, expected
