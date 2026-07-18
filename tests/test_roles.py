import json

from clinical_agent.agents import base, roles
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.store import PatientStore
from clinical_agent.synthetic import generate_synthetic_patient


def setup(tmp_path):
    store = PatientStore(tmp_path)
    generate_synthetic_patient(store)
    return Settings(mock_claude=True, data_dir=tmp_path), EventBus(), store


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


async def test_pre_visit_brief(tmp_path):
    settings, bus, store = setup(tmp_path)
    base.MOCK_RESPONSES["previsit"] = json.dumps(
        {"brief": "b", "vocal_trends": ["mood up"], "topics_to_discuss": ["sleep"]})
    q = bus.subscribe()
    out = await roles.pre_visit_brief(settings, bus, store, "demo-synthetic")
    assert out["topics_to_discuss"] == ["sleep"]
    assert store.read_artifact("demo-synthetic", 10, "pre_visit_brief")["brief"] == "b"
    assert any(e["type"] == "pre_visit_brief" for e in _drain(q))


async def test_post_visit_persists(tmp_path):
    settings, bus, store = setup(tmp_path)
    payload = {"summary": "s", "vocal_findings": [], "transcript_findings": [],
               "discordance": "none", "screener_recommendations": ["PHQ-9"],
               "chart_update_draft": [], "next_visit_topics": ["mood"]}
    base.MOCK_RESPONSES["postvisit"] = json.dumps(payload)
    out = await roles.post_visit_summary(settings, bus, store, "demo-synthetic", 10,
                                         ["hi"], [], [], {})
    assert out["screener_recommendations"] == ["PHQ-9"]
    assert store.read_artifact("demo-synthetic", 10, "summary")["summary"] == "s"


async def test_longitudinal(tmp_path):
    settings, bus, store = setup(tmp_path)
    base.MOCK_RESPONSES["longitudinal"] = json.dumps(
        {"narrative": "voice flagged early", "deltas": [
            {"condition": "depression", "first_voice_flag_visit": 3,
             "first_coded_visit": 9, "visits_early": 6}]})
    q = bus.subscribe()
    out = await roles.longitudinal_analysis(settings, bus, store, "demo-synthetic")
    assert out["deltas"][0]["visits_early"] == 6
    assert any(e["type"] == "longitudinal_delta" for e in _drain(q))
