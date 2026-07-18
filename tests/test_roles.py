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


async def test_shared_visit_memory(tmp_path):
    """The reasoner (per chunk) and post-visit share ONE conversation — the final
    analysis sees every prior turn (true memory continuity, not re-injected context)."""
    settings, bus, store = setup(tmp_path)
    base.MOCK_RESPONSES["reasoner"] = "observation for this chunk"
    base.MOCK_RESPONSES["postvisit"] = json.dumps(
        {"summary": "s", "vocal_findings": [], "transcript_findings": [], "discordance": "none",
         "screener_recommendations": [], "chart_update_draft": [], "next_visit_topics": []})
    history: list = [{"role": "user", "content": "VISIT START"}]

    await roles.reason_over_chunk(settings, bus, store, "demo-synthetic", 0,
                                  "patient mentions periods are irregular", [], {}, history=history)
    await roles.reason_over_chunk(settings, bus, store, "demo-synthetic", 1,
                                  "second chunk transcript", [], {}, history=history)
    n_after_chunks = len(history)

    await roles.post_visit_summary(settings, bus, store, "demo-synthetic", 10,
                                   ["c0", "c1"], [], [], {}, history=history)

    # conversation grew across chunks AND into the post-visit turn (same memory)
    assert n_after_chunks >= 5           # seed + 2 chunks * (user + assistant)
    assert len(history) > n_after_chunks  # post-visit appended to the same conversation
    # the first chunk's content is still present when the post-visit analysis runs
    assert any("irregular" in json.dumps(m) for m in history)


async def test_build_visit_note_writes_and_emits(tmp_path):
    settings, bus, store = setup(tmp_path)
    base.MOCK_RESPONSES["notewriter"] = json.dumps({
        "chief_complaint": "anxiety",
        "subjective": "reports feeling anxious and tired",
        "objective": "alert, cooperative",
        "assessment": [{"code": "F41.9", "description": "Anxiety disorder", "note": "coded this visit"}],
        "plan": ["start PHQ-9"],
    })
    q = bus.subscribe()
    transcript = [{"speaker": "patient", "text": "I've been very anxious and tired."}]
    note = await roles.build_visit_note(settings, bus, store, "demo-synthetic", 1, transcript,
                                        [{"code": "F41.9", "description": "Anxiety disorder"}],
                                        [{"name": "anxiety", "level": "moderate"}])
    assert note["assessment"][0]["code"] == "F41.9"
    assert store.read_artifact("demo-synthetic", 1, "note")["plan"] == ["start PHQ-9"]
    assert any(e["type"] == "visit_note" for e in _drain(q))
