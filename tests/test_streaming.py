import json

from clinical_agent.streaming import AudioBucketer

SR = 16000


def _pcm(seconds: float) -> bytes:
    return b"\x00\x00" * int(seconds * SR)  # silent 16-bit mono PCM


def test_bucketer_emits_overlapping_windows_at_hop_cadence():
    b = AudioBucketer(sample_rate=SR, window_s=30.0, hop_s=15.0)
    # feed 15s → first hop boundary → one 15s window
    b.feed(_pcm(15))
    w1 = b.pop_ready()
    assert len(w1) == 1 and w1[0].index == 1
    assert (w1[0].start_s, w1[0].end_s) == (0.0, 15.0)
    # feed another 15s (30s total) → next window is the full 30s (overlaps the first 15s)
    b.feed(_pcm(15))
    w2 = b.pop_ready()
    assert len(w2) == 1 and (w2[0].start_s, w2[0].end_s) == (0.0, 30.0)
    # at 45s the window slides: [15, 45]
    b.feed(_pcm(15))
    w3 = b.pop_ready()
    assert len(w3) == 1 and (w3[0].start_s, w3[0].end_s) == (15.0, 45.0)


def test_bucketer_flush_drops_subminimum_tail():
    b = AudioBucketer(sample_rate=SR, window_s=30.0, hop_s=15.0, min_s=15.0)
    b.feed(_pcm(15)); b.pop_ready()      # emit the 15s window
    b.feed(_pcm(5))                       # only 5s more — below the 15s floor
    assert b.flush() is None
    b.feed(_pcm(12))                      # now 17s past the last boundary — clears the floor
    assert b.flush() is not None


def test_websocket_stream_runs_a_full_visit(monkeypatch):
    # Real WS transport end-to-end with mocked Claude/Whisper/Amplifier — proves the boundary works.
    from fastapi.testclient import TestClient

    from clinical_agent.agents import base
    from clinical_agent.amplifier import AmplifierClient
    from clinical_agent.api import create_app
    from clinical_agent.config import Settings
    from clinical_agent.events import EventBus
    from clinical_agent.store import PatientStore
    from clinical_agent.synthetic import generate_synthetic_patient
    from clinical_agent.transcribe import Transcriber

    base.MOCK_RESPONSES["reasoner"] = "nothing notable"
    base.MOCK_RESPONSES["previsit"] = json.dumps({"brief": "b", "vocal_trends": [], "topics_to_discuss": []})
    base.MOCK_RESPONSES["postvisit"] = json.dumps({
        "summary": "s", "vocal_findings": [], "transcript_findings": [], "discordance": "none",
        "screener_recommendations": [], "chart_update_draft": [], "next_visit_topics": []})
    base.MOCK_RESPONSES["notewriter"] = json.dumps({
        "chief_complaint": "c", "subjective": "s", "objective": "o", "assessment": [], "plan": []})

    import tempfile
    from pathlib import Path
    settings = Settings(mock_claude=True, mock_whisper=True, amplifier_offline=True,
                        data_dir=Path(tempfile.mkdtemp()))
    store = PatientStore(settings.data_dir)
    generate_synthetic_patient(store)
    planned = next(v for v in store.list_visits("demo-synthetic") if v.status == "planned")
    bus = EventBus()
    app = create_app(settings, store, bus, Transcriber("base", mock=True), AmplifierClient(settings, bus))

    client = TestClient(app)
    with client.websocket_connect(f"/patients/demo-synthetic/visits/stream?visit={planned.number}") as ws:
        ws.send_text(json.dumps({"type": "visit.start", "sample_rate": SR}))
        ws.send_bytes(_pcm(45))  # ~3 windows
        ws.send_text(json.dumps({"type": "visit.end"}))

    assert store.list_visits("demo-synthetic")  # visit marked complete + summary written
    v = next(v for v in store.list_visits("demo-synthetic") if v.number == planned.number)
    assert v.status == "complete"
    assert store.read_artifact("demo-synthetic", planned.number, "summary") is not None
