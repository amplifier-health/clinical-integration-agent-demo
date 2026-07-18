from fastapi.testclient import TestClient

from clinical_agent.amplifier import AmplifierClient
from clinical_agent.api import create_app
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.store import PatientStore
from clinical_agent.synthetic import generate_synthetic_patient
from clinical_agent.transcribe import Transcriber


async def test_bus_fanout():
    bus = EventBus()
    q1, q2 = bus.subscribe(), bus.subscribe()
    await bus.emit("chunk_created", patient="p", chunk=1, start_s=0.0, end_s=30.0)
    e1, e2 = q1.get_nowait(), q2.get_nowait()
    assert e1["type"] == "chunk_created" and e1["chunk"] == 1 and "ts" in e1
    assert e2 == e1
    bus.unsubscribe(q1)
    await bus.emit("chunk_created", patient="p", chunk=2, start_s=30.0, end_s=60.0)
    assert q1.empty() and not q2.empty()


def test_patients_endpoint(tmp_path):
    store = PatientStore(tmp_path)
    generate_synthetic_patient(store)
    settings = Settings(data_dir=tmp_path)
    bus = EventBus()
    app = create_app(settings, store, bus, Transcriber(mock=True),
                     AmplifierClient(settings, bus, cache_dir=tmp_path / "cache"))
    client = TestClient(app)
    resp = client.get("/patients")
    assert resp.status_code == 200
    assert resp.json()[0]["id"] == "demo-synthetic"
    chart = client.get("/patients/demo-synthetic/chart").json()
    assert len(chart["visits"]) == 10
