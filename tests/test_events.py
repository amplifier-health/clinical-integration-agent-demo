from fastapi.testclient import TestClient

from clinical_agent.api import create_app
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.store import PatientStore
from clinical_agent.synthetic import generate_synthetic_patient


async def test_bus_fanout():
    bus = EventBus()
    q1, q2 = bus.subscribe(), bus.subscribe()
    await bus.emit("chunk_created", index=1)
    e1, e2 = q1.get_nowait(), q2.get_nowait()
    assert e1["type"] == "chunk_created" and e1["index"] == 1 and "ts" in e1
    assert e2 == e1
    bus.unsubscribe(q1)
    await bus.emit("x")
    assert q1.empty() and not q2.empty()


def test_patients_endpoint(tmp_path):
    store = PatientStore(tmp_path)
    generate_synthetic_patient(store)
    app = create_app(Settings(), store, EventBus())
    client = TestClient(app)
    resp = client.get("/patients")
    assert resp.status_code == 200
    assert resp.json()[0]["id"] == "demo-synthetic"
    chart = client.get("/patients/demo-synthetic/chart").json()
    assert len(chart["visits"]) == 10
