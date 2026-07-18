from clinical_agent.store import PatientStore
from clinical_agent.synthetic import generate_synthetic_patient


def test_generates_loadable_bundle(tmp_path):
    store = PatientStore(tmp_path)
    generate_synthetic_patient(store, n_visits=10)
    visits = store.list_visits("demo-synthetic")
    assert len(visits) == 10
    assert visits[-1].status == "planned" and visits[-1].has_audio
    # PCOS coded at visit 6, mental health at visit 9
    assert any(c.code == "E28.2" for c in visits[5].icd10)
    assert any(c.code.startswith("F") for c in visits[8].icd10)
    # voice flagged mood-disruption well before visit 9
    sig = store.read_artifact("demo-synthetic", 3, "signals")
    mood = next(s for s in sig if s["name"] == "mood-disruption")
    assert mood["flagged"] is True
    # past visits carry summaries; today's visit carries none
    assert store.read_artifact("demo-synthetic", 2, "summary") is not None
    assert store.read_artifact("demo-synthetic", 10, "summary") is None
