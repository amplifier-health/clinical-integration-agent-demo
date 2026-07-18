from clinical_agent.store import Icd10Code, PatientMeta, PatientStore, VisitMeta


def make_store(tmp_path):
    store = PatientStore(tmp_path)
    store.save_patient(PatientMeta(id="p1", alias="Jane D.", age=32, sex="F"))
    store.save_visits("p1", [
        VisitMeta(number=1, date="2026-01-10", reason="Ear infection",
                  icd10=[Icd10Code(code="H66.90", description="Otitis media")], has_audio=True),
        VisitMeta(number=2, date="2026-02-14", reason="Wellness visit"),
    ])
    return store


def test_round_trip(tmp_path):
    store = make_store(tmp_path)
    assert store.get_patient("p1").alias == "Jane D."
    visits = store.list_visits("p1")
    assert len(visits) == 2 and visits[0].icd10[0].code == "H66.90"


def test_artifacts_and_chart(tmp_path):
    store = make_store(tmp_path)
    store.write_artifact("p1", 1, "signals", [{"name": "mood-disruption", "score": 0.7}])
    store.write_artifact("p1", 1, "summary", {"summary": "flagged"})
    assert store.read_artifact("p1", 1, "signals")[0]["name"] == "mood-disruption"
    assert store.read_artifact("p1", 1, "missing") is None
    chart = store.chart("p1")
    assert chart["patient"]["alias"] == "Jane D."
    assert chart["visits"][0]["signals"][0]["score"] == 0.7
    assert chart["visits"][1]["summary"] is None
