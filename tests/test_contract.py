import json
from pathlib import Path

import pytest

from clinical_agent import contract
from clinical_agent.events import EventBus, current_session, start_session


def test_version_pinned():
    # Bumping the contract must be deliberate — update this when you change it on purpose.
    assert contract.CONTRACT_VERSION == "1.0"


def test_registry_covers_every_emitted_type():
    # Every type string emitted in the codebase must have a registered model.
    import re
    src = Path("clinical_agent").rglob("*.py")
    emitted = set()
    for f in src:
        for m in re.finditer(r'emit\(\s*"([a-z_]+)"', f.read_text()):
            emitted.add(m.group(1))
    missing = emitted - set(contract.REGISTRY)
    assert not missing, f"emitted but unmodeled event types: {missing}"


def test_validate_rejects_unknown_field():
    with pytest.raises(Exception):
        contract.validate("chunk_created",
                          {"patient": "p", "chunk": 1, "start_s": 0.0, "end_s": 30.0, "bogus": 1})


def test_validate_normalizes_and_drops_none():
    out = contract.validate("api_job_result",
                            {"chunk": 2, "cached": True, "signals": [
                                {"name": "anxiety", "score": 0.5, "level": "moderate", "flagged": True}]})
    assert out["chunk"] == 2 and out["cached"] is True
    assert "offline_miss" not in out  # None fields are dropped from the wire


async def test_emit_stamps_envelope_and_sequences():
    bus = EventBus()
    q = bus.subscribe()
    start_session(patient_id="p1", visit=3)
    await bus.emit("chunk_created", patient="p1", chunk=1, start_s=0.0, end_s=30.0)
    await bus.emit("observation", patient="p1", chunk=1, text="hi")
    e1, e2 = q.get_nowait(), q.get_nowait()
    for e in (e1, e2):
        assert e["contract_version"] == "1.0"
        assert e["session_id"] == current_session().session_id
        assert "phase" in e and "ts" in e
    assert e1["seq"] == 1 and e2["seq"] == 2          # monotonic per session
    assert e1["phase"] == "telemetry" and e2["phase"] == "reasoning"


async def test_emit_rejects_offcontract_payload():
    bus = EventBus()
    bus.subscribe()
    start_session(patient_id="p1", visit=1)
    with pytest.raises(Exception):
        await bus.emit("visit_note", patient="p1", visit=1, chief_complaint="x")  # missing required


def test_signal_tolerates_extra_api_fields():
    # Live Amplifier signal objects may carry fields we don't model — they must pass
    # and be preserved (regression: extra="forbid" would break the live path).
    out = contract.validate("api_job_result", {
        "chunk": 1, "cached": False,
        "signals": [{"name": "anxiety", "score": 0.5, "level": "moderate",
                     "flagged": True, "recommended_action": "monitor", "extra_metric": 0.9}],
    })
    sig = out["signals"][0]
    assert sig["recommended_action"] == "monitor" and sig["extra_metric"] == 0.9


def test_committed_schema_is_current():
    # The published contract in docs/ must match the models. Regenerate with
    # `python scripts/dump_contract.py` when this fails.
    import scripts.dump_contract as d
    committed = Path("docs/contract/contract.json")
    assert committed.exists(), "run: python scripts/dump_contract.py"
    assert json.loads(committed.read_text()) == d.build(), \
        "contract drifted from models — run scripts/dump_contract.py and commit"
