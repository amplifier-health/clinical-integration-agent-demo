import pytest

from clinical_agent.agents import roles
from clinical_agent.agents.base import ClaudeAgent
from clinical_agent.clinician_config import ClinicianConfig, current_config, set_config
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.store import PatientStore
from clinical_agent.synthetic import generate_synthetic_patient


def test_default_is_standard():
    assert ClinicianConfig().explainability == "standard"


def test_from_override_ignores_unknown_keys():
    # UI may send fields this version doesn't implement yet (enabled outputs, alert…) — tolerate them.
    cfg = ClinicianConfig.from_override({"explainability": "detailed", "alert": {"sensitivity": "x"}})
    assert cfg.explainability == "detailed"


def test_from_override_rejects_bad_value():
    with pytest.raises(Exception):
        ClinicianConfig.from_override({"explainability": "verbose"})  # not a valid depth


def test_depth_prompt_differs_per_level():
    prompts = {d: ClinicianConfig(explainability=d).depth_prompt()
               for d in ("minimal", "standard", "detailed")}
    assert len(set(prompts.values())) == 3
    assert "minimal" in prompts["minimal"] and "detailed" in prompts["detailed"]


def test_output_enabled_logic():
    assert ClinicianConfig().output_enabled("trial_match") is True          # None = all enabled
    cfg = ClinicianConfig(enabled_outputs=["topics"])
    assert cfg.output_enabled("topics") is True
    assert cfg.output_enabled("trial_match") is False


async def test_emit_suppresses_disabled_clinical_outputs():
    from clinical_agent.events import EventBus as _Bus
    bus = _Bus()
    q = bus.subscribe()
    set_config(ClinicianConfig(enabled_outputs=["topics"]))
    await bus.emit("chart_draft", patient="p", visit=1, items=[])   # clinical, disabled → dropped
    await bus.emit("topics", patient="p", visit=1, items=["x"])     # clinical, enabled → delivered
    await bus.emit("chunk_created", patient="p", chunk=1, start_s=0.0, end_s=30.0)  # telemetry → always
    types = []
    while not q.empty():
        types.append(q.get_nowait()["type"])
    assert types == ["topics", "chunk_created"]
    set_config(ClinicianConfig())  # reset for other tests


async def test_reasoner_system_prompt_reflects_depth(tmp_path, monkeypatch):
    store = PatientStore(tmp_path)
    generate_synthetic_patient(store)
    settings = Settings(mock_claude=True, data_dir=tmp_path)
    bus = EventBus()
    captured = {}

    async def fake_run(self, system, user, **kw):
        captured["system"] = system
        return "ok"

    monkeypatch.setattr(ClaudeAgent, "run", fake_run)
    set_config(ClinicianConfig(explainability="detailed"))
    await roles.reason_over_chunk(settings, bus, store, "demo-synthetic", 1, "hi", [], {})
    assert "Explanation depth = detailed" in captured["system"]
    assert current_config().explainability == "detailed"
