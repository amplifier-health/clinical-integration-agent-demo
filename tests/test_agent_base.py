from clinical_agent.agents import base
from clinical_agent.agents.base import ClaudeAgent
from clinical_agent.config import Settings
from clinical_agent.events import EventBus


async def test_mock_mode_no_network(tmp_path):
    base.MOCK_RESPONSES["previsit"] = "mock brief"
    bus = EventBus()
    q = bus.subscribe()
    agent = ClaudeAgent("previsit", Settings(mock_claude=True), bus, cache_dir=tmp_path)
    out = await agent.run("sys", "user prompt")
    assert out == "mock brief"
    e = q.get_nowait()
    assert e["type"] == "agent_token" and e["agent"] == "previsit"


async def test_rehearsal_cache_fallback(tmp_path, monkeypatch):
    bus = EventBus()
    agent = ClaudeAgent("reasoner", Settings(), bus, cache_dir=tmp_path)
    key = agent._cache_key("sys", "hello")
    (tmp_path / f"{key}.txt").write_text("cached answer")

    async def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(agent, "_run_live", boom)
    out = await agent.run("sys", "hello")
    assert out == "cached answer"


async def test_history_intact_after_live_failure(tmp_path, monkeypatch):
    """A failed live call must not corrupt the SHARED visit history — no dangling user turn,
    roles stay alternating, and the fallback exchange is recorded."""
    bus = EventBus()
    agent = ClaudeAgent("reasoner", Settings(), bus, cache_dir=tmp_path)
    key = agent._cache_key("sys", "hello")
    (tmp_path / f"{key}.txt").write_text("cached answer")

    async def boom(system, user, *, history=None, **k):
        history.append({"role": "user", "content": user})  # simulate a partial live call
        raise RuntimeError("network down")

    monkeypatch.setattr(agent, "_run_live", boom)
    history = [{"role": "user", "content": "prior"}, {"role": "assistant", "content": "ok"}]
    out = await agent.run("sys", "hello", history=history)
    assert out == "cached answer"
    roles = [m["role"] for m in history]
    assert not any(roles[i] == roles[i + 1] == "user" for i in range(len(roles) - 1)), roles
    assert history[-1] == {"role": "assistant", "content": "cached answer"}
