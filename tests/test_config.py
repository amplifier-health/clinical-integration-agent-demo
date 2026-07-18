from clinical_agent.config import Settings


def test_from_env_defaults(monkeypatch):
    for var in ("AMPLIFIER_BASE_URL", "MOCK_CLAUDE", "SPEED"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env()
    assert s.anthropic_model == "claude-opus-4-8"
    assert s.amplifier_cache == "off"
    assert s.speed == 1.0


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("MOCK_CLAUDE", "1")
    monkeypatch.setenv("SPEED", "50")
    monkeypatch.setenv("AMPLIFIER_CACHE", "warm")
    s = Settings.from_env()
    assert s.mock_claude is True
    assert s.speed == 50.0
    assert s.amplifier_cache == "warm"
