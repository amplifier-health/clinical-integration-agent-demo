import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    amplifier_base_url: str = "https://api.amplifierhealth.com"
    amplifier_account_id: str = ""
    amplifier_api_key: str = ""
    # Amplifier use-case models to score each audio chunk against. Each name is a
    # model on POST /v2/models/{name}/analyze. Multiple => every chunk is scored by
    # each model and the signals are merged (tagged with `model`). This demo uses
    # only "aria" (the model we precomputed for the example patient).
    amplifier_use_cases: list[str] = field(default_factory=lambda: ["aria"])
    anthropic_model: str = "claude-opus-4-8"
    mock_claude: bool = False
    amplifier_cache: str = "off"  # off | warm | record
    # Never call the live Amplifier API — read cached results only; a cache miss returns a
    # benign empty result. Use this to run the demo entirely off precomputed results.
    amplifier_offline: bool = False
    whisper_model: str = "base"
    mock_whisper: bool = False
    speed: float = 1.0
    data_dir: Path = field(default_factory=lambda: Path("data"))

    @classmethod
    def from_env(cls) -> "Settings":
        from dotenv import load_dotenv
        load_dotenv()  # reads .env from the working directory; real env vars win
        raw = os.environ.get("AMPLIFIER_USE_CASES", "aria")  # comma-separated
        use_cases = [u.strip() for u in raw.split(",") if u.strip()] or ["aria"]
        return cls(
            amplifier_base_url=os.environ.get("AMPLIFIER_BASE_URL", cls.amplifier_base_url),
            amplifier_account_id=os.environ.get("AMPLIFIER_ACCOUNT_ID", ""),
            amplifier_api_key=os.environ.get("AMPLIFIER_API_KEY", ""),
            amplifier_use_cases=use_cases,
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", cls.anthropic_model),
            mock_claude=os.environ.get("MOCK_CLAUDE", "") == "1",
            amplifier_cache=os.environ.get("AMPLIFIER_CACHE", "off"),
            amplifier_offline=os.environ.get("AMPLIFIER_OFFLINE", "") == "1",
            whisper_model=os.environ.get("WHISPER_MODEL", "base"),
            mock_whisper=os.environ.get("MOCK_WHISPER", "") == "1",
            speed=float(os.environ.get("SPEED", "1.0")),
            data_dir=Path(os.environ.get("DATA_DIR", "data")),
        )
