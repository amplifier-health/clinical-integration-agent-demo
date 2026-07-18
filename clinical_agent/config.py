import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    amplifier_base_url: str = "https://api.amplifierhealth.com"
    amplifier_account_id: str = ""
    amplifier_api_key: str = ""
    anthropic_model: str = "claude-opus-4-8"
    mock_claude: bool = False
    amplifier_cache: str = "off"  # off | warm | record
    whisper_model: str = "base"
    mock_whisper: bool = False
    speed: float = 1.0
    data_dir: Path = field(default_factory=lambda: Path("data"))

    @classmethod
    def from_env(cls) -> "Settings":
        from dotenv import load_dotenv
        load_dotenv()  # reads .env from the working directory; real env vars win
        return cls(
            amplifier_base_url=os.environ.get("AMPLIFIER_BASE_URL", cls.amplifier_base_url),
            amplifier_account_id=os.environ.get("AMPLIFIER_ACCOUNT_ID", ""),
            amplifier_api_key=os.environ.get("AMPLIFIER_API_KEY", ""),
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", cls.anthropic_model),
            mock_claude=os.environ.get("MOCK_CLAUDE", "") == "1",
            amplifier_cache=os.environ.get("AMPLIFIER_CACHE", "off"),
            whisper_model=os.environ.get("WHISPER_MODEL", "base"),
            mock_whisper=os.environ.get("MOCK_WHISPER", "") == "1",
            speed=float(os.environ.get("SPEED", "1.0")),
            data_dir=Path(os.environ.get("DATA_DIR", "data")),
        )
