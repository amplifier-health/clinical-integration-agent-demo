"""Per-clinician settings that control agent behavior at the point of care.

Distinct from `Settings` (deployment/env): this is clinical behavior a doctor
tunes, editable and eventually UI-exposed. Loaded from a JSON file, overridable
per request (so the UI can toggle live), and bound to the visit via a contextvar
so the agents read it without threading it through every signature.

Three settings (see docs): alert sensitivity, screener suggestions, explainability
depth. Enforcement split: alert cap/floor + screener gating are enforced in CODE
here; explainability depth is a prompt fragment the agents inject.
"""
import contextvars
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from clinical_agent.contract import TIER

_TIER_ORDER = list(TIER.__args__)  # inconclusive < none < low < consider < moderate < elevated < high


def _tier_ge(level: str | None, floor: str) -> bool:
    if level not in _TIER_ORDER:
        return False
    return _TIER_ORDER.index(level) >= _TIER_ORDER.index(floor)


# Preset → (min live tier that may interrupt, max live interruptions per visit)
_PRESETS = {
    "conservative": ("elevated", 1),
    "balanced": ("moderate", 3),
    "sensitive": ("consider", 6),
}


class AlertConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sensitivity: Literal["conservative", "balanced", "sensitive"] = "balanced"
    # Explicit knobs override the preset when set.
    min_live_tier: Optional[str] = None
    max_live_per_visit: Optional[int] = None

    def resolved(self) -> tuple[str, int]:
        floor, cap = _PRESETS[self.sensitivity]
        return (self.min_live_tier or floor, self.max_live_per_visit if self.max_live_per_visit is not None else cap)


class ScreenerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    # signal name -> screening instrument to offer when that signal is flagged
    instruments: dict[str, str] = Field(default_factory=lambda: {
        "anxiety": "GAD-7",
        "mood-disruption": "PHQ-9",
    })


class ClinicianConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alert: AlertConfig = Field(default_factory=AlertConfig)
    screeners: ScreenerConfig = Field(default_factory=ScreenerConfig)
    explainability: Literal["minimal", "standard", "detailed"] = "standard"

    # -- loading / overrides ----------------------------------------------------
    @classmethod
    def load(cls, path: Path | None) -> "ClinicianConfig":
        if path and Path(path).exists():
            return cls.model_validate_json(Path(path).read_text())
        return cls()

    def merged(self, override: dict | None) -> "ClinicianConfig":
        """Return a copy with a per-request override deep-merged over it."""
        if not override:
            return self
        base = self.model_dump()
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = {**base[k], **v}
            else:
                base[k] = v
        return ClinicianConfig.model_validate(base)

    # -- enforcement helpers (code, not prompt) ---------------------------------
    def signal_clears_alert_floor(self, cumulative_signals: list) -> bool:
        floor, _cap = self.alert.resolved()
        return any(s.get("flagged") and _tier_ge(s.get("level"), floor) for s in cumulative_signals)

    def max_live(self) -> int:
        return self.alert.resolved()[1]

    def screeners_for(self, cumulative_signals: list) -> list[str]:
        """Instruments to offer given which signals are flagged (deduped, in signal order)."""
        if not self.screeners.enabled:
            return []
        out, seen = [], set()
        for s in cumulative_signals:
            inst = self.screeners.instruments.get(s.get("name"))
            if s.get("flagged") and inst and inst not in seen:
                seen.add(inst)
                out.append(inst)
        return out

    def depth_prompt(self) -> str:
        return {
            "minimal": "Explainability: state only the finding in one short clause; no rationale.",
            "standard": "Explainability: one-sentence finding plus a brief reason and your confidence.",
            "detailed": "Explainability: the finding, the specific vocal signals and their trend across "
                        "chunks, corroborating transcript, and your confidence.",
        }[self.explainability]


_config: contextvars.ContextVar[ClinicianConfig] = contextvars.ContextVar(
    "clinician_config", default=ClinicianConfig())


def set_config(cfg: ClinicianConfig) -> None:
    _config.set(cfg)


def current_config() -> ClinicianConfig:
    return _config.get()
