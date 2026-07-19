"""Per-clinician settings that control agent behavior at the point of care.

Distinct from `Settings` (deployment/env): this is clinical behavior a clinician
tunes from the UI. Sent per-visit on `POST /visits/start` and bound to the visit
via a contextvar, so the agents read it without threading it through every call.

This module currently implements **Setting 2: explanation depth**. Alert
sensitivity and screener controls are intended to land here as sibling fields.
"""
import contextvars
from typing import Literal

from pydantic import BaseModel, ConfigDict

_DEPTH_PROMPT = {
    "minimal": "Explanation depth = minimal: state only the finding in one short clause; no rationale.",
    "standard": "Explanation depth = standard: a one-sentence finding plus a brief reason and your confidence.",
    "detailed": "Explanation depth = detailed: the finding, the specific vocal signals and their trend "
                "across chunks, any corroborating transcript, and your confidence.",
}


class ClinicianConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    explainability: Literal["minimal", "standard", "detailed"] = "standard"
    # Setting: enabled clinical outputs. None = all enabled (full contract). A list names
    # the clinical event types the clinician wants; others are suppressed at the emit gate.
    enabled_outputs: list[str] | None = None

    def depth_prompt(self) -> str:
        return _DEPTH_PROMPT[self.explainability]

    def output_enabled(self, clinical_type: str) -> bool:
        return self.enabled_outputs is None or clinical_type in self.enabled_outputs

    @classmethod
    def from_override(cls, override: dict | None) -> "ClinicianConfig":
        """Build from a per-request config dict (unknown keys ignored, so the UI can
        send fields this version doesn't implement yet without erroring)."""
        if not override:
            return cls()
        known = {k: v for k, v in override.items() if k in cls.model_fields}
        return cls.model_validate(known)


_config: contextvars.ContextVar[ClinicianConfig] = contextvars.ContextVar(
    "clinician_config", default=ClinicianConfig())


def set_config(cfg: ClinicianConfig) -> None:
    _config.set(cfg)


def current_config() -> ClinicianConfig:
    return _config.get()
