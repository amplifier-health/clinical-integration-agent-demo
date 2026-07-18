"""The output contract: typed, versioned models for every event the agent emits.

Enforcement model
-----------------
These pydantic models are the single source of truth. `EventBus.emit` validates
every event's payload against the model registered for its `type` (drift → error,
because every model is `extra="forbid"`), then stamps the shared envelope fields.

Wire format is FLAT — envelope fields and payload fields sit at the top level of
the SSE frame — so existing UI consumers that read `d.chunk` / `d.text` keep
working. The envelope adds `contract_version`, `session_id`, `seq`, and `phase`
alongside them.

`scripts/dump_contract.py` renders these models to JSON Schema + examples in
`docs/contract/` — the artifact a plugin consumer integrates against.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

CONTRACT_VERSION = "1.0"

TIER = Literal["inconclusive", "none", "low", "consider", "moderate", "elevated", "high"]
PHASE = Literal["lifecycle", "pre_visit", "live", "post_visit", "longitudinal", "telemetry", "reasoning"]


class _Payload(BaseModel):
    """Base for every event payload. Forbids unknown fields so schema drift is a
    hard error at the emit site, not a silent shape change on the wire."""
    model_config = ConfigDict(extra="forbid")


# ---- shared sub-objects (referenced across events) ----------------------------

class Signal(BaseModel):
    # extra="allow": signals in `api_job_result` come straight from the Amplifier API,
    # whose objects may carry fields we don't model. Tolerate and preserve them on the
    # wire (don't reject live results) while still type-checking the fields we rely on.
    model_config = ConfigDict(extra="allow")
    name: str
    score: Optional[float] = None
    level: TIER
    flagged: Optional[bool] = None
    label: Optional[str] = None
    model: Optional[str] = None


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["voice", "transcript", "chart"] = "voice"
    quote: Optional[str] = None
    t_start: Optional[float] = None
    chunk: Optional[int] = None


class Explanation(BaseModel):
    """Per-event 'why'. Depth-controlled: `minimal` populates summary only,
    `standard` adds confidence, `detailed` adds drivers + trend."""
    model_config = ConfigDict(extra="forbid")
    summary: str
    confidence: Optional[float] = None
    drivers: list[str] = []
    trend: list[float] = []


# ---- lifecycle ----------------------------------------------------------------

class VisitStarted(_Payload):
    patient: str
    visit: int
    date: str
    reason: str


class VisitComplete(_Payload):
    patient: str
    visit: int


class ErrorEvent(_Payload):
    patient: Optional[str] = None
    chunk: Optional[int] = None
    message: str


# ---- telemetry (the machine working; namespaced, non-clinical) ----------------

class ChunkCreated(_Payload):
    patient: Optional[str] = None
    chunk: int
    start_s: float
    end_s: float


class Transcript(_Payload):
    patient: Optional[str] = None
    chunk: int
    text: str


class ApiJobCreated(_Payload):
    patient: Optional[str] = None
    chunk: int
    model: Optional[str] = None
    job_id: str


class ApiJobResult(_Payload):
    patient: Optional[str] = None
    chunk: int
    cached: bool
    model: Optional[str] = None
    offline_miss: Optional[bool] = None
    signals: Optional[list[Signal]] = None
    summary: Optional[dict] = None


class AgentToken(_Payload):
    agent: str
    text: str
    cached: Optional[bool] = None


class AgentToolCall(_Payload):
    agent: str
    tool: str
    input: dict


# ---- reasoning ----------------------------------------------------------------

class Observation(_Payload):
    patient: str
    chunk: int
    text: str


# ---- clinical output ----------------------------------------------------------

class PreVisitBrief(_Payload):
    patient: str
    brief: str
    vocal_trends: list[str]
    topics_to_discuss: list[str]


class VocalFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sign: str
    level: str
    note: str


class VisitSummary(_Payload):
    patient: str
    visit: int
    summary: str
    vocal_findings: list[VocalFinding]
    discordance: str
    screener_recommendations: list[str]


class ChartDraftItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    rationale: str


class ChartDraft(_Payload):
    patient: str
    visit: int
    items: list[ChartDraftItem]


class Topics(_Payload):
    patient: str
    visit: int
    items: list[str]


class AssessmentItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    description: str
    note: str


class VisitNote(_Payload):
    """SOAP visit note (a `documentation` artifact)."""
    patient: str
    visit: int
    chief_complaint: str
    subjective: str
    objective: str
    assessment: list[AssessmentItem]
    plan: list[str]


class TrialMatch(_Payload):
    patient: str
    nct_id: str
    title: str
    why_relevant: Optional[str] = None
    eligibility_hint: Optional[str] = None


class LongitudinalDelta(_Payload):
    patient: str
    condition: str
    first_voice_flag_visit: int
    first_coded_visit: int
    visits_early: int


class LongitudinalNarrative(_Payload):
    patient: str
    text: str


# ---- registry: type string -> (model, phase) ---------------------------------

REGISTRY: dict[str, tuple[type[_Payload], PHASE]] = {
    "visit_started": (VisitStarted, "lifecycle"),
    "visit_complete": (VisitComplete, "lifecycle"),
    "error": (ErrorEvent, "lifecycle"),
    "chunk_created": (ChunkCreated, "telemetry"),
    "transcript": (Transcript, "telemetry"),
    "api_job_created": (ApiJobCreated, "telemetry"),
    "api_job_result": (ApiJobResult, "telemetry"),
    "agent_token": (AgentToken, "telemetry"),
    "agent_tool_call": (AgentToolCall, "reasoning"),
    "observation": (Observation, "reasoning"),
    "pre_visit_brief": (PreVisitBrief, "pre_visit"),
    "visit_summary": (VisitSummary, "post_visit"),
    "chart_draft": (ChartDraft, "post_visit"),
    "topics": (Topics, "post_visit"),
    "visit_note": (VisitNote, "post_visit"),
    "trial_match": (TrialMatch, "post_visit"),
    "longitudinal_delta": (LongitudinalDelta, "longitudinal"),
    "longitudinal_narrative": (LongitudinalNarrative, "longitudinal"),
}

# Which types are the stable, sellable clinical contract vs. telemetry/reasoning noise.
CLINICAL_TYPES = {
    "pre_visit_brief", "visit_summary", "chart_draft", "topics", "visit_note",
    "trial_match", "longitudinal_delta", "longitudinal_narrative",
}


def validate(type: str, data: dict) -> dict:
    """Validate a payload against its registered model; return the normalized dict.
    Unregistered types pass through untouched (telemetry we haven't modeled yet)."""
    entry = REGISTRY.get(type)
    if entry is None:
        return data
    model, _phase = entry
    return model.model_validate(data).model_dump(exclude_none=True)


def phase_for(type: str) -> PHASE:
    entry = REGISTRY.get(type)
    return entry[1] if entry else "telemetry"
