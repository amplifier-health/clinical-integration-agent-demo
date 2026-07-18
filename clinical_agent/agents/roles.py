import json

from clinical_agent.agents.base import ClaudeAgent
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.store import PatientStore

_NEVER_DIAGNOSE = (
    "You never diagnose. You describe vocal biomarker signals and transcript observations, "
    "and you may suggest validated screening instruments (e.g. PHQ-9, GAD-7) for clinician review. "
    "All chart changes are drafts a clinician must approve."
)

PREVISIT_SYSTEM = (
    "You are a pre-visit preparation agent for a clinician. Given a patient's chart — visits, "
    "ICD-10 codes, prior vocal biomarker results and visit summaries — produce a short brief: "
    "vocal signal trends across visits, discordance between what the voice showed and what the "
    "chart coded, and concrete topics to discuss today. " + _NEVER_DIAGNOSE
)

REASONER_SYSTEM = (
    "You are a clinical reasoning agent running during a live visit. You receive vocal biomarker "
    "signals for the latest audio chunk plus the running transcript. In one or two sentences, note "
    "anything a clinician should know now — especially discordance between the patient's words and "
    "their vocal signals, or trends across chunks. Use the read_chart tool if chart history would "
    "change your assessment. If nothing is notable, say so in a few words. " + _NEVER_DIAGNOSE
)

POSTVISIT_SYSTEM = (
    "You are a post-visit documentation agent. Produce a visit summary with a vocal-findings "
    "section, transcript findings, any voice/words discordance, screener recommendations, a chart "
    "update draft for clinician approval, and topics to raise at the next visit. " + _NEVER_DIAGNOSE
)

LONGITUDINAL_SYSTEM = (
    "You are a longitudinal analyst. Compare, per condition, when vocal biomarkers first flagged a "
    "signal versus when the condition (or a related ICD-10 code) first appeared in the chart. "
    "Report each gap and a short narrative of the early-detection story. " + _NEVER_DIAGNOSE
)

_STR_ARR = {"type": "array", "items": {"type": "string"}}

PREVISIT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"brief": {"type": "string"}, "vocal_trends": _STR_ARR, "topics_to_discuss": _STR_ARR},
    "required": ["brief", "vocal_trends", "topics_to_discuss"],
}

POSTVISIT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "vocal_findings": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"sign": {"type": "string"}, "level": {"type": "string"}, "note": {"type": "string"}},
            "required": ["sign", "level", "note"]}},
        "transcript_findings": _STR_ARR,
        "discordance": {"type": "string"},
        "screener_recommendations": _STR_ARR,
        "chart_update_draft": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"description": {"type": "string"}, "rationale": {"type": "string"}},
            "required": ["description", "rationale"]}},
        "next_visit_topics": _STR_ARR,
    },
    "required": ["summary", "vocal_findings", "transcript_findings", "discordance",
                 "screener_recommendations", "chart_update_draft", "next_visit_topics"],
}

LONGITUDINAL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "narrative": {"type": "string"},
        "deltas": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"condition": {"type": "string"},
                           "first_voice_flag_visit": {"type": "integer"},
                           "first_coded_visit": {"type": "integer"},
                           "visits_early": {"type": "integer"}},
            "required": ["condition", "first_voice_flag_visit", "first_coded_visit", "visits_early"]}},
    },
    "required": ["narrative", "deltas"],
}


async def pre_visit_brief(settings: Settings, bus: EventBus, store: PatientStore, pid: str) -> dict:
    agent = ClaudeAgent("previsit", settings, bus)
    chart = json.dumps(store.chart(pid), indent=1)
    text = await agent.run(PREVISIT_SYSTEM, f"Patient chart:\n{chart}", output_schema=PREVISIT_SCHEMA)
    brief = json.loads(text)
    planned = next((v for v in store.list_visits(pid) if v.status == "planned"), None)
    if planned is not None:
        store.write_artifact(pid, planned.number, "pre_visit_brief", brief)
    await bus.emit("pre_visit_brief", patient=pid, **brief)
    return brief


async def reason_over_chunk(settings: Settings, bus: EventBus, store: PatientStore, pid: str,
                            chunk_no: int, transcript: str, cumulative_signals: list,
                            brief: dict) -> str:
    agent = ClaudeAgent("reasoner", settings, bus)

    async def _read_chart(_input: dict) -> str:
        return json.dumps(store.chart(pid))

    tools = {"read_chart": ({"name": "read_chart",
                             "description": "Read the patient's full chart: visits, ICD-10 codes, "
                                            "prior vocal results and summaries.",
                             "input_schema": {"type": "object", "properties": {},
                                              "additionalProperties": False}}, _read_chart)}
    user = (f"Pre-visit brief: {json.dumps(brief)}\n"
            f"Chunk {chunk_no} transcript: {transcript}\n"
            f"Cumulative signals so far: {json.dumps(cumulative_signals)}")
    text = await agent.run(REASONER_SYSTEM, user, tools=tools, effort="low")
    await bus.emit("observation", patient=pid, chunk=chunk_no, text=text)
    return text


async def post_visit_summary(settings: Settings, bus: EventBus, store: PatientStore, pid: str,
                             visit_no: int, transcript_parts: list, all_signals: list,
                             observations: list, brief: dict) -> dict:
    agent = ClaudeAgent("postvisit", settings, bus)
    user = (f"Pre-visit brief: {json.dumps(brief)}\n"
            f"Full transcript: {' '.join(transcript_parts)}\n"
            f"All chunk signals: {json.dumps(all_signals)}\n"
            f"Live observations: {json.dumps(observations)}")
    summary = json.loads(await agent.run(POSTVISIT_SYSTEM, user, output_schema=POSTVISIT_SCHEMA))
    store.write_artifact(pid, visit_no, "summary", summary)
    await bus.emit("visit_summary", patient=pid, visit=visit_no, summary=summary["summary"],
                   vocal_findings=summary["vocal_findings"], discordance=summary["discordance"],
                   screener_recommendations=summary["screener_recommendations"])
    await bus.emit("chart_draft", patient=pid, visit=visit_no, items=summary["chart_update_draft"])
    await bus.emit("topics", patient=pid, visit=visit_no, items=summary["next_visit_topics"])
    return summary


async def longitudinal_analysis(settings: Settings, bus: EventBus, store: PatientStore, pid: str) -> dict:
    agent = ClaudeAgent("longitudinal", settings, bus)
    chart = json.dumps(store.chart(pid), indent=1)
    out = json.loads(await agent.run(LONGITUDINAL_SYSTEM, f"Patient chart:\n{chart}",
                                     output_schema=LONGITUDINAL_SCHEMA))
    for d in out["deltas"]:
        await bus.emit("longitudinal_delta", patient=pid, **d)
    await bus.emit("longitudinal_narrative", patient=pid, text=out["narrative"])
    return out
