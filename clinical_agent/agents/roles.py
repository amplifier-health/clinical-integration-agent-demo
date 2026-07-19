import json
import urllib.parse

import httpx

from clinical_agent.agents.base import ClaudeAgent
from clinical_agent.clinician_config import current_config
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.store import PatientStore

_NEVER_DIAGNOSE = (
    "You never diagnose. You describe vocal biomarker signals and transcript observations, "
    "and you may suggest validated screening instruments (e.g. PHQ-9, GAD-7) for clinician review. "
    "All chart changes are drafts a clinician must approve."
)

_QUALITATIVE = (
    "You are given the FULL voice-biomarker object — the condition signals, the wellness dimensions "
    "(each with anchor words, e.g. invigorated↔exhausted), and the speech/prosody metrics (pitch, "
    "loudness, speech rate). Reason over ALL of it: use the wellness and prosody context to "
    "corroborate or temper the condition signals and describe the coherent clinical picture — do not "
    "just restate individual signals. NEVER surface raw numeric scores, probabilities, or metric "
    "values to the clinician; translate everything into qualitative clinical language (e.g. 'markedly "
    "elevated', 'reduced pitch variability consistent with flat affect'). A number like 0.08 is "
    "meaningless to a physician. "
)

PREVISIT_SYSTEM = (
    "You are a pre-visit preparation agent for a clinician. Given a patient's chart — visits, "
    "ICD-10 codes, prior vocal biomarker results and visit summaries — produce a short brief: "
    "vocal signal trends across visits, discordance between what the voice showed and what the "
    "chart coded, and concrete topics to discuss today. " + _QUALITATIVE + _NEVER_DIAGNOSE
)

REASONER_SYSTEM = (
    "You are a clinical reasoning agent running during a live visit. You receive vocal biomarker "
    "signals for the latest audio window plus the running transcript. "
    "Default to silence: reply with exactly 'nothing notable' UNLESS there is a specific, "
    "high-confidence, actionable signal a clinician must know right now — a clear discordance "
    "between the patient's words and their vocal signals, or a strong emerging trend. When you do "
    "speak, use at most one sentence; never restate the transcript or enumerate every signal. "
    "Favor precision over recall — a missed minor cue is fine; a noisy false alarm is not. "
    "Never suggest 'establishing a baseline' or generic monitoring — say something specific or nothing. "
    "Use the read_chart tool only if chart history would change your assessment. " + _QUALITATIVE + _NEVER_DIAGNOSE
)

POSTVISIT_SYSTEM = (
    "You are a post-visit documentation agent. Produce a visit summary with a vocal-findings "
    "section, transcript findings, any voice/words discordance, screener recommendations, a chart "
    "update draft for clinician approval, and topics to raise at the next visit. "
    "Be high-precision and conservative: include only findings you are confident about, and only "
    "vocal findings that are actually flagged (consider/moderate/elevated) — omit weak, borderline, "
    "or speculative signals entirely. Prefer a few strong items over a long list; a clinician's "
    "attention is scarce. " + _QUALITATIVE + _NEVER_DIAGNOSE
)

VISITNOTE_SYSTEM = (
    "You are a clinical documentation agent. From a visit transcript and the ICD-10 codes the "
    "clinician assigned to THIS encounter, write a concise, factual visit note in SOAP form. "
    "Ground every statement in the transcript. Treat the ICD-10 codes as the clinician's coded "
    "assessment for this visit and reflect each one in the Assessment. Do not invent findings. "
    + _QUALITATIVE + _NEVER_DIAGNOSE
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

VISITNOTE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "chief_complaint": {"type": "string"},
        "subjective": {"type": "string"},
        "objective": {"type": "string"},
        "assessment": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"code": {"type": "string"}, "description": {"type": "string"},
                           "note": {"type": "string"}},
            "required": ["code", "description", "note"]}},
        "plan": _STR_ARR,
    },
    "required": ["chief_complaint", "subjective", "objective", "assessment", "plan"],
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


async def pre_visit_brief(settings: Settings, bus: EventBus, store: PatientStore, pid: str,
                          before: int | None = None) -> dict:
    agent = ClaudeAgent("previsit", settings, bus)
    chart = json.dumps(store.chart(pid, before=before), indent=1)
    text = await agent.run(PREVISIT_SYSTEM, f"Patient chart:\n{chart}", output_schema=PREVISIT_SCHEMA)
    brief = json.loads(text)
    planned = next((v for v in store.list_visits(pid) if v.status == "planned"), None)
    if planned is not None:
        store.write_artifact(pid, planned.number, "pre_visit_brief", brief)
    await bus.emit("pre_visit_brief", patient=pid, **brief)
    return brief


def _transcript_text(parts) -> str:
    """Flatten a stored transcript to plain text. Handles both shapes we persist:
    historical `[{speaker, text}]` (from GCS diarization) and live `[{chunk, text}]`."""
    if not parts:
        return ""
    if isinstance(parts, str):
        return parts
    lines = []
    for p in parts:
        if isinstance(p, dict):
            who = p.get("speaker") or (f"chunk {p['chunk']}" if "chunk" in p else "")
            lines.append(f"{who + ': ' if who else ''}{p.get('text', '')}")
        else:
            lines.append(str(p))
    return "\n".join(lines)


def _code_lines(icd10) -> str:
    out = []
    for c in icd10 or []:
        code = c.get("code") if isinstance(c, dict) else c.code
        desc = c.get("description") if isinstance(c, dict) else c.description
        out.append(f"- {code}: {desc}")
    return "\n".join(out) or "(no ICD-10 codes assigned to this visit)"


async def build_visit_note(settings: Settings, bus: EventBus, store: PatientStore, pid: str,
                           visit_no: int, transcript, icd10, signals: list | None = None,
                           history: list | None = None) -> dict:
    """Draft a basic SOAP visit note from the transcript + the ICD-10 codes coded for this
    visit (and any vocal signals). Stores a `note` artifact and emits `visit_note`."""
    agent = ClaudeAgent("notewriter", settings, bus)
    user = (f"Visit transcript:\n{_transcript_text(transcript) or '(no transcript available)'}\n\n"
            f"ICD-10 codes the clinician assigned to THIS visit:\n{_code_lines(icd10)}")
    if signals:
        user += f"\n\nVocal-biomarker signals observed this visit:\n{json.dumps(signals)}"
    note = json.loads(await agent.run(VISITNOTE_SYSTEM, user, output_schema=VISITNOTE_SCHEMA,
                                      history=history))
    store.write_artifact(pid, visit_no, "note", note)
    await bus.emit("visit_note", patient=pid, visit=visit_no, **note)
    return note


async def reason_over_chunk(settings: Settings, bus: EventBus, store: PatientStore, pid: str,
                            chunk_no: int, transcript: str, cumulative_signals: list,
                            brief: dict, history: list | None = None) -> str:
    agent = ClaudeAgent("reasoner", settings, bus)

    async def _read_chart(_input: dict) -> str:
        return json.dumps(store.chart(pid))

    tools = {"read_chart": ({"name": "read_chart",
                             "description": "Read the patient's full chart: visits, ICD-10 codes, "
                                            "prior vocal results and summaries.",
                             "input_schema": {"type": "object", "properties": {},
                                              "additionalProperties": False}}, _read_chart)}
    # With a shared visit history, only the NEW information for this chunk is sent — the
    # brief and prior chunks are already in the conversation memory.
    if history:
        user = (f"LIVE TICK — chunk {chunk_no}.\nNew transcript: {transcript}\n"
                f"New cumulative signals: {json.dumps(cumulative_signals)}\n"
                "Triage: note only what a clinician should know right now; otherwise say so briefly.")
    else:
        user = (f"Pre-visit brief: {json.dumps(brief)}\n"
                f"Chunk {chunk_no} transcript: {transcript}\n"
                f"Cumulative signals so far: {json.dumps(cumulative_signals)}")
    system = REASONER_SYSTEM + " " + current_config().depth_prompt()  # Setting 2: explanation depth
    text = await agent.run(system, user, tools=tools, effort="low", history=history)
    await bus.emit("observation", patient=pid, chunk=chunk_no, text=text)
    return text


# Validated screening instrument per vocal signal (voice flag → offer this screener).
_SIGNAL_SCREENERS = {
    "anxiety": ("GAD-7", "Anxiety"),
    "mood-disruption": ("PHQ-9", "Depression"),
}


def _pubmed_tool(bus: EventBus, pid: str):
    """A tool that searches PubMed (NCBI E-utilities) and emits a `literature_ref` per article.
    Used only at 'detailed' explanation depth, to let the agent cite supporting evidence."""
    async def _search(inp: dict) -> str:
        query = inp.get("query", "")
        n = int(inp.get("max_results", 3))
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        try:
            async with httpx.AsyncClient(timeout=30) as h:
                ids = (await h.get(f"{base}/esearch.fcgi", params={
                    "db": "pubmed", "term": query, "retmax": n, "retmode": "json",
                    "sort": "relevance"})).raise_for_status().json()
                idlist = ids.get("esearchresult", {}).get("idlist", [])
                if not idlist:
                    return "no PubMed results"
                summ = (await h.get(f"{base}/esummary.fcgi", params={
                    "db": "pubmed", "id": ",".join(idlist), "retmode": "json"})).raise_for_status().json()
        except Exception as exc:
            return f"PubMed search failed: {exc}"
        result, found = summ.get("result", {}), []
        for pmid in result.get("uids", []):
            a = result.get(pmid, {})
            title = a.get("title", "(untitled)")
            journal = a.get("fulljournalname") or a.get("source")
            year = (a.get("pubdate") or "")[:4]
            await bus.emit("literature_ref", patient=pid, pmid=pmid, title=title,
                           journal=journal, year=year, query=query,
                           url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
            found.append(f"{pmid}: {title}")
        return "found: " + " | ".join(found) if found else "no PubMed results"

    return {"search_pubmed": ({
        "name": "search_pubmed",
        "description": "Search PubMed for peer-reviewed literature supporting a clinical finding "
                       "(e.g. a vocal-biomarker association). Emits a literature_ref per article.",
        "input_schema": {"type": "object", "additionalProperties": False,
                         "properties": {"query": {"type": "string"},
                                        "max_results": {"type": "integer"}},
                         "required": ["query"]}}, _search)}


def _clinical_trials_tool(bus: EventBus, pid: str):
    """A tool that searches ClinicalTrials.gov and emits a `trial_match` per result."""
    async def _search(inp: dict) -> str:
        condition = inp.get("condition", "")
        n = int(inp.get("max_results", 3))

        async def _q(recruiting_only: bool):
            # NB: no `fields` filter — the v2 API rejects display-name fields (400); the full
            # study object carries protocolSection.identificationModule, which we read below.
            params = {"query.cond": condition, "pageSize": n}
            if recruiting_only:
                params["filter.overallStatus"] = "RECRUITING"
            url = "https://clinicaltrials.gov/api/v2/studies?" + urllib.parse.urlencode(params)
            async with httpx.AsyncClient(timeout=30) as h:
                r = await h.get(url)
                r.raise_for_status()
                return r.json()
        try:
            data = await _q(True)
            if not data.get("studies"):
                data = await _q(False)  # fall back to any status so we still surface matches
        except Exception as exc:
            return f"trial search failed: {exc}"
        found = []
        for st in data.get("studies", [])[:n]:
            idm = st.get("protocolSection", {}).get("identificationModule", {})
            nct, title = idm.get("nctId", "?"), (idm.get("briefTitle") or "(untitled)")
            await bus.emit("trial_match", patient=pid, nct_id=nct, title=title,
                           why_relevant=f"Matched on '{condition}' (the undiscussed gap).",
                           eligibility_hint=condition)
            found.append(f"{nct}: {title}")
        return "found: " + " | ".join(found) if found else "no trials found"

    return {"search_clinical_trials": ({
        "name": "search_clinical_trials",
        "description": "Search ClinicalTrials.gov for trials relevant to a condition the visit did "
                       "NOT focus on. Emits a trial_match per result.",
        "input_schema": {"type": "object", "additionalProperties": False,
                         "properties": {"condition": {"type": "string"},
                                        "max_results": {"type": "integer"}},
                         "required": ["condition"]}}, _search)}


async def post_visit_summary(settings: Settings, bus: EventBus, store: PatientStore, pid: str,
                             visit_no: int, transcript_parts: list, all_signals: list,
                             observations: list, brief: dict, history: list | None = None) -> dict:
    agent = ClaudeAgent("postvisit", settings, bus)
    if history:  # the visit is already in memory — just ask for the final structured analysis
        user = ("POST-VISIT. The recording has stopped. Using everything you observed this visit "
                "(all in your memory above), produce the final structured summary.")
    else:
        user = (f"Pre-visit brief: {json.dumps(brief)}\n"
                f"Full transcript: {' '.join(transcript_parts)}\n"
                f"All chunk signals: {json.dumps(all_signals)}\n"
                f"Live observations: {json.dumps(observations)}")
    system = POSTVISIT_SYSTEM + " " + current_config().depth_prompt()  # Setting 2: explanation depth
    summary = json.loads(await agent.run(system, user, output_schema=POSTVISIT_SCHEMA,
                                         history=history))
    store.write_artifact(pid, visit_no, "summary", summary)
    await bus.emit("visit_summary", patient=pid, visit=visit_no, summary=summary["summary"],
                   vocal_findings=summary["vocal_findings"], discordance=summary["discordance"],
                   screener_recommendations=summary["screener_recommendations"])
    await bus.emit("chart_draft", patient=pid, visit=visit_no, items=summary["chart_update_draft"])
    await bus.emit("topics", patient=pid, visit=visit_no, items=summary["next_visit_topics"])

    # Suggested screening — deterministic (validated instrument per flagged signal), so it works
    # even in mock mode. Opt-in: only when the clinician enabled the screener output.
    if current_config().output_enabled("screener_suggested"):
        offered = set()
        for s in all_signals:
            if not s.get("flagged"):
                continue
            m = _SIGNAL_SCREENERS.get(s.get("name"))
            if m and m[0] not in offered:
                offered.add(m[0])
                await bus.emit("screener_suggested", patient=pid, visit=visit_no,
                               instrument=m[0], condition=m[1], signal=s.get("name"))

    # Clinical trials — only when the clinician enabled that output (opt-in; needs live Claude to
    # call the tool). Searches for the single undiscussed gap; emits a trial_match per result.
    if current_config().output_enabled("trial_match") and not settings.mock_claude:
        gap = "; ".join(summary.get("next_visit_topics", [])) or summary.get("discordance", "")
        if gap:
            trials = ClaudeAgent("trials", settings, bus)
            await trials.run(
                POSTVISIT_SYSTEM,
                f"For the follow-up gap you identified (\"{gap}\"), call search_clinical_trials to "
                "surface relevant trials the clinician might consider. Search the single most "
                "important undiscussed condition.",
                tools=_clinical_trials_tool(bus, pid), effort="low", history=history)

    # Supporting literature — ONLY at 'detailed' explanation depth (needs live Claude to call the
    # tool). Cites PubMed evidence for the visit's main vocal finding.
    if current_config().explainability == "detailed" and not settings.mock_claude:
        finding = (summary.get("vocal_findings") or [{}])[0].get("sign") or summary.get("discordance", "")
        if finding:
            lit = ClaudeAgent("literature", settings, bus)
            await lit.run(
                POSTVISIT_SYSTEM,
                f"Call search_pubmed to find peer-reviewed evidence relating the vocal-biomarker "
                f"finding \"{finding}\" to its condition (e.g. voice biomarkers and that condition). "
                "Cite the most relevant results.",
                tools=_pubmed_tool(bus, pid), effort="low", history=history)
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
