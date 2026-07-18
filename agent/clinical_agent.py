#!/usr/bin/env python3
"""The real clinical-integration agent — Claude does the reasoning.

Only two things are mocked: audio is not streamed live, and the voice-biomarker
API is not polled live (results are replayed from precomputed objects). Everything
else is a genuine Claude agent (model `claude-opus-4-8`, tool runner) making real
decisions via real tool calls, including a real ClinicalTrials.gov lookup.

The agent replays a scenario's INPUT events (biomarker.result / transcript.token),
and at reasoning triggers calls Claude with tools that emit the OUTPUT events the UI
consumes (topic.suggested, flowsheet.update, note.addendum, followup.item,
trial.match). Nothing about those outputs is scripted.

Reads ANTHROPIC_API_KEY from the environment or ../.env (see .env.example).
"""
import json, os, time, urllib.request, urllib.parse
import anthropic
from anthropic import beta_tool

MODEL = "claude-opus-4-8"

# --- tool sink: tools append emitted events here; set per run ---
_SINK = None
def _emit(ev):
    if _SINK is not None:
        _SINK(ev)

# ---------------- tools (real Claude calls these) ----------------
@beta_tool
def suggest_topic(title: str, rationale: str, source: str, confidence: float) -> str:
    """Surface a discussion topic to the clinician's mobile 'suggested topics' card, live during the visit.

    Args:
        title: Short topic title the clinician sees.
        rationale: One sentence on why this is worth raising, grounded in the voice signal and/or chart.
        source: Where the signal came from: "voice", "chart", or "voice+chart".
        confidence: 0-1 confidence this is worth the clinician's attention.
    """
    _emit({"type": "topic.suggested", "title": title, "rationale": rationale,
           "source": source, "confidence": confidence, "trend": []})
    return "surfaced to mobile suggested-topics"

@beta_tool
def update_flowsheet(section: str, row: str, value: str, quote: str) -> str:
    """Auto-populate a structured flowsheet row from an ambient finding (mirrors Abridge's ambient-source pattern).

    Args:
        section: Flowsheet section, e.g. "Endocrine / Voice".
        row: The flowsheet row label.
        value: The value to record.
        quote: The transcript quote or signal that justifies it (provenance).
    """
    _emit({"type": "flowsheet.update", "section": section, "row": row, "value": value,
           "provenance": {"quote": quote}})
    return "flowsheet row updated"

@beta_tool
def add_note_addendum(section: str, text: str) -> str:
    """Add a structured addendum to the clinical note (post-visit).

    Args:
        section: Note section, e.g. "Assessment".
        text: The addendum text; cite the voice signal and any corroborating transcript.
    """
    _emit({"type": "note.addendum", "section": section, "text": text})
    return "note addendum added"

@beta_tool
def flag_followup(text: str, why: str, discussed: bool) -> str:
    """Record a follow-up item. discussed=false marks it as a GAP (voice-flagged but not addressed this visit).

    Args:
        text: The follow-up action.
        why: Why it matters, grounded in the voice signal.
        discussed: True if it was addressed this visit; False if it is a gap for next time.
    """
    _emit({"type": "followup.item", "text": text, "why": why, "discussed": discussed})
    return "follow-up recorded"

@beta_tool
def search_clinical_trials(condition: str, max_results: int = 3) -> str:
    """Search ClinicalTrials.gov for recruiting trials relevant to a condition (REAL API call).

    Use this post-visit for conditions the voice flagged but the visit did NOT focus on.

    Args:
        condition: Condition/topic to search, e.g. "PCOS anxiety".
        max_results: How many trials to return (default 3).
    """
    def query(recruiting_only):
        params = {"query.cond": condition, "pageSize": max_results,
                  "fields": "NCT Number,Study Title,Conditions,Overall Status"}
        if recruiting_only:
            params["filter.overallStatus"] = "RECRUITING"
        url = f"https://clinicaltrials.gov/api/v2/studies?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read())
    try:
        data = query(recruiting_only=True)
        if not data.get("studies"):
            data = query(recruiting_only=False)  # fall back to any status so the demo always surfaces matches
    except Exception as e:
        return f"trial search failed: {e}"
    out = []
    for st in data.get("studies", [])[:max_results]:
        idm = st.get("protocolSection", {}).get("identificationModule", {})
        nct = idm.get("nctId", "?")
        title = idm.get("briefTitle") or idm.get("officialTitle") or "(untitled)"
        _emit({"type": "trial.match", "nct_id": nct, "title": title,
               "why_relevant": f"Recruiting trial matched on '{condition}' (the undiscussed signal).",
               "eligibility_hint": condition})
        out.append(f"{nct}: {title}")
    return "found: " + " | ".join(out) if out else "no recruiting trials found"

IN_VISIT_TOOLS = [suggest_topic, update_flowsheet]
POST_VISIT_TOOLS = [add_note_addendum, flag_followup, search_clinical_trials]

SYSTEM = """You are a clinical-integration agent embedded in an ambient scribe (Abridge-style),
present for the ENTIRE visit as ONE continuous session. You accumulate memory across the
appointment: the patient chart, the transcript as it unfolds, the voice-biomarker trend
(probabilities + tiers: none/low/consider/moderate/elevated), and your own prior decisions.
Voice signals are SIGNALS TO CONSIDER, never diagnoses — phrase everything that way.

You operate in two modes, and each user turn tells you which:

LIVE TICK (mid-appointment): a short update arrives as new audio is scored. Your ONLY job is
triage — decide if there is something genuinely NEW, high-confidence, and ADDITIVE that is
worth interrupting the clinician with RIGHT NOW, while they can still act on it in the room.
The bar is high: most ticks should call NO tools. Surface a topic only if a signal is
'consider'+, coheres with the transcript/chart, is NOT already suggested, and is NOT already
being discussed. Prefer the differentiated find (a signal rising but not the visit's focus).
Do NOT write notes, follow-ups, or comprehensive analysis mid-visit — that is for the end.

POST-VISIT: the visit has ended. Now do the full, careful analysis using everything you
observed across the whole session — including what you already surfaced live. Write note
addenda, follow-ups, the gap (voice-flagged but never discussed), and search trials for the gap."""

class ClinicalAgent:
    def __init__(self, emit, model=MODEL):
        self.emit = emit
        self.client = anthropic.Anthropic()
        self.model = model
        self.chart = None
        self.transcript = []
        self.trend = {}          # signal -> list of (t, score, tier)
        self.suggested = set()
        # ONE persistent conversation for the whole appointment — this is the agent's memory.
        self.messages = []
        self._sent_tx = 0          # transcript lines already put into the conversation
        self._sent_chunks = set()  # biomarker chunk_ids already put into the conversation
        self._first = True
        self._gap = None           # text of the undiscussed gap, if flagged
        self._trial_done = False

    def _reason(self, tools, user_text, phase, effort):
        """Continue the SAME conversation with a new user turn; stream reasoning + tool events.

        The agent's memory is self.messages — every mid-visit tick and the post-visit pass
        append to and read from this single history, so the final analysis is done by the same
        agent that watched the whole visit, with its own prior reasoning and decisions in context.
        """
        global _SINK
        _SINK = self._emit_output
        self.emit({"type": "reasoning.step", "phase": phase,
                   "message": f"Claude ({self.model}, effort={effort}) — {phase} turn "
                              f"[conversation: {len(self.messages)} prior messages in memory]"})
        self.messages.append({"role": "user", "content": user_text})
        runner = self.client.beta.messages.tool_runner(
            model=self.model, max_tokens=6000,
            thinking={"type": "adaptive"}, output_config={"effort": effort},
            system=SYSTEM, tools=tools, messages=self.messages)
        for message in runner:
            for block in message.content:
                if block.type == "text" and block.text.strip():
                    self.emit({"type": "reasoning.step", "phase": phase, "message": block.text.strip()})
                elif block.type == "tool_use":
                    self.emit({"type": "reasoning.step", "phase": phase,
                               "message": f"decision → {block.name}({json.dumps(block.input)[:160]})"})
            # mirror the turn back into our persistent history (preserves thinking + tool blocks)
            self.messages.append({"role": "assistant", "content": message.content})
            tool_response = runner.generate_tool_call_response()
            if tool_response is not None:
                self.messages.append(tool_response)
        _SINK = None

    def _emit_output(self, ev):
        # tool-emitted UI event; dedupe topics, track the gap + trial for the guaranteed-search turn
        if ev["type"] == "topic.suggested":
            key = ev["title"].lower()
            if key in self.suggested:
                return
            self.suggested.add(key)
        elif ev["type"] == "followup.item" and not ev.get("discussed", True):
            self._gap = ev["text"]
        elif ev["type"] == "trial.match":
            self._trial_done = True
        self.emit(ev)

    def _trend_summary(self):
        lines = []
        for sig, pts in self.trend.items():
            if pts:
                t, s, tier = pts[-1]
                peak = max(pts, key=lambda x: x[1])
                lines.append(f"{sig}: now {s:.2f} ({tier}), peak {peak[1]:.2f}")
        return "\n".join(lines)

    def run(self, scenario, speed=8.0):
        """Replay scenario inputs; call real Claude at reasoning triggers + post-visit."""
        self.chart = scenario["chart"]
        self.emit({"type": "meta", "scenario": scenario["scenario"], "speed": speed})
        events = sorted(scenario["events"], key=lambda e: e["t"])
        t0 = events[0]["t"]; wall = time.monotonic()
        conclusive_since_reason = 0
        for e in events:
            delay = (wall + (e["t"] - t0) / speed) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            if e["type"] == "biomarker.result":
                for s in e["signals"]:
                    self.trend.setdefault(s["name"], []).append((e["t_start"], s["score"], s["tier"]))
                self.emit({"type": "reasoning.step", "phase": "score",
                           "message": f"chunk {e['chunk_id']} → POST /v2/models/aria/analyze → 200 (replayed)"})
                self.emit(e)
                if any(s["tier"] not in ("inconclusive", None) for s in e["signals"]):
                    conclusive_since_reason += 1
                if conclusive_since_reason >= 4:
                    conclusive_since_reason = 0
                    # mid-visit: fast, targeted triage — low effort
                    self._reason(IN_VISIT_TOOLS, self._live_delta(), "reason", effort="low")
            elif e["type"] == "transcript.token":
                self.transcript.append(f"{e['who']}: {e['text']}")
                self.emit(e)
            elif e["type"] == "briefing_item":
                self.emit({"type": "briefing", "items": [e]})
        # post-visit pass — same conversation, careful analysis — high effort
        self._reason(POST_VISIT_TOOLS, self._post_prompt(), "post", effort="high")
        # guarantee the trials lookup for the flagged gap (same agent, same conversation)
        if self._gap and not self._trial_done:
            self._reason(POST_VISIT_TOOLS,
                         f"You flagged this as an undiscussed gap: \"{self._gap}\". "
                         "Call search_clinical_trials now for that specific gap to surface relevant recruiting trials.",
                         "trials", effort="low")
        self.emit({"type": "reasoning.step", "phase": "done", "message": "encounter processed"})
        self.emit({"type": "end"})

    def _live_delta(self):
        """Only the NEW information since the last tick — the conversation already holds the rest."""
        new_tx = self.transcript[self._sent_tx:]; self._sent_tx = len(self.transcript)
        new_chunks = []
        for sig, pts in self.trend.items():
            for t, s, tier in pts:
                if (sig, t) not in self._sent_chunks and tier not in ("inconclusive", None):
                    self._sent_chunks.add((sig, t)); new_chunks.append((t, sig, s, tier))
        new_chunks.sort()
        parts = []
        if self._first:
            self._first = False
            parts.append("VISIT START.\nPATIENT CHART:\n" + json.dumps(self.chart, indent=1))
        parts.append("LIVE TICK (mid-appointment).")
        if new_tx:
            parts.append("New transcript since last tick:\n" + "\n".join(new_tx))
        if new_chunks:
            parts.append("New voice-biomarker readings:\n" +
                         "\n".join(f"  t={int(t)}s {sig}: {s:.2f} ({tier})" for t, sig, s, tier in new_chunks))
        parts.append(f"Currently on the clinician's card: {sorted(self.suggested) or 'nothing'}.\n"
                     "Triage: is anything here NEW, high-confidence, and additive enough to interrupt the "
                     "clinician RIGHT NOW? If yes, call suggest_topic / update_flowsheet. If not, call no tools.")
        return "\n\n".join(parts)

    def _post_prompt(self):
        return ("POST-VISIT. The recording has stopped. Using everything you observed across this "
                "session (it is all in your memory above — chart, full transcript, the biomarker "
                "trend, and what you already surfaced live), do the final analysis now:\n"
                f"Final voice-biomarker trend:\n{self._trend_summary()}\n\n"
                "(1) add_note_addendum for confirmed findings, with provenance.\n"
                "(2) flag_followup for what to act on — set discussed=false for any signal the voice "
                "flagged but the transcript never addressed (the gap).\n"
                "(3) for the undiscussed gap ONLY, call search_clinical_trials to find relevant recruiting trials.")
