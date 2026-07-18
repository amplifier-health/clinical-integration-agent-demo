#!/usr/bin/env python3
"""Build a synthetic demo scenario (chart + timed event stream) for the mock agent.

Everything here is SYNTHETIC and illustrative — no real patient data. The event
stream is what the mock agent replays; the UI subscribes to it. Times are in
seconds into the encounter (the server compresses them by a speed multiplier).
"""
import json, os

CHART = {
    "name": "James Whitaker", "mrn": "55803", "age": 64, "sex": "M",
    "problems": ["COPD (GOLD group E, 2 exacerbations/12mo)", "Type 2 diabetes (A1c 9.2%)",
                 "Hypertension", "Hyperlipidemia", "Active tobacco use (~45 pack-years)"],
    "meds": ["Tiotropium-olodaterol (maintenance)", "Albuterol PRN"],
    "chief_complaint": "Shortness of breath and worsening cough.",
    "prior_gap": "Low mood flagged by voice last visit — never discussed.",
}

# each entry: (t_seconds, event_dict)
E = []
def ev(t, type, **kw): E.append({"t": t, "type": type, **kw})

# --- pre-visit briefing (t<0 shown before record) ---
ev(-2, "briefing", items=[
    {"text": "Reassess inhaler technique & adherence", "source": "chart"},
    {"text": "Smoking cessation check-in (~0.5 ppd)", "source": "chart"},
    {"text": "Watch: low mood flagged by voice last visit, not addressed", "source": "voice-history"}])

# --- rolling transcript (compressed snippets) ---
for t, who, txt in [
    (2,"DR","How's the breathing been since we adjusted the inhaler?"),
    (8,"PT","Still short of breath walking up stairs, coughing a lot at night."),
    (20,"DR","Any yellow or green phlegm?"),(26,"PT","Yeah, greenish the last few days."),
    (40,"DR","Let me listen to your lungs. Big breaths."),
    (58,"PT","I've been using the rescue inhaler almost every couple hours."),
    (74,"DR","Okay. And the smoking — still about half a pack?"),(80,"PT","Yeah, about that."),
    (96,"PT","Honestly I've just been really run down, no energy for anything."),
    (110,"DR","Let's get a chest x-ray and step up treatment for this flare."),
]:
    ev(t, "transcript.token", who=who, text=txt)

# --- biomarker chunks (breath model) replayed every 15s; mocked API call each ---
# signals: airway-obstruction-pattern (rising), fatigue, mood-disruption (the quiet gap)
chunks = [
    (30, {"airway-obstruction-pattern":(0.09,"low"), "allergy":(0.04,"none"), "fatigue":(0.12,"low"), "mood-disruption":(0.10,"low")}),
    (45, {"airway-obstruction-pattern":(0.22,"consider"), "allergy":(0.06,"low"), "fatigue":(0.19,"consider"), "mood-disruption":(0.17,"consider")}),
    (60, {"airway-obstruction-pattern":(0.34,"moderate"), "allergy":(0.05,"none"), "fatigue":(0.26,"moderate"), "mood-disruption":(0.24,"consider")}),
    (75, {"airway-obstruction-pattern":(0.41,"moderate"), "allergy":(0.07,"low"), "fatigue":(0.31,"moderate"), "mood-disruption":(0.29,"moderate")}),
    (90, {"airway-obstruction-pattern":(0.47,"elevated"), "allergy":(0.06,"low"), "fatigue":(0.35,"moderate"), "mood-disruption":(0.33,"moderate")}),
    (105,{"airway-obstruction-pattern":(0.44,"moderate"), "allergy":(0.05,"none"), "fatigue":(0.34,"moderate"), "mood-disruption":(0.36,"moderate")}),
]
for i,(t,sig) in enumerate(chunks):
    ev(t-0.5, "reasoning.step", phase="score",
       message=f"chunk {int(t-30)}–{int(t)}s → POST /v2/models/breath/analyze → 200 (replayed)")
    ev(t, "biomarker.result", chunk_id=i, t_start=int(t-30),
       signals=[{"name":n,"score":s,"tier":tier} for n,(s,tier) in sig.items()])

# --- live reasoning cycles → suggested topics / flowsheet (during visit) ---
ev(46, "reasoning.step", phase="reason", message="airway-obstruction ↑ to 'consider'; transcript confirms dyspnea+cough → surface auscultation prompt")
ev(47, "topic.suggested", id="airway", title="Airway obstruction pattern rising",
   rationale="Voice signal trending up (consider→moderate) alongside reported dyspnea and productive cough.",
   source="voice", confidence=0.71, trend=[0.09,0.22,0.34])
ev(62, "reasoning.step", phase="reason", message="spoken exam finding 'crackles' detected → map to flowsheet row")
ev(63, "flowsheet.update", section="Respiratory", row="Bilateral Breath Sounds",
   value="Crackles", provenance={"quote":"still have crackles on both sides","t":61})
ev(78, "reasoning.step", phase="reason", message="rescue-inhaler overuse in transcript + obstruction signal → adherence topic")
ev(79, "topic.suggested", id="inhaler", title="Rescue inhaler overuse — technique & adherence",
   rationale="Patient reports albuterol q2h; maintenance adherence worth confirming.",
   source="voice+chart", confidence=0.66, trend=[])
ev(100, "reasoning.step", phase="reason", message="mood-disruption ↑ to 'moderate' but NOT raised in transcript → hold for post-visit gap list")

# --- post-visit pass (fires after 'stop', t>=115) ---
ev(116, "reasoning.step", phase="post", message="Aggregating full-visit trajectory across 6 chunks…")
ev(118, "note.addendum", section="Respiratory",
   text="Voice analysis: airway-obstruction pattern peaked 'elevated' this encounter; bilateral crackles noted. Consistent with COPD exacerbation.",
   citations=[{"quote":"crackles on both sides","t":61}])
ev(120, "followup.item", text="Confirm inhaler technique and maintenance adherence", why="Rescue overuse + obstruction signal", discussed=True)
ev(122, "reasoning.step", phase="gap", message="GAP: mood-disruption trended to 'moderate' and was never discussed this visit")
ev(123, "followup.item", text="Screen for depression / low mood next visit", why="Voice mood-disruption reached 'moderate'; not addressed today; flagged prior visit too", discussed=False)
ev(126, "reasoning.step", phase="trials", message="Querying Clinical Trials MCP: COPD + persistent low mood (the undiscussed gap)…")
ev(129, "trial.match", nct_id="NCTXXXXXXX1", title="Pulmonary rehabilitation + behavioral activation for COPD with depressive symptoms",
   why_relevant="Targets the exact overlap the visit missed: COPD exacerbation + emerging low mood.", eligibility_hint="COPD, active symptoms, age 40–75")
ev(131, "trial.match", nct_id="NCTXXXXXXX2", title="Triple therapy vs LAMA-LABA in frequent COPD exacerbators",
   why_relevant="GOLD group E with ≥2 exacerbations/yr — matches escalation discussed.", eligibility_hint="≥2 exacerbations/12mo")
ev(133, "reasoning.step", phase="done", message="Encounter processed. 2 topics surfaced live · 1 gap · 2 trials.")

E.sort(key=lambda e: e["t"])
out = {"scenario":"respiratory_copd_synthetic", "chart": CHART,
       "encounter_seconds": 115, "events": E,
       "note": "SYNTHETIC — illustrative only, not real patient data."}
here = os.path.dirname(os.path.abspath(__file__))
json.dump(out, open(os.path.join(here,"scenario_respiratory.json"),"w"), indent=1)
print(f"wrote scenario_respiratory.json — {len(E)} events, {out['encounter_seconds']}s encounter")
