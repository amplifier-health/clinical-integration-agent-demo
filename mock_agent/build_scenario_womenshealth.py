#!/usr/bin/env python3
"""Build the women's-health demo scenario for the mock agent.

SYNTHETIC and illustrative — a fictional patient. The clinical arc mirrors the
motivating finding: the visit is focused on ANXIETY while the voice model's
ELEVATED-ANDROGENS signal quietly rises across the encounter and becomes the
post-visit gap (an un-worked-up PCOS signal). Model = aria.
"""
import json, os

CHART = {
    "name": "Elena Ross", "mrn": "48120", "age": 31, "sex": "F",
    "problems": ["Generalized anxiety disorder (F41.1)", "Recurrent depressive episodes (F33)",
                 "Fatigue, unspecified"],
    "meds": ["Sertraline 100 mg daily"],
    "chief_complaint": "Anxiety follow-up; ongoing fatigue and feeling 'off'.",
    "prior_gap": "Mentioned irregular periods once in passing — never worked up.",
}

E = []
def ev(t, type, **kw): E.append({"t": t, "type": type, **kw})

ev(-2, "briefing", items=[
    {"text": "Review anxiety symptom control & sertraline response", "source": "chart"},
    {"text": "Follow up on persistent fatigue", "source": "chart"},
    {"text": "Watch: irregular periods mentioned last visit, not worked up", "source": "voice-history"}])

for t, who, txt in [
    (2,"DR","How have things been since we increased the sertraline?"),
    (9,"PT","The panic is a bit better, but I'm exhausted all the time and just not myself."),
    (24,"DR","Sleep okay? Any change in appetite or mood?"),
    (30,"PT","Sleep's rough. I feel low a lot, and honestly kind of foggy."),
    (48,"DR","Let's talk about coping strategies and whether the dose is right."),
    (62,"PT","Work stress is a huge trigger. I've had a couple of bad weeks."),
    (82,"PT","Oh — and my periods have been all over the place, but that's probably the stress."),
    (100,"DR","We'll keep the dose and add a therapy referral. Let's recheck in a month."),
]:
    ev(t, "transcript.token", who=who, text=txt)

# aria biomarker chunks — elevated-androgens rises, anxiety+mood elevated, fatigue up
chunks = [
    (30, {"anxiety":(0.19,"consider"),"mood-disruption":(0.17,"consider"),"elevated-androgens":(0.08,"low"),"fatigue":(0.16,"consider"),"iron-deficiency":(0.10,"low")}),
    (45, {"anxiety":(0.28,"moderate"),"mood-disruption":(0.22,"consider"),"elevated-androgens":(0.18,"consider"),"fatigue":(0.24,"moderate"),"iron-deficiency":(0.13,"consider")}),
    (60, {"anxiety":(0.33,"moderate"),"mood-disruption":(0.27,"moderate"),"elevated-androgens":(0.29,"moderate"),"fatigue":(0.30,"moderate"),"iron-deficiency":(0.15,"consider")}),
    (75, {"anxiety":(0.31,"moderate"),"mood-disruption":(0.29,"moderate"),"elevated-androgens":(0.38,"moderate"),"fatigue":(0.33,"moderate"),"iron-deficiency":(0.18,"consider")}),
    (90, {"anxiety":(0.35,"moderate"),"mood-disruption":(0.30,"moderate"),"elevated-androgens":(0.46,"elevated"),"fatigue":(0.35,"moderate"),"iron-deficiency":(0.20,"consider")}),
    (105,{"anxiety":(0.30,"moderate"),"mood-disruption":(0.28,"moderate"),"elevated-androgens":(0.43,"moderate"),"fatigue":(0.32,"moderate"),"iron-deficiency":(0.19,"consider")}),
]
for i,(t,sig) in enumerate(chunks):
    ev(t-0.5, "reasoning.step", phase="score",
       message=f"chunk {int(t-30)}–{int(t)}s → POST /v2/models/aria/analyze → 200 (replayed)")
    ev(t, "biomarker.result", chunk_id=i, t_start=int(t-30),
       signals=[{"name":n,"score":s,"tier":tier} for n,(s,tier) in sig.items()])

# live reasoning → suggested topics / flowsheet
ev(46, "reasoning.step", phase="reason", message="anxiety 'moderate' + fatigue ↑; visit is on-topic → surface symptom-control check (aligned)")
ev(47, "topic.suggested", id="anx", title="Anxiety signal elevated — confirm symptom control",
   rationale="Voice anxiety + mood-disruption at 'moderate' this visit; assess sertraline response.",
   source="voice+chart", confidence=0.70, trend=[0.19,0.28,0.33])
ev(61, "reasoning.step", phase="reason", message="elevated-androgens crossing 'moderate' — NOT the visit focus; cross-check chart → no PCOS/menstrual workup on file")
ev(62, "topic.suggested", id="androgen", title="Voice signal: elevated androgen-associated features",
   rationale="Rising androgen-associated vocal profile (low→moderate→elevated). Consider asking about menstrual regularity, hair/skin changes.",
   source="voice", confidence=0.68, trend=[0.08,0.18,0.29,0.38])
ev(84, "reasoning.step", phase="reason", message="patient volunteers irregular periods but attributes to stress; androgen signal corroborates → strengthen for post-visit")
ev(86, "flowsheet.update", section="Endocrine / Voice", row="Androgen-associated vocal profile",
   value="Elevated", provenance={"quote":"periods have been all over the place","t":82})

# post-visit
ev(116, "reasoning.step", phase="post", message="Aggregating full-visit trajectory across 6 chunks…")
ev(118, "note.addendum", section="Assessment",
   text="Voice analysis: anxiety and mood-disruption 'moderate', consistent with presentation. Incidental — elevated androgen-associated vocal profile peaked 'elevated'; patient reports menstrual irregularity.",
   citations=[{"quote":"periods have been all over the place","t":82}])
ev(120, "followup.item", text="Continue sertraline; therapy referral placed; recheck anxiety in 1 month", why="Visit focus, addressed", discussed=True)
ev(122, "reasoning.step", phase="gap", message="GAP: elevated-androgens reached 'elevated' + reported oligomenorrhea, but no PCOS workup was ordered")
ev(123, "followup.item", text="PCOS workup next visit: menstrual history, free testosterone / SHBG, consider pelvic ultrasound", why="Voice androgen signal 'elevated' + irregular periods; not worked up today (or previously)", discussed=False)
ev(126, "reasoning.step", phase="trials", message="Querying Clinical Trials MCP: PCOS / hyperandrogenism with comorbid anxiety-depression (the undiscussed gap)…")
ev(129, "trial.match", nct_id="NCTXXXXXXX1", title="Metabolic and mental-health outcomes in PCOS with comorbid anxiety/depression",
   why_relevant="Targets the exact overlap surfaced but not addressed: androgen excess + mood symptoms.", eligibility_hint="Reproductive-age women, PCOS features, mood symptoms")
ev(131, "trial.match", nct_id="NCTXXXXXXX2", title="Anti-androgen vs combined oral contraceptive for hyperandrogenism",
   why_relevant="If PCOS workup confirms hyperandrogenism, relevant treatment-comparison trial.", eligibility_hint="Confirmed hyperandrogenism, age 18–40")
ev(133, "reasoning.step", phase="done", message="Encounter processed. 2 topics surfaced live · 1 gap (PCOS) · 2 trials.")

E.sort(key=lambda e: e["t"])
out = {"scenario":"womens_health_synthetic", "chart": CHART, "encounter_seconds": 115,
       "events": E, "note": "SYNTHETIC — fictional patient, illustrative only."}
here = os.path.dirname(os.path.abspath(__file__))
json.dump(out, open(os.path.join(here,"scenario_womenshealth.json"),"w"), indent=1)
print(f"wrote scenario_womenshealth.json — {len(E)} events")
