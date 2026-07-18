#!/usr/bin/env python3
"""Build the women's-health demo scenario for the offline mock server.

Emits the SAME event contract as the real backend (clinical_agent) so the viewer
speaks one language everywhere: pre_visit_brief / transcript / api_job_created /
api_job_result / agent_token / observation / visit_summary / chart_draft / topics /
trial_match / longitudinal_delta / visit_complete.

SYNTHETIC, fictional patient. Model = aria (elevated-androgens, anxiety, mood-disruption,
fatigue, iron-deficiency). The clinical arc: an anxiety-focused visit while the voice's
elevated-androgens signal quietly rises and becomes the post-visit PCOS gap.
"""
import json, os

CHART = {
    "name": "Elena Ross", "mrn": "48120", "age": 31, "sex": "F",
    "problems": ["Generalized anxiety disorder (F41.1)", "Recurrent depressive episodes (F33)", "Fatigue"],
    "meds": ["Sertraline 100 mg daily"],
    "chief_complaint": "Anxiety follow-up; ongoing fatigue and feeling 'off'.",
}

E = []
def ev(t, type, **kw): E.append({"t": t, "type": type, **kw})

ev(-2, "pre_visit_brief", brief="Anxiety follow-up on sertraline; watch fatigue and a prior unworked-up menstrual complaint.",
   topics_to_discuss=["Review anxiety symptom control & sertraline response", "Follow up on persistent fatigue"],
   vocal_trends=["Irregular periods mentioned last visit — never worked up"])

for t, who, txt in [
    (2, "DR", "How have things been since we increased the sertraline?"),
    (9, "PT", "The panic is a bit better, but I'm exhausted all the time and just not myself."),
    (30, "PT", "Sleep's rough. I feel low a lot, and honestly kind of foggy."),
    (62, "PT", "Work stress is a huge trigger. I've had a couple of bad weeks."),
    (82, "PT", "Oh — and my periods have been all over the place, but that's probably the stress."),
    (100, "DR", "We'll keep the dose and add a therapy referral. Let's recheck in a month."),
]:
    ev(t, "transcript", chunk=int(t // 15), text=f"{who}: {txt}")

chunks = [
    (30, {"anxiety": (0.19, "consider"), "mood-disruption": (0.17, "consider"), "elevated-androgens": (0.08, "low"), "fatigue": (0.16, "consider"), "iron-deficiency": (0.10, "low")}),
    (45, {"anxiety": (0.28, "moderate"), "mood-disruption": (0.22, "consider"), "elevated-androgens": (0.18, "consider"), "fatigue": (0.24, "moderate"), "iron-deficiency": (0.13, "consider")}),
    (60, {"anxiety": (0.33, "moderate"), "mood-disruption": (0.27, "moderate"), "elevated-androgens": (0.29, "moderate"), "fatigue": (0.30, "moderate"), "iron-deficiency": (0.15, "consider")}),
    (75, {"anxiety": (0.31, "moderate"), "mood-disruption": (0.29, "moderate"), "elevated-androgens": (0.38, "moderate"), "fatigue": (0.33, "moderate"), "iron-deficiency": (0.18, "consider")}),
    (90, {"anxiety": (0.35, "moderate"), "mood-disruption": (0.30, "moderate"), "elevated-androgens": (0.46, "elevated"), "fatigue": (0.35, "moderate"), "iron-deficiency": (0.20, "consider")}),
    (105, {"anxiety": (0.30, "moderate"), "mood-disruption": (0.28, "moderate"), "elevated-androgens": (0.43, "moderate"), "fatigue": (0.32, "moderate"), "iron-deficiency": (0.19, "consider")}),
]
for i, (t, sig) in enumerate(chunks):
    ev(t - 1, "api_job_created", chunk=i, model="aria", job_id=f"job-{i}")
    ev(t, "api_job_result", chunk=i, model="aria",
       signals=[{"name": n, "score": s, "level": lv, "flagged": lv in ("consider", "moderate", "elevated")}
                for n, (s, lv) in sig.items()],
       summary={"overall_level": max(sig.values(), key=lambda x: x[0])[1]})

# live reasoner output (streamed tokens + observations)
ev(46, "agent_token", agent="reasoner", text="Anxiety and mood at moderate — consistent with the visit's focus; already known. ")
ev(47, "observation", chunk=1, text="Anxiety and mood-disruption at moderate, consistent with the anxiety follow-up.")
ev(62, "agent_token", agent="reasoner", text="Elevated-androgens crossing moderate and NOT the visit focus; chart shows no PCOS workup. ")
ev(63, "observation", chunk=2, text="Elevated androgen-associated vocal features are rising and are not the focus of this visit — consider asking about menstrual regularity and hair/skin changes.")
ev(84, "observation", chunk=4, text="Patient volunteered irregular periods; the androgen signal corroborates. Worth an endocrine screen.")

# post-visit
ev(116, "agent_token", agent="postvisit", text="Aggregating the visit: anxiety on-topic; elevated-androgens the incidental, undiscussed signal. ")
ev(118, "visit_summary", patient="demo", visit=1,
   summary="Anxiety follow-up: since the sertraline increase, panic is somewhat improved but the patient reports persistent fatigue and low mood. Continue current management with a therapy referral.",
   vocal_findings=[{"sign": "elevated-androgens", "level": "elevated", "note": "Androgen-associated vocal profile peaked 'elevated' this visit; incidental to the anxiety focus."},
                   {"sign": "anxiety", "level": "moderate", "note": "Consistent with the presenting complaint."}],
   discordance="The visit centered on anxiety, but the strongest rising voice signal (androgen-associated) went unaddressed.",
   screener_recommendations=["GAD-7", "PHQ-9"])
ev(120, "chart_draft", patient="demo", visit=1, items=[
   {"description": "Add problem: fatigue, evaluate", "rationale": "Persistent, patient-reported; supported by voice fatigue signal."}])
ev(122, "agent_tool_call", agent="postvisit", tool="search_clinical_trials", input={"condition": "PCOS anxiety"})
ev(123, "topics", patient="demo", visit=1, items=[
   "PCOS workup: menstrual history, free testosterone / SHBG, consider pelvic ultrasound",
   "Reassess fatigue; check CBC / ferritin"])
ev(129, "trial_match", patient="demo", nct_id="NCT05123456",
   title="Metabolic and mental-health outcomes in PCOS with comorbid anxiety",
   why_relevant="Targets the exact overlap surfaced but not addressed: androgen excess + mood symptoms.",
   eligibility_hint="Reproductive-age women, PCOS features, mood symptoms")
ev(131, "trial_match", patient="demo", nct_id="NCT04987654",
   title="Anti-androgen vs combined oral contraceptive for hyperandrogenism",
   why_relevant="If PCOS workup confirms hyperandrogenism, a relevant treatment-comparison trial.",
   eligibility_hint="Confirmed hyperandrogenism, age 18–40")
ev(133, "longitudinal_delta", patient="demo", condition="PCOS / hyperandrogenism",
   first_voice_flag_visit=1, first_coded_visit=4, visits_early=3)
ev(135, "visit_complete", patient="demo", visit=1)

E.sort(key=lambda e: e["t"])
out = {"scenario": "womens_health_synthetic", "chart": CHART, "encounter_seconds": 135,
       "events": E, "note": "SYNTHETIC — fictional patient; backend event contract."}
here = os.path.dirname(os.path.abspath(__file__))
json.dump(out, open(os.path.join(here, "scenario_womenshealth.json"), "w"), indent=1)
print(f"wrote scenario_womenshealth.json — {len(E)} backend-contract events")
