"""Deterministic synthetic patient bundle for development and tests.

Story: female patient, somatic complaints early; voice flags mood-disruption
and related signals visits before the chart codes PCOS (visit 6) and
depression/anxiety (visit 9).
"""
from clinical_agent.store import Icd10Code, PatientMeta, PatientStore, VisitMeta

SOMATIC = [
    ("H66.90", "Otitis media, unspecified"),
    ("R53.83", "Other fatigue"),
    ("R51.9", "Headache, unspecified"),
    ("L70.0", "Acne vulgaris"),
    ("N92.6", "Irregular menstruation"),
]
SIGNS = ["mood-disruption", "anxiety", "stress", "fatigue", "hypervigilance", "attention-dysregulation"]


def _signals(visit_no: int) -> list[dict]:
    out = []
    for i, name in enumerate(SIGNS):
        base = 0.15 + 0.02 * i
        if name in ("mood-disruption", "anxiety", "fatigue"):
            score = min(0.9, base + 0.08 * visit_no)  # trends up over visits
        else:
            score = base
        level = "high" if score >= 0.6 else "moderate" if score >= 0.35 else "low"
        out.append({
            "name": name,
            "label": name.replace("-", " ").title(),
            "score": round(score, 2),
            "level": level,
            "flagged": score >= 0.35,
        })
    return out


def generate_synthetic_patient(store: PatientStore, pid: str = "demo-synthetic", n_visits: int = 10) -> None:
    store.save_patient(PatientMeta(id=pid, alias="Jane D. (synthetic)", age=31, sex="F"))
    visits = []
    for n in range(1, n_visits + 1):
        codes = [Icd10Code(code=c, description=d) for c, d in [SOMATIC[(n - 1) % len(SOMATIC)]]]
        if n == 6:
            codes.append(Icd10Code(code="E28.2", description="Polycystic ovarian syndrome"))
        if n == 9:
            codes += [
                Icd10Code(code="F32.9", description="Major depressive disorder, single episode"),
                Icd10Code(code="F41.9", description="Anxiety disorder, unspecified"),
            ]
        is_today = n == n_visits
        visits.append(VisitMeta(
            number=n,
            date=f"2025-{(n % 12) + 1:02d}-10",
            reason="Wellness visit" if n % 2 else "Follow-up",
            icd10=[] if is_today else codes,
            has_audio=is_today or n >= 2,
            status="planned" if is_today else "complete",
        ))
        if not is_today and n >= 2:
            store.write_artifact(pid, n, "signals", _signals(n))
            store.write_artifact(pid, n, "transcript", [
                {"chunk": 1, "text": "Doctor: How have you been? Patient: I'm okay, just tired lately."},
            ])
            store.write_artifact(pid, n, "summary", {
                "summary": f"Visit {n}: patient reports feeling okay; voice signals show "
                           f"mood-disruption {_signals(n)[0]['level']}.",
                "next_visit_topics": ["Sleep quality", "Energy levels"],
            })
    store.save_visits(pid, visits)
