import json
from pathlib import Path

from pydantic import BaseModel, Field


class Icd10Code(BaseModel):
    code: str
    description: str


class VisitMeta(BaseModel):
    number: int
    date: str
    reason: str
    icd10: list[Icd10Code] = Field(default_factory=list)
    has_audio: bool = False
    status: str = "planned"


class PatientMeta(BaseModel):
    id: str
    alias: str
    age: int
    sex: str


class PatientStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _pdir(self, pid: str) -> Path:
        return self.root / "patients" / pid

    def save_patient(self, meta: PatientMeta) -> None:
        d = self._pdir(meta.id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "patient.json").write_text(meta.model_dump_json(indent=2))

    def get_patient(self, pid: str) -> PatientMeta:
        return PatientMeta.model_validate_json((self._pdir(pid) / "patient.json").read_text())

    def list_patients(self) -> list[PatientMeta]:
        base = self.root / "patients"
        if not base.exists():
            return []
        return [self.get_patient(p.name) for p in sorted(base.iterdir()) if (p / "patient.json").exists()]

    def save_visits(self, pid: str, visits: list[VisitMeta]) -> None:
        payload = [v.model_dump() for v in visits]
        (self._pdir(pid) / "visits.json").write_text(json.dumps(payload, indent=2))

    def list_visits(self, pid: str) -> list[VisitMeta]:
        path = self._pdir(pid) / "visits.json"
        if not path.exists():
            return []
        return [VisitMeta.model_validate(v) for v in json.loads(path.read_text())]

    def write_artifact(self, pid: str, visit: int, name: str, payload) -> None:
        d = self._pdir(pid) / "visits" / str(visit)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(json.dumps(payload, indent=2))

    def read_artifact(self, pid: str, visit: int, name: str):
        path = self._pdir(pid) / "visits" / str(visit) / f"{name}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def chart(self, pid: str) -> dict:
        visits = []
        for v in self.list_visits(pid):
            visits.append({
                **v.model_dump(),
                "signals": self.read_artifact(pid, v.number, "signals"),
                "summary": self.read_artifact(pid, v.number, "summary"),
            })
        return {"patient": self.get_patient(pid).model_dump(), "visits": visits}
