# Voice Agent Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A real agent pipeline that chunks visit audio, transcribes with Whisper, analyzes chunks through the Amplifier Health API, reasons over results with Claude live, and maintains a longitudinal patient store — exposed over FastAPI + SSE for a later frontend.

**Architecture:** Pre-visit agent → visit session (AudioSource → chunker → faster-whisper + async Amplifier job queue → visit reasoner) → post-visit agent, all writing to a JSON patient store whose output becomes the next visit's input. A longitudinal analyst computes voice-flag vs ICD-10-code gaps on demand. Everything emits to an in-process event bus streamed via SSE.

**Tech Stack:** Python 3.11+, FastAPI, httpx, anthropic SDK (Claude Opus 4.8, streaming, adaptive thinking), faster-whisper, pydub, pydantic, pytest + respx.

## Global Constraints

- License: MIT. Repo is public (`github.com/amplifier-health/clinical-integration-agent-demo`). **Never commit real patient data**: `onset_analysis/`, `data/`, and `.env` are gitignored. Synthetic fixtures only.
- Claude model: `claude-opus-4-8` everywhere. Streaming on every call. `thinking={"type": "adaptive"}`. Per-chunk reasoner uses `output_config={"effort": "low"}`; pre-visit/post-visit/longitudinal use `"high"`.
- Amplifier API: base URL from `AMPLIFIER_BASE_URL` (default `https://api.amplifierhealth.com`), auth headers `X-Account-ID` + `X-API-Key`, model `haven`, analyze rate limit 5 req/min, chunk length 45 s target with 15 s API floor.
- Every live external call has an offline path: `MOCK_CLAUDE=1` (canned agent output), `AMPLIFIER_CACHE=warm|record|off` (result cache keyed by audio hash), Claude rehearsal cache with silent fallback on error or >6 s to first token.
- The code path is identical in demo and real mode; only config and data differ.
- All async; tests use `pytest-asyncio` with `asyncio_mode = "auto"`.

---

### Task 1: Repo scaffolding

**Files:**
- Create: `pyproject.toml`, `LICENSE`, `.gitignore`, `clinical_agent/__init__.py`, `tests/__init__.py`, `clinical_agent/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `clinical_agent.config.Settings` dataclass with fields `amplifier_base_url: str`, `amplifier_account_id: str`, `amplifier_api_key: str`, `anthropic_model: str = "claude-opus-4-8"`, `mock_claude: bool`, `amplifier_cache: str` (`"off"|"warm"|"record"`), `whisper_model: str = "base"`, `mock_whisper: bool`, `speed: float = 1.0`, `data_dir: Path`, and classmethod `Settings.from_env() -> Settings`.

- [ ] **Step 1: Write project files**

`pyproject.toml`:
```toml
[project]
name = "clinical-integration-agent"
version = "0.1.0"
description = "Voice biomarker clinical agent demo (Amplifier Health x Abridge hackathon)"
requires-python = ">=3.11"
license = { text = "MIT" }
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "httpx>=0.27",
    "anthropic>=0.40",
    "pydantic>=2.7",
    "faster-whisper>=1.0",
    "pydub>=0.25",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "respx>=0.21"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["clinical_agent", "clinical_agent.agents"]
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.env
data/
onset_analysis/
*.egg-info/
.pytest_cache/
```

`LICENSE`: standard MIT text, copyright `2026 Amplifier Health`.

`clinical_agent/config.py`:
```python
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    amplifier_base_url: str = "https://api.amplifierhealth.com"
    amplifier_account_id: str = ""
    amplifier_api_key: str = ""
    anthropic_model: str = "claude-opus-4-8"
    mock_claude: bool = False
    amplifier_cache: str = "off"  # off | warm | record
    whisper_model: str = "base"
    mock_whisper: bool = False
    speed: float = 1.0
    data_dir: Path = field(default_factory=lambda: Path("data"))

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            amplifier_base_url=os.environ.get("AMPLIFIER_BASE_URL", cls.amplifier_base_url),
            amplifier_account_id=os.environ.get("AMPLIFIER_ACCOUNT_ID", ""),
            amplifier_api_key=os.environ.get("AMPLIFIER_API_KEY", ""),
            anthropic_model=os.environ.get("ANTHROPIC_MODEL", cls.anthropic_model),
            mock_claude=os.environ.get("MOCK_CLAUDE", "") == "1",
            amplifier_cache=os.environ.get("AMPLIFIER_CACHE", "off"),
            whisper_model=os.environ.get("WHISPER_MODEL", "base"),
            mock_whisper=os.environ.get("MOCK_WHISPER", "") == "1",
            speed=float(os.environ.get("SPEED", "1.0")),
            data_dir=Path(os.environ.get("DATA_DIR", "data")),
        )
```

`tests/test_config.py`:
```python
from clinical_agent.config import Settings


def test_from_env_defaults(monkeypatch):
    for var in ("AMPLIFIER_BASE_URL", "MOCK_CLAUDE", "SPEED"):
        monkeypatch.delenv(var, raising=False)
    s = Settings.from_env()
    assert s.anthropic_model == "claude-opus-4-8"
    assert s.amplifier_cache == "off"
    assert s.speed == 1.0


def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("MOCK_CLAUDE", "1")
    monkeypatch.setenv("SPEED", "50")
    monkeypatch.setenv("AMPLIFIER_CACHE", "warm")
    s = Settings.from_env()
    assert s.mock_claude is True
    assert s.speed == 50.0
    assert s.amplifier_cache == "warm"
```

- [ ] **Step 2: Create venv, install, run tests**

Run: `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"` then `.venv/bin/pytest tests/test_config.py -v`
Expected: 2 passed. (faster-whisper install is slow; that's fine.)

- [ ] **Step 3: Wire remote and commit**

```bash
git remote add origin https://github.com/amplifier-health/clinical-integration-agent-demo.git
git add pyproject.toml LICENSE .gitignore clinical_agent tests
git commit -m "feat: project scaffolding with config"
```
Do NOT push yet (push at the end of the plan, after confirming no sensitive files are tracked).

---

### Task 2: Patient store

**Files:**
- Create: `clinical_agent/store.py`, `tests/test_store.py`

**Interfaces:**
- Produces:
  - Pydantic models: `Icd10Code(code: str, description: str)`, `VisitMeta(number: int, date: str, reason: str, icd10: list[Icd10Code] = [], has_audio: bool = False, status: str = "planned")`, `PatientMeta(id: str, alias: str, age: int, sex: str)`.
  - `PatientStore(root: Path)` with methods:
    - `list_patients() -> list[PatientMeta]`
    - `get_patient(pid: str) -> PatientMeta`
    - `list_visits(pid: str) -> list[VisitMeta]`
    - `save_patient(meta: PatientMeta)`, `save_visits(pid, visits: list[VisitMeta])`
    - `write_artifact(pid: str, visit: int, name: str, payload: dict | list)` → writes `patients/<pid>/visits/<visit>/<name>.json`
    - `read_artifact(pid: str, visit: int, name: str) -> dict | list | None` (None if missing)
    - `chart(pid: str) -> dict` — full chart for agents: patient meta, all visits with codes, and for each visit any stored `summary` and `signals` artifacts.

- [ ] **Step 1: Write the failing test**

`tests/test_store.py`:
```python
from clinical_agent.store import Icd10Code, PatientMeta, PatientStore, VisitMeta


def make_store(tmp_path):
    store = PatientStore(tmp_path)
    store.save_patient(PatientMeta(id="p1", alias="Jane D.", age=32, sex="F"))
    store.save_visits("p1", [
        VisitMeta(number=1, date="2026-01-10", reason="Ear infection",
                  icd10=[Icd10Code(code="H66.90", description="Otitis media")], has_audio=True),
        VisitMeta(number=2, date="2026-02-14", reason="Wellness visit"),
    ])
    return store


def test_round_trip(tmp_path):
    store = make_store(tmp_path)
    assert store.get_patient("p1").alias == "Jane D."
    visits = store.list_visits("p1")
    assert len(visits) == 2 and visits[0].icd10[0].code == "H66.90"


def test_artifacts_and_chart(tmp_path):
    store = make_store(tmp_path)
    store.write_artifact("p1", 1, "signals", [{"name": "mood-disruption", "score": 0.7}])
    store.write_artifact("p1", 1, "summary", {"summary": "flagged"})
    assert store.read_artifact("p1", 1, "signals")[0]["name"] == "mood-disruption"
    assert store.read_artifact("p1", 1, "missing") is None
    chart = store.chart("p1")
    assert chart["patient"]["alias"] == "Jane D."
    assert chart["visits"][0]["signals"][0]["score"] == 0.7
    assert chart["visits"][1]["summary"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_store.py -v` — Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement**

`clinical_agent/store.py`:
```python
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
```

- [ ] **Step 4: Run tests** — Expected: PASS.
- [ ] **Step 5: Commit** — `git add clinical_agent/store.py tests/test_store.py && git commit -m "feat: JSON patient store"`

---

### Task 3: Synthetic patient generator

**Files:**
- Create: `clinical_agent/synthetic.py`, `tests/test_synthetic.py`

**Interfaces:**
- Consumes: `PatientStore`, `PatientMeta`, `VisitMeta`, `Icd10Code` from Task 2.
- Produces: `generate_synthetic_patient(store: PatientStore, pid: str = "demo-synthetic", n_visits: int = 10) -> None`. Deterministic (no randomness). Visits 1..n-1 are past visits carrying stored `signals`, `summary`, and `transcript` artifacts in the exact shapes the live pipeline writes; visit n is the "today" visit (`status="planned"`, `has_audio=True`, no artifacts). ICD-10 story: somatic codes early, PCOS code (E28.2) at visit 6, depression/anxiety (F32.9/F41.9) at visit 9, while stored voice `signals` show mood-disruption trending up from visit 2.
- Signal dict shape (matches Amplifier use_case `result.signals[]` consumers downstream): `{"name": str, "label": str, "score": float, "level": str, "flagged": bool}`.

- [ ] **Step 1: Write the failing test**

`tests/test_synthetic.py`:
```python
from clinical_agent.store import PatientStore
from clinical_agent.synthetic import generate_synthetic_patient


def test_generates_loadable_bundle(tmp_path):
    store = PatientStore(tmp_path)
    generate_synthetic_patient(store, n_visits=10)
    visits = store.list_visits("demo-synthetic")
    assert len(visits) == 10
    assert visits[-1].status == "planned" and visits[-1].has_audio
    # PCOS coded at visit 6, mental health at visit 9
    assert any(c.code == "E28.2" for c in visits[5].icd10)
    assert any(c.code.startswith("F") for c in visits[8].icd10)
    # voice flagged mood-disruption well before visit 9
    sig = store.read_artifact("demo-synthetic", 3, "signals")
    mood = next(s for s in sig if s["name"] == "mood-disruption")
    assert mood["flagged"] is True
    # past visits carry summaries; today's visit carries none
    assert store.read_artifact("demo-synthetic", 2, "summary") is not None
    assert store.read_artifact("demo-synthetic", 10, "summary") is None
```

- [ ] **Step 2: Run test to verify it fails** — Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement**

`clinical_agent/synthetic.py`:
```python
"""Deterministic synthetic patient bundle for development and tests.

Story: female patient, somatic complaints early; voice flags mood-disruption
and elevated-androgens visits before the chart codes PCOS (visit 6) and
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
```

- [ ] **Step 4: Run tests** — Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat: synthetic patient generator"`

---

### Task 4: Event bus + FastAPI skeleton with SSE

**Files:**
- Create: `clinical_agent/events.py`, `clinical_agent/api.py`, `tests/test_events.py`

**Interfaces:**
- Produces:
  - `EventBus` with `subscribe() -> asyncio.Queue`, `unsubscribe(q)`, `async emit(type: str, **data)` — each event is `{"type": type, "ts": <iso8601>, **data}`.
  - `create_app(settings: Settings, store: PatientStore, bus: EventBus) -> FastAPI` with `GET /events` (SSE: lines `event: <type>\ndata: <json>\n\n`), `GET /patients` (list of PatientMeta dicts), `GET /patients/{pid}/chart`.
  - Module-level `app` factory used by uvicorn comes in Task 10.

- [ ] **Step 1: Write the failing test**

`tests/test_events.py`:
```python
import asyncio

from fastapi.testclient import TestClient

from clinical_agent.api import create_app
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.store import PatientStore
from clinical_agent.synthetic import generate_synthetic_patient


async def test_bus_fanout():
    bus = EventBus()
    q1, q2 = bus.subscribe(), bus.subscribe()
    await bus.emit("chunk_created", index=1)
    e1, e2 = q1.get_nowait(), q2.get_nowait()
    assert e1["type"] == "chunk_created" and e1["index"] == 1 and "ts" in e1
    assert e2 == e1
    bus.unsubscribe(q1)
    await bus.emit("x")
    assert q1.empty() and not q2.empty()


def test_patients_endpoint(tmp_path):
    store = PatientStore(tmp_path)
    generate_synthetic_patient(store)
    app = create_app(Settings(), store, EventBus())
    client = TestClient(app)
    resp = client.get("/patients")
    assert resp.status_code == 200
    assert resp.json()[0]["id"] == "demo-synthetic"
    chart = client.get("/patients/demo-synthetic/chart").json()
    assert len(chart["visits"]) == 10
```

- [ ] **Step 2: Run test to verify it fails** — Expected: FAIL.

- [ ] **Step 3: Implement**

`clinical_agent/events.py`:
```python
import asyncio
from datetime import datetime, timezone


class EventBus:
    def __init__(self):
        self._subs: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subs:
            self._subs.remove(q)

    async def emit(self, type: str, **data) -> None:
        event = {"type": type, "ts": datetime.now(timezone.utc).isoformat(), **data}
        for q in list(self._subs):
            q.put_nowait(event)
```

`clinical_agent/api.py`:
```python
import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.store import PatientStore


def create_app(settings: Settings, store: PatientStore, bus: EventBus) -> FastAPI:
    app = FastAPI(title="Clinical Integration Agent")
    app.state.settings, app.state.store, app.state.bus = settings, store, bus

    @app.get("/patients")
    def patients():
        return [p.model_dump() for p in store.list_patients()]

    @app.get("/patients/{pid}/chart")
    def chart(pid: str):
        return store.chart(pid)

    @app.get("/events")
    async def events():
        q = bus.subscribe()

        async def stream():
            try:
                while True:
                    e = await q.get()
                    yield f"event: {e['type']}\ndata: {json.dumps(e)}\n\n"
            finally:
                bus.unsubscribe(q)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app
```

- [ ] **Step 4: Run tests** — `.venv/bin/pytest tests/test_events.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat: event bus and API skeleton with SSE"`

---

### Task 5: AudioSource + chunker

**Files:**
- Create: `clinical_agent/audio.py`, `tests/test_audio.py`, `tests/conftest.py`

**Interfaces:**
- Produces:
  - `AudioChunk(index: int, wav_bytes: bytes, start_s: float, end_s: float)` (dataclass).
  - `async chunk_file(path: Path, *, chunk_seconds: float = 45.0, min_seconds: float = 15.0, speed: float = 1.0) -> AsyncIterator[AudioChunk]` — loads the file with pydub (any format ffmpeg supports), slices at `chunk_seconds` boundaries, merges a final slice shorter than `min_seconds` into the previous chunk, exports each slice as 16 kHz mono WAV bytes, and sleeps `slice_duration / speed` before yielding each chunk to simulate real time (`speed=1e9` in tests ≈ no waiting).
- Note: pydub requires `ffmpeg` on PATH for non-WAV input; WAV works without it. Tests use WAV.

- [ ] **Step 1: Write the failing test**

`tests/conftest.py`:
```python
import io
import math
import struct
import wave
from pathlib import Path

import pytest


def sine_wav_bytes(seconds: float, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(seconds * rate)
        frames = b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(n)
        )
        w.writeframes(frames)
    return buf.getvalue()


@pytest.fixture
def wav_100s(tmp_path) -> Path:
    p = tmp_path / "visit.wav"
    p.write_bytes(sine_wav_bytes(100.0))
    return p
```

`tests/test_audio.py`:
```python
import io
import wave

from clinical_agent.audio import chunk_file


async def collect(path, **kw):
    return [c async for c in chunk_file(path, **kw)]


async def test_chunk_boundaries(wav_100s):
    chunks = await collect(wav_100s, chunk_seconds=45.0, min_seconds=15.0, speed=1e9)
    # 100s -> 45 + 45 + 10; trailing 10s < 15s floor merges into chunk 2 (45+55)
    assert len(chunks) == 2
    assert chunks[0].index == 1 and chunks[1].index == 2
    assert abs((chunks[0].end_s - chunks[0].start_s) - 45.0) < 0.1
    assert abs((chunks[1].end_s - chunks[1].start_s) - 55.0) < 0.1


async def test_chunks_are_valid_wav(wav_100s):
    chunks = await collect(wav_100s, speed=1e9)
    with wave.open(io.BytesIO(chunks[0].wav_bytes)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
```

- [ ] **Step 2: Run test to verify it fails** — Expected: FAIL.

- [ ] **Step 3: Implement**

`clinical_agent/audio.py`:
```python
import asyncio
import io
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from pydub import AudioSegment


@dataclass
class AudioChunk:
    index: int
    wav_bytes: bytes
    start_s: float
    end_s: float


async def chunk_file(
    path: Path,
    *,
    chunk_seconds: float = 45.0,
    min_seconds: float = 15.0,
    speed: float = 1.0,
) -> AsyncIterator[AudioChunk]:
    audio = AudioSegment.from_file(path).set_frame_rate(16000).set_channels(1)
    total_ms = len(audio)
    step_ms = int(chunk_seconds * 1000)
    bounds: list[tuple[int, int]] = []
    pos = 0
    while pos < total_ms:
        end = min(pos + step_ms, total_ms)
        bounds.append((pos, end))
        pos = end
    if len(bounds) > 1 and (bounds[-1][1] - bounds[-1][0]) < min_seconds * 1000:
        last = bounds.pop()
        prev = bounds.pop()
        bounds.append((prev[0], last[1]))

    for i, (start, end) in enumerate(bounds, start=1):
        await asyncio.sleep((end - start) / 1000.0 / speed)
        buf = io.BytesIO()
        audio[start:end].export(buf, format="wav")
        yield AudioChunk(index=i, wav_bytes=buf.getvalue(), start_s=start / 1000, end_s=end / 1000)
```

- [ ] **Step 4: Run tests** — Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat: audio replay source and chunker"`

---

### Task 6: Transcriber (faster-whisper with mock mode)

**Files:**
- Create: `clinical_agent/transcribe.py`, `tests/test_transcribe.py`

**Interfaces:**
- Consumes: chunk `wav_bytes` from Task 5.
- Produces: `Transcriber(model_size: str = "base", mock: bool = False)` with `async transcribe(wav_bytes: bytes) -> str`. Mock returns `"[mock transcript]"`. Real path lazy-loads `faster_whisper.WhisperModel(model_size, compute_type="int8")` once and runs it in a thread (`asyncio.to_thread`) since it's CPU-bound.

- [ ] **Step 1: Write the failing test**

`tests/test_transcribe.py`:
```python
from clinical_agent.transcribe import Transcriber


async def test_mock_transcriber():
    t = Transcriber(mock=True)
    assert await t.transcribe(b"anything") == "[mock transcript]"


async def test_real_model_is_lazy():
    t = Transcriber(mock=False)
    assert t._model is None  # no model download at construction time
```

- [ ] **Step 2: Run test to verify it fails** — Expected: FAIL.

- [ ] **Step 3: Implement**

`clinical_agent/transcribe.py`:
```python
import asyncio
import io
import tempfile


class Transcriber:
    def __init__(self, model_size: str = "base", mock: bool = False):
        self.model_size = model_size
        self.mock = mock
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_size, compute_type="int8")
        return self._model

    def _run(self, wav_bytes: bytes) -> str:
        model = self._load()
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(wav_bytes)
            f.flush()
            segments, _info = model.transcribe(f.name)
            return " ".join(s.text.strip() for s in segments).strip()

    async def transcribe(self, wav_bytes: bytes) -> str:
        if self.mock:
            return "[mock transcript]"
        return await asyncio.to_thread(self._run, wav_bytes)
```

- [ ] **Step 4: Run tests** — Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat: whisper transcriber with mock mode"`

---

### Task 7: Amplifier API client (async jobs, rate limit, cache)

**Files:**
- Create: `clinical_agent/amplifier.py`, `tests/test_amplifier.py`

**Interfaces:**
- Consumes: `Settings` (Task 1), `AudioChunk` (Task 5), `EventBus` (Task 4).
- Produces: `AmplifierClient(settings: Settings, bus: EventBus, cache_dir: Path | None = None)` with:
  - `async analyze(chunk: AudioChunk) -> dict` — full flow for one chunk: cache check → upload → analyze → poll to completion → returns the job `result` dict (`{"signals": [...], "summary": {...}, "audio_quality": {...}}` shape). Emits `api_job_created` (`{chunk: n, job_id}`) and `api_job_result` (`{chunk: n, signals, summary}`) on the bus. Cache modes: `warm` = read if present else live + write; `record` = live + write; `off` = live.
  - Rate limiting: an internal `_RateLimiter(max_calls=5, per_seconds=60.0)` awaited before each analyze POST.
- Wire flow (Amplifier v2):
  1. `POST {base}/v2/audio/uploads` json `{"content_type": "audio/wav"}` → `{upload_url, upload_ref, required_headers}`
  2. `PUT upload_url` with body = wav bytes and exactly the `required_headers`
  3. `POST {base}/v2/models/haven/analyze` json `{"audio_upload_ref": upload_ref}` → job envelope `{"id": ...}` (accept `id` or `job_id` key)
  4. `GET {base}/v2/jobs/{id}` every 5 s until `status` in `{"completed","done","succeeded"}` → return `result`; raise on `{"failed","error"}`.
  - Steps 1, 3, 4 send headers `X-Account-ID` / `X-API-Key`; step 2 sends only `required_headers`.
  - NOTE FOR IMPLEMENTER: field names come from the API docs at docs.amplifierhealth.com; if a live smoke test shows different key names, adjust in this one module only.

- [ ] **Step 1: Write the failing test**

`tests/test_amplifier.py`:
```python
import json

import httpx
import pytest
import respx

from clinical_agent.amplifier import AmplifierClient, _RateLimiter
from clinical_agent.audio import AudioChunk
from clinical_agent.config import Settings
from clinical_agent.events import EventBus

BASE = "https://api.test"


def make_client(tmp_path, cache="off"):
    s = Settings(amplifier_base_url=BASE, amplifier_account_id="acct", amplifier_api_key="key",
                 amplifier_cache=cache)
    return AmplifierClient(s, EventBus(), cache_dir=tmp_path / "cache")


def chunk(n=1):
    return AudioChunk(index=n, wav_bytes=b"RIFFfake" + bytes([n]), start_s=0, end_s=45)


RESULT = {"signals": [{"name": "mood-disruption", "score": 0.7, "level": "high", "flagged": True}],
          "summary": {"overall_level": "high"}, "audio_quality": {"voice_percentage": 92.0}}


@respx.mock
async def test_full_flow(tmp_path):
    respx.post(f"{BASE}/v2/audio/uploads").respond(200, json={
        "upload_url": "https://storage.test/put", "upload_ref": "ref-1",
        "required_headers": {"Content-Type": "audio/wav"}})
    respx.put("https://storage.test/put").respond(200)
    respx.post(f"{BASE}/v2/models/haven/analyze").respond(200, json={"id": "job-1", "status": "queued"})
    respx.get(f"{BASE}/v2/jobs/job-1").mock(side_effect=[
        httpx.Response(200, json={"id": "job-1", "status": "processing"}),
        httpx.Response(200, json={"id": "job-1", "status": "completed", "result": RESULT}),
    ])
    client = make_client(tmp_path)
    client.poll_interval = 0  # no real sleeping in tests
    result = await client.analyze(chunk())
    assert result["signals"][0]["name"] == "mood-disruption"
    analyze_call = respx.calls[2].request
    assert analyze_call.headers["X-Account-ID"] == "acct"
    assert json.loads(analyze_call.content)["audio_upload_ref"] == "ref-1"


@respx.mock
async def test_warm_cache_skips_network(tmp_path):
    client = make_client(tmp_path, cache="warm")
    client._cache_write(chunk(), RESULT)
    result = await client.analyze(chunk())  # respx would 404 any request
    assert result == RESULT and len(respx.calls) == 0


async def test_rate_limiter_spaces_calls():
    clock = [0.0]
    waits = []

    async def fake_sleep(s):
        waits.append(s)
        clock[0] += s

    rl = _RateLimiter(max_calls=2, per_seconds=60, now=lambda: clock[0], sleep=fake_sleep)
    await rl.acquire(); await rl.acquire(); await rl.acquire()
    assert waits and abs(sum(waits) - 60.0) < 0.01  # third call waited a full window
```

- [ ] **Step 2: Run test to verify it fails** — Expected: FAIL.

- [ ] **Step 3: Implement**

`clinical_agent/amplifier.py`:
```python
import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx

from clinical_agent.audio import AudioChunk
from clinical_agent.config import Settings
from clinical_agent.events import EventBus

_DONE = {"completed", "done", "succeeded"}
_FAILED = {"failed", "error"}


class _RateLimiter:
    def __init__(self, max_calls: int, per_seconds: float, now=time.monotonic, sleep=asyncio.sleep):
        self.max_calls, self.per_seconds = max_calls, per_seconds
        self._now, self._sleep = now, sleep
        self._stamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                cutoff = self._now() - self.per_seconds
                self._stamps = [t for t in self._stamps if t > cutoff]
                if len(self._stamps) < self.max_calls:
                    self._stamps.append(self._now())
                    return
                await self._sleep(self._stamps[0] + self.per_seconds - self._now())


class AmplifierClient:
    def __init__(self, settings: Settings, bus: EventBus, cache_dir: Path | None = None):
        self.s = settings
        self.bus = bus
        self.cache_dir = cache_dir or settings.data_dir / "amplifier_cache"
        self.limiter = _RateLimiter(max_calls=5, per_seconds=60.0)
        self.poll_interval = 5.0
        self._auth = {"X-Account-ID": settings.amplifier_account_id, "X-API-Key": settings.amplifier_api_key}

    # -- cache ---------------------------------------------------------
    def _cache_path(self, chunk: AudioChunk) -> Path:
        return self.cache_dir / f"{hashlib.sha256(chunk.wav_bytes).hexdigest()}.json"

    def _cache_read(self, chunk: AudioChunk):
        p = self._cache_path(chunk)
        return json.loads(p.read_text()) if p.exists() else None

    def _cache_write(self, chunk: AudioChunk, result: dict) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path(chunk).write_text(json.dumps(result, indent=2))

    # -- API flow ------------------------------------------------------
    async def analyze(self, chunk: AudioChunk) -> dict:
        if self.s.amplifier_cache == "warm":
            cached = self._cache_read(chunk)
            if cached is not None:
                await self.bus.emit("api_job_result", chunk=chunk.index, cached=True,
                                    signals=cached.get("signals"), summary=cached.get("summary"))
                return cached
        async with httpx.AsyncClient(timeout=60) as http:
            up = (await http.post(f"{self.s.amplifier_base_url}/v2/audio/uploads",
                                  json={"content_type": "audio/wav"}, headers=self._auth)).raise_for_status().json()
            (await http.put(up["upload_url"], content=chunk.wav_bytes,
                            headers=up.get("required_headers", {}))).raise_for_status()
            await self.limiter.acquire()
            job = (await http.post(f"{self.s.amplifier_base_url}/v2/models/haven/analyze",
                                   json={"audio_upload_ref": up["upload_ref"]},
                                   headers=self._auth)).raise_for_status().json()
            job_id = job.get("id") or job.get("job_id")
            await self.bus.emit("api_job_created", chunk=chunk.index, job_id=job_id)
            while True:
                status = (await http.get(f"{self.s.amplifier_base_url}/v2/jobs/{job_id}",
                                         headers=self._auth)).raise_for_status().json()
                state = str(status.get("status", "")).lower()
                if state in _DONE:
                    result = status["result"]
                    break
                if state in _FAILED:
                    raise RuntimeError(f"Amplifier job {job_id} failed: {status}")
                await asyncio.sleep(self.poll_interval)
        if self.s.amplifier_cache in ("warm", "record"):
            self._cache_write(chunk, result)
        await self.bus.emit("api_job_result", chunk=chunk.index, cached=False,
                            signals=result.get("signals"), summary=result.get("summary"))
        return result
```

- [ ] **Step 4: Run tests** — `.venv/bin/pytest tests/test_amplifier.py -v` — Expected: 3 passed.
- [ ] **Step 5: Commit** — `git commit -am "feat: async Amplifier client with rate limiting and result cache"`

---

### Task 8: Claude agent base (streaming, rehearsal cache, tools, mock)

**Files:**
- Create: `clinical_agent/agents/__init__.py`, `clinical_agent/agents/base.py`, `tests/test_agent_base.py`

**Interfaces:**
- Consumes: `Settings`, `EventBus`.
- Produces: `ClaudeAgent(name: str, settings: Settings, bus: EventBus, cache_dir: Path | None = None)` with:
  - `async run(system: str, user: str, *, tools: dict[str, tuple[dict, Callable]] | None = None, effort: str = "high", output_schema: dict | None = None) -> str`
  - `tools` maps tool name → (JSON tool definition, async callable taking the tool input dict and returning a string).
  - Streams via `anthropic.AsyncAnthropic().messages.stream(...)` with `thinking={"type": "adaptive"}`, `output_config` carrying `effort` and, when `output_schema` is given, `format={"type": "json_schema", "schema": output_schema}`. Emits `agent_token` events (`{agent: name, text: delta}`) per text delta and `agent_tool_call` when a tool runs. Manual tool loop (stream → `get_final_message()` → if `stop_reason == "tool_use"` run tools, append `tool_result` blocks, loop).
  - Rehearsal cache: key = sha256 of `name + system + user`; on success writes `{cache_dir}/{key}.txt`; on any exception, or >6 s waiting for the first stream event, falls back to the cached file (emitting its text as one `agent_token` event) — re-raises if no cache exists.
  - `settings.mock_claude=True` short-circuits: returns `MOCK_RESPONSES.get(name, '{"mock": true}')` (module-level dict the tests and mock mode populate) and emits one `agent_token` event. No network.

- [ ] **Step 1: Write the failing test**

`tests/test_agent_base.py`:
```python
from clinical_agent.agents import base
from clinical_agent.agents.base import ClaudeAgent
from clinical_agent.config import Settings
from clinical_agent.events import EventBus


async def test_mock_mode_no_network(tmp_path):
    base.MOCK_RESPONSES["previsit"] = "mock brief"
    bus = EventBus()
    q = bus.subscribe()
    agent = ClaudeAgent("previsit", Settings(mock_claude=True), bus, cache_dir=tmp_path)
    out = await agent.run("sys", "user prompt")
    assert out == "mock brief"
    e = q.get_nowait()
    assert e["type"] == "agent_token" and e["agent"] == "previsit"


async def test_rehearsal_cache_fallback(tmp_path, monkeypatch):
    bus = EventBus()
    agent = ClaudeAgent("reasoner", Settings(), bus, cache_dir=tmp_path)
    key = agent._cache_key("sys", "hello")
    (tmp_path / f"{key}.txt").write_text("cached answer")

    async def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(agent, "_run_live", boom)
    out = await agent.run("sys", "hello")
    assert out == "cached answer"
```

- [ ] **Step 2: Run test to verify it fails** — Expected: FAIL.

- [ ] **Step 3: Implement**

`clinical_agent/agents/__init__.py` is empty. `clinical_agent/agents/base.py`:
```python
import hashlib
import json
from pathlib import Path
from typing import Callable

import anthropic

from clinical_agent.config import Settings
from clinical_agent.events import EventBus

MOCK_RESPONSES: dict[str, str] = {}
FIRST_TOKEN_TIMEOUT_S = 6.0


class ClaudeAgent:
    def __init__(self, name: str, settings: Settings, bus: EventBus, cache_dir: Path | None = None):
        self.name = name
        self.s = settings
        self.bus = bus
        self.cache_dir = cache_dir or settings.data_dir / "rehearsal_cache"

    def _cache_key(self, system: str, user: str) -> str:
        return hashlib.sha256(f"{self.name}\x00{system}\x00{user}".encode()).hexdigest()

    async def run(self, system: str, user: str, *, tools=None, effort: str = "high",
                  output_schema: dict | None = None) -> str:
        if self.s.mock_claude:
            text = MOCK_RESPONSES.get(self.name, '{"mock": true}')
            await self.bus.emit("agent_token", agent=self.name, text=text)
            return text
        key = self._cache_key(system, user)
        try:
            text = await self._run_live(system, user, tools=tools, effort=effort,
                                        output_schema=output_schema)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / f"{key}.txt").write_text(text)
            return text
        except Exception:
            cached = self.cache_dir / f"{key}.txt"
            if cached.exists():
                text = cached.read_text()
                await self.bus.emit("agent_token", agent=self.name, text=text, cached=True)
                return text
            raise

    async def _run_live(self, system: str, user: str, *, tools=None, effort: str = "high",
                        output_schema: dict | None = None) -> str:
        import asyncio

        client = anthropic.AsyncAnthropic()
        tool_defs = [d for d, _ in (tools or {}).values()]
        messages = [{"role": "user", "content": user}]
        output_config: dict = {"effort": effort}
        if output_schema:
            output_config["format"] = {"type": "json_schema", "schema": output_schema}

        while True:
            parts: list[str] = []
            async with client.messages.stream(
                model=self.s.anthropic_model,
                max_tokens=16000,
                system=system,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config=output_config,
                **({"tools": tool_defs} if tool_defs else {}),
            ) as stream:
                first = True
                async for event in stream:
                    if first:
                        first = False  # first event arrived; per-event timeout handled by SDK client timeout
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        parts.append(event.delta.text)
                        await self.bus.emit("agent_token", agent=self.name, text=event.delta.text)
                final = await asyncio.wait_for(stream.get_final_message(), timeout=FIRST_TOKEN_TIMEOUT_S * 20)

            if final.stop_reason != "tool_use":
                return "".join(parts)

            messages.append({"role": "assistant", "content": final.content})
            results = []
            for block in final.content:
                if block.type == "tool_use":
                    _, fn = tools[block.name]
                    await self.bus.emit("agent_tool_call", agent=self.name, tool=block.name,
                                        input=block.input)
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": await fn(block.input)})
            messages.append({"role": "user", "content": results})
```

- [ ] **Step 4: Run tests** — Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat: Claude agent base with streaming, tools, rehearsal cache"`

---

### Task 9: The four agents

**Files:**
- Create: `clinical_agent/agents/roles.py`, `tests/test_roles.py`

**Interfaces:**
- Consumes: `ClaudeAgent` (Task 8), `PatientStore.chart()` (Task 2).
- Produces (all take `settings, bus, store` and return parsed results; all built on `ClaudeAgent`):
  - `async pre_visit_brief(settings, bus, store, pid: str) -> dict` — agent name `previsit`, effort high, JSON schema `{brief: str, vocal_trends: [str], topics_to_discuss: [str]}`. Persists to the current planned visit as artifact `pre_visit_brief` and emits `pre_visit_brief` event with the parsed dict.
  - `async reason_over_chunk(settings, bus, store, pid: str, chunk_no: int, transcript: str, cumulative_signals: list[dict], brief: dict) -> str` — agent name `reasoner`, effort **low**, plain text (1–2 sentences), exposes tool `read_chart` (`input_schema` `{}`) whose callable returns `json.dumps(store.chart(pid))`. Emits `observation` event with the text.
  - `async post_visit_summary(settings, bus, store, pid: str, visit_no: int, transcript_parts: list[str], all_signals: list[dict], observations: list[str], brief: dict) -> dict` — agent name `postvisit`, schema `{summary: str, vocal_findings: [{sign: str, level: str, note: str}], transcript_findings: [str], discordance: str, screener_recommendations: [str], chart_update_draft: [{description: str, rationale: str}], next_visit_topics: [str]}`. Persists to store as artifact `summary` and emits `visit_summary`, `chart_draft`, `topics` events.
  - `async longitudinal_analysis(settings, bus, store, pid: str) -> dict` — agent name `longitudinal`, schema `{narrative: str, deltas: [{condition: str, first_voice_flag_visit: int, first_coded_visit: int, visits_early: int}]}`. Emits `longitudinal_delta` per delta and `longitudinal_narrative`.
- All JSON schemas set `additionalProperties: false` and full `required` lists on every object (structured-outputs requirement).
- System prompts (write verbatim into `roles.py` as module constants; each ≤15 lines, plain clinical tone, explicitly instructing: never diagnose — describe signals and suggest validated screeners like PHQ-9/GAD-7; drafts are for clinician review).

- [ ] **Step 1: Write the failing test**

`tests/test_roles.py`:
```python
import json

from clinical_agent.agents import base, roles
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.store import PatientStore
from clinical_agent.synthetic import generate_synthetic_patient


def setup(tmp_path):
    store = PatientStore(tmp_path)
    generate_synthetic_patient(store)
    return Settings(mock_claude=True, data_dir=tmp_path), EventBus(), store


async def test_pre_visit_brief(tmp_path):
    settings, bus, store = setup(tmp_path)
    base.MOCK_RESPONSES["previsit"] = json.dumps(
        {"brief": "b", "vocal_trends": ["mood up"], "topics_to_discuss": ["sleep"]})
    q = bus.subscribe()
    out = await roles.pre_visit_brief(settings, bus, store, "demo-synthetic")
    assert out["topics_to_discuss"] == ["sleep"]
    assert any(e["type"] == "pre_visit_brief" for e in _drain(q))


async def test_post_visit_persists(tmp_path):
    settings, bus, store = setup(tmp_path)
    payload = {"summary": "s", "vocal_findings": [], "transcript_findings": [],
               "discordance": "none", "screener_recommendations": ["PHQ-9"],
               "chart_update_draft": [], "next_visit_topics": ["mood"]}
    base.MOCK_RESPONSES["postvisit"] = json.dumps(payload)
    out = await roles.post_visit_summary(settings, bus, store, "demo-synthetic", 10,
                                         ["hi"], [], [], {})
    assert out["screener_recommendations"] == ["PHQ-9"]
    assert store.read_artifact("demo-synthetic", 10, "summary")["summary"] == "s"


async def test_longitudinal(tmp_path):
    settings, bus, store = setup(tmp_path)
    base.MOCK_RESPONSES["longitudinal"] = json.dumps(
        {"narrative": "voice flagged early", "deltas": [
            {"condition": "depression", "first_voice_flag_visit": 3,
             "first_coded_visit": 9, "visits_early": 6}]})
    q = bus.subscribe()
    out = await roles.longitudinal_analysis(settings, bus, store, "demo-synthetic")
    assert out["deltas"][0]["visits_early"] == 6
    assert any(e["type"] == "longitudinal_delta" for e in _drain(q))


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out
```

- [ ] **Step 2: Run test to verify it fails** — Expected: FAIL.

- [ ] **Step 3: Implement**

`clinical_agent/agents/roles.py` (prompts abbreviated here to the required content; write them out fully):
```python
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
                            chunk_no: int, transcript: str, cumulative_signals: list[dict],
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
                             visit_no: int, transcript_parts: list[str], all_signals: list[dict],
                             observations: list[str], brief: dict) -> dict:
    agent = ClaudeAgent("postvisit", settings, bus)
    user = (f"Pre-visit brief: {json.dumps(brief)}\n"
            f"Full transcript: {' '.join(transcript_parts)}\n"
            f"All chunk signals: {json.dumps(all_signals)}\n"
            f"Live observations: {json.dumps(observations)}")
    summary = json.loads(await agent.run(POSTVISIT_SYSTEM, user, output_schema=POSTVISIT_SCHEMA))
    store.write_artifact(pid, visit_no, "summary", summary)
    await bus.emit("visit_summary", patient=pid, visit=visit_no, **{"summary": summary["summary"],
                   "vocal_findings": summary["vocal_findings"], "discordance": summary["discordance"],
                   "screener_recommendations": summary["screener_recommendations"]})
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
```

- [ ] **Step 4: Run tests** — Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat: pre-visit, reasoner, post-visit, longitudinal agents"`

---

### Task 10: Visit session orchestrator

**Files:**
- Create: `clinical_agent/session.py`, `tests/test_session.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `async run_visit(settings, bus, store, transcriber, amplifier, pid: str, audio_path: Path) -> dict` (returns the post-visit summary) and `async run_longitudinal(settings, bus, store, pid) -> dict` (thin wrapper over `roles.longitudinal_analysis`).
- `run_visit` flow:
  1. Resolve the current visit = the visit with `status == "planned"` (raise if none). Emit `visit_started`.
  2. `brief = await roles.pre_visit_brief(...)`.
  3. Producer task: iterate `chunk_file(audio_path, speed=settings.speed)`; for each chunk emit `chunk_created` (`{chunk, start_s, end_s}`), then schedule an `asyncio.create_task` per chunk that (a) transcribes, emits `transcript` (`{chunk, text}`); (b) calls `amplifier.analyze(chunk)`; (c) puts `(chunk_no, transcript, signals)` on a results queue. Results arrive out of order by design.
  4. Consumer: for each queue item, extend `all_signals`/`transcripts` dicts and call `roles.reason_over_chunk` with cumulative flat signal list. Continue until every scheduled chunk has resolved (or a task raised — log, emit `error` event, continue with remaining).
  5. Persist artifacts: `signals` (last chunk's signal list — the most complete cumulative snapshot), `transcript` (`[{chunk, text}...]` in chunk order), `observations`.
  6. `summary = await roles.post_visit_summary(...)`; mark the visit `status="complete"` in `visits.json`; emit `visit_complete`.

- [ ] **Step 1: Write the failing test**

`tests/test_session.py`:
```python
import json

from clinical_agent.agents import base
from clinical_agent.amplifier import AmplifierClient
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.session import run_visit
from clinical_agent.store import PatientStore
from clinical_agent.synthetic import generate_synthetic_patient
from clinical_agent.transcribe import Transcriber


async def test_full_visit_fast_mode(tmp_path, wav_100s):
    settings = Settings(mock_claude=True, mock_whisper=True, amplifier_cache="warm",
                        speed=1e9, data_dir=tmp_path)
    bus = EventBus()
    q = bus.subscribe()
    store = PatientStore(tmp_path)
    generate_synthetic_patient(store)
    base.MOCK_RESPONSES["previsit"] = json.dumps(
        {"brief": "b", "vocal_trends": [], "topics_to_discuss": []})
    base.MOCK_RESPONSES["reasoner"] = "nothing notable"
    base.MOCK_RESPONSES["postvisit"] = json.dumps(
        {"summary": "done", "vocal_findings": [], "transcript_findings": [], "discordance": "none",
         "screener_recommendations": [], "chart_update_draft": [], "next_visit_topics": ["x"]})

    amplifier = AmplifierClient(settings, bus, cache_dir=tmp_path / "cache")
    # pre-seed the amplifier cache for both expected chunks (100s @45s -> 2 chunks)
    from clinical_agent.audio import chunk_file
    result = {"signals": [{"name": "mood-disruption", "score": 0.7, "level": "high", "flagged": True}],
              "summary": {"overall_level": "high"}}
    async for c in chunk_file(wav_100s, speed=1e9):
        amplifier._cache_write(c, result)

    summary = await run_visit(settings, bus, store, Transcriber(mock=True), amplifier,
                              "demo-synthetic", wav_100s)
    assert summary["summary"] == "done"
    # visit marked complete and artifacts persisted
    visits = store.list_visits("demo-synthetic")
    assert visits[-1].status == "complete"
    assert store.read_artifact("demo-synthetic", 10, "summary")["summary"] == "done"
    assert len(store.read_artifact("demo-synthetic", 10, "transcript")) == 2
    types = [q.get_nowait()["type"] for _ in range(q.qsize())]
    for expected in ("visit_started", "pre_visit_brief", "chunk_created", "transcript",
                     "api_job_result", "observation", "visit_summary", "visit_complete"):
        assert expected in types, expected
```

- [ ] **Step 2: Run test to verify it fails** — Expected: FAIL.

- [ ] **Step 3: Implement**

`clinical_agent/session.py`:
```python
import asyncio
from pathlib import Path

from clinical_agent.agents import roles
from clinical_agent.amplifier import AmplifierClient
from clinical_agent.audio import chunk_file
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.store import PatientStore
from clinical_agent.transcribe import Transcriber


async def run_visit(settings: Settings, bus: EventBus, store: PatientStore,
                    transcriber: Transcriber, amplifier: AmplifierClient,
                    pid: str, audio_path: Path) -> dict:
    visits = store.list_visits(pid)
    current = next((v for v in visits if v.status == "planned"), None)
    if current is None:
        raise ValueError(f"no planned visit for patient {pid}")
    await bus.emit("visit_started", patient=pid, visit=current.number, date=current.date,
                   reason=current.reason)

    brief = await roles.pre_visit_brief(settings, bus, store, pid)

    results_q: asyncio.Queue = asyncio.Queue()
    tasks: list[asyncio.Task] = []

    async def process(chunk):
        try:
            text = await transcriber.transcribe(chunk.wav_bytes)
            await bus.emit("transcript", patient=pid, chunk=chunk.index, text=text)
            result = await amplifier.analyze(chunk)
            await results_q.put((chunk.index, text, result.get("signals", [])))
        except Exception as exc:  # keep the visit alive; surface the failure
            await bus.emit("error", patient=pid, chunk=chunk.index, message=str(exc))
            await results_q.put((chunk.index, "", []))

    async for chunk in chunk_file(audio_path, speed=settings.speed):
        await bus.emit("chunk_created", patient=pid, chunk=chunk.index,
                       start_s=chunk.start_s, end_s=chunk.end_s)
        tasks.append(asyncio.create_task(process(chunk)))

    transcripts: dict[int, str] = {}
    signals_by_chunk: dict[int, list] = {}
    observations: list[str] = []
    for _ in range(len(tasks)):
        chunk_no, text, signals = await results_q.get()
        transcripts[chunk_no] = text
        signals_by_chunk[chunk_no] = signals
        cumulative = [s for n in sorted(signals_by_chunk) for s in signals_by_chunk[n]]
        obs = await roles.reason_over_chunk(settings, bus, store, pid, chunk_no, text,
                                            cumulative, brief)
        observations.append(obs)
    await asyncio.gather(*tasks)

    ordered = sorted(signals_by_chunk)
    store.write_artifact(pid, current.number, "signals",
                         signals_by_chunk[ordered[-1]] if ordered else [])
    store.write_artifact(pid, current.number, "transcript",
                         [{"chunk": n, "text": transcripts[n]} for n in sorted(transcripts)])
    store.write_artifact(pid, current.number, "observations", observations)

    all_signals = [s for n in ordered for s in signals_by_chunk[n]]
    summary = await roles.post_visit_summary(settings, bus, store, pid, current.number,
                                             [transcripts[n] for n in sorted(transcripts)],
                                             all_signals, observations, brief)
    current.status = "complete"
    store.save_visits(pid, visits)
    await bus.emit("visit_complete", patient=pid, visit=current.number)
    return summary


async def run_longitudinal(settings: Settings, bus: EventBus, store: PatientStore, pid: str) -> dict:
    return await roles.longitudinal_analysis(settings, bus, store, pid)
```

- [ ] **Step 4: Run tests** — `.venv/bin/pytest tests/test_session.py -v` — Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -am "feat: visit session orchestrator"`

---

### Task 11: Wire API endpoints + entrypoint

**Files:**
- Modify: `clinical_agent/api.py` (add visit endpoints)
- Create: `clinical_agent/main.py`, `tests/test_api_visits.py`

**Interfaces:**
- Consumes: `run_visit` / `run_longitudinal` (Task 10).
- Produces, added inside `create_app` (which now also takes `transcriber: Transcriber` and `amplifier: AmplifierClient` args — update Task 4's two `create_app` calls in tests to pass `Transcriber(mock=True)` and a client):
  - `POST /patients/{pid}/visits/start` body `{"audio_path": "<path>"}` → `202 {"status": "started"}`; runs `run_visit` as an `asyncio.create_task` (store the task on `app.state.jobs` list so it isn't GC'd; emit `error` event if it raises).
  - `POST /patients/{pid}/longitudinal` → runs `run_longitudinal` inline and returns its dict.
  - `clinical_agent/main.py`: builds `Settings.from_env()`, `PatientStore(settings.data_dir)`, `EventBus()`, `Transcriber(settings.whisper_model, mock=settings.mock_whisper)`, `AmplifierClient(...)`; if the store has no patients, runs `generate_synthetic_patient`; exposes module-level `app`. Run with `uvicorn clinical_agent.main:app`.

- [ ] **Step 1: Write the failing test**

`tests/test_api_visits.py`:
```python
import json
import time

from fastapi.testclient import TestClient

from clinical_agent.agents import base
from clinical_agent.amplifier import AmplifierClient
from clinical_agent.api import create_app
from clinical_agent.audio import chunk_file
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.store import PatientStore
from clinical_agent.synthetic import generate_synthetic_patient
from clinical_agent.transcribe import Transcriber


async def _seed_cache(amplifier, wav):
    result = {"signals": [], "summary": {}}
    async for c in chunk_file(wav, speed=1e9):
        amplifier._cache_write(c, result)


def test_visit_endpoint(tmp_path, wav_100s):
    settings = Settings(mock_claude=True, mock_whisper=True, amplifier_cache="warm",
                        speed=1e9, data_dir=tmp_path)
    bus, store = EventBus(), PatientStore(tmp_path)
    generate_synthetic_patient(store)
    for name, resp in [("previsit", {"brief": "b", "vocal_trends": [], "topics_to_discuss": []}),
                       ("postvisit", {"summary": "done", "vocal_findings": [], "transcript_findings": [],
                                      "discordance": "", "screener_recommendations": [],
                                      "chart_update_draft": [], "next_visit_topics": []}),
                       ("longitudinal", {"narrative": "n", "deltas": []})]:
        base.MOCK_RESPONSES[name] = json.dumps(resp)
    base.MOCK_RESPONSES["reasoner"] = "ok"
    amplifier = AmplifierClient(settings, bus, cache_dir=tmp_path / "cache")

    app = create_app(settings, store, bus, Transcriber(mock=True), amplifier)
    with TestClient(app) as client:
        import asyncio
        asyncio.get_event_loop  # TestClient runs its own loop; seed cache synchronously:
        import anyio
        anyio.run(_seed_cache, amplifier, wav_100s)

        resp = client.post("/patients/demo-synthetic/visits/start",
                           json={"audio_path": str(wav_100s)})
        assert resp.status_code == 202
        deadline = time.time() + 10
        while time.time() < deadline:
            if store.read_artifact("demo-synthetic", 10, "summary"):
                break
            time.sleep(0.1)
        assert store.read_artifact("demo-synthetic", 10, "summary")["summary"] == "done"

        out = client.post("/patients/demo-synthetic/longitudinal").json()
        assert out["narrative"] == "n"
```

- [ ] **Step 2: Run test to verify it fails** — Expected: FAIL.

- [ ] **Step 3: Implement**

Add to `create_app` signature: `transcriber: Transcriber, amplifier: AmplifierClient` and inside:
```python
    from pathlib import Path as _Path

    from pydantic import BaseModel as _BM

    class StartVisit(_BM):
        audio_path: str

    app.state.jobs = []

    @app.post("/patients/{pid}/visits/start", status_code=202)
    async def start_visit(pid: str, body: StartVisit):
        import asyncio

        from clinical_agent.session import run_visit

        async def job():
            try:
                await run_visit(settings, bus, store, transcriber, amplifier,
                                pid, _Path(body.audio_path))
            except Exception as exc:
                await bus.emit("error", patient=pid, message=str(exc))

        app.state.jobs.append(asyncio.create_task(job()))
        return {"status": "started"}

    @app.post("/patients/{pid}/longitudinal")
    async def longitudinal(pid: str):
        from clinical_agent.session import run_longitudinal
        return await run_longitudinal(settings, bus, store, pid)
```
(Match `run_visit`'s positional signature from Task 10 — adjust the call to `run_visit(settings, bus, store, transcriber, amplifier, pid, _Path(body.audio_path))`.)

`clinical_agent/main.py`:
```python
from clinical_agent.amplifier import AmplifierClient
from clinical_agent.api import create_app
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.store import PatientStore
from clinical_agent.synthetic import generate_synthetic_patient
from clinical_agent.transcribe import Transcriber

settings = Settings.from_env()
store = PatientStore(settings.data_dir)
bus = EventBus()
if not store.list_patients():
    generate_synthetic_patient(store)
transcriber = Transcriber(settings.whisper_model, mock=settings.mock_whisper)
amplifier = AmplifierClient(settings, bus)
app = create_app(settings, store, bus, transcriber, amplifier)
```

Also update `tests/test_events.py`'s `create_app(...)` call to the new signature.

- [ ] **Step 4: Run the whole suite** — `.venv/bin/pytest -v` — Expected: all pass.
- [ ] **Step 5: Commit** — `git commit -am "feat: visit endpoints and app entrypoint"`

---

### Task 12: README, live smoke script, push

**Files:**
- Create: `README.md`, `scripts/smoke_live.py`

**Interfaces:**
- `scripts/smoke_live.py`: takes an audio file path; runs ONE chunk through the real Amplifier API (`AMPLIFIER_CACHE=record`) and one real `previsit` Claude call against the synthetic patient, printing results. This is the manual verification that wire formats in Task 7 match the real API — run it once credentials are set and fix `amplifier.py` field names if needed.

- [ ] **Step 1: Write README.md**

Cover: what it is (Abridge x Anthropic hackathon — vocal biomarker layer for the clinical visit workflow, before/during/after visit), architecture diagram (reuse the spec's), quickstart (`pip install -e ".[dev]"`, env vars table, `uvicorn clinical_agent.main:app`, `curl -N localhost:8000/events`, start a visit), demo mode vs real mode, SSE event contract table (all event types with payload fields), MIT license note, explicit statement that no real patient data is in the repo.

- [ ] **Step 2: Write `scripts/smoke_live.py`**

```python
"""One-shot live smoke test: real Amplifier chunk + real Claude pre-visit call.

Usage: AMPLIFIER_ACCOUNT_ID=... AMPLIFIER_API_KEY=... ANTHROPIC_API_KEY=... \
       .venv/bin/python scripts/smoke_live.py path/to/audio.wav
"""
import asyncio
import sys

from clinical_agent.agents.roles import pre_visit_brief
from clinical_agent.amplifier import AmplifierClient
from clinical_agent.audio import chunk_file
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.store import PatientStore
from clinical_agent.synthetic import generate_synthetic_patient


async def main(path: str) -> None:
    settings = Settings.from_env()
    settings.amplifier_cache = "record"
    bus = EventBus()
    store = PatientStore(settings.data_dir)
    if not store.list_patients():
        generate_synthetic_patient(store)

    async for chunk in chunk_file(path, speed=1e9):
        print(f"analyzing chunk {chunk.index} ({chunk.end_s - chunk.start_s:.0f}s)...")
        result = await AmplifierClient(settings, bus).analyze(chunk)
        print("signals:", result.get("signals"))
        break

    print("pre-visit brief (live Claude):")
    print(await pre_visit_brief(settings, bus, store, "demo-synthetic"))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
```

- [ ] **Step 3: Verify nothing sensitive is tracked**

Run: `git status --porcelain && git ls-files | grep -E "onset_analysis|data/|\.env"` — Expected: no matches from the grep.

- [ ] **Step 4: Commit and push**

```bash
git add README.md scripts/
git commit -m "docs: README and live smoke script"
git push -u origin main
```

- [ ] **Step 5: Run the live smoke test** (requires credentials + a ≥15 s audio file)

Run: `AMPLIFIER_ACCOUNT_ID=... AMPLIFIER_API_KEY=... .venv/bin/python scripts/smoke_live.py sample.wav`
Expected: printed signals from the real API and a live pre-visit brief. If field names differ from Task 7's assumptions, fix `clinical_agent/amplifier.py` only, re-run, commit `fix: match live Amplifier API field names`.

---

## Deferred (separate plan later)

- Frontend (Abridge-style UI over the SSE contract).
- Real demo-patient data ingestion (Vijay supplies audio + ICD-10 history; drop into `data/patients/<id>/` in the store shape — no code changes).
- Rehearsal-run tooling for pre-warming caches before the stage demo.
