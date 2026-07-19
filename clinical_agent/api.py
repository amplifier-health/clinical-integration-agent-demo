import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from clinical_agent import contract
from clinical_agent.amplifier import AmplifierClient
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.store import PatientStore
from clinical_agent.transcribe import Transcriber


class StartVisit(BaseModel):
    audio_path: str | None = None  # omit to replay the selected appointment's own audio
    visit: int | None = None       # which appointment to run; default = the planned visit
    config: dict | None = None     # per-visit clinician settings (e.g. {"explainability": "detailed"})


def create_app(settings: Settings, store: PatientStore, bus: EventBus,
               transcriber: Transcriber, amplifier: AmplifierClient) -> FastAPI:
    app = FastAPI(title="Clinical Integration Agent")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # demo app, no auth; lock down if this outlives the hackathon
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings, app.state.store, app.state.bus = settings, store, bus
    app.state.jobs = []

    @app.get("/patients")
    def patients():
        return [p.model_dump() for p in store.list_patients()]

    @app.get("/patients/{pid}/chart")
    def chart(pid: str):
        return store.chart(pid)

    @app.get("/patients/{pid}/visits")
    def visits(pid: str):
        return [v.model_dump() for v in store.list_visits(pid)]

    @app.get("/contract")
    def contract_doc():
        """The published output contract (version, typed events by phase, clinical vs
        telemetry, envelope) — what a plugin consumer integrates against."""
        p = Path(__file__).resolve().parent.parent / "docs" / "contract" / "contract.json"
        return json.loads(p.read_text()) if p.exists() else {
            "contract_version": contract.CONTRACT_VERSION,
            "clinical_types": sorted(contract.CLINICAL_TYPES),
            "error": "contract.json not generated — run scripts/dump_contract.py",
        }

    @app.get("/patients/{pid}/visits/{visit}/result")
    def visit_result(pid: str, visit: int):
        """The folded snapshot of a visit — the same clinical outputs the stream
        delivers as deltas, reduced to final state. For late joiners / reconnects /
        EHR write-back that want the finished visit rather than the live stream."""
        meta = next((v for v in store.list_visits(pid) if v.number == visit), None)
        summary = store.read_artifact(pid, visit, "summary") or {}
        return {
            "contract_version": contract.CONTRACT_VERSION,
            "patient_id": pid,
            "visit": visit,
            "status": meta.status if meta else "unknown",
            "signals": store.read_artifact(pid, visit, "signals") or [],
            "observations": store.read_artifact(pid, visit, "observations") or [],
            "summary": summary.get("summary"),
            "vocal_findings": summary.get("vocal_findings", []),
            "discordance": summary.get("discordance"),
            "screener_recommendations": summary.get("screener_recommendations", []),
            "chart_draft": summary.get("chart_update_draft", []),
            "next_visit_topics": summary.get("next_visit_topics", []),
            "note": store.read_artifact(pid, visit, "note"),
        }

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

    @app.post("/patients/{pid}/visits/start", status_code=202)
    async def start_visit(pid: str, body: StartVisit):
        from clinical_agent.clinician_config import ClinicianConfig
        from clinical_agent.session import replay_visit, run_visit

        cfg = ClinicianConfig.from_override(body.config)

        async def job():
            try:
                if body.audio_path:  # true live stream from a WAV
                    await run_visit(settings, bus, store, transcriber, amplifier, pid,
                                    Path(body.audio_path), visit_number=body.visit, config=cfg)
                else:  # demo default: replay the appointment from its precomputed aria results
                    await replay_visit(settings, bus, store, pid, visit_number=body.visit, config=cfg)
            except Exception as exc:
                await bus.emit("error", patient=pid, message=str(exc))

        app.state.jobs.append(asyncio.create_task(job()))
        return {"status": "started"}

    @app.websocket("/patients/{pid}/visits/stream")
    async def visit_stream(ws: WebSocket, pid: str, visit: int | None = None):
        """REAL ingestion boundary: the ambient scribe streams raw PCM frames here and we bucket
        them into the 30s/15s pipeline. Results come back on GET /events (same typed contract).
        The demo doesn't use this — it replays precomputed results — but this is the production wire."""
        from clinical_agent.clinician_config import ClinicianConfig
        from clinical_agent.session import run_streaming_visit
        await ws.accept()
        try:
            await run_streaming_visit(settings, bus, store, transcriber, amplifier, pid, ws,
                                      visit_number=visit, config=ClinicianConfig())
        except Exception as exc:
            await bus.emit("error", patient=pid, message=str(exc))
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    @app.post("/patients/{pid}/longitudinal")
    async def longitudinal(pid: str):
        from clinical_agent.session import run_longitudinal
        return await run_longitudinal(settings, bus, store, pid)

    return app
