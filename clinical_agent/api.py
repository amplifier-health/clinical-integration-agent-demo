import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from clinical_agent.amplifier import AmplifierClient
from clinical_agent.config import Settings
from clinical_agent.events import EventBus
from clinical_agent.store import PatientStore
from clinical_agent.transcribe import Transcriber


class StartVisit(BaseModel):
    audio_path: str | None = None  # omit to replay the selected appointment's own audio
    visit: int | None = None       # which appointment to run; default = the planned visit


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
        from clinical_agent.session import replay_visit, run_visit

        async def job():
            try:
                if body.audio_path:  # true live stream from a WAV
                    await run_visit(settings, bus, store, transcriber, amplifier, pid,
                                    Path(body.audio_path), visit_number=body.visit)
                else:  # demo default: replay the appointment from its precomputed aria results
                    await replay_visit(settings, bus, store, pid, visit_number=body.visit)
            except Exception as exc:
                await bus.emit("error", patient=pid, message=str(exc))

        app.state.jobs.append(asyncio.create_task(job()))
        return {"status": "started"}

    @app.post("/patients/{pid}/longitudinal")
    async def longitudinal(pid: str):
        from clinical_agent.session import run_longitudinal
        return await run_longitudinal(settings, bus, store, pid)

    return app
