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
