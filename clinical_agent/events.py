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
