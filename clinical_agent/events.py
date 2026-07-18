import asyncio
import contextvars
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from clinical_agent import contract

log = logging.getLogger(__name__)


@dataclass
class SessionContext:
    session_id: str
    patient_id: str | None = None
    visit: int | None = None
    _seq: int = field(default=0)

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq


_session: contextvars.ContextVar[SessionContext | None] = contextvars.ContextVar("session", default=None)


def start_session(patient_id: str | None = None, visit: int | None = None) -> SessionContext:
    """Bind a session to the current async context. Tasks spawned afterwards inherit it,
    so every emit from the visit is stamped with the same session_id and a monotonic seq."""
    ctx = SessionContext(session_id=f"sesn_{uuid.uuid4().hex[:16]}", patient_id=patient_id, visit=visit)
    _session.set(ctx)
    return ctx


def current_session() -> SessionContext | None:
    return _session.get()


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
        # Validate against the registered model, then stamp the envelope. Wire stays
        # flat for UI back-compat. Fail-soft: a validation error must never abort a
        # live visit (many call sites emit unwrapped), so on drift we log, mark the
        # frame `contract_error`, and still deliver it. The test suite enforces the
        # contract strictly via contract.validate — the teeth are at dev time.
        contract_error = None
        try:
            data = contract.validate(type, data)
        except Exception as exc:
            contract_error = str(exc).splitlines()[0]
            log.warning("event %r failed contract validation: %s", type, contract_error)
        ctx = _session.get()
        envelope = {
            "type": type,
            "contract_version": contract.CONTRACT_VERSION,
            "phase": contract.phase_for(type),
            "session_id": ctx.session_id if ctx else None,
            "seq": ctx.next_seq() if ctx else None,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if contract_error:
            envelope["contract_error"] = contract_error
        event = {**envelope, **data}
        for q in list(self._subs):
            q.put_nowait(event)
