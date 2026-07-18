import asyncio
import json
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

    # One conversation for the whole visit — the reasoner (per chunk) and the post-visit agent
    # share this memory, so the final analysis is done by an agent that saw the entire encounter.
    visit_history: list = [{"role": "user", "content":
        f"VISIT START for patient {pid}. Chart:\n{json.dumps(store.chart(pid))}\n"
        f"Pre-visit brief:\n{json.dumps(brief)}\n"
        "Live voice-biomarker ticks and transcript follow as the visit is recorded."}]

    results_q: asyncio.Queue = asyncio.Queue()
    tasks: list[asyncio.Task] = []

    async def process(chunk):
        text = ""
        try:
            text = await transcriber.transcribe(chunk.wav_bytes)
            await bus.emit("transcript", patient=pid, chunk=chunk.index, text=text)
            result = await amplifier.analyze(chunk)
            await results_q.put((chunk.index, text, result.get("signals", [])))
        except Exception as exc:  # keep the visit alive; surface the failure
            await bus.emit("error", patient=pid, chunk=chunk.index, message=str(exc))
            await results_q.put((chunk.index, text, []))  # keep the transcript if we got one

    async for chunk in chunk_file(audio_path, speed=settings.speed):
        await bus.emit("chunk_created", patient=pid, chunk=chunk.index,
                       start_s=chunk.start_s, end_s=chunk.end_s)
        tasks.append(asyncio.create_task(process(chunk)))

    transcripts: dict = {}
    signals_by_chunk: dict = {}
    observations: list = []
    for _ in range(len(tasks)):
        chunk_no, text, signals = await results_q.get()
        transcripts[chunk_no] = text
        signals_by_chunk[chunk_no] = signals
        cumulative = [s for n in sorted(signals_by_chunk) for s in signals_by_chunk[n]]
        obs = await roles.reason_over_chunk(settings, bus, store, pid, chunk_no, text,
                                            cumulative, brief, history=visit_history)
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
                                             all_signals, observations, brief, history=visit_history)
    await roles.build_visit_note(settings, bus, store, pid, current.number,
                                 [transcripts[n] for n in sorted(transcripts)],
                                 current.icd10, all_signals, history=visit_history)
    current.status = "complete"
    store.save_visits(pid, visits)
    await bus.emit("visit_complete", patient=pid, visit=current.number)
    return summary


async def run_longitudinal(settings: Settings, bus: EventBus, store: PatientStore, pid: str) -> dict:
    return await roles.longitudinal_analysis(settings, bus, store, pid)
