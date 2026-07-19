import asyncio
import json
import re
from pathlib import Path

from clinical_agent.agents import roles
from clinical_agent.amplifier import AmplifierClient
from clinical_agent.audio import chunk_file
from clinical_agent.config import Settings
from clinical_agent.clinician_config import ClinicianConfig, set_config
from clinical_agent.events import EventBus, start_session
from clinical_agent.store import PatientStore
from clinical_agent.transcribe import Transcriber


async def run_visit(settings: Settings, bus: EventBus, store: PatientStore,
                    transcriber: Transcriber, amplifier: AmplifierClient,
                    pid: str, audio_path: Path | None = None,
                    visit_number: int | None = None,
                    config: ClinicianConfig | None = None) -> dict:
    visits = store.list_visits(pid)
    if visit_number is not None:  # demo toggle: replay any appointment as if it were live
        current = next((v for v in visits if v.number == visit_number), None)
    else:
        current = next((v for v in visits if v.status == "planned"), None)
    if current is None:
        raise ValueError(f"no {'visit ' + str(visit_number) if visit_number else 'planned visit'} "
                         f"for patient {pid}")
    if audio_path is None:  # resolve the appointment's own audio from its stored artifact
        audio = store.read_artifact(pid, current.number, "audio") or {}
        pick = audio.get("postdiarized_patient") or next(iter(audio.values()), None)
        if not pick:
            raise ValueError(f"no audio on file for visit {current.number} of patient {pid}")
        audio_path = Path(pick)
    start_session(patient_id=pid, visit=current.number)
    set_config(config or ClinicianConfig())
    await bus.emit("visit_started", patient=pid, visit=current.number, date=current.date,
                   reason=current.reason)

    # Prior appointments only — the agent reasons causally from history it could actually have had.
    brief = await roles.pre_visit_brief(settings, bus, store, pid, before=current.number)

    # One conversation for the whole visit — the reasoner (per chunk) and the post-visit agent
    # share this memory, so the final analysis is done by an agent that saw the entire encounter.
    _wellness = _wellness_context(store, pid, current.number)
    visit_history: list = [{"role": "user", "content":
        f"VISIT START for patient {pid}. Chart (prior appointments only):\n"
        f"{json.dumps(store.chart(pid, before=current.number))}\n"
        f"Pre-visit brief:\n{json.dumps(brief)}\n"
        + (f"Full voice wellness & speech-prosody object for this visit (reason over ALL of it to "
           f"contextualize the condition signals; never surface raw numbers to the clinician):\n"
           f"{json.dumps(_wellness)}\n" if _wellness else "")
        + "Live voice-biomarker ticks and transcript follow as the visit is recorded."}]

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
    await bus.emit("visit_analyzing", patient=pid, visit=current.number)  # live reasoning done → phone stops recording
    # Summary and note are independent — run them in parallel, each on its own history copy.
    tp = [transcripts[n] for n in sorted(transcripts)]
    summary, _ = await asyncio.gather(
        roles.post_visit_summary(settings, bus, store, pid, current.number, tp,
                                 all_signals, observations, brief, history=list(visit_history)),
        roles.build_visit_note(settings, bus, store, pid, current.number, tp,
                               current.icd10, all_signals, history=list(visit_history)))
    current.status = "complete"
    store.save_visits(pid, visits)
    await bus.emit("visit_complete", patient=pid, visit=current.number)
    return summary


_TIER_FLAGGED = {"consider", "moderate", "elevated", "high"}
_SEG_RE = re.compile(r"\s*([\d.]+)-([\d.]+)\s+\[([^\]]+)\]:\s*(.*)")


def _wellness_context(store: PatientStore, pid: str, visit_no: int) -> dict | None:
    """The WHOLE wellness + speech/prosody object for a visit — 18 wellness dimensions (each with
    anchor words) and the speech prosody metrics. Handed to the agent verbatim so it can reason over
    the full picture and CONTEXTUALIZE the condition signals itself, not receive a pre-curated slice."""
    return store.read_artifact(pid, visit_no, "wellness") or None


def _timed_segments(transcript) -> list[tuple[float, str, str]]:
    """Parse diarized transcript turns (`"12.3-15.0 [SPEAKER_01]: words"` lines) into a single
    time-ordered list of (start_s, speaker, text) so the demo can pretend-stream them like live ASR."""
    segs: list[tuple[float, str, str]] = []
    for turn in transcript or []:
        speaker = turn.get("speaker", "?")
        for line in (turn.get("text") or "").splitlines():
            m = _SEG_RE.match(line)
            if m:
                segs.append((float(m.group(1)), speaker, m.group(4).strip()))
    segs.sort(key=lambda s: s[0])
    return segs


async def replay_visit(settings: Settings, bus: EventBus, store: PatientStore,
                       pid: str, visit_number: int | None = None,
                       config: ClinicianConfig | None = None) -> dict:
    """Replay a visit from its precomputed per-chunk aria results — no audio, no Whisper, no
    chunker. This is the 'mock the API, speed through the appointment' demo path: the signals
    are the real precomputed ones, the agent reasoning is live."""
    visits = store.list_visits(pid)
    if visit_number is not None:
        current = next((v for v in visits if v.number == visit_number), None)
    else:
        current = next((v for v in visits if v.status == "planned"), None)
    if current is None:
        raise ValueError(f"no visit {visit_number or '(planned)'} for patient {pid}")
    chunks = store.read_artifact(pid, current.number, "chunks") or []
    if not chunks:
        raise ValueError(f"no precomputed chunks for visit {current.number}; run the real "
                         f"pipeline for this visit or import its aria results")

    start_session(patient_id=pid, visit=current.number)
    set_config(config or ClinicianConfig())
    await bus.emit("visit_started", patient=pid, visit=current.number, date=current.date,
                   reason=current.reason)
    brief = await roles.pre_visit_brief(settings, bus, store, pid, before=current.number)
    _wellness = _wellness_context(store, pid, current.number)
    visit_history: list = [{"role": "user", "content":
        f"VISIT START for patient {pid}. Chart (prior appointments only):\n"
        f"{json.dumps(store.chart(pid, before=current.number))}\n"
        f"Pre-visit brief:\n{json.dumps(brief)}\n"
        + (f"Full voice wellness & speech-prosody object for this visit (reason over ALL of it to "
           f"contextualize the condition signals; never surface raw numbers to the clinician):\n"
           f"{json.dumps(_wellness)}\n" if _wellness else "")
        + "Precomputed voice-biomarker ticks follow as the visit is replayed."}]

    # Pretend-stream the transcript we already have, time-aligned — standing in for the live ASR
    # we'd run on the scribe's audio stream in the real scenario.
    transcript = store.read_artifact(pid, current.number, "transcript")
    segments = _timed_segments(transcript)
    seg_ptr = 0
    heard: list = []

    pace = 15.0 / max(settings.speed, 1.0)  # ~15s of audio per hop, compressed by SPEED
    signals_by_chunk: dict = {}
    observations: list = []
    for ch in sorted(chunks, key=lambda c: c["chunk"]):
        idx = ch["chunk"]
        win_end = (ch.get("start", 0.0) or 0.0) + 30.0
        await bus.emit("chunk_created", patient=pid, chunk=idx,
                       start_s=ch.get("start", 0.0), end_s=win_end)
        # reveal the transcript that "arrived" during this window
        new_parts = []
        while seg_ptr < len(segments) and segments[seg_ptr][0] < win_end:
            _, spk, txt = segments[seg_ptr]
            if txt:
                new_parts.append(f"{spk}: {txt}")
            seg_ptr += 1
        new_text = " ".join(new_parts)
        if new_text:
            heard.append(new_text)
            await bus.emit("transcript", patient=pid, chunk=idx, text=new_text)
        sigs = [{"name": n, "score": s.get("score"), "level": s.get("level"),
                 "flagged": s.get("level") in _TIER_FLAGGED}
                for n, s in (ch.get("signals") or {}).items()]
        await bus.emit("api_job_created", patient=pid, chunk=idx, model="aria",
                       job_id=f"replay-{current.number}-{idx}")
        await bus.emit("api_job_result", patient=pid, chunk=idx, cached=True, signals=sigs,
                       summary={"overall_level": ch.get("overall_level")})
        signals_by_chunk[idx] = sigs
        cumulative = [s for n in sorted(signals_by_chunk) for s in signals_by_chunk[n]]
        try:
            obs = await roles.reason_over_chunk(settings, bus, store, pid, idx, new_text, cumulative,
                                                brief, history=visit_history)
            observations.append(obs)
        except Exception as exc:  # keep the replay alive if a single reasoning tick fails
            await bus.emit("error", patient=pid, chunk=idx, message=str(exc))
        await asyncio.sleep(pace)

    ordered = sorted(signals_by_chunk)
    all_signals = [s for n in ordered for s in signals_by_chunk[n]]
    store.write_artifact(pid, current.number, "signals", signals_by_chunk[ordered[-1]] if ordered else [])
    store.write_artifact(pid, current.number, "observations", observations)

    await bus.emit("visit_analyzing", patient=pid, visit=current.number)  # live reasoning done → phone stops recording
    # Summary and note are independent — run them in parallel, each on its own history copy.
    summary, _ = await asyncio.gather(
        roles.post_visit_summary(settings, bus, store, pid, current.number,
                                 heard or [roles._transcript_text(transcript)], all_signals,
                                 observations, brief, history=list(visit_history)),
        roles.build_visit_note(settings, bus, store, pid, current.number,
                               transcript, current.icd10, all_signals, history=list(visit_history)))
    if current.status != "complete":
        current.status = "complete"
        store.save_visits(pid, visits)
    await bus.emit("visit_complete", patient=pid, visit=current.number)
    return summary


async def run_streaming_visit(settings: Settings, bus: EventBus, store: PatientStore,
                              transcriber: Transcriber, amplifier: AmplifierClient, pid: str, ws,
                              visit_number: int | None = None,
                              config: ClinicianConfig | None = None) -> dict:
    """The REAL ingestion path: the scribe streams PCM frames over a WebSocket; we bucket them
    into 30s/15s windows and run the exact same pipeline as a file visit. (The demo mocks this
    with replay_visit — this is the production transport.)"""
    from fastapi import WebSocketDisconnect

    from clinical_agent.streaming import AudioBucketer

    visits = store.list_visits(pid)
    current = (next((v for v in visits if v.number == visit_number), None) if visit_number is not None
               else next((v for v in visits if v.status == "planned"), None))
    if current is None:
        raise ValueError(f"no visit {visit_number or '(planned)'} for patient {pid}")

    start_session(patient_id=pid, visit=current.number)
    set_config(config or ClinicianConfig())
    await bus.emit("visit_started", patient=pid, visit=current.number, date=current.date,
                   reason=current.reason)
    brief = await roles.pre_visit_brief(settings, bus, store, pid, before=current.number)
    visit_history: list = [{"role": "user", "content":
        f"VISIT START for patient {pid}. Chart (prior appointments only):\n"
        f"{json.dumps(store.chart(pid, before=current.number))}\n"
        f"Pre-visit brief:\n{json.dumps(brief)}\n"
        "Live voice-biomarker ticks and transcript follow as the scribe streams audio."}]

    bucketer = AudioBucketer()
    transcripts: dict = {}
    signals_by_chunk: dict = {}
    observations: list = []

    async def process(chunk):
        try:
            text = await transcriber.transcribe(chunk.wav_bytes)
            await bus.emit("transcript", patient=pid, chunk=chunk.index, text=text)
            result = await amplifier.analyze(chunk)
            transcripts[chunk.index] = text
            signals_by_chunk[chunk.index] = result.get("signals", [])
            await bus.emit("chunk_created", patient=pid, chunk=chunk.index,
                           start_s=chunk.start_s, end_s=chunk.end_s)
            cumulative = [s for n in sorted(signals_by_chunk) for s in signals_by_chunk[n]]
            obs = await roles.reason_over_chunk(settings, bus, store, pid, chunk.index, text,
                                                cumulative, brief, history=visit_history)
            observations.append(obs)
        except Exception as exc:  # a bad window must not kill the stream
            await bus.emit("error", patient=pid, chunk=chunk.index, message=str(exc))

    while True:  # scribe → us: {"type":"visit.start", sample_rate?} · binary PCM frames · {"type":"visit.end"}
        try:
            msg = await ws.receive()
        except WebSocketDisconnect:
            break
        if msg.get("type") == "websocket.disconnect":
            break
        if msg.get("bytes") is not None:
            bucketer.feed(msg["bytes"])
            for chunk in bucketer.pop_ready():
                await process(chunk)
        elif msg.get("text") is not None:
            ctrl = json.loads(msg["text"])
            if ctrl.get("type") == "visit.start" and ctrl.get("sample_rate"):
                bucketer = AudioBucketer(sample_rate=int(ctrl["sample_rate"]))
            elif ctrl.get("type") == "visit.end":
                break

    tail = bucketer.flush()
    if tail is not None:
        await process(tail)

    ordered = sorted(signals_by_chunk)
    all_signals = [s for n in ordered for s in signals_by_chunk[n]]
    store.write_artifact(pid, current.number, "signals",
                         signals_by_chunk[ordered[-1]] if ordered else [])
    store.write_artifact(pid, current.number, "transcript",
                         [{"chunk": n, "text": transcripts[n]} for n in ordered])
    store.write_artifact(pid, current.number, "observations", observations)
    await bus.emit("visit_analyzing", patient=pid, visit=current.number)  # live reasoning done → phone stops recording
    # Summary and note are independent — run them in parallel, each on its own history copy.
    tp = [transcripts[n] for n in ordered]
    summary, _ = await asyncio.gather(
        roles.post_visit_summary(settings, bus, store, pid, current.number, tp, all_signals,
                                 observations, brief, history=list(visit_history)),
        roles.build_visit_note(settings, bus, store, pid, current.number, tp, current.icd10,
                               all_signals, history=list(visit_history)))
    current.status = "complete"
    store.save_visits(pid, visits)
    await bus.emit("visit_complete", patient=pid, visit=current.number)
    return summary


async def run_longitudinal(settings: Settings, bus: EventBus, store: PatientStore, pid: str) -> dict:
    start_session(patient_id=pid)  # scope the longitudinal events to a session too
    return await roles.longitudinal_analysis(settings, bus, store, pid)
