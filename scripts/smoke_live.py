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
