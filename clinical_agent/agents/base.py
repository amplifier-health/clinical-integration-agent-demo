import asyncio
import hashlib
from pathlib import Path

import anthropic

from clinical_agent.config import Settings
from clinical_agent.events import EventBus

MOCK_RESPONSES: dict[str, str] = {}


class ClaudeAgent:
    """Streaming Claude call with tool loop, mock mode, and rehearsal cache.

    The rehearsal cache makes stage demos resilient: every successful live
    call is recorded; if a later identical call fails (conference wifi), the
    recording plays back silently.
    """

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
        client = anthropic.AsyncAnthropic()
        tool_defs = [d for d, _ in (tools or {}).values()]
        messages: list[dict] = [{"role": "user", "content": user}]
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
                async for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        parts.append(event.delta.text)
                        await self.bus.emit("agent_token", agent=self.name, text=event.delta.text)
                final = await stream.get_final_message()

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
