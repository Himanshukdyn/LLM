"""Chat orchestration: history from SQLite -> LLM gateway -> store the turn."""

from __future__ import annotations

import asyncio
import time
import weakref
from typing import Any

from .config import Settings
from .db import Database
from .llm import LLMGateway


class ChatService:
    def __init__(self, settings: Settings, db: Database, gateway: LLMGateway) -> None:
        self.settings = settings
        self.db = db
        self.gateway = gateway
        # One lock per session keeps turns of the same chat in order.
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def _lock_for(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[session_id] = lock
        return lock

    async def chat(
        self,
        session_id: str,
        message: str,
        system_prompt: str | None = None,
        simulation: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Run one turn. Raises AllProvidersUnavailable; nothing is stored then."""
        started = time.perf_counter()
        async with self._lock_for(session_id):
            stored_prompt = await asyncio.to_thread(self.db.get_system_prompt, session_id)
            prompt = system_prompt or stored_prompt or self.settings.default_system_prompt
            history = await asyncio.to_thread(self.db.get_messages, session_id)
            history = history[-self.settings.max_history_messages:]

            llm_messages = [{"role": "system", "content": prompt}]
            llm_messages += [{"role": m["role"], "content": m["content"]} for m in history]
            llm_messages.append({"role": "user", "content": message})

            result = await self.gateway.complete(llm_messages, simulation)

            metadata = {
                "resolution_source": result.resolution_source,
                "model": result.model,
                "latency_ms": max(1, round((time.perf_counter() - started) * 1000)),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "key_id": result.key_id,
                "fallback_attempts": result.fallback_attempts,
            }
            await asyncio.to_thread(
                self.db.save_exchange, session_id, prompt, message, result.reply, metadata
            )
        return {"session_id": session_id, "reply": result.reply, "metadata": metadata}

    async def history(self, session_id: str) -> list[dict[str, Any]] | None:
        """Stored messages, oldest first, or None for an unknown session."""
        if not await asyncio.to_thread(self.db.session_exists, session_id):
            return None
        return await asyncio.to_thread(self.db.get_messages, session_id)
