"""Shared fixtures. The LLM is replaced by FakeLLM, so tests need no network or keys."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.db import Database  # noqa: E402
from app.llm import LLMGateway  # noqa: E402
from app.service import ChatService  # noqa: E402

KEYS = ("secret-key-one", "secret-key-two")
MODEL = "fake/model-1"


class ProviderError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"provider returned {status_code}")
        self.status_code = status_code


class FakeLLM:
    """Stands in for litellm.acompletion, including its response cache."""

    def __init__(self) -> None:
        self.cache: dict[str, tuple[str, int, int]] = {}
        self.failing: dict[str, Exception] = {}  # api_key -> error to raise
        self.live_calls: list[dict] = []
        self.empty_reply_keys: set[str] = set()

    async def __call__(self, *, model, messages, api_key, **kwargs):
        cache_id = json.dumps([model, messages])
        skip_read = bool((kwargs.get("cache") or {}).get("no-cache"))
        if not skip_read and cache_id in self.cache:
            return self._response(*self.cache[cache_id], cache_hit=True)
        if api_key in self.failing:
            raise self.failing[api_key]
        await asyncio.sleep(0.03)  # a live call is slower than a cache hit
        self.live_calls.append({"api_key": api_key, "messages": messages, "kwargs": kwargs})
        if api_key in self.empty_reply_keys:
            return self._response("", 1, 0, cache_hit=False)
        user_turns = [m["content"] for m in messages if m["role"] == "user"]
        reply = f"reply #{len(self.live_calls)} to: {user_turns[-1]} | seen: {' / '.join(user_turns)}"
        prompt_tokens = sum(len(m["content"].split()) + 4 for m in messages)
        entry = (reply, prompt_tokens, len(reply.split()))
        self.cache[cache_id] = entry
        return self._response(*entry, cache_hit=False)

    @staticmethod
    def _response(reply, prompt_tokens, completion_tokens, cache_hit):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            _hidden_params={"cache_hit": True} if cache_hit else {},
        )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        api_keys=KEYS,
        models=(MODEL,),
        database_path=str(tmp_path / "app.db"),
        allowed_origins=("http://localhost:3000",),
        enable_failure_simulation=True,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def gateway(settings, fake_llm) -> LLMGateway:
    return LLMGateway(settings, completion_fn=fake_llm)


@pytest.fixture
def db(settings) -> Database:
    database = Database(settings.database_path)
    database.init()
    return database


@pytest.fixture
def service(settings, db, gateway) -> ChatService:
    return ChatService(settings, db, gateway)
