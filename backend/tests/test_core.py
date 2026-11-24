"""Framework-free tests: config, header parser, gateway failover, chat service."""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from app.config import Settings, sqlite_path_from_url
from app.db import Database
from app.llm import AllProvidersUnavailable, cache_key
from app.simulation import InvalidSimulationHeader, parse_simulation_header

from .conftest import KEYS, MODEL, ProviderError

run = asyncio.run
MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello there"}]


# --- config -----------------------------------------------------------------

def test_sqlite_url_parsing():
    assert sqlite_path_from_url("sqlite:////data/app.db") == "/data/app.db"
    assert sqlite_path_from_url("sqlite:///./app.db") == "./app.db"
    with pytest.raises(ValueError):
        sqlite_path_from_url("postgresql://host/db")


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEYS", " a , b ,, c ")
    monkeypatch.setenv("LLM_MODEL", "m1,m2")
    monkeypatch.setenv("DATABASE_URL", "sqlite:////data/app.db")
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.app, https://b.app")
    monkeypatch.setenv("ENABLE_FAILURE_SIMULATION", "true")
    s = Settings.from_env()
    assert s.api_keys == ("a", "b", "c")
    assert [s.model_for(i) for i in range(3)] == ["m1", "m2", "m2"]
    assert s.database_path == "/data/app.db"
    assert s.allowed_origins == ("https://a.app", "https://b.app")
    assert s.enable_failure_simulation is True


# --- X-Simulate-Failure parser ----------------------------------------------

def test_parse_simulation_header():
    assert parse_simulation_header(None) == {}
    assert parse_simulation_header("key_1=429") == {"key_1": 429}
    assert parse_simulation_header("key_1=429, key_2=500") == {"key_1": 429, "key_2": 500}
    assert parse_simulation_header("*=503") == {"*": 503}


@pytest.mark.parametrize("bad", ["key_1", "key_1=404", "foo=429", "key_1=abc", "key_0=429", "=429"])
def test_parse_simulation_header_rejects_bad_values(bad):
    with pytest.raises(InvalidSimulationHeader):
        parse_simulation_header(bad)


# --- gateway ----------------------------------------------------------------

def test_cache_key_depends_on_model_and_messages_only():
    base = cache_key(model="m", messages=MSGS, api_key="k1")
    assert base == cache_key(model="m", messages=MSGS, api_key="k2", timeout=5)
    assert base != cache_key(model="other", messages=MSGS)
    assert base != cache_key(model="m", messages=MSGS + [{"role": "user", "content": "x"}])


def test_live_call_uses_first_key(gateway, fake_llm):
    result = run(gateway.complete(MSGS))
    assert result.resolution_source == "LIVE_INFERENCE"
    assert (result.key_id, result.fallback_attempts, result.model) == ("key_1", 0, MODEL)
    assert result.input_tokens > 0 and result.output_tokens > 0
    assert fake_llm.live_calls[0]["api_key"] == KEYS[0]
    assert fake_llm.live_calls[0]["kwargs"]["num_retries"] == 0


def test_second_identical_call_is_cache_hit(gateway, fake_llm):
    first = run(gateway.complete(MSGS))
    second = run(gateway.complete(MSGS))
    assert second.resolution_source == "LLM_CACHE"
    assert second.reply == first.reply
    assert (second.key_id, second.fallback_attempts) == (None, 0)
    assert (second.input_tokens, second.output_tokens) == (first.input_tokens, first.output_tokens)
    assert len(fake_llm.live_calls) == 1


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_real_provider_error_fails_over(gateway, fake_llm, status):
    fake_llm.failing[KEYS[0]] = ProviderError(status)
    result = run(gateway.complete(MSGS))
    assert (result.key_id, result.fallback_attempts) == ("key_2", 1)
    assert result.resolution_source == "LIVE_INFERENCE"


def test_empty_reply_counts_as_key_failure(gateway, fake_llm):
    fake_llm.empty_reply_keys.add(KEYS[0])
    result = run(gateway.complete(MSGS))
    assert result.key_id == "key_2" and result.reply


def test_all_keys_failing_raises(gateway, fake_llm):
    fake_llm.failing = {k: ProviderError(429) for k in KEYS}
    with pytest.raises(AllProvidersUnavailable):
        run(gateway.complete(MSGS))


@pytest.mark.parametrize("status", [429, 500])
def test_simulated_failure_of_key_1(gateway, fake_llm, status):
    result = run(gateway.complete(MSGS, {"key_1": status}))
    assert (result.key_id, result.fallback_attempts) == ("key_2", 1)
    assert fake_llm.live_calls[0]["api_key"] == KEYS[1]


def test_simulation_skips_cache_read_and_applies_to_one_request_only(gateway):
    run(gateway.complete(MSGS))  # warm the cache
    simulated = run(gateway.complete(MSGS, {"key_1": 429}))
    assert (simulated.resolution_source, simulated.key_id) == ("LIVE_INFERENCE", "key_2")
    with pytest.raises(AllProvidersUnavailable):
        run(gateway.complete(MSGS, {"*": 429}))
    after = run(gateway.complete([MSGS[0], {"role": "user", "content": "fresh question"}]))
    assert (after.key_id, after.fallback_attempts) == ("key_1", 0)


# --- chat service + SQLite --------------------------------------------------

def test_schema_is_created_on_startup(db, settings):
    with sqlite3.connect(settings.database_path) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(messages)")]
    assert columns == ["id", "session_id", "role", "content", "metadata", "created_at"]
    assert db.ping()


def test_multi_turn_memory_and_history(service, fake_llm):
    run(service.chat("s-1", "my name is Asha"))
    second = run(service.chat("s-1", "what is my name?"))
    assert "my name is Asha" in second["reply"]  # earlier turn reached the LLM
    sent = fake_llm.live_calls[-1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]

    history = run(service.history("s-1"))
    assert [m["role"] for m in history] == ["user", "assistant", "user", "assistant"]
    assert "metadata" not in history[0]
    assert history[1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert all(m["role"] != "system" for m in history)


def test_sessions_are_isolated(service):
    run(service.chat("s-a", "apples"))
    reply = run(service.chat("s-b", "bananas"))["reply"]
    assert "apples" not in reply
    assert len(run(service.history("s-b"))) == 2


def test_same_first_message_in_new_session_is_cached_and_faster(service):
    live = run(service.chat("s-1", "What is the capital of France?"))
    cached = run(service.chat("s-2", "What is the capital of France?"))
    assert live["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert cached["metadata"]["resolution_source"] == "LLM_CACHE"
    assert cached["reply"] == live["reply"]
    assert cached["metadata"]["key_id"] is None
    assert cached["metadata"]["fallback_attempts"] == 0
    assert cached["metadata"]["input_tokens"] == live["metadata"]["input_tokens"]
    assert cached["metadata"]["latency_ms"] < live["metadata"]["latency_ms"]
    different = run(service.chat("s-3", "What is the capital of Spain?"))
    assert different["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_tokens_are_per_request_not_running_totals(service):
    a = run(service.chat("s-1", "one two three"))["metadata"]["input_tokens"]
    b = run(service.chat("s-2", "four five six"))["metadata"]["input_tokens"]
    assert a == b
    third = run(service.chat("s-1", "seven eight nine"))["metadata"]
    assert isinstance(third["input_tokens"], int) and isinstance(third["latency_ms"], int)


def test_system_prompt_override_is_kept_for_the_session(service, fake_llm):
    run(service.chat("s-1", "hi", system_prompt="Answer like a pirate."))
    run(service.chat("s-1", "again"))
    assert fake_llm.live_calls[-1]["messages"][0] == {
        "role": "system", "content": "Answer like a pirate."
    }
    run(service.chat("s-2", "hi"))
    assert fake_llm.live_calls[-1]["messages"][0]["content"] == service.settings.default_system_prompt


def test_failed_request_is_not_stored(service):
    with pytest.raises(AllProvidersUnavailable):
        run(service.chat("s-new", "hello", simulation={"*": 429}))
    assert run(service.history("s-new")) is None  # session was never created

    run(service.chat("s-old", "first"))
    with pytest.raises(AllProvidersUnavailable):
        run(service.chat("s-old", "second", simulation={"*": 503}))
    assert len(run(service.history("s-old"))) == 2


def test_history_survives_a_restart(service, settings):
    run(service.chat("s-1", "remember me"))
    reopened = Database(settings.database_path)
    assert reopened.get_messages("s-1")[0]["content"] == "remember me"
