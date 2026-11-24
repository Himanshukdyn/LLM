"""Runs the gateway against the real LiteLLM library (no network, no real keys).

LiteLLM's `mock_response` produces a provider-shaped answer locally, which lets
us check that the built-in cache is wired up and that cache hits are detected.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

litellm = pytest.importorskip("litellm")

from app.llm import LLMGateway, configure_litellm  # noqa: E402


def test_litellm_cache_roundtrip(settings):
    acompletion = configure_litellm(settings)
    assert litellm.cache is not None

    async def mocked(**kwargs):
        return await acompletion(mock_response="Paris is the capital of France.", **kwargs)

    gateway = LLMGateway(settings, completion_fn=mocked)
    question = f"capital of France? {uuid.uuid4()}"
    messages = [{"role": "user", "content": question}]

    async def scenario():
        first = await gateway.complete(messages)
        await asyncio.sleep(0.3)  # LiteLLM may write the cache entry in a background task
        second = await gateway.complete(messages)
        other = await gateway.complete([{"role": "user", "content": question + " again"}])
        bypass = await gateway.complete(messages, {"key_1": 429})
        return first, second, other, bypass

    first, second, other, bypass = asyncio.run(scenario())
    assert first.resolution_source == "LIVE_INFERENCE" and first.key_id == "key_1"
    assert second.resolution_source == "LLM_CACHE" and second.key_id is None
    assert second.reply == first.reply
    assert (second.input_tokens, second.output_tokens) == (first.input_tokens, first.output_tokens)
    assert other.resolution_source == "LIVE_INFERENCE"
    assert bypass.resolution_source == "LIVE_INFERENCE" and bypass.key_id == "key_2"
