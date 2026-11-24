"""LiteLLM gateway: key pool, ordered failover and response caching.

Every call goes through `litellm.acompletion`, which gives us one interface for
any provider, typed provider errors (429 / quota / 5xx) and the built-in
response cache. The gateway walks the key pool in order (key_1, key_2, ...) and
moves to the next key whenever one fails.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .config import Settings
from .simulation import simulated_status

logger = logging.getLogger("chatbot.llm")

LIVE_INFERENCE = "LIVE_INFERENCE"
LLM_CACHE = "LLM_CACHE"

CompletionFn = Callable[..., Awaitable[Any]]


class AllProvidersUnavailable(Exception):
    """Every key in the pool failed for this request."""


class SimulatedProviderError(Exception):
    """Raised in place of a provider call by the X-Simulate-Failure hook."""

    def __init__(self, key_id: str, status_code: int) -> None:
        super().__init__(f"simulated HTTP {status_code} for {key_id}")
        self.status_code = status_code


class EmptyReplyError(Exception):
    """The provider answered 200 but with no text; treated as a key failure."""


@dataclass(frozen=True)
class LLMResult:
    reply: str
    resolution_source: str
    model: str
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int


def cache_key(*_args: Any, **kwargs: Any) -> str:
    """Cache key = model + the full message list sent to the LLM.

    The API key is deliberately not part of it, so a reply produced by key_2
    after a failover is still a hit when key_1 is asked the same thing later.
    """
    payload = json.dumps(
        {"model": kwargs.get("model"), "messages": kwargs.get("messages")},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def configure_litellm(settings: Settings) -> CompletionFn:
    """Turn on LiteLLM's built-in cache and return its async completion function."""
    import litellm

    try:
        from litellm.caching.caching import Cache
    except ImportError:  # older LiteLLM releases
        from litellm.caching import Cache

    litellm.drop_params = True  # ignore optional params a provider does not support
    litellm.suppress_debug_info = True
    litellm.telemetry = False

    if settings.cache_enabled:
        if settings.cache_type == "disk":
            cache = Cache(
                type="disk", disk_cache_dir=settings.cache_dir, ttl=settings.cache_ttl_seconds
            )
        else:
            cache = Cache(type="local", ttl=settings.cache_ttl_seconds)
        cache.get_cache_key = cache_key
        litellm.cache = cache
    else:
        litellm.cache = None
    return litellm.acompletion


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class LLMGateway:
    def __init__(self, settings: Settings, completion_fn: CompletionFn | None = None) -> None:
        self.settings = settings
        self._completion = completion_fn or configure_litellm(settings)

    @property
    def keys_configured(self) -> int:
        return len(self.settings.api_keys)

    async def complete(
        self, messages: list[dict[str, str]], simulation: dict[str, int] | None = None
    ) -> LLMResult:
        """Answer from cache if possible, otherwise from the first key that works."""
        simulation = simulation or {}
        failed = 0
        for index, api_key in enumerate(self.settings.api_keys):
            key_id = f"key_{index + 1}"
            model = self.settings.model_for(index)
            try:
                status = simulated_status(simulation, key_id)
                if status is not None:
                    raise SimulatedProviderError(key_id, status)
                response = await self._call(model, api_key, messages, bool(simulation))
                reply = self._reply_text(response)
            except Exception as exc:  # any failure of one key must not break the request
                failed += 1
                # Log type and status only: provider messages can echo credentials.
                logger.warning(
                    "%s failed (%s, status=%s); trying next key",
                    key_id,
                    type(exc).__name__,
                    getattr(exc, "status_code", None),
                )
                continue

            usage = _get(response, "usage")
            cache_hit = bool(_get(_get(response, "_hidden_params") or {}, "cache_hit"))
            return LLMResult(
                reply=reply,
                resolution_source=LLM_CACHE if cache_hit else LIVE_INFERENCE,
                model=model,
                input_tokens=int(_get(usage, "prompt_tokens", 0) or 0),
                output_tokens=int(_get(usage, "completion_tokens", 0) or 0),
                key_id=None if cache_hit else key_id,
                fallback_attempts=0 if cache_hit else failed,
            )
        raise AllProvidersUnavailable()

    async def _call(
        self, model: str, api_key: str, messages: list[dict[str, str]], skip_cache_read: bool
    ) -> Any:
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "api_key": api_key,
            "timeout": self.settings.llm_timeout_seconds,
            "num_retries": 0,  # failover to the next key is the retry strategy
            "caching": self.settings.cache_enabled,
        }
        if self.settings.max_output_tokens:
            params["max_tokens"] = self.settings.max_output_tokens
        if skip_cache_read:
            # A request that simulates failures must exercise the key pool, so it
            # skips the cache lookup. Its answer is still written to the cache.
            params["cache"] = {"no-cache": True}
        return await self._completion(**params)

    @staticmethod
    def _reply_text(response: Any) -> str:
        choices = _get(response, "choices") or []
        content = _get(_get(choices[0], "message"), "content") if choices else None
        if not isinstance(content, str) or not content.strip():
            raise EmptyReplyError("provider returned an empty reply")
        return content
