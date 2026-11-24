"""Request and response models for the API contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

MAX_MESSAGE_CHARS = 4000
MAX_SESSION_ID_CHARS = 128
MAX_SYSTEM_PROMPT_CHARS = 4000


class ChatRequest(BaseModel):
    session_id: str
    message: str
    system_prompt: str | None = None

    @field_validator("session_id")
    @classmethod
    def _session_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("session_id must not be empty")
        if len(value) > MAX_SESSION_ID_CHARS:
            raise ValueError(f"session_id must be at most {MAX_SESSION_ID_CHARS} characters")
        return value

    @field_validator("message")
    @classmethod
    def _message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be empty")
        if len(value) > MAX_MESSAGE_CHARS:
            raise ValueError(f"message must be at most {MAX_MESSAGE_CHARS} characters")
        return value

    @field_validator("system_prompt")
    @classmethod
    def _system_prompt(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip()
        if len(value) > MAX_SYSTEM_PROMPT_CHARS:
            raise ValueError(
                f"system_prompt must be at most {MAX_SYSTEM_PROMPT_CHARS} characters"
            )
        return value


class Metadata(BaseModel):
    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"]
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: Metadata


class HealthResponse(BaseModel):
    status: str
    keys_configured: int
    cache_enabled: bool
