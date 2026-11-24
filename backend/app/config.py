"""All configuration comes from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL = "gemini/gemini-3.5-flash"
DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant."
SQLITE_PREFIX = "sqlite:///"


def _csv(value: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in (value or "").split(",") if part.strip())


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def sqlite_path_from_url(url: str) -> str:
    """sqlite:////data/app.db -> /data/app.db ; sqlite:///./app.db -> ./app.db"""
    if not url.startswith(SQLITE_PREFIX):
        raise ValueError("DATABASE_URL must be a SQLite URL, e.g. sqlite:////data/app.db")
    path = url[len(SQLITE_PREFIX):]
    if not path or path == ":memory:":
        raise ValueError("DATABASE_URL must point to a file, e.g. sqlite:////data/app.db")
    return path


@dataclass(frozen=True)
class Settings:
    api_keys: tuple[str, ...] = ()
    models: tuple[str, ...] = (DEFAULT_MODEL,)
    database_path: str = "./data/app.db"
    allowed_origins: tuple[str, ...] = ("http://localhost:3000",)
    enable_failure_simulation: bool = False
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    cache_enabled: bool = True
    cache_type: str = "local"  # "local" (in-memory) or "disk"
    cache_dir: str = "./data/llm_cache"
    cache_ttl_seconds: int = 3600
    llm_timeout_seconds: float = 30.0
    max_history_messages: int = 40
    max_output_tokens: int | None = None

    def model_for(self, key_index: int) -> str:
        """Model used with the key at this index; the last model repeats."""
        return self.models[min(key_index, len(self.models) - 1)]

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        max_out = (env("LLM_MAX_OUTPUT_TOKENS") or "").strip()
        return cls(
            api_keys=_csv(env("LLM_API_KEYS")),
            models=_csv(env("LLM_MODEL")) or (DEFAULT_MODEL,),
            database_path=sqlite_path_from_url(env("DATABASE_URL") or "sqlite:///./data/app.db"),
            allowed_origins=_csv(env("ALLOWED_ORIGINS")) or ("http://localhost:3000",),
            enable_failure_simulation=_bool(env("ENABLE_FAILURE_SIMULATION")),
            default_system_prompt=(env("DEFAULT_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT).strip(),
            cache_enabled=_bool(env("CACHE_ENABLED"), default=True),
            cache_type=(env("CACHE_TYPE") or "local").strip().lower(),
            cache_dir=env("CACHE_DIR") or "./data/llm_cache",
            cache_ttl_seconds=int(env("CACHE_TTL_SECONDS") or 3600),
            llm_timeout_seconds=float(env("LLM_TIMEOUT_SECONDS") or 30),
            max_history_messages=int(env("MAX_HISTORY_MESSAGES") or 40),
            max_output_tokens=int(max_out) if max_out else None,
        )
