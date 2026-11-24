"""SQLite storage for sessions and messages.

Each operation opens a short-lived connection, which keeps the class safe to
call from worker threads. An exchange (user + assistant message) is written in
one transaction, so a failed request never leaves half a turn in the history.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    system_prompt TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    metadata    TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Database:
    def __init__(self, path: str) -> None:
        self.path = path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:  # commit on success, roll back on error
                yield conn
        finally:
            conn.close()

    def init(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    def ping(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1 FROM messages LIMIT 1")
            return True
        except sqlite3.Error:
            return False

    def get_system_prompt(self, session_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT system_prompt FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row["system_prompt"] if row else None

    def session_exists(self, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row is not None

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        """History of one session, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, metadata FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            message: dict[str, Any] = {"role": row["role"], "content": row["content"]}
            if row["metadata"]:
                message["metadata"] = json.loads(row["metadata"])
            messages.append(message)
        return messages

    def save_exchange(
        self,
        session_id: str,
        system_prompt: str,
        user_content: str,
        assistant_content: str,
        metadata: dict[str, Any],
    ) -> None:
        """Store one successful turn and create/update the session atomically."""
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, system_prompt, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    system_prompt = excluded.system_prompt,
                    updated_at = excluded.updated_at
                """,
                (session_id, system_prompt, now, now),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at)"
                " VALUES (?, 'user', ?, NULL, ?)",
                (session_id, user_content, now),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at)"
                " VALUES (?, 'assistant', ?, ?, ?)",
                (session_id, assistant_content, json.dumps(metadata), utc_now()),
            )
