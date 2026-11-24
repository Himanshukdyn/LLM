"""Parser for the X-Simulate-Failure test hook.

Header format: comma-separated <key_id>=<status> pairs, "*" meaning every key.
    X-Simulate-Failure: key_1=429,key_2=500
    X-Simulate-Failure: *=429
"""

from __future__ import annotations

import re

ALLOWED_STATUSES = {429, 500, 502, 503}
_KEY_RE = re.compile(r"^(\*|key_[1-9]\d*)$")


class InvalidSimulationHeader(ValueError):
    pass


def parse_simulation_header(value: str | None) -> dict[str, int]:
    """Return {key_id or "*": status}. Empty dict when the header is absent."""
    if value is None or not value.strip():
        return {}
    plan: dict[str, int] = {}
    for pair in value.split(","):
        key_id, sep, status = pair.strip().partition("=")
        key_id, status = key_id.strip(), status.strip()
        if not sep or not _KEY_RE.match(key_id) or not status.isdigit():
            raise InvalidSimulationHeader(
                "X-Simulate-Failure must be comma-separated <key_id>=<status> pairs, "
                "e.g. key_1=429 or *=503"
            )
        if int(status) not in ALLOWED_STATUSES:
            raise InvalidSimulationHeader(
                "X-Simulate-Failure status must be one of 429, 500, 502, 503"
            )
        plan[key_id] = int(status)
    return plan


def simulated_status(plan: dict[str, int], key_id: str) -> int | None:
    """Status this key should fail with in this request, if any."""
    return plan.get(key_id, plan.get("*"))
