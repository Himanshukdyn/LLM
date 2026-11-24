"""HTTP contract tests (sections 3 and 4 of the assignment)."""

from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.llm import LLMGateway  # noqa: E402
from app.main import create_app  # noqa: E402

from .conftest import KEYS  # noqa: E402

META_FIELDS = {
    "resolution_source", "model", "latency_ms", "input_tokens",
    "output_tokens", "key_id", "fallback_attempts",
}


@pytest.fixture
def client(settings, gateway):
    with TestClient(create_app(settings, gateway), raise_server_exceptions=False) as c:
        yield c


def chat(client, session_id, message, headers=None, **extra):
    return client.post(
        "/chat", json={"session_id": session_id, "message": message, **extra}, headers=headers
    )


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "keys_configured": 2, "cache_enabled": True}


def test_chat_response_shape(client):
    response = chat(client, "s-7f3a", "What is the capital of France?")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "s-7f3a" and body["reply"]
    meta = body["metadata"]
    assert META_FIELDS <= set(meta)
    assert meta["resolution_source"] == "LIVE_INFERENCE"
    assert meta["key_id"] == "key_1" and meta["fallback_attempts"] == 0
    for field in ("latency_ms", "input_tokens", "output_tokens"):
        assert isinstance(meta[field], int)
    assert not any(key in response.text for key in KEYS)  # the key itself never leaks


def test_cache_hit_in_new_session(client):
    live = chat(client, "s-1", "Same question").json()
    cached = chat(client, "s-2", "Same question").json()
    assert cached["metadata"]["resolution_source"] == "LLM_CACHE"
    assert cached["reply"] == live["reply"]
    assert cached["metadata"]["key_id"] is None
    assert cached["metadata"]["fallback_attempts"] == 0
    assert chat(client, "s-3", "Other question").json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_history_endpoint(client):
    chat(client, "s-1", "  first  ")
    chat(client, "s-1", "second")
    response = client.get("/sessions/s-1/messages")
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "s-1"
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user", "assistant"]
    assert body["messages"][0] == {"role": "user", "content": "first"}
    assert body["messages"][1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_unknown_session_is_404(client):
    response = client.get("/sessions/nope/messages")
    assert response.status_code == 404
    assert response.json() == {"detail": "session not found"}


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "s-1"},
        {"message": "hi"},
        {"session_id": "s-1", "message": ""},
        {"session_id": "s-1", "message": "   "},
        {"session_id": "", "message": "hi"},
        {"session_id": "x" * 129, "message": "hi"},
        {"session_id": "s-1", "message": "x" * 4001},
        {"session_id": 5, "message": "hi"},
    ],
)
def test_validation_errors_are_422_and_not_stored(client, payload):
    response = client.post("/chat", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()
    assert client.get("/sessions/s-1/messages").status_code == 404


def test_message_length_is_checked_after_trimming(client):
    assert chat(client, "s-1", " " * 50 + "x" * 4000 + " " * 50).status_code == 200


def test_malformed_json_is_422(client):
    response = client.post(
        "/chat", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.parametrize("header", ["key_1=429", "key_1=500"])
def test_simulated_key_1_failure_falls_over_to_key_2(client, header):
    response = chat(client, "s-1", "hello", headers={"X-Simulate-Failure": header})
    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["key_id"] == "key_2"
    assert body["metadata"]["fallback_attempts"] == 1
    assert body["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert body["reply"].startswith("reply #")  # a real answer, not an error in a 200


def test_all_keys_failing_is_503_and_not_stored(client):
    response = chat(client, "s-1", "hello", headers={"X-Simulate-Failure": "*=429"})
    assert response.status_code == 503
    assert response.json() == {"detail": "all providers unavailable"}
    assert client.get("/sessions/s-1/messages").status_code == 404

    response = chat(client, "s-2", "hello", headers={"X-Simulate-Failure": "key_1=429,key_2=500"})
    assert response.status_code == 503


def test_simulation_affects_one_request_only(client):
    chat(client, "s-1", "one", headers={"X-Simulate-Failure": "key_1=429"})
    assert chat(client, "s-1", "two").json()["metadata"]["key_id"] == "key_1"


def test_invalid_simulation_header_is_422(client):
    response = chat(client, "s-1", "hello", headers={"X-Simulate-Failure": "key_1=418"})
    assert response.status_code == 422 and "detail" in response.json()


def test_simulation_header_ignored_when_switch_is_off(settings, fake_llm):
    off = replace(settings, enable_failure_simulation=False)
    with TestClient(create_app(off, LLMGateway(off, completion_fn=fake_llm))) as c:
        response = chat(c, "s-1", "hello", headers={"X-Simulate-Failure": "*=429"})
    assert response.status_code == 200
    assert response.json()["metadata"]["key_id"] == "key_1"


def test_health_is_503_without_keys(settings, fake_llm):
    empty = replace(settings, api_keys=())
    with TestClient(create_app(empty, LLMGateway(empty, completion_fn=fake_llm))) as c:
        assert c.get("/health").status_code == 503
        response = chat(c, "s-1", "hello")
    assert response.status_code == 503
    assert response.json() == {"detail": "all providers unavailable"}


def test_unexpected_errors_are_json_500(settings, fake_llm, monkeypatch):
    gateway = LLMGateway(settings, completion_fn=fake_llm)
    with TestClient(create_app(settings, gateway), raise_server_exceptions=False) as c:
        monkeypatch.setattr(c.app.state.db, "save_exchange", lambda *a, **k: 1 / 0)
        response = chat(c, "s-1", "hello")
    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}
    assert "Traceback" not in response.text


def test_cors_allows_the_frontend_origin(client):
    response = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-simulate-failure",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
