from __future__ import annotations

import asyncio
import json

import pytest

from attention_memory_service import MemoryService
from attention_memory_service.identity import store_id
from attention_memory_service.memory_service_api import create_app


def _request(app, path: str, *, key: str | None = None):
    headers = [] if key is None else [(b"x-memory-service-key", key.encode())]
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "GET", "scheme": "http", "path": path,
        "raw_path": path.encode(), "query_string": b"", "headers": headers,
        "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 8768),
    }
    messages = []
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    asyncio.run(app(scope, receive, send))
    status = next(item["status"] for item in messages if item["type"] == "http.response.start")
    content = b"".join(item.get("body", b"") for item in messages if item["type"] == "http.response.body")
    return status, json.loads(content)


def test_standalone_service_starts_with_empty_store_and_auth(tmp_path):
    store_path = tmp_path / "attention_memory.sqlite3"
    service = MemoryService(store_path)
    assert service.retrieve(
        {"perception_mode": "sim_gt", "suite": "robocasa", "task_id": "counter_to_sink"},
        now=1.0,
    ) == []
    app = create_app(store_path, api_key="memory-service-test-key")
    assert _request(app, "/health")[0] == 200
    assert _request(app, "/memories/unknown")[0] == 401
    assert _request(app, "/memories/unknown", key="memory-service-test-key")[0] == 409


def test_service_refuses_weak_key(tmp_path):
    with pytest.raises(ValueError, match="at least 16"):
        create_app(tmp_path / "memory.sqlite3", api_key="short")
    with pytest.raises(ValueError, match="must differ"):
        create_app(
            tmp_path / "memory.sqlite3", api_key="memory-service-test-key",
            operator_key="memory-service-test-key",
        )


def test_store_identity_is_stable_and_distinguishes_databases(tmp_path):
    first = tmp_path / "first.sqlite3"
    second = tmp_path / "second.sqlite3"
    assert store_id(first) == store_id(first)
    assert store_id(first) != store_id(second)
