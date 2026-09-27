"""Development guidance is evidence, not an attempt-bound Memory treatment."""

import asyncio
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.memory_service_api import create_app
from tests.test_service import _request


def _post(app, path, payload, *, key=None):
    body = json.dumps(payload).encode()
    headers = [(b"content-type", b"application/json")]
    if key is not None:
        headers.append((b"x-memory-service-key", key.encode()))
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": path,
        "raw_path": path.encode(), "query_string": b"", "headers": headers,
        "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 8768),
    }
    messages = []
    delivered = False

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)

    asyncio.run(app(scope, receive, send))
    status = next(message["status"] for message in messages
                  if message["type"] == "http.response.start")
    content = b"".join(message.get("body", b"") for message in messages
                       if message["type"] == "http.response.body")
    return status, json.loads(content)


def test_dev_grant_is_scoped_idempotent_and_revocable(tmp_path, monkeypatch):
    service = MemoryService(tmp_path / "memory.sqlite3")
    context = {"suite": "robosuite", "task_id": "cube_lift", "perception_mode": "sim_gt"}
    memory = SimpleNamespace(
        memory_id="m1", version=2, guidance="Use the gripper after alignment.",
        source_trace_id="trace:t1", evidence_refs=("raw:r1",),
    )
    monkeypatch.setattr(service, "retrieve", lambda value, now: [memory] if value == context else [])
    with sqlite3.connect(service.store.path) as db:
        db.execute(
            "INSERT INTO memories(id, payload) VALUES (?, ?)",
            ("m1", json.dumps({"status": "trusted", "version": 2, "expires_at": None})),
        )
    arguments = dict(
        memory_id="m1", development_id="graph:lift:session-1", context=context,
        source_policy_sha256="a" * 64, now=1.0,
    )
    grant = service.authorize_dev_use(**arguments)
    assert grant["schema_version"] == "attentionbench.dev-memory-exposure.v1"
    assert grant["evidence_refs"] == ["raw:r1"]
    assert service.authorize_dev_use(**arguments)["grant_id"] == grant["grant_id"]
    with pytest.raises(StateConflictError):
        service.authorize_dev_use(**{**arguments, "source_policy_sha256": "b" * 64})
    with sqlite3.connect(service.store.path) as db:
        db.execute(
            "UPDATE memories SET payload=? WHERE id=?",
            (json.dumps({"status": "disabled", "version": 3}), "m1"),
        )
    with pytest.raises(StateConflictError):
        service.authorize_dev_use(**{**arguments, "development_id": "graph:lift:session-2"})


def test_dev_grant_rejects_unmatched_scope(tmp_path, monkeypatch):
    service = MemoryService(tmp_path / "memory.sqlite3")
    monkeypatch.setattr(service, "retrieve", lambda context, now: [])
    with pytest.raises(StateConflictError):
        service.authorize_dev_use(
            memory_id="m1", development_id="graph:lift:session-1",
            context={"suite": "robosuite"}, source_policy_sha256="a" * 64,
            now=1.0,
        )



def test_dev_result_binds_real_native_summary_and_is_immutable(tmp_path, monkeypatch):
    service = MemoryService(tmp_path / "memory.sqlite3")
    context = {"suite": "robosuite", "task_id": "cube_lift", "perception_mode": "sim_gt"}
    memory = SimpleNamespace(
        memory_id="m1", version=1, guidance="Try again",
        source_trace_id="trace:t1", evidence_refs=("raw:r1",),
    )
    monkeypatch.setattr(service, "retrieve", lambda value, now: [memory])
    with sqlite3.connect(service.store.path) as db:
        db.execute("INSERT INTO memories(id,payload) VALUES (?,?)",
                   ("m1", json.dumps({"status": "trusted", "version": 1})))
    grant = service.authorize_dev_use(
        memory_id="m1", development_id="graph:lift:session-1",
        context=context, source_policy_sha256="a" * 64, now=1.0,
    )
    artifact_dir = tmp_path / "attention-robosuite-cube_lift-seed101-test"
    artifact_dir.mkdir()
    artifact = artifact_dir / "attention_run.json"
    run_id = f"run:{artifact_dir.name}"
    summary = {"schema_version": "attentionbench.sim-gt-attention-run.v1",
               "artifact_dir": str(artifact_dir.resolve()), "suite": "robosuite",
               "task_id": "cube_lift", "native_success": True}
    artifact.write_text(json.dumps(summary))
    with sqlite3.connect(service.store.path) as db:
        db.execute("INSERT INTO runs(id,payload) VALUES (?,?)",
                   (run_id, json.dumps({"status": "completed", "suite": "robosuite",
                                        "task_id": "cube_lift"})))
    args = dict(grant_id=grant["grant_id"], run_id=run_id,
                result_policy_sha256="b" * 64, artifact_uri=str(artifact),
                artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    recorded = service.record_dev_result(**args)
    assert recorded["result"]["claim"] == "development_exposure_only_not_causal_impact"
    assert service.record_dev_result(**args)["result"]["run_id"] == run_id
    assert service.get_dev_use(grant["grant_id"])["result"]["run_id"] == run_id
    app = create_app(service.store.path, api_key="memory-service-test-key")
    path = f"/dev-use-grants/{grant['grant_id']}"
    assert _request(app, path)[0] == 401
    assert _request(app, path, key="memory-service-test-key")[0] == 200

    with pytest.raises(StateConflictError):
        service.record_dev_result(**{**args, "result_policy_sha256": "c" * 64})
    with pytest.raises(StateConflictError, match="digest mismatch"):
        service.record_dev_result(**{**args, "artifact_sha256": "0" * 64})
    artifact.write_text("{}")
    with pytest.raises(StateConflictError, match="changed after recording"):
        service.get_dev_use(grant["grant_id"])
    assert _request(app, path, key="memory-service-test-key")[0] == 409



def test_dev_use_grant_http_requires_auth_and_returns_evidence(tmp_path, monkeypatch):
    store_path = tmp_path / "memory.sqlite3"
    service = MemoryService(store_path)
    context = {"suite": "robosuite", "task_id": "cube_lift", "perception_mode": "sim_gt"}
    memory = SimpleNamespace(
        memory_id="m1", version=1, guidance="Align then grasp",
        source_trace_id="trace:t1", evidence_refs=("raw:r1",),
    )
    monkeypatch.setattr(MemoryService, "retrieve",
                        lambda self, value, now: [memory] if value == context else [])
    with sqlite3.connect(store_path) as db:
        db.execute("INSERT INTO memories(id,payload) VALUES (?,?)",
                   ("m1", json.dumps({"status": "trusted", "version": 1})))
    app = create_app(store_path, api_key="memory-service-test-key")
    payload = {"memory_id": "m1", "development_id": "graph:lift:session-1",
               "context": context, "source_policy_sha256": "a" * 64, "now": 1.0}
    assert _post(app, "/dev-use-grants", payload)[0] == 401
    status, result = _post(app, "/dev-use-grants", payload,
                           key="memory-service-test-key")
    assert status == 200
    assert result["grant_id"].startswith("devgrant:")
    assert result["evidence_refs"] == ["raw:r1"]
