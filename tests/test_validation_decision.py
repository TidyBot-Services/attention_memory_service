"""Validation verdicts (promote or reject with reasons) and the operator lifecycle."""

from __future__ import annotations

import pytest

from attention_memory_service.core.memory import MemoryManager
from attention_memory_service.core.models import MemoryRecord, MemoryStatus, MemoryUseRecord
from attention_memory_service.core.store import AttentionStore, StateConflictError
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.memory_v2 import MemoryV2Manager
from tests.test_store import populated_store


def candidate(memory_id="memory-1") -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id, version=1, source_trace_id="trace-1",
        guidance="lower the grasp height", candidate_repair="reduce lateral offset",
        applicability={"suite": "robosuite", "task_id": "cube_lift", "perception_mode": "sim_gt"},
        evidence_refs=("artifact://run-1/frame-17",), created_at=10.0)


def report(**changes):
    value = {
        "paired_dev_seeds": 5, "control_successes": 0, "treatment_successes": 5,
        "empirical_confidence": 1.0, "success_gain": 5, "success_rate_gain": 1.0,
        "both_success_assistance_credits_saved": 0, "validation_policy_hashes": ["policy"],
        "safety_regressions": [], "control_unsafe_attempts": 0, "treatment_unsafe_attempts": 0,
        "validated_scope": [{"scene_id": "a"}], "counterexamples": [],
    }
    value.update(changes)
    return value


@pytest.fixture
def manager(tmp_path, monkeypatch):
    store, _ = populated_store(tmp_path / "memory.sqlite3")
    MemoryManager(store).add_candidate(candidate())
    v2 = MemoryV2Manager(store)
    monkeypatch.setattr(v2, "verify_source_evidence", lambda memory_id: None)
    monkeypatch.setattr(v2, "_plan_complete", lambda memory_id: {"seeds": [1, 2, 3, 4, 5]})
    return v2


def with_report(manager, monkeypatch, **changes):
    monkeypatch.setattr(manager, "impact_report", lambda memory_id: report(**changes))


def test_passing_validation_promotes_and_records_the_decision(manager, monkeypatch):
    with_report(manager, monkeypatch)
    memory = manager.finalize_validation("memory-1")
    assert memory.status is MemoryStatus.TRUSTED
    decision = manager.get_decision("memory-1")
    assert decision["profile"] == "default"
    assert decision["decision"] == "promoted" and decision["failed_gates"] == []
    assert decision["report"]["treatment_successes"] == 5
    assert manager.finalize_validation("memory-1").status is MemoryStatus.TRUSTED  # idempotent


@pytest.mark.parametrize("changes,gates", [
    ({"safety_regressions": [{"seed": 103}], "treatment_unsafe_attempts": 1}, ["safety_regression"]),
    ({"success_gain": -1, "treatment_successes": 3}, ["success_regression", "treatment_fraction_below_0.75"]),
    ({"success_gain": 0, "treatment_successes": 5}, ["no_measured_gain"]),
    ({"treatment_successes": 2, "success_gain": 2},
     ["insufficient_treatment_successes", "treatment_fraction_below_0.75"]),
    ({"validated_scope": []}, ["no_validated_scope"]),
    ({"paired_dev_seeds": 4, "treatment_successes": 4, "success_gain": 4}, ["insufficient_pairs"]),
    ({"validation_policy_hashes": ["a", "b"]}, ["mixed_policy_versions"]),
])
def test_failed_gates_reject_the_candidate_with_reasons(manager, monkeypatch, changes, gates):
    with_report(manager, monkeypatch, **changes)
    memory = manager.finalize_validation("memory-1")
    assert memory.status is MemoryStatus.REJECTED
    assert memory.status_reason == "validation_gate_failed: " + ", ".join(gates)
    decision = manager.get_decision("memory-1")
    assert decision["decision"] == "rejected" and decision["failed_gates"] == gates
    assert len(decision["failed_gate_messages"]) == len(gates)
    assert decision["report"]["paired_dev_seeds"] == changes.get("paired_dev_seeds", 5)
    # The verdict is final: it never promotes later and repeating it changes nothing.
    assert manager.finalize_validation("memory-1").status is MemoryStatus.REJECTED
    with pytest.raises(StateConflictError):
        manager.validate_and_promote("memory-1")


def test_incomplete_plan_is_not_a_verdict(manager, monkeypatch):
    def incomplete(memory_id):
        raise StateConflictError("all predefined variation cases must be completed")

    monkeypatch.setattr(manager, "_plan_complete", incomplete)
    with pytest.raises(StateConflictError, match="must be completed"):
        manager.finalize_validation("memory-1")
    assert manager.store.get_memory("memory-1").status is MemoryStatus.CANDIDATE
    assert manager.get_decision("memory-1") is None


def test_validate_and_promote_keeps_its_original_failure_messages(manager, monkeypatch):
    with_report(manager, monkeypatch, safety_regressions=[{"seed": 1}], treatment_unsafe_attempts=1)
    with pytest.raises(StateConflictError, match="safety regression blocks"):
        manager.validate_and_promote("memory-1")
    assert manager.store.get_memory("memory-1").status is MemoryStatus.CANDIDATE


def trusted_store(tmp_path):
    store, _ = populated_store(tmp_path / "lifecycle.sqlite3")
    lifecycle = MemoryManager(store)
    lifecycle.add_candidate(candidate())
    lifecycle.record_validation("memory-1", successes=5, failures=0)
    lifecycle.promote("memory-1")
    return store, lifecycle


def test_disable_enable_disable_cycle_keeps_every_transition(tmp_path):
    store, lifecycle = trusted_store(tmp_path)
    lifecycle.disable("memory-1", reason="camera mismatch", actor="alice")
    lifecycle.enable("memory-1", reason="fixed camera", actor="bob")
    assert store.get_memory("memory-1").status is MemoryStatus.TRUSTED
    second = lifecycle.disable("memory-1", reason="regressed again", actor="carol")
    assert second.status is MemoryStatus.DISABLED and second.status_reason == "regressed again"
    lifecycle.enable("memory-1", reason="verified", actor="dave")
    rolled = lifecycle.rollback("memory-1", reason="withdrawn", actor="erin")
    assert rolled.status is MemoryStatus.ROLLED_BACK
    history = store.memory_events("memory-1")
    assert [(h["status"], h["actor"]) for h in history] == [
        ("validated", None), ("trusted", None), ("disabled", "alice"), ("trusted", "bob"),
        ("disabled", "carol"), ("trusted", "dave"), ("rolled_back", "erin")]
    assert history[2]["reason"] == "camera mismatch"
    with pytest.raises(StateConflictError):
        lifecycle.enable("memory-1", reason="too late", actor="frank")  # rolled back is permanent


def test_service_lifecycle_is_idempotent_and_requires_actor_and_reason(tmp_path, monkeypatch):
    store, _ = trusted_store(tmp_path)
    service = MemoryService(store.path)
    monkeypatch.setattr(service.artifacts, "publish", lambda memory_id: None)
    monkeypatch.setattr(service.v2, "provenance", lambda memory_id: {"source_kind": "advisor_proxy"})
    with pytest.raises(ValueError):
        service.disable("memory-1", actor=" ", reason="x")
    with pytest.raises(ValueError):
        service.enable("memory-1", actor="alice", reason="")
    first = service.disable("memory-1", actor="alice", reason="mismatch")
    assert service.disable("memory-1", actor="alice", reason="mismatch").status is first.status
    assert service.enable("memory-1", actor="bob", reason="fixed").status is MemoryStatus.TRUSTED
    assert service.enable("memory-1", actor="bob", reason="fixed").status is MemoryStatus.TRUSTED
    assert [h["status"] for h in service.history("memory-1")][-2:] == ["disabled", "trusted"]


def test_service_catalogue_reports_validation_progress_and_uses(tmp_path, monkeypatch):
    store, _ = trusted_store(tmp_path)
    store.record_memory_use(MemoryUseRecord(
        "use-1", "memory-1", 1, "run-1", "attempt-1", 21.0, "native_success", ("artifact://r",)))
    service = MemoryService(store.path)
    monkeypatch.setattr(service.v2, "_provenance", lambda memory_id: {
        "source_kind": "advisor_proxy", "source_run_id": "run-1", "source_attempt_id": "attempt-0",
        "source_trace_id": "trace-1", "request_id": "request-1", "responder": "advisor_proxy"})
    monkeypatch.setattr(service.v2, "provenance", lambda memory_id: {"source_kind": "advisor_proxy"})
    monkeypatch.setattr(service.v2, "get_plan", lambda memory_id: {"seeds": [1, 2, 3, 4, 5]})
    monkeypatch.setattr(service.v2, "list_pairs", lambda memory_id: [{"seed": 1}, {"seed": 2}])
    rows = service.list_memories()
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "trusted" and row["use_count"] == 1
    assert row["validation"] == {"planned_seeds": 5, "completed_pairs": 2,
                                 "decision": None, "failed_gates": []}
    assert service.list_memories(status="rejected") == []
    assert service.uses("memory-1")[0]["outcome"] == "native_success"


def test_new_routes_enforce_the_service_and_operator_keys(tmp_path):
    from attention_memory_service.memory_service_api import create_app
    from tests.test_sync_legacy_routes import sync_request

    store, _ = trusted_store(tmp_path)
    app = create_app(store.path, api_key="service-key-0123456789", operator_key="operator-key-0123456789")
    assert sync_request(app, "/memories")[0] == 401
    status, value = sync_request(app, "/memories", key="service-key-0123456789")
    assert status == 200 and value == {"memories": []}  # nothing with v2 provenance yet
    payload = {"actor": "alice", "reason": "camera mismatch"}
    for action in ("enable", "disable"):
        assert sync_request(app, f"/memories/memory-1/{action}", method="POST", payload=payload,
                            key="service-key-0123456789")[0] == 403
    for path in ("uses", "history", "decision"):
        assert sync_request(app, f"/memories/memory-1/{path}")[0] == 401
