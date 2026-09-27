"""Service-owned checks that do not depend on a simulator or Universe code."""

from __future__ import annotations

import json

import pytest

from attention_memory_service.core.memory import MemoryManager
from attention_memory_service.core.models import MemoryRecord, MemoryStatus, MemoryUseRecord
from attention_memory_service.core.store import AttentionStore, StateConflictError
from attention_memory_service.memory_v2 import MemoryV2Manager, _wilson_95


def _memory() -> MemoryRecord:
    return MemoryRecord(
        memory_id="memory-1", version=1, source_trace_id="trace-1",
        guidance="grasp from above", candidate_repair="shift the approach",
        applicability={"suite": "robocasa", "task_id": "pick", "perception_mode": "sim_gt"},
        evidence_refs=("evidence-1",), created_at=1.0, status=MemoryStatus.TRUSTED,
    )


def test_confidence_reference_reports_small_sample_uncertainty():
    assert _wilson_95(0, 0) is None
    all_success = _wilson_95(5, 5)
    assert all_success["n"] == 5 and all_success["upper"] == 1.0
    assert all_success["lower"] < 0.6
    assert "not assumed IID" in all_success["interpretation"]


def test_impact_separates_help_saved_on_success_from_help_saved_on_failure(monkeypatch, tmp_path):
    manager = MemoryV2Manager(AttentionStore(tmp_path / "memory.sqlite3"))
    monkeypatch.setattr(manager, "provenance", lambda memory_id: {
        "source_kind": "advisor_proxy", "human_attention_seconds": 0.0,
        "source_policy_sha256": "source-policy",
    })
    monkeypatch.setattr(manager, "list_pairs", lambda memory_id: [
        {"seed": seed, "control_attempt_id": f"c-{seed}",
         "treatment_attempt_id": f"t-{seed}",
         "control_safety": {"arm": "c", "seed": seed},
         "treatment_safety": {"arm": "t", "seed": seed}}
        for seed in (102, 103)
    ])
    monkeypatch.setattr(manager, "_verify_safety", lambda reference: int(
        reference in ({"arm": "c", "seed": 102}, {"arm": "t", "seed": 103})
    ))

    def trial(attempt_id):
        arm, seed = attempt_id.split("-")
        seed = int(seed)
        success = seed == 103
        return {
            "run": {"seed": seed}, "memory_ids": [] if arm == "c" else ["memory-1"],
            "exposure": "none" if arm == "c" else "candidate_validation",
            "success": success, "credits": 1 if arm == "c" else 0,
            "policy_sha256": "validation-policy",
            "seconds": 1.0, "gpu_seconds": 0.0, "tokens": 0,
            "variation": {
                "scene_id": f"scene-{seed}", "object_set_id": f"objects-{seed}",
                "camera_config_id": f"camera-{seed}", "task_variant_id": f"task-{seed}",
            },
        }

    monkeypatch.setattr(manager, "_trial", trial)
    report = manager.impact_report("memory-1")
    assert report["future_assistance_credits_saved"] == 2
    assert report["both_success_assistance_credits_saved"] == 1
    assert report["both_failed_assistance_credits_saved"] == 1
    assert len(report["supporting_successes"]) == 1
    assert report["control_unsafe_attempts"] == report["treatment_unsafe_attempts"] == 1
    assert [item["seed"] for item in report["safety_regressions"]] == [103]


def test_retrieval_requires_concrete_view_and_prompt(monkeypatch, tmp_path):
    manager = MemoryV2Manager(AttentionStore(tmp_path / "memory.sqlite3"))
    monkeypatch.setattr(MemoryManager, "retrieve", lambda self, context, now: [_memory()])
    monkeypatch.setattr(manager, "_provenance", lambda memory_id: {"memory_id": memory_id})
    monkeypatch.setattr(manager, "provenance", lambda memory_id: {"memory_id": memory_id})
    monkeypatch.setattr(manager, "verify_source_evidence", lambda memory_id: None)
    case = {
        "scene_id": "scene-a", "object_set_id": "objects-a",
        "camera_config_id": "camera-a", "task_variant_id": "task-a",
        "camera_names": ["wrist_camera"], "task_prompt": "pick up the cup",
    }
    monkeypatch.setattr(manager, "get_plan", lambda memory_id: {"cases": [case]})
    monkeypatch.setattr(manager, "impact_report", lambda memory_id: {
        "validated_scope": [{key: case[key] for key in (
            "scene_id", "object_set_id", "camera_config_id", "task_variant_id"
        )}],
    })
    context = {**_memory().applicability, **case}
    assert [item.memory_id for item in manager.retrieve(context, now=10.0)] == ["memory-1"]
    assert manager.retrieve({**context, "camera_names": ["base_camera"]}, now=10.0) == []
    assert manager.retrieve({**context, "task_prompt": "different task"}, now=10.0) == []
    assert manager.retrieve({key: value for key, value in context.items() if key != "camera_names"}, now=10.0) == []


def test_use_requires_trusted_trace_and_native_outcome(monkeypatch, tmp_path):
    manager = MemoryV2Manager(AttentionStore(tmp_path / "memory.sqlite3"))
    use = MemoryUseRecord(
        use_id="use-1", memory_id="memory-1", memory_version=1,
        run_id="run-1", attempt_id="attempt-1", used_at=10.0,
        outcome="native_success",
    )
    context = {"suite": "robocasa", "task_id": "pick", "perception_mode": "sim_gt"}
    trial = {
        "run_id": "run-1", "exposure": "trusted",
        "run": {"suite": "robocasa", "task_id": "pick"},
        "mode": "sim_gt", "memory_context": context,
        "memory_ids": ["memory-1"], "retrieval_versions": [1],
        "retrieval_timestamps": [10.0], "retrieval_grants": ["grant-1"],
        "success": True, "attempt_status": "succeeded", "evaluator_authoritative": True,
    }
    monkeypatch.setattr(manager, "_trial", lambda attempt_id, **kwargs: trial)
    monkeypatch.setattr(manager, "get_use_grant", lambda grant_id: {
        "memory_id": "memory-1", "version": 1,
        "attempt_id": "attempt-1", "used_at": 10.0, "context": context,
    } if grant_id == "grant-1" else None)
    manager.verify_use(use)
    trial["attempt_status"] = "timed_out"
    with pytest.raises(StateConflictError, match="outcome disagrees"):
        manager.verify_use(use)
    manager.verify_use(MemoryUseRecord(**{**use.__dict__, "outcome": "timeout"}))
    trial["attempt_status"] = "succeeded"
    with pytest.raises(StateConflictError, match="outcome disagrees"):
        manager.verify_use(MemoryUseRecord(**{**use.__dict__, "outcome": "failed"}))
    trial["retrieval_versions"] = [2]
    with pytest.raises(StateConflictError, match="version disagrees"):
        manager.verify_use(use)
    trial["retrieval_versions"] = [1]
    trial["retrieval_timestamps"] = [11.0]
    with pytest.raises(StateConflictError, match="time disagrees"):
        manager.verify_use(use)
    trial["retrieval_timestamps"] = [10.0]
    trial["retrieval_grants"] = ["forged-grant"]
    with pytest.raises(StateConflictError, match="Service-issued grant"):
        manager.verify_use(use)
    trial["retrieval_grants"] = ["grant-1"]
    trial["run"]["task_id"] = "other-task"
    with pytest.raises(StateConflictError, match="context disagrees"):
        manager.verify_use(use)
    trial["run"]["task_id"] = "pick"
    trial["memory_ids"] = []
    with pytest.raises(StateConflictError, match="retrieval event"):
        manager.verify_use(use)


def test_safety_count_must_agree_with_monitor_events(tmp_path):
    path = tmp_path / "safety.json"
    artifact = {
        "schema_version": "attentionbench.safety-monitor.v1",
        "source": "independent_safety_monitor",
        "attempt_id": "attempt-1",
        "unsafe_attempts": 0,
        "events": [{"kind": "state_sample"}],
        "violations": [{"kind": "observed_step_exceeds_limit"}],
    }
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="count disagrees"):
        MemoryV2Manager._safety(path, "attempt-1")
    artifact["unsafe_attempts"] = 1
    path.write_text(json.dumps(artifact))
    assert MemoryV2Manager._safety(path, "attempt-1")["attempt_id"] == "attempt-1"
    artifact["events"] = []
    artifact["violations"] = []
    artifact["unsafe_attempts"] = 0
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="event evidence"):
        MemoryV2Manager._safety(path, "attempt-1")
