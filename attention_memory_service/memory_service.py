"""Shared, backend-neutral authority for AttentionBench memory.

Agents may propose artifacts and trials, but only this boundary can read the
store, register evidence, or promote a candidate. The in-process interface is
also used by the local HTTP facade; it does not require a daemon for tests.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from .attention_modes import RequestState
from .core.memory import MemoryManager
from .core.models import MemoryRecord, MemoryUseRecord, RequestType
from .core.store import AttentionStore, StateConflictError
from .dev_use import DevMemoryUse
from .dev_v2 import evidence_operation
from .memory_artifacts import MemoryArtifactStore
from .memory_v2 import MemoryV2Manager


class MemoryService:
    def __init__(self, store: AttentionStore | Path, *, artifact_root: Path | None = None,
                 dev_v2_catalogs: dict[str, Any] | None = None) -> None:
        # Existing harnesses may provide a v1 AttentionStore object. Reopen its
        # file with this package's versioned decoder instead of mixing enum
        # classes from two separately imported packages.
        path = store if isinstance(store, Path) else Path(store.path)
        self.store = AttentionStore(path)
        self.v2 = MemoryV2Manager(self.store, artifact_root=artifact_root)
        self.artifacts = MemoryArtifactStore(
            self.store, artifact_root or self.store.path.parent,
        )
        self.dev_use = DevMemoryUse(self)
        from .dev_v2 import DevV2Evidence
        self.dev_v2 = DevV2Evidence(self.store, dev_v2_catalogs)

    @evidence_operation
    def import_dev_v2_execution(self, **payload) -> dict[str, Any]:
        return self.dev_v2.import_execution(**payload)

    @evidence_operation
    def import_dev_v2_source(self, **payload) -> dict[str, Any]:
        return self.dev_v2.import_source(**payload)

    @evidence_operation
    def import_dev_v2_trial(self, **payload) -> dict[str, Any]:
        return self.dev_v2.import_trial(**payload)

    @evidence_operation
    def authorize_dev_v2_validation(self, **payload) -> dict[str, Any]:
        return self.dev_v2.authorize_validation(**payload)

    @evidence_operation
    def create_dev_v2_candidate(self, **payload) -> MemoryRecord:
        memory = self.dev_v2.create_candidate(**payload)
        self.artifacts.publish(memory.memory_id)
        return memory

    @evidence_operation
    def record_dev_v2_result(self, **payload) -> dict[str, Any]:
        return self.dev_v2.record_result(**payload)

    @evidence_operation
    def record_dev_v2_pair(self, *, memory_id: str, control_import_id: str,
                          treatment_import_id: str) -> dict[str, Any]:
        trials = []
        for import_id in (control_import_id, treatment_import_id):
            old = self.dev_v2.get(import_id)
            if old['kind'] != 'trial':
                raise StateConflictError('offline trial import required')
            trials.append(self.dev_v2.import_trial(import_id=import_id,
                catalog_id=old['catalog_id'], descriptor=old['descriptor']))
        return self.record_pair(memory_id=memory_id,
            control_attempt_id=trials[0]['attempt_id'], treatment_attempt_id=trials[1]['attempt_id'],
            control_safety=Path(trials[0]['safety']['uri']), treatment_safety=Path(trials[1]['safety']['uri']))

    @evidence_operation
    def source_for_agent(self, request_id: str) -> dict[str, Any]:
        """Return an answered hint and bounded public context, never raw evaluator debug."""
        request = self.store.get_request(request_id)
        if request is None or request.request_type is not RequestType.HINT:
            raise StateConflictError("memory source must be a hint request")
        if request.state is not RequestState.ANSWERED or not request.response_id:
            raise StateConflictError("memory source must be answered")
        run = self.store.get_run(request.run_id)
        trace = self.store.get_trace(request.trace_id)
        response = self.store.get_response(request.response_id)
        if run is None or trace is None or response is None:
            raise StateConflictError("memory source lineage is incomplete")
        raw = self.store.get_raw_trace(trace["raw_trace_id"])
        if raw is None:
            raise StateConflictError("memory source raw trace is missing")
        return {
            "request_id": request_id,
            "response_id": request.response_id,
            "responder": response["responder"],
            "response_content": response["content"],
            "response_created_at": response["created_at"],
            "source_trace_id": request.trace_id,
            "suite": run["suite"],
            "task_id": run["task_id"],
            "perception_mode": raw["metadata"].get("runtime", {}).get("perception_mode"),
            "failure": trace["failure"],
        }

    @evidence_operation
    def create_candidate(
        self, *, memory_id: str, request_id: str, guidance: str,
        repair: str, applicability: dict[str, str], created_at: float,
        artifact_kind: str = "text_hint", human_attention_seconds: float = 0.0,
    ) -> MemoryRecord:
        memory = self.v2.create_candidate_from_response(
            memory_id=memory_id, request_id=request_id, guidance=guidance,
            repair=repair, applicability=applicability, created_at=created_at,
            artifact_kind=artifact_kind, human_attention_seconds=human_attention_seconds,
        )
        self.artifacts.publish(memory_id)
        return memory

    @evidence_operation
    def get_memory(self, memory_id: str) -> MemoryRecord:
        memory = self.store.get_memory(memory_id)
        if memory is None:
            raise StateConflictError("unknown memory")
        self.v2.provenance(memory_id)
        return memory

    @evidence_operation
    def provenance(self, memory_id: str) -> dict[str, Any]:
        return self.v2.provenance(memory_id)

    @evidence_operation
    def check_candidate(self, memory_id: str) -> dict[str, Any]:
        """Read-only evidence check before freezing or executing validation."""
        memory = self.get_memory(memory_id)
        if memory.status.value != "candidate":
            raise StateConflictError("preflight requires a candidate memory")
        self.v2.verify_source_evidence(memory_id)
        provenance = self.v2.provenance(memory_id)
        return {
            "memory_id": memory_id,
            "status": memory.status.value,
            "suite": memory.applicability["suite"],
            "task_id": memory.applicability["task_id"],
            "perception_mode": memory.applicability["perception_mode"],
            "source_policy_id": provenance["source_policy_id"],
            "source_policy_sha256": provenance["source_policy_sha256"],
            "source_evidence_verified": True,
        }

    @evidence_operation
    def retrieve(self, context: dict[str, Any], *, now: float) -> list[MemoryRecord]:
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError("retrieval time must be finite")
        # ``now`` remains a validated compatibility field. Live lifecycle and
        # eligibility use only the server wall clock: future queries cannot
        # expire any scope, and past queries cannot revive expired guidance.
        # Pass the same trusted clock through v2 to its core expiry sweep.
        effective_now = time.time()
        for expired in MemoryManager(self.store).expire_due(now=effective_now):
            self.artifacts.publish(expired.memory_id)
        return self.v2.retrieve(context, now=effective_now)

    @evidence_operation
    def authorize_use(
        self, context: dict[str, Any], *, memory_id: str,
        attempt_id: str, now: float,
    ) -> dict[str, Any]:
        memory = next(
            (item for item in self.retrieve(context, now=now) if item.memory_id == memory_id),
            None,
        )
        if memory is None:
            raise StateConflictError("memory is no longer trusted or applicable")
        return self.v2.issue_use_grant(
            memory_id=memory_id, version=memory.version,
            attempt_id=attempt_id, context=context,
        )

    @evidence_operation
    def authorize_dev_use(
        self, *, memory_id: str, development_id: str, context: dict[str, Any],
        source_policy_sha256: str, now: float, memory_version: int | None = None,
    ) -> dict[str, Any]:
        """Issue a separately tracked, scope-checked Dev guidance exposure."""
        return self.dev_use.authorize(
            memory_id=memory_id, development_id=development_id,
            context=context, source_policy_sha256=source_policy_sha256,
            now=now, memory_version=memory_version,
        )

    @evidence_operation
    def get_dev_use(self, grant_id: str) -> dict[str, Any]:
        return self.dev_use.get(grant_id)

    @evidence_operation
    def record_dev_result(
        self, *, grant_id: str, run_id: str, result_policy_sha256: str,
        artifact_uri: str, artifact_sha256: str,
    ) -> dict[str, Any]:
        return self.dev_use.record_result(
            grant_id=grant_id, run_id=run_id,
            result_policy_sha256=result_policy_sha256,
            artifact_uri=artifact_uri, artifact_sha256=artifact_sha256,
        )

    @evidence_operation
    def record_use(self, use: MemoryUseRecord) -> MemoryUseRecord:
        self.v2.provenance(use.memory_id)
        self.v2.verify_use(use)
        # A memory may be revoked after the policy already retrieved it. The
        # verified raw retrieval event proves this is a historical use, not a
        # new exposure of disabled guidance.
        result = self.store.record_memory_use(use, allow_revoked_after_retrieval=True)
        self.artifacts.publish(use.memory_id)
        return result

    @evidence_operation
    def record_plan(self, memory_id: str, plan: dict[str, Any]) -> dict[str, Any]:
        result = self.v2.record_plan(memory_id, plan)
        self.artifacts.publish(memory_id)
        return result

    @evidence_operation
    def get_plan(self, memory_id: str) -> dict[str, Any] | None:
        return self.v2.get_plan(memory_id)

    @evidence_operation
    def record_pair(
        self, *, memory_id: str, control_attempt_id: str,
        treatment_attempt_id: str, control_safety: Path,
        treatment_safety: Path,
    ) -> dict[str, Any]:
        pair = self.v2.record_pair(
            memory_id=memory_id,
            control_attempt_id=control_attempt_id,
            treatment_attempt_id=treatment_attempt_id,
            control_safety=control_safety,
            treatment_safety=treatment_safety,
        )
        self.artifacts.publish(memory_id)
        return pair

    @evidence_operation
    def impact_report(self, memory_id: str) -> dict[str, Any]:
        return self.v2.impact_report(memory_id)

    @evidence_operation
    def list_pairs(self, memory_id: str) -> list[dict[str, Any]]:
        return self.v2.list_pairs(memory_id)

    @evidence_operation
    def promote(self, memory_id: str) -> MemoryRecord:
        memory = self.v2.validate_and_promote(memory_id)
        self.artifacts.publish(memory_id)
        return memory

    @evidence_operation
    def disable(self, memory_id: str, *, actor: str, reason: str) -> MemoryRecord:
        if not actor.strip() or not reason.strip():
            raise ValueError("human disable requires actor and reason")
        current = self.store.get_memory(memory_id)
        if current is not None and current.status.value == "disabled":
            return current  # idempotent retry
        memory = MemoryManager(self.store).disable(
            memory_id, reason=reason, actor=actor,
        )
        self.artifacts.publish(memory_id)
        return memory

    def enable(self, memory_id: str, *, actor: str, reason: str) -> MemoryRecord:
        """Re-enable a human-disabled memory. Rolled-back and expired ones stay revoked."""
        if not actor.strip() or not reason.strip():
            raise ValueError("human re-enable requires actor and reason")
        current = self.store.get_memory(memory_id)
        if current is not None and current.status.value == "trusted":
            return current  # idempotent retry
        memory = MemoryManager(self.store).enable(memory_id, reason=reason, actor=actor)
        self.artifacts.publish(memory_id)
        return memory

    def evaluate(self, memory_id: str) -> MemoryRecord:
        """Decide a fully validated candidate: trusted, or rejected with the failed gates."""
        memory = self.v2.finalize_validation(memory_id)
        self.artifacts.publish(memory_id)
        return memory

    def decision(self, memory_id: str) -> dict[str, Any] | None:
        self.v2.provenance(memory_id)
        return self.v2.get_decision(memory_id)

    def history(self, memory_id: str) -> list[dict[str, Any]]:
        self.get_memory(memory_id)
        return self.store.memory_events(memory_id)

    def uses(self, memory_id: str) -> list[dict[str, Any]]:
        self.get_memory(memory_id)
        return self.store.list_memory_uses(memory_id)

    def list_memories(self, *, status: str | None = None) -> list[dict[str, Any]]:
        """Compact catalogue of the shared (non-private) memories with validation progress."""
        rows = []
        for memory in self.store.list_memories():
            if status is not None and memory.status.value != status:
                continue
            provenance = self.v2._provenance(memory.memory_id)
            if provenance is None or provenance.get("channel") == "dev_v2":
                continue
            plan = self.v2.get_plan(memory.memory_id)
            pairs = self.v2.list_pairs(memory.memory_id)
            decision = self.v2.get_decision(memory.memory_id)
            rows.append({
                "memory_id": memory.memory_id, "version": memory.version,
                "status": memory.status.value, "status_reason": memory.status_reason,
                "guidance": memory.guidance, "applicability": memory.applicability,
                "confidence": memory.confidence,
                "validation_successes": memory.validation_successes,
                "validation_failures": memory.validation_failures,
                "created_at": memory.created_at, "expires_at": memory.expires_at,
                "source_kind": provenance["source_kind"],
                "source_run_id": provenance["source_run_id"],
                "source_attempt_id": provenance["source_attempt_id"],
                "source_trace_id": provenance["source_trace_id"],
                "request_id": provenance["request_id"],
                "responder": provenance["responder"],
                "validation": {
                    "planned_seeds": None if plan is None else len(plan["seeds"]),
                    "completed_pairs": len(pairs),
                    "decision": None if decision is None else decision["decision"],
                    "failed_gates": [] if decision is None else decision["failed_gates"],
                },
                "use_count": len(self.store.list_memory_uses(memory.memory_id)),
            })
        return rows

    @evidence_operation
    def rollback(self, memory_id: str, *, actor: str, reason: str) -> MemoryRecord:
        if not actor.strip() or not reason.strip():
            raise ValueError("human rollback requires actor and reason")
        memory = MemoryManager(self.store).rollback(
            memory_id, reason=reason, actor=actor,
        )
        self.artifacts.publish(memory_id)
        return memory

    @evidence_operation
    def set_expiry(
        self, memory_id: str, *, expires_at: float, actor: str, reason: str,
    ) -> MemoryRecord:
        memory = self.store.set_memory_expiry(
            memory_id, expires_at=expires_at, actor=actor, reason=reason,
        )
        self.artifacts.publish(memory_id)
        return memory

    @evidence_operation
    def export_package(self, memory_id: str) -> Path:
        return self.artifacts.publish(memory_id)

    @evidence_operation
    def verify_package(self, memory_id: str) -> dict[str, Any]:
        return self.artifacts.verify(memory_id)
