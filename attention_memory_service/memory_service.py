"""Shared, backend-neutral authority for AttentionBench memory.

Agents may propose artifacts and trials, but only this boundary can read the
store, register evidence, or promote a candidate. The in-process interface is
also used by the local HTTP facade; it does not require a daemon for tests.
"""

from __future__ import annotations

import math
import hashlib
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
from .memory_v2 import MemoryV2Manager, trusted_here
from .applicability import require_profile_match, validate_profile, profile_fields, LAYER, LEVELS, SAME_SCENE, marker
from . import provisional_notes
from .seeded_notes import create as create_test_seeded, flags, is_test_seeded, is_provisional


class MemoryService:
    def __init__(self, store: AttentionStore | Path, *, artifact_root: Path | None = None,
                 dev_v2_catalogs: dict[str, Any] | None = None,
                 dev_v2_registration_roots: dict[str, Any] | None = None) -> None:
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
        provisional_notes.initialize(self.store)
        from .dev_v2 import DevV2Evidence
        self.dev_v2 = DevV2Evidence(self.store, dev_v2_catalogs, dev_v2_registration_roots)
        from .cross_seed import CrossSeed
        self.cross_seed = CrossSeed(self)
        from .positive_controls import PositiveControls
        self.positive_controls = PositiveControls(self)

    def register_dev_v2_catalog(self, **payload):
        return self.dev_v2.register_online(**payload)

    def validation_queue(self):
        from .auto_validation import validation_queue
        return validation_queue(self)

    def memory_events(self, *, after=0):
        from .auto_validation import memory_events
        return memory_events(self.store, after=after)

    def _publish_if_validated_source(self, memory_id: str) -> None:
        if not flags(self.v2._provenance(memory_id)):
            self.artifacts.publish(memory_id)

    def _refuse_provisional(self, memory_id):
        if is_provisional(self.v2._provenance(memory_id)):
            raise StateConflictError('provisional notes cannot enter validation, promotion or results')

    def create_provisional_note(self, **payload) -> dict[str, Any]:
        memory = provisional_notes.create(self, **payload)
        return {'memory_id': memory.memory_id, 'version': memory.version,
                'status': memory.status.value, 'expires_at': memory.expires_at,
                'source_kind': 'advisor_proxy', **flags(self.v2.provenance(memory.memory_id))}

    def record_provisional_outcome(self, **payload) -> dict[str, Any]:
        return provisional_notes.record_outcome(self, **payload)

    def create_test_seeded_note(self, **payload) -> dict[str, Any]:
        memory = create_test_seeded(self, **payload)
        return {'memory_id': memory.memory_id, 'version': memory.version,
                'status': memory.status.value, 'source_kind': 'seeded_for_tests',
                'source_label': 'seeded for tests', **flags(self.v2.provenance(memory.memory_id))}

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
    def record_dev_v2_failed_check(self, **payload) -> dict[str, Any]:
        return self.v2.record_failed_check(**payload)

    @evidence_operation
    def authorize_dev_v2_validation(self, **payload) -> dict[str, Any]:
        self._refuse_provisional(payload['memory_id'])
        return self.dev_v2.authorize_validation(**payload)

    @evidence_operation
    def create_dev_v2_candidate(self, **payload) -> MemoryRecord:
        self._refuse_provisional(payload['memory_id'])
        memory = self.dev_v2.create_candidate(**payload)
        self.artifacts.publish(memory.memory_id)
        return memory

    @evidence_operation
    def record_dev_v2_result(self, **payload) -> dict[str, Any]:
        result = self.dev_v2.record_result(**payload)
        plan = self.v2.get_plan(result['memory_id'])
        if plan and plan.get('profile') == 'single_scene_v1':
            execution = self.dev_v2.get(payload['execution_import_id'])
            data = self.dev_v2.ref(execution['catalog_id'], execution['public_feedback'], raw=True)
            self.record_provisional_outcome(grant_id=result['grant_id'], development_id=result['development_id'],
                public_feedback_text=data.decode('utf-8'), public_feedback_sha256=hashlib.sha256(data).hexdigest())
        return result

    @evidence_operation
    def record_dev_v2_pair(self, *, memory_id: str, control_import_id: str,
                          treatment_import_id: str) -> dict[str, Any]:
        self._refuse_provisional(memory_id)
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
        return [memory for memory, _ in self.retrieve_levels(context, now=now, levels=(SAME_SCENE,))]

    @evidence_operation
    def retrieve_levels(self, context: dict[str, Any], *, now: float,
                        levels=(SAME_SCENE,)) -> list[tuple[MemoryRecord, str]]:
        """Layered retrieval; callers opt in to the reference-only same_task level."""
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError("retrieval time must be finite")
        validate_profile(context)
        effective_now = time.time()
        for expired in MemoryManager(self.store).expire_due(now=effective_now):
            self._publish_if_validated_source(expired.memory_id)
        return self.v2.retrieve_levels(context, now=effective_now, levels=tuple(levels))

    @evidence_operation
    def authorize_use(
        self, context: dict[str, Any], *, memory_id: str,
        attempt_id: str, now: float,
    ) -> dict[str, Any]:
        # Attempt-bound grants feed validation/use evidence: never provisional.
        self._refuse_provisional(memory_id)
        memory = next(
            (item for item in self.retrieve(context, now=now) if item.memory_id == memory_id),
            None,
        )
        if memory is None:
            note = self.store.get_memory(memory_id)
            if note is not None:
                require_profile_match(note.applicability, context)
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
        grant = self.dev_use.get(grant_id)
        if grant.get('unvalidated'):
            raise StateConflictError('provisional exposure is excluded from Dev results')
        if grant.get('trusted_here') is False:
            raise StateConflictError('reference exposure is excluded from Dev results')
        if grant.get('seeded_for_tests'):
            raise StateConflictError('test-seeded exposure is excluded from Dev results')
        return self.dev_use.record_result(
            grant_id=grant_id, run_id=run_id,
            result_policy_sha256=result_policy_sha256,
            artifact_uri=artifact_uri, artifact_sha256=artifact_sha256,
        )

    @evidence_operation
    def record_use(self, use: MemoryUseRecord) -> MemoryUseRecord:
        self._refuse_provisional(use.memory_id)
        self.v2.provenance(use.memory_id)
        self.v2.verify_use(use)
        # A memory may be revoked after the policy already retrieved it. The
        # verified raw retrieval event proves this is a historical use, not a
        # new exposure of disabled guidance.
        result = self.store.record_memory_use(
            use, allow_revoked_after_retrieval=True,
            markers=flags(self.v2._provenance(use.memory_id)))
        self._publish_if_validated_source(use.memory_id)
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
        self._refuse_provisional(memory_id)
        if is_test_seeded(self.v2._provenance(memory_id)):
            raise StateConflictError('test-seeded notes are excluded from impact reports')
        return self.v2.impact_report(memory_id)

    @evidence_operation
    def list_pairs(self, memory_id: str) -> list[dict[str, Any]]:
        return self.v2.list_pairs(memory_id)

    @evidence_operation
    def promote(self, memory_id: str) -> MemoryRecord:
        self._refuse_provisional(memory_id)
        if is_test_seeded(self.v2._provenance(memory_id)):
            raise StateConflictError('test-seeded notes cannot be promoted')
        plan = self.v2.get_plan(memory_id)
        memory = (self.v2.finalize_validation(memory_id)
                  if plan and plan.get('profile') == 'single_scene_v1'
                  else self.v2.validate_and_promote(memory_id))
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
        self._publish_if_validated_source(memory_id)
        return memory

    def enable(self, memory_id: str, *, actor: str, reason: str) -> MemoryRecord:
        """Re-enable a human-disabled memory. Rolled-back and expired ones stay revoked."""
        if not actor.strip() or not reason.strip():
            raise ValueError("human re-enable requires actor and reason")
        current = self.store.get_memory(memory_id)
        if current is not None and current.status.value in {"trusted", "provisional"}:
            if current.status.value == "provisional" and current.expires_at <= time.time():
                raise StateConflictError("provisional note expired; set_expiry before enabling")
            return current  # idempotent retry
        memory = MemoryManager(self.store).enable(memory_id, reason=reason, actor=actor)
        self._publish_if_validated_source(memory_id)
        return memory

    def evaluate(self, memory_id: str) -> MemoryRecord:
        """Decide a fully validated candidate: trusted, or rejected with the failed gates."""
        self._refuse_provisional(memory_id)
        if is_test_seeded(self.v2._provenance(memory_id)):
            raise StateConflictError('test-seeded notes cannot be evaluated')
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
        mark = flags(self.v2._provenance(memory_id))
        return [{**use, **mark} for use in self.store.list_memory_uses(memory_id)]

    def list_memories(self, *, status: str | None = None) -> list[dict[str, Any]]:
        """Compact catalogue of the shared (non-private) memories with validation progress."""
        rows = []
        for memory in self.store.list_memories():
            if status is not None and memory.status.value != status:
                continue
            provenance = self.v2._provenance(memory.memory_id)
            if provenance is None or provenance.get("channel") in {'dev_v2', 'test_seeded', 'dev_v2_provisional'}:
                continue
            plan = self.v2.get_plan(memory.memory_id)
            pairs = self.v2.list_pairs(memory.memory_id)
            decision = self.v2.get_decision(memory.memory_id)
            rows.append({
                "memory_id": memory.memory_id, "version": memory.version,
                "status": memory.status.value, "status_reason": memory.status_reason,
                "guidance": memory.guidance, "applicability": memory.applicability,
                **profile_fields(memory.applicability),
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

    def list_dev_v2_memories(self, *, status: str | None = None,
                             context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Explicit public allowlist; importer evidence never enters this projection.

        With a Dev context, each row also says at which level it would be read
        there (``match_level``) and whether it is trusted there; without one
        both are None.
        """
        levels = None
        if context is not None:
            levels = {memory.memory_id: level for memory, level in
                      self.retrieve_levels(context, now=time.time(), levels=LEVELS)}
        rows = []
        for memory in self.store.list_memories():
            if status is not None and memory.status.value != status:
                continue
            provenance = self.v2._provenance(memory.memory_id)
            if not provenance or provenance.get('channel') not in {'dev_v2', 'test_seeded', 'dev_v2_provisional'}:
                continue
            plan = self.v2.get_plan(memory.memory_id)
            decision = self.v2.get_decision(memory.memory_id)
            seeded = is_test_seeded(provenance)
            provisional = is_provisional(provenance)
            case = plan['cases'][0] if plan and plan.get('profile') == 'single_scene_v1' else None
            if seeded or provisional and provenance.get('channel') == 'dev_v2_provisional':
                case = memory.applicability
            trusted = not provisional and memory.status.value in {'trusted', 'disabled', 'expired', 'rolled_back'}
            scope = ('test_seeded' if seeded else 'scene_scoped' if plan and plan.get('profile') == 'single_scene_v1' else 'multi_scene') if trusted else None
            if trusted and provenance.get('trust_scope') == 'task_scoped':
                scope = 'task_scoped'
            from .motion_permission import public_summary
            motion_permission = public_summary(self.store, memory.memory_id, provenance)
            rows.append({
                **({'motion_permission': motion_permission} if motion_permission else {}),
                'memory_id': memory.memory_id, 'version': memory.version,
                'layer': 'task' if scope == 'task_scoped' else 'scene',
                'from_provisional': provenance.get('from_provisional'),
                **({'decision_link': provenance['decision_link']} if 'decision_link' in provenance else {}),
                **({'read_layer': LAYER.get(levels.get(memory.memory_id))} if levels is not None else {}),
                **({'origin_summary': {k: provenance['origin'][k] for k in ('experiment_id', 'help_version', 'policy_id')},
                    'from_run_note': provenance.get('from_run_note')} if provisional and 'origin' in provenance else {}),
                'title': provenance.get('title'),
                'summary': provenance.get('summary'),
                'group': provenance.get('group'),
                'status': memory.status.value, 'trust_scope': scope,
                'trust_display': 'seeded_for_tests' if seeded else 'scene_scoped_trusted' if scope == 'scene_scoped'
                                 else 'task_scoped_trusted' if scope == 'task_scoped' else None,
                'profile': None if plan is None else plan.get('profile', 'default'),
                **profile_fields(memory.applicability),
                'applicability_summary': {
                    **profile_fields(memory.applicability),
                    'benchmark': memory.applicability['benchmark'],
                    'suite': memory.applicability['suite'],
                    'task_id': memory.applicability['task_id'], 'seed': provenance['source_seed'],
                    'scene_id': None if case is None else case['scene_id'],
                    'object_set_id': None if case is None else case['object_set_id'],
                    'robot': memory.applicability['robot'],
                    'controller': memory.applicability['controller'],
                    'perception_mode': memory.applicability['perception_mode'],
                    'camera_config_id': None if case is None else case['camera_config_id'],
                    'task_variant_id': None if case is None else case['task_variant_id'],
                    'camera_names': None if case is None else case['camera_names'],
                    'task_prompt': None if case is None else case['task_prompt'],
                    'scene_contract_ref': memory.applicability.get('scene_contract_ref') if seeded or provisional else None,
                    'sdk_sha256': memory.applicability['sdk_sha256'],
                    'sdk_interface_sha256': memory.applicability.get('sdk_interface_sha256'),
                    'namespace': memory.applicability['namespace'],
                    'planned_cases': None if plan is None else plan['cases'],
                },
                'failure_signature_summary': provenance.get('failure_signature'),
                'validation_summary': None if seeded or provisional else {
                    'planned_pairs': None if plan is None else len(plan.get('pairs', plan['seeds'])),
                    'completed_pairs': len(self.v2.list_pairs(memory.memory_id)),
                    'decision': None if decision is None else decision['decision'],
                    'failed_gates': [] if decision is None else decision['failed_gates'],
                    'control_signature_count': None if decision is None else decision['report'].get('control_signature_count'),
                    'treatment_signature_count': None if decision is None else decision['report'].get('treatment_signature_count'),
                },
                'source_label': 'seeded for tests' if seeded else provenance['source_kind'],
                'excluded_from_results': seeded or provisional,
                'seeded_for_tests': seeded,
                'source_seed': provenance['source_seed'],
                'match_level': None if levels is None else levels.get(memory.memory_id),
                'trusted_here': None if levels is None else (
                    memory.memory_id in levels and trusted_here(memory, levels[memory.memory_id], provenance, context)),
                'guidance_length': len(memory.guidance) if seeded else None,
                'guidance_sha256': hashlib.sha256(memory.guidance.encode()).hexdigest() if seeded else None,
                **({'expires_at': memory.expires_at, 'uses': provisional_notes.uses_summary(self.store, memory.memory_id),
                    **flags(provenance)} if provisional else {}),
            })
        return rows

    def get_dev_v2_memory_public(self, memory_id: str) -> dict[str, Any]:
        row = next((item for item in self.list_dev_v2_memories() if item['memory_id'] == memory_id), None)
        if row is None:
            raise FileNotFoundError('unknown Dev v2 memory')
        return row

    @evidence_operation
    def rollback(self, memory_id: str, *, actor: str, reason: str) -> MemoryRecord:
        if not actor.strip() or not reason.strip():
            raise ValueError("human rollback requires actor and reason")
        memory = MemoryManager(self.store).rollback(
            memory_id, reason=reason, actor=actor,
        )
        self._publish_if_validated_source(memory_id)
        return memory

    @evidence_operation
    def set_expiry(
        self, memory_id: str, *, expires_at: float, actor: str, reason: str,
    ) -> MemoryRecord:
        memory = self.store.set_memory_expiry(
            memory_id, expires_at=expires_at, actor=actor, reason=reason,
        )
        self._publish_if_validated_source(memory_id)
        return memory

    @evidence_operation
    def export_package(self, memory_id: str) -> Path:
        self._refuse_provisional(memory_id)
        if is_test_seeded(self.v2._provenance(memory_id)):
            raise StateConflictError('test-seeded notes have no validation package')
        return self.artifacts.publish(memory_id)

    @evidence_operation
    def verify_package(self, memory_id: str) -> dict[str, Any]:
        return self.artifacts.verify(memory_id)
