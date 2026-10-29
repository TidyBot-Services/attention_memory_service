"""Evidence-gated v2 memory on top of the frozen v1 AttentionStore.

The sidecar tables are additive: v1 records and its freeze manifest are not
modified. A v1 ``trusted`` label alone is deliberately insufficient for v2.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from .attention_modes import AssistanceMode, RequestState
from .core.memory import MemoryManager, _matches
from .core.models import MemoryRecord, MemoryStatus, MemoryUseRecord, RequestType
from .core.store import AttentionStore, StateConflictError
from .seed_guard import classify_seed
from .seeded_notes import is_test_seeded, is_provisional, flags, TUPLE_KEYS
from .applicability import (INTERFACE_KEY, LEVELS, SAME_SCENE, SAME_TASK, SAME_FAMILY, same_family, same_scene,
                            same_task, valid_interface, PROFILE_KEY, DEV_LOCAL_KEY, profile_matches, validate_profile,
                            require_profile_match, profile_fields)


def trusted_here(memory, level, provenance=None, context=None):
    """Task authority is explicit; legacy and seeded notes keep exact trust."""
    if context is not None and not profile_matches(memory.applicability, context):
        return False
    task = (provenance or {}).get('channel') == 'dev_v2' and (provenance or {}).get('trust_scope') == 'task_scoped'
    return memory.status is MemoryStatus.TRUSTED and (level == SAME_SCENE or task and level == SAME_TASK)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encoded(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _wilson_95(successes: int, trials: int) -> dict[str, Any] | None:
    """Descriptive uncertainty for a small binomial treatment sample."""
    if trials == 0:
        return None
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return {
        "method": "wilson_score", "level": 0.95, "n": trials,
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
        "interpretation": "binomial reference only; frozen variation cases are not assumed IID",
    }


class MemoryV2Manager:
    """Convert costed advice into artifacts and validate it with paired runs."""

    def __init__(self, store: AttentionStore, *, artifact_root: Path | None = None) -> None:
        # Reopen a caller's v1 store through this package's record decoder.
        # The shared SQLite schema is versioned; Python enum identities are not.
        self.store = AttentionStore(Path(store.path))
        self.artifact_root = (artifact_root or self.store.path.parent).resolve()
        with sqlite3.connect(self.store.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS memory_v2_provenance "
                "(memory_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS memory_v2_pairs "
                "(pair_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, "
                "seed INTEGER NOT NULL, payload TEXT NOT NULL, "
                "UNIQUE(memory_id, seed))"
            )
            db.execute("CREATE TABLE IF NOT EXISTS memory_v2_scene_pairs "
                       "(pair_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, payload TEXT NOT NULL)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS memory_v2_plans "
                "(memory_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS memory_v2_decisions "
                "(memory_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS memory_v2_use_grants "
                "(grant_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, "
                "attempt_id TEXT NOT NULL, payload TEXT NOT NULL, "
                "UNIQUE(memory_id, attempt_id))"
            )

    def issue_use_grant(
        self, *, memory_id: str, version: int, attempt_id: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Bind a live trusted lookup to one attempt before releasing guidance."""
        validate_profile(context)
        if not attempt_id or not isinstance(attempt_id, str):
            raise ValueError("use grant requires an attempt ID")
        # BEGIN IMMEDIATE serializes this final status check with operator
        # disable/rollback/expiry writes to the same SQLite database.
        with sqlite3.connect(self.store.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM memories WHERE id=?", (memory_id,)).fetchone()
            if row is None:
                raise StateConflictError("unknown memory")
            memory = json.loads(row[0])
            require_profile_match(memory['applicability'], context)
            issued_at = time.time()
            if (
                memory["status"] != MemoryStatus.TRUSTED.value
                or memory["version"] != version
                or memory.get("expires_at") is not None
                and memory["expires_at"] <= issued_at
            ):
                raise StateConflictError("memory is no longer trusted or unexpired")
            existing = db.execute(
                "SELECT payload FROM memory_v2_use_grants WHERE memory_id=? AND attempt_id=?",
                (memory_id, attempt_id),
            ).fetchone()
            if existing is not None:
                grant = json.loads(existing[0])
                if grant["version"] != version or grant["context"] != context:
                    raise StateConflictError("attempt already has a different memory-use grant")
                return grant
            grant = {
                "schema_version": "attentionbench.memory-use-grant.v1",
                "grant_id": f"grant:{uuid.uuid4().hex}",
                "memory_id": memory_id,
                "version": version,
                "attempt_id": attempt_id,
                "context": context,
                "used_at": issued_at,
            }
            grant.update(flags(self._provenance(memory_id)))
            grant.update(profile_fields(memory['applicability']))
            db.execute(
                "INSERT INTO memory_v2_use_grants(grant_id, memory_id, attempt_id, payload) "
                "VALUES (?, ?, ?, ?)",
                (grant["grant_id"], memory_id, attempt_id, _encoded(grant)),
            )
            return grant

    def get_use_grant(self, grant_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.store.path) as db:
            row = db.execute(
                "SELECT payload FROM memory_v2_use_grants WHERE grant_id=?", (grant_id,)
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def record_plan(self, memory_id: str, plan: dict[str, Any]) -> dict[str, Any]:
        validate_profile(plan)
        if is_provisional(self._provenance(memory_id)):
            raise StateConflictError('provisional notes cannot enter validation or impact results')
        provenance = self.provenance(memory_id)
        self.verify_source_evidence(memory_id)
        memory = self.store.get_memory(memory_id)
        if memory is None or memory.status is not MemoryStatus.CANDIDATE:
            raise StateConflictError("validation plan requires a candidate memory")
        applicability = provenance["artifact"]["applicability"]
        require_profile_match(applicability, plan)
        if provenance.get('channel') == 'dev_v2':
            from .dev_v2 import SCOPE, AXES
            if any(plan.get(k) != applicability[k] for k in SCOPE if k not in AXES):
                raise StateConflictError('validation plan needs exact SDK/namespace/robot scope')
        if plan.get("schema_version") != "attentionbench.memory-validation-plan.v2":
            raise ValueError("unsupported memory validation plan")
        if plan.get("memory_id") != memory_id:
            raise StateConflictError("validation plan memory ID mismatch")
        if any(plan.get(key) != applicability[key] for key in ("suite", "task_id", "perception_mode")):
            raise StateConflictError("validation plan applicability mismatch")
        if plan.get("policy_id") != provenance["source_policy_id"]:
            raise StateConflictError("validation plan policy mismatch")
        profile = plan.get('profile', 'default')
        if 'arm_generation' in plan and (profile != 'single_scene_v1' or plan['arm_generation'] != 'dev_from_origin'):
            raise ValueError('arm_generation requires single_scene_v1 dev_from_origin')
        if profile == 'single_scene_v1':
            return self._record_single_scene_plan(memory_id, plan, provenance)
        if profile != 'default':
            raise ValueError('unknown validation profile')
        seeds = plan.get("seeds")
        if (not isinstance(seeds, list) or len(seeds) < 5
                or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
                or len(set(seeds)) != len(seeds)):
            raise ValueError("validation plan needs five distinct development seeds")
        if any(classify_seed(seed) != "dev" for seed in seeds):
            raise PermissionError("validation plan is development-only")
        cases = plan.get("cases")
        axes = ("scene_id", "object_set_id", "camera_config_id", "task_variant_id")
        if not isinstance(cases, list) or len(cases) != len(seeds):
            raise ValueError("validation plan needs a variation case for every seed")
        if [case.get("seed") for case in cases if isinstance(case, dict)] != seeds:
            raise ValueError("variation cases must follow the frozen seed order")
        for case in cases:
            if not isinstance(case, dict) or set(case) != {"seed", *axes, "camera_names", "task_prompt"} or any(
                not isinstance(case[axis], str) or not case[axis].strip() for axis in axes
            ) or not isinstance(case["camera_names"], list) or not case["camera_names"] or any(
                not isinstance(name, str) or not name for name in case["camera_names"]
            ) or not isinstance(case["task_prompt"], str) or not case["task_prompt"].strip():
                raise ValueError("variation case needs explicit scene/object/camera/task IDs")
        if any(len({case[axis] for case in cases}) < 2 for axis in axes):
            raise ValueError("validation must vary each declared axis")
        if len({tuple(case["camera_names"]) for case in cases}) < 2 or len({case["task_prompt"] for case in cases}) < 2:
            raise ValueError("camera and task variants need different concrete views and prompts")
        camera_map = {case["camera_config_id"]: tuple(case["camera_names"]) for case in cases}
        task_map = {case["task_variant_id"]: case["task_prompt"] for case in cases}
        if (
            len(camera_map) != len({tuple(case["camera_names"]) for case in cases})
            or len(task_map) != len({case["task_prompt"] for case in cases})
            or any(camera_map[case["camera_config_id"]] != tuple(case["camera_names"]) for case in cases)
            or any(task_map[case["task_variant_id"]] != case["task_prompt"] for case in cases)
        ):
            raise ValueError("camera/task variant IDs must map one-to-one to concrete settings")
        credits = plan.get("assistance_credits")
        if isinstance(credits, bool) or not isinstance(credits, int) or credits < 0:
            raise ValueError("validation plan credits must be nonnegative")
        encoded = _encoded(plan)
        with sqlite3.connect(self.store.path) as db:
            row = db.execute(
                "SELECT payload FROM memory_v2_plans WHERE memory_id=?", (memory_id,)
            ).fetchone()
            if row is not None:
                if row[0] != encoded:
                    raise StateConflictError("validation plan is immutable once recorded")
                return json.loads(row[0])
            if db.execute(
                "SELECT 1 FROM memory_v2_pairs WHERE memory_id=? LIMIT 1", (memory_id,)
            ).fetchone() is not None:
                raise StateConflictError("validation plan cannot be backfilled after pairs")
            db.execute(
                "INSERT INTO memory_v2_plans(memory_id, payload) VALUES (?, ?)",
                (memory_id, encoded),
            )
        return plan

    def _record_single_scene_plan(self, memory_id, plan, provenance):
        from .dev_v2 import SCOPE, AXES, SHA
        if provenance.get('channel') != 'dev_v2' or not provenance.get('failure_signature'):
            raise StateConflictError('single-scene profile needs a frozen Dev v2 failure signature')
        seed = provenance['source_seed']
        case = {'seed': seed, **{k: provenance['artifact']['applicability'][k] for k in AXES}}
        # Camera names and prompt come from the sealed source context, not the
        # policy-patch applicability (which predates this profile).
        source = self.provenance(memory_id)
        from .dev_v2 import DevV2Evidence
        context = DevV2Evidence(self.store).source(source['source_import_id'])['context']
        case.update(camera_names=context['camera_names'], task_prompt=context['task_prompt'])
        validate_single_scene_plan(plan, context, seed, case, provenance['failure_signature'], provenance['source_policy_sha256'])
        encoded = _encoded(plan)
        with sqlite3.connect(self.store.path) as db:
            row = db.execute('SELECT payload FROM memory_v2_plans WHERE memory_id=?', (memory_id,)).fetchone()
            if row:
                if row[0] != encoded:
                    raise StateConflictError('validation plan is immutable once recorded')
                return json.loads(row[0])
            if db.execute('SELECT 1 FROM memory_v2_scene_pairs WHERE memory_id=? LIMIT 1', (memory_id,)).fetchone():
                raise StateConflictError('validation plan cannot be backfilled after pairs')
            db.execute('INSERT INTO memory_v2_plans(memory_id,payload) VALUES (?,?)', (memory_id, encoded))
        return plan

    def get_plan(self, memory_id: str) -> dict[str, Any] | None:
        self.provenance(memory_id)
        with sqlite3.connect(self.store.path) as db:
            row = db.execute(
                "SELECT payload FROM memory_v2_plans WHERE memory_id=?", (memory_id,)
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def create_candidate_from_response(
        self,
        *,
        memory_id: str,
        request_id: str,
        guidance: str,
        repair: str,
        applicability: dict[str, str],
        created_at: float,
        artifact_kind: str = "text_hint",
        human_attention_seconds: float = 0.0,
    ) -> MemoryRecord:
        validate_profile(applicability)
        if 'sdk_sha256' in applicability or 'namespace' in applicability:
            raise StateConflictError('exact Dev v2 scope requires trusted source importer')
        if artifact_kind not in {"text_hint", "policy_patch", "skill"}:
            raise ValueError("unsupported structured memory artifact kind")
        if not guidance.strip() or not repair.strip():
            raise ValueError("guidance and repair must be nonempty")
        if set(("perception_mode", "suite", "task_id")) - applicability.keys():
            raise ValueError("memory requires explicit mode, suite, and task")
        if applicability["perception_mode"] not in {"sim_gt", "vision"}:
            raise ValueError("unknown perception mode")
        if human_attention_seconds < 0:
            raise ValueError("human attention cost cannot be negative")
        request = self.store.get_request(request_id)
        if request is None or request.state is not RequestState.ANSWERED or not request.response_id:
            raise StateConflictError("memory requires an answered request")
        if request.request_type is not RequestType.HINT:
            raise StateConflictError("only a hint may become a reusable memory artifact")
        response = self.store.get_response(request.response_id)
        trace = self.store.get_trace(request.trace_id)
        run = self.store.get_run(request.run_id)
        if response is None or trace is None or run is None:
            raise StateConflictError("request, response, trace, and run must exist")
        if response["request_id"] != request_id or trace["run_id"] != request.run_id:
            raise StateConflictError("response or trace lineage mismatch")
        if run["assistance_mode"] != request.mode.value:
            raise StateConflictError("request mode differs from source run")
        if self.store.budget_status(request.run_id)["used"] < 1:
            raise StateConflictError("source assistance request has no committed cost")
        if run["suite"] != applicability["suite"] or run["task_id"] != applicability["task_id"]:
            raise StateConflictError("memory applicability does not match source run")
        raw = self.store.get_raw_trace(trace["raw_trace_id"])
        if raw is None or raw["metadata"].get("runtime", {}).get("perception_mode") != applicability["perception_mode"]:
            raise StateConflictError("source trace lacks matching perception provenance")
        source_policy_sha256 = raw["metadata"]["runtime"].get("policy_sha256")
        if not isinstance(source_policy_sha256, str) or len(source_policy_sha256) != 64:
            raise StateConflictError("source trace lacks a policy code digest")
        source_attempt = self.store.get_attempt(request.attempt_id)
        if (
            source_attempt is None
            or source_attempt["run_id"] != request.run_id
            or source_attempt["native_success"] is not False
            or raw["outcome"].get("native_success") is not False
            or raw["outcome"].get("evaluator_authoritative") is not True
        ):
            raise StateConflictError("memory source must be an evaluator-confirmed failed attempt")
        responder = response["responder"]
        if responder == "advisor_proxy":
            source_kind = "advisor_proxy"
        elif responder == "human" and request.mode is AssistanceMode.LIVE_HUMAN_FIRST:
            source_kind = "human"
        elif responder == "curated_human_packet":
            source_kind = "curated_human_packet"
        else:
            raise StateConflictError("response origin is not a recognized assistance source")
        if source_kind == "advisor_proxy" and human_attention_seconds:
            raise ValueError("proxy latency is not human attention")
        if source_kind != "advisor_proxy" and human_attention_seconds <= 0:
            raise ValueError("human-sourced guidance requires positive attention cost")
        received_guidance = self._received_guidance(response["content"], source_kind)
        artifact = {
            "schema_version": "attentionbench.memory-artifact.v2",
            "kind": artifact_kind,
            "guidance": guidance.strip(),
            "repair": repair.strip(),
            "applicability": applicability,
        }
        artifact_hash = _digest(_encoded(artifact).encode())
        evidence_ids = tuple(item["evidence_id"] for item in trace["evidence"])
        candidate = MemoryRecord(
            memory_id=memory_id,
            version=1,
            source_trace_id=request.trace_id,
            guidance=guidance.strip(),
            candidate_repair=repair.strip(),
            applicability=applicability,
            evidence_refs=evidence_ids,
            created_at=created_at,
        )
        provenance = {
            "schema_version": "attentionbench.memory-provenance.v2",
            "memory_id": memory_id,
            "source_kind": source_kind,
            "source_run_id": request.run_id,
            "source_attempt_id": request.attempt_id,
            "source_trace_id": request.trace_id,
            "source_raw_trace_id": trace["raw_trace_id"],
            "raw_evidence_refs": [
                {
                    "evidence_id": item["evidence_id"],
                    "kind": item["kind"],
                    "uri": item["uri"],
                    "sha256": item["sha256"],
                }
                for item in raw["evidence"]
            ],
            "source_seed": run["seed"],
            "source_policy_id": run["policy_id"],
            "source_policy_sha256": source_policy_sha256,
            "source_developer_model": run["developer_model"],
            "source_execution_target": run["execution_target"],
            "request_id": request_id,
            "response_id": request.response_id,
            "responder": response["responder"],
            "response_content_sha256": _digest(response["content"].encode("utf-8")),
            "received_guidance": received_guidance,
            "assistance_credits": 1,
            "human_attention_seconds": human_attention_seconds,
            "artifact": artifact,
            "artifact_sha256": artifact_hash,
        }
        existing = self._provenance(memory_id)
        if existing is not None:
            if existing != provenance:
                raise StateConflictError("conflicting memory provenance")
            return self.store.get_memory(memory_id)
        MemoryManager(self.store).add_candidate(candidate)
        with sqlite3.connect(self.store.path) as db:
            db.execute(
                "INSERT INTO memory_v2_provenance(memory_id, payload) VALUES (?, ?)",
                (memory_id, _encoded(provenance)),
            )
        return candidate

    def provenance(self, memory_id: str) -> dict[str, Any]:
        value = self._provenance(memory_id)
        if value is None:
            raise StateConflictError("memory has no v2 provenance")
        memory = self.store.get_memory(memory_id)
        if memory is None or memory.guidance != value["artifact"]["guidance"] or memory.candidate_repair != value["artifact"]["repair"]:
            raise StateConflictError("memory artifact differs from core record")
        if _digest(_encoded(value["artifact"]).encode()) != value["artifact_sha256"]:
            raise StateConflictError("memory artifact digest mismatch")
        if value.get("channel") == "dev_v2" or value.get("channel") == "dev_agent" and value.get("source_import_id"):
            from .dev_v2 import DevV2Evidence, SCOPE
            source = DevV2Evidence(self.store).source(value["source_import_id"])
            if value.get("channel") == "dev_agent" and source.get("channel") != "dev_agent":
                raise StateConflictError("agent source channel differs")
            if (memory.applicability != value["artifact"]["applicability"]
                    or not profile_matches(memory.applicability, source["context"])
                    or any(memory.applicability.get(k) != source["context"][k] for k in SCOPE)
                    or memory.guidance != source["guidance"]
                    or memory.candidate_repair != source["repair"]
                    or value["source_policy_sha256"] != source["source_policy_sha256"]
                    or value.get('failure_signature') != source.get('failure_signature')
                    or value['artifact'].get('failure_signature') != source.get('failure_signature')):
                raise StateConflictError("Dev v2 source/Memory binding changed")
            return value
        if value.get('channel') in {'dev_v2_provisional', 'dev_agent'}:
            if (not is_provisional(value)
                    or memory.applicability != value['artifact']['applicability']
                    or memory.source_trace_id != value['source_trace_id']
                    or value['failure_signature'] != value['artifact']['failure_signature']
                    or value['source_seed'] != memory.applicability['seed']):
                raise StateConflictError('provisional provenance binding changed')
            return value
        if value.get('channel') == 'test_seeded':
            if (not is_test_seeded(value)
                    or memory.applicability != value['artifact']['applicability']
                    or memory.source_trace_id != value['source_trace_id']
                    or value['source_seed'] != memory.applicability['seed']):
                raise StateConflictError('test-seeded provenance binding changed')
            return value
        request = self.store.get_request(value["request_id"])
        response = self.store.get_response(value["response_id"])
        if (
            request is None or request.state is not RequestState.ANSWERED
            or request.response_id != value["response_id"]
            or request.run_id != value["source_run_id"]
            or request.attempt_id != value["source_attempt_id"]
            or request.trace_id != value["source_trace_id"]
            or response is None or response["request_id"] != value["request_id"]
            or response["responder"] != value["responder"]
        ):
            raise StateConflictError("memory source request/response lineage is missing")
        received_guidance = self._received_guidance(response["content"], value["source_kind"])
        response_hash = _digest(response["content"].encode("utf-8"))
        if (
            "response_content_sha256" in value
            and value["response_content_sha256"] != response_hash
        ) or (
            "received_guidance" in value
            and value["received_guidance"] != received_guidance
        ):
            raise StateConflictError("memory source response content changed")
        trace = self.store.get_trace(value["source_trace_id"])
        if (
            trace is None
            or trace["run_id"] != value["source_run_id"]
            or trace["attempt_id"] != value["source_attempt_id"]
            or not trace.get("raw_trace_id")
        ):
            raise StateConflictError("memory source trace lineage is missing")
        raw = self.store.get_raw_trace(trace["raw_trace_id"])
        if raw is None or raw["run_id"] != trace["run_id"] or raw["attempt_id"] != trace["attempt_id"]:
            raise StateConflictError("memory source raw trace is missing")
        source_attempt = self.store.get_attempt(value["source_attempt_id"])
        source_run = self.store.get_run(value["source_run_id"])
        if (
            source_attempt is None or source_run is None
            or source_attempt["run_id"] != value["source_run_id"]
            or source_attempt["native_success"] is not False
            or raw["outcome"].get("native_success") is not False
            or raw["outcome"].get("evaluator_authoritative") is not True
            or source_run["policy_id"] != value["source_policy_id"]
            or source_run["seed"] != value["source_seed"]
            or source_run["execution_target"] != value["source_execution_target"]
        ):
            raise StateConflictError("memory source is no longer an evaluator-confirmed failed attempt")
        source_policy_sha256 = raw["metadata"].get("runtime", {}).get("policy_sha256")
        if (
            not isinstance(source_policy_sha256, str) or len(source_policy_sha256) != 64
            or "source_policy_sha256" in value
            and value["source_policy_sha256"] != source_policy_sha256
        ):
            raise StateConflictError("memory source policy code digest changed or missing")
        raw_refs = [
            {
                "evidence_id": item["evidence_id"],
                "kind": item["kind"],
                "uri": item["uri"],
                "sha256": item["sha256"],
            }
            for item in raw["evidence"]
        ]
        if (
            "source_raw_trace_id" in value
            and value["source_raw_trace_id"] != trace["raw_trace_id"]
        ) or ("raw_evidence_refs" in value and value["raw_evidence_refs"] != raw_refs):
            raise StateConflictError("memory raw evidence links differ from source trace")
        # Older v2 candidates retain the advisor trace ID but predate explicit
        # raw links. Derive those links from the immutable source trace without
        # changing the authoritative provenance row on a read.
        return {
            **value,
            "response_content_sha256": response_hash,
            "received_guidance": received_guidance,
            "source_raw_trace_id": trace["raw_trace_id"],
            "source_policy_sha256": source_policy_sha256,
            "raw_evidence_refs": raw_refs,
        }

    @staticmethod
    def _received_guidance(content: str, source_kind: str) -> str:
        if not isinstance(content, str) or not content.strip():
            raise StateConflictError("memory source response has no guidance")
        if source_kind == "advisor_proxy":
            try:
                advice = json.loads(content)
            except json.JSONDecodeError as exc:
                raise StateConflictError("memory source Advisor response is not JSON") from exc
            if not isinstance(advice, dict) or not isinstance(advice.get("guidance"), str):
                raise StateConflictError("memory source Advisor guidance is missing")
            guidance = advice["guidance"].strip()
        else:
            guidance = content.strip()
        if not guidance:
            raise StateConflictError("memory source response has no guidance")
        return guidance

    def retrieve(self, context: dict[str, Any], *, now: float) -> list[MemoryRecord]:
        """Exact (same-scene) retrieval; trial grants and legacy callers use this."""
        return [item for item, _ in self.retrieve_levels(context, now=now, levels=(SAME_SCENE,))]

    def retrieve_levels(self, context: dict[str, Any], *, now: float,
                        levels=(SAME_SCENE, SAME_TASK)) -> list[tuple[MemoryRecord, str]]:
        """Return ``(memory, match_level)``: trusted same-scene notes first, then
        notes not trusted here (provisional, other-scene), newest first."""
        validate_profile(context)
        if context.get("perception_mode") not in {"sim_gt", "vision"}:
            raise ValueError("retrieval requires explicit perception mode")
        if not isinstance(levels, (list, tuple)) or not levels or any(level not in LEVELS for level in levels):
            raise ValueError("unknown Memory match level")
        if not valid_interface(context):
            raise ValueError("sdk_interface_sha256 must be SHA-256")
        matches = MemoryManager(self.store).retrieve(context, now=now)
        for memory in self.store.list_memories():
            if memory in matches:
                continue
            provenance = self._provenance(memory.memory_id)
            if (memory.status is MemoryStatus.TRUSTED and provenance
                    and provenance.get('channel') in {'dev_v2', 'test_seeded'}
                    or memory.status is MemoryStatus.PROVISIONAL and is_provisional(provenance)):
                matches.append(memory)
        result = []
        for item in matches:
            if self._provenance(item.memory_id) is None:
                continue
            provenance = self.provenance(item.memory_id)
            level = self._match_level(item, provenance, context)
            if level in levels:
                result.append((item, level))
        here = [pair for pair in result if trusted_here(*pair, self._provenance(pair[0].memory_id), context)]
        rest = sorted((pair for pair in result if not trusted_here(*pair, self._provenance(pair[0].memory_id), context)),
                      key=lambda pair: (-pair[0].created_at, pair[0].memory_id))
        return here + rest

    def _match_level(self, item, provenance, context):
        if not profile_matches(item.applicability, context):
            return None
        from .dev_v2 import SCOPE
        if is_provisional(provenance) and provenance.get('channel') == 'dev_v2':
            plan = self.get_plan(item.memory_id)
            case = plan['cases'][0]
            if not set(context) - {INTERFACE_KEY, PROFILE_KEY, DEV_LOCAL_KEY} >= TUPLE_KEYS - {'scene_contract_ref'}:
                return None
            if (context.get('seed') == provenance['source_seed']
                    and same_scene(item.applicability, context, SCOPE)
                    and all(context.get(k) == case[k] for k in ('camera_names', 'task_prompt'))):
                return SAME_SCENE
            if same_task(item.applicability, context, {case['task_prompt']}):
                return SAME_TASK
            if same_family(item.applicability, context):
                return SAME_FAMILY
            return None
        if is_test_seeded(provenance) or is_provisional(provenance):
            if set(context) - {INTERFACE_KEY, PROFILE_KEY, DEV_LOCAL_KEY} != TUPLE_KEYS:
                return None
            if same_scene(item.applicability, context, TUPLE_KEYS):
                return SAME_SCENE
            if same_task(item.applicability, context, {item.applicability.get('task_prompt')}):
                return SAME_TASK
            if same_family(item.applicability, context):
                return SAME_FAMILY
            return None
        if self._trusted_exact(item, provenance, context):
            return SAME_SCENE
        if (item.status is MemoryStatus.TRUSTED and provenance.get('channel') == 'dev_v2'
                and set(context) - {INTERFACE_KEY, PROFILE_KEY, DEV_LOCAL_KEY} >= TUPLE_KEYS - {'scene_contract_ref'}):
            plan = self.get_plan(item.memory_id)
            prompts = {case.get('task_prompt') for case in (plan or {}).get('cases', [])}
            if plan is not None and same_task(item.applicability, context, prompts):
                self.verify_source_evidence(item.memory_id)
                return SAME_TASK
            if plan is not None and same_family(item.applicability, context):
                self.verify_source_evidence(item.memory_id)
                return SAME_FAMILY
        return None

    def _trusted_exact(self, item, provenance, context):
        if not profile_matches(item.applicability, context):
            return False
        from .dev_v2 import SCOPE
        plan = self.get_plan(item.memory_id)
        if plan and plan.get('profile', 'default') == 'single_scene_v1':
            if (context.get('seed') != provenance['source_seed']
                    or not same_scene(item.applicability, context, SCOPE)
                    or context.get('camera_names') != plan['cases'][0]['camera_names']
                    or context.get('task_prompt') != plan['cases'][0]['task_prompt']):
                return False
        if 'sdk_sha256' in context or 'namespace' in context:
            if provenance.get('channel') != 'dev_v2' or not same_scene(item.applicability, context, SCOPE):
                return False
        elif not _matches(item.applicability, context):
            # Legacy notes keep the core exact-condition match.
            return False
        if not all(item.applicability.get(key) == context.get(key) for key in ("perception_mode", "suite", "task_id")):
            return False
        axes = ("scene_id", "object_set_id", "camera_config_id", "task_variant_id")
        if not all(isinstance(context.get(axis), str) and context[axis] for axis in axes):
            return False
        if (
            not isinstance(context.get("camera_names"), list)
            or not context["camera_names"]
            or any(not isinstance(name, str) or not name for name in context["camera_names"])
            or not isinstance(context.get("task_prompt"), str)
            or not context["task_prompt"].strip()
        ):
            return False
        variation = {axis: context[axis] for axis in axes}
        if plan is None or not any(
            all(case[axis] == variation[axis] for axis in axes)
            and case["camera_names"] == context["camera_names"]
            and case["task_prompt"] == context["task_prompt"]
            for case in plan["cases"]
        ):
            return False
        scope = self.impact_report(item.memory_id)["validated_scope"]
        if variation in scope:
            self.verify_source_evidence(item.memory_id)
            return True
        return False

    def verify_use(self, use: MemoryUseRecord) -> None:
        """Require the recorded outcome to agree with the raw policy trace."""
        trial = self._trial(use.attempt_id, allow_terminal_failure=True)
        if trial["run_id"] != use.run_id:
            raise StateConflictError("memory use run does not match its attempt")
        if trial["exposure"] != "trusted" or use.memory_id not in trial["memory_ids"]:
            raise StateConflictError("memory use lacks an explicit trusted retrieval event")
        if trial["memory_ids"].count(use.memory_id) != 1 or trial["retrieval_versions"][
            trial["memory_ids"].index(use.memory_id)
        ] != use.memory_version:
            raise StateConflictError("memory use version disagrees with retrieval trace")
        if trial["retrieval_timestamps"][trial["memory_ids"].index(use.memory_id)] != use.used_at:
            raise StateConflictError("memory use time disagrees with retrieval trace")
        grants = trial.get("retrieval_grants", [])
        grant_id = grants[trial["memory_ids"].index(use.memory_id)] if len(grants) == len(trial["memory_ids"]) else None
        grant = self.get_use_grant(grant_id) if isinstance(grant_id, str) else None
        if (
            grant is None or grant["memory_id"] != use.memory_id
            or grant["version"] != use.memory_version
            or grant["attempt_id"] != use.attempt_id
            or grant["used_at"] != use.used_at
        ):
            raise StateConflictError("memory use lacks a matching Service-issued grant")
        context = grant.get("context")
        if (
            not isinstance(context, dict)
            or context != trial["memory_context"]
            or context.get("suite") != trial["run"]["suite"]
            or context.get("task_id") != trial["run"]["task_id"]
            or context.get("perception_mode") != trial["mode"]
        ):
            raise StateConflictError("memory use grant context disagrees with attempt")
        expected = (
            "timeout" if trial["attempt_status"] == "timed_out"
            else "cancelled" if trial["attempt_status"] == "cancelled"
            else "evaluator_unavailable" if not trial["evaluator_authoritative"]
            else "native_success" if trial["success"] else "failed"
        )
        if use.outcome != expected:
            raise StateConflictError("memory use outcome disagrees with native evaluator")

    def record_pair(
        self,
        *,
        memory_id: str,
        control_attempt_id: str,
        treatment_attempt_id: str,
        control_safety: Path,
        treatment_safety: Path,
    ) -> dict[str, Any]:
        """Register a dev-seed pair; outcomes and costs are read from the store.

        Safety files must be independent monitor outputs with an integer
        ``unsafe_attempts`` field. They are hashed here and rechecked at gate.
        """
        closed = self.get_decision(memory_id)
        if closed and 'failed_check' in closed:
            raise StateConflictError('validation batch is closed')
        if is_provisional(self._provenance(memory_id)):
            raise StateConflictError('provisional notes cannot enter validation or impact results')
        provenance = self.provenance(memory_id)
        self.verify_source_evidence(memory_id)
        control = self._trial(control_attempt_id)
        treatment = self._trial(treatment_attempt_id)
        for trial in (control, treatment):
            require_profile_match(provenance["artifact"]["applicability"],
                                  trial.get("context") or trial.get("memory_context") or {})
        if provenance.get("channel") == "dev_v2":
            from .dev_v2 import SCOPE, AXES
            for trial in (control, treatment):
                if trial.get('validation_memory_id') != memory_id:
                    raise StateConflictError('pair arm grant belongs to another memory')
                if any(trial.get("context", {}).get(k) != provenance["artifact"]["applicability"][k]
                       for k in SCOPE if k not in AXES):
                    raise StateConflictError("Dev v2 validation exact SDK/namespace/robot scope mismatch")
        for field in ("suite", "task_id", "seed", "policy_id", "execution_target", "developer_model"):
            if control["run"][field] != treatment["run"][field]:
                raise StateConflictError(f"paired runs differ on {field}")
        seed = control["run"]["seed"]
        if classify_seed(seed) != "dev":
            raise PermissionError("memory validation must use development seeds")
        if control["run_id"] == treatment["run_id"]:
            raise StateConflictError("control and treatment must be different runs")
        if control["mode"] != treatment["mode"] or control["mode"] != provenance["artifact"]["applicability"]["perception_mode"]:
            raise StateConflictError("paired perception mode mismatch")
        if control["run"]["suite"] != provenance["artifact"]["applicability"]["suite"] or control["run"]["task_id"] != provenance["artifact"]["applicability"]["task_id"]:
            raise StateConflictError("paired task mismatch")
        if control["run"]["budget"] != treatment["run"]["budget"]:
            raise StateConflictError("paired budgets differ")
        plan = self.get_plan(memory_id)
        if plan is None:
            raise StateConflictError("paired validation requires a frozen variation plan")
        expected_case = next((case for case in plan["cases"] if case["seed"] == seed), None)
        expected_variation = None if expected_case is None else {
            axis: expected_case[axis]
            for axis in ("scene_id", "object_set_id", "camera_config_id", "task_variant_id", "camera_names", "task_prompt")
        }
        if expected_variation is None or control["variation"] != expected_variation or treatment["variation"] != expected_variation:
            raise StateConflictError("paired runs did not attest the planned variation")
        if (
            seed not in plan["seeds"]
            or control["run"]["policy_id"] != plan["policy_id"]
            or control["run"]["budget"]["assistance_credits"] != plan["assistance_credits"]
        ):
            raise StateConflictError("paired run differs from recorded validation plan")
        generated = plan.get('arm_generation') == 'dev_from_origin'
        if generated and any(t.get('origin_source_sha256') != plan['origin_source_sha256']
                             or t.get('dev_model') != plan['dev_model'] for t in (control, treatment)):
            raise StateConflictError('paired generation origin/model differs from plan')
        if not control["policy_sha256"] or (not generated and control["policy_sha256"] != treatment["policy_sha256"]):
            raise StateConflictError("paired policy code differs")
        if not control["config_sha256"] or control["config_sha256"] != treatment["config_sha256"]:
            raise StateConflictError("paired execution configuration differs")
        if control["memory_ids"] or control["exposure"] != "none":
            raise StateConflictError("control must not receive memory")
        if treatment["memory_ids"] != [memory_id] or treatment["exposure"] != "candidate_validation":
            raise StateConflictError("treatment trace must attest use of this candidate only")
        safety = [
            self._safety(control_safety, control_attempt_id),
            self._safety(treatment_safety, treatment_attempt_id),
        ]
        pair_id = f"pair:{memory_id}:{seed}"
        payload = {
            "schema_version": "attentionbench.memory-pair.v2",
            "pair_id": pair_id,
            "memory_id": memory_id,
            "seed": seed,
            "control_attempt_id": control_attempt_id,
            "treatment_attempt_id": treatment_attempt_id,
            "control_safety": safety[0],
            "treatment_safety": safety[1],
        }
        if plan.get('profile', 'default') == 'single_scene_v1':
            match = next((p for p in plan['pairs']
                          if p['control_cell'] == control.get('validation_cell')
                          and p['treatment_cell'] == treatment.get('validation_cell')), None)
            if match is None:
                raise StateConflictError('arms do not match a frozen single-scene pair')
            pair_id = f"pair:{memory_id}:{match['pair_id']}"
            payload['pair_id'] = pair_id
            payload['profile'] = 'single_scene_v1'
            with sqlite3.connect(self.store.path) as db:
                try:
                    db.execute('INSERT INTO memory_v2_scene_pairs(pair_id,memory_id,payload) VALUES (?,?,?)',
                               (pair_id, memory_id, _encoded(payload)))
                except sqlite3.IntegrityError as exc:
                    row = db.execute('SELECT payload FROM memory_v2_scene_pairs WHERE pair_id=?', (pair_id,)).fetchone()
                    if row is None or json.loads(row[0]) != payload:
                        raise StateConflictError('conflicting single-scene pair') from exc
            return payload
        with sqlite3.connect(self.store.path) as db:
            try:
                db.execute(
                    "INSERT INTO memory_v2_pairs(pair_id, memory_id, seed, payload) "
                    "VALUES (?, ?, ?, ?)",
                    (pair_id, memory_id, seed, _encoded(payload)),
                )
            except sqlite3.IntegrityError as exc:
                row = db.execute("SELECT payload FROM memory_v2_pairs WHERE pair_id=?", (pair_id,)).fetchone()
                if row is None or json.loads(row[0]) != payload:
                    raise StateConflictError("conflicting validation pair or reused seed") from exc
        return payload

    def impact_report(self, memory_id: str) -> dict[str, Any]:
        if is_provisional(self._provenance(memory_id)):
            raise StateConflictError('provisional notes cannot enter validation or impact results')
        provenance = self.provenance(memory_id)
        if is_test_seeded(provenance):
            raise StateConflictError('test-seeded notes are excluded from impact reports')
        plan = self.get_plan(memory_id)
        single = plan is not None and plan.get('profile', 'default') == 'single_scene_v1'
        generated = single and plan.get('arm_generation') == 'dev_from_origin'
        pairs = self.list_pairs(memory_id)
        control_success = treatment_success = control_credits = treatment_credits = 0
        control_seconds = treatment_seconds = control_unsafe = treatment_unsafe = 0.0
        control_gpu = treatment_gpu = control_tokens = treatment_tokens = 0.0
        both_success_credits_saved = failed_pair_credits_saved = 0
        supporting_successes = []
        counterexamples = []
        safety_regressions = []
        validation_policy_hashes: set[str] = set()
        signature_arms = []
        validation_config_hashes: set[str] = set()
        source_drifts=[]
        generated_policy_hashes, origins, models = [], set(), set()
        for pair in pairs:
            control = self._trial(pair["control_attempt_id"])
            treatment = self._trial(pair["treatment_attempt_id"])
            for trial in (control,treatment):
                if trial.get('source_drift') and trial['source_drift'] not in source_drifts:source_drifts.append(trial['source_drift'])
            if provenance.get('channel') == 'dev_v2' and any(
                    trial.get('validation_memory_id') != memory_id for trial in (control, treatment)):
                raise StateConflictError('pair arm grant belongs to another memory')
            if not control["policy_sha256"] or (not generated and control["policy_sha256"] != treatment["policy_sha256"]):
                raise StateConflictError("paired policy code differs")
            validation_policy_hashes.add(control["policy_sha256"])
            if generated:
                validation_policy_hashes.add(treatment['policy_sha256'])
                generated_policy_hashes.append(dict(pair_id=pair['pair_id'],
                    control=control['policy_sha256'], treatment=treatment['policy_sha256']))
                origins.update(t.get('origin_source_sha256') for t in (control, treatment))
                models.update(_encoded(t.get('dev_model')) for t in (control, treatment))
            if single:
                validation_config_hashes.update((control['config_sha256'], treatment['config_sha256']))
                signature_arms.append({
                    'pair_id': pair['pair_id'],
                    'control_signature_present': control['signature_present'],
                    'treatment_signature_present': treatment['signature_present'],
                    'control_stop_position': control['stop_position'],
                    'treatment_stop_position': treatment['stop_position'],
                    'treatment_not_earlier': treatment['stop_position'] >= control['stop_position'],
                })
            if control["run"]["seed"] != pair["seed"] or treatment["run"]["seed"] != pair["seed"]:
                raise StateConflictError("validation seed changed")
            if control["memory_ids"] or control["exposure"] != "none" or treatment["memory_ids"] != [memory_id] or treatment["exposure"] != "candidate_validation":
                raise StateConflictError("validation exposure changed")
            control_success += int(control["success"])
            treatment_success += int(treatment["success"])
            control_credits += control["credits"]
            treatment_credits += treatment["credits"]
            if control["success"] and treatment["success"]:
                both_success_credits_saved += control["credits"] - treatment["credits"]
            elif not control["success"] and not treatment["success"]:
                failed_pair_credits_saved += control["credits"] - treatment["credits"]
            control_seconds += control["seconds"]
            treatment_seconds += treatment["seconds"]
            control_gpu += control["gpu_seconds"]
            treatment_gpu += treatment["gpu_seconds"]
            control_tokens += control["tokens"]
            treatment_tokens += treatment["tokens"]
            control_pair_unsafe = self._verify_safety(pair["control_safety"])
            treatment_pair_unsafe = self._verify_safety(pair["treatment_safety"])
            control_unsafe += control_pair_unsafe
            treatment_unsafe += treatment_pair_unsafe
            evidence = {
                "seed": pair["seed"],
                "variation": treatment["variation"],
                "control_attempt_id": pair["control_attempt_id"],
                "treatment_attempt_id": pair["treatment_attempt_id"],
            }
            if treatment_pair_unsafe > control_pair_unsafe:
                safety_regressions.append({
                    **evidence,
                    "control_unsafe_attempts": control_pair_unsafe,
                    "treatment_unsafe_attempts": treatment_pair_unsafe,
                })
            if treatment["success"] and (
                not control["success"] or treatment["credits"] < control["credits"]
            ):
                supporting_successes.append({
                    **evidence,
                    "reason": "success_gain" if not control["success"] else "assistance_saved",
                    "assistance_credits_saved": control["credits"] - treatment["credits"],
                })
            if (
                not treatment["success"]
                or treatment_pair_unsafe > control_pair_unsafe
                or treatment["credits"] > control["credits"]
            ):
                counterexamples.append({
                    **evidence,
                    "reason": (
                        "treatment_failed" if not treatment["success"]
                        else "safety_regression" if treatment_pair_unsafe > control_pair_unsafe
                        else "assistance_regression"
                    ),
                    "control_native_success": bool(control["success"]),
                    "treatment_native_success": bool(treatment["success"]),
                    "control_unsafe_attempts": control_pair_unsafe,
                    "treatment_unsafe_attempts": treatment_pair_unsafe,
                })
        n = len(pairs)
        axes = ("scene_id", "object_set_id", "camera_config_id", "task_variant_id")
        scope_tuple = lambda variation: {axis: variation[axis] for axis in axes}
        blocked = {_encoded(scope_tuple(item["variation"])) for item in counterexamples}
        validated_scope = []
        seen_scope = set()
        for item in supporting_successes:
            variation = scope_tuple(item["variation"])
            key = _encoded(variation)
            if key not in blocked and key not in seen_scope:
                validated_scope.append(variation)
                seen_scope.add(key)
        report = {
            "memory_id": memory_id,
            "source_kind": provenance["source_kind"],
            "source_policy_sha256": provenance["source_policy_sha256"],
            "validation_policy_hashes": sorted(validation_policy_hashes),
            "policy_changed_since_source": (
                next(iter(validation_policy_hashes)) != provenance["source_policy_sha256"]
                if len(validation_policy_hashes) == 1 else None
            ),
            "source_human_attention_seconds": provenance["human_attention_seconds"],
            "paired_dev_seeds": n,
            "control_successes": control_success,
            "treatment_successes": treatment_success,
            "empirical_confidence": treatment_success / n if n else None,
            "treatment_success_rate_95ci": _wilson_95(treatment_success, n),
            "supporting_successes": supporting_successes,
            "counterexamples": counterexamples,
            "safety_regressions": safety_regressions,
            "validated_scope": validated_scope,
            "success_gain": treatment_success - control_success,
            "success_rate_gain": (treatment_success - control_success) / n if n else None,
            "future_assistance_credits_saved": control_credits - treatment_credits,
            "both_success_assistance_credits_saved": both_success_credits_saved,
            "both_failed_assistance_credits_saved": failed_pair_credits_saved,
            "robot_seconds_saved": control_seconds - treatment_seconds,
            "gpu_seconds_saved": control_gpu - treatment_gpu,
            "tokens_saved": int(control_tokens - treatment_tokens),
            "control_unsafe_attempts": int(control_unsafe),
            "treatment_unsafe_attempts": int(treatment_unsafe),
            "human_seconds_per_success_gain": (
                provenance["human_attention_seconds"] / (treatment_success - control_success)
                if provenance["source_kind"] in {"human", "curated_human_packet"} and treatment_success > control_success
                else None
            ),
        }
        if single:
            report.update(profile='single_scene_v1', cross_scene_eligible=False,
                          failure_signature=provenance['failure_signature'],
                          paired_runs=n, signature_arms=signature_arms,
                          control_signature_count=sum(a['control_signature_present'] for a in signature_arms),
                          treatment_signature_count=sum(a['treatment_signature_present'] for a in signature_arms),
                          source_drift=source_drifts,validation_config_hashes=sorted(validation_config_hashes),
                          validated_scope=[{axis: plan['cases'][0][axis] for axis in axes}])
        if generated:
            report.update(arm_generation='dev_from_origin', generated_policy_hashes=generated_policy_hashes,
                          validation_origin_hashes=sorted(origins),
                          validation_dev_models=[json.loads(m) for m in sorted(models)])
        return report

    def list_pairs(self, memory_id: str) -> list[dict[str, Any]]:
        self.provenance(memory_id)
        plan = self.get_plan(memory_id)
        if plan and plan.get('profile', 'default') == 'single_scene_v1':
            with sqlite3.connect(self.store.path) as db:
                rows = db.execute('SELECT payload FROM memory_v2_scene_pairs WHERE memory_id=? ORDER BY pair_id',
                                  (memory_id,)).fetchall()
            return [json.loads(row[0]) for row in rows]
        with sqlite3.connect(self.store.path) as db:
            rows = db.execute(
                "SELECT payload FROM memory_v2_pairs WHERE memory_id=? ORDER BY seed",
                (memory_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    GATE_MESSAGES = {
        "insufficient_pairs": "at least five distinct paired dev seeds are required",
        "mixed_policy_versions": "validation cases used different policy code versions",
        "safety_regression": "paired safety regression blocks memory promotion",
        "success_regression": "success regression blocks memory promotion",
        "no_measured_gain": "no measured success gain or successful-pair assistance saving",
        "insufficient_treatment_successes": "insufficient treatment successes",
        "treatment_fraction_below_0.75": "treatment success fraction is below 0.75",
        "no_validated_scope": "no validated retrieval scope remains after counterexamples",
        'control_signature_not_reproduced': 'target failure did not reproduce in at least 60% of controls',
        'treatment_signature_remains': 'target failure remains in more than 20% of treatments',
        'treatment_stopped_earlier': 'a treatment stopped before its paired control',
        'mixed_validation_configuration': 'single-scene arms used different frozen configurations',
    }

    def _single_scene_failures(self, report, plan):
        n = report['paired_runs']
        failures = []
        if n < 5:
            failures.append('insufficient_pairs')
        if plan.get('arm_generation') == 'dev_from_origin':
            mixed = (report['validation_origin_hashes'] != [plan['origin_source_sha256']]
                     or report['validation_dev_models'] != [plan['dev_model']])
        else:
            mixed = report['validation_policy_hashes'] != [plan['validation_policy_sha256']]
        if mixed:
            failures.append('mixed_policy_versions')
        if report['validation_config_hashes'] != [plan['validation_config_sha256']]:
            failures.append('mixed_validation_configuration')
        if report['safety_regressions'] or report['treatment_unsafe_attempts'] > report['control_unsafe_attempts']:
            failures.append('safety_regression')
        if report['control_signature_count'] < math.ceil(0.6 * n):
            failures.append('control_signature_not_reproduced')
        if report['treatment_signature_count'] > math.floor(0.2 * n):
            failures.append('treatment_signature_remains')
        if any(not arm['treatment_not_earlier'] for arm in report['signature_arms']):
            failures.append('treatment_stopped_earlier')
        return failures

    def _gate_failures(self, report: dict[str, Any], *, minimum_pairs: int) -> list[str]:
        """Failed promotion gates, in the order they have always been checked."""
        failures = []
        if report["paired_dev_seeds"] < minimum_pairs or minimum_pairs < 5:
            failures.append("insufficient_pairs")
        if len(report["validation_policy_hashes"]) != 1:
            failures.append("mixed_policy_versions")
        if report["safety_regressions"] or report["treatment_unsafe_attempts"] > report["control_unsafe_attempts"]:
            failures.append("safety_regression")
        if report["success_gain"] < 0:
            failures.append("success_regression")
        if report["success_gain"] == 0 and report["both_success_assistance_credits_saved"] <= 0:
            failures.append("no_measured_gain")
        if report["treatment_successes"] < 3:
            failures.append("insufficient_treatment_successes")
        if report["paired_dev_seeds"] and report["treatment_successes"] / report["paired_dev_seeds"] < 0.75:
            failures.append("treatment_fraction_below_0.75")
        if not report["validated_scope"]:
            failures.append("no_validated_scope")
        return failures

    def _plan_complete(self, memory_id: str) -> dict[str, Any]:
        decision = self.get_decision(memory_id)
        if decision and 'failed_check' in decision:
            raise StateConflictError('validation batch is closed')
        plan = self.get_plan(memory_id)
        if plan and plan.get('profile', 'default') == 'single_scene_v1':
            expected = {f"pair:{memory_id}:{p['pair_id']}" for p in plan['pairs']}
            if {p['pair_id'] for p in self.list_pairs(memory_id)} != expected:
                raise StateConflictError('all predefined single-scene pairs must be completed')
            return plan
        if plan is None or {pair["seed"] for pair in self.list_pairs(memory_id)} != set(plan["seeds"]):
            raise StateConflictError("all predefined variation cases must be completed")
        return plan

    def validate_and_promote(self, memory_id: str, *, minimum_pairs: int = 5) -> MemoryRecord:
        if is_provisional(self._provenance(memory_id)):
            raise StateConflictError('provisional notes cannot enter validation or impact results')
        report = self.impact_report(memory_id)
        self.verify_source_evidence(memory_id)
        plan = self._plan_complete(memory_id)
        memory = self.store.get_memory(memory_id)
        if memory is None or memory.status not in {
            MemoryStatus.CANDIDATE, MemoryStatus.VALIDATED, MemoryStatus.TRUSTED,
        }:
            raise StateConflictError("memory is not eligible for v2 validation")
        single = plan.get('profile', 'default') == 'single_scene_v1'
        failures = self._single_scene_failures(report, plan) if single else self._gate_failures(report, minimum_pairs=minimum_pairs)
        if failures:
            raise StateConflictError(self.GATE_MESSAGES[failures[0]])
        if memory.status is MemoryStatus.TRUSTED:
            return memory
        manager = MemoryManager(self.store)
        if single:
            if memory.status is MemoryStatus.CANDIDATE:
                self.store.transition_memory(memory_id, MemoryStatus.VALIDATED,
                    event_key=f'memory-validation:{memory_id}:single_scene_v1',
                    confidence=report['control_signature_count'] / report['paired_runs'],
                    validation_successes=report['paired_runs'] - report['treatment_signature_count'],
                    validation_failures=report['treatment_signature_count'],
                    reason='single_scene_v1_validation_passed')
            return manager.promote(memory_id)
        if memory.status is MemoryStatus.CANDIDATE:
            manager.record_validation(
                memory_id,
                successes=report["treatment_successes"],
                failures=report["paired_dev_seeds"] - report["treatment_successes"],
            )
        updated = self.store.get_memory(memory_id)
        if updated.status is not MemoryStatus.VALIDATED:
            raise StateConflictError("core validation confidence gate failed")
        return manager.promote(memory_id)

    def finalize_validation(self, memory_id: str, *, minimum_pairs: int = 5) -> MemoryRecord:
        """Close a completed validation: promote to trusted, or reject with the failed gates.

        The decision, its gates and the full impact report are kept so a rejected
        candidate stays explainable. An incomplete plan is not a verdict.
        """
        if is_provisional(self._provenance(memory_id)):
            raise StateConflictError('provisional notes cannot enter validation or impact results')
        if is_test_seeded(self._provenance(memory_id)):
            raise StateConflictError('test-seeded notes cannot be evaluated')
        memory = self.store.get_memory(memory_id)
        if memory is None:
            raise StateConflictError("unknown memory")
        closed = self.get_decision(memory_id)
        if closed and 'failed_check' in closed:
            return memory
        if memory.status in {MemoryStatus.TRUSTED, MemoryStatus.REJECTED}:
            return memory
        if memory.status not in {MemoryStatus.CANDIDATE, MemoryStatus.VALIDATED}:
            raise StateConflictError("memory is not eligible for v2 validation")
        self.verify_source_evidence(memory_id)
        plan = self._plan_complete(memory_id)
        report = self.impact_report(memory_id)
        single = plan.get('profile', 'default') == 'single_scene_v1'
        failures = self._single_scene_failures(report, plan) if single else self._gate_failures(report, minimum_pairs=minimum_pairs)
        decision = {
            "schema_version": "attentionbench.memory-validation-decision.v1",
            "memory_id": memory_id, "profile": 'single_scene_v1' if single else 'default',
            "decision": "rejected" if failures else "promoted",
            "failed_gates": failures,
            "failed_gate_messages": [self.GATE_MESSAGES[code] for code in failures],
            "minimum_pairs": minimum_pairs, "report": report, "decided_at": time.time(),
        }
        if report.get('source_drift'):decision['source_drift']=report['source_drift']
        if single:
            decision.update(trust_scope='scene_scoped' if not failures else None,
                            cross_scene_eligible=False,
                            gate_thresholds={'minimum_pairs': 5, 'control_signature_fraction_min': 0.6,
                                             'treatment_signature_fraction_max': 0.2,
                                             'stop_position': 'each treatment >= paired control',
                                             'safety': 'no per-pair or net regression'})
            return self._finalize_single_scene(memory_id, decision)
        if not failures:
            promoted = self.validate_and_promote(memory_id, minimum_pairs=minimum_pairs)
            self._record_decision(memory_id, decision)
            return promoted
        rejected = self.store.transition_memory(
            memory_id, MemoryStatus.REJECTED, event_key=f"memory-reject:{memory_id}",
            confidence=report["empirical_confidence"] or 0.0,
            validation_successes=report["treatment_successes"],
            validation_failures=report["paired_dev_seeds"] - report["treatment_successes"],
            reason="validation_gate_failed: " + ", ".join(failures),
        )
        self._record_decision(memory_id, decision)
        return rejected

    def record_failed_check(self, *, memory_id, validation_grant_id, execution_import_id, catalog_id, descriptor):
        """Close an inadmissible arm without storing a trial or permitting promotion."""
        from .dev_v2 import DevV2Evidence, require
        evidence = DevV2Evidence(self.store)
        grant = evidence.validation_grant(validation_grant_id)
        require(grant['binding']['memory_id'] == memory_id, 'failed check grant belongs to another memory')
        execution, ed, bundle, artifacts, _, verdict, _ = evidence.verdict(
            execution_import_id, catalog_id, descriptor, for_validation=False)
        checks = verdict['checks']
        require(not checks['safety'] or not checks['scene_reproduction'], 'all checks passed; use normal trial path')
        _, plan, _, _ = evidence._verify_validation_arm(execution, ed, bundle, artifacts, grant,
                                                       grant['binding']['exposure'], memory_id)
        require(plan.get('profile') == 'single_scene_v1', 'failed check closure requires single_scene_v1')
        self.verify_source_evidence(memory_id)
        for ref in verdict.get('scene_replay_states', []):
            evidence.ref(catalog_id, ref)
        failed = dict(validation_grant_id=validation_grant_id, execution_import_id=execution_import_id,
                      catalog_id=catalog_id, descriptor=descriptor,
                      descriptor_sha256=evidence.catalog(catalog_id)['files'][descriptor], checks=checks)
        old = self.get_decision(memory_id)
        if old:
            require(old.get('failed_check') == failed, 'conflicting closed validation decision')
            return old
        scene_failed = not checks['scene_reproduction']
        rejected = not scene_failed and grant['binding']['exposure'] == 'candidate_validation'
        gate = 'safety_regression' if rejected else 'scene_reproduction_failed' if scene_failed else 'control_safety_failed'
        reason = ('Rejected: a run with the note failed the independent safety check' if rejected else
                  'Could not finish the check: ' + ('the scene did not reproduce' if scene_failed else
                                                   'a run without the note failed the safety check'))
        report = self.impact_report(memory_id)  # Contains only previously accepted pairs.
        with self.store._transaction() as db:
            row = db.execute('SELECT payload FROM memory_v2_decisions WHERE memory_id=?', (memory_id,)).fetchone()
            if row:
                old = json.loads(row[0])
                require(old.get('failed_check') == failed, 'conflicting closed validation decision')
                return old
            memory = self.store._decode_memory(json.loads(db.execute(
                'SELECT payload FROM memories WHERE id=?', (memory_id,)).fetchone()[0]))
            require(memory.status is MemoryStatus.CANDIDATE and memory.version == grant['binding']['memory_version'],
                    'failed check candidate changed or is no longer eligible')
            decision = dict(schema_version='attentionbench.memory-validation-decision.v1', memory_id=memory_id,
                profile='single_scene_v1', decision='rejected' if rejected else 'inconclusive',
                failed_gates=[gate], failed_gate_messages=[reason], reason=reason, failed_check=failed,
                minimum_pairs=5, report=report, decided_at=time.time(), trust_scope=None, cross_scene_eligible=False)
            if verdict.get('source_drift'):decision['source_drift']=[verdict['source_drift']]
            if rejected:
                memory = self.store._transition_memory_tx(db, memory_id, MemoryStatus.REJECTED,
                    event_key=f'memory-reject:{memory_id}', reason=reason, actor='system:validation')
            self._record_decision_tx(db, memory, decision)
        return decision

    def _finalize_single_scene(self, memory_id, decision):
        """Commit status, decision, notices and the source note's retirement together."""
        report = decision['report']
        with self.store._transaction() as db:
            memory = self.store._decode_memory(json.loads(db.execute(
                'SELECT payload FROM memories WHERE id=?', (memory_id,)).fetchone()[0]))
            if db.execute('SELECT 1 FROM memory_v2_decisions WHERE memory_id=?', (memory_id,)).fetchone():
                return memory
            if memory.status not in {MemoryStatus.CANDIDATE, MemoryStatus.VALIDATED}:
                raise StateConflictError('memory is not eligible for v2 validation')
            transition = self.store._transition_memory_tx
            if decision['decision'] == 'promoted':
                if memory.status is MemoryStatus.CANDIDATE:
                    transition(db, memory_id, MemoryStatus.VALIDATED,
                        event_key=f'memory-validation:{memory_id}:single_scene_v1',
                        confidence=report['control_signature_count'] / report['paired_runs'],
                        validation_successes=report['paired_runs'] - report['treatment_signature_count'],
                        validation_failures=report['treatment_signature_count'],
                        reason='single_scene_v1_validation_passed')
                memory = transition(db, memory_id, MemoryStatus.TRUSTED,
                    event_key=f'memory-promote:{memory_id}', reason='validated_memory_promoted')
            else:
                memory = transition(db, memory_id, MemoryStatus.REJECTED,
                    event_key=f'memory-reject:{memory_id}', confidence=report['empirical_confidence'] or 0.0,
                    validation_successes=report['treatment_successes'],
                    validation_failures=report['paired_dev_seeds'] - report['treatment_successes'],
                    reason='validation_gate_failed: ' + ', '.join(decision['failed_gates']))
            self._record_decision_tx(db, memory, decision)
        return memory

    def _record_decision_tx(self, db, memory, decision):
        from .auto_validation import lifecycle, validation_reason
        memory_id = memory.memory_id
        db.execute('INSERT INTO memory_v2_decisions(memory_id,payload) VALUES (?,?)',
                   (memory_id, _encoded(decision)))
        provenance = json.loads(db.execute('SELECT payload FROM memory_v2_provenance WHERE memory_id=?',
                                           (memory_id,)).fetchone()[0])
        lifecycle(db, memory.artifact(), provenance, decision['decision'],
                  validation_reason(decision), 'system:validation', at=decision['decided_at'])
        linked_id = provenance.get('from_provisional')
        if provenance.get('channel') != 'dev_v2' or linked_id is None:
            return
        linked = db.execute('SELECT payload FROM memory_v2_provenance WHERE memory_id=?', (linked_id,)).fetchone()
        prior = json.loads(linked[0]) if linked else None
        if not is_provisional(prior):
            raise StateConflictError('validation source must remain a provisional note')
        note = self.store._decode_memory(json.loads(db.execute(
            'SELECT payload FROM memories WHERE id=?', (linked_id,)).fetchone()[0]))
        if decision['decision'] == 'inconclusive':
            prior['decision_link'] = dict(memory_id=memory_id, decision='inconclusive',
                                          reason=decision['reason'], decided_at=decision['decided_at'])
            db.execute('UPDATE memory_v2_provenance SET payload=? WHERE memory_id=?', (_encoded(prior), linked_id))
            lifecycle(db, note.artifact(), prior, 'inconclusive', decision['reason'], 'system:validation',
                      at=decision['decided_at'], decision_link=prior['decision_link'])
            return
        event = 'superseded' if decision['decision'] == 'promoted' else 'rejected'
        reason = (f'replaced by its checked version {memory_id}' if event == 'superseded'
                  else 'rejected after automatic check')
        if note.status is MemoryStatus.PROVISIONAL:
            note = self.store._transition_memory_tx(db, linked_id, MemoryStatus.DISABLED,
                event_key=f'memory-check-disable:{linked_id}:{memory_id}',
                reason=reason, actor='system:validation')
        elif note.status not in {MemoryStatus.DISABLED, MemoryStatus.EXPIRED, MemoryStatus.ROLLED_BACK}:
            raise StateConflictError('validation source is no longer provisional or off')
        # Preserve an existing operator revocation. The public link names only
        # the checked candidate and outcome, never its private evidence.
        prior['decision_link'] = dict(memory_id=memory_id, decision=decision['decision'],
                                      reason=reason, decided_at=decision['decided_at'])
        db.execute('UPDATE memory_v2_provenance SET payload=? WHERE memory_id=?', (_encoded(prior), linked_id))
        lifecycle(db, note.artifact(), prior, event, reason, 'system:validation',
                  at=decision['decided_at'], decision_link=prior['decision_link'])

    def _record_decision(self, memory_id: str, decision: dict[str, Any]) -> None:
        with sqlite3.connect(self.store.path) as db:
            db.execute("INSERT OR IGNORE INTO memory_v2_decisions(memory_id, payload) VALUES (?, ?)",
                       (memory_id, _encoded(decision)))

    def get_decision(self, memory_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.store.path) as db:
            row = db.execute("SELECT payload FROM memory_v2_decisions WHERE memory_id=?",
                             (memory_id,)).fetchone()
        return None if row is None else json.loads(row[0])

    def verify_source_evidence(self, memory_id: str) -> None:
        """Fail closed if the candidate's original artifact bytes are gone."""
        if is_provisional(self._provenance(memory_id)):
            raise StateConflictError('provisional notes cannot enter validation or impact results')
        provenance = self.provenance(memory_id)
        if is_test_seeded(provenance):
            raise StateConflictError('test-seeded notes are excluded from validation evidence')
        if provenance.get("channel") == "dev_v2":
            # provenance() reopens the sealed ledger and all original receipts.
            return
        refs = provenance["raw_evidence_refs"]
        if not refs:
            raise StateConflictError("memory has no raw source evidence")
        for ref in refs:
            uri = ref["uri"]
            if not isinstance(uri, str) or not uri.startswith("artifact://"):
                raise StateConflictError("unsupported raw evidence URI")
            relative = Path(uri.removeprefix("artifact://"))
            path = (self.artifact_root / relative).resolve()
            if not path.is_relative_to(self.artifact_root) or not path.is_file():
                raise StateConflictError("raw source evidence is missing or outside artifact root")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != ref["sha256"]:
                raise StateConflictError("raw source evidence digest changed")

    def _trial(self, attempt_id: str, *, allow_terminal_failure: bool = False) -> dict[str, Any]:
        from .dev_v2 import DevV2Evidence
        imported = DevV2Evidence(self.store).trial(attempt_id)
        if imported is not None:
            return imported
        attempt = self.store.get_attempt(attempt_id)
        allowed = {"succeeded", "failed"}
        if allow_terminal_failure:
            allowed.update({"timed_out", "cancelled"})
        if attempt is None or attempt["status"] not in allowed or attempt["native_success"] is None:
            raise StateConflictError("trial attempt lacks completed native outcome")
        run = self.store.get_run(attempt["run_id"])
        if run is None:
            raise StateConflictError("trial run missing")
        raw = next((item for item in self.store._list_payloads("raw_traces") if item["attempt_id"] == attempt_id), None)
        if raw is None or raw["outcome"].get("native_success") != attempt["native_success"]:
            raise StateConflictError("trial lacks agreeing native trace outcome")
        if raw["outcome"].get("evaluator_authoritative") is not True:
            # Deadline, cancellation and evaluator failures remain memory-use
            # outcomes, but without a verdict they cannot validate a memory.
            if not allow_terminal_failure or attempt["status"] not in {"failed", "timed_out", "cancelled"}:
                raise StateConflictError("trial lacks an authoritative native evaluator verdict")
        runtime = raw["metadata"].get("runtime", {})
        ids = runtime.get("memory_ids", [])
        if not isinstance(ids, list):
            raise StateConflictError("invalid memory exposure trace")
        retrieval_events = [
            event
            for event in raw["events"]
            if event["event_type"] == "attention.memory_retrieval"
            and event["status"] == "completed"
        ]
        if ids != [event["arguments"].get("memory_id") for event in retrieval_events]:
            raise StateConflictError("memory exposure list disagrees with retrieval events")
        resources = self.store.resource_status(attempt["run_id"])
        return {
            "run_id": attempt["run_id"], "run": run, "success": attempt["native_success"],
            "attempt_status": attempt["status"],
            "evaluator_authoritative": raw["outcome"].get("evaluator_authoritative") is True,
            "mode": runtime.get("perception_mode"), "memory_ids": ids,
            "retrieval_versions": [event["result"].get("memory_version") for event in retrieval_events],
            "retrieval_timestamps": [event["timestamp"] for event in retrieval_events],
            "retrieval_grants": [event["result"].get("grant_id") for event in retrieval_events],
            "exposure": runtime.get("memory_exposure"),
            "memory_context": runtime.get("memory_context"),
            "policy_sha256": runtime.get("policy_sha256"),
            "config_sha256": runtime.get("validation_config_sha256"),
            "variation": runtime.get("validation_variation"),
            "credits": self.store.budget_status(attempt["run_id"])["used"],
            "seconds": float(raw["outcome"]["elapsed_seconds"]),
            "gpu_seconds": float(resources["gpu_seconds"]["used"]),
            "tokens": int(resources["tokens"]["used"]),
        }

    @staticmethod
    def _safety(path: Path, expected_attempt_id: str) -> dict[str, str]:
        path = path.resolve()
        data = path.read_bytes()
        value = json.loads(data)
        count = value.get("unsafe_attempts")
        if value.get("source") != "independent_safety_monitor":
            raise ValueError("safety evidence must come from independent monitor")
        if value.get("schema_version") != "attentionbench.safety-monitor.v1":
            raise ValueError("unsupported safety monitor artifact schema")
        events = value.get("events")
        violations = value.get("violations")
        if (
            not isinstance(events, list) or not isinstance(violations, list)
            or any(not isinstance(item, dict) or not isinstance(item.get("kind"), str)
                   for item in (*events, *violations))
            or not events and not violations
        ):
            raise ValueError("safety monitor artifact lacks event evidence")
        if (
            isinstance(count, bool) or not isinstance(count, int)
            or count != int(bool(violations))
        ):
            raise ValueError("safety count disagrees with monitor violations")
        if not isinstance(value.get("attempt_id"), str) or not value["attempt_id"]:
            raise ValueError("safety evidence must bind to an attempt")
        if value["attempt_id"] != expected_attempt_id:
            raise StateConflictError("safety evidence belongs to another attempt")
        return {"uri": str(path), "sha256": _digest(data), "attempt_id": value["attempt_id"]}

    @staticmethod
    def _verify_safety(reference: dict[str, str]) -> int:
        path = Path(reference["uri"])
        if _digest(path.read_bytes()) != reference["sha256"]:
            raise StateConflictError("safety evidence changed after registration")
        MemoryV2Manager._safety(path, reference["attempt_id"])
        return int(json.loads(path.read_text())["unsafe_attempts"])

    def _provenance(self, memory_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.store.path) as db:
            row = db.execute(
                "SELECT payload FROM memory_v2_provenance WHERE memory_id=?", (memory_id,)
            ).fetchone()
        return None if row is None else json.loads(row[0])


def validate_single_scene_plan(plan, context, seed, case, signature, origin_sha):
    from .dev_v2 import SCOPE, AXES, SHA
    pairs = plan.get('pairs')
    if (plan.get('failure_signature') != signature
            or plan.get('seeds') != [seed] or plan.get('cases') != [case]
            or not isinstance(pairs, list) or len(pairs) < 5
            or any(not isinstance(p, dict) or set(p) != {'pair_id', 'control_cell', 'treatment_cell'}
                   or any(not isinstance(v, str) or not v for v in p.values()) for p in pairs)
            or len({p['pair_id'] for p in pairs}) != len(pairs)
            or len({cell for p in pairs for cell in (p['control_cell'], p['treatment_cell'])}) != 2 * len(pairs)):
        raise ValueError('single-scene plan needs one exact source tuple and five distinct pairs')
    if classify_seed(seed) != 'dev':
        raise PermissionError('validation plan is development-only')
    if any(plan.get(k) != context[k] for k in SCOPE if k not in AXES):
        raise StateConflictError('single-scene plan differs from source scope')
    generated = plan.get('arm_generation') == 'dev_from_origin'
    policy_key = 'origin_source_sha256' if generated else 'validation_policy_sha256'
    if any(not isinstance(plan.get(k), str) or SHA.fullmatch(plan[k]) is None
           for k in (policy_key, 'validation_config_sha256')):
        raise ValueError('single-scene plan needs frozen policy and config SHA-256')
    if generated:
        if plan['origin_source_sha256'] != origin_sha:
            raise StateConflictError('validation origin differs from sealed note origin')
        model = plan.get('dev_model')
        if (not isinstance(model, dict) or set(model) != {'provider', 'model'}
                or any(not isinstance(v, str) or not v.strip() for v in model.values())):
            raise ValueError('dev_model requires provider and model identifiers')
    credits = plan.get('assistance_credits')
    if type(credits) is not int or credits < 0:
        raise ValueError('validation plan credits must be nonnegative')
