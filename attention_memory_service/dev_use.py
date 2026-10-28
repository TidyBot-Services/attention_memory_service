"""Service-owned evidence for trusted Memory shown to a development agent.

This is deliberately separate from attempt-bound runtime memory use.  A Dev
Agent may change source code after reading guidance, so its exposure cannot be
counted as a policy-attempt treatment or promotion result.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import uuid
from typing import Any
from pathlib import Path

from .core.models import MemoryStatus
from .memory_v2 import trusted_here
from .core.store import StateConflictError
from .seeded_notes import flags, is_test_seeded, is_provisional
from .applicability import profile_fields, require_profile_match, SAME_SCENE, SAME_TASK, SAME_FAMILY, layer_of, marker as level_marker


_SHA = re.compile(r"[0-9a-f]{64}\Z")


class DevMemoryUse:
    def __init__(self, service: Any) -> None:
        self.service = service
        with sqlite3.connect(service.store.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS memory_dev_exposures ("
                "grant_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, "
                "development_id TEXT NOT NULL, payload TEXT NOT NULL, "
                "UNIQUE(memory_id, development_id))"
            )

    def _find(self, memory_id, context, now):
        """Exact match first; otherwise a same-task reference, else nothing."""
        for item in self.service.retrieve(context, now=now):
            if item.memory_id == memory_id:
                return item, SAME_SCENE
        if 'sdk_sha256' not in context:
            return None, None  # Legacy contexts have no layered levels.
        for item, level in self.service.retrieve_levels(context, now=now, levels=(SAME_TASK, SAME_FAMILY)):
            if item.memory_id == memory_id:
                return item, level
        return None, None

    def authorize(
        self, *, memory_id: str, development_id: str, context: dict[str, Any],
        source_policy_sha256: str, now: float, memory_version: int | None = None,
    ) -> dict[str, Any]:
        if not isinstance(development_id, str) or not development_id.strip():
            raise ValueError("development_id is required")
        if not isinstance(source_policy_sha256, str) or not _SHA.fullmatch(source_policy_sha256):
            raise ValueError("source policy SHA-256 is required")
        if not isinstance(context, dict):
            raise ValueError("context must be an object")
        # The existing retrieval guard checks exact validated variation,
        # provenance, trusted status and expiry.  Recheck status atomically
        # before the guidance leaves the Service.
        # Dev exposure may read every level; explicit task authority also
        # applies at SAME_TASK. Anything not trusted here is excluded.
        memory, level = self._find(memory_id, context, now)
        if memory is None:
            note = self.service.store.get_memory(memory_id)
            if note is not None:
                require_profile_match(note.applicability, context)
            raise StateConflictError("memory is not trusted or applicable to Dev context")
        provenance = self.service.v2._provenance(memory_id)
        marker = flags(provenance)
        if provenance and provenance.get('channel') == 'dev_v2':
            if type(memory_version) is not int or memory_version != memory.version:
                raise StateConflictError('explicit exact trusted Memory version required')
        elif memory_version is not None and (type(memory_version) is not int or memory_version != memory.version):
            raise StateConflictError('requested Memory version changed')
        with sqlite3.connect(self.service.store.path) as db:
            prior = db.execute('SELECT payload FROM memory_dev_exposures WHERE memory_id=? AND development_id=?', (memory_id, development_id)).fetchone()
        if prior:
            old = json.loads(prior[0])
            if (old.get('result') or {}).get('schema_version') == 'attentionbench.dev-v2-memory-result.v1':
                self.service.dev_v2.verify_result(old)
        with sqlite3.connect(self.service.store.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM memories WHERE id=?", (memory_id,)).fetchone()
            if row is None:
                raise StateConflictError("unknown memory")
            current = json.loads(row[0])
            if (current["status"] != (MemoryStatus.PROVISIONAL.value if is_provisional(provenance) else MemoryStatus.TRUSTED.value)
                    or current["version"] != memory.version
                    or current.get("expires_at") is not None
                    and current["expires_at"] <= time.time()):
                raise StateConflictError("memory was revoked or expired before Dev exposure")
            require_profile_match(current.get('applicability', {}), context)
            marker.update(profile_fields(current.get('applicability', {})))
            # Scene and task authority share one trust predicate.
            trusted = current['status'] == MemoryStatus.TRUSTED.value and (
                level == SAME_SCENE or trusted_here(memory, level, provenance, context))
            marker = {**marker, **level_marker(level, trusted),
                      'source_task_id': current.get('applicability', context).get('task_id')}
            if provenance and provenance.get('channel') in {'dev_v2', 'test_seeded', 'dev_v2_provisional'}:
                marker.update(source_seed=provenance.get('source_seed'),
                              note_sdk_sha256=memory.applicability.get('sdk_sha256'))
            row = db.execute(
                "SELECT payload FROM memory_dev_exposures WHERE memory_id=? AND development_id=?",
                (memory_id, development_id),
            ).fetchone()
            if row is not None:
                old = json.loads(row[0])
                if (old["context"] != context or old["source_policy_sha256"] != source_policy_sha256
                        or old["memory_version"] != memory.version
                        or old.get("match_level", SAME_SCENE) != level):
                    raise StateConflictError("development ID already binds different Memory evidence")
                old.setdefault('source_task_id', current.get('applicability', context).get('task_id'))
                return _online_grant(old)
            record = {
                "schema_version": "attentionbench.dev-memory-exposure.v1",
                "grant_id": f"devgrant:{uuid.uuid4().hex}",
                "memory_id": memory_id,
                "memory_version": memory.version,
                "development_id": development_id,
                "source_policy_sha256": source_policy_sha256,
                "context": context,
                "guidance": memory.guidance,
                "source_trace_id": memory.source_trace_id,
                "evidence_refs": list(memory.evidence_refs),
                "exposed_at": time.time(),
                "result": None,
                **marker,
            }
            db.execute(
                "INSERT INTO memory_dev_exposures(grant_id, memory_id, development_id, payload) "
                "VALUES (?, ?, ?, ?)",
                (record["grant_id"], memory_id, development_id,
                 json.dumps(record, sort_keys=True, separators=(",", ":"))),
            )
            return record

    def record_result(
        self, *, grant_id: str, run_id: str, result_policy_sha256: str,
        artifact_uri: str, artifact_sha256: str,
    ) -> dict[str, Any]:
        """Attach native-run evidence; never infer causal benefit from one run."""
        if not all(isinstance(value, str) and value.strip() for value in
                   (grant_id, run_id, artifact_uri)):
            raise ValueError("grant, run and artifact IDs are required")
        if not all(isinstance(value, str) and _SHA.fullmatch(value) for value in
                   (result_policy_sha256, artifact_sha256)):
            raise ValueError("result source and artifact SHA-256 are required")
        with sqlite3.connect(self.service.store.path) as db:
            grant_row = db.execute('SELECT payload FROM memory_dev_exposures WHERE grant_id=?', (grant_id,)).fetchone()
        if grant_row:
            grant = json.loads(grant_row[0])
            provenance = self.service.v2._provenance(grant['memory_id'])
            if is_provisional(provenance):
                raise StateConflictError('provisional exposure is excluded from Dev results')
            if grant.get('trusted_here') is False:
                raise StateConflictError('reference exposure is excluded from Dev results')
            if is_test_seeded(provenance):
                raise StateConflictError('test-seeded exposure is excluded from Dev results')
            if provenance and provenance.get('channel') == 'dev_v2':
                raise StateConflictError('Dev v2 results require offline sealed bundle import')
        run = self.service.store.get_run(run_id)
        if run is None or run.get("status") not in {"failed", "completed", "cancelled"}:
            raise StateConflictError("Dev result requires a terminal native run")
        artifact = Path(artifact_uri)
        if not artifact.is_file() or artifact.name != "attention_run.json":
            raise StateConflictError("Dev result summary artifact is missing")
        content = artifact.read_bytes()
        if hashlib.sha256(content).hexdigest() != artifact_sha256:
            raise StateConflictError("Dev result artifact digest mismatch")
        try:
            summary = json.loads(content)
        except ValueError as exc:
            raise StateConflictError("Dev result artifact is not JSON") from exc
        if (not isinstance(summary, dict)
                or summary.get("schema_version") != "attentionbench.sim-gt-attention-run.v1"
                or summary.get("artifact_dir") != str(artifact.parent.resolve())
                or f"run:{artifact.parent.name}" != run_id
                or summary.get("suite") != run.get("suite")
                or summary.get("task_id") != run.get("task_id")
                or not isinstance(summary.get("native_success"), bool)):
            raise StateConflictError("Dev result artifact disagrees with native run")
        if (run["status"] == "completed") != summary["native_success"]:
            raise StateConflictError("Dev result outcome disagrees with native run status")
        result = {
            "run_id": run_id, "native_run_status": run["status"],
            "result_policy_sha256": result_policy_sha256,
            "artifact_uri": artifact_uri, "artifact_sha256": artifact_sha256,
            "recorded_at": time.time(),
            "claim": "development_exposure_only_not_causal_impact",
        }
        with sqlite3.connect(self.service.store.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT payload FROM memory_dev_exposures WHERE grant_id=?", (grant_id,)
            ).fetchone()
            if row is None:
                raise StateConflictError("unknown Dev Memory grant")
            record = json.loads(row[0])
            if record["result"] is not None:
                if any(record["result"][key] != result[key] for key in
                       ("run_id", "result_policy_sha256", "artifact_uri", "artifact_sha256")):
                    raise StateConflictError("Dev Memory result is immutable")
                return record
            if run.get("suite") != record["context"].get("suite") or run.get("task_id") != record["context"].get("task_id"):
                raise StateConflictError("Dev result run is outside memory context")
            record["result"] = result
            db.execute(
                "UPDATE memory_dev_exposures SET payload=? WHERE grant_id=?",
                (json.dumps(record, sort_keys=True, separators=(",", ":")), grant_id),
            )
            return record

    def get(self, grant_id: str) -> dict[str, Any]:
        with sqlite3.connect(self.service.store.path) as db:
            row = db.execute(
                "SELECT payload FROM memory_dev_exposures WHERE grant_id=?", (grant_id,)
            ).fetchone()
        if row is None:
            raise StateConflictError("unknown Dev Memory grant")
        record = json.loads(row[0])
        provenance = self.service.v2._provenance(record['memory_id'])
        if provenance and provenance.get('channel') in {'dev_v2', 'test_seeded', 'dev_v2_provisional'}:
            # New-channel reads release guidance again. Preserve the legacy
            # historical-read API; enforce live eligibility on Dev v2 grants.
            with sqlite3.connect(self.service.store.path) as db:
                memory = db.execute('SELECT payload FROM memories WHERE id=?',
                                    (record['memory_id'],)).fetchone()
            current = {} if memory is None else json.loads(memory[0])
            if (current.get('status') != ('provisional' if is_provisional(provenance) else 'trusted')
                    or current.get('version') != record['memory_version']
                    or current.get('expires_at') is not None and current['expires_at'] <= time.time()):
                raise StateConflictError('Dev Memory grant was revoked, expired or changed version')
            found, level = self._find(record['memory_id'], record['context'], time.time())
            if (found is None or found.version != record['memory_version']
                    or level != record.get('match_level', SAME_SCENE)):
                raise StateConflictError('Dev v2 grant no longer has verified exact scope')
        result = record.get("result")
        if result is not None and result.get('schema_version') == 'attentionbench.dev-v2-memory-result.v1':
            self.service.dev_v2.verify_result(record)
            # No private bundle/native result pointers leave the offline side.
            record['result'] = {k: result[k] for k in ('schema_version', 'execution_import_id',
                'run_id', 'cell', 'version', 'source_policy_sha256', 'result_policy_sha256', 'claim')}
            if result.get('channel') == 'dev_agent':
                record['result']['channel'] = 'dev_agent'
        elif result is not None:
            artifact = Path(result["artifact_uri"])
            if (not artifact.is_file()
                    or hashlib.sha256(artifact.read_bytes()).hexdigest()
                    != result["artifact_sha256"]):
                raise StateConflictError("Dev result artifact changed after recording")
        if 'source_task_id' not in record:
            # Layered grants have just rechecked their source note. Legacy
            # grants were exact-task reads and retain their original context.
            record['source_task_id'] = (found.applicability if provenance and provenance.get('channel')
                in {'dev_v2', 'test_seeded', 'dev_v2_provisional'} else record['context']).get('task_id')
        return _online_grant(record)


def _online_grant(record):
    record = dict(record)
    record.setdefault('layer', layer_of(record.get('match_level')))
    result = record.get('result')
    if result and result.get('schema_version') == 'attentionbench.dev-v2-memory-result.v1':
        record['result'] = {k: result[k] for k in ('schema_version', 'execution_import_id', 'run_id',
            'cell', 'version', 'source_policy_sha256', 'result_policy_sha256', 'claim')}
        if result.get('channel') == 'dev_agent':
            record['result']['channel'] = 'dev_agent'
    return record
