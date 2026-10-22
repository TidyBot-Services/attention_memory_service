"""Stdlib HTTP client used by a separately running Memory Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .core.models import MemoryRecord, MemoryStatus, MemoryUseRecord
from .core.store import StateConflictError


def _memory(value: dict[str, Any]) -> MemoryRecord:
    payload = dict(value)
    for key in ('seeded_for_tests', 'excluded_from_results', 'trust_scope', 'unvalidated',
                'match_level', 'trusted_here', 'source_seed'):
        payload.pop(key, None)
    payload["status"] = MemoryStatus(payload["status"])
    payload["evidence_refs"] = tuple(payload["evidence_refs"])
    return MemoryRecord(**payload)


class MemoryServiceClient:
    def __init__(
        self, base_url: str, *, api_key: str, timeout: float = 30.0,
        operator_key: str | None = None,
        importer_key: str | None = None,
    ) -> None:
        if not base_url.startswith(("http://127.0.0.1:", "http://localhost:", "https://")):
            raise ValueError("remote Memory Service requires HTTPS")
        if not api_key:
            raise ValueError("Memory Service API key is required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.operator_key = operator_key
        self.importer_key = importer_key
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        return self._call("GET", "/health")

    def store_id(self) -> str:
        return str(self._call("GET", "/store")["store_id"])

    def source_for_agent(self, request_id: str) -> dict[str, Any]:
        return self._call("GET", f"/sources/{quote(request_id, safe='')}")

    def create_candidate(self, **payload: Any) -> MemoryRecord:
        return _memory(self._call("POST", "/candidates", payload))

    def get_memory(self, memory_id: str) -> MemoryRecord:
        return _memory(self._call("GET", f"/memories/{quote(memory_id, safe='')}"))

    def list_dev_v2_memories(self, *, status: str | None = None,
                             context: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if context is not None:
            return self._call('POST', '/dev-v2/memories/query',
                              {'status': status, 'context': context})['memories']
        suffix = '' if status is None else '?status=' + quote(status, safe='')
        return self._call('GET', '/dev-v2/memories' + suffix)['memories']

    def get_dev_v2_memory_public(self, memory_id: str) -> dict[str, Any]:
        return self._call('GET', f"/dev-v2/memories/{quote(memory_id, safe='')}")

    def provenance(self, memory_id: str) -> dict[str, Any]:
        return self._call("GET", f"/memories/{quote(memory_id, safe='')}/provenance")

    def check_candidate(self, memory_id: str) -> dict[str, Any]:
        return self._call("GET", f"/memories/{quote(memory_id, safe='')}/preflight")

    def record_plan(self, memory_id: str, plan: dict[str, Any]) -> dict[str, Any]:
        return self._call("PUT", f"/memories/{quote(memory_id, safe='')}/plan", plan)

    def get_plan(self, memory_id: str) -> dict[str, Any] | None:
        return self._call("GET", f"/memories/{quote(memory_id, safe='')}/plan")

    def import_dev_v2_execution(self, **payload):
        return self._call('POST', '/dev-v2/executions', payload, importer=True)

    def import_dev_v2_source(self, **payload):
        return self._call('POST', '/dev-v2/sources', payload, importer=True)

    def import_dev_v2_trial(self, **payload):
        return self._call('POST', '/dev-v2/trials', payload, importer=True)

    def authorize_dev_v2_validation(self, **payload):
        return self._call('POST', '/dev-v2/validation-grants', payload, importer=True)

    def create_dev_v2_candidate(self, **payload):
        return _memory(self._call('POST', '/dev-v2/candidates', payload, importer=True))

    def create_test_seeded_note(self, **payload) -> dict[str, Any]:
        return self._call('POST', '/dev-v2/test-seeded-notes', payload, importer=True)

    def create_provisional_note(self, **payload) -> dict[str, Any]:
        return self._call('POST', '/dev-v2/provisional-notes', payload, importer=True)

    def record_provisional_outcome(self, **payload) -> dict[str, Any]:
        return self._call('POST', '/dev-v2/provisional-outcomes', payload, importer=True)

    def record_dev_v2_pair(self, **payload):
        return self._call('POST', '/dev-v2/pairs', payload, importer=True)

    def record_dev_v2_result(self, **payload):
        return self._call('POST', '/dev-v2/results', payload, importer=True)

    def export_package(self, memory_id: str) -> Path:
        result = self._call("POST", f"/memories/{quote(memory_id, safe='')}/export")
        return Path(result["directory"])

    def verify_package(self, memory_id: str) -> dict[str, Any]:
        return self._call("GET", f"/memories/{quote(memory_id, safe='')}/artifact")

    def retrieve(self, context: dict[str, Any], *, now: float) -> list[MemoryRecord]:
        return [
            _memory(item) for item in self.retrieve_entries(context, now=now)
        ]

    def retrieve_entries(self, context: dict[str, Any], *, now: float,
                         match_levels: list[str] | None = None) -> list[dict[str, Any]]:
        """Raw retrieval records retain test-seed/result-exclusion flags and,
        per record, ``match_level``/``trusted_here``/``source_seed``."""
        payload = {'context': context, 'now': now}
        if match_levels is not None:
            payload['match_levels'] = list(match_levels)
        return self._call('POST', '/retrieve', payload)

    def authorize_use(
        self, context: dict[str, Any], *, memory_id: str,
        attempt_id: str, now: float,
    ) -> dict[str, Any]:
        return self._call("POST", "/use-grants", {
            "context": context, "memory_id": memory_id,
            "attempt_id": attempt_id, "now": now,
        })

    def authorize_dev_use(
        self, *, memory_id: str, development_id: str, context: dict[str, Any],
        source_policy_sha256: str, now: float, memory_version: int | None = None,
    ) -> dict[str, Any]:
        return self._call("POST", "/dev-use-grants", {
            "memory_id": memory_id, "development_id": development_id,
            "context": context, "source_policy_sha256": source_policy_sha256,
            "now": now, "memory_version": memory_version,
        })

    def get_dev_use(self, grant_id: str) -> dict[str, Any]:
        return self._call("GET", f"/dev-use-grants/{quote(grant_id, safe='')}")

    def record_dev_result(
        self, *, grant_id: str, run_id: str, result_policy_sha256: str,
        artifact_uri: str, artifact_sha256: str,
    ) -> dict[str, Any]:
        return self._call("POST", "/dev-use-results", {
            "grant_id": grant_id, "run_id": run_id,
            "result_policy_sha256": result_policy_sha256,
            "artifact_uri": artifact_uri,
            "artifact_sha256": artifact_sha256,
        })


    def record_use(self, use: MemoryUseRecord) -> MemoryUseRecord:
        payload = self._call("POST", "/uses", use.artifact())
        for key in ('seeded_for_tests', 'excluded_from_results', 'trust_scope', 'unvalidated'):
            payload.pop(key, None)
        payload["evidence_refs"] = tuple(payload.get("evidence_refs", ()))
        return MemoryUseRecord(**payload)

    def record_pair(
        self, *, memory_id: str, control_attempt_id: str,
        treatment_attempt_id: str, control_safety: Path,
        treatment_safety: Path,
    ) -> dict[str, Any]:
        return self._call("POST", "/pairs", {
            "memory_id": memory_id,
            "control_attempt_id": control_attempt_id,
            "treatment_attempt_id": treatment_attempt_id,
            "control_safety": json.loads(control_safety.read_text(encoding="utf-8")),
            "treatment_safety": json.loads(treatment_safety.read_text(encoding="utf-8")),
        })

    def impact_report(self, memory_id: str) -> dict[str, Any]:
        return self._call("GET", f"/memories/{quote(memory_id, safe='')}/impact")

    def list_pairs(self, memory_id: str) -> list[dict[str, Any]]:
        return self._call("GET", f"/memories/{quote(memory_id, safe='')}/pairs")

    def promote(self, memory_id: str) -> MemoryRecord:
        return _memory(self._call("POST", f"/memories/{quote(memory_id, safe='')}/promote"))

    def list_memories(self, *, status: str | None = None) -> list[dict[str, Any]]:
        suffix = "" if status is None else f"?status={quote(status, safe='')}"
        return self._call("GET", f"/memories{suffix}")["memories"]

    def uses(self, memory_id: str) -> list[dict[str, Any]]:
        return self._call("GET", f"/memories/{quote(memory_id, safe='')}/uses")["uses"]

    def history(self, memory_id: str) -> list[dict[str, Any]]:
        return self._call("GET", f"/memories/{quote(memory_id, safe='')}/history")["history"]

    def decision(self, memory_id: str) -> dict[str, Any] | None:
        return self._call("GET", f"/memories/{quote(memory_id, safe='')}/decision")["decision"]

    def evaluate(self, memory_id: str) -> MemoryRecord:
        return _memory(self._call("POST", f"/memories/{quote(memory_id, safe='')}/evaluate", {}))

    def enable(self, memory_id: str, *, actor: str, reason: str) -> MemoryRecord:
        return _memory(self._call(
            "POST", f"/memories/{quote(memory_id, safe='')}/enable",
            {"actor": actor, "reason": reason}, operator=True))

    def disable(self, memory_id: str, *, actor: str, reason: str) -> MemoryRecord:
        return _memory(self._call(
            "POST", f"/memories/{quote(memory_id, safe='')}/disable",
            {"actor": actor, "reason": reason}, operator=True,
        ))

    def rollback(self, memory_id: str, *, actor: str, reason: str) -> MemoryRecord:
        return _memory(self._call(
            "POST", f"/memories/{quote(memory_id, safe='')}/rollback",
            {"actor": actor, "reason": reason}, operator=True,
        ))

    def set_expiry(
        self, memory_id: str, *, expires_at: float, actor: str, reason: str,
    ) -> MemoryRecord:
        return _memory(self._call(
            "PUT", f"/memories/{quote(memory_id, safe='')}/expiry",
            {"expires_at": expires_at, "actor": actor, "reason": reason},
            operator=True,
        ))

    def _call(
        self, method: str, path: str, payload: dict[str, Any] | None = None,
        *, operator: bool = False, importer: bool = False,
    ) -> Any:
        if operator and not self.operator_key:
            raise PermissionError("operator key is required")
        data = None if payload is None else json.dumps(payload).encode()
        headers = {
            "X-Memory-Service-Key": self.api_key,
            "Content-Type": "application/json",
        }
        if operator:
            headers["X-Memory-Operator-Key"] = self.operator_key
        if importer and not self.importer_key:
            raise PermissionError('offline importer key is required')
        # A separately configured offline client can use the existing plan /
        # preflight / promote paths; online clients never acquire this header.
        if self.importer_key:
            headers['X-Memory-Importer-Key'] = self.importer_key
        request = Request(
            self.base_url + path, data=data, method=method,
            headers=headers,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("detail", str(exc))
            except (ValueError, OSError):
                detail = str(exc)
            if exc.code == 409:
                raise StateConflictError(detail) from exc
            if exc.code == 403:
                raise PermissionError(detail) from exc
            if exc.code == 404:
                raise FileNotFoundError(detail) from exc
            if exc.code == 422:
                raise ValueError(detail) from exc
            raise RuntimeError(f"Memory Service HTTP {exc.code}: {detail}") from exc
