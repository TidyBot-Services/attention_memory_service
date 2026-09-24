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
    payload["status"] = MemoryStatus(payload["status"])
    payload["evidence_refs"] = tuple(payload["evidence_refs"])
    return MemoryRecord(**payload)


class MemoryServiceClient:
    def __init__(
        self, base_url: str, *, api_key: str, timeout: float = 30.0,
        operator_key: str | None = None,
    ) -> None:
        if not base_url.startswith(("http://127.0.0.1:", "http://localhost:", "https://")):
            raise ValueError("remote Memory Service requires HTTPS")
        if not api_key:
            raise ValueError("Memory Service API key is required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.operator_key = operator_key
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

    def provenance(self, memory_id: str) -> dict[str, Any]:
        return self._call("GET", f"/memories/{quote(memory_id, safe='')}/provenance")

    def record_plan(self, memory_id: str, plan: dict[str, Any]) -> dict[str, Any]:
        return self._call("PUT", f"/memories/{quote(memory_id, safe='')}/plan", plan)

    def get_plan(self, memory_id: str) -> dict[str, Any] | None:
        return self._call("GET", f"/memories/{quote(memory_id, safe='')}/plan")

    def export_package(self, memory_id: str) -> Path:
        result = self._call("POST", f"/memories/{quote(memory_id, safe='')}/export")
        return Path(result["directory"])

    def verify_package(self, memory_id: str) -> dict[str, Any]:
        return self._call("GET", f"/memories/{quote(memory_id, safe='')}/artifact")

    def retrieve(self, context: dict[str, Any], *, now: float) -> list[MemoryRecord]:
        return [
            _memory(item) for item in self._call(
                "POST", "/retrieve", {"context": context, "now": now},
            )
        ]

    def record_use(self, use: MemoryUseRecord) -> MemoryUseRecord:
        payload = self._call("POST", "/uses", use.artifact())
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
        *, operator: bool = False,
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
