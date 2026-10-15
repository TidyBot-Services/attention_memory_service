"""Authenticated loopback HTTP surface for the shared Memory Service.

This is an optional transport; the Memory Agent and runner can use the same
service boundary in-process. No daemon is started merely by importing it.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, StrictInt

from .core.models import MemoryUseRecord
from .core.store import StateConflictError
from .identity import store_id
from .memory_service import MemoryService


class CandidateInput(BaseModel):
    memory_id: str
    request_id: str
    guidance: str
    repair: str
    applicability: dict[str, str]
    created_at: float
    artifact_kind: str = "text_hint"
    human_attention_seconds: float = 0.0


class RetrieveInput(BaseModel):
    context: dict[str, Any]
    now: float


class UseGrantInput(RetrieveInput):
    memory_id: str
    attempt_id: str


class DevUseInput(BaseModel):
    memory_version: StrictInt | None = None
    memory_id: str
    development_id: str
    context: dict[str, Any]
    source_policy_sha256: str
    now: float


class DevResultInput(BaseModel):
    grant_id: str
    run_id: str
    result_policy_sha256: str
    artifact_uri: str
    artifact_sha256: str


class PairInput(BaseModel):
    memory_id: str
    control_attempt_id: str
    treatment_attempt_id: str
    control_safety: dict[str, Any]
    treatment_safety: dict[str, Any]


class UseInput(BaseModel):
    use_id: str
    memory_id: str
    memory_version: int
    run_id: str
    attempt_id: str
    used_at: float
    outcome: str
    evidence_refs: tuple[str, ...] = ()


class OperatorInput(BaseModel):
    actor: str
    reason: str


class ExpiryInput(OperatorInput):
    expires_at: float


class ImportInput(BaseModel):
    model_config = {'extra': 'forbid'}
    import_id: str
    catalog_id: str
    descriptor: str


class ImportedCandidateInput(BaseModel):
    model_config = {'extra': 'forbid'}
    memory_id: str
    source_import_id: str
    created_at: float


class ImportedPairInput(BaseModel):
    model_config = {'extra': 'forbid'}
    memory_id: str
    control_import_id: str
    treatment_import_id: str


class ImportedResultInput(BaseModel):
    model_config = {'extra': 'forbid'}
    grant_id: str
    execution_import_id: str


class ValidationGrantInput(BaseModel):
    model_config = {'extra': 'forbid'}
    memory_id: str
    cell: str
    version: StrictInt
    seed: StrictInt
    context: dict[str, Any]
    policy_sha256: str
    config_sha256: str
    exposure: str


def create_app(store_path: Path, *, api_key: str, operator_key: str | None = None,
               importer_key: str | None = None,
               dev_v2_catalogs: dict[str, Any] | None = None) -> FastAPI:
    if not api_key or len(api_key) < 16:
        raise ValueError("Memory Service requires an API key of at least 16 characters")
    if operator_key is not None and len(operator_key) < 16:
        raise ValueError("Memory Service operator key must have at least 16 characters")
    if operator_key is not None and hmac.compare_digest(operator_key, api_key):
        raise ValueError("operator key must differ from the service API key")
    if importer_key is not None and (len(importer_key) < 16 or importer_key in {api_key, operator_key}):
        raise ValueError('importer key must be distinct and have at least 16 characters')
    service = MemoryService(store_path, dev_v2_catalogs=dev_v2_catalogs)
    app = FastAPI(title="TidyBot Attention Memory Service", version="2")

    def authorized(x_memory_service_key: str | None = Header(default=None)) -> None:
        if not x_memory_service_key or not hmac.compare_digest(x_memory_service_key, api_key):
            raise HTTPException(status_code=401, detail="invalid Memory Service key")

    def operator_authorized(
        x_memory_service_key: str | None = Header(default=None),
        x_memory_operator_key: str | None = Header(default=None),
    ) -> None:
        authorized(x_memory_service_key)
        if operator_key is None or not x_memory_operator_key or not hmac.compare_digest(x_memory_operator_key, operator_key):
            raise HTTPException(status_code=403, detail="operator authorization required")

    def importer_authorized(x_memory_service_key: str | None = Header(default=None),
                            x_memory_importer_key: str | None = Header(default=None)):
        authorized(x_memory_service_key)
        if importer_key is None or not x_memory_importer_key or not hmac.compare_digest(importer_key, x_memory_importer_key):
            raise HTTPException(status_code=403, detail='offline importer authorization required')

    def private_memory_authorized(request: Request,
                                  x_memory_service_key: str | None = Header(default=None),
                                  x_memory_importer_key: str | None = Header(default=None)):
        provenance = service.v2._provenance(request.path_params['memory_id'])
        if provenance and provenance.get('channel') == 'dev_v2':
            importer_authorized(x_memory_service_key, x_memory_importer_key)

    def online_memory(memory):
        value = memory.artifact()
        provenance = service.v2._provenance(memory.memory_id)
        if provenance and provenance.get('channel') == 'dev_v2':
            for key in ('confidence', 'validation_successes', 'validation_failures', 'status_reason'):
                value.pop(key, None)
        return value

    @app.post('/dev-v2/executions', dependencies=[Depends(importer_authorized)])
    def import_execution(payload: ImportInput):
        return guarded(service.import_dev_v2_execution, **payload.model_dump())

    @app.post('/dev-v2/sources', dependencies=[Depends(importer_authorized)])
    def import_source(payload: ImportInput):
        return guarded(service.import_dev_v2_source, **payload.model_dump())

    @app.post('/dev-v2/trials', dependencies=[Depends(importer_authorized)])
    def import_trial(payload: ImportInput):
        return guarded(service.import_dev_v2_trial, **payload.model_dump())

    @app.post('/dev-v2/validation-grants', dependencies=[Depends(importer_authorized)])
    def validation_grant(payload: ValidationGrantInput):
        return guarded(service.authorize_dev_v2_validation, **payload.model_dump())

    @app.post('/dev-v2/candidates', dependencies=[Depends(importer_authorized)])
    def imported_candidate(payload: ImportedCandidateInput):
        return guarded(service.create_dev_v2_candidate, **payload.model_dump()).artifact()

    @app.post('/dev-v2/pairs', dependencies=[Depends(importer_authorized)])
    def imported_pair(payload: ImportedPairInput):
        return guarded(service.record_dev_v2_pair, **payload.model_dump())

    @app.post('/dev-v2/results', dependencies=[Depends(importer_authorized)])
    def imported_result(payload: ImportedResultInput):
        return guarded(service.record_dev_v2_result, **payload.model_dump())

    def guarded(call, *args, **kwargs):
        try:
            return call(*args, **kwargs)
        except StateConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, TypeError, KeyError, sqlite3.DatabaseError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/health")
    def health():
        return {
            "status": "ok", "schema_version": "attentionbench.memory-service.v2",
            "capabilities": ["attempt_bound_use_grants", "development_use_evidence", 'sealed_dev_v2_imports'],
        }

    @app.get("/store", dependencies=[Depends(authorized)])
    def identity():
        return {"store_id": store_id(store_path)}

    @app.get("/sources/{request_id}", dependencies=[Depends(authorized)])
    def source(request_id: str):
        return guarded(service.source_for_agent, request_id)

    @app.post("/candidates", dependencies=[Depends(authorized)])
    def candidate(payload: CandidateInput):
        return guarded(service.create_candidate, **payload.model_dump()).artifact()

    @app.post("/retrieve", dependencies=[Depends(authorized)])
    def retrieve(payload: RetrieveInput):
        return [online_memory(item) for item in guarded(
            service.retrieve, payload.context, now=payload.now,
        )]

    @app.post("/use-grants", dependencies=[Depends(authorized)])
    def use_grant(payload: UseGrantInput):
        return guarded(
            service.authorize_use, payload.context,
            memory_id=payload.memory_id, attempt_id=payload.attempt_id,
            now=payload.now,
        )

    @app.post("/dev-use-grants", dependencies=[Depends(authorized)])
    def dev_use_grant(payload: DevUseInput):
        return guarded(service.authorize_dev_use, **payload.model_dump())

    @app.get("/dev-use-grants/{grant_id}", dependencies=[Depends(authorized)])
    def get_dev_use(grant_id: str):
        return guarded(service.get_dev_use, grant_id)

    @app.post("/dev-use-results", dependencies=[Depends(authorized)])
    def dev_use_result(payload: DevResultInput):
        return guarded(service.record_dev_result, **payload.model_dump())

    @app.post("/uses", dependencies=[Depends(authorized)])
    def use(payload: UseInput):
        return guarded(
            service.record_use, MemoryUseRecord(**payload.model_dump())
        ).artifact()

    @app.post("/pairs", dependencies=[Depends(authorized)])
    def pair(payload: PairInput):
        if (service.v2._provenance(payload.memory_id) or {}).get('channel') == 'dev_v2':
            raise HTTPException(status_code=403, detail='Dev v2 pairs require offline imported trial evidence')
        def save_safety(value: dict[str, Any]) -> Path:
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            if len(encoded) > 8 * 1024 * 1024:
                raise ValueError("safety monitor artifact exceeds 8 MiB")
            digest = hashlib.sha256(encoded).hexdigest()
            directory = store_path.parent / "memory-safety-evidence"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{digest}.json"
            if path.exists():
                if path.read_bytes() != encoded:
                    raise StateConflictError("safety artifact digest collision")
            else:
                path.write_bytes(encoded)
            return path

        return guarded(
            service.record_pair,
            memory_id=payload.memory_id,
            control_attempt_id=payload.control_attempt_id,
            treatment_attempt_id=payload.treatment_attempt_id,
            control_safety=guarded(save_safety, payload.control_safety),
            treatment_safety=guarded(save_safety, payload.treatment_safety),
        )

    @app.get("/memories/{memory_id}", dependencies=[Depends(authorized), Depends(private_memory_authorized)])
    def memory(memory_id: str):
        return online_memory(guarded(service.get_memory, memory_id))

    @app.get("/memories/{memory_id}/provenance", dependencies=[Depends(authorized), Depends(private_memory_authorized)])
    def provenance(memory_id: str):
        return guarded(service.provenance, memory_id)

    @app.get("/memories/{memory_id}/preflight", dependencies=[Depends(authorized), Depends(private_memory_authorized)])
    def preflight(memory_id: str):
        return guarded(service.check_candidate, memory_id)

    @app.get("/memories/{memory_id}/impact", dependencies=[Depends(authorized), Depends(private_memory_authorized)])
    def impact(memory_id: str):
        return guarded(service.impact_report, memory_id)

    @app.get("/memories/{memory_id}/pairs", dependencies=[Depends(authorized), Depends(private_memory_authorized)])
    def pairs(memory_id: str):
        return guarded(service.list_pairs, memory_id)

    @app.put("/memories/{memory_id}/plan", dependencies=[Depends(authorized), Depends(private_memory_authorized)])
    def plan(memory_id: str, payload: dict[str, Any]):
        return guarded(service.record_plan, memory_id, payload)

    @app.get("/memories/{memory_id}/plan", dependencies=[Depends(authorized), Depends(private_memory_authorized)])
    def get_plan(memory_id: str):
        return guarded(service.get_plan, memory_id)

    @app.post("/memories/{memory_id}/export", dependencies=[Depends(authorized), Depends(private_memory_authorized)])
    def export(memory_id: str):
        return {"directory": str(guarded(service.export_package, memory_id))}

    @app.get("/memories/{memory_id}/artifact", dependencies=[Depends(authorized), Depends(private_memory_authorized)])
    def artifact(memory_id: str):
        return guarded(service.verify_package, memory_id)

    @app.post("/memories/{memory_id}/promote", dependencies=[Depends(authorized), Depends(private_memory_authorized)])
    def promote(memory_id: str):
        return guarded(service.promote, memory_id).artifact()

    @app.post("/memories/{memory_id}/disable", dependencies=[Depends(operator_authorized)])
    def disable(memory_id: str, payload: OperatorInput):
        return guarded(service.disable, memory_id, **payload.model_dump()).artifact()

    @app.post("/memories/{memory_id}/rollback", dependencies=[Depends(operator_authorized)])
    def rollback(memory_id: str, payload: OperatorInput):
        return guarded(service.rollback, memory_id, **payload.model_dump()).artifact()

    @app.put("/memories/{memory_id}/expiry", dependencies=[Depends(operator_authorized)])
    def set_expiry(memory_id: str, payload: ExpiryInput):
        return guarded(service.set_expiry, memory_id, **payload.model_dump()).artifact()

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--api-key-env", default="ATTENTION_MEMORY_API_KEY")
    parser.add_argument("--operator-key-env", default="ATTENTION_MEMORY_OPERATOR_KEY")
    parser.add_argument('--importer-key-env', default='ATTENTION_MEMORY_IMPORTER_KEY')
    parser.add_argument('--dev-v2-catalogs', type=Path)
    args = parser.parse_args()
    key = os.environ.get(args.api_key_env, "")
    app = create_app(
        args.store_path, api_key=key,
        operator_key=os.environ.get(args.operator_key_env) or None,
        importer_key=os.environ.get(args.importer_key_env) or None,
        dev_v2_catalogs=json.loads(args.dev_v2_catalogs.read_text()) if args.dev_v2_catalogs else None,
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
