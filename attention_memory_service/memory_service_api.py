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
from pydantic import Field, BaseModel, StrictInt, field_validator, model_validator

from .core.models import MemoryUseRecord
from .core.store import StateConflictError
from .applicability import marker
from .memory_v2 import trusted_here
from .identity import store_id
from .memory_service import MemoryService
from .seeded_notes import flags
from .note_title import validate_group, validate_summary, validate_title


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


class LayeredRetrieveInput(RetrieveInput):
    model_config = {'extra': 'forbid'}
    # Default stays exact; the Dev prompt path opts in to same_task references.
    match_levels: list[str] = ['same_scene']


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


class CatalogInput(BaseModel):
    model_config = {'extra': 'forbid'}
    catalog_id: str
    catalog: dict[str, Any]


class FailedCheckInput(BaseModel):
    model_config = {'extra': 'forbid'}
    memory_id: str
    validation_grant_id: str
    execution_import_id: str
    catalog_id: str
    descriptor: str


class ImportedCandidateInput(BaseModel):
    model_config = {'extra': 'forbid'}
    memory_id: str
    source_import_id: str
    created_at: float
    from_provisional: str | None = None


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
    origin_source_sha256: str | None = None
    target_seed: StrictInt | None = None


class CrossSeedProbeInput(BaseModel):
    model_config = {'extra': 'forbid'}
    memory_id: str
    target_seed: StrictInt
    execution_import_id: str
    catalog_id: str
    descriptor: str
    scene_contract_ref: dict[str, str]


class CrossSeedPlanInput(BaseModel):
    model_config = {'extra': 'forbid'}
    memory_id: str
    target_seed: StrictInt
    pairs: list[dict[str, str]]
    dev_model: dict[str, str]


class CrossSeedPairInput(BaseModel):
    model_config = {'extra': 'forbid'}
    memory_id: str
    target_seed: StrictInt
    control_import_id: str
    treatment_import_id: str


class CrossSeedFailedInput(BaseModel):
    model_config = {'extra': 'forbid'}
    memory_id: str
    target_seed: StrictInt
    validation_grant_id: str
    execution_import_id: str
    catalog_id: str
    descriptor: str


class SeededNoteInput(BaseModel):
    model_config = {'extra': 'forbid'}
    actor: str
    reason: str
    memory_id: str
    context: dict[str, Any]
    guidance: str
    repair: str = ''
    title: str | None = None
    summary: str | None = None
    group: str | None = None

    @model_validator(mode='before')
    @classmethod
    def display_fields_are_strings_when_present(cls, value):
        if isinstance(value, dict):
            for field in ('title', 'summary', 'group'):
                if field in value and value[field] is None:
                    raise ValueError(f'{field} must be a string')
        return value

    @field_validator('title')
    @classmethod
    def check_title(cls, value):
        return validate_title(value) if value is not None else None

    @field_validator('summary')
    @classmethod
    def check_summary(cls, value):
        return validate_summary(value) if value is not None else None

    @field_validator('group')
    @classmethod
    def check_group(cls, value):
        return validate_group(value) if value is not None else None


class ProvisionalNoteInput(BaseModel):
    model_config = {'extra': 'forbid'}
    actor: str
    memory_id: str
    context: dict[str, Any]
    guidance: str
    failure_signature: dict[str, Any]
    origin: dict[str, Any]
    from_run_note: dict | None = None
    agent_evidence: dict | None = None
    source_version_evidence: dict | None = None
    origin_public_feedback_text: str
    help_public_feedback_text: str
    title: str | None = None
    summary: str | None = None
    group: str | None = None
    ttl_days: float = Field(default=14, strict=True, gt=0, le=14)


class MemoryQueryInput(BaseModel):
    model_config = {'extra': 'forbid'}
    status: str | None = None
    context: dict[str, Any]


class ProvisionalOutcomeInput(BaseModel):
    model_config = {'extra': 'forbid'}
    execution_import_id: str | None = None
    grant_id: str
    development_id: str
    public_feedback_text: str
    public_feedback_sha256: str


def create_app(store_path: Path, *, api_key: str, operator_key: str | None = None,
               importer_key: str | None = None,
               dev_v2_catalogs: dict[str, Any] | None = None,
               dev_v2_registration_roots: dict[str, Any] | None = None) -> FastAPI:
    if not api_key or len(api_key) < 16:
        raise ValueError("Memory Service requires an API key of at least 16 characters")
    if operator_key is not None and len(operator_key) < 16:
        raise ValueError("Memory Service operator key must have at least 16 characters")
    if operator_key is not None and hmac.compare_digest(operator_key, api_key):
        raise ValueError("operator key must differ from the service API key")
    if importer_key is not None and (len(importer_key) < 16 or importer_key in {api_key, operator_key}):
        raise ValueError('importer key must be distinct and have at least 16 characters')
    service = MemoryService(store_path, dev_v2_catalogs=dev_v2_catalogs,
                            dev_v2_registration_roots=dev_v2_registration_roots)
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
        if provenance and provenance.get('channel') in {'dev_v2', 'test_seeded', 'dev_v2_provisional'}:
            importer_authorized(x_memory_service_key, x_memory_importer_key)

    def online_memory(memory):
        value = memory.artifact()
        provenance = service.v2._provenance(memory.memory_id)
        value.update(flags(provenance))
        if provenance and provenance.get('channel') in {'dev_v2', 'test_seeded', 'dev_v2_provisional'}:
            for key in ('confidence', 'validation_successes', 'validation_failures', 'status_reason'):
                value.pop(key, None)
        return value

    @app.get('/dev-v2/family', dependencies=[Depends(authorized)])
    def family_public():
        return guarded(service.family.public)

    @app.get('/dev-v2/family/{key}/{action}', dependencies=[Depends(importer_authorized)])
    def family_read(key: str, action: str):
        return guarded(service.family.read, key, action)

    @app.post('/dev-v2/family/{action}', dependencies=[Depends(importer_authorized)])
    def family_write(action: str, payload: dict):
        return guarded(service.family.write, action, payload)

    @app.get('/dev-v2/positive-controls', dependencies=[Depends(authorized)])
    def positive_controls():
        return {'items': guarded(service.positive_controls.public)}

    @app.get('/dev-v2/positive-controls/{control_id}/{action}', dependencies=[Depends(importer_authorized)])
    def positive_control_read(control_id: str, action: str):
        return guarded(service.positive_controls.read, control_id, action)

    @app.post('/dev-v2/positive-controls/{action}', dependencies=[Depends(importer_authorized)])
    def positive_control_write(action: str, payload: dict):
        return guarded(service.positive_controls.write, action, payload)

    @app.post('/dev-v2/cross-seed/probes', dependencies=[Depends(importer_authorized)])
    def cross_seed_probe(payload: CrossSeedProbeInput):
        return guarded(service.cross_seed.probe, **payload.model_dump())

    @app.post('/dev-v2/cross-seed/plans', dependencies=[Depends(importer_authorized)])
    def cross_seed_plan(payload: CrossSeedPlanInput):
        return guarded(service.cross_seed.plan, **payload.model_dump())

    @app.post('/dev-v2/cross-seed/pairs', dependencies=[Depends(importer_authorized)])
    def cross_seed_pair(payload: CrossSeedPairInput):
        return guarded(service.cross_seed.pair, **payload.model_dump())

    @app.post('/dev-v2/cross-seed/failed-checks', dependencies=[Depends(importer_authorized)])
    def cross_seed_failed(payload: CrossSeedFailedInput):
        return guarded(service.cross_seed.failed_check, **payload.model_dump())

    @app.get('/dev-v2/cross-seed/status', dependencies=[Depends(authorized)])
    def cross_seed_status(memory_id: str):
        return guarded(service.cross_seed.status, memory_id=memory_id)

    @app.get('/dev-v2/cross-seed/queue', dependencies=[Depends(authorized)])
    def cross_seed_queue():
        return {'queue': guarded(service.cross_seed.queue)}

    @app.post('/dev-v2/catalogs', dependencies=[Depends(importer_authorized)])
    def register_catalog(payload: CatalogInput):
        return guarded(service.register_dev_v2_catalog, **payload.model_dump())

    @app.get('/dev-v2/validation-queue', dependencies=[Depends(authorized)])
    def validation_queue():
        return {'items': guarded(service.validation_queue)}

    @app.get('/dev-v2/memory-events', dependencies=[Depends(authorized)])
    def memory_events(after: int = 0):
        return {'events': guarded(service.memory_events, after=after)}

    @app.post('/dev-v2/executions', dependencies=[Depends(importer_authorized)])
    def import_execution(payload: ImportInput):
        return guarded(service.import_dev_v2_execution, **payload.model_dump())

    @app.post('/dev-v2/sources', dependencies=[Depends(importer_authorized)])
    def import_source(payload: ImportInput):
        return guarded(service.import_dev_v2_source, **payload.model_dump())

    @app.post('/dev-v2/trials', dependencies=[Depends(importer_authorized)])
    def import_trial(payload: ImportInput):
        return guarded(service.import_dev_v2_trial, **payload.model_dump())

    @app.post('/dev-v2/failed-checks', dependencies=[Depends(importer_authorized)])
    def failed_check(payload: FailedCheckInput):
        return guarded(service.record_dev_v2_failed_check, **payload.model_dump())

    @app.post('/dev-v2/validation-grants', dependencies=[Depends(importer_authorized)])
    def validation_grant(payload: ValidationGrantInput):
        return guarded(service.authorize_dev_v2_validation, **payload.model_dump())

    @app.post('/dev-v2/candidates', dependencies=[Depends(importer_authorized)])
    def imported_candidate(payload: ImportedCandidateInput):
        return guarded(service.create_dev_v2_candidate, **payload.model_dump()).artifact()

    @app.post('/dev-v2/test-seeded-notes', dependencies=[Depends(importer_authorized)])
    def test_seeded_note(payload: SeededNoteInput):
        return guarded(service.create_test_seeded_note, **payload.model_dump())

    @app.get('/dev-v2/reuse', dependencies=[Depends(authorized)])
    def reuse_notes_list():
        return guarded(service.reuse_notes_list)

    @app.get('/dev-v2/reuse/{memory_id}', dependencies=[Depends(authorized)])
    def reuse_note_detail(memory_id: str):
        return guarded(service.reuse_note_detail, memory_id)

    @app.post('/dev-v2/reuse/generalize', dependencies=[Depends(importer_authorized)])
    def generalize_note(payload: dict):
        return guarded(service.generalize_note, **payload)

    @app.post('/dev-v2/reuse/confirm', dependencies=[Depends(operator_authorized)])
    def confirm_note_level(payload: dict):
        return guarded(service.confirm_note_level, **payload)

    @app.post('/dev-v2/provisional-notes', dependencies=[Depends(importer_authorized)])
    def provisional_note(payload: ProvisionalNoteInput):
        return guarded(service.create_provisional_note, **payload.model_dump())

    @app.post('/dev-v2/provisional-outcomes', dependencies=[Depends(importer_authorized)])
    def provisional_outcome(payload: ProvisionalOutcomeInput):
        return guarded(service.record_provisional_outcome, **payload.model_dump())

    @app.post('/dev-v2/pairs', dependencies=[Depends(importer_authorized)])
    def imported_pair(payload: ImportedPairInput):
        return guarded(service.record_dev_v2_pair, **payload.model_dump())

    @app.post('/dev-v2/results', dependencies=[Depends(importer_authorized)])
    def imported_result(payload: ImportedResultInput):
        return guarded(service.record_dev_v2_result, **payload.model_dump())

    @app.get('/dev-v2/memories', dependencies=[Depends(authorized)])
    def dev_v2_memories(status: str | None = None):
        return guarded(service.dev_v2_memory_listing, status=status)

    @app.post('/dev-v2/memories/query', dependencies=[Depends(authorized)])
    def dev_v2_memories_query(payload: MemoryQueryInput):
        return guarded(service.dev_v2_memory_listing, status=payload.status,
                       context=payload.context)

    @app.get('/dev-v2/memories/{memory_id}', dependencies=[Depends(authorized)])
    def dev_v2_memory(memory_id: str):
        return guarded(service.get_dev_v2_memory_public, memory_id)

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
    def retrieve(payload: LayeredRetrieveInput):
        rows = []
        for item, level in guarded(service.retrieve_levels, payload.context, now=payload.now,
                                   levels=tuple(getattr(payload, 'match_levels', ['same_scene']))):
            value = online_memory(item)
            provenance = service.v2._provenance(item.memory_id) or {}
            from .reuse_notes import FIELDS
            value.update({k:provenance[k] for k in FIELDS if k in provenance})
            value.update(marker(level, trusted_here(item, level, provenance, payload.context)), source_seed=provenance.get('source_seed'),
                         source_task_id=item.applicability.get('task_id'))
            rows.append(value)
        if 'same_family' in getattr(payload, 'match_levels', ['same_scene']):
            rows.extend(guarded(service.family.references,payload.context))
        return rows

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
        result = guarded(
            service.record_use, MemoryUseRecord(**payload.model_dump())
        )
        return {**result.artifact(), **flags(service.v2._provenance(result.memory_id))}

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

    @app.get("/memories", dependencies=[Depends(authorized)])
    def list_memories(status: str | None = None):
        return {"memories": guarded(service.list_memories, status=status)}

    @app.get("/memories/{memory_id}/uses", dependencies=[Depends(authorized), Depends(private_memory_authorized)])
    def memory_uses(memory_id: str):
        return {"uses": guarded(service.uses, memory_id)}

    @app.get("/memories/{memory_id}/history", dependencies=[Depends(authorized), Depends(private_memory_authorized)])
    def memory_history(memory_id: str):
        return {"history": guarded(service.history, memory_id)}

    @app.get("/memories/{memory_id}/decision", dependencies=[Depends(authorized), Depends(private_memory_authorized)])
    def memory_decision(memory_id: str):
        return {"decision": guarded(service.decision, memory_id)}

    @app.post("/memories/{memory_id}/evaluate", dependencies=[Depends(authorized), Depends(private_memory_authorized)])
    def evaluate(memory_id: str):
        return guarded(service.evaluate, memory_id).artifact()

    @app.post("/memories/{memory_id}/enable", dependencies=[Depends(operator_authorized)])
    def enable(memory_id: str, payload: OperatorInput):
        return guarded(service.enable, memory_id, **payload.model_dump()).artifact()

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
    parser.add_argument('--dev-v2-registration-roots', type=Path)
    args = parser.parse_args()
    key = os.environ.get(args.api_key_env, "")
    app = create_app(
        args.store_path, api_key=key,
        operator_key=os.environ.get(args.operator_key_env) or None,
        importer_key=os.environ.get(args.importer_key_env) or None,
        dev_v2_catalogs=json.loads(args.dev_v2_catalogs.read_text()) if args.dev_v2_catalogs else None,
        dev_v2_registration_roots=json.loads(args.dev_v2_registration_roots.read_text()) if args.dev_v2_registration_roots else None,
    )
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
