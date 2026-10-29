"""Importer-attested development hints and observational (never validation) uses."""
from __future__ import annotations

import json
import math
import sqlite3
import time

from .core.models import MemoryRecord, MemoryStatus
from .core.store import StateConflictError
from .dev_v2 import SHA, digest, encoded, public, view_no_oracle, validate_signature, signature_present
from .applicability import require_profile_match
from .note_title import validate_group, validate_summary, validate_title
from .seeded_notes import is_provisional, flags, validate_context

DIGEST_KEYS = frozenset(('origin_public_feedback', 'origin_source', 'help',
                         'help_model_input', 'help_source', 'help_public_feedback'))
ORIGIN_KEYS = frozenset(('experiment_id', 'cell', 'policy_id', 'origin_version',
                         'help_version', 'provider', 'effective_reason', 'eval_reason_class', 'digests'))


def feedback(text, sha256):
    if not isinstance(text, str) or not isinstance(sha256, str) or not SHA.fullmatch(sha256):
        raise ValueError('feedback text and SHA-256 are required')
    if digest(text.encode('utf-8')) != sha256:
        raise StateConflictError('public feedback digest mismatch')
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('public feedback must be a JSON object')
    # Validate the public fields consumed by the shared signature matcher.
    trace = value.get('action_trace', [])
    if (not isinstance(value.get('public_error', ''), str) or not isinstance(trace, list)
            or any(not isinstance(e, dict) or any(k in e and not isinstance(e[k], str)
                   for k in ('source', 'operation', 'status')) for e in trace)):
        raise ValueError('invalid public feedback action trace')
    return value


def progress(value):
    completed = 0
    for event in value.get('action_trace', []):
        if event.get('status') == 'failed' or event.get('error'):
            break
        completed += event.get('status') in {'completed', 'succeeded'}
    return completed


def create(service, *, actor, memory_id, context, guidance, failure_signature, origin,
           origin_public_feedback_text, help_public_feedback_text,
           title=None, summary=None, group=None, ttl_days=14, from_run_note=None, agent_evidence=None):
    for name, value in (('actor', actor), ('memory_id', memory_id), ('guidance', guidance)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} is required')
    validate_context(context)
    if 'sdk_interface_sha256' not in context:
        # New tier: always matched on the SDK interface identity.
        raise ValueError('provisional notes require sdk_interface_sha256')
    public(guidance)
    view_no_oracle({'guidance': guidance})
    if type(ttl_days) not in (int, float) or not math.isfinite(ttl_days) or not 0 < ttl_days <= 14:
        raise ValueError('ttl_days must be positive and at most 14')
    display = {key: validate(value) for key, value, validate in (
        ('title', title, validate_title), ('summary', summary, validate_summary),
        ('group', group, validate_group)) if value is not None}
    agent = isinstance(origin, dict) and origin.get('channel') == 'dev_agent'
    if not isinstance(origin, dict) or set(origin) != (ORIGIN_KEYS | {'channel'} if agent else ORIGIN_KEYS):
        raise ValueError('origin requires exactly the documented fields')
    if any(not isinstance(origin[k], str) or not origin[k].strip()
           for k in ('experiment_id', 'cell', 'policy_id', 'eval_reason_class')):
        raise ValueError('origin identifiers and eval_reason_class are required')
    if (type(origin['origin_version']) is not int or origin['origin_version'] < 1
            or type(origin['help_version']) is not int
            or origin['help_version'] != origin['origin_version'] + 1):
        raise ValueError('help_version must equal origin_version + 1')
    if (origin['provider'] not in ('advisor', 'human_proxy')
            or origin['effective_reason'] not in ('signature_changed', 'progressed')):
        raise ValueError('invalid origin provider or effective_reason')
    digests = origin['digests']
    keys = (DIGEST_KEYS - {'help_model_input'} | {'agent_help_evidence'}) if agent else DIGEST_KEYS
    if (not isinstance(digests, dict) or set(digests) != keys
            or any(not isinstance(v, str) or not SHA.fullmatch(v) for v in digests.values())):
        raise ValueError('origin digests must be SHA-256 hex strings')
    if agent:
        from .agent_provisional import verify
        verify(origin, guidance, agent_evidence)
        if (agent_evidence['refs']['origin_public_feedback'] != origin_public_feedback_text
                or agent_evidence['refs']['help_public_feedback'] != help_public_feedback_text):
            raise StateConflictError('agent feedback differs from evidence')
    elif agent_evidence is not None:
        raise ValueError('agent evidence requires dev_agent origin')
    before = feedback(origin_public_feedback_text, digests['origin_public_feedback'])
    after = feedback(help_public_feedback_text, digests['help_public_feedback'])
    validate_signature(failure_signature, before)
    origin_progress, help_progress = progress(before), progress(after)
    effective = ('signature_changed' if not signature_present(after, failure_signature)
                 else 'progressed' if help_progress > origin_progress else None)
    if effective is None:
        raise StateConflictError('intervention not effective')
    if effective != origin['effective_reason']:
        raise StateConflictError('effective_reason disagrees with public feedback')
    requested = dict(actor=actor, memory_id=memory_id, context=context, guidance=guidance,
                     failure_signature=failure_signature, origin=origin,
                     origin_public_feedback_text=origin_public_feedback_text,
                     help_public_feedback_text=help_public_feedback_text, ttl_days=ttl_days,
                     title=title, summary=summary, group=group)
    if agent:
        requested['agent_evidence'] = agent_evidence
    if from_run_note is not None:
        if (not isinstance(from_run_note, dict)
                or set(from_run_note) != {'experiment_id', 'cell', 'version', 'run_note_id'}
                or from_run_note['experiment_id'] != origin['experiment_id']
                or from_run_note['cell'] != origin['cell']
                or type(from_run_note['version']) is not int
                or from_run_note['version'] != origin['help_version']
                or from_run_note['run_note_id'] != 'r-' + digest(
                    f"{origin['experiment_id']}:{origin['cell']}:{origin['help_version']}".encode())[:24]):
            raise ValueError('invalid from_run_note binding')
        requested['from_run_note'] = from_run_note
    fingerprint = digest(encoded(requested).encode())
    trace = f'provisional:{memory_id}'
    with sqlite3.connect(service.store.path) as db:
        db.execute('BEGIN IMMEDIATE')
        old = db.execute('SELECT payload FROM memory_v2_provenance WHERE memory_id=?', (memory_id,)).fetchone()
        occupied = db.execute('SELECT 1 FROM memories WHERE id=?', (memory_id,)).fetchone()
        if old or occupied:
            if not old or not is_provisional(json.loads(old[0])) or json.loads(old[0]).get('request_sha256') != fingerprint:
                raise StateConflictError('memory ID already binds different content or origin')
            return service.store.get_memory(memory_id)
        created_at = time.time()
        artifact = dict(kind='text_hint', guidance=guidance, repair='Importer-attested guidance; no validated repair.',
                        applicability=context, failure_signature=failure_signature)
        provenance = dict(schema_version='attentionbench.memory-provenance.v2',
                          channel='dev_agent' if agent else 'dev_v2_provisional', source_kind='advisor_proxy',
                          trust_scope='provisional', unvalidated=True, excluded_from_results=True,
                          attested_by='offline_importer', origin=origin, layer='scene',
                          **({'from_run_note': from_run_note} if from_run_note is not None else {}),
                          failure_signature=failure_signature, origin_progress=origin_progress,
                          help_progress=help_progress, source_trace_id=trace, source_seed=context['seed'],
                          request_sha256=fingerprint, artifact=artifact,
                          artifact_sha256=digest(encoded(artifact).encode()), **display)
        memory = MemoryRecord(memory_id=memory_id, version=1, source_trace_id=trace,
                              guidance=guidance, candidate_repair='Importer-attested guidance; no validated repair.', applicability=context,
                              evidence_refs=(trace,), created_at=created_at,
                              expires_at=created_at + ttl_days * 86400,
                              status=MemoryStatus.PROVISIONAL,
                              status_reason='provisional_unvalidated_excluded_from_results')
        db.execute('INSERT INTO memories(id,payload) VALUES (?,?)', (memory_id, encoded(memory.artifact())))
        db.execute('INSERT INTO memory_v2_provenance(memory_id,payload) VALUES (?,?)', (memory_id, encoded(provenance)))
        db.execute("INSERT INTO events(event_key,entity_type,entity_id,event_type,payload) "
                   "VALUES (?,'memory',?,'memory.provisional_created',?)",
                   (f'memory-provisional:{memory_id}', memory_id,
                    encoded(dict(memory=memory.artifact(), actor=actor, recorded_at=created_at, **flags(provenance)))))
    return memory


def initialize(store):
    with sqlite3.connect(store.path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS memory_provisional_outcomes ('
                   'grant_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, payload TEXT NOT NULL)')


def record_outcome(service, *, grant_id, development_id, public_feedback_text, public_feedback_sha256,
                   execution_import_id=None):
    observed = feedback(public_feedback_text, public_feedback_sha256)
    request = dict(grant_id=grant_id, development_id=development_id,
                   public_feedback_text=public_feedback_text, public_feedback_sha256=public_feedback_sha256)
    execution_provenance = {}
    if execution_import_id is not None:
        execution, d, bundle, *_ = service.dev_v2.execution(execution_import_id)
        from .dev_v2 import require
        require(bundle.get('channel') == 'dev_agent', 'execution-linked outcome requires agent evidence')
        require(development_id == f"{d['cell']}:v{d['version']}"
                and observed == service.dev_v2.ref(execution['catalog_id'], execution['public_feedback']),
                'agent outcome differs from sealed execution feedback')
        request['execution_import_id'] = execution_import_id
        execution_provenance = dict(channel='dev_agent', execution_import_id=execution_import_id,
                                    provenance=dict(channel='dev_agent', execution_import_id=execution_import_id))
    fingerprint = digest(encoded(request).encode())
    # Sealed-source verification may reopen immutable imports. Do it before
    # acquiring the outcome write lock, then recheck provenance in that lock.
    with sqlite3.connect(service.store.path) as db:
        row = db.execute('SELECT payload FROM memory_dev_exposures WHERE grant_id=?', (grant_id,)).fetchone()
    if row is None:
        raise StateConflictError('unknown Dev Memory grant')
    grant = json.loads(row[0])
    if execution_import_id is not None:
        require(grant['context'] == d['context'], 'agent outcome grant scope mismatch')
    provenance = service.v2.provenance(grant['memory_id'])
    require_profile_match(provenance['artifact']['applicability'], grant['context'])
    plan = service.v2.get_plan(grant['memory_id'])
    with sqlite3.connect(service.store.path) as db:
        db.execute('BEGIN IMMEDIATE')
        old = db.execute('SELECT payload FROM memory_provisional_outcomes WHERE grant_id=?', (grant_id,)).fetchone()
        if old:
            record = json.loads(old[0])
            if record['request_sha256'] != fingerprint:
                # An attested observation may already have been auto-imported.
                # Attach verified execution provenance only to those exact bytes;
                # do not replace its observation, timing, or original metrics.
                unlinked = {k: v for k, v in request.items() if k != 'execution_import_id'}
                if (execution_import_id is not None and 'execution_import_id' not in record
                        and record['request_sha256'] == digest(encoded(unlinked).encode())):
                    record.update(execution_provenance, request_sha256=fingerprint)
                    db.execute('UPDATE memory_provisional_outcomes SET payload=? WHERE grant_id=?',
                               (encoded(record), grant_id))
                elif (execution_import_id is None and record.get('execution_import_id') is not None
                        and record['request_sha256'] == digest(encoded(dict(request,
                            execution_import_id=record['execution_import_id'])).encode())):
                    pass  # Idempotent retry of the original observational request.
                else:
                    raise StateConflictError('provisional outcome is immutable')
            return record
        current = json.loads(db.execute('SELECT payload FROM memory_v2_provenance WHERE memory_id=?',
                                        (grant['memory_id'],)).fetchone()[0])
        if current != provenance:
            raise StateConflictError('memory provenance changed while recording outcome; retry')
        eligible = (is_provisional(provenance) and grant.get('trust_scope') == 'provisional'
                    or provenance.get('channel') == 'dev_v2'
                    and plan and plan.get('profile') == 'single_scene_v1')
        if (not eligible or grant['development_id'] != development_id):
            raise StateConflictError('outcome requires a matching provisional development grant')
        record = dict(schema_version='attentionbench.provisional-outcome.v1',
                      grant_id=grant_id, memory_id=grant['memory_id'], memory_version=grant['memory_version'],
                      development_id=development_id, public_feedback_sha256=public_feedback_sha256,
                      signature_present=signature_present(observed, provenance['failure_signature']),
                      progress=progress(observed), origin_progress=provenance.get('origin_progress'),
                      observational=True, attested_by='offline_importer', recorded_at=time.time(),
                      request_sha256=fingerprint, **flags(provenance), **execution_provenance)
        db.execute('INSERT INTO memory_provisional_outcomes(grant_id,memory_id,payload) VALUES (?,?,?)',
                   (grant_id, grant['memory_id'], encoded(record)))
        from .auto_validation import demote_if_due
        demote_if_due(db, grant['memory_id'])
    return record


def uses_summary(store, memory_id):
    with sqlite3.connect(store.path) as db:
        grants = db.execute('SELECT COUNT(*) FROM memory_dev_exposures WHERE memory_id=?', (memory_id,)).fetchone()[0]
        outcomes = [json.loads(r[0]) for r in db.execute(
            'SELECT payload FROM memory_provisional_outcomes WHERE memory_id=?', (memory_id,))]
    return dict(grants=grants, outcomes=len(outcomes),
                failure_gone=sum(r['signature_present'] is False for r in outcomes))
