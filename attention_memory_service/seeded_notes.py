"""Importer-only test notes. These are never validation evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time

from .core.models import MemoryRecord, MemoryStatus
from .core.store import StateConflictError
from .dev_v2 import SCOPE, SHA
from .note_title import validate_group, validate_summary, validate_title

TUPLE_KEYS = frozenset((*SCOPE, 'seed', 'camera_names', 'task_prompt', 'scene_contract_ref'))


def is_test_seeded(provenance):
    return bool(provenance and provenance.get('source_kind') == 'seeded_for_tests'
                and provenance.get('trust_scope') == 'test_seeded'
                and provenance.get('excluded_from_results') is True)


def is_provisional(provenance):
    return bool(provenance and (provenance.get('channel') == 'dev_v2_provisional'
                               or provenance.get('channel') == 'dev_v2' and provenance.get('demoted_at') is not None)
                and provenance.get('trust_scope') == 'provisional'
                and provenance.get('unvalidated') is True
                and provenance.get('excluded_from_results') is True)


def flags(provenance):
    if is_provisional(provenance):
        return {'unvalidated': True, 'excluded_from_results': True,
                'trust_scope': 'provisional'}
    if is_test_seeded(provenance):
        return {'seeded_for_tests': True, 'excluded_from_results': True,
                'trust_scope': 'test_seeded'}
    return {}


def validate_context(context):
    # sdk_interface_sha256 is optional only for notes seeded before it existed;
    # see applicability.py for the matching rule.
    if not isinstance(context, dict) or set(context) - {'sdk_interface_sha256'} != TUPLE_KEYS:
        raise ValueError('test seed requires the complete exact single-scene tuple')
    if 'sdk_interface_sha256' in context and (
            not isinstance(context['sdk_interface_sha256'], str)
            or not SHA.fullmatch(context['sdk_interface_sha256'])):
        raise ValueError('sdk_interface_sha256 must be SHA-256')
    if type(context['seed']) is not int:
        raise ValueError('seed must be an integer')
    if any(not isinstance(context[k], str) or not context[k].strip()
           for k in SCOPE if k != 'sdk_sha256'):
        raise ValueError('scope values are required')
    if not isinstance(context['sdk_sha256'], str) or not SHA.fullmatch(context['sdk_sha256']):
        raise ValueError('sdk_sha256 must be SHA-256')
    if context['perception_mode'] not in {'sim_gt', 'vision'}:
        raise ValueError('unsupported perception mode')
    if (not isinstance(context['camera_names'], list) or not context['camera_names']
            or any(not isinstance(n, str) or not n.strip() for n in context['camera_names'])
            or len(set(context['camera_names'])) != len(context['camera_names'])
            or not isinstance(context['task_prompt'], str) or not context['task_prompt'].strip()):
        raise ValueError('camera names and task prompt are required')
    ref = context['scene_contract_ref']
    if (not isinstance(ref, dict) or set(ref) != {'uri', 'sha256'}
            or not isinstance(ref['uri'], str) or not ref['uri'].strip()
            or not isinstance(ref['sha256'], str) or not SHA.fullmatch(ref['sha256'])):
        raise ValueError('scene contract reference needs uri and SHA-256')


def create(service, *, actor, reason, memory_id, context, guidance, repair='',
           title=None, summary=None, group=None):
    if title is not None:
        title = validate_title(title)
    if summary is not None:
        summary = validate_summary(summary)
    if group is not None:
        group = validate_group(group)
    for name, value in (('actor', actor), ('reason', reason), ('memory_id', memory_id),
                        ('guidance', guidance)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name} is required')
    if not isinstance(repair, str):
        raise ValueError('repair must be text')
    validate_context(context)
    if any(type(value) is not str for value in (actor, reason, memory_id, guidance)):
        raise ValueError('invalid text')
    requested = {'context': context, 'guidance': guidance, 'repair': repair}
    if title is not None:
        requested['title'] = title
    if summary is not None:
        requested['summary'] = summary
    if group is not None:
        requested['group'] = group
    stored_repair = repair or 'No repair specified for test seed.'
    encoded = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False,
                                       separators=(',', ':'), allow_nan=False)
    fingerprint = hashlib.sha256(encoded(requested).encode()).hexdigest()
    trace = f'seeded-for-tests:{memory_id}'
    with sqlite3.connect(service.store.path) as db:
        db.execute('BEGIN IMMEDIATE')
        old = db.execute('SELECT payload FROM memory_v2_provenance WHERE memory_id=?',
                         (memory_id,)).fetchone()
        occupied = db.execute('SELECT 1 FROM memories WHERE id=?', (memory_id,)).fetchone()
        if old or occupied:
            if not old or not is_test_seeded(json.loads(old[0])) or json.loads(old[0]).get('request_sha256') != fingerprint:
                raise StateConflictError('memory ID already binds different content or origin')
            return service.store.get_memory(memory_id)
        created_at = time.time()
        artifact = {'kind': 'text_hint', 'guidance': guidance, 'repair': stored_repair,
                    'applicability': context}
        provenance = {'schema_version': 'attentionbench.memory-provenance.v2',
                      'channel': 'test_seeded', 'source_kind': 'seeded_for_tests',
                      'trust_scope': 'test_seeded', 'excluded_from_results': True,
                      'seeded_for_tests': True, 'source_trace_id': trace,
                      'source_seed': context['seed'], 'request_sha256': fingerprint,
                      **({'title': title} if title is not None else {}),
                      **({'summary': summary} if summary is not None else {}),
                      **({'group': group} if group is not None else {}),
                      'artifact': artifact,
                      'artifact_sha256': hashlib.sha256(encoded(artifact).encode()).hexdigest()}
        memory = MemoryRecord(memory_id=memory_id, version=1, source_trace_id=trace,
                              guidance=guidance, candidate_repair=stored_repair,
                              applicability=context, evidence_refs=(trace,),
                              created_at=created_at, status=MemoryStatus.TRUSTED,
                              status_reason='test_seeded_excluded_from_results')
        db.execute('INSERT INTO memories(id,payload) VALUES (?,?)',
                   (memory_id, encoded(memory.artifact())))
        db.execute('INSERT INTO memory_v2_provenance(memory_id,payload) VALUES (?,?)',
                   (memory_id, encoded(provenance)))
        db.execute('INSERT INTO events(event_key,entity_type,entity_id,event_type,payload) '
                   'VALUES (?,\'memory\',?,\'memory.test_seeded\',?)',
                   (f'memory-test-seeded:{memory_id}', memory_id,
                    encoded({'memory': memory.artifact(), 'actor': actor, 'reason': reason,
                             'recorded_at': created_at, 'excluded_from_results': True})))
    return memory
