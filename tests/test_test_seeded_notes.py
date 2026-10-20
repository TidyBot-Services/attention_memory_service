"""Importer-only seeded notes use a disposable database and fake scope."""

import copy
import time

import pytest
from pydantic import ValidationError

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.core.models import MemoryUseRecord
from attention_memory_service.dev_v2 import SCOPE
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.memory_service_api import SeededNoteInput, create_app
from tests.test_sync_legacy_routes import sync_request
from tests.test_store import populated_store

API = 'online-key-123456789'
OPERATOR = 'operator-key-123456789'
IMPORTER = 'importer-key-123456789'


def payload():
    context = {key: f'fake-{key}' for key in SCOPE}
    context.update(sdk_sha256='a' * 64, perception_mode='sim_gt', seed=101,
                   camera_names=['left', 'right'], task_prompt='Move fake cup',
                   scene_contract_ref={'uri': 'fixture://scene', 'sha256': 'b' * 64})
    return dict(actor='tester', reason='exercise read and revoke', memory_id='seed-1',
                context=context, guidance='private test guidance', repair='fake repair')


def service(tmp_path):
    return MemoryService(tmp_path / 'memory.sqlite')


def test_seeded_creation_scope_audit_and_validation_isolation(tmp_path):
    svc = service(tmp_path)
    req = payload()
    created = svc.create_test_seeded_note(**req)
    assert created == dict(memory_id='seed-1', version=1, status='trusted',
                           source_kind='seeded_for_tests', source_label='seeded for tests',
                           seeded_for_tests=True, excluded_from_results=True,
                           trust_scope='test_seeded')
    assert svc.create_test_seeded_note(**req) == created
    assert len(svc.history('seed-1')) == 1
    assert svc.history('seed-1')[0]['actor'] == 'tester'
    assert svc.history('seed-1')[0]['reason'] == req['reason']
    audit = next(event['payload'] for event in svc.store.events()
                 if event['event_type'] == 'memory.test_seeded')
    assert audit['recorded_at'] > 0
    assert [m.memory_id for m in svc.retrieve(req['context'], now=time.time())] == ['seed-1']
    assert svc.list_memories() == []
    for key in SCOPE:
        context = copy.deepcopy(req['context'])
        context[key] = ('vision' if key == 'perception_mode' else 'c' * 64 if key == 'sdk_sha256' else 'other')
        assert svc.retrieve(context, now=time.time()) == [], key
    for key, different in [('seed', 102), ('scene_id', 'other'), ('task_id', 'other'),
                           ('camera_names', ['right', 'left']), ('task_prompt', 'other'),
                           ('scene_contract_ref', {'uri': 'other', 'sha256': 'b' * 64}),
                           ('robot', 'other')]:
        context = copy.deepcopy(req['context']); context[key] = different
        assert svc.retrieve(context, now=time.time()) == [], key
    context = copy.deepcopy(req['context']); context.pop('scene_contract_ref')
    assert svc.retrieve(context, now=time.time()) == []
    with pytest.raises(StateConflictError):
        svc.create_test_seeded_note(**{**req, 'guidance': 'changed'})
    for call in (lambda: svc.record_plan('seed-1', {'profile': 'default'}),
                 lambda: svc.promote('seed-1'), lambda: svc.evaluate('seed-1'),
                 lambda: svc.impact_report('seed-1')):
        with pytest.raises(StateConflictError):
            call()
    assert svc.decision('seed-1') is None
    assert svc.list_pairs('seed-1') == []


def test_seeded_title_is_public_trimmed_and_part_of_idempotence(tmp_path):
    svc = service(tmp_path)
    req = {**payload(), 'title': '  Take small, retried arm steps  '}
    svc.create_test_seeded_note(**req)
    assert svc.v2.provenance('seed-1')['title'] == 'Take small, retried arm steps'
    assert svc.get_dev_v2_memory_public('seed-1')['title'] == 'Take small, retried arm steps'
    assert svc.create_test_seeded_note(**{**req, 'title': 'Take small, retried arm steps'})['memory_id'] == 'seed-1'
    with pytest.raises(StateConflictError, match='different content'):
        svc.create_test_seeded_note(**{**req, 'title': 'A different title'})
    with pytest.raises(StateConflictError, match='different content'):
        svc.create_test_seeded_note(**{k: v for k, v in req.items() if k != 'title'})
    untitled = {**payload(), 'memory_id': 'seed-2'}
    svc.create_test_seeded_note(**untitled)
    assert svc.create_test_seeded_note(**untitled)['memory_id'] == 'seed-2'
    assert svc.get_dev_v2_memory_public('seed-2')['title'] is None
    assert 'title' not in svc.v2.provenance('seed-2')


@pytest.mark.parametrize('title, message', [
    ('   ', '1–80'), ('x' * 81, '1–80'), ('First\nsecond', 'newline'),
    ('A\x00B', 'control'), ('See https://example.com', 'URL'),
    ('See docs/guide', 'path'), ('C:\\private\\key', 'path'),
    ('sk-' + 'a' * 24, 'key-like'), ('api_key=topsecret', 'key-like'),
    ('AKIA' + 'A' * 16, 'key-like'), ('oracle result', 'oracle'),
    ('native evaluator score', 'oracle'), ('原生评估器得分', 'oracle'),
    ('<b>markup</b>', 'plain text'),
])
def test_seeded_title_validation(tmp_path, title, message):
    with pytest.raises(ValidationError, match=message):
        SeededNoteInput.model_validate({**payload(), 'title': title})
    with pytest.raises(ValueError, match=message):
        service(tmp_path).create_test_seeded_note(**{**payload(), 'title': title})


def test_seeded_title_http_directory_and_detail_do_not_leak(tmp_path):
    app = create_app(tmp_path / 'memory.sqlite', api_key=API, importer_key=IMPORTER)
    req = {**payload(), 'title': '  Take small, retried arm steps  '}
    status, _ = sync_request(app, '/dev-v2/test-seeded-notes', method='POST',
                             payload=req, key=API, importer_key=IMPORTER)
    assert status == 200
    for path in ('/dev-v2/memories', '/dev-v2/memories/seed-1'):
        status, result = sync_request(app, path, key=API)
        assert status == 200
        row = result['memories'][0] if 'memories' in result else result
        assert row['title'] == 'Take small, retried arm steps'
        assert req['guidance'] not in str(row) and req['repair'] not in str(row)
        assert not {'artifact', 'request_sha256', 'source_trace_id', 'grant', 'descriptor'} & row.keys()


def test_seeded_grants_catalog_and_revocation(tmp_path):
    svc = service(tmp_path); req = payload(); svc.create_test_seeded_note(**req)
    grant = svc.authorize_use(req['context'], memory_id='seed-1', attempt_id='fake-attempt', now=1)
    assert grant['seeded_for_tests'] is True and grant['excluded_from_results'] is True
    dev = svc.authorize_dev_use(memory_id='seed-1', development_id='fake-dev',
                                context=req['context'], source_policy_sha256='c' * 64,
                                now=1, memory_version=1)
    assert dev['seeded_for_tests'] is True and dev['excluded_from_results'] is True
    assert svc.get_dev_use(dev['grant_id'])['excluded_from_results'] is True
    with pytest.raises(StateConflictError):
        svc.record_dev_result(grant_id=dev['grant_id'], run_id='fake-run',
                              result_policy_sha256='d' * 64, artifact_uri='fake', artifact_sha256='e' * 64)
    public = svc.get_dev_v2_memory_public('seed-1')
    assert public['trust_scope'] == 'test_seeded'
    assert public['source_label'] == 'seeded for tests'
    assert public['excluded_from_results'] is True
    assert public['guidance_length'] == len(req['guidance'])
    assert req['guidance'] not in str(public) and req['repair'] not in str(public)
    svc.disable('seed-1', actor='tester', reason='stop test')
    assert svc.retrieve(req['context'], now=1) == []
    with pytest.raises(StateConflictError):
        svc.get_dev_use(dev['grant_id'])
    svc.enable('seed-1', actor='tester', reason='retry test')
    svc.set_expiry('seed-1', expires_at=time.time() - 1, actor='tester', reason='end test')
    assert svc.retrieve(req['context'], now=1) == []
    with pytest.raises(StateConflictError):
        svc.enable('seed-1', actor='tester', reason='cannot revive')
    req2 = {**req, 'memory_id': 'seed-2'}
    svc.create_test_seeded_note(**req2)
    svc.rollback('seed-2', actor='tester', reason='end test')
    with pytest.raises(StateConflictError):
        svc.enable('seed-2', actor='tester', reason='cannot revive')


def test_seeded_importer_only_and_required_actor_reason(tmp_path):
    app = create_app(tmp_path / 'memory.sqlite', api_key=API, operator_key=OPERATOR,
                     importer_key=IMPORTER)
    req = payload()
    for absent in ('actor', 'reason'):
        with pytest.raises(ValidationError):
            SeededNoteInput.model_validate({k: v for k, v in req.items() if k != absent})
        status, _ = sync_request(app, '/dev-v2/test-seeded-notes', method='POST',
                                 payload={**req, absent: '   '}, key=API, importer_key=IMPORTER)
        assert status == 422
    for kwargs in ({'key': API}, {'key': API, 'operator_key': OPERATOR}):
        status, _ = sync_request(app, '/dev-v2/test-seeded-notes', method='POST',
                                 payload=req, **kwargs)
        assert status == 403
    status, result = sync_request(app, '/dev-v2/test-seeded-notes', method='POST',
                                  payload=req, key=API, importer_key=IMPORTER)
    assert status == 200 and result['excluded_from_results'] is True
    status, rows = sync_request(app, '/dev-v2/memories', key=API)
    assert status == 200 and len(rows['memories']) == 1
    assert req['guidance'] not in str(rows)
    status, records = sync_request(app, '/retrieve', method='POST',
                                   payload={'context': req['context'], 'now': 1}, key=API)
    assert status == 200 and records[0]['seeded_for_tests'] is True


def test_seeded_use_record_persists_exclusion_flags(tmp_path, monkeypatch):
    path = tmp_path / 'memory.sqlite'
    populated_store(path)
    svc = MemoryService(path); svc.create_test_seeded_note(**payload())
    # This test isolates persistence from the separate native trace verifier.
    monkeypatch.setattr(svc.v2, 'verify_use', lambda use: None)
    use = MemoryUseRecord('fake-use', 'seed-1', 1, 'run-1', 'attempt-1', 3.0, 'failed')
    svc.record_use(use)
    assert svc.store.list_memory_uses('seed-1')[0]['excluded_from_results'] is True
    assert svc.uses('seed-1')[0]['seeded_for_tests'] is True
