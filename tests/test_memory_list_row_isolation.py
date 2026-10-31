"""Listing isolates unavailable evidence; authority operations still fail closed."""
import ast
import json
from pathlib import Path
import sqlite3
import subprocess

import pytest

from attention_memory_service import memory_service, motion_permission
from attention_memory_service.core.store import StateConflictError
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.memory_service_api import create_app
from attention_memory_service.memory_service_client import MemoryServiceClient
from tests.test_provisional_notes import context, payload
from tests.test_sync_legacy_routes import sync_request
from tests.test_test_seeded_notes import payload as seeded_payload

BASE = '980199e95966db334c2d43652a72f019a423f38b'
API_KEY = 'row-isolation-test-key'


def base_listing(service, **kwargs):
    """Execute the frozen base implementation against the same synthetic store."""
    source = subprocess.check_output(
        ['git', 'show', f'{BASE}:attention_memory_service/memory_service.py'],
        cwd=Path(__file__).resolve().parents[1], text=True)
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == 'MemoryService')
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'list_dev_v2_memories')
    namespace = dict(vars(memory_service))
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<base-listing>', 'exec'), namespace)
    return namespace[method.name](service, **kwargs)


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr('time.time', lambda: 2000000000.0)
    svc = MemoryService(tmp_path / 'memory.sqlite')
    for mid in ('p-healthy', 'p-bad', 'p-disabled'):
        svc.create_provisional_note(**payload(mid))
    svc.disable('p-disabled', actor='test', reason='test disabled projection')
    svc.create_test_seeded_note(**seeded_payload())
    return svc


def corrupt(service, mid='p-bad'):
    with sqlite3.connect(service.store.path) as db:
        raw, = db.execute('SELECT payload FROM memory_v2_provenance WHERE memory_id=?', (mid,)).fetchone()
        value = json.loads(raw)
        value['artifact_sha256'] = '0' * 64
        db.execute('UPDATE memory_v2_provenance SET payload=? WHERE memory_id=?',
                   (json.dumps(value), mid))


@pytest.mark.parametrize('query', [
    {}, {'status': 'provisional'}, {'status': 'disabled'},
    {'context': context()}, {'context': context(seed=103)},
    {'context': context(task_id='another-task')},
    {'context': context(suite='another-suite')},
    {'context': context(), 'status': 'trusted'},
])
def test_healthy_rows_are_byte_identical_to_base(service, query):
    golden = wire(base_listing(service, **query))
    result = service.dev_v2_memory_listing(**query)
    assert result['unavailable'] == []
    assert wire(result['memories']) == golden
    assert wire(service.list_dev_v2_memories(**query)) == golden


@pytest.mark.parametrize('with_context', [False, True])
def test_corrupt_row_is_omitted_and_diagnosed(service, with_context, caplog):
    query = {'context': context()} if with_context else {}
    golden = base_listing(service, **query)
    corrupt(service)
    with pytest.raises(StateConflictError, match='artifact digest mismatch'):
        base_listing(service, **query)
    result = service.dev_v2_memory_listing(**query)
    assert wire(result['memories']) == wire([row for row in golden if row['memory_id'] != 'p-bad'])
    assert result['unavailable'] == [dict(memory_id='p-bad', error_class='StateConflictError',
                                          reason='memory artifact digest mismatch')]
    assert 'p-bad' in caplog.text and 'StateConflictError' in caplog.text
    assert service.dev_v2_memory_listing(status='trusted')['unavailable'] == []
    with pytest.raises(FileNotFoundError):
        service.get_dev_v2_memory_public('p-bad')


@pytest.mark.parametrize('stage', ['get_plan', 'get_decision', 'public_summary'])
@pytest.mark.parametrize('error', [StateConflictError, FileNotFoundError])
def test_late_verification_failure_is_local(service, monkeypatch, stage, error):
    owner = motion_permission if stage == 'public_summary' else service.v2
    original = getattr(owner, stage)

    def fail(*args, **kwargs):
        mid = args[1] if stage == 'public_summary' else args[0]
        if mid == 'p-bad':
            raise error('sealed evidence unavailable')
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, stage, fail)
    result = service.dev_v2_memory_listing(context=context())
    assert len(result['memories']) == 3
    assert result['unavailable'] == [dict(memory_id='p-bad', error_class=error.__name__,
                                          reason='sealed evidence unavailable')]


@pytest.mark.parametrize('error', [ValueError, TypeError, KeyError, RuntimeError])
def test_programming_errors_still_raise(service, monkeypatch, error):
    def broken(*args, **kwargs):
        raise error('projection bug')
    monkeypatch.setattr(motion_permission, 'public_summary', broken)
    with pytest.raises(error, match='projection bug'):
        service.dev_v2_memory_listing()


@pytest.mark.parametrize('bad_context', [
    {'perception_mode': 'unknown'},
    {'perception_mode': 'sim_gt', 'sdk_interface_sha256': 'bad'},
    {'perception_mode': 'sim_gt', 'execution_profile': 'legacy'},
])
def test_invalid_query_context_still_raises(service, bad_context):
    with pytest.raises(ValueError):
        service.dev_v2_memory_listing(context=bad_context)


def test_retrieval_and_dev_grant_remain_strict(service):
    corrupt(service)
    with pytest.raises(StateConflictError, match='artifact digest mismatch'):
        service.retrieve_levels(context(), now=2000000000.0)
    with pytest.raises(StateConflictError, match='artifact digest mismatch'):
        service.authorize_dev_use(memory_id='p-bad', development_id='test-dev',
            context=context(), source_policy_sha256='1' * 64, now=2000000000.0)
    with sqlite3.connect(service.store.path) as db:
        assert db.execute('SELECT COUNT(*) FROM memory_dev_exposures').fetchone()[0] == 0


def test_http_shape_and_unchanged_client(service, monkeypatch):
    corrupt(service)
    app = create_app(service.store.path, api_key=API_KEY)
    expected = service.dev_v2_memory_listing()
    code, response = sync_request(app, '/dev-v2/memories', key=API_KEY)
    assert code == 200 and response == expected
    code, response = sync_request(app, '/dev-v2/memories/query', method='POST',
                                  payload={'context': context()}, key=API_KEY)
    assert code == 200 and response == service.dev_v2_memory_listing(context=context())
    assert sync_request(app, '/dev-v2/memories')[0] == 401
    client = MemoryServiceClient('https://unused', api_key=API_KEY)
    monkeypatch.setattr(client, '_call', lambda *args, **kwargs: expected)
    assert client.list_dev_v2_memories() == expected['memories']
    assert client.list_dev_v2_memories(context=context()) == expected['memories']


def test_agent_provisional_rows_stay_out_of_the_listing(service):
    # Isolation only: the listing allowlist is unchanged, so dev_agent rows stay
    # out of this public projection (they are delivered only via retrieval grants).
    with sqlite3.connect(service.store.path) as db:
        raw, = db.execute('SELECT payload FROM memory_v2_provenance WHERE memory_id=?', ('p-healthy',)).fetchone()
        value = json.loads(raw)
        value['channel'] = 'dev_agent'
        value['origin']['channel'] = 'dev_agent'
        db.execute('UPDATE memory_v2_provenance SET payload=? WHERE memory_id=?',
                   (json.dumps(value), 'p-healthy'))
    result = service.dev_v2_memory_listing(context=context())
    assert all(row['memory_id'] != 'p-healthy' for row in result['memories'])
    assert result['unavailable'] == []


@pytest.mark.parametrize('with_context', [False, True])
def test_changed_sealed_source_omits_only_its_candidate(tmp_path, with_context):
    from tests.dev_v2_fixtures import Fixture

    fixture = Fixture(tmp_path)
    failed = fixture.make('failed')
    fixture.add_advice(failed)
    svc = MemoryService(tmp_path / 'authority.sqlite', dev_v2_catalogs=fixture.seal())
    svc.import_dev_v2_execution(import_id='failed', catalog_id='failed', descriptor='failed/execution.json')
    svc.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')
    svc.create_dev_v2_candidate(memory_id='candidate', source_import_id='source', created_at=1.0)
    svc.create_test_seeded_note(**seeded_payload())
    query = {'context': {**failed['context'], 'seed': 101}} if with_context else {}
    golden = base_listing(svc, **query)
    assert wire(svc.list_dev_v2_memories(**query)) == wire(golden)
    Path(failed['record']['source']['uri']).write_text('# source changed after sealing\n')
    with pytest.raises(StateConflictError):
        base_listing(svc, **query)
    result = svc.dev_v2_memory_listing(**query)
    assert wire(result['memories']) == wire([row for row in golden if row['memory_id'] != 'candidate'])
    assert len(result['unavailable']) == 1
    assert result['unavailable'][0]['memory_id'] == 'candidate'
    assert result['unavailable'][0]['error_class'] == 'StateConflictError'
    with pytest.raises(StateConflictError):
        svc.provenance('candidate')
