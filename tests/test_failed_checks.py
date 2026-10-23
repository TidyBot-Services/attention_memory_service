"""Negative independent receipts close a batch, never create a passing trial."""
import sqlite3
import time

import pytest

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.memory_service_api import create_app
from attention_memory_service.memory_service_client import MemoryServiceClient
from tests.test_auto_validation import outcome, OPERATOR
from tests.test_provisional_notes import API, IMPORTER, CHANGED, sha
from tests.test_single_scene import setup_scene
from tests.test_sync_legacy_routes import sync_request

PID = 'p-' + 'a' * 24


def setup(tmp_path, *, arm='t0', safety=False, scene=True, fault=None):
    f, service, plan = setup_scene(tmp_path, linked=True, interface=True)
    ctx = service.store.get_memory(PID).applicability
    outcome(service, PID, 0, CHANGED, ctx)
    service.record_plan('m1', plan)
    r = f.records[arm]
    grant_args = dict(memory_id='m1', cell=arm, version=1, seed=101, context=r['context'],
        policy_sha256=r['record']['source']['sha256'], config_sha256=r['descriptor']['config']['sha256'],
        exposure=r['trial']['exposure'])
    grant = service.authorize_dev_v2_validation(**grant_args)
    f.bind_validation(r, grant)
    r['verdict']['checks'].update(safety=safety, scene_reproduction=scene)
    if fault == 'forged': r['verdict']['bindings']['seed'] = 102
    if fault in {'native_evaluation', 'permissions', 'source_identity'}: r['verdict']['checks'][fault] = False
    f.write(f.validator / arm / 'verdict.json', r['verdict'])
    f.write(f.validator / arm / 'alias.json', r['verdict'])
    cats = f.seal()
    for cid in (arm, 'validator-' + arm): service.dev_v2.register(cid, cats[cid])
    service.import_dev_v2_execution(import_id=arm, catalog_id=arm, descriptor=arm + '/execution.json')
    args = dict(memory_id='m1', validation_grant_id=grant['grant_id'], execution_import_id=arm,
                catalog_id='validator-' + arm, descriptor=arm + '/verdict.json')
    if fault == 'digest': (f.validator / arm / 'verdict.json').write_text('{}')
    if fault == 'no_grant': args['validation_grant_id'] = 'unissued'
    if fault == 'wrong_memory': args['memory_id'] = PID
    if fault == 'wrong_role': args['catalog_id'] = arm
    return f, service, args, grant_args, ctx


@pytest.mark.parametrize('arm,safety,scene,decision,gate', [
    ('t0', False, True, 'rejected', 'safety_regression'),
    ('c0', False, True, 'inconclusive', 'control_safety_failed'),
    ('c0', True, False, 'inconclusive', 'scene_reproduction_failed'),
    ('t0', True, False, 'inconclusive', 'scene_reproduction_failed'),
    ('c0', False, False, 'inconclusive', 'scene_reproduction_failed'),
    ('t0', False, False, 'inconclusive', 'scene_reproduction_failed'),
])
def test_failed_check_closes_batch_and_never_imports_trial(tmp_path, arm, safety, scene, decision, gate):
    _, service, args, grant_args, ctx = setup(tmp_path, arm=arm, safety=safety, scene=scene)
    delayed = service.authorize_dev_use(memory_id=PID, memory_version=1, development_id='delayed:v1',
        context=ctx, source_policy_sha256='c' * 64, now=time.time())
    with pytest.raises(StateConflictError, match='safety/scene'):
        service.import_dev_v2_trial(import_id='bad-trial', catalog_id=args['catalog_id'], descriptor=arm + '/trial.json')
    result = service.record_dev_v2_failed_check(**args)
    assert result['decision'] == decision and result['failed_gates'] == [gate]
    assert result['report']['paired_runs'] == 0
    assert service.record_dev_v2_failed_check(**args) == result
    assert MemoryService(service.store.path).record_dev_v2_failed_check(**args) == result
    with pytest.raises(StateConflictError, match='conflicting'):
        service.record_dev_v2_failed_check(**{**args, 'descriptor': arm + '/alias.json'})
    assert service.store.get_memory('m1').status.value == ('rejected' if decision == 'rejected' else 'candidate')
    assert service.evaluate('m1').status == service.store.get_memory('m1').status
    assert service.promote('m1').status == service.store.get_memory('m1').status
    with pytest.raises(StateConflictError): service.v2.validate_and_promote('m1')
    with pytest.raises(StateConflictError, match='closed|candidate'):
        service.authorize_dev_v2_validation(**grant_args)
    assert service.validation_queue() == []
    assert [e['event'] for e in service.memory_events()] == [decision, decision]
    assert service.memory_events()[0]['reason'] == result['reason']
    assert service.memory_events()[1]['reason'] == ('rejected after automatic check' if decision == 'rejected'
                                                    else result['reason'])
    public = service.get_dev_v2_memory_public('m1')
    assert public['validation_summary']['decision'] == decision
    assert service.v2.list_pairs('m1') == []
    with sqlite3.connect(service.store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM dev_v2_imports WHERE kind='trial'").fetchone()[0] == 0
    if decision == 'rejected':
        assert result['reason'] == 'Rejected: a run with the note failed the independent safety check'
        assert service.store.get_memory(PID).status.value == 'disabled'
        return
    assert service.store.get_memory(PID).status.value == 'provisional'
    assert PID in [m.memory_id for m in service.retrieve(ctx, now=time.time())]
    expected = 'the scene did not reproduce' if not scene else 'a run without the note failed the safety check'
    assert result['reason'] == 'Could not finish the check: ' + expected
    # A late report of a pre-decision use does not restart validation.
    service.record_provisional_outcome(grant_id=delayed['grant_id'], development_id=delayed['development_id'],
        public_feedback_text=CHANGED, public_feedback_sha256=sha(CHANGED))
    assert service.validation_queue() == []
    outcome(service, PID, 1, CHANGED, ctx)
    assert [r['memory_id'] for r in service.validation_queue()] == [PID]
    service.create_dev_v2_candidate(memory_id='m2', source_import_id='source', created_at=time.time(), from_provisional=PID)
    assert service.validation_queue() == []
    assert service.decision('m1') == result


@pytest.mark.parametrize('fault', ['forged', 'digest', 'native_evaluation', 'permissions', 'source_identity',
                                  'no_grant', 'wrong_memory', 'wrong_role', 'all_true'])
def test_failed_check_refuses_untrusted_evidence(tmp_path, fault):
    _, service, args, _, _ = setup(tmp_path, safety=fault == 'all_true', fault=fault)
    with pytest.raises(StateConflictError): service.record_dev_v2_failed_check(**args)
    assert service.decision('m1') is None
    assert service.store.get_memory('m1').status.value == 'candidate'
    assert service.memory_events() == []


def test_failed_check_route_auth_and_client(tmp_path, monkeypatch):
    _, service, args, _, _ = setup(tmp_path)
    app = create_app(service.store.path, api_key=API, importer_key=IMPORTER, operator_key=OPERATOR)
    for headers, status in [({}, 401), ({'key': API}, 403), ({'key': API, 'operator_key': OPERATOR}, 403)]:
        assert sync_request(app, '/dev-v2/failed-checks', method='POST', payload=args, **headers)[0] == status
    client = MemoryServiceClient('http://localhost:1', api_key=API, importer_key=IMPORTER)
    def call(method, path, payload, *, importer=False):
        assert importer
        status, response = sync_request(app, path, method=method, payload=payload, key=API, importer_key=IMPORTER)
        assert status == 200
        return response
    monkeypatch.setattr(client, '_call', call)
    result = client.record_dev_v2_failed_check(**args)
    assert result['decision'] == 'rejected'
    assert sync_request(app, '/dev-v2/failed-checks', method='POST',
        payload={**args, 'descriptor': 't0/alias.json'}, key=API, importer_key=IMPORTER)[0] == 409


def test_failed_check_transaction_rolls_back_on_notice_error(tmp_path, monkeypatch):
    from attention_memory_service import auto_validation
    _, service, args, _, _ = setup(tmp_path)
    def broken(*args, **kwargs): raise RuntimeError('injected notice failure')
    monkeypatch.setattr(auto_validation, 'lifecycle', broken)
    with pytest.raises(RuntimeError): service.record_dev_v2_failed_check(**args)
    assert service.decision('m1') is None
    assert service.store.get_memory('m1').status.value == 'candidate'
    assert service.store.get_memory(PID).status.value == 'provisional'


def test_generated_arm_can_close_with_failed_check(tmp_path):
    from tests.test_arm_generation import setup as generated_setup, prepare
    f, service, plan = generated_setup(tmp_path)
    service.record_plan('m1', plan)
    _, grant = prepare(f, service, plan, 't0', checks={'safety': False})
    result = service.record_dev_v2_failed_check(memory_id='m1', validation_grant_id=grant['grant_id'],
        execution_import_id='t0', catalog_id='validator-t0', descriptor='t0/verdict.json')
    assert result['decision'] == 'rejected'
    assert result['failed_gates'] == ['safety_regression']
