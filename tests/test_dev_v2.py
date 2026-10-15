"""Controlled synthetic boundary tests; never evidence of real provider/native use."""
import copy
import json
import sqlite3
import time
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.memory_service_api import create_app, ImportInput, RetrieveInput
from attention_memory_service.dev_v2 import SCOPE, digest
from tests.dev_v2_fixtures import Fixture


@pytest.fixture
def source_setup(tmp_path):
    f = Fixture(tmp_path)
    failed = f.make('failed')
    f.add_advice(failed)
    service = MemoryService(tmp_path / 'authority.sqlite', dev_v2_catalogs=f.seal())
    service.import_dev_v2_execution(import_id='failed', catalog_id='failed', descriptor='failed/execution.json')
    return f, service


def candidate(source_setup):
    f, service = source_setup
    s = service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')
    m = service.create_dev_v2_candidate(memory_id='m1', source_import_id='source', created_at=1.0)
    return f, service, s, m


def route(app, path, method='POST'):
    return next(r for r in app.routes if r.path == path and method in r.methods)


def test_source_candidate_and_restart_are_evidence_backed(source_setup):
    f, service, s, m = candidate(source_setup)
    assert m.status.value == 'candidate'
    assert m.guidance == s['guidance'] and m.candidate_repair == s['repair']
    assert service.retrieve(f.records['failed']['context'], now=time.time()) == []
    assert service.create_dev_v2_candidate(memory_id='m1', source_import_id='source', created_at=1.0) == m
    assert service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json') == s
    reopened = MemoryService(service.store.path)
    assert reopened.check_candidate('m1')['source_evidence_verified']
    assert reopened.verify_package('m1')['status'] == 'candidate'
    with sqlite3.connect(service.store.path) as db:
        assert db.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 0
        assert db.execute('SELECT COUNT(*) FROM requests').fetchone()[0] == 0
        assert db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == 0
    with pytest.raises(StateConflictError, match='all predefined|five distinct'):
        service.promote('m1')
    Path(f.records['failed']['record']['source']['uri']).write_text('tampered')
    with pytest.raises(StateConflictError, match='digest changed'):
        reopened.check_candidate('m1')


@pytest.mark.parametrize('file', ['failed/ledger.sqlite', 'failed/advisor-wire.json', 'failed/trace.json', 'failed/native.json', 'failed/help.json', 'failed/repair.py', 'lock.json', 'openarm365_dev_v2/sdk.py'])
def test_missing_or_tampered_source_rejected(source_setup, file):
    f, service = source_setup
    (f.root / file).write_bytes(b'{}')
    with pytest.raises((StateConflictError, KeyError)):
        service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')


@pytest.mark.parametrize('defect', ['unknown_cost', 'not_adopted', 'unsettled', 'provider_id', 'wrong_wire', 'scope', 'success', 'scene', 'no_evaluation', 'permission'])
def test_semantically_invalid_sealed_source_rejected(tmp_path, defect):
    f = Fixture(tmp_path); failed = f.make('failed'); f.add_advice(failed)
    directory = f.root / 'failed'; ledger = directory / 'ledger.sqlite'
    if defect in {'unknown_cost', 'not_adopted', 'unsettled'}:
        with sqlite3.connect(ledger) as db:
            if defect == 'not_adopted':
                row = json.loads(db.execute('SELECT payload FROM v2_planner_requests').fetchone()[0]); row['adopted'] = False
                db.execute('UPDATE v2_planner_requests SET payload=?', (json.dumps(row),))
            else:
                row = json.loads(db.execute("SELECT payload FROM v2_jobs WHERE kind='advisor'").fetchone()[0])
                if defect == 'unknown_cost': row['actual'].pop('provider_tokens')
                db.execute("UPDATE v2_jobs SET payload=?,state=? WHERE kind='advisor'", (json.dumps(row), 'needs_review' if defect == 'unsettled' else 'settled'))
        for file in ('execution.json', 'source.json'):
            value = json.loads((directory / file).read_bytes()); value['ledger']['sha256'] = digest(ledger.read_bytes()); f.write(directory / file, value)
    elif defect in {'provider_id', 'wrong_wire'}:
        value = json.loads((directory / 'advisor-wire.json').read_bytes())
        if defect == 'provider_id': value['request_id'] = None
        else: value['body']['choices'][0]['message']['content'] = 'different hint'
        f.write(directory / 'advisor-wire.json', value)
    elif defect == 'scope':
        value = json.loads((directory / 'source.json').read_bytes()); value['context']['sdk_sha256'] = 'b' * 64; f.write(directory / 'source.json', value)
    else:
        verdict = failed['verdict']
        if defect == 'success': verdict['native_success'] = True
        elif defect == 'scene':
            f.write(f.validator / 'failed/obs2.json', {'objects': {'cup': [9, 2, 3]}, 'evidence_kind': 'controlled_test'})
        elif defect == 'no_evaluation': verdict['checks']['native_evaluation'] = False
        else: verdict['permission_scope'] = {'mode': 'unapproved'}
        f.write(f.validator / 'failed/verdict.json', verdict)
    service = MemoryService(tmp_path / 'authority.sqlite', dev_v2_catalogs=f.seal())
    # Changed wire/observation refs may fail as early as preregistered SHA checks.
    with pytest.raises((StateConflictError, KeyError)):
        service.import_dev_v2_execution(import_id='failed', catalog_id='failed', descriptor='failed/execution.json')
        service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')


def test_roots_roles_paths_and_catalog_immutability(source_setup):
    f, service = source_setup
    for path in ('../failed/execution.json', '/etc/passwd', 'not-registered.json'):
        with pytest.raises(StateConflictError):
            service.import_dev_v2_execution(import_id='bad', catalog_id='failed', descriptor=path)
    with pytest.raises(StateConflictError, match='role'):
        service.import_dev_v2_execution(import_id='bad', catalog_id='validator-failed', descriptor='failed/verdict.json')
    with pytest.raises(StateConflictError, match='unregistered'):
        service.import_dev_v2_execution(import_id='bad', catalog_id='absent', descriptor='failed/execution.json')
    changed = copy.deepcopy(f.catalogs['failed']); changed['context']['namespace'] = 'other'
    with pytest.raises(StateConflictError, match='conflict'):
        service.dev_v2.register('failed', changed)
    linked = f.root / 'linked'; linked.symlink_to(f.root / 'failed/execution.json')
    changed = copy.deepcopy(f.catalogs['failed']); changed['files']['linked'] = digest(linked.read_bytes())
    with pytest.raises(StateConflictError, match='symlink'):
        service.dev_v2.register('linked', changed)


def make_promoted(tmp_path):
    f = Fixture(tmp_path); failed = f.make('failed'); f.add_advice(failed)
    cases = []
    for i, seed in enumerate(range(101, 106)):
        c = f.make(f'c{seed}', seed=seed, variant=i % 2, success=False)
        t = f.make(f't{seed}', seed=seed, variant=i % 2, success=True, exposure='candidate_validation', memory_id='m1')
        cases.append({'seed': seed, **{k: c['context'][k] for k in ('scene_id', 'object_set_id', 'camera_config_id', 'task_variant_id', 'camera_names', 'task_prompt')}})
    service = MemoryService(tmp_path / 'authority.sqlite', dev_v2_catalogs={k: v for k, v in f.seal().items() if k in {'failed', 'validator-failed'}})
    service.import_dev_v2_execution(import_id='failed', catalog_id='failed', descriptor='failed/execution.json')
    service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')
    service.create_dev_v2_candidate(memory_id='m1', source_import_id='source', created_at=1.0)
    ctx = failed['context']
    plan = {'schema_version': 'attentionbench.memory-validation-plan.v2', 'memory_id': 'm1', **{k: ctx[k] for k in SCOPE if k not in ('scene_id', 'object_set_id', 'camera_config_id', 'task_variant_id')}, 'policy_id': 'full_trace_aware_attention_planner', 'seeds': list(range(101, 106)), 'cases': cases, 'assistance_credits': 1}
    service.record_plan('m1', plan)
    for seed in range(101, 106):
        for arm in ('c', 't'):
            name = f'{arm}{seed}'
            r = f.records[name]
            grant = service.authorize_dev_v2_validation(memory_id='m1', cell=name, version=1, seed=seed, context=r['context'], policy_sha256=r['record']['source']['sha256'], config_sha256=r['descriptor']['config']['sha256'], exposure=r['trial']['exposure'])
            f.bind_validation(r, grant)
            catalogs = f.seal()
            for cid in (name, 'validator-' + name):
                service.dev_v2.register(cid, catalogs[cid])
            service.import_dev_v2_execution(import_id=name, catalog_id=name, descriptor=name + '/execution.json')
            service.import_dev_v2_trial(import_id='trial-' + name, catalog_id='validator-' + name, descriptor=name + '/trial.json')
        service.record_dev_v2_pair(memory_id='m1', control_import_id=f'trial-c{seed}', treatment_import_id=f'trial-t{seed}')
    trusted = service.promote('m1')
    assert trusted.status.value == 'trusted'
    return f, service, trusted


def test_empty_authority_to_promotion_exact_grants_and_lifecycle(tmp_path):
    f, service, trusted = make_promoted(tmp_path)
    ctx = f.records['failed']['context']
    assert [m.memory_id for m in service.retrieve(ctx, now=time.time())] == ['m1']
    for k in SCOPE:
        assert service.retrieve({**ctx, k: 'other' if k != 'perception_mode' else 'vision'}, now=time.time()) == []
    for key in ('sdk_sha256', 'namespace', 'camera_names', 'task_prompt'):
        assert service.retrieve({k: v for k, v in ctx.items() if k != key}, now=time.time()) == []
    args = {'memory_id': 'm1', 'development_id': 'future:v1', 'context': ctx, 'source_policy_sha256': 'a' * 64, 'now': time.time(), 'memory_version': trusted.version}
    grant = service.authorize_dev_use(**args)
    assert grant['memory_version'] == trusted.version
    assert service.get_dev_use(grant['grant_id']) == grant
    service.disable('m1', actor='controlled_test_operator', reason='test lifecycle')
    with pytest.raises(StateConflictError): service.get_dev_use(grant['grant_id'])
    with pytest.raises(StateConflictError): service.authorize_dev_use(**args)


def test_importer_http_roles_and_private_projection(source_setup):
    f, service, s, m = candidate(source_setup)
    app = create_app(service.store.path, api_key='controlled-online-key', importer_key='controlled-importer-key', operator_key='controlled-operator-key')
    r = route(app, '/dev-v2/sources')
    auth = r.dependant.dependencies[0].call
    for online, importer, status in [(None, None, 401), ('controlled-online-key', None, 403), ('controlled-online-key', 'controlled-operator-key', 403)]:
        with pytest.raises(HTTPException) as exc: auth(online, importer)
        assert exc.value.status_code == status
    auth('controlled-online-key', 'controlled-importer-key')
    result = r.endpoint(ImportInput(import_id='source', catalog_id='failed', descriptor='failed/source.json'))
    assert result['kind'] == 'source'
    private = route(app, '/memories/{memory_id}/impact', 'GET').dependant.dependencies[1].call
    req = Request({'type': 'http', 'path_params': {'memory_id': 'm1'}})
    with pytest.raises(HTTPException) as exc: private(req, 'controlled-online-key', None)
    assert exc.value.status_code == 403
    private(req, 'controlled-online-key', 'controlled-importer-key')
    online = route(app, '/memories/{memory_id}', 'GET').endpoint('m1')
    assert not {'confidence', 'validation_successes', 'validation_failures', 'status_reason'} & online.keys()
    assert 'native_success' not in json.dumps(online)
    with pytest.raises(ValueError, match='distinct'):
        create_app(service.store.path, api_key='controlled-online-key', importer_key='controlled-online-key')
    with pytest.raises(StateConflictError, match='importer'):
        service.create_candidate(memory_id='spoof', request_id='not-real', guidance='hint', repair='repair', applicability={k: s['context'][k] for k in SCOPE}, created_at=1.0)


def test_dev_bundle_result_generation_binding_and_idempotence(tmp_path):
    from attention_memory_service.identity import store_id
    f, service, trusted = make_promoted(tmp_path)
    result = f.make('result', success=False)
    previous = 'controlled_test previous code\n'
    grant = service.authorize_dev_use(memory_id='m1', memory_version=trusted.version,
        development_id='result:v1', context=result['context'],
        source_policy_sha256=digest(previous.encode()), now=time.time())
    f.add_result_model(result, grant, store_id(service.store.path))
    catalogs = f.seal()
    service.dev_v2.register('result', catalogs['result'])
    service.import_dev_v2_execution(import_id='result', catalog_id='result', descriptor='result/execution.json')
    recorded = service.record_dev_v2_result(grant_id=grant['grant_id'], execution_import_id='result')
    assert recorded['result']['result_policy_sha256'] == result['record']['source']['sha256']
    assert service.record_dev_v2_result(grant_id=grant['grant_id'], execution_import_id='result') == recorded
    public = service.get_dev_use(grant['grant_id'])
    assert not {'bundle', 'ledger', 'native_success', 'native_run_status', 'public_feedback', 'permission_scope'} & public['result'].keys()
    assert service.authorize_dev_use(memory_id='m1', memory_version=trusted.version, development_id='result:v1', context=result['context'], source_policy_sha256=digest(previous.encode()), now=time.time()) == public
    with pytest.raises(StateConflictError, match='another run'):
        service.record_dev_v2_result(grant_id=grant['grant_id'], execution_import_id='failed')
    with pytest.raises(StateConflictError, match='sealed bundle'):
        service.record_dev_result(grant_id=grant['grant_id'], run_id='old', result_policy_sha256='a' * 64, artifact_uri='old', artifact_sha256='a' * 64)
    invalid = f.make('bool-result')
    wrong_grant = service.authorize_dev_use(memory_id='m1', memory_version=trusted.version, development_id='bool-result:v1', context=invalid['context'], source_policy_sha256=digest(previous.encode()), now=time.time())
    f.add_result_model(invalid, {**wrong_grant, 'memory_version': True}, store_id(service.store.path))
    service.dev_v2.register('bool-result', f.seal()['bool-result'])
    service.import_dev_v2_execution(import_id='bool-result', catalog_id='bool-result', descriptor='bool-result/execution.json')
    with pytest.raises(StateConflictError, match='exact integer'):
        service.record_dev_v2_result(grant_id=wrong_grant['grant_id'], execution_import_id='bool-result')
    model_input = f.root / 'result/model-input.json'
    original = model_input.read_bytes()
    model_input.write_bytes(b'{}')
    with pytest.raises(StateConflictError, match='digest changed'):
        service.get_dev_use(grant['grant_id'])
    model_input.write_bytes(original)
    Path(result['descriptor']['ledger']['uri']).write_bytes(b'tampered')
    with pytest.raises(StateConflictError, match='digest changed'):
        service.get_dev_use(grant['grant_id'])


@pytest.mark.parametrize('action', ['rollback', 'expiry'])
def test_new_grant_exact_version_and_lifecycle(tmp_path, action):
    f, service, trusted = make_promoted(tmp_path)
    args = dict(memory_id='m1', development_id='result:v1', context=f.records['failed']['context'], source_policy_sha256='a' * 64, now=time.time())
    for version in (None, True, trusted.version + 1):
        with pytest.raises(StateConflictError, match='exact trusted'):
            service.authorize_dev_use(**args, memory_version=version)
    grant = service.authorize_dev_use(**args, memory_version=trusted.version)
    if action == 'expiry':
        service.set_expiry('m1', expires_at=time.time() - 1, actor='controlled_test', reason='controlled_test expiry')
    else:
        service.rollback('m1', actor='controlled_test', reason='controlled_test rollback')
    with pytest.raises(StateConflictError): service.get_dev_use(grant['grant_id'])
    assert service.retrieve(args['context'], now=time.time()) == []


def test_scene_reproduction_failure_preserves_candidate_without_promotion(tmp_path):
    f = Fixture(tmp_path); failed = f.make('failed'); f.add_advice(failed)
    failed['verdict']['checks']['scene_reproduction'] = False
    f.write(f.validator / 'failed/verdict.json', failed['verdict'])
    service = MemoryService(tmp_path / 'authority.sqlite', dev_v2_catalogs=f.seal())
    service.import_dev_v2_execution(import_id='failed', catalog_id='failed', descriptor='failed/execution.json')
    service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')
    assert service.create_dev_v2_candidate(memory_id='m1', source_import_id='source', created_at=1.0).status.value == 'candidate'
    with pytest.raises(StateConflictError, match='scene reproduction'):
        service.import_dev_v2_trial(import_id='trial-failed', catalog_id='validator-failed', descriptor='failed/trial.json')
    assert service.retrieve(failed['context'], now=time.time()) == []


@pytest.mark.parametrize('defect', ['dispatch', 'consumption'])
def test_validation_grants_require_prior_plan_and_original_dispatch(source_setup, defect):
    f, service, s, m = candidate(source_setup)
    r = f.make('offline-trial')
    args = dict(memory_id='m1', cell=r['name'], version=1, seed=101, context=r['context'], policy_sha256=r['record']['source']['sha256'], config_sha256=r['descriptor']['config']['sha256'], exposure='none')
    with pytest.raises(StateConflictError, match='plan before'):
        service.authorize_dev_v2_validation(**args)
    # Freeze five concrete cases; only one synthetic trial is needed to check
    # that retrospective exposure and a different policy version are rejected.
    cases = []
    for i, seed in enumerate(range(101, 106)):
        view = f.make('view-' + str(seed), seed=seed, variant=i % 2)
        cases.append({'seed': seed, **{k: view['context'][k] for k in ('scene_id', 'object_set_id', 'camera_config_id', 'task_variant_id', 'camera_names', 'task_prompt')}})
    ctx = r['context']
    service.record_plan('m1', {'schema_version': 'attentionbench.memory-validation-plan.v2', 'memory_id': 'm1', **{k: ctx[k] for k in SCOPE if k not in ('scene_id', 'object_set_id', 'camera_config_id', 'task_variant_id')}, 'policy_id': 'full_trace_aware_attention_planner', 'seeds': list(range(101, 106)), 'cases': cases, 'assistance_credits': 1})
    grant = service.authorize_dev_v2_validation(**args)
    assert service.authorize_dev_v2_validation(**args) == grant
    with pytest.raises(StateConflictError, match='policy version'):
        service.authorize_dev_v2_validation(**{**args, 'cell': 'other', 'policy_sha256': 'b' * 64})
    with pytest.raises(StateConflictError, match='scope'):
        service.authorize_dev_v2_validation(**{**args, 'cell': 'other', 'context': {**ctx, 'namespace': 'other'}})
    f.bind_validation(r, grant, consume=defect != 'consumption')
    with sqlite3.connect(r['descriptor']['ledger']['uri']) as db:
        if defect == 'dispatch': db.execute('DELETE FROM v2_events')  # Controlled fixture corruption only.
    r['descriptor']['ledger']['sha256'] = digest(Path(r['descriptor']['ledger']['uri']).read_bytes())
    f.write(f.root / r['name'] / 'execution.json', r['descriptor'])
    catalogs = f.seal()
    for cid in (r['name'], 'validator-' + r['name']): service.dev_v2.register(cid, catalogs[cid])
    service.import_dev_v2_execution(import_id=r['name'], catalog_id=r['name'], descriptor=r['name'] + '/execution.json')
    with pytest.raises(StateConflictError, match='pre-execution validation dispatch' if defect == 'dispatch' else 'consumption receipt'):
        service.import_dev_v2_trial(import_id='bad-trial', catalog_id='validator-' + r['name'], descriptor=r['name'] + '/trial.json')


def test_evidence_cache_is_per_operation_and_rejects_mid_call_changes(source_setup):
    from attention_memory_service.dev_v2 import evidence_operation
    f, service, s, m = candidate(source_setup)

    @evidence_operation
    def change_during_read():
        service.check_candidate('m1')
        (f.root / 'openarm365_dev_v2/sdk.py').write_text('# altered controlled test\n')
        return service.check_candidate('m1')

    with pytest.raises(StateConflictError, match='changed during authority'):
        change_during_read()
    with pytest.raises(StateConflictError, match='digest changed'):
        service.check_candidate('m1')


def test_scene_verifier_preserves_frozen_tolerance_and_full_state():
    from attention_memory_service.dev_v2 import scene_matches
    original = {'robot': {'eef_quat': [0, 0, 0, 1]}, 'objects': {'cup_pos': [0, 1, 0]}, 'fixtures': {'sink_pos': [1, 0, 0]}}
    same = copy.deepcopy(original); same['robot']['eef_quat'] = [0, 0, 0, -1]
    assert scene_matches(original, same)
    changed = copy.deepcopy(original); changed['objects']['cup_pos'][1] += 1.1824655973261569
    assert not scene_matches(original, changed)
    changed = copy.deepcopy(original); changed['fixtures']['sink_pos'][0] += 1e-4
    assert not scene_matches(original, changed)
    assert not scene_matches(original, {k: v for k, v in original.items() if k != 'fixtures'})


def test_sdk_source_file_set_drift_is_rejected(source_setup):
    f, service, s, m = candidate(source_setup)
    (f.root / 'openarm365_dev_v2/unlocked.py').write_text('# controlled_test drift\n')
    with pytest.raises(StateConflictError, match='source file set drift'):
        service.check_candidate('m1')
