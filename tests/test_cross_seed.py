"""Cross-seed fixtures only: sealed executions, no models or simulators."""
import copy
import json
import sqlite3
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.dev_v2 import digest, encoded
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.memory_service_api import create_app
from attention_memory_service.scene_contract import V2, fingerprint
from tests.dev_v2_fixtures import Fixture
from tests.test_single_scene import setup_scene, import_pairs
from tests.test_arm_generation import prepare, import_trial, MODEL, setup as setup_generated
from tests.test_scene_contract import STATE, receipt
from tests.test_auto_validation import outcome, scene_feedback
from tests.test_provisional_notes import API, IMPORTER
from tests.test_sync_legacy_routes import sync_request


@pytest.fixture
def trusted(tmp_path, request):
    generated = getattr(request, 'param', False)
    f, service, plan = (setup_generated if generated else setup_scene)(tmp_path, interface=True)
    assert isinstance(f, Fixture)
    service.record_plan('m1', plan)
    if generated:
        for pair in plan['pairs']:
            for arm in ('control_cell', 'treatment_cell'):
                prepare(f, service, plan, pair[arm])
                import_trial(service, pair[arm])
            service.record_dev_v2_pair(memory_id='m1', control_import_id='trial-' + pair['control_cell'],
                                       treatment_import_id='trial-' + pair['treatment_cell'])
    else:
        import_pairs(f, service)
    assert service.evaluate('m1').status.value == 'trusted'
    return f, service


def target(f, service, seed, *, failure=True, fault=None, checks=None):
    root = f.root.parent / f'contract-{seed}'
    contract = dict(schema=V2, tolerance=1e-6, tasks={'pick': {'seeds': {str(seed): {
        'normalized_sha256': fingerprint(STATE, True)}}}})
    if fault == 'missing_seed': contract['tasks']['pick']['seeds'] = {}
    ref = f.write(root / 'contract.json', contract)
    scene = dict(contract=ref, state=STATE, validation=receipt(STATE, ref, seed),
                 replays=[dict(state=STATE, validation=receipt(STATE, ref, seed)) for _ in range(2)])
    if fault == 'fingerprint':
        scene['replays'][1] = dict(state={'robot': {'x': [99]}}, validation=receipt({'robot': {'x': [99]}}, ref, seed))
    r = f.make(f'probe{seed}', seed=seed, failure=failure, completed_calls=1, scene=scene,
               code='def policy(sdk):\n    return 0\n' if fault == 'source' else 'def policy(sdk):\n    pass\n')
    if fault == 'other_task': r['context']['task_prompt'] = 'another task'
    if fault == 'interface': r['context']['sdk_interface_sha256'] = '0' * 64
    if fault == 'notes':
        value = json.loads(Path(r['descriptor']['validation_input']['uri']).read_text())
        value['memory_ids'] = ['other']
        r['descriptor']['validation_input'] = f.write(Path(r['descriptor']['validation_input']['uri']), value)
        f.write(f.root / r['name'] / 'execution.json', r['descriptor'])
    r['verdict']['checks'].update(checks or {})
    f.write(f.validator / r['name'] / 'verdict.json', r['verdict'])
    catalogs = f.seal()
    if fault == 'root_confusion':
        catalogs['validator-' + r['name']] = {**catalogs[r['name']], 'role': 'validator'}
    for cid in (r['name'], 'validator-' + r['name']): service.dev_v2.register(cid, catalogs[cid])
    if fault != 'no_catalog':
        service.dev_v2.register(f'contract-{seed}', dict(root=str(root), role='contract', context=r['context'],
                                                       files={'contract.json': ref['sha256']}))
    service.import_dev_v2_execution(import_id=r['name'], catalog_id=r['name'], descriptor=r['name'] + '/execution.json')
    args = dict(memory_id='m1', target_seed=seed, execution_import_id=r['name'], catalog_id='validator-' + r['name'],
                descriptor=r['name'] + '/verdict.json', scene_contract_ref=ref)
    if fault == 'digest': (f.validator / r['name'] / 'verdict.json').write_text('{}')
    if fault == 'wrong_ref': args['scene_contract_ref'] = {**ref, 'sha256': '0' * 64}
    if fault == 'executor': args['catalog_id'] = r['name']
    return args, scene


def batch(f, service, seed, *, failure=True, fault=None, checks=None):
    args, scene = target(f, service, seed, failure=failure, checks=checks)
    probe = service.cross_seed.probe(**args)
    if probe['role'] == 'could_not_finish': return None, args
    n = 5 if failure else 3
    pairs = [dict(pair_id=str(i), control_cell=f'c{seed}-{i}', treatment_cell=f't{seed}-{i}') for i in range(n)]
    for pair in pairs:
        for arm in ('control_cell', 'treatment_cell'):
            name = pair[arm]
            treatment = arm == 'treatment_cell'
            f.make(name, seed=seed, failure=failure and (not treatment or fault == 'remains'),
                   completed_calls=0 if treatment and fault == 'earlier' else 1,
                   scene=scene, exposure='candidate_validation' if treatment else 'none',
                   memory_id='m1' if treatment else None, code=f'def policy(sdk):\n    # {name}\n    pass\n')
    plan = service.cross_seed.plan(memory_id='m1', target_seed=seed, pairs=pairs, dev_model=MODEL)
    return plan, args


def finish(f, service, plan, *, fault=None, count=None):
    for pair in plan['pairs'][:count]:
        for arm in ('control_cell', 'treatment_cell'):
            prepare(f, service, plan, pair[arm], unsafe=fault == 'safety' and arm == 'treatment_cell')
            import_trial(service, pair[arm])
        service.cross_seed.pair(memory_id='m1', target_seed=plan['target_seed'],
            control_import_id='trial-' + pair['control_cell'], treatment_import_id='trial-' + pair['treatment_cell'])
    return service.cross_seed.status(memory_id='m1')


@pytest.mark.parametrize('failure,checks,role', [(True, {}, 'transfer'), (False, {}, 'neutral'),
    (True, {'safety': False}, 'could_not_finish'), (False, {'scene_reproduction': False}, 'could_not_finish')])
def test_probe_computed_and_idempotent(trusted, failure, checks, role):
    f, s = trusted
    args, _ = target(f, s, 114, failure=failure, checks=checks)
    value = s.cross_seed.probe(**args)
    assert value['role'] == role and value['signature_present'] is failure
    assert s.cross_seed.probe(**args) == value
    assert MemoryService(s.store.path).cross_seed.probe(**args) == value
    with pytest.raises(TypeError): s.cross_seed.probe(**args, role='neutral')
    if role == 'could_not_finish':
        assert s.cross_seed.status(memory_id='m1')['task_trust']['scenes'] == 1
        with pytest.raises(StateConflictError): s.cross_seed.plan(memory_id='m1', target_seed=114, pairs=[], dev_model=MODEL)
    (f.validator / 'probe114/verdict.json').write_text('{}')
    with pytest.raises(StateConflictError): s.cross_seed.probe(**args)


@pytest.mark.parametrize('fault', ['origin', 'other_task', 'no_catalog', 'wrong_ref', 'fingerprint', 'digest', 'executor', 'missing_seed', 'notes', 'source', 'interface', 'root_confusion'])
def test_probe_refusals(trusted, fault):
    f, s = trusted
    with pytest.raises((StateConflictError, ValueError)):
        args, _ = target(f, s, 101 if fault == 'origin' else 114, fault=fault)
        s.cross_seed.probe(**args)
    assert s.cross_seed.status(memory_id='m1')['seeds'] == []


@pytest.mark.parametrize('transfer,fault,expected', [(True, None, 'transfer_passed'),
    (True, 'safety', 'transfer_failed'), (True, 'earlier', 'transfer_failed'), (True, 'remains', 'transfer_failed'),
    (False, None, 'neutral_no_harm'), (False, 'safety', 'neutral_harm'), (False, 'earlier', 'neutral_harm')])
def test_pair_rules(trusted, transfer, fault, expected):
    f, s = trusted
    plan, _ = batch(f, s, 114, failure=transfer, fault=fault)
    assert len(plan['pairs']) == (5 if transfer else 3)
    with pytest.raises(StateConflictError):
        s.cross_seed.plan(memory_id='m1', target_seed=114, pairs=plan['pairs'][:-1], dev_model=MODEL)
    status = finish(f, s, plan, fault=fault, count=len(plan['pairs']) - 1)
    assert status['seeds'][0]['verdict'] is None and s.decision('m1')['trust_scope'] == 'scene_scoped'
    # Finish only the final pair; sealed records are immutable.
    status = finish(f, s, {**plan, 'pairs': plan['pairs'][-1:]}, fault=fault)
    assert status['seeds'][0]['verdict'] == expected
    assert status['task_trust']['missing']  # Only two scenes so far.


@pytest.mark.parametrize('trusted', [False, True], indirect=True)
def test_task_promotion_retrieval_and_ladder(trusted):
    f, s = trusted
    origin = {**f.records['failed']['context'], 'seed': 101}
    other = {**origin, 'seed': 114}
    before_origin = s.authorize_dev_use(memory_id='m1', memory_version=s.store.get_memory('m1').version,
        development_id='before-origin:v1', context=origin, source_policy_sha256='c' * 64, now=time.time())
    def grant(ctx, name):
        return s.authorize_dev_use(memory_id='m1', memory_version=s.store.get_memory('m1').version,
            development_id=name + ':v1', context=ctx, source_policy_sha256='c' * 64, now=time.time())
    assert grant(other, 'reference')['trusted_here'] is False
    origin_decision = copy.deepcopy(s.decision('m1'))
    plan, _ = batch(f, s, 114)
    finish(f, s, plan)
    assert s.decision('m1')['trust_scope'] == 'scene_scoped'
    plan2, _ = batch(f, s, 120, failure=False)
    status = finish(f, s, plan2)
    assert status['task_trust']['missing'] == []
    decision = s.decision('m1')
    assert decision['trust_scope'] == 'task_scoped' and decision['cross_scene_eligible'] is False
    assert decision['report'] == origin_decision['report']
    assert grant(other, 'task')['trusted_here'] is True
    assert s.get_dev_v2_memory_public('m1')['trust_display'] == 'task_scoped_trusted'
    assert grant(other, 'task2')['match_level'] == 'same_task'
    after_origin = grant(origin, 'after-origin')
    for k in ('trusted_here', 'match_level', 'layer', 'excluded_from_results'):
        assert after_origin.get(k) == before_origin.get(k)
    assert s.retrieve_levels({**other, 'task_id': 'other'}, now=time.time()) == []
    pair = plan2['pairs'][-1]
    s.cross_seed.pair(memory_id='m1', target_seed=120, control_import_id='trial-' + pair['control_cell'],
                       treatment_import_id='trial-' + pair['treatment_cell'])
    assert len([e for e in s.memory_events() if e['event'] == 'promoted']) == 2
    assert s.cross_seed.queue() == []
    for i in range(5): _, retry = outcome(s, 'm1', i, scene_feedback(f, i < 3), other)
    assert s.store.get_memory('m1').status.value == 'trusted'
    assert s.decision('m1')['trust_scope'] == 'scene_scoped'
    assert s.memory_events()[-1]['reason'] == 'target failure in 3 of last 5 uses on other seeds'
    assert grant(other, 'reference-again')['trusted_here'] is False
    s.record_provisional_outcome(**retry)
    assert len([e for e in s.memory_events() if e['event'] == 'demoted']) == 1
    for i in range(5, 10): _, retry = outcome(s, 'm1', i, scene_feedback(f, i < 8), origin)
    assert s.store.get_memory('m1').status.value == 'provisional'
    s.record_provisional_outcome(**retry)
    assert len([e for e in s.memory_events() if e['event'] == 'demoted']) == 2


@pytest.mark.parametrize('transfer,arm,checks,expected', [(True, 'control_cell', {'safety': False}, 'could_not_finish'),
    (True, 'treatment_cell', {'safety': False}, 'transfer_failed'),
    (False, 'treatment_cell', {'safety': False}, 'neutral_harm'),
    (False, 'treatment_cell', {'scene_reproduction': False}, 'could_not_finish')])
def test_failed_checks(trusted, transfer, arm, checks, expected):
    f, s = trusted
    plan, _ = batch(f, s, 114, failure=transfer)
    name = plan['pairs'][0][arm]
    # prepare imports the execution before attempting the refused trial.
    prepare(f, s, plan, name, checks=checks)
    with pytest.raises(StateConflictError): import_trial(s, name)
    trial = f.records[name]['trial']
    args = dict(memory_id='m1', target_seed=114, validation_grant_id=trial['validation_grant_id'],
                execution_import_id=name, catalog_id='validator-' + name, descriptor=name + '/verdict.json')
    value = s.cross_seed.failed_check(**args)
    assert value['verdict'] == expected
    assert s.cross_seed.failed_check(**args) == value
    assert s.store.get_memory('m1').status.value == 'trusted'


@pytest.mark.parametrize('path', ['probes', 'plans', 'pairs', 'failed-checks'])
def test_importer_auth_and_client_role_rejected(tmp_path, path):
    app = create_app(tmp_path / 'auth.sqlite', api_key=API, importer_key=IMPORTER)
    route = '/dev-v2/cross-seed/' + path
    payload = dict(memory_id='m1', target_seed=114)
    payload.update({'probes': dict(execution_import_id='p', catalog_id='v', descriptor='v.json', scene_contract_ref={}),
                    'plans': dict(pairs=[], dev_model=MODEL),
                    'pairs': dict(control_import_id='c', treatment_import_id='t'),
                    'failed-checks': dict(validation_grant_id='g', execution_import_id='p', catalog_id='v', descriptor='v.json')}[path])
    assert sync_request(app, route, method='POST', payload=payload, key=API)[0] == 403
    assert sync_request(app, route, method='POST', payload=payload)[0] == 401
    assert sync_request(app, route, method='POST', payload=payload, importer_key=IMPORTER)[0] == 401
    with pytest.raises(ValidationError):
        sync_request(app, route, method='POST', payload={**payload, 'role': 'transfer'}, key=API, importer_key=IMPORTER)
    assert sync_request(app, '/dev-v2/cross-seed/queue', key=API)[0] == 200
    assert sync_request(app, '/dev-v2/cross-seed/queue')[0] == 401


@pytest.mark.parametrize('missing', ['origin', 'transfer', 'neutral', 'total'])
def test_each_task_trust_condition(trusted, missing):
    f, s = trusted
    first_transfer = missing != 'transfer'
    first, _ = batch(f, s, 114, failure=first_transfer)
    finish(f, s, first)
    if missing != 'total':
        second, _ = batch(f, s, 120, failure=False, fault='earlier' if missing == 'neutral' else None)
        if missing == 'origin':
            s.disable('m1', actor='operator', reason='stop before promotion')
            with pytest.raises(StateConflictError): finish(f, s, second)
        else:
            finish(f, s, second)
    status = s.cross_seed.status(memory_id='m1')
    expected = dict(origin='Trusted on the origin scene at the current version',
                    transfer='Worked on at least one other transfer seed', neutral='No neutral seed showed harm',
                    total='At least three scenes have verdicts, including the origin')
    assert expected[missing] in status['task_trust']['missing']
    assert s.decision('m1')['trust_scope'] == 'scene_scoped'
    assert len([e for e in s.memory_events() if e['event'] == 'promoted']) == 1


def test_transfer_failed_counts_and_does_not_block(trusted):
    f, s = trusted
    failed, _ = batch(f, s, 114, fault='remains')
    finish(f, s, failed)
    passed, _ = batch(f, s, 120)
    status = finish(f, s, passed)
    assert status['task_trust']['missing'] == []
    assert status['task_trust']['transfer_failed'] == [114]
    assert s.decision('m1')['cross_seed_validation']['transfer_failed'] == [114]


@pytest.mark.parametrize('fault', ['control_note', 'second_note', 'wrong_parent', 'old_version', 'tamper', 'extra_section'])
def test_cross_seed_generation_checks(trusted, fault):
    f, s = trusted
    plan, _ = batch(f, s, 114)
    name = plan['pairs'][0]['control_cell' if fault in {'control_note', 'extra_section'} else 'treatment_cell']
    prepare(f, s, plan, name, fault=fault)
    with pytest.raises(StateConflictError): import_trial(s, name)
    assert s.cross_seed.status(memory_id='m1')['seeds'][0]['pairs_done'] == 0


def test_old_evidence_tamper_blocks_promotion(trusted):
    f, s = trusted
    first, _ = batch(f, s, 114)
    finish(f, s, first)
    second, _ = batch(f, s, 120, failure=False)
    finish(f, s, second, count=2)
    name = first['pairs'][0]['control_cell']
    (f.validator / name / 'verdict.json').write_text('{}')
    with pytest.raises(StateConflictError): finish(f, s, {**second, 'pairs': second['pairs'][-1:]})
    assert s.decision('m1')['trust_scope'] == 'scene_scoped'
    status = s.cross_seed.status(memory_id='m1')
    assert status['seeds'][1]['verdict'] is None and status['seeds'][1]['pairs_done'] == 2


def test_task_origin_failure_demotes_only_one_level(trusted):
    f, s = trusted
    first, _ = batch(f, s, 114)
    finish(f, s, first)
    second, _ = batch(f, s, 120, failure=False)
    finish(f, s, second)
    ctx = {**f.records['failed']['context'], 'seed': 101}
    for i in range(5): _, args = outcome(s, 'm1', i, scene_feedback(f, True), ctx)
    s.record_provisional_outcome(**args)
    assert s.store.get_memory('m1').status.value == 'trusted'
    assert s.decision('m1')['trust_scope'] == 'scene_scoped'
    assert s.memory_events()[-1]['reason'] == 'target failure in 3 of last 5 uses'
    for i in range(5, 10): outcome(s, 'm1', i, scene_feedback(f, True), ctx)
    assert s.store.get_memory('m1').status.value == 'provisional'


def test_queue_open_need_and_read_auth(trusted):
    f, s = trusted
    assert [r['memory_id'] for r in s.cross_seed.queue()] == ['m1']
    ctx = {**f.records['failed']['context'], 'seed': 101}
    outcome(s, 'm1', 0, scene_feedback(f, False), ctx)
    row = s.cross_seed.queue()[0]
    assert row['uses'] == row['failure_gone'] == 1
    plan, args = batch(f, s, 114, failure=False)
    assert s.cross_seed.queue()[0]['seeds'][0]['role'] == 'neutral'
    finish(f, s, plan)
    assert s.cross_seed.queue()[0]['seeds'][0]['verdict'] == 'neutral_no_harm'
    with pytest.raises(StateConflictError):
        s.cross_seed.probe(**{**args, 'descriptor': 'another.json'})
    s.disable('m1', actor='operator', reason='off')
    assert s.cross_seed.queue() == []
