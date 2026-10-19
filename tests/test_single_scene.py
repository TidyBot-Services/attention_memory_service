"""Synthetic, temporary-store checks for the scene-scoped admission rule."""
import json
import time
from pathlib import Path

import pytest

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.dev_v2 import SCOPE, public_trace_projection, stop_position
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.memory_service_api import create_app
from tests.dev_v2_fixtures import Fixture
from tests.test_sync_legacy_routes import sync_request


SIGNATURE = {'public_error_class': 'RuntimeError: TimeoutError', 'operation': 'left_arm.move_delta'}


def setup_scene(tmp_path, *, controls=(True,) * 5, treatments=(False,) * 5,
                control_calls=(1,) * 5, treatment_calls=(1,) * 5):
    f = Fixture(tmp_path)
    failed = f.make('failed', failure=True)
    f.add_advice(failed, SIGNATURE)
    for i in range(len(controls)):
        f.make(f'c{i}', failure=controls[i], completed_calls=control_calls[i])
        f.make(f't{i}', failure=treatments[i], completed_calls=treatment_calls[i],
               exposure='candidate_validation', memory_id='m1')
    service = MemoryService(tmp_path / 'authority.sqlite',
                            dev_v2_catalogs={k: v for k, v in f.seal().items() if k in {'failed', 'validator-failed'}})
    service.import_dev_v2_execution(import_id='failed', catalog_id='failed', descriptor='failed/execution.json')
    service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')
    service.create_dev_v2_candidate(memory_id='m1', source_import_id='source', created_at=1.0)
    ctx = failed['context']
    case = {'seed': 101, **{k: ctx[k] for k in ('scene_id', 'object_set_id', 'camera_config_id',
                                                'task_variant_id', 'camera_names', 'task_prompt')}}
    plan = {'schema_version': 'attentionbench.memory-validation-plan.v2', 'profile': 'single_scene_v1',
            'memory_id': 'm1', **{k: ctx[k] for k in SCOPE if k not in
                                    ('scene_id', 'object_set_id', 'camera_config_id', 'task_variant_id')},
            'policy_id': 'full_trace_aware_attention_planner', 'seeds': [101], 'cases': [case],
            'pairs': [{'pair_id': str(i), 'control_cell': f'c{i}', 'treatment_cell': f't{i}'}
                      for i in range(len(controls))],
            'failure_signature': SIGNATURE, 'assistance_credits': 1,
            'validation_policy_sha256': f.records['c0']['record']['source']['sha256'],
            'validation_config_sha256': f.records['c0']['descriptor']['config']['sha256']}
    return f, service, plan


def import_pairs(f, service, count=5, *, unsafe_treatment=None):
    for i in range(count):
        for arm in ('c', 't'):
            name = f'{arm}{i}'
            r = f.records[name]
            grant = service.authorize_dev_v2_validation(
                memory_id='m1', cell=name, version=1, seed=101, context=r['context'],
                policy_sha256=r['record']['source']['sha256'],
                config_sha256=r['descriptor']['config']['sha256'], exposure=r['trial']['exposure'])
            f.bind_validation(r, grant)
            trace = json.loads(Path(r['bundle']['artifacts']['trace']['uri']).read_text())
            r['verdict'].update(failure_signature=SIGNATURE,
                                signature_present=public_trace_projection(trace)['public_error'].startswith(
                                    SIGNATURE['public_error_class']),
                                stop_position=stop_position(trace))
            if unsafe_treatment == i and arm == 't':
                monitor = Path(r['verdict']['safety']['uri'])
                value = json.loads(monitor.read_text())
                value['unsafe_attempts'] = 1
                value['violations'] = [{'kind': 'controlled_test_sample'}]
                r['verdict']['safety'] = f.write(monitor, value)
            f.write(f.validator / name / 'verdict.json', r['verdict'])
            catalogs = f.seal()
            for cid in (name, 'validator-' + name):
                service.dev_v2.register(cid, catalogs[cid])
            service.import_dev_v2_execution(import_id=name, catalog_id=name, descriptor=name + '/execution.json')
            service.import_dev_v2_trial(import_id='trial-' + name, catalog_id='validator-' + name,
                                        descriptor=name + '/trial.json')
        service.record_dev_v2_pair(memory_id='m1', control_import_id=f'trial-c{i}',
                                   treatment_import_id=f'trial-t{i}')


@pytest.mark.parametrize('controls,treatments,ccalls,tcalls,unsafe,expected', [
    ((True,) * 5, (False,) * 5, (1,) * 5, (1,) * 5, None, None),
    ((True,) * 3 + (False,) * 2, (False,) * 5, (1,) * 5, (1,) * 5, None, 'control_signature_not_reproduced'),
    ((True,) * 5, (True,) * 2 + (False,) * 3, (1,) * 5, (1,) * 5, None, 'treatment_signature_remains'),
    ((True,) * 5, (False,) * 5, (1,) * 5, (1,) * 5, 0, 'safety_regression'),
    ((True,) * 5, (False,) * 5, (2,) * 5, (1,) * 5, None, 'treatment_stopped_earlier'),
])
def test_single_scene_verdict(tmp_path, controls, treatments, ccalls, tcalls, unsafe, expected):
    f, service, plan = setup_scene(tmp_path, controls=controls, treatments=treatments,
                                    control_calls=ccalls, treatment_calls=tcalls)
    service.record_plan('m1', plan)
    import_pairs(f, service, unsafe_treatment=unsafe)
    memory = service.evaluate('m1')
    decision = service.decision('m1')
    assert decision['profile'] == 'single_scene_v1'
    assert decision['cross_scene_eligible'] is False
    assert decision['report']['cross_scene_eligible'] is False
    assert decision['report']['control_signature_count'] == sum(controls)
    assert decision['report']['treatment_signature_count'] == sum(treatments)
    assert len(decision['report']['signature_arms']) == 5
    assert memory.status.value == ('rejected' if expected else 'trusted')
    assert (expected in decision['failed_gates']) if expected else decision['failed_gates'] == []
    if expected:
        return
    assert decision['trust_scope'] == 'scene_scoped'
    context = {**f.records['failed']['context'], 'seed': 101}
    assert [m.memory_id for m in service.retrieve(context, now=time.time())] == ['m1']
    for altered in ({**context, 'seed': 102}, {**context, 'camera_names': ['different']},
                    {**context, 'task_prompt': 'different'}, {k: v for k, v in context.items() if k != 'seed'}):
        assert service.retrieve(altered, now=time.time()) == []
    public = service.get_dev_v2_memory_public('m1')
    assert public['trust_scope'] == 'scene_scoped' and public['trust_display'] == 'scene_scoped_trusted'
    assert not {'guidance', 'repair', 'grant', 'ledger', 'bundle', 'raw_evidence_refs'} & public.keys()
    assert not {'guidance', 'repair', 'grant_id', 'source_import_id', 'descriptor', 'evidence_refs'} & {
        key for obj in (public, public['applicability_summary'], public['validation_summary']) for key in obj}
    app = create_app(service.store.path, api_key='controlled-online-key', operator_key='controlled-operator-key')
    assert sync_request(app, '/dev-v2/memories')[0] == 401
    assert sync_request(app, '/dev-v2/memories', key='controlled-online-key')[1]['memories'][0] == public
    assert sync_request(app, '/dev-v2/memories/m1', key='controlled-online-key')[1] == public
    service.disable('m1', actor='test', reason='test')
    assert service.retrieve(context, now=time.time()) == []
    service.enable('m1', actor='test', reason='test')
    service.set_expiry('m1', expires_at=time.time() - 1, actor='test', reason='test')
    assert service.retrieve(context, now=time.time()) == []


def test_single_scene_plan_rejects_short_and_wrong_signature(tmp_path):
    _, service, plan = setup_scene(tmp_path)
    with pytest.raises(ValueError, match='five distinct pairs'):
        service.record_plan('m1', {**plan, 'pairs': plan['pairs'][:4]})
    with pytest.raises(ValueError, match='five distinct pairs'):
        service.record_plan('m1', {**plan, 'failure_signature':
                                       {**SIGNATURE, 'operation': 'right_arm.move_delta'}})


def test_source_signature_cannot_be_invented(tmp_path):
    f = Fixture(tmp_path)
    failed = f.make('failed', failure=True)
    f.add_advice(failed, {**SIGNATURE, 'operation': 'right_arm.move_delta'})
    service = MemoryService(tmp_path / 'authority.sqlite', dev_v2_catalogs=f.seal())
    service.import_dev_v2_execution(import_id='failed', catalog_id='failed', descriptor='failed/execution.json')
    with pytest.raises(StateConflictError, match='signature differs'):
        service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')


def test_scene_scoped_rollback_blocks_retrieval(tmp_path):
    f, service, plan = setup_scene(tmp_path)
    service.record_plan('m1', plan)
    import_pairs(f, service)
    assert service.evaluate('m1').status.value == 'trusted'
    context = {**f.records['failed']['context'], 'seed': 101}
    service.rollback('m1', actor='test', reason='withdrawn')
    assert service.retrieve(context, now=time.time()) == []
    assert service.get_dev_v2_memory_public('m1')['trust_scope'] == 'scene_scoped'


def test_validator_signature_result_must_match_public_trace(tmp_path):
    f, service, plan = setup_scene(tmp_path)
    service.record_plan('m1', plan)
    r = f.records['c0']
    grant = service.authorize_dev_v2_validation(
        memory_id='m1', cell='c0', version=1, seed=101, context=r['context'],
        policy_sha256=r['record']['source']['sha256'],
        config_sha256=r['descriptor']['config']['sha256'], exposure='none')
    f.bind_validation(r, grant)
    trace = json.loads(Path(r['bundle']['artifacts']['trace']['uri']).read_text())
    r['verdict'].update(failure_signature=SIGNATURE, signature_present=False,
                        stop_position=stop_position(trace))
    f.write(f.validator / 'c0/verdict.json', r['verdict'])
    catalogs = f.seal()
    for cid in ('c0', 'validator-c0'):
        service.dev_v2.register(cid, catalogs[cid])
    service.import_dev_v2_execution(import_id='c0', catalog_id='c0', descriptor='c0/execution.json')
    with pytest.raises(StateConflictError, match='signature/stop-position verdict'):
        service.import_dev_v2_trial(import_id='trial-c0', catalog_id='validator-c0',
                                    descriptor='c0/trial.json')
