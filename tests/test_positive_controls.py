"""C18 isolated records, shared decision code, sealed fake arms and exclusions."""
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from attention_memory_service.memory_service import MemoryService
from attention_memory_service.core.store import StateConflictError
from attention_memory_service.memory_service_api import create_app
from attention_memory_service.positive_controls import LABEL, validate_spec, PositiveControls
from attention_memory_service.memory_v2 import MemoryV2Manager
from tests.test_arm_generation import prepare, import_trial, MODEL
from tests.test_single_scene import SIGNATURE
from tests.dev_v2_fixtures import Fixture as BaseFixture
from tests.test_sync_legacy_routes import sync_request

API='online-key-123456789'; IMPORTER='importer-key-123456789'


class Fixture(BaseFixture):
    def make(self,*args,**kwargs):
        r=super().make(*args,**kwargs)
        r['context']['seed']=r['config']['seed']
        self.write(self.root/r['name']/'execution.json',r['descriptor'])
        clean=json.loads((self.root/r['name']/'validation-input.json').read_text()) if (self.root/r['name']/'validation-input.json').exists() else None
        if clean is not None:
            clean['context']=r['context']
            r['descriptor']['validation_input']=self.write(self.root/r['name']/'validation-input.json',clean)
            self.write(self.root/r['name']/'execution.json',r['descriptor'])
        return r


def setup(tmp_path, treatments=(False,)*5):
    f=Fixture(tmp_path,interface=True)
    failed=f.make('failed',failure=True)
    # No Advisor or source import. The origin's independently sealed signature.
    failed['verdict'].update(failure_signature=SIGNATURE,signature_present=True)
    f.write(f.validator/'failed/verdict.json',failed['verdict'])
    s=MemoryService(tmp_path/'authority.sqlite',dev_v2_catalogs=f.seal())
    s.import_dev_v2_execution(import_id='failed',catalog_id='failed',descriptor='failed/execution.json')
    ctx=failed['context']
    spec=dict(control_id='m1',title='Controlled test',guidance='Rotate +0.3 rad; lift 45 mm then 15 mm.',
              label=LABEL,origin=dict(experiment_id='failed',cell='failed',version=1),context=ctx,
              cross_seed_targets=[dict(seed=114,contract=str(tmp_path/'contract-114/contract.json')),
                                  dict(seed=104,contract=str(tmp_path/'contract-104/contract.json'))])
    pc=s.positive_controls
    pc.create(spec=spec,spec_sha256=validate_spec(spec),execution_import_id='failed',catalog_id='validator-failed',descriptor='failed/verdict.json')
    from attention_memory_service.dev_v2 import SCOPE,AXES
    plan=dict(schema_version='attentionbench.memory-validation-plan.v2',profile='single_scene_v1',memory_id='m1',
              **{k:ctx[k] for k in SCOPE if k not in AXES},policy_id=failed['cell']['policy_id'],
              seeds=[101],cases=[dict(seed=101,**{k:ctx[k] for k in (*AXES,'camera_names','task_prompt')})],
              pairs=[dict(pair_id=str(i),control_cell=f'c{i}',treatment_cell=f't{i}') for i in range(5)],
              failure_signature=SIGNATURE,assistance_credits=1,arm_generation='dev_from_origin',
              origin_source_sha256=failed['record']['source']['sha256'],dev_model=MODEL,
              validation_config_sha256=failed['descriptor']['config']['sha256'])
    pc.record_plan('m1',plan)
    for i in range(5):
        for arm in ('c','t'):
            f.make(f'{arm}{i}',failure=True if arm=='c' else treatments[i],completed_calls=1,
                exposure='none' if arm=='c' else 'candidate_validation',memory_id=None if arm=='c' else 'm1',
                code=f'def policy(sdk):\n    # {arm}{i}\n    pass\n')
    proxy=SimpleNamespace(store=SimpleNamespace(get_memory=pc.get_memory,path=s.store.path),
        dev_v2=pc.dev_v2,authorize_dev_v2_validation=pc.authorize_dev_v2_validation,
        import_dev_v2_execution=s.import_dev_v2_execution,import_dev_v2_trial=pc.dev_v2.import_trial,
        cross_seed=pc.cross_seed)
    return f,s,pc,proxy,plan,spec


def finish(f,pc,proxy,plan):
    for pair in plan['pairs']:
        for key in ('control_cell','treatment_cell'):
            prepare(f,proxy,plan,pair[key]);import_trial(proxy,pair[key])
        pc.record_dev_v2_pair(memory_id='m1',control_import_id='trial-'+pair['control_cell'],treatment_import_id='trial-'+pair['treatment_cell'])
    return pc.evaluate('m1')


@pytest.mark.parametrize('treatments,decision', [((False,)*5,'would_pass'),((True,)*5,'would_fail')])
def test_decision_and_exclusions(tmp_path,treatments,decision):
    f,s,pc,proxy,plan,spec=setup(tmp_path,treatments)
    verdict=finish(f,pc,proxy,plan)
    assert verdict['decision']==decision
    assert PositiveControls._single_scene_failures is MemoryV2Manager._single_scene_failures
    assert pc.evaluate('m1')==verdict
    assert s.list_memories()==[] and s.list_dev_v2_memories()==[]
    assert s.retrieve(spec['context'],now=time.time())==[]
    assert s.validation_queue()==[] and s.cross_seed.queue()==[]
    for action in (lambda:s.get_memory('m1'),lambda:s.promote('m1'),lambda:s.evaluate('m1'),
                   lambda:s.impact_report('m1'),lambda:s.export_package('m1'),
                   lambda:s.authorize_use(memory_id='m1',context=spec['context'],attempt_id='attempt-test',now=time.time()),
                   lambda:s.authorize_dev_use(memory_id='m1',development_id='dev:v1',context=spec['context'],source_policy_sha256=plan['origin_source_sha256'],now=time.time())):
        with pytest.raises((ValueError,KeyError,StateConflictError,FileNotFoundError)):action()
    with sqlite3.connect(s.store.path) as db:
        for table in ('memories','memory_v2_provenance','memory_v2_decisions','memory_v2_use_grants','memory_dev_exposures'):
            assert db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]==0
        assert not db.execute("SELECT 1 FROM dev_v2_imports WHERE kind='source'").fetchone()
    # Normal trial import cannot consume a positive-control grant.
    with pytest.raises((ValueError,StateConflictError),match='grant'):
        s.import_dev_v2_trial(import_id='ordinary-trial',catalog_id='validator-c0',descriptor='c0/trial.json')


def test_cross_seed_same_logic(tmp_path):
    from tests.test_cross_seed import batch,finish as cross_finish
    f,s,pc,proxy,plan,spec=setup(tmp_path)
    finish(f,pc,proxy,plan)
    for seed,failure in ((114,True),(104,False)):
        cross_plan,_=batch(f,proxy,seed,failure=failure)
        status=cross_finish(f,proxy,cross_plan)
    assert [r['verdict'] for r in status['seeds']]==['neutral_no_harm','transfer_passed']
    assert status['task_trust']['verdict']=='would be trusted for this task'
    assert not status['task_trust']['missing']
    assert s.list_memories()==[]


def test_importer_only_and_binding(tmp_path):
    f,s,pc,proxy,plan,spec=setup(tmp_path)
    app=create_app(s.store.path,api_key=API,importer_key=IMPORTER)
    payload=dict(spec=spec,spec_sha256=validate_spec(spec),execution_import_id='failed',catalog_id='validator-failed',descriptor='failed/verdict.json')
    assert sync_request(app,'/dev-v2/positive-controls/create',method='POST',key=API,payload=payload)[0]==403
    assert sync_request(app,'/dev-v2/positive-controls/create',method='POST',key=API,importer_key=IMPORTER,payload=payload)[0]==200
    with pytest.raises((ValueError,StateConflictError)):pc.create(**dict(payload,spec_sha256='0'*64))
    changed=dict(spec,guidance='different')
    with pytest.raises((ValueError,StateConflictError)):pc.create(**dict(payload,spec=changed,spec_sha256=validate_spec(changed)))
    with pytest.raises((ValueError,StateConflictError)):pc.record_plan('m1',dict(plan,pairs=plan['pairs'][:4]))
    assert sync_request(app,'/dev-v2/positive-controls',key=API)[1]['items'][0]['label']==LABEL


@pytest.mark.parametrize('fault,missing', [('remains','Worked on at least one other transfer seed'),
                                         ('earlier','No neutral seed showed harm')])
def test_cross_seed_missing_conditions(tmp_path,fault,missing):
    from tests.test_cross_seed import batch,finish as cross_finish
    f,s,pc,proxy,plan,spec=setup(tmp_path)
    finish(f,pc,proxy,plan)
    for seed,failure in ((114,True),(104,False)):
        cross_plan,_=batch(f,proxy,seed,failure=failure,fault=fault if (failure and fault=='remains' or not failure and fault=='earlier') else None)
        status=cross_finish(f,proxy,cross_plan)
    assert missing in status['task_trust']['missing']
    assert status['task_trust']['verdict']=='missing checks' and s.list_memories()==[]


def test_real_result_path_rejects_positive_grants(tmp_path):
    f,s,pc,proxy,plan,spec=setup(tmp_path)
    prepare(f,proxy,plan,'t0');import_trial(proxy,'t0')
    with sqlite3.connect(s.store.path) as db:
        grant_id=db.execute('SELECT id FROM positive_control_grants').fetchone()[0]
    with pytest.raises(StateConflictError):
        s.record_dev_v2_result(grant_id=grant_id,execution_import_id='t0')
    with pytest.raises(StateConflictError):
        s.authorize_dev_v2_validation(memory_id='m1',cell='t0',version=1,seed=101,context=spec['context'],
            policy_sha256=f.records['t0']['record']['source']['sha256'],config_sha256=plan['validation_config_sha256'],
            exposure='candidate_validation',origin_source_sha256=plan['origin_source_sha256'])
    with pytest.raises(StateConflictError):s.uses('m1')
