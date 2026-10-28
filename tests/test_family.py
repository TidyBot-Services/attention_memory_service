"""Sealed synthetic executions only. No provider, robot or live service."""
import copy
import json
import sqlite3
from types import SimpleNamespace
from pathlib import Path
import pytest
from attention_memory_service.core.store import StateConflictError
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.family import LABEL,STATEMENT,REWRITE_PROMPT,validate_spec,family_conditions,FamilyCrossSeed
from attention_memory_service.dev_v2 import digest,encoded
from attention_memory_service.cross_seed import CrossSeed
from tests.test_positive_controls import Fixture as BaseFixture
from tests.test_single_scene import SIGNATURE
from tests.test_arm_generation import prepare,import_trial,MODEL
from tests.test_cross_seed import target

SOURCE='pick_place_counter_to_sink';TARGET='pick_place_counter_to_cabinet'

class Fixture(BaseFixture):
    task=SOURCE
    mid='m1'
    treatment_label=None
    def write(self,path,value):
        def convert(v):
            if isinstance(v,dict):
                for k in list(v):
                    item=convert(v.pop(k));v[self.task+k[4:] if k.startswith('pick:') else self.task if k=='pick' else k]=item
            elif isinstance(v,list):
                for i in range(len(v)):v[i]=convert(v[i])
            elif v=='pick':return self.task
            elif v=='m1':return self.mid
            elif v=='pick cup variant 0':return 'Pick the object from the counter and place it in the '+('sink.' if self.task==SOURCE else 'cabinet.')
            return v
        if self.treatment_label and isinstance(value,dict) and len(value.get('messages',[]))==2:
            packet=json.loads(value['messages'][1]['content'])
            if packet.get('attention_control',{}).get('memory') and not value['messages'][0]['content'].endswith(self.treatment_label):value['messages'][0]['content']+='\n'+self.treatment_label
        return super().write(path,convert(value))


def rewrite(source,text='Adjust the grasp heading and lift in small increments.'):
    prompt=dict(instruction=REWRITE_PROMPT,title=source['title'],guidance=source['guidance'],task_id=source['context']['task_id'])
    raw=encoded(dict(title='Grasp heading',guidance=text))
    return dict(title='Grasp heading',guidance=text,receipt=dict(provider='codex',model='controlled_test_dev',prompt=prompt,prompt_sha256=digest(encoded(prompt).encode()),response_sha256=digest(raw.encode()),raw_response=raw,started_at=1.,finished_at=2.))


def setup(tmp_path):
    f=Fixture(tmp_path,interface=True);failed=f.make('failed',failure=True)
    failed['verdict'].update(failure_signature=SIGNATURE,signature_present=True)
    f.write(f.validator/'failed/verdict.json',failed['verdict'])
    s=MemoryService(tmp_path/'authority.sqlite',dev_v2_catalogs=f.seal())
    s.import_dev_v2_execution(import_id='failed',catalog_id='failed',descriptor='failed/execution.json')
    spec=dict(fixture_id='fixture-one',title='Origin heading',guidance='Rotate the grasp then lift.',origin=dict(experiment_id='failed',cell='failed',version=1),context=failed['context'],label=LABEL,user_quotes=['Operator test confirmation'])
    args=dict(spec=spec,spec_sha256=validate_spec(spec),execution_import_id='failed',catalog_id='validator-failed',descriptor='failed/verdict.json')
    s.family.fixture_create(**args)
    return f,s,spec,args


def candidate(f,s,kind='fixture'):
    source=dict(kind=kind,id='fixture-one' if kind=='fixture' else 'm1')
    c=s.family.generalize(source=source,rewrite=rewrite(s.family.source(source)))
    f.mid=c['candidate_id'];f.treatment_label=c['treatment_label'];mid=f.mid
    s.family.queue_check(spec=dict(family_check_id='check-one',candidate_id=mid,target_task_id=TARGET,label=LABEL if kind=='fixture' else 'Family check',
        targets=[dict(seed=seed,contract=str(f.root.parent/f'contract-{seed}'/'contract.json'),baseline_source=dict(path=str(f.root/'failed/policy.py'),sha256=f.records['failed']['record']['source']['sha256'])) for seed in (114,124)]))
    return c


class CrossProxy:
    def __init__(self,cross,mid):self.cross,self.mid=cross,mid
    def __getattr__(self,key):
        method=getattr(self.cross,key)
        def call(**kw):
            if kw.get('memory_id')=='m1':kw['memory_id']=self.mid
            return method(**kw)
        return call


def proxy(f,s):
    def authorize(**kw):kw['memory_id']=f.mid;return s.family.cross_seed.authorize(**kw)
    return SimpleNamespace(dev_v2=s.family.dev_v2,store=SimpleNamespace(get_memory=lambda mid:s.family.get_memory(f.mid)),
        authorize_dev_v2_validation=authorize,import_dev_v2_execution=s.import_dev_v2_execution,
        import_dev_v2_trial=s.family.dev_v2.import_trial,cross_seed=CrossProxy(s.family.cross_seed,f.mid))


def scene(f,s,seed,*,failure=False,fault=None,checks=None):
    f.task=TARGET;p=proxy(f,s)
    args,contract=target(f,p,seed,failure=failure,checks=checks)
    probe=p.cross_seed.probe(**args)
    if probe['role']=='could_not_finish':return s.family.status(f.mid)
    n=5 if failure else 3
    pairs=[dict(pair_id=str(i),control_cell=f'c{seed}-{i}',treatment_cell=f't{seed}-{i}') for i in range(n)]
    plan=p.cross_seed.plan(memory_id='m1',target_seed=seed,pairs=pairs,dev_model=MODEL)
    # The sealed test fixtures use one assistance credit. Production templates
    # use zero; bind the plan to the same actual allocation for these fixtures.
    for pair in pairs:
        for arm in ('control_cell','treatment_cell'):
            name=pair[arm];treatment=arm=='treatment_cell'
            r=f.make(name,seed=seed,failure=failure and (not treatment or fault=='remains'),completed_calls=0 if treatment and fault=='earlier' else 1,scene=contract,
                exposure='candidate_validation' if treatment else 'none',memory_id=f.mid if treatment else None,code=f'def policy(sdk):\n    # {name}\n    pass\n')
            prepare(f,p,plan,name)
            import_trial(p,name)
        p.cross_seed.pair(memory_id='m1',target_seed=seed,control_import_id='trial-'+pair['control_cell'],treatment_import_id='trial-'+pair['treatment_cell'])
    return s.family.status(f.mid)


def test_fixture_and_rewrite_isolation(tmp_path):
    f,s,spec,args=setup(tmp_path);r=s.family.fixture_create(**args)
    assert r['statement']==STATEMENT and r['operator_preset'] and r['preset_levels']==dict(scene='trusted',task='trusted')
    assert r['user_quotes']==spec['user_quotes'] and r['label']==LABEL
    bad=copy.deepcopy(spec);bad['guidance']='changed'
    with pytest.raises(StateConflictError):s.family.fixture_create(**dict(args,spec=bad,spec_sha256=validate_spec(bad)))
    c=candidate(f,s);before=digest(spec['guidance'].encode())
    assert c['from_source']['guidance_sha256']==before and s.family._read('fixtures','fixture-one')['guidance']==spec['guidance']
    source=dict(kind='fixture',id='fixture-one');rw=rewrite(s.family.source(source))
    assert s.family.generalize(source=source,rewrite=rw)==c
    assert s.list_memories()==s.list_dev_v2_memories()==s.validation_queue()==s.cross_seed.queue()==[]
    assert s.retrieve(spec['context'],now=1)==[]
    for action in (lambda:s.export_package('fixture-one'),lambda:s.get_memory(f.mid),lambda:s.authorize_dev_use(memory_id=f.mid,development_id='dev:1',context=spec['context'],source_policy_sha256='a'*64,now=1,memory_version=1)):
        with pytest.raises((StateConflictError,ValueError,KeyError,FileNotFoundError)):action()
    s.family.fixture_revoke(fixture_id='fixture-one',reason='test complete')
    assert s.family.get(f.mid)['status']=='revoked'
    with pytest.raises(StateConflictError):s.family.get_memory(f.mid)
    with pytest.raises(StateConflictError):s.family.generalize(source=source,rewrite=rw)


def test_fixture_spec_and_pending_scene_cards(tmp_path):
    f,s,spec,_=setup(tmp_path)
    for bad in (dict(spec,label='ordinary'),dict(spec,user_quotes=[]),dict(spec,user_quotes=[' ']),
                {k:v for k,v in spec.items() if k!='user_quotes'},dict(spec,extra=True)):
        with pytest.raises((StateConflictError,ValueError)):validate_spec(bad)
    candidate(f,s)
    status=s.family.status(f.mid)
    assert [(r['target_seed'],r['role']) for r in status['scenes']]==[(114,'pending'),(124,'pending')]
    assert status['conditions']['missing']
    s.family.fixture_revoke(fixture_id='fixture-one',reason='end')
    assert s.family.public()['fixtures'][0]['status']=='revoked'


@pytest.mark.parametrize('text',['','x'*601,'Move into the sink.','Use pick_place_counter_to_sink','Seed 102 needs a lift.','Move to world x=0.4 y=0.5 z=1.2','Move to [0.1, 0.2, 0.3].'])
def test_bad_rewrites(tmp_path,text):
    f,s,*_=setup(tmp_path);source=dict(kind='fixture',id='fixture-one')
    with pytest.raises((StateConflictError,ValueError)):s.family.generalize(source=source,rewrite=rewrite(s.family.source(source),text))


def test_family_neutral_pipeline_and_revoke(tmp_path):
    f,s,*_=setup(tmp_path);candidate(f,s)
    scene(f,s,114);result=scene(f,s,124)
    assert result['decision']['decision']=='would be trusted for the family'
    assert result['decision']['label']==LABEL and result['decision']['excluded_from_results']
    assert result['conditions']['neutral_only'] and not result['conditions']['worked_on']
    assert [(r['pairs_done'],r['pairs_needed'],r['verdict']) for r in result['scenes']]==[(3,3,'neutral_no_harm')]*2
    assert FamilyCrossSeed._report is CrossSeed._report and FamilyCrossSeed._no_harm is CrossSeed._no_harm
    s.family.fixture_revoke(fixture_id='fixture-one',reason='end')
    assert s.family.status(f.mid)['decision']['decision']=='Revoked'
    plan=s.family.cross_seed.get('plans',f.mid,114)
    with pytest.raises(StateConflictError):s.family.cross_seed.plan(memory_id=f.mid,target_seed=114,pairs=plan['pairs'],dev_model=MODEL)


@pytest.mark.parametrize('failure,fault,verdict',[(True,None,'transfer_passed'),(True,'remains','transfer_failed'),(False,'earlier','neutral_harm')])
def test_scene_rules(tmp_path,failure,fault,verdict):
    f,s,*_=setup(tmp_path);candidate(f,s)
    result=scene(f,s,114,failure=failure,fault=fault)
    assert result['scenes'][0]['verdict']==verdict
    assert result['scenes'][0]['pairs_needed']==(5 if failure else 3)


def test_family_conditions():
    scenes=[dict(target_task_id=TARGET,target_seed=s,verdict='neutral_no_harm') for s in (114,124)]
    assert not family_conditions(True,scenes)['missing']
    assert family_conditions(False,scenes)['missing']
    assert family_conditions(True,scenes[:1])['missing']
    assert not family_conditions(True,scenes+[dict(target_task_id=TARGET,target_seed=117,verdict='could_not_finish')])['missing']
    assert family_conditions(True,scenes+[dict(target_task_id=TARGET,target_seed=117,verdict='transfer_failed')])['missing']


@pytest.mark.parametrize('checks',[{'safety':False},{'scene_reproduction':False}])
def test_unsafe_or_unreproduced_probe_never_creates_pairs(tmp_path,checks):
    f,s,*_=setup(tmp_path);candidate(f,s)
    result=scene(f,s,114,checks=checks)
    assert result['scenes'][0]['verdict']=='could_not_finish'
    assert result['scenes'][0]['pairs_needed']==0
    assert result['decision']['decision']=='would not'
    assert s.family.cross_seed.get('plans',f.mid,114) is None


def real_source(tmp_path,monkeypatch):
    import tests.test_single_scene as single
    from tests.test_cross_seed import batch,finish
    monkeypatch.setattr(single,'Fixture',Fixture)
    f,s,plan=single.setup_scene(tmp_path,interface=True)
    s.record_plan('m1',plan);single.import_pairs(f,s)
    assert s.evaluate('m1').status.value=='trusted'
    with pytest.raises(StateConflictError):s.family.source(dict(kind='memory',id='m1'))
    for seed,failure in ((103,True),(104,False)):
        plan,_=batch(f,s,seed,failure=failure);finish(f,s,plan)
    assert s.v2.get_decision('m1')['trust_scope']=='task_scoped'
    return f,s


def test_real_promotion_read_and_use_demotion(tmp_path,monkeypatch):
    f,s=real_source(tmp_path,monkeypatch);candidate(f,s,kind='memory')
    scene(f,s,114);result=scene(f,s,124)
    assert result['status']=='family_trusted'
    context=s.family.cross_seed.get('plans',f.mid,114)['context']
    refs=s.family.references(context);assert refs[0]['memory_id']==f.mid
    grant=s.authorize_dev_use(memory_id=f.mid,memory_version=1,context=context,development_id='family:use:1',source_policy_sha256='a'*64,now=1)
    assert grant['trust_scope']=='family_scoped' and s.get_dev_use(grant['grant_id'])==grant
    for i,failed in enumerate([True,False,True,False,True]):s.family.record_use(candidate_id=f.mid,use_id=str(i),signature_present=failed)
    assert s.family.get(f.mid)['status']=='candidate' and not s.family.references(context)
    with pytest.raises(StateConflictError):s.get_dev_use(grant['grant_id'])


def test_source_demote_and_kill(tmp_path,monkeypatch):
    f,s=real_source(tmp_path,monkeypatch);candidate(f,s,kind='memory')
    scene(f,s,114);scene(f,s,124)
    s.disable('m1',actor='test operator',reason='source off')
    assert s.family.get(f.mid)['status']=='candidate'
    with pytest.raises(StateConflictError):s.family.get_memory(f.mid)
    assert s.family.disable(candidate_id=f.mid,reason='kill')['status']=='disabled'


@pytest.mark.parametrize('change',[{'task_id':SOURCE},{'task_id':'open_cabinet'},{'execution_profile':'r3-20261025'}, {'sdk_interface_sha256':'0'*64},{'planner_dev_local_sha256':'a'*64}])
def test_target_refuses_scope(tmp_path,change):
    f,s,*_=setup(tmp_path);candidate(f,s);f.task=TARGET;p=proxy(f,s)
    args,_=target(f,p,114,failure=False)
    e=s.dev_v2.get(args['execution_import_id']);e=copy.deepcopy(e);e['context'].update(change)
    with pytest.raises((StateConflictError,ValueError)):s.family.cross_seed._target(f.mid,114,e,args['scene_contract_ref'])


def test_importer_only(tmp_path):
    from attention_memory_service.memory_service_api import create_app
    from tests.test_sync_legacy_routes import sync_request
    f,s,_,args=setup(tmp_path);key='online-key-123456789';imp='importer-key-123456789'
    app=create_app(s.store.path,api_key=key,importer_key=imp)
    assert sync_request(app,'/dev-v2/family/fixture-create',method='POST',key=key,payload=args)[0]==403
    assert sync_request(app,'/dev-v2/family/fixture-create',method='POST',key=key,importer_key=imp,payload=args)[0]==200
