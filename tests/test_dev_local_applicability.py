"""Snapshot hashes isolate reads, grants and validation at all applicability levels."""
import json
from pathlib import Path
import subprocess
import time
from types import ModuleType

import pytest
from attention_memory_service.applicability import (DEV_LOCAL_KEY as KEY, LEVELS, config_profile,
    profile_fields, profile_matches, require_profile_match, same_scene, same_task, same_family)
from attention_memory_service.core.store import StateConflictError
from attention_memory_service.memory_service_api import create_app
from attention_memory_service.seeded_notes import TUPLE_KEYS
from tests.test_provisional_notes import API, IMPORTER, context, payload, service
from tests.test_test_seeded_notes import payload as seeded_payload
from tests.test_sync_legacy_routes import sync_request
from tests.legacy_profile_flow import capture


def scoped(value,sha):
    value=dict(value,execution_profile='r3-20261025')
    if sha is not None:value[KEY]=sha
    return value


@pytest.mark.parametrize('a,b',[(None,None),('a'*64,'a'*64),('a'*64,'b'*64),(None,'a'*64),('a'*64,None)])
def test_all_levels(a,b):
    note=scoped(context(task_id='pick_place_cup'),a);ctx=scoped(context(task_id='pick_place_cup'),b)
    assert profile_matches(note,ctx)==(a==b)
    assert same_scene(note,ctx,TUPLE_KEYS)==(a==b)
    assert same_task(note,dict(ctx,seed=103),{ctx['task_prompt']})==(a==b)
    assert same_family(note,dict(ctx,task_id='pick_place_plate'))==(a==b)
    if a!=b:
        with pytest.raises(StateConflictError,match=f'planner_dev_local_sha256 note {a}, context {b}'):
            require_profile_match(note,ctx)


@pytest.mark.parametrize('kind',['provisional','seeded'])
@pytest.mark.parametrize('a,b',[('a'*64,'a'*64),('a'*64,'b'*64),(None,'a'*64),('a'*64,None)])
def test_import_retrieve_grants_and_queue(tmp_path,kind,a,b):
    svc=service(tmp_path)
    req=payload() if kind=='provisional' else seeded_payload()
    req['context']=scoped(context(task_id='pick_place_cup'),a)
    create=svc.create_provisional_note if kind=='provisional' else svc.create_test_seeded_note
    create(**req)
    assert svc.store.get_memory(req['memory_id']).applicability==req['context']
    ctx=scoped(context(task_id='pick_place_cup'),b)
    for changes,level in [({},'same_scene'),({'seed':103},'same_task'),({'task_id':'pick_place_plate'},'same_family')]:
        here=dict(ctx,**changes)
        rows=svc.retrieve_levels(here,now=time.time(),levels=LEVELS)
        assert [(m.memory_id,l) for m,l in rows]==([(req['memory_id'],level)] if a==b else [])
        args=dict(memory_id=req['memory_id'],development_id=level,context=here,source_policy_sha256='c'*64,now=time.time())
        if a==b:assert svc.authorize_dev_use(**args)[KEY]==a
        else:
            with pytest.raises(StateConflictError,match='planner_dev_local_sha256'):svc.authorize_dev_use(**args)
    row,=svc.list_dev_v2_memories(context=ctx)
    assert row.get(KEY)==a and row['applicability_summary'].get(KEY)==a
    if kind=='provisional':
        from tests.test_auto_validation import outcome
        outcome(svc,req['memory_id'],0,ctx=req['context'])
        queued,=svc.validation_queue()
        assert queued.get(KEY)==a


@pytest.mark.parametrize('bad',['','a'*63,'a'*65,'A'*64,'z'*64,'a'*64+'\n',None,1,True,[],{}])
def test_invalid_sha_http_422(tmp_path,bad):
    svc=service(tmp_path);app=create_app(svc.store.path,api_key=API,importer_key=IMPORTER)
    ctx=dict(context(),**{KEY:bad})
    for path,body in [('/retrieve',dict(context=ctx,now=time.time())),
                      ('/dev-v2/provisional-notes',payload(context=ctx)),
                      ('/dev-v2/test-seeded-notes',dict(seeded_payload(),context=ctx)),
                      ('/memories/missing/plan',{KEY:bad})]:
        code,response=sync_request(app,path,method='PUT' if path.endswith('/plan') else 'POST',key=API,importer_key=IMPORTER,payload=body)
        assert code==422 and KEY in response['detail']


def test_config_and_plan_fields():
    config=dict(planner_source=dict(kind='dev-local',source_root='/source'),
                execution_settings=dict(profile='r3-20261025',planner_deployment=dict(sha256='a'*64)))
    fields={'execution_profile':'r3-20261025',KEY:'a'*64}
    assert config_profile(config)==fields and profile_fields(fields)==fields
    config['execution_settings']['planner_deployment']['sha256']='bad'
    with pytest.raises(ValueError,match=KEY):config_profile(config)


def test_registered_fields_and_old_note_response_bytes(tmp_path):
    old=ModuleType('baseline')
    source=subprocess.check_output(['git','show','dc7f5ef2af172f8e3516ae55121c667638de7ba9:attention_memory_service/applicability.py'],text=True)
    exec(compile(source,'applicability.py','exec'),old.__dict__)
    for value in ({},{'execution_profile':'r3-20261025'}):
        assert json.dumps(profile_fields(value))==json.dumps(old.profile_fields(value))
    for config in ({},{'execution_settings':{'profile':'r3-20261025'}}):
        assert json.dumps(config_profile(config))==json.dumps(old.config_profile(config))
    assert capture(tmp_path)==Path(__file__).with_name('legacy_profile_flow.json').read_bytes()


def test_validation_and_cross_seed_propagation(tmp_path,monkeypatch):
    from tests.dev_v2_fixtures import Fixture
    from tests.test_single_scene import setup_scene,import_pairs
    from tests.test_cross_seed import batch,finish,target
    original=Fixture.write
    current={'sha':'a'*64}
    def write(path,value):
        if path.name=='config.json' and isinstance(value,dict):
            value['planner_source']=dict(kind='dev-local',source_root=str(tmp_path))
            value['execution_settings']['planner_deployment']=dict(sha256=current['sha'])
        if isinstance(value,dict) and isinstance(value.get('context'),dict):
            value['context'][KEY]=current['sha']
        return original(path,value)
    monkeypatch.setattr(Fixture,'write',staticmethod(write))
    fixture,svc,plan=setup_scene(tmp_path,interface=True,execution_profile='r3-20261025')
    assert svc.store.get_memory('m1').applicability[KEY]=='a'*64
    with pytest.raises(StateConflictError,match=KEY):svc.record_plan('m1',plan)
    plan[KEY]='a'*64;svc.record_plan('m1',plan)
    record=fixture.records['c0']
    with pytest.raises(StateConflictError,match=KEY):
        svc.authorize_dev_v2_validation(memory_id='m1',cell='c0',version=1,seed=101,
            context=dict(record['context'],**{KEY:'b'*64}),policy_sha256=record['record']['source']['sha256'],
            config_sha256=record['descriptor']['config']['sha256'],exposure='none')
    import_pairs(fixture,svc);svc.evaluate('m1')
    queued,=svc.cross_seed.queue()
    assert queued[KEY]=='a'*64
    cross_plan,_=batch(fixture,svc,114,failure=True)
    assert cross_plan[KEY]==cross_plan['context'][KEY]=='a'*64
    finish(fixture,svc,cross_plan)
    current['sha']='b'*64
    args,_=target(fixture,svc,120)
    with pytest.raises(StateConflictError,match=KEY):svc.cross_seed.probe(**args)
