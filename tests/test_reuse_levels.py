"""T2: disposable stores, synthetic versions and model responses only."""
import copy
import json
import sqlite3
import time

import pytest

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.dev_v2 import digest, encoded
from attention_memory_service.memory_service_api import create_app
from attention_memory_service.reuse_notes import REWRITE_PROMPT
from attention_memory_service.skill_tags import TASKS, tags_for
from tests.test_provisional_notes import payload, service, API, IMPORTER
from tests.test_sync_legacy_routes import sync_request

LEVELS = ('same_scene','same_task','same_skill','general','same_family')
OPERATOR = 'operator-key-123456789'


def request(mid='p-level', verified=True):
    req = payload(mid)
    req['context'].update(task_id='pick_place_counter_to_cabinet', task_prompt='Pick the object',
                          execution_profile='r5-fixture-20261031', planner_behavior_sha256='d'*64,
                          planner_dev_local_sha256='c'*64)
    req['title'] = 'Retreat from the cabinet before lifting'
    req['guidance'] = 'Retreat at x=1.23 before lifting in this cabinet.'
    raw = encoded(dict(version=2, valid=True, source=dict(sha256=req['origin']['digests']['help_source']),
                       native_success=verified, eval_verdict='pass', eval=dict(passed=True)))
    req['source_version_evidence'] = dict(raw_version=raw, sha256=digest(raw.encode()))
    return req


def rewrite(svc, mid='p-level', **changes):
    p = svc.provenance(mid);m = svc.get_memory(mid)
    response = dict(title='Retreat before lifting a carried object',
                    guidance='Retreat in small steps until clear of the door before lifting.',
                    applies_when='When carrying an object inside a cabinet.')
    response.update(changes)
    raw = encoded(response)
    prompt = dict(instruction=REWRITE_PROMPT,title=p['title'],guidance=m.guidance,task_id=m.applicability['task_id'])
    return dict(response, receipt=dict(provider='codex',model='synthetic-model',prompt=prompt,
        prompt_sha256=digest(encoded(prompt).encode()),raw_response=raw,response_sha256=digest(raw.encode()),
        started_at=1.,finished_at=2.))


def create(svc, **kw):
    req=request(**kw);svc.create_provisional_note(**req)
    svc.generalize_note(memory_id=req['memory_id'],rewrite=rewrite(svc,req['memory_id']))
    return req


def matching(svc, ctx):
    return {m.memory_id:level for m,level in svc.retrieve_levels(ctx,now=time.time(),levels=LEVELS)}


def test_table_has_fourteen_tasks_cross_family_overlap_and_fixture_precision():
    assert len(TASKS)==14
    assert set(tags_for('turn_off_stove')) & set(tags_for('kettle_boiling')) == {'twist_knob:stove'}
    assert not set(tags_for('open_cabinet')) & set(tags_for('open_drawer'))
    assert not tags_for('unknown')


def test_l1_l2_l3_identity_and_legacy(tmp_path):
    svc=service(tmp_path);req=create(svc);ctx=req['context']
    assert matching(svc,ctx)=={'p-level':'same_scene'}
    target=dict(ctx,task_id='categorize_condiments',task_prompt='Sort condiments',scene_id='other',seed=101)
    assert matching(svc,target)=={'p-level-g':'same_skill'}
    assert not matching(svc,dict(target,task_id='open_drawer'))
    for key,value in [('planner_behavior_sha256','e'*64),('sdk_interface_sha256','e'*64),
                      ('execution_profile','other'),('benchmark','other'),('suite','other'),
                      ('robot','other'),('controller','other'),('perception_mode','vision'),('namespace','other')]:
        assert matching(svc,dict(target,**{key:value}))=={}, key
    assert matching(svc,dict(target,planner_dev_local_sha256='e'*64)) # T1 audit-only snapshot
    result=svc.confirm_note_level(memory_id='p-level',level='general',actor='alice',opened_at=time.time()-3)
    assert result['generalized']['confirmation']['human_attention_seconds']>=3
    assert matching(svc,dict(target,task_id='open_drawer'))=={'p-level-g':'general'}
    assert not matching(svc,dict(target,planner_behavior_sha256='e'*64))
    svc.confirm_note_level(memory_id='p-level',level='same_task',actor='alice',opened_at=time.time()-1)
    assert 'p-level-g' in matching(svc,ctx)
    assert not matching(svc,target)
    # Legacy rows retain the old family matcher; the A-D analysis field never participates.
    req=payload('legacy',context=dict(ctx,task_id='pick_place_counter_to_sink'))
    svc.create_provisional_note(**req)
    with sqlite3.connect(svc.store.path) as db:
        p=svc.v2._provenance('legacy')
        for k in ('level','skill_tags','skill_tags_version'):p.pop(k,None)
        db.execute('UPDATE memory_v2_provenance SET payload=? WHERE memory_id=?',(encoded(p),'legacy'))
    assert matching(svc,ctx)['legacy']=='same_family'


@pytest.mark.parametrize('changes',[
    {'guidance':'Move to world coordinates x=1.23.'},
    {'guidance':'Use seed 101.'},
    {'guidance':'Repeat pick_place_counter_to_cabinet.'},
    {'guidance':'Move to [1, 2, 3].'},
    {'title':'note'}, {'applies_when':''}, {'guidance':'Move 1.23 m left.'},
])
def test_generalization_rejects_specifics_and_placeholders(tmp_path,changes):
    svc=service(tmp_path);svc.create_provisional_note(**request())
    with pytest.raises((ValueError,StateConflictError)):
        svc.generalize_note(memory_id='p-level',rewrite=rewrite(svc,**changes))
    assert svc.get_memory('p-level').guidance==request()['guidance']
    assert svc.store.get_memory('p-level-g') is None


def test_raw_model_receipt_and_copy_provenance_are_rechecked(tmp_path):
    svc=service(tmp_path);svc.create_provisional_note(**request())
    r=rewrite(svc);r['guidance']='A changed response.'
    with pytest.raises(StateConflictError,match='raw model response'):
        svc.generalize_note(memory_id='p-level',rewrite=r)
    r=rewrite(svc);svc.generalize_note(memory_id='p-level',rewrite=r)
    assert svc.generalize_note(memory_id='p-level',rewrite=r)['original']['guidance']==request()['guidance']
    p=svc.provenance('p-level-g');assert p['model_rewrite']==r
    with sqlite3.connect(svc.store.path) as db:
        p['model_rewrite']['receipt']['response_sha256']='f'*64
        db.execute('UPDATE memory_v2_provenance SET payload=? WHERE memory_id=?',(encoded(p),'p-level-g'))
    with pytest.raises(StateConflictError,match='digest mismatch'):
        svc.provenance('p-level-g')


def test_source_verified_requires_both_and_bound_version_bytes(tmp_path):
    svc=service(tmp_path)
    for mid,native,verdict in [('yes',True,'pass'),('native-fail',False,'pass'),('eval-fail',True,'fail')]:
        req=request(mid);v=json.loads(req['source_version_evidence']['raw_version'])
        v.update(native_success=native,eval_verdict=verdict,eval=dict(passed=verdict=='pass'))
        raw=encoded(v);req['source_version_evidence']=dict(raw_version=raw,sha256=digest(raw.encode()))
        svc.create_provisional_note(**req)
        p=svc.provenance(mid)
        assert p['source_verified'] is (native and verdict=='pass')
        assert p['source_verified_evidence']['sha256']==digest(raw.encode())
        assert svc.get_memory(mid).status.value=='provisional'
        with pytest.raises(StateConflictError,match='provisional'):
            svc.promote(mid)
    for change in ['digest','version','source']:
        req=request(change)
        if change=='digest':req['source_version_evidence']['sha256']='f'*64
        else:
            v=json.loads(req['source_version_evidence']['raw_version'])
            v[change]=3 if change=='version' else dict(sha256='f'*64)
            raw=encoded(v);req['source_version_evidence']=dict(raw_version=raw,sha256=digest(raw.encode()))
        with pytest.raises(StateConflictError):svc.create_provisional_note(**req)
    req=request('no-evidence');req.pop('source_version_evidence');svc.create_provisional_note(**req)
    assert svc.provenance('no-evidence')['source_verified'] is False
    assert 'no-evidence' in matching(svc,req['context'])


def test_edit_attention_versioning_parent_lifecycle_and_grants(tmp_path):
    svc=service(tmp_path);req=create(svc);ctx=dict(req['context'],task_id='categorize_condiments',task_prompt='Sort')
    grant=svc.authorize_dev_use(memory_id='p-level-g',memory_version=1,development_id='run:v1',
        context=ctx,source_policy_sha256='e'*64,now=time.time())
    assert grant['match_level']=='same_skill' and grant['trusted_here'] is False
    assert grant['source_verified'] is True and 'source_verified_evidence' not in grant
    old=svc.provenance('p-level-g')['model_rewrite']
    svc.confirm_note_level(memory_id='p-level',level='general',actor='alice',opened_at=time.time()-2,
                          edit=dict(guidance='Retreat slowly before lifting.',applies_when='When carrying an object.'))
    assert svc.get_memory('p-level-g').version==2
    assert svc.provenance('p-level-g')['model_rewrite']==old
    with pytest.raises(StateConflictError):
        svc.authorize_dev_use(memory_id='p-level-g',memory_version=1,development_id='run:v2',
            context=ctx,source_policy_sha256='e'*64,now=time.time())
    assert any(e['status']=='level_confirmed' for e in svc.history('p-level-g'))
    svc.disable('p-level',actor='alice',reason='source withdrawn')
    assert not matching(svc,ctx)


def test_api_roles_metadata_and_no_eval_leak(tmp_path):
    app=create_app(tmp_path/'api.sqlite',api_key=API,importer_key=IMPORTER,operator_key=OPERATOR)
    from attention_memory_service import MemoryService
    svc=MemoryService(tmp_path/'api.sqlite')
    create(svc)
    body=dict(memory_id='p-level',level='general',actor='alice',opened_at=time.time()-1)
    assert sync_request(app,'/dev-v2/reuse/confirm',method='POST',key=API,payload=body)[0]==403
    assert sync_request(app,'/dev-v2/reuse/confirm',method='POST',key=API,operator_key=OPERATOR,payload=body)[0]==200
    code,rows=sync_request(app,'/retrieve',method='POST',key=API,payload=dict(context=request()['context'],now=time.time(),match_levels=list(LEVELS)))
    assert code==200 and rows and all('level' in row for row in rows)
    assert all('raw_version' not in encoded(row) and 'native_success' not in encoded(row) for row in rows)


def test_agent_source_label_reads_nested_export_verdict():
    from attention_memory_service.reuse_notes import source_label
    req=request();origin=dict(req['origin'],channel='dev_agent')
    record=json.loads(req['source_version_evidence']['raw_version'])
    nested={k:record.pop(k) for k in ('native_success','eval_verdict','eval')}
    record['agent']=nested
    raw=encoded(record)
    evidence=dict(raw_version=raw,sha256=digest(raw.encode()))
    assert source_label(origin,evidence)['source_verified'] is True
    record['agent']['eval_verdict']='fail'
    raw=encoded(record)
    assert source_label(origin,dict(raw_version=raw,sha256=digest(raw.encode())))['source_verified'] is False


def test_one_call_version_label_uses_its_exact_settled_outcomes(tmp_path):
    req=request();record=json.loads(req['source_version_evidence']['raw_version'])
    for k in ('native_success','eval_verdict','eval'):record.pop(k)
    record['config_sha256']='a'*64
    def ref(value):
        raw=encoded(value);return dict(raw=raw,sha256=digest(raw.encode()))
    native=dict(native_success=True,evaluated=True)
    native_ref=ref(native)
    bundle=dict(dev_version=2,policy_id=req['origin']['policy_id'],policy_sha256=req['origin']['digests']['help_source'],
                config_sha256='a'*64,run_id='run:cell-1',attempt_id='attempt:cell-1:1',native_success=True,
                native_evaluator=native,artifacts=dict(native_result=dict(sha256=native_ref['sha256'])))
    bundle_ref=ref(bundle);review_ref=ref(dict(status='completed',passed=True))
    execute_job=dict(cell='cell-1',version=2,kind='execute',state='settled',
                     receipt=dict(bundle=dict(sha256=bundle_ref['sha256']),native_result=dict(sha256=native_ref['sha256'])))
    eval_job=dict(cell='cell-1',version=2,kind='eval',state='settled',receipt=dict(eval_review=dict(sha256=review_ref['sha256'])))
    raw=encoded(record)
    evidence=dict(raw_version=raw,sha256=digest(raw.encode()),outcomes=dict(bundle=bundle_ref,native=native_ref,
        eval=review_ref,execute_job=ref(execute_job),eval_job=ref(eval_job)))
    svc=service(tmp_path);req['source_version_evidence']=evidence;svc.create_provisional_note(**req)
    assert svc.provenance('p-level')['source_verified'] is True
    req['memory_id']='p-wrong';req['source_version_evidence']=copy.deepcopy(evidence)
    eval_job['version']=3
    req['source_version_evidence']['outcomes']['eval_job']=ref(eval_job)
    with pytest.raises(StateConflictError,match='job identity'):svc.create_provisional_note(**req)


def test_confirming_an_edited_copy_preserves_edit_and_immutable_model_response(tmp_path):
    svc=service(tmp_path);create(svc)
    edit=dict(guidance='Retreat slowly before lifting.',applies_when='When carrying an object.')
    svc.confirm_note_level(memory_id='p-level',level='same_skill',actor='alice',opened_at=time.time()-1,edit=edit)
    svc.confirm_note_level(memory_id='p-level',level='general',actor='bob',opened_at=time.time()-1)
    p=svc.provenance('p-level-g')
    assert svc.get_memory('p-level-g').guidance==edit['guidance'] and p['applies_when']==edit['applies_when']
    assert p['edit_confirmation']['actor']=='alice' and p['level_confirmation']['actor']=='bob'
    assert p['model_rewrite']['guidance']!=edit['guidance']


def test_sdk_limit_can_be_general_without_world_coordinates(tmp_path):
    svc=service(tmp_path);svc.create_provisional_note(**request())
    r=rewrite(svc,title='Keep base steps within the SDK limit',
              guidance='The SDK limits each base step to at most 0.60 m per call.',
              applies_when='When moving the base.')
    svc.generalize_note(memory_id='p-level',rewrite=r)
    svc.confirm_note_level(memory_id='p-level',level='general',actor='alice',opened_at=time.time()-1)
    assert svc.provenance('p-level-g')['level']=='general'


def test_copy_lifecycle_preserves_readable_confirmation_and_provisional_trust(tmp_path):
    svc=service(tmp_path);req=create(svc)
    svc.confirm_note_level(memory_id='p-level',level='general',actor='alice',opened_at=time.time()-1)
    svc.disable('p-level-g',actor='alice',reason='pause')
    assert svc.provenance('p-level-g')['level_confirmation']['actor']=='alice'
    svc.enable('p-level-g',actor='alice',reason='resume')
    assert svc.get_memory('p-level-g').status.value=='provisional'
    assert matching(svc,dict(req['context'],task_id='open_drawer',task_prompt='Open drawer'))['p-level-g']=='general'


def test_generalized_copy_drops_source_scene_summary(tmp_path):
    svc=service(tmp_path);req=request();req['summary']='Lift at x=1.23 in this cabinet'
    svc.create_provisional_note(**req)
    svc.generalize_note(memory_id='p-level',rewrite=rewrite(svc))
    assert svc.provenance('p-level')['summary']==req['summary']
    assert 'summary' not in svc.provenance('p-level-g')
    assert svc.get_dev_v2_memory_public('p-level-g')['summary'] is None
