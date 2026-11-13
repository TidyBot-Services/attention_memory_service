"""New notes reuse behavior; validation still binds the exact deployment."""
import time

import pytest

from attention_memory_service.applicability import (BEHAVIOR_KEY, DEV_LOCAL_KEY, LEVELS,
    require_profile_match)
from attention_memory_service.core.store import StateConflictError
from tests.test_provisional_notes import context, payload, service
from tests.test_test_seeded_notes import payload as seeded_payload
from tests.test_sync_legacy_routes import sync_request
from tests.test_provisional_notes import API, IMPORTER
from attention_memory_service.memory_service_api import create_app


def scoped(snapshot,behavior='e'*64):
    value=dict(context(task_id='pick_place_cup'),execution_profile='r4-test',
               planner_dev_local_sha256=snapshot)
    if behavior is not None:value[BEHAVIOR_KEY]=behavior
    return value


@pytest.mark.parametrize('kind',['provisional','seeded'])
@pytest.mark.parametrize('behavior',['e'*64,None])
def test_import_retrieve_expose_and_visible_rejection(tmp_path,kind,behavior):
    svc=service(tmp_path)
    req=payload() if kind=='provisional' else seeded_payload()
    req['context']=scoped('a'*64,behavior)
    create=svc.create_provisional_note if kind=='provisional' else svc.create_test_seeded_note
    create(**req)
    assert svc.store.get_memory(req['memory_id']).applicability==req['context']
    same=scoped('b'*64,behavior)
    for index,changes in enumerate(({},dict(seed=103),dict(task_id='pick_place_plate'))):
        here=dict(same,**changes)
        rows=svc.retrieve_levels(here,now=time.time(),levels=LEVELS)
        assert bool(rows)==(behavior is not None)
        args=dict(memory_id=req['memory_id'],development_id='use-'+str(index),context=here,
                  source_policy_sha256='c'*64,now=time.time())
        if behavior:
            grant=svc.authorize_dev_use(**args)
            assert grant[BEHAVIOR_KEY]==behavior and grant[DEV_LOCAL_KEY]=='a'*64
        else:
            with pytest.raises(StateConflictError,match=DEV_LOCAL_KEY):svc.authorize_dev_use(**args)
    bad=scoped('b'*64,'f'*64)
    assert svc.retrieve_levels(bad,now=time.time(),levels=LEVELS)==[]
    with pytest.raises(StateConflictError,match=BEHAVIOR_KEY if behavior else DEV_LOCAL_KEY):
        svc.authorize_dev_use(memory_id=req['memory_id'],development_id='bad',context=bad,
                             source_policy_sha256='c'*64,now=time.time())
    app=create_app(svc.store.path,api_key=API,importer_key=IMPORTER)
    code,response=sync_request(app,'/dev-use-grants',method='POST',key=API,importer_key=IMPORTER,
        payload=dict(memory_id=req['memory_id'],development_id='bad-http',context=bad,
                     source_policy_sha256='c'*64,now=time.time()))
    assert code==409 and (BEHAVIOR_KEY if behavior else DEV_LOCAL_KEY) in response['detail']
    row,=svc.list_dev_v2_memories(context=bad)
    assert row.get(BEHAVIOR_KEY)==behavior
    # Validation is strict even if behavior matches; defaults cannot opt in.
    with pytest.raises(StateConflictError,match=DEV_LOCAL_KEY):
        require_profile_match(req['context'],same)
    assert svc.retrieve_levels(req['context'],now=time.time(),levels=LEVELS)


@pytest.mark.parametrize('bad',[None,'', 'z'*64,'A'*64,'a'*63,1])
def test_behavior_digest_validation_http(tmp_path,bad):
    svc=service(tmp_path)
    app=create_app(svc.store.path,api_key=API,importer_key=IMPORTER)
    code,response=sync_request(app,'/retrieve',method='POST',key=API,importer_key=IMPORTER,
        payload=dict(context=dict(context(),**{BEHAVIOR_KEY:bad}),now=time.time()))
    assert code==422 and BEHAVIOR_KEY in response['detail']
