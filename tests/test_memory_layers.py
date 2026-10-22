"""Nested-layer boundaries, public projections and provenance compatibility."""
import hashlib
import time

import pytest
from pathlib import Path

from attention_memory_service.applicability import same_family, layer_of
from attention_memory_service.families import family_of
from attention_memory_service.dev_v2 import digest, encoded
from attention_memory_service.memory_service_client import _memory
from attention_memory_service.memory_service_api import create_app
from attention_memory_service.core.store import StateConflictError
from tests.test_provisional_notes import payload, context, service, API, IMPORTER
from tests.test_sync_legacy_routes import sync_request
from tests.test_dev_v2 import source_setup, candidate


def link(req):
    origin = req['origin']
    return dict(experiment_id=origin['experiment_id'], cell=origin['cell'], version=origin['help_version'],
                run_note_id='r-' + hashlib.sha256(
                    f"{origin['experiment_id']}:{origin['cell']}:{origin['help_version']}".encode()).hexdigest()[:24])


def test_families():
    assert family_of('pick_place_cup') == family_of('pick_place_plate') == 'pick_and_place'
    assert family_of('pick') is None and family_of(None) is None
    a, b = context(task_id='pick_place_cup'), context(task_id='pick_place_plate', task_prompt='Other prompt')
    assert same_family(a, b)
    for key in ('benchmark', 'suite', 'robot', 'controller', 'perception_mode', 'namespace', 'sdk_interface_sha256'):
        assert not same_family(a, {**b, key: 'different'})
    assert not same_family(a, a)
    assert [layer_of(k) for k in ('same_scene', 'same_task', 'same_family')] == ['scene', 'task', 'family']


def test_read_layers_family_opt_in_and_public_catalogue(tmp_path):
    svc = service(tmp_path)
    req = payload(context=context(task_id='pick_place_cup'))
    svc.create_provisional_note(**req)
    app = create_app(svc.store.path, api_key=API, importer_key=IMPORTER)
    for changes, level, layer in (({}, 'same_scene', 'scene'), ({'seed': 103}, 'same_task', 'task'),
                                  ({'task_id': 'pick_place_plate', 'task_prompt': 'Other'}, 'same_family', 'family')):
        ctx = {**req['context'], **changes}
        if layer == 'family':
            assert svc.retrieve(ctx, now=time.time()) == []
            assert svc.retrieve_levels(ctx, now=time.time(), levels=('same_scene', 'same_task')) == []
            assert sync_request(app, '/retrieve', method='POST', key=API,
                                payload={'context': ctx, 'now': time.time()}) == (200, [])
        code, rows = sync_request(app, '/retrieve', method='POST', key=API,
                                 payload={'context': ctx, 'now': time.time(), 'match_levels': [level]})
        assert code == 200 and len(rows) == 1
        assert (rows[0]['layer'], rows[0]['source_task_id'], rows[0]['trusted_here']) == (layer, 'pick_place_cup', False)
        assert _memory(dict(rows[0], future_key='ignored')).memory_id == req['memory_id']
        grant = svc.authorize_dev_use(memory_id=req['memory_id'], memory_version=1, development_id=layer,
                    context=ctx, source_policy_sha256='a'*64, now=time.time())
        assert (grant['layer'], grant['source_task_id']) == (layer, 'pick_place_cup')
        assert grant['excluded_from_results'] and not grant['trusted_here']
        row, = svc.list_dev_v2_memories(context=ctx)
        assert row['layer'] == 'scene' and row['read_layer'] == layer
    row, = svc.list_dev_v2_memories()
    assert row['layer'] == 'scene' and 'read_layer' not in row
    assert row['origin_summary'] == {k: req['origin'][k] for k in ('experiment_id', 'help_version', 'policy_id')}
    assert row['from_run_note'] is None and row['from_provisional'] is None
    assert not {'guidance', 'artifact', 'origin', 'digests'} & row.keys()


def test_run_note_link_route_and_old_fingerprint(tmp_path):
    svc = service(tmp_path)
    req = payload()
    old_formula = {**req, 'ttl_days': 14, 'summary': None, 'group': None}
    svc.create_provisional_note(**req)
    assert svc.v2.provenance(req['memory_id'])['request_sha256'] == digest(encoded(old_formula).encode())
    assert svc.create_provisional_note(**req) == svc.create_provisional_note(**req)
    req = payload(memory_id='p-linked')
    req['from_run_note'] = link(req)
    app = create_app(svc.store.path, api_key=API, importer_key=IMPORTER)
    code, _ = sync_request(app, '/dev-v2/provisional-notes', method='POST', key=API, importer_key=IMPORTER, payload=req)
    assert code == 200
    assert svc.v2.provenance('p-linked')['from_run_note'] == req['from_run_note']
    assert svc.v2.provenance('p-linked')['layer'] == 'scene'
    req['from_run_note']['version'] = True
    assert sync_request(app, '/dev-v2/provisional-notes', method='POST', key=API, importer_key=IMPORTER, payload=req)[0] == 422


@pytest.mark.parametrize('change', [lambda v: v.pop('cell'), lambda v: v.update(extra='x'),
    lambda v: v.update(experiment_id='other'), lambda v: v.update(cell='other'),
    lambda v: v.update(version=3), lambda v: v.update(version=True), lambda v: v.update(run_note_id='r-wrong')])
def test_malformed_links(tmp_path, change):
    req = payload(); req['from_run_note'] = link(req); change(req['from_run_note'])
    with pytest.raises(ValueError, match='from_run_note'):
        service(tmp_path).create_provisional_note(**req)


def test_candidate_link_and_absent_provenance_unchanged(source_setup):
    _, svc, source, original = candidate(source_setup)
    before = encoded(svc.v2.provenance('m1'))
    svc.create_dev_v2_candidate(memory_id='m1', source_import_id='source', created_at=1.0, from_provisional=None)
    assert encoded(svc.v2.provenance('m1')) == before
    assert 'from_provisional' not in svc.v2.provenance('m1') and 'layer' not in svc.v2.provenance('m1')
    pid = 'p-' + 'a'*24
    svc.create_provisional_note(**payload(memory_id=pid, context=context(task_id=source['context']['task_id'], seed=source['source_seed'])))
    svc.create_dev_v2_candidate(memory_id='linked', source_import_id='source', created_at=1.0, from_provisional=pid)
    assert svc.v2.provenance('linked')['from_provisional'] == pid
    assert svc.get_dev_v2_memory_public('linked')['from_provisional'] == pid
    for bad in ('m1', 'p-'+'b'*24, 'p-bad'):
        with pytest.raises((StateConflictError, ValueError)):
            svc.create_dev_v2_candidate(memory_id='bad', source_import_id='source', created_at=1.0, from_provisional=bad)
    for n, changes in enumerate(({'task_id':'other'}, {'seed':999})):
        other = 'p-' + str(n)*24
        svc.create_provisional_note(**payload(memory_id=other, context=context(task_id=source['context']['task_id'], seed=source['source_seed']) | changes))
        with pytest.raises((StateConflictError, ValueError)):
            svc.create_dev_v2_candidate(memory_id='bad', source_import_id='source', created_at=1.0, from_provisional=other)


def test_seeded_family_and_trusted_candidate_evidence_guard(tmp_path, monkeypatch):
    from attention_memory_service import families
    from tests.test_test_seeded_notes import payload as seeded_payload
    from tests.test_single_scene import setup_scene, import_pairs
    seeded = service(tmp_path)
    req = seeded_payload();req['context']['task_id']='pick_place_cup'
    seeded.create_test_seeded_note(**req)
    other = dict(req['context'],task_id='pick_place_plate',task_prompt='Other task')
    assert seeded.retrieve(other,now=time.time()) == []
    assert [(m.memory_id,level) for m,level in seeded.retrieve_levels(other,now=time.time(),levels=('same_family',))] == [('seed-1','same_family')]
    grant=seeded.authorize_dev_use(memory_id='seed-1',development_id='family',context=other,source_policy_sha256='a'*64,now=time.time())
    assert grant['layer']=='family' and grant['trusted_here'] is False
    # Controlled evidence fixtures use the task name "pick". Extend only this
    # test's family mapping so the real verification path is exercised intact.
    monkeypatch.setattr(families,'FAMILIES',{'pick_and_place':('pick','pick_place_*')})
    location=tmp_path/'validated';location.mkdir()
    fixture,svc,plan=setup_scene(location,interface=True)
    svc.record_plan('m1',plan);import_pairs(fixture,svc);svc.evaluate('m1')
    ctx=dict(fixture.records['failed']['context'],seed=101,task_id='pick_place_plate',task_prompt='Another task')
    assert svc.retrieve(ctx,now=time.time())==[]
    assert [(m.memory_id,l) for m,l in svc.retrieve_levels(ctx,now=time.time(),levels=('same_family',))]==[('m1','same_family')]
    grant=svc.authorize_dev_use(memory_id='m1',memory_version=1,development_id='family',context=ctx,source_policy_sha256='a'*64,now=time.time())
    assert grant['layer']=='family' and grant['source_task_id']=='pick' and not grant['trusted_here']
    with pytest.raises(StateConflictError,match='excluded'):
        svc.record_dev_result(grant_id=grant['grant_id'],run_id='r',result_policy_sha256='a'*64,artifact_uri='x',artifact_sha256='b'*64)
    original_plan=svc.v2.get_plan
    monkeypatch.setattr(svc.v2,'get_plan',lambda _:None)
    assert svc.retrieve_levels(ctx,now=time.time(),levels=('same_family',))==[]
    monkeypatch.setattr(svc.v2,'get_plan',original_plan)
    # Removing a sealed source artifact invalidates the family reference too.
    Path(fixture.records['failed']['descriptor']['config']['uri']).unlink()
    with pytest.raises((StateConflictError,ValueError,FileNotFoundError)):
        svc.retrieve_levels(ctx,now=time.time(),levels=('same_family',))


def test_old_grant_layer_is_derived_without_rewriting(tmp_path):
    import json
    import sqlite3
    svc=service(tmp_path);req=payload();svc.create_provisional_note(**req)
    kwargs=dict(memory_id=req['memory_id'],memory_version=1,development_id='old',context=req['context'],
                source_policy_sha256='a'*64,now=time.time())
    grant=svc.authorize_dev_use(**kwargs)
    grant.pop('layer');grant.pop('source_task_id')
    raw=encoded(grant)
    with sqlite3.connect(svc.store.path) as db:
        db.execute('UPDATE memory_dev_exposures SET payload=? WHERE grant_id=?',(raw,grant['grant_id']))
    assert svc.authorize_dev_use(**kwargs)['layer']=='scene'
    assert svc.get_dev_use(grant['grant_id'])['source_task_id']==req['context']['task_id']
    with sqlite3.connect(svc.store.path) as db:
        assert db.execute('SELECT payload FROM memory_dev_exposures WHERE grant_id=?',(grant['grant_id'],)).fetchone()[0]==raw
