"""Sealed synthetic evidence only; no execution, network or model calls."""
import json
import sqlite3
from pathlib import Path

import pytest

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.dev_v2 import digest, encoded
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.motion_permission import ACTION
from tests.dev_v2_fixtures import Fixture

CODE = 'from robot_sdk import left_arm, left_gripper\ndef policy(sdk):\n    left_arm.move_delta([0, 0, 0.001])\n    left_gripper.close()\n'


class MotionFixture(Fixture):
    def __init__(self, root, config=None, bundle=None):
        self.config_changes = {'budget_tier':'dev', 'runner_boundary':'dev_loop', **(config or {})}
        self.bundle_changes = {'runner_boundary':'dev_loop', **(bundle or {})}
        super().__init__(root)

    def write(self, path, value):
        if path.name == 'config.json':value.update(self.config_changes)
        if path.name == 'bundle.json':value.update(self.bundle_changes)
        return super().write(path, value)

    def make(self, *args, **kwargs):
        r = super().make(*args, code=CODE, **kwargs)
        c, d = r['config'], r['descriptor']
        # Derive fixture scope independently from the production verifier.
        c.update(self.config_changes)
        d.update(import_purpose='memory_validation')
        d['permission_scope'].update(action=ACTION, actor=ACTION, execution_target='robocasa365_sim',
            risk='bounded simulated robot motion under existing SDK safety limits',
            scope=dict(operations=['left_arm.move_delta','left_gripper.close'], robot=c['robot'], task_id=c['task_id'], seed=c['seed'], sdk_calls=c['sdk_call_limit'], control_steps=c['horizon'], wall_seconds=c['attempt_wall_seconds']))
        self.save(r)
        return r

    def save(self, r):
        self.write(self.root/r['name']/'execution.json', r['descriptor'])
        r['verdict']['permission_scope'] = r['descriptor']['permission_scope']
        self.write(self.validator/r['name']/'verdict.json', r['verdict'])


def setup(tmp_path, **kwargs):
    f = MotionFixture(tmp_path, **kwargs)
    r = f.make('motion')
    return f, r


def service(f, tmp_path):
    return MemoryService(tmp_path/'authority.sqlite', dev_v2_catalogs=f.seal())


def imported(s):
    return s.import_dev_v2_execution(import_id='motion', catalog_id='motion', descriptor='motion/execution.json')


def test_validation_motion_records_action_actor_and_rechecks(tmp_path):
    f, r = setup(tmp_path); s = service(f, tmp_path)
    value = imported(s)
    assert value['permission'] == r['descriptor']['permission_scope']
    assert value['permission']['actor'] == ACTION
    assert value['import_purpose'] == 'memory_validation'
    assert 'approval_id' not in r['descriptor']
    assert s.dev_v2.execution('motion')[0] == value
    with pytest.raises(StateConflictError, match='validation-only'):
        s.dev_v2._result_evidence({}, 'motion')


@pytest.mark.parametrize('config,bundle', [
    ({'budget_tier':'formal'}, {}), ({'budget_tier':'legacy'}, {}),
    ({'budget_tier':None}, {}), ({'runner_boundary':'formal'}, {}),
    ({'budget_boundary':'formal'}, {}), ({}, {'runner_boundary':'formal'}),
    ({}, {'execution_target':'real_robot'}), ({'execution_target':'real_robot'}, {}),
    ({}, {'suite':'real_robot'}), ({}, {'entry_lock':{'main_matrix':True}}),
])
def test_non_dev_and_non_simulation_refused(tmp_path, config, bundle):
    f, _ = setup(tmp_path, config=config, bundle=bundle)
    with pytest.raises(StateConflictError):imported(service(f, tmp_path))


@pytest.mark.parametrize('field,value', [
    ('actor','authenticated_human'), ('run_id','run:other'), ('cell','other'),
    ('code_version',2), ('code_sha256','0'*64), ('config_sha256','0'*64),
    ('execution_target','real_robot'), ('action_version',2),
    ('scope',{'operations':['sensors']}),
])
def test_scope_and_actor_mismatch_refused(tmp_path, field, value):
    f, r = setup(tmp_path); r['descriptor']['permission_scope'][field] = value; f.save(r)
    with pytest.raises(StateConflictError):imported(service(f, tmp_path))


@pytest.mark.parametrize('purpose', [None,'result','trusted_use'])
def test_ordinary_import_cannot_claim_new_action(tmp_path, purpose):
    f, r = setup(tmp_path)
    r['descriptor']['import_purpose'] = purpose; f.save(r)
    with pytest.raises(StateConflictError):imported(service(f, tmp_path))


def test_new_action_cannot_claim_approval_row(tmp_path):
    f, r = setup(tmp_path); r['descriptor']['approval_id']='made-up'; f.save(r)
    with pytest.raises(StateConflictError):imported(service(f, tmp_path))


@pytest.mark.parametrize('approved', [False,True])
def test_existing_consumed_human_path(tmp_path, approved):
    f, r = setup(tmp_path); d=r['descriptor']; d.pop('import_purpose')
    d['permission_scope'].update(action='execute_bounded_motion_policy'); d['permission_scope'].pop('actor')
    d['approval_id']='approval'
    if approved:
        approval=dict(request_type='approval',state='consumed',binding=d['permission_scope'],consumed_by='motion-execute',created_at=0,responded_at=.25,consumed_at=.5,deadline_epoch=2,response=dict(approved=True,actor='authenticated_human',response_id='response'))
        with sqlite3.connect(d['ledger']['uri']) as db:db.execute('INSERT INTO v2_planner_requests VALUES (?,?,?)',('approval','motion',encoded(approval)))
        d['ledger']['sha256']=digest(Path(d['ledger']['uri']).read_bytes())
    f.save(r); s=service(f,tmp_path)
    if approved:assert imported(s)['kind']=='execution'
    else:
        with pytest.raises(StateConflictError):imported(s)


def test_safety_receipt_limits_unchanged(tmp_path):
    f, r=setup(tmp_path); ref=r['bundle']['artifacts']['sandbox_receipt']
    raw=json.loads(Path(ref['uri']).read_bytes()); raw['sdk_call_budget']['dispatched']=81
    new=f.write(Path(ref['uri']),raw); r['bundle']['artifacts']['sandbox_receipt']=new
    bref=f.write(f.root/'motion/bundle.json',r['bundle'])
    with sqlite3.connect(r['descriptor']['ledger']['uri']) as db:
        job=json.loads(db.execute('SELECT payload FROM v2_jobs').fetchone()[0]); job['receipt']['bundle']=bref
        db.execute('UPDATE v2_jobs SET payload=?',(encoded(job),))
    r['descriptor']['ledger']['sha256']=digest(Path(r['descriptor']['ledger']['uri']).read_bytes()); f.save(r)
    with pytest.raises(StateConflictError,match='frozen bounds'):imported(service(f,tmp_path))


def test_public_result_entrypoint_and_recheck_refuse_validation_evidence(tmp_path):
    f, r=setup(tmp_path); s=service(f,tmp_path); imported(s)
    grant=dict(grant_id='trusted-grant',memory_id='trusted-note',development_id='motion:v1')
    with sqlite3.connect(s.store.path) as db:
        db.execute('INSERT INTO memory_dev_exposures VALUES (?,?,?,?)',('trusted-grant','trusted-note','motion:v1',encoded(grant)))
    with pytest.raises(StateConflictError,match='validation-only'):
        s.record_dev_v2_result(grant_id='trusted-grant',execution_import_id='motion')
    with pytest.raises(StateConflictError,match='validation-only'):
        s.dev_v2.verify_result({**grant,'result':{'execution_import_id':'motion'}})
    with sqlite3.connect(s.store.path) as db:
        assert 'result' not in json.loads(db.execute('SELECT payload FROM memory_dev_exposures').fetchone()[0])


def test_imported_purpose_cannot_be_removed_later(tmp_path):
    f,r=setup(tmp_path);s=service(f,tmp_path);imported(s)
    r['descriptor'].pop('import_purpose');f.save(r)
    with pytest.raises(StateConflictError):s.dev_v2.execution('motion')


@pytest.mark.parametrize('actor', [None,'authenticated_human'])
def test_motion_cannot_downgrade_to_sensor_only(tmp_path, actor):
    f,r=setup(tmp_path);d=r['descriptor'];d.pop('import_purpose')
    d['permission_scope']['action']='sensor_only'
    if actor is None:d['permission_scope'].pop('actor')
    else:d['permission_scope']['actor']=actor
    f.save(r)
    with pytest.raises(StateConflictError,match='sensor-only evidence'):
        imported(service(f,tmp_path))
