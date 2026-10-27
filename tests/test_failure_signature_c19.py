"""C19 public planner reasons, frozen legacy signatures and unchanged decisions."""
import sqlite3
import time

import pytest

from attention_memory_service import dev_v2
from attention_memory_service.failure_signature import derive_signature, matches_frozen_derivation
from attention_memory_service.core.store import StateConflictError

LEGACY = dict(public_error_class='RuntimeError: PlanningRejected', operation='left_arm.move_delta')
REASON = 'ik_joint_margin_below_execution_limit'


def feedback(code=REASON, suffix=''):
    return dict(public_error='RuntimeError: PlanningRejected: cuRoboV2 planning rejected: '+code+suffix,
                action_trace=[dict(source='left_arm', operation='move_delta', status='failed')])


@pytest.mark.parametrize('suffix', ['', '; best evaluated IK candidate has closest joint openarm_left_joint7', ' extra context'])
def test_reason_and_real_control_treatment_shape(suffix):
    control=feedback(suffix=suffix); treatment=feedback('official_ik_non_success')
    signature=derive_signature(control)
    assert signature == dict(LEGACY, reason_code=REASON)
    dev_v2.validate_signature(signature, control)
    assert dev_v2.signature_present(control, signature)
    assert not dev_v2.signature_present(treatment, signature)
    assert dev_v2.signature_present(treatment, LEGACY)
    assert matches_frozen_derivation(control, LEGACY)
    assert not matches_frozen_derivation(treatment, signature)


@pytest.mark.parametrize('code', ['', 'Bad', '_bad', '1bad', 'bad-code', 'bad.code', 'bad:extra', 'bad\n', 'a'*121])
def test_unparseable_reason_retains_two_keys(code):
    assert derive_signature(feedback(code)) == LEGACY


@pytest.mark.parametrize('code', ['a', 'a0_', 'a'*120])
def test_bounded_identifier(code):
    value=feedback(code); signature=derive_signature(value)
    assert signature['reason_code']==code
    dev_v2.validate_signature(signature,value)


@pytest.mark.parametrize('code', [None, 1, '', 'Upper', '_code', 'a-b', 'a '*2, 'a'*121])
def test_invalid_signature_reason(code):
    with pytest.raises((ValueError, StateConflictError)):
        dev_v2.validate_signature(dict(LEGACY, reason_code=code),feedback())


@pytest.mark.parametrize('error', ['RuntimeError: TimeoutError: timed out',
    'PlanningRejected: V2 client stopped, canceled, or deadline expired; result invalidated',
    "PlanningRejected: V2 planning requires this client's explicit complete world upload",
    'PlanningRejected: Invalid or unvalidated V2 result; dispatch forbidden'])
def test_non_planner_code_errors(error):
    value=dict(feedback(),public_error=error)
    assert set(derive_signature(value))=={'public_error_class','operation'}


@pytest.mark.parametrize('change', ['class','operation','reason','missing','class_prefix'])
def test_three_keys_require_exact_match(change):
    value=feedback(); signature=derive_signature(value)
    if change=='class':value['public_error']=value['public_error'].replace('RuntimeError','ValueError')
    if change=='class_prefix':signature['public_error_class']='RuntimeError'
    if change=='operation':value['action_trace'][0]['operation']='move_joints'
    if change=='reason':value=feedback('official_ik_non_success')
    if change=='missing':value['public_error']='RuntimeError: PlanningRejected: rejected'
    assert not dev_v2.signature_present(value,signature)


def legacy_present(feedback,signature):
    # Byte-for-byte behavior of the pre-C19 matcher, including prefix boundaries.
    error_class=signature['public_error_class'];operation=signature['operation']
    error=feedback.get('public_error','')
    if not error.startswith(error_class) or error[len(error_class):len(error_class)+1] not in {'',':',' ','('}:
        return False
    return any(e.get('status')=='failed' and '.'.join(x for x in (e.get('source'),e.get('operation')) if x)==operation
               for e in feedback.get('action_trace',[]))


@pytest.mark.parametrize('ending', ['', ': other', ' more', '(detail)', 'Suffix'])
def test_legacy_prefix_boundaries_unchanged(ending):
    value=dict(feedback(),public_error=LEGACY['public_error_class']+ending)
    assert dev_v2.signature_present(value,LEGACY)==legacy_present(value,LEGACY)


@pytest.mark.parametrize('controls,treatments,ccalls,tcalls,unsafe', [
    ((True,)*5,(False,)*5,(1,)*5,(1,)*5,None),
    ((True,)*3+(False,)*2,(True,)+(False,)*4,(1,)*5,(1,)*5,None),
    ((True,)*2+(False,)*3,(False,)*5,(1,)*5,(1,)*5,None),
    ((True,)*5,(True,)*2+(False,)*3,(1,)*5,(1,)*5,None),
    ((True,)*5,(False,)*5,(1,)*5,(1,)*5,0),
    ((True,)*5,(False,)*5,(2,)*5,(1,)*5,None),
])
def test_existing_fixture_decision_bytes_unchanged(tmp_path,monkeypatch,controls,treatments,ccalls,tcalls,unsafe):
    from tests.test_single_scene import setup_scene,import_pairs
    f,service,plan=setup_scene(tmp_path,controls=controls,treatments=treatments,control_calls=ccalls,treatment_calls=tcalls)
    service.record_plan('m1',plan)
    import_pairs(f,service,unsafe_treatment=unsafe)
    # Evaluate the identical sealed fixtures twice, including evidence rechecks.
    monkeypatch.setattr(time,'time',lambda now=time.time():now)
    with sqlite3.connect(service.store.path) as db, sqlite3.connect(tmp_path/'before.sqlite') as saved:
        db.backup(saved)
    def decision_bytes():
        service.evaluate('m1')
        with sqlite3.connect(service.store.path) as db:
            return db.execute('SELECT payload FROM memory_v2_decisions WHERE memory_id=?',('m1',)).fetchone()[0].encode()
    current=decision_bytes()
    with sqlite3.connect(tmp_path/'before.sqlite') as saved, sqlite3.connect(service.store.path) as db:
        saved.backup(db)
    monkeypatch.setattr(dev_v2,'signature_present',legacy_present)
    assert decision_bytes()==current


def test_existing_control_stays_frozen_new_id_rederives_same_origin(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    from attention_memory_service import positive_controls
    from tests import test_positive_controls as fixture
    from tests.dev_v2_fixtures import Fixture
    write=Fixture.write
    def planner_write(path,value):
        if path.name=='trace.json' and isinstance(value,dict) and value.get('error'):
            value['error']=feedback()['public_error']
        return write(path,value)
    monkeypatch.setattr(Fixture,'write',staticmethod(planner_write))
    monkeypatch.setattr(fixture,'SIGNATURE',LEGACY)
    # Create the old two-key record exactly as the pre-C19 authority did.
    with monkeypatch.context() as old:
        old.setattr(positive_controls,'derive_signature',lambda value:dict(LEGACY))
        f,service,pc,proxy,plan,spec=fixture.setup(tmp_path)
    def frozen_bytes():
        with sqlite3.connect(service.store.path) as db:
            return [db.execute('SELECT payload FROM positive_control_'+table+' WHERE id=?',('m1',)).fetchone()[0].encode()
                    for table in ('records','plans')]
    before=frozen_bytes()
    pc.verify_source_evidence('m1');pc.record_plan('m1',plan)
    assert frozen_bytes()==before
    # Seal a fresh independent verdict under a new catalog, preserving old bytes.
    old_verdict=Path(f.validator/'failed/verdict.json').read_bytes()
    fresh=dict(json.loads(old_verdict),failure_signature=dict(LEGACY,reason_code=REASON))
    ref=f.write(f.validator/'rerun/verdict.json',fresh)
    catalog=dict(f.catalogs['validator-failed'])
    catalog['files']=dict(catalog['files'],**{'rerun/verdict.json':ref['sha256']})
    pc.dev_v2.register('validator-rerun',catalog)
    rerun=dict(spec,control_id='m1-b')
    result=pc.create(spec=rerun,spec_sha256=positive_controls.validate_spec(rerun),
        execution_import_id='failed',catalog_id='validator-rerun',descriptor='rerun/verdict.json')
    assert result['failure_signature']==dict(LEGACY,reason_code=REASON)
    assert result['spec']['origin']==spec['origin'] and result['spec']['guidance']==spec['guidance']
    assert frozen_bytes()==before and Path(f.validator/'failed/verdict.json').read_bytes()==old_verdict
