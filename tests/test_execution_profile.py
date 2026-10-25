"""C1 execution profiles isolate all note reads and validation authority."""
import sqlite3
import time
from functools import partial
from pathlib import Path

import pytest

from attention_memory_service.applicability import (
    LEVELS, profile_matches, same_scene, same_task, same_family, validate_profile,
)
from attention_memory_service.core.memory import _matches
from attention_memory_service.core.models import MemoryRecord, MemoryStatus
from attention_memory_service.core.store import StateConflictError
from attention_memory_service.dev_v2 import encoded
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.memory_service_api import create_app
from attention_memory_service.memory_v2 import trusted_here
from attention_memory_service.seeded_notes import TUPLE_KEYS
from tests.dev_v2_fixtures import Fixture
from tests.legacy_profile_flow import capture
from tests.test_provisional_notes import API, IMPORTER, context, payload, service
from tests.test_test_seeded_notes import payload as seeded_payload
from tests.test_single_scene import setup_scene, import_pairs
from tests.test_cross_seed import target, batch, finish, MODEL
from tests.test_sync_legacy_routes import sync_request

R3 = 'r3-20261025'
INVALID = ['', 'legacy', 'R3', '-r3', 'r3_20261025', 'a' * 49, 'r3\n',
           ' r3', 'r3 ', 'é', None, 3, True, [], {}]


def profiled(value, profile):
    return dict(value, execution_profile=profile) if profile is not None else dict(value)


@pytest.mark.parametrize('note_profile,run_profile', [(None, None), (None, R3), (R3, None), (R3, R3), (R3, 'r4')])
def test_shared_rule_all_levels_and_task_trust(note_profile, run_profile):
    note = profiled(context(task_id='pick_place_cup'), note_profile)
    ctx = profiled(context(task_id='pick_place_cup'), run_profile)
    expected = note_profile == run_profile
    assert profile_matches(note, ctx) is expected
    assert same_scene(note, ctx, TUPLE_KEYS) is expected
    assert same_task(note, dict(ctx, seed=103), {ctx['task_prompt']}) is expected
    assert same_family(note, dict(ctx, task_id='pick_place_plate')) is expected
    # Non-layered conditions use the same profile guard in either direction.
    assert _matches(profiled({'suite': 's'}, note_profile), profiled({'suite': 's'}, run_profile)) is expected
    memory = MemoryRecord(memory_id='m', version=1, source_trace_id='t', guidance='g',
                          candidate_repair='r', applicability=note, evidence_refs=('e',), created_at=1.0,
                          status=MemoryStatus.TRUSTED)
    for level in ('same_scene', 'same_task'):
        assert trusted_here(memory, level, {'channel': 'dev_v2', 'trust_scope': 'task_scoped'}, ctx) is expected
    assert 'execution_profile' not in TUPLE_KEYS


@pytest.mark.parametrize('kind', ['provisional', 'seeded'])
@pytest.mark.parametrize('note_profile,run_profile', [(None, None), (None, R3), (R3, None), (R3, R3)])
def test_import_retrieval_grants_and_listings(tmp_path, kind, note_profile, run_profile):
    svc = service(tmp_path)
    req = payload() if kind == 'provisional' else seeded_payload()
    req['context'] = profiled(context(task_id='pick_place_cup'), note_profile)
    create = svc.create_provisional_note if kind == 'provisional' else svc.create_test_seeded_note
    create(**req)
    assert svc.store.get_memory(req['memory_id']).applicability == req['context']
    ctx = profiled(context(task_id='pick_place_cup'), run_profile)
    for changes, level in (({}, 'same_scene'), ({'seed': 103}, 'same_task'),
                           ({'task_id': 'pick_place_plate'}, 'same_family')):
        here = dict(ctx, **changes)
        rows = svc.retrieve_levels(here, now=time.time(), levels=LEVELS)
        assert [(m.memory_id, match) for m, match in rows] == (
            [(req['memory_id'], level)] if note_profile == run_profile else [])
        args = dict(memory_id=req['memory_id'], development_id=level, context=here,
                    source_policy_sha256='a' * 64, now=time.time())
        if note_profile == run_profile:
            grant = svc.authorize_dev_use(**args)
            assert grant.get('execution_profile', 'legacy') == (note_profile or 'legacy')
        else:
            with pytest.raises(StateConflictError, match='execution profile mismatch'):
                svc.authorize_dev_use(**args)
    row, = svc.list_dev_v2_memories(context=ctx)
    assert row.get('execution_profile', 'legacy') == (note_profile or 'legacy')
    assert row['applicability_summary'].get('execution_profile', 'legacy') == (note_profile or 'legacy')
    if note_profile != run_profile:
        assert row['match_level'] is None and row['trusted_here'] is False
    if kind == 'provisional':
        from tests.test_auto_validation import outcome
        outcome(svc, req['memory_id'], 0, ctx=req['context'])
        queued, = svc.validation_queue()
        assert queued.get('execution_profile', 'legacy') == (note_profile or 'legacy')


@pytest.mark.parametrize('bad', INVALID)
def test_invalid_profiles_rejected_at_http_boundaries(tmp_path, bad):
    svc = service(tmp_path)
    app = create_app(svc.store.path, api_key=API, importer_key=IMPORTER)
    ctx = dict(context(), execution_profile=bad)
    for path, body in [
        ('/retrieve', dict(context=ctx, now=time.time())),
        ('/dev-v2/provisional-notes', payload(context=ctx)),
        ('/dev-v2/test-seeded-notes', dict(seeded_payload(), context=ctx)),
    ]:
        code, response = sync_request(app, path, method='POST', key=API,
                                      importer_key=IMPORTER, payload=body)
        assert code == 422 and 'execution_profile' in response['detail']
    code, response = sync_request(app, '/memories/missing/plan', method='PUT', key=API,
                                  payload=dict(execution_profile=bad))
    assert code == 422 and 'execution_profile' in response['detail']
    with pytest.raises(ValueError, match='execution_profile'):
        svc.create_candidate(memory_id='bad', request_id='missing', guidance='g', repair='r',
                             applicability={'execution_profile': bad}, created_at=1.0)
    with pytest.raises(ValueError, match='execution_profile'):
        svc.cross_seed.authorize(memory_id='missing', target_seed=114, cell='c', version=1,
            seed=114, context=ctx, policy_sha256='a' * 64, config_sha256='b' * 64, exposure='none')
    with pytest.raises(ValueError, match='execution_profile'):
        svc.create_test_seeded_note(**dict(seeded_payload(), context=ctx))
    with pytest.raises(ValueError, match='execution_profile'):
        svc.dev_v2.register('bad', {'context': ctx, 'root': str(tmp_path.resolve()), 'role': 'executor'})


@pytest.mark.parametrize('value', ['a', '0', 'a' * 48, R3, 'r3-'])
def test_valid_profile_syntax(value):
    validate_profile({'execution_profile': value})


@pytest.mark.parametrize('note_profile,config_value', [(R3, None), (None, R3), (R3, 'r4')])
def test_catalog_import_refuses_frozen_config_mismatch(tmp_path, note_profile, config_value):
    f = Fixture(tmp_path, interface=True)
    failed = f.make('failed', execution_profile=note_profile,
           execution_settings=None if config_value is None else {'profile': config_value})
    f.add_advice(failed)
    svc = MemoryService(tmp_path / 'authority.sqlite', dev_v2_catalogs=f.seal())
    app = create_app(svc.store.path, api_key=API, importer_key=IMPORTER)
    code, response = sync_request(app, '/dev-v2/executions', method='POST', key=API,
        importer_key=IMPORTER, payload=dict(import_id='failed', catalog_id='failed', descriptor='failed/execution.json'))
    assert code == 409 and 'execution profile mismatch' in response['detail']
    code, response = sync_request(app, '/dev-v2/sources', method='POST', key=API,
        importer_key=IMPORTER, payload=dict(import_id='source', catalog_id='failed', descriptor='failed/source.json'))
    assert code == 409 and 'execution profile mismatch' in response['detail']
    # Direct importer also proves rejection happens before an import is stored.
    with pytest.raises(StateConflictError, match='execution profile mismatch'):
        svc.import_dev_v2_execution(import_id='failed', catalog_id='failed', descriptor='failed/execution.json')
    with pytest.raises(StateConflictError, match='unknown Dev v2 import'):
        svc.dev_v2.get('failed')


@pytest.mark.parametrize('profile', [None, R3])
def test_validation_plan_and_grant_refuse_different_profile(tmp_path, profile):
    f, svc, plan = setup_scene(tmp_path, interface=True, execution_profile=profile)
    other = R3 if profile is None else None
    bad_plan = {k: v for k, v in plan.items() if k != 'execution_profile'}
    bad_plan = profiled(bad_plan, other)
    with pytest.raises(StateConflictError, match='execution profile mismatch'):
        svc.record_plan('m1', bad_plan)
    assert svc.v2.get_plan('m1') is None
    svc.record_plan('m1', plan)
    r = f.records['c0']
    ctx = {k: v for k, v in r['context'].items() if k != 'execution_profile'}
    with pytest.raises(StateConflictError, match='execution profile mismatch'):
        svc.authorize_dev_v2_validation(memory_id='m1', cell='c0', version=1, seed=101,
            context=profiled(ctx, other), policy_sha256=r['record']['source']['sha256'],
            config_sha256=r['descriptor']['config']['sha256'], exposure='none')


@pytest.mark.parametrize('profile', [None, R3])
def test_cross_seed_probe_and_plan_refuse_different_profile(tmp_path, profile):
    f, svc, plan = setup_scene(tmp_path, interface=True, execution_profile=profile)
    svc.record_plan('m1', plan)
    import_pairs(f, svc)
    svc.evaluate('m1')
    # Import a fully sealed execution with the *other* profile consistently
    # declared in both config and context. This must still not test this note.
    other = R3 if profile is None else None
    make = f.make.func if isinstance(f.make, partial) else f.make
    f.make = partial(make, execution_profile=other,
                     execution_settings=None if other is None else {'profile': other})
    args, _ = target(f, svc, 114)
    with pytest.raises(StateConflictError, match='execution profile mismatch'):
        svc.cross_seed.probe(**args)
    # Simulate a probe left by an older importer. Plan creation rechecks real
    # sealed target evidence rather than trusting the persisted probe row.
    probe = dict(role='transfer', memory_version=1, request=args,
                 scene_contract_ref=args['scene_contract_ref'])
    with sqlite3.connect(svc.store.path) as db:
        db.execute('INSERT INTO memory_cross_seed_probes VALUES (?,?,?)', ('m1', 114, encoded(probe)))
    pairs = [dict(pair_id=str(i), control_cell=f'cx{i}', treatment_cell=f'tx{i}') for i in range(5)]
    with pytest.raises(StateConflictError, match='execution profile mismatch'):
        svc.cross_seed.plan(memory_id='m1', target_seed=114, pairs=pairs, dev_model=MODEL)
    assert svc.cross_seed.get('plans', 'm1', 114) is None


def test_profiled_validation_cross_seed_task_trust_and_read_isolation(tmp_path):
    f, svc, plan = setup_scene(tmp_path, interface=True, execution_profile=R3)
    assert svc.store.get_memory('m1').applicability['execution_profile'] == R3
    svc.record_plan('m1', plan)
    import_pairs(f, svc)
    svc.evaluate('m1')
    queued, = svc.cross_seed.queue()
    assert queued['execution_profile'] == R3
    for seed, failure in ((114, True), (120, False)):
        cross_plan, _ = batch(f, svc, seed, failure=failure)
        assert cross_plan['execution_profile'] == R3
        assert cross_plan['context']['execution_profile'] == R3
        finish(f, svc, cross_plan)
    assert svc.decision('m1')['trust_scope'] == 'task_scoped'
    ctx = dict(f.records['failed']['context'], seed=125)
    grant = svc.authorize_dev_use(memory_id='m1', memory_version=1, development_id='task',
        context=ctx, source_policy_sha256='a' * 64, now=time.time())
    assert grant['trusted_here'] and grant['execution_profile'] == R3
    legacy = {k: v for k, v in ctx.items() if k != 'execution_profile'}
    assert svc.retrieve_levels(legacy, now=time.time(), levels=LEVELS) == []
    row, = svc.list_dev_v2_memories(context=legacy)
    assert row['trusted_here'] is False and row['execution_profile'] == R3


def test_legacy_flow_byte_identical_to_before_c1(tmp_path):
    # Captured from the unchanged base before implementing C1. Includes core
    # import/validation/promotion/retrieval and provisional import/list/retrieval.
    assert capture(tmp_path) == Path(__file__).with_name('legacy_profile_flow.json').read_bytes()
