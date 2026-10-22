"""Importer-attested provisional notes on a disposable store with fake scope."""

import copy
import hashlib
import json
import time

import pytest

from attention_memory_service import provisional_notes
from attention_memory_service.core.store import StateConflictError
from attention_memory_service.core.models import MemoryUseRecord
from attention_memory_service.dev_v2 import SCOPE
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.memory_service_api import ProvisionalNoteInput, create_app
from pydantic import ValidationError
from attention_memory_service.memory_service_client import MemoryServiceClient
from tests.test_sync_legacy_routes import sync_request

API = 'online-key-123456789'
IMPORTER = 'importer-key-123456789'
SIGNATURE = {'public_error_class': 'RuntimeError: PlanningRejected', 'operation': 'left_arm.move_delta'}
BOTH = ('same_scene', 'same_task')


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def feedback(error, events):
    return json.dumps({'schema_version': 'attentionbench.dev-public-feedback.v1',
                       'execution_status': 'failed', 'public_error': error,
                       'action_trace': [{'source': s, 'operation': o, 'status': st} for s, o, st in events]})


ORIGIN = feedback('RuntimeError: PlanningRejected: cuRoboV2 planning rejected: goal',
                  [('sensors', 'get_observation', 'completed'), ('left_arm', 'move_delta', 'failed')])
CHANGED = feedback('RuntimeError: TimeoutError: timed out',
                   [('sensors', 'get_observation', 'completed'), ('right_arm', 'move_delta', 'failed')])
PROGRESSED = feedback('RuntimeError: PlanningRejected: cuRoboV2 planning rejected: goal',
                      [('sensors', 'get_observation', 'completed'), ('base', 'move_delta', 'completed'),
                       ('left_arm', 'move_delta', 'failed')])
SAME = ORIGIN.replace('goal', 'goal again')


def context(**changes):
    value = {key: f'fake-{key}' for key in SCOPE}
    value.update(sdk_sha256='a' * 64, sdk_interface_sha256='9' * 64, perception_mode='sim_gt', seed=102,
                 camera_names=['left', 'right'], task_prompt='Move fake cup',
                 scene_contract_ref={'uri': 'fixture://scene', 'sha256': 'b' * 64})
    value.update(changes)
    return value


def payload(memory_id='p-1', help_text=CHANGED, reason='signature_changed', **changes):
    digests = dict(origin_public_feedback=sha(ORIGIN), origin_source='1' * 64, help='2' * 64,
                   help_model_input='3' * 64, help_source='4' * 64, help_public_feedback=sha(help_text))
    origin = dict(experiment_id='devexp-fake', cell='cell-1', policy_id='reactive_help',
                  origin_version=1, help_version=2, provider='advisor', effective_reason=reason,
                  eval_reason_class='planner_rejected', digests=digests)
    value = dict(actor='importer', memory_id=memory_id, context=context(), guidance='Lift before moving left.',
                 failure_signature=dict(SIGNATURE), origin=origin, origin_public_feedback_text=ORIGIN,
                 help_public_feedback_text=help_text, title='Avoid PlanningRejected on left_arm.move_delta')
    value.update(changes)
    return value


def service(tmp_path):
    return MemoryService(tmp_path / 'memory.sqlite')


def test_create_flags_retrieve_and_idempotency(tmp_path):
    svc = service(tmp_path)
    req = payload()
    created = svc.create_provisional_note(**req)
    assert created['status'] == 'provisional' and created['unvalidated'] is True
    assert created['excluded_from_results'] is True and created['trust_scope'] == 'provisional'
    assert 13.9 * 86400 < created['expires_at'] - time.time() <= 14 * 86400
    assert svc.create_provisional_note(**req) == created
    provenance = svc.v2.provenance('p-1')
    assert (provenance['origin_progress'], provenance['help_progress']) == (1, 1)
    assert provenance['attested_by'] == 'offline_importer' and provenance['channel'] == 'dev_v2_provisional'
    assert any(e['event_type'] == 'memory.provisional_created' for e in svc.store.events())
    for changed in ({'guidance': 'other'}, {'title': 'Other'}, {'context': context(seed=103)}):
        with pytest.raises(StateConflictError, match='different content'):
            svc.create_provisional_note(**{**req, **changed})
    assert [m.memory_id for m in svc.retrieve(req['context'], now=time.time())] == ['p-1']
    for key, value in (('seed', 103), ('task_id', 'x'), ('robot', 'x'), ('camera_names', ['x']),
                       ('scene_contract_ref', {'uri': 'x', 'sha256': 'b' * 64})):
        assert svc.retrieve({**req['context'], key: value}, now=time.time()) == [], key


def test_progressed_and_not_effective(tmp_path):
    svc = service(tmp_path)
    svc.create_provisional_note(**payload('p-2', PROGRESSED, 'progressed'))
    assert svc.v2.provenance('p-2')['help_progress'] == 2
    with pytest.raises(StateConflictError, match='not effective'):
        svc.create_provisional_note(**payload('p-3', SAME, 'progressed'))
    with pytest.raises(StateConflictError, match='disagrees'):
        svc.create_provisional_note(**payload('p-4', CHANGED, 'progressed'))
    assert svc.store.get_memory('p-3') is None


@pytest.mark.parametrize('mutate,error', [
    (lambda r: r['context'].pop('seed'), ValueError),
    (lambda r: r['context'].pop('sdk_interface_sha256'), ValueError),
    (lambda r: r['context'].update(sdk_interface_sha256='bad'), ValueError),
    (lambda r: r['origin']['digests'].update(origin_source='XYZ'), ValueError),
    (lambda r: r['origin']['digests'].pop('help'), ValueError),
    (lambda r: r['origin']['digests'].update(origin_public_feedback='0' * 64), StateConflictError),
    (lambda r: r['origin'].update(help_version=3), ValueError),
    (lambda r: r['origin'].update(provider='human'), ValueError),
    (lambda r: r['origin'].update(extra='x'), ValueError),
    (lambda r: r['origin'].update(cell=''), ValueError),
    (lambda r: r.update(failure_signature={**SIGNATURE, 'operation': 'right_arm.move_delta'}), StateConflictError),
    (lambda r: r.update(guidance='   '), ValueError),
    (lambda r: r.update(guidance='use the native_success flag'), StateConflictError),
    (lambda r: r.update(origin_public_feedback_text='[]',
                        origin={**r['origin'], 'digests': {**r['origin']['digests'],
                                                           'origin_public_feedback': sha('[]')}}), ValueError),
])
def test_creation_validation(tmp_path, mutate, error):
    req = copy.deepcopy(payload())
    mutate(req)
    with pytest.raises(error):
        service(tmp_path).create_provisional_note(**req)


def test_trusted_first_then_not_trusted_here_newest_first(tmp_path):
    svc = service(tmp_path)
    seeded = dict(actor='t', reason='r', memory_id='seed-1', context=context(), guidance='seeded hint')
    svc.create_test_seeded_note(**seeded)
    svc.create_provisional_note(**payload('p-old'))
    svc.create_provisional_note(**payload('p-other', context=context(seed=101, camera_names=['top']),
                                          guidance='other scene hint'))
    svc.create_provisional_note(**payload('p-new', guidance='newer hint'))
    rows = svc.retrieve_levels(context(), now=time.time(), levels=BOTH)
    assert [(m.memory_id, level) for m, level in rows] == [
        ('seed-1', 'same_scene'), ('p-new', 'same_scene'), ('p-other', 'same_task'), ('p-old', 'same_scene')]
    assert [m.memory_id for m in svc.retrieve(context(), now=time.time())] == ['seed-1', 'p-new', 'p-old']


def test_dev_grant_flags_outcomes_and_listing(tmp_path):
    svc = service(tmp_path)
    svc.create_provisional_note(**payload())
    grant = svc.authorize_dev_use(memory_id='p-1', memory_version=1, development_id='cell:v3',
                                  context=context(), source_policy_sha256='c' * 64, now=time.time())
    assert {k: grant[k] for k in ('unvalidated', 'excluded_from_results', 'trust_scope', 'match_level',
                                  'trusted_here', 'source_seed')} == dict(
        unvalidated=True, excluded_from_results=True, trust_scope='provisional',
        match_level='same_scene', trusted_here=False, source_seed=102)
    other = svc.authorize_dev_use(memory_id='p-1', memory_version=1, development_id='other:v1',
                                  context=context(seed=101, camera_names=['top']),
                                  source_policy_sha256='c' * 64, now=time.time())
    assert (other['match_level'], other['unvalidated'], other['trusted_here']) == ('same_task', True, False)
    with pytest.raises(StateConflictError, match='excluded from Dev results'):
        svc.record_dev_result(grant_id=grant['grant_id'], run_id='r', result_policy_sha256='d' * 64,
                              artifact_uri='x', artifact_sha256='e' * 64)
    with pytest.raises(StateConflictError):
        svc.authorize_use(context(), memory_id='p-1', attempt_id='a', now=1)
    with pytest.raises(StateConflictError):
        svc.record_use(MemoryUseRecord('u', 'p-1', 1, 'run-1', 'attempt-1', 3.0, 'failed'))
    outcome = dict(grant_id=grant['grant_id'], development_id='cell:v3',
                   public_feedback_text=CHANGED, public_feedback_sha256=sha(CHANGED))
    record = svc.record_provisional_outcome(**outcome)
    assert (record['signature_present'], record['progress'], record['origin_progress']) == (False, 1, 1)
    assert record['observational'] is True and record['attested_by'] == 'offline_importer'
    assert svc.record_provisional_outcome(**outcome) == record
    with pytest.raises(StateConflictError, match='immutable'):
        svc.record_provisional_outcome(**{**outcome, 'public_feedback_text': SAME,
                                          'public_feedback_sha256': sha(SAME)})
    with pytest.raises(StateConflictError):
        svc.record_provisional_outcome(**{**outcome, 'development_id': 'cell:v4'})
    with pytest.raises(StateConflictError, match='digest'):
        svc.record_provisional_outcome(grant_id=other['grant_id'], development_id='other:v1',
                                       public_feedback_text=SAME, public_feedback_sha256=sha(CHANGED))
    svc.record_provisional_outcome(grant_id=other['grant_id'], development_id='other:v1',
                                   public_feedback_text=SAME, public_feedback_sha256=sha(SAME))
    assert svc.store.get_memory('p-1').status.value == 'provisional'
    row = svc.list_dev_v2_memories(status='provisional')[0]
    assert row['uses'] == {'grants': 2, 'outcomes': 2, 'failure_gone': 1}
    assert (row['status'], row['unvalidated'], row['excluded_from_results'], row['trust_scope']) == (
        'provisional', True, True, 'provisional')
    assert row['expires_at'] > time.time() and row['validation_summary'] is None
    assert row['applicability_summary']['sdk_interface_sha256'] == '9' * 64
    assert 'Lift before' not in json.dumps(row)
    assert svc.list_dev_v2_memories(status='trusted') == []
    ref = svc.list_dev_v2_memories(context=context(seed=101, camera_names=['top']))[0]
    assert (ref['match_level'], ref['trusted_here']) == ('same_task', False)


def test_sdk_interface_matching_and_legacy_rule(tmp_path):
    svc = service(tmp_path)
    svc.create_provisional_note(**payload())
    assert len(svc.retrieve(context(sdk_sha256='f' * 64), now=time.time())) == 1
    assert svc.retrieve(context(sdk_interface_sha256='8' * 64), now=time.time()) == []
    without = context(); without.pop('sdk_interface_sha256')
    assert svc.retrieve(without, now=time.time()) == []
    legacy = dict(actor='t', reason='r', memory_id='seed-legacy', context=without, guidance='legacy hint')
    svc.create_test_seeded_note(**legacy)
    # Legacy notes have no interface identity: exact implementation hash only.
    assert [m.memory_id for m in svc.retrieve(context(), now=time.time())] == ['seed-legacy', 'p-1']
    assert [m.memory_id for m in svc.retrieve(context(sdk_sha256='f' * 64), now=time.time())] == ['p-1']
    with pytest.raises(ValueError):
        svc.retrieve(context(sdk_interface_sha256='nope'), now=time.time())


def test_expiry_kill_switch_and_no_promotion(tmp_path, monkeypatch):
    svc = service(tmp_path)
    svc.create_provisional_note(**payload())
    svc.disable('p-1', actor='op', reason='stop')
    assert svc.retrieve(context(), now=time.time()) == []
    assert svc.enable('p-1', actor='op', reason='resume').status.value == 'provisional'
    for call in (lambda: svc.promote('p-1'), lambda: svc.evaluate('p-1'),
                 lambda: svc.impact_report('p-1'), lambda: svc.record_plan('p-1', {'profile': 'single_scene_v1'}),
                 lambda: svc.create_dev_v2_candidate(memory_id='p-1', source_import_id='x', created_at=1.0),
                 lambda: svc.record_dev_v2_pair(memory_id='p-1', control_import_id='a', treatment_import_id='b'),
                 lambda: svc.authorize_dev_v2_validation(memory_id='p-1', cell='c', version=1, seed=1,
                                                         context=context(), policy_sha256='a' * 64,
                                                         config_sha256='a' * 64, exposure='none'),
                 lambda: svc.export_package('p-1')):
        with pytest.raises(StateConflictError):
            call()
    assert svc.store.get_memory('p-1').status.value == 'provisional'
    real = time.time
    monkeypatch.setattr(provisional_notes.time, 'time', lambda: real() - 15 * 86400)
    svc.create_provisional_note(**payload('p-old'))
    monkeypatch.setattr(provisional_notes.time, 'time', real)
    assert [m.memory_id for m in svc.retrieve(context(), now=time.time())] == ['p-1']
    assert svc.store.get_memory('p-old').status.value == 'expired'
    with pytest.raises(StateConflictError):
        svc.enable('p-old', actor='op', reason='revive')
    svc.set_expiry('p-old', expires_at=time.time() + 3600, actor='op', reason='extend')
    assert svc.enable('p-old', actor='op', reason='revive').status.value == 'provisional'
    svc.rollback('p-1', actor='op', reason='withdraw')
    assert [m.memory_id for m in svc.retrieve(context(), now=time.time())] == ['p-old']
    with pytest.raises(StateConflictError):
        svc.enable('p-1', actor='op', reason='revive')


def test_http_importer_only_and_client(tmp_path, monkeypatch):
    app = create_app(tmp_path / 'memory.sqlite', api_key=API, importer_key=IMPORTER)
    req = payload()
    assert sync_request(app, '/dev-v2/provisional-notes', method='POST', payload=req, key=API)[0] == 403
    status, created = sync_request(app, '/dev-v2/provisional-notes', method='POST', payload=req,
                                   key=API, importer_key=IMPORTER)
    assert status == 200 and created['status'] == 'provisional'
    status, rows = sync_request(app, '/retrieve', method='POST', key=API,
                                payload={'context': req['context'], 'now': 1})
    assert status == 200 and (rows[0]['unvalidated'], rows[0]['trusted_here'], rows[0]['match_level']) == (
        True, False, 'same_scene')
    assert sync_request(app, '/dev-v2/provisional-outcomes', method='POST', key=API,
                        payload=dict(grant_id='x', development_id='d', public_feedback_text='{}',
                                     public_feedback_sha256=sha('{}')))[0] == 403
    with pytest.raises(ValidationError):
        ProvisionalNoteInput.model_validate({**req, 'unexpected': 1})
    calls = []
    client = MemoryServiceClient('http://127.0.0.1:1', api_key=API, importer_key=IMPORTER)
    monkeypatch.setattr(client, '_call', lambda method, path, body=None, importer=False:
                        calls.append((method, path, importer)) or {})
    client.create_provisional_note(**req)
    client.record_provisional_outcome(grant_id='g', development_id='d', public_feedback_text='{}',
                                      public_feedback_sha256=sha('{}'))
    client.retrieve_entries(req['context'], now=1, match_levels=list(BOTH))
    assert calls == [('POST', '/dev-v2/provisional-notes', True),
                     ('POST', '/dev-v2/provisional-outcomes', True), ('POST', '/retrieve', False)]
