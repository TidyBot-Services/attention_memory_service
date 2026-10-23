"""Synthetic service-only automatic-validation checks; no external services."""
import copy
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from attention_memory_service.auto_validation import rank_validation_queue
from attention_memory_service.core.store import StateConflictError
from attention_memory_service.dev_v2 import digest, encoded
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.memory_service_api import create_app
from tests.test_provisional_notes import API, IMPORTER, payload, context, CHANGED, SAME, sha
from tests.test_single_scene import setup_scene, import_pairs
from tests.test_sync_legacy_routes import sync_request

OPERATOR = 'operator-key-123456789'


def catalog(root, role='executor'):
    root.mkdir(parents=True, exist_ok=True)
    file = root / 'evidence.json'
    file.write_text('{}')
    return dict(role=role, root=str(root), context=context(), files={file.name: digest(file.read_bytes())})


def registration(tmp_path):
    executor, validator = tmp_path / 'executor', tmp_path / 'validator'
    executor.mkdir(); validator.mkdir()
    roots = dict(executor=[str(executor)], validator=[str(validator)])
    app = create_app(tmp_path / 'store.sqlite', api_key=API, operator_key=OPERATOR,
                     importer_key=IMPORTER, dev_v2_registration_roots=roots)
    return app, roots


@pytest.mark.parametrize('role,nested', [('executor', False), ('validator', False), ('executor', True), ('validator', True)])
def test_catalog_registration_roles_nested_idempotence_and_audit(tmp_path, role, nested):
    app, roots = registration(tmp_path)
    root = tmp_path / role / 'nested' if nested else tmp_path / role
    value = dict(catalog_id='snapshot', catalog=catalog(root, role))
    args = dict(method='POST', payload=value, key=API, importer_key=IMPORTER)
    status, result = sync_request(app, '/dev-v2/catalogs', **args)
    assert status == 200 and result == dict(catalog_id='snapshot', role=role, root=str(root))
    assert sync_request(app, '/dev-v2/catalogs', **args) == (status, result)
    changed = copy.deepcopy(value)
    changed['catalog']['context']['namespace'] = 'different'
    assert sync_request(app, '/dev-v2/catalogs', **{**args, 'payload': changed})[0] == 409
    service = MemoryService(tmp_path / 'store.sqlite')
    rows = [r for r in service.store.events() if r['event_type'] == 'catalog.registered']
    assert len(rows) == 1
    assert rows[0]['payload']['actor'] == 'importer'
    assert rows[0]['payload']['recorded_at'] <= time.time()
    assert service.dev_v2.catalog('snapshot') == value['catalog']


@pytest.mark.parametrize('case', ['outside', 'sibling_prefix', 'root_symlink', 'wrong_role', 'tampered',
                                  'missing', 'parent_path', 'absolute_path', 'symlink_escape'])
def test_catalog_registration_refuses_unsafe_inputs(tmp_path, case):
    app, roots = registration(tmp_path)
    root = tmp_path / 'executor'
    if case == 'outside': root = tmp_path / 'outside'
    if case == 'sibling_prefix': root = tmp_path / 'executor2'
    value = catalog(root)
    if case == 'wrong_role': value['role'] = 'validator'
    if case == 'root_symlink':
        outside = tmp_path / 'outside'; catalog(outside)
        link = root / 'link'; link.symlink_to(outside, target_is_directory=True)
        value['root'] = str(link)
    if case == 'tampered': (root / 'evidence.json').write_text('changed')
    if case == 'missing': (root / 'evidence.json').unlink()
    if case == 'parent_path': value['files'] = {'../evidence.json': 'a' * 64}
    if case == 'absolute_path': value['files'] = {str(root / 'evidence.json'): sha('{}')}
    if case == 'symlink_escape':
        outside = tmp_path / 'outside'; outside.write_text('{}')
        (root / 'link').symlink_to(outside)
        value['files']['link'] = sha('{}')
        value['symlinks'] = {'link': '../outside'}
    assert sync_request(app, '/dev-v2/catalogs', method='POST', key=API, importer_key=IMPORTER,
                        payload=dict(catalog_id='bad', catalog=value))[0] == 409
    with pytest.raises(StateConflictError):
        MemoryService(tmp_path / 'store.sqlite').dev_v2.catalog('bad')


@pytest.mark.parametrize('headers,expected', [({}, 401), ({'importer_key': IMPORTER}, 401),
    ({'key': API}, 403), ({'key': API, 'operator_key': OPERATOR}, 403)])
def test_catalog_registration_auth(tmp_path, headers, expected):
    app, roots = registration(tmp_path)
    assert sync_request(app, '/dev-v2/catalogs', method='POST',
                        payload=dict(catalog_id='c', catalog=catalog(tmp_path / 'executor')),
                        **headers)[0] == expected


def test_registration_unconfigured_and_startup_digest_checks(tmp_path):
    value = catalog(tmp_path / 'executor')
    app = create_app(tmp_path / 'store.sqlite', api_key=API, importer_key=IMPORTER)
    status, response = sync_request(app, '/dev-v2/catalogs', method='POST', key=API, importer_key=IMPORTER,
                                   payload=dict(catalog_id='c', catalog=value))
    assert status == 403 and response['detail'] == 'registration not configured'
    (tmp_path / 'executor/evidence.json').write_text('tampered')
    with pytest.raises(StateConflictError, match='digest'):
        MemoryService(tmp_path / 'startup.sqlite', dev_v2_catalogs={'c': value})


@pytest.mark.parametrize('which', ['same', 'validator_inside', 'executor_inside'])
def test_registration_roots_must_not_overlap(tmp_path, which):
    outer, inner = tmp_path / 'outer', tmp_path / 'outer/inner'
    inner.mkdir(parents=True)
    a, b = (outer, outer) if which == 'same' else (outer, inner) if which == 'validator_inside' else (inner, outer)
    with pytest.raises(StateConflictError, match='overlap'):
        create_app(tmp_path / 'store.sqlite', api_key=API,
                   dev_v2_registration_roots={'executor': [str(a)], 'validator': [str(b)]})


def outcome(service, mid, index, text=CHANGED, ctx=None, version=None):
    memory = service.store.get_memory(mid)
    grant = service.authorize_dev_use(memory_id=mid, memory_version=version or memory.version,
        development_id=f'{mid}-use{index}:v1', context=ctx or context(),
        source_policy_sha256='c' * 64, now=time.time())
    args = dict(grant_id=grant['grant_id'], development_id=grant['development_id'],
                public_feedback_text=text, public_feedback_sha256=sha(text))
    return service.record_provisional_outcome(**args), args


def test_queue_ranking_is_pure():
    rows = [dict(memory_id=mid, uses=uses, failure_gone=gone, created_at=at)
            for mid, uses, gone, at in [('b', 2, 1, 2), ('a', 2, 1, 1), ('d', 6, 0, 1), ('c', 3, 1, 4)]]
    original = copy.deepcopy(rows)
    assert [r['memory_id'] for r in rank_validation_queue(rows)] == ['c', 'a', 'b', 'd']
    assert rows == original


def test_queue_eligibility_counts_and_auth(tmp_path):
    service = MemoryService(tmp_path / 'store.sqlite')
    for mid in ('p-one', 'p-two', 'p-unused', 'p-off', 'p-expired', 'p-withdrawn', 'p-planned', 'p-decided', 'p-linked'):
        service.create_provisional_note(**payload(mid))
        if mid != 'p-unused': outcome(service, mid, 0)
    outcome(service, 'p-one', 1, SAME)
    outcome(service, 'p-two', 1)
    service.disable('p-off', actor='op', reason='off')
    service.rollback('p-withdrawn', actor='op', reason='withdrawn')
    service.set_expiry('p-expired', expires_at=time.time() - 1, actor='op', reason='expiry')
    service.create_test_seeded_note(actor='op', reason='test', memory_id='test', context=context(), guidance='Test hint')
    with sqlite3.connect(service.store.path) as db:
        db.execute('INSERT INTO memory_v2_plans VALUES (?,?)', ('p-planned', '{}'))
        db.execute('INSERT INTO memory_v2_decisions VALUES (?,?)', ('p-decided', '{}'))
        db.execute('INSERT INTO memory_v2_provenance VALUES (?,?)', ('candidate', encoded({'from_provisional': 'p-linked'})))
    rows = service.validation_queue()
    assert [r['memory_id'] for r in rows] == ['p-two', 'p-one']
    assert [(r['uses'], r['failure_gone']) for r in rows] == [(2, 2), (2, 1)]
    assert rows[0]['source_summary'] == dict(task=context()['task_id'], seed=102,
                                           experiment_id='devexp-fake', cell='cell-1', version=1)
    assert rows[0]['layer'] == 'scene' and rows[0]['version'] == 1
    app = create_app(service.store.path, api_key=API)
    assert sync_request(app, '/dev-v2/validation-queue')[0] == 401
    assert sync_request(app, '/dev-v2/validation-queue', key=API) == (200, {'items': rows})


@pytest.fixture
def promoted(tmp_path):
    f, service, plan = setup_scene(tmp_path)
    service.record_plan('m1', plan)
    import_pairs(f, service)
    service.evaluate('m1')
    return f, service, plan


def scene_feedback(f, failure):
    return (f.root / ('failed' if failure else 't0') / 'feedback.json').read_text()


def test_demotion_five_use_window_retries_restart_and_revalidation(promoted):
    f, service, plan = promoted
    ctx = {**f.records['failed']['context'], 'seed': 101}
    before = service.store.get_memory('m1')
    for i, failure in enumerate((True, True, False, False, True)):
        record, args = outcome(service, 'm1', i, scene_feedback(f, failure), ctx)
        assert record['signature_present'] is failure
        assert service.store.get_memory('m1').status.value == ('provisional' if i == 4 else 'trusted')
    assert service.record_provisional_outcome(**args) == record
    restarted = MemoryService(service.store.path)
    assert restarted.record_provisional_outcome(**args) == record
    note = restarted.store.get_memory('m1')
    assert note.version == before.version + 1
    public = restarted.get_dev_v2_memory_public('m1')
    assert public['trust_scope'] == 'provisional' and public['unvalidated']
    assert public['trust_display'] is None
    assert [m.memory_id for m in restarted.retrieve(ctx, now=time.time())] == ['m1']
    new, _ = outcome(restarted, 'm1', 5, scene_feedback(f, False), ctx)
    assert new['unvalidated'] and new['excluded_from_results']
    with pytest.raises(StateConflictError): restarted.get_dev_use(args['grant_id'])
    with pytest.raises(StateConflictError): restarted.evaluate('m1')
    with pytest.raises(StateConflictError): restarted.promote('m1')
    assert restarted.decision('m1')['decision'] == 'promoted'  # Historical decision retained.
    events = restarted.memory_events()
    assert [r['event'] for r in events] == ['promoted', 'demoted']
    assert events[0]['reason'] == ('Passed: the failure stopped in 5 of 5 runs with the note '
                                    '(it happened in 5 of 5 without)')
    assert events[-1]['reason'] == 'target failure in 3 of last 5 uses'
    raw = [e['payload'] for e in restarted.store.events() if e['event_type'] == 'memory.demoted'][0]
    assert raw['actor'] == 'system:auto-demotion' and len(raw['use_ids']) == 5
    assert restarted.memory_events(after=events[0]['seq']) == events[1:]
    assert restarted.memory_events(after=events[-1]['seq']) == []
    queue = restarted.validation_queue()
    assert [r['memory_id'] for r in queue] == ['m1']
    assert queue[0]['source_summary']['cell'] == 'failed'
    restarted.disable('m1', actor='op', reason='off')
    assert restarted.validation_queue() == []
    assert restarted.get_dev_v2_memory_public('m1')['trust_display'] is None
    assert restarted.enable('m1', actor='op', reason='on').status.value == 'provisional'
    fresh = restarted.create_dev_v2_candidate(memory_id='revalidation', source_import_id='source',
                                              created_at=time.time(), from_provisional='m1')
    assert fresh.status.value == 'candidate'
    assert restarted.validation_queue() == []
    new_plan = {**plan, 'memory_id': 'revalidation',
                'pairs': [dict(pair_id=str(i), control_cell=f'new-c{i}', treatment_cell=f'new-t{i}') for i in range(5)]}
    assert restarted.record_plan('revalidation', new_plan) == new_plan
    # A complete second validation uses fresh grants, imports and pairs; the
    # prior decision cannot be recycled for the demoted note.
    from pathlib import Path
    from attention_memory_service.dev_v2 import public_trace_projection, signature_present, stop_position
    from tests.test_single_scene import SIGNATURE
    for i in range(5):
        for arm in ('c', 't'):
            name = f'new-{arm}{i}'
            r = f.make(name, failure=arm == 'c', completed_calls=1,
                       exposure='none' if arm == 'c' else 'candidate_validation',
                       memory_id=None if arm == 'c' else 'revalidation')
            grant = restarted.authorize_dev_v2_validation(memory_id='revalidation', cell=name, version=1,
                seed=101, context=r['context'], policy_sha256=new_plan['validation_policy_sha256'],
                config_sha256=new_plan['validation_config_sha256'], exposure=r['trial']['exposure'])
            f.bind_validation(r, grant)
            trace = json.loads(Path(r['bundle']['artifacts']['trace']['uri']).read_text())
            r['verdict'].update(failure_signature=SIGNATURE, stop_position=stop_position(trace),
                                signature_present=signature_present(public_trace_projection(trace), SIGNATURE))
            f.write(f.validator / name / 'verdict.json', r['verdict'])
            catalogs = f.seal()
            for cid in (name, 'validator-' + name): restarted.dev_v2.register(cid, catalogs[cid])
            restarted.import_dev_v2_execution(import_id=name, catalog_id=name, descriptor=f'{name}/execution.json')
            restarted.import_dev_v2_trial(import_id='trial-' + name, catalog_id='validator-' + name,
                                          descriptor=f'{name}/trial.json')
        restarted.record_dev_v2_pair(memory_id='revalidation', control_import_id=f'trial-new-c{i}',
                                     treatment_import_id=f'trial-new-t{i}')
    assert restarted.evaluate('revalidation').status.value == 'trusted'
    assert restarted.provenance('revalidation')['from_provisional'] == 'm1'
    assert [e['event'] for e in restarted.memory_events()[-2:]] == ['promoted', 'superseded']
    assert restarted.store.get_memory('m1').status.value == 'disabled'


def test_last_five_current_version_only_and_concurrent_crossing(promoted):
    f, service, _ = promoted
    ctx = {**f.records['failed']['context'], 'seed': 101}
    # Old-version failures must not count toward the current window.
    with sqlite3.connect(service.store.path) as db:
        for i in range(5):
            db.execute('INSERT INTO memory_provisional_outcomes VALUES (?,?,?)', (f'old{i}', 'm1', encoded(dict(
                grant_id=f'old{i}', memory_id='m1', memory_version=0, recorded_at=time.time(), signature_present=True))))
    for i, failure in enumerate((True, False, False, False, False, True, True)):
        outcome(service, 'm1', i, scene_feedback(f, failure), ctx)
    assert service.store.get_memory('m1').status.value == 'trusted'
    grant = service.authorize_dev_use(memory_id='m1', memory_version=1, development_id='last:v1',
        context=ctx, source_policy_sha256='c' * 64, now=time.time())
    args = dict(grant_id=grant['grant_id'], development_id='last:v1',
                public_feedback_text=scene_feedback(f, True), public_feedback_sha256=sha(scene_feedback(f, True)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(lambda _: service.record_provisional_outcome(**args), range(2)))
    assert records[0] == records[1]
    assert [e['event'] for e in service.memory_events()] == ['promoted', 'demoted']


def test_validation_notices_plain_reasons_and_read_only_auth(tmp_path):
    f, service, plan = setup_scene(tmp_path, controls=(True, True, False, False, False))
    service.record_plan('m1', plan); import_pairs(f, service)
    service.evaluate('m1'); service.evaluate('m1')
    events = service.memory_events()
    assert len(events) == 1 and events[0]['event'] == 'rejected'
    assert events[0]['reason'] == 'Rejected: without the note the failure only happened in 2 of 5 runs'
    assert set(events[0]) == {'seq', 'memory_id', 'title', 'event', 'reason', 'time'}
    app = create_app(service.store.path, api_key=API)
    assert sync_request(app, '/dev-v2/memory-events')[0] == 401
    assert sync_request(app, '/dev-v2/memory-events', key=API) == (200, {'events': events})
    with pytest.raises(StateConflictError): service.memory_events(after=-1)


@pytest.mark.parametrize('cell,exposure', [('c0', 'candidate_validation'), ('t0', 'none')])
def test_validation_grants_reject_swapped_arms(tmp_path, cell, exposure):
    f, service, plan = setup_scene(tmp_path)
    service.record_plan('m1', plan)
    r = f.records[cell]
    with pytest.raises(StateConflictError, match='cell/exposure'):
        service.authorize_dev_v2_validation(memory_id='m1', cell=cell, version=1, seed=101,
            context=r['context'], policy_sha256=r['record']['source']['sha256'],
            config_sha256=r['descriptor']['config']['sha256'], exposure=exposure)


def test_sealed_dev_results_feed_demotion(promoted):
    from attention_memory_service.identity import store_id
    f, service, _ = promoted
    for i, failure in enumerate((False, True, True, False, True)):
        name = f'result{i}'
        result = f.make(name, failure=failure, completed_calls=1)
        ctx = {**result['context'], 'seed': 101}
        # The sealed execution context must match the exact grant context.
        result['descriptor']['context'] = ctx
        result['context'] = ctx
        grant = service.authorize_dev_use(memory_id='m1', memory_version=1,
            development_id=f'{name}:v1', context=ctx,
            source_policy_sha256=digest(b'controlled_test previous code\n'), now=time.time())
        f.add_result_model(result, grant, store_id(service.store.path))
        service.dev_v2.register(name, f.seal()[name])
        service.import_dev_v2_execution(import_id=name, catalog_id=name, descriptor=f'{name}/execution.json')
        record = service.record_dev_v2_result(grant_id=grant['grant_id'], execution_import_id=name)
        assert service.record_dev_v2_result(grant_id=grant['grant_id'], execution_import_id=name) == record
    assert service.store.get_memory('m1').status.value == 'provisional'
    assert len(service.memory_events()) == 2
    assert service.validation_queue()[0]['uses'] == 5


def test_pair_rejects_control_grant_for_another_memory(promoted):
    _, service, plan = promoted
    service.create_dev_v2_candidate(memory_id='m2', source_import_id='source', created_at=time.time())
    service.record_plan('m2', {**plan, 'memory_id': 'm2'})
    # Even the empty control arm must be granted for the memory being tested.
    with pytest.raises(StateConflictError, match='arm grant belongs to another memory'):
        service.record_dev_v2_pair(memory_id='m2', control_import_id='trial-c0', treatment_import_id='trial-t0')


@pytest.mark.parametrize('arm,ids', [('c0', ['other']), ('t0', ['m1', 'other']), ('t0', ['other'])])
def test_validation_input_must_be_clean(tmp_path, arm, ids):
    from pathlib import Path
    f, service, plan = setup_scene(tmp_path)
    service.record_plan('m1', plan)
    r = f.records[arm]
    grant = service.authorize_dev_v2_validation(memory_id='m1', cell=arm, version=1, seed=101,
        context=r['context'], policy_sha256=r['record']['source']['sha256'],
        config_sha256=r['descriptor']['config']['sha256'], exposure=r['trial']['exposure'])
    path = Path(r['descriptor']['validation_input']['uri'])
    value = json.loads(path.read_text()); value['memory_ids'] = ids
    r['descriptor']['validation_input'] = f.write(path, value)
    f.bind_validation(r, grant)
    catalogs = f.seal()
    for cid in (arm, 'validator-' + arm): service.dev_v2.register(cid, catalogs[cid])
    service.import_dev_v2_execution(import_id=arm, catalog_id=arm, descriptor=f'{arm}/execution.json')
    with pytest.raises(StateConflictError, match='original validation exposure/config mismatch'):
        service.import_dev_v2_trial(import_id='bad-trial', catalog_id='validator-' + arm, descriptor=f'{arm}/trial.json')


def test_client_routes_and_cursor(monkeypatch):
    from attention_memory_service.memory_service_client import MemoryServiceClient
    client = MemoryServiceClient('http://127.0.0.1:1', api_key=API, importer_key=IMPORTER)
    calls = []
    monkeypatch.setattr(client, '_call', lambda method, path, body=None, **kwargs:
                        calls.append((method, path, body, kwargs)) or {'items': [], 'events': []})
    client.register_dev_v2_catalog(catalog_id='c', catalog={})
    assert client.validation_queue() == client.memory_events(after=12) == []
    assert calls == [('POST', '/dev-v2/catalogs', {'catalog_id': 'c', 'catalog': {}}, {'importer': True}),
                     ('GET', '/dev-v2/validation-queue', None, {}),
                     ('GET', '/dev-v2/memory-events?after=12', None, {})]
    with pytest.raises(ValueError): client.memory_events(after=-1)


def test_registration_cli_loads_roots_without_starting_a_server(tmp_path, monkeypatch):
    import uvicorn
    from attention_memory_service import memory_service_api
    roots = {'executor': ['/configured/executor'], 'validator': ['/configured/validator']}
    path = tmp_path / 'roots.json'; path.write_text(json.dumps(roots))
    captured = {}
    def app(store_path, **kwargs):
        captured.update(kwargs)
        return 'controlled-app'
    monkeypatch.setattr(memory_service_api, 'create_app', app)
    monkeypatch.setattr(uvicorn, 'run', lambda *args, **kwargs: None)
    monkeypatch.setattr('sys.argv', ['service', '--store-path', str(tmp_path / 'unused.sqlite'),
                                   '--dev-v2-registration-roots', str(path)])
    assert memory_service_api.main() == 0
    assert captured['dev_v2_registration_roots'] == roots


def test_demotion_preserves_operator_expiry(promoted):
    f, service, _ = promoted
    ctx = {**f.records['failed']['context'], 'seed': 101}
    grants = [service.authorize_dev_use(memory_id='m1', memory_version=1,
        development_id=f'expiry{i}:v1', context=ctx, source_policy_sha256='c' * 64, now=time.time())
        for i in range(5)]
    service.set_expiry('m1', expires_at=0, actor='op', reason='expire now')
    text = scene_feedback(f, True)
    for grant in grants:
        service.record_provisional_outcome(grant_id=grant['grant_id'], development_id=grant['development_id'],
                                          public_feedback_text=text, public_feedback_sha256=sha(text))
    assert service.store.get_memory('m1').expires_at == 0
    assert service.validation_queue() == []
    assert service.retrieve(ctx, now=time.time()) == []
    assert service.store.get_memory('m1').status.value == 'expired'
    with pytest.raises(StateConflictError): service.enable('m1', actor='op', reason='retry')
