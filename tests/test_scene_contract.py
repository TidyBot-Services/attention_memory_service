"""Controlled, local scene-contract receipts; no simulator or model calls."""
import copy
from pathlib import Path

import pytest

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.dev_v2 import digest, encoded
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.memory_service_api import create_app
from attention_memory_service.scene_contract import V1, V2, fingerprint, verify_v2_receipt
from tests.dev_v2_fixtures import Fixture
from tests.test_auto_validation import catalog, OPERATOR
from tests.test_provisional_notes import API, IMPORTER
from tests.test_sync_legacy_routes import sync_request


# Literal vectors shared with the producer: preserve int/float, signed zero,
# key ordering and canonical JSON's default ASCII escaping.
@pytest.mark.parametrize('state,raw,normalized', [
    ({}, '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a',
     '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a'),
    ({'z': [1, 1.0, -0.0], 'a': {'x': 1.23456789, 'y': -0.00000049}},
     'e0c942f260b070ae1eef9ea039e623e185c7a06038e2c3d93a50d549c2390ffa',
     '63ae03bffe689f5dd30a90c8c478d2e6471f732536acd826c19a249ce0a1a003'),
    ({'杯': [0.00000149, 0.00000151, -1.23456789]},
     '3b8b83b4fdeb2f9f57271130ff3c4afd540c76aae236c3c53bdafa2d3e790548',
     'bb8827827fadd0eb720834d94e1eb95aa07e68a17ccd31b1b4c6e479df9b75fe'),
    ({'robot': {'quat': [0.0, 0.0, 0.0, 1.0]}, 'objects': {'cup': [1.00000049, 2.0, 3.0]}},
     '3940d36d258185ae8733f0eeb40c2e00e80a165e6b09973460b22aa22f8750ea',
     'b8cacf00454f83a8c454bc50f9a9c4029afbe2e4ba6087398d1b68c9b05a1b7a'),
])
def test_fingerprint_golden_vectors(state, raw, normalized):
    before = copy.deepcopy(state)
    assert fingerprint(state) == raw
    assert fingerprint(state, rounded=True) == normalized
    assert state == before


@pytest.mark.parametrize('value', [True, False, None, '1', (1,), float('nan'), float('inf'), -float('inf')])
def test_fingerprint_refuses_non_numeric_or_nonfinite(value):
    for rounded in (False, True):
        with pytest.raises(ValueError):
            fingerprint({'nested': [value]}, rounded=rounded)


STATE = {'robot': {'quat': [0.0, 0.0, 0.0, 1.0]}, 'objects': {'cup': [1.0, 2.0, 3.0]}}


def receipt(state, ref, seed=101, expected=STATE):
    return dict(contract=ref, task_id='pick', seed=seed, tolerance=1e-6,
                raw_sha256=fingerprint(state), normalized_sha256=fingerprint(state, True),
                matched=fingerprint(state, True) == fingerprint(expected, True))


def setup(tmp_path, *, location='contract', seed=101, original=None, original_patch=None,
          replay=None, replay_patch=None, reference_seed='101', v1=False):
    f = Fixture(tmp_path)
    root = {'executor': f.root, 'validator': f.validator}.get(location, tmp_path / 'contracts')
    contract = {'schema': V1 if v1 else V2, 'tolerance': 1e-6, 'tasks': {'pick':
        {'state': STATE} if v1 else {'seeds': {reference_seed: {
            'normalized_sha256': fingerprint(STATE, True), 'raw_sha256': fingerprint(STATE),
            'resets': 2, 'fresh_processes': 2}}}}}
    ref = f.write(root / 'contract.json', contract)
    original = copy.deepcopy(STATE if original is None else original)
    validation = receipt(original, ref, seed)
    replays = [{'state': copy.deepcopy(STATE), 'validation': receipt(STATE, ref, seed)} for _ in range(2)]
    if replay is not None:
        index, state = replay
        replays[index] = {'state': state, 'validation': receipt(state, ref, seed)}
    if replay_patch is not None:
        index, patch = replay_patch
        replays[index]['validation'].update(patch)
    if v1:
        # v1 does not require a seed/normalized hash and keeps bare replay states.
        validation = dict(contract=ref, task_id='pick', tolerance=1e-6, matched=True,
                          raw_sha256=digest(encoded(original).encode()))
        replays = [entry['state'] for entry in replays]
    validation.update(original_patch or {})
    row = f.make('episode', seed=seed, scene=dict(contract=ref, state=original,
                  validation=validation, replays=replays))
    catalogs = f.seal()
    if location == 'contract':
        catalogs['contract'] = dict(root=str(root), role='contract', context=row['context'],
                                    files={'contract.json': ref['sha256']})
    return f, catalogs, ref, contract


def service_for(tmp_path, catalogs):
    return MemoryService(tmp_path / 'store.sqlite', dev_v2_catalogs=catalogs)


def import_execution(service):
    return service.import_dev_v2_execution(import_id='episode', catalog_id='episode',
                                            descriptor='episode/execution.json')


def validate(service):
    import_execution(service)
    return service.dev_v2.verdict('episode', 'validator-episode', 'episode/verdict.json')


@pytest.mark.parametrize('location', ['executor', 'contract'])
def test_v2_original_and_both_replays_match(tmp_path, location):
    # Raw differs, rounded state matches: the reference raw hash is not the gate.
    state = copy.deepcopy(STATE)
    state['objects']['cup'][0] += 0.0000004
    _, catalogs, _, _ = setup(tmp_path, location=location, original=state, replay=(1, state))
    assert validate(service_for(tmp_path, catalogs))[5]['checks']['scene_reproduction']


@pytest.mark.parametrize('case', ['sealed', 'unlisted', 'different_digest'])
def test_scene_contract_in_source_catalog(tmp_path, case):
    f, catalogs, ref, contract = setup(tmp_path)
    source = catalogs.pop('contract')
    source['role'] = 'source'
    sdk = f.write(Path(source['root']) / 'sdk.py', '# controlled test SDK\n')
    source['files']['sdk.py'] = sdk['sha256']
    catalogs['source'] = source
    if case == 'unlisted':
        source['files'].pop('contract.json')
    elif case == 'different_digest':
        path = Path(ref['uri'])
        path.write_bytes(path.read_bytes() + b'\n')
        source['files']['contract.json'] = digest(path.read_bytes())
    # Registration succeeds even for the negative cases: the config reference
    # must independently match a listed file at its exact sealed digest.
    service = service_for(tmp_path, catalogs)
    if case == 'sealed':
        assert service.dev_v2.scene_contract('episode', ref) == contract
        assert validate(service)[5]['checks']['scene_reproduction']
    else:
        with pytest.raises(StateConflictError, match='exact digest'):
            import_execution(service)


@pytest.mark.parametrize('index', ['original', 0, 1])
def test_v2_coordinate_mismatch_cannot_validate(tmp_path, index):
    state = copy.deepcopy(STATE)
    state['objects']['cup'][0] += 0.000002
    args = {'original': state} if index == 'original' else {'replay': (index, state)}
    _, catalogs, _, _ = setup(tmp_path, **args)
    service = service_for(tmp_path, catalogs)
    import_execution(service)  # A truthful mismatch remains historical evidence.
    with pytest.raises(StateConflictError, match='scene'):
        service.dev_v2.verdict('episode', 'validator-episode', 'episode/verdict.json')


@pytest.mark.parametrize('index', ['original', 0, 1])
@pytest.mark.parametrize('field,value', [('normalized_sha256', '0' * 64), ('raw_sha256', '0' * 64),
                                         ('matched', False), ('seed', 102)])
def test_v2_forged_receipt_rejected(tmp_path, index, field, value):
    patch = {field: value}
    args = {'original_patch': patch} if index == 'original' else {'replay_patch': (index, patch)}
    _, catalogs, _, _ = setup(tmp_path, **args)
    with pytest.raises(StateConflictError, match='scene'):
        validate(service_for(tmp_path, catalogs))


@pytest.mark.parametrize('seed', [True, 101.0, '101', None])
def test_v2_non_integer_seed_rejected(tmp_path, seed):
    _, catalogs, _, contract = setup(tmp_path, seed=seed)
    with pytest.raises(StateConflictError):
        validate(service_for(tmp_path, catalogs))
    with pytest.raises(StateConflictError, match='integer'):
        verify_v2_receipt(contract, {'task_id': 'pick', 'seed': seed}, STATE, {})


def test_v2_missing_seed_entry_rejected(tmp_path):
    _, catalogs, _, contract = setup(tmp_path, reference_seed='102')
    with pytest.raises(StateConflictError, match='task/seed'):
        validate(service_for(tmp_path, catalogs))
    with pytest.raises(StateConflictError, match='integer'):
        verify_v2_receipt(contract, {'task_id': 'pick'}, STATE, {})


@pytest.mark.parametrize('location', ['outside', 'validator'])
def test_contract_outside_accepted_catalogs_rejected(tmp_path, location):
    _, catalogs, _, _ = setup(tmp_path, location=location)
    with pytest.raises(StateConflictError, match='registered contract catalogs'):
        validate(service_for(tmp_path, catalogs))


@pytest.mark.parametrize('when', ['startup', 'read'])
def test_contract_changed_digest_rejected(tmp_path, when):
    _, catalogs, ref, _ = setup(tmp_path)
    service = service_for(tmp_path, catalogs) if when == 'read' else None
    Path(ref['uri']).write_text('{}')
    with pytest.raises(StateConflictError, match='digest changed'):
        validate(service or service_for(tmp_path, catalogs))


def test_contract_ref_must_equal_sealed_digest(tmp_path):
    _, catalogs, ref, _ = setup(tmp_path)
    service = service_for(tmp_path, catalogs)
    with pytest.raises(StateConflictError, match='exact digest'):
        service.dev_v2.scene_contract('episode', {**ref, 'sha256': '0' * 64})


@pytest.mark.parametrize('case', ['parent_path', 'absolute_path', 'symlink_escape', 'root_symlink'])
def test_contract_catalog_path_rules(tmp_path, case):
    _, catalogs, ref, _ = setup(tmp_path)
    cat = catalogs['contract']
    root = Path(cat['root'])
    if case == 'parent_path':
        cat['files'] = {'../contracts/contract.json': ref['sha256']}
    elif case == 'absolute_path':
        cat['files'] = {ref['uri']: ref['sha256']}
    elif case == 'symlink_escape':
        outside = tmp_path / 'outside.json'
        outside.write_bytes(Path(ref['uri']).read_bytes())
        (root / 'link.json').symlink_to(outside)
        cat['files']['link.json'] = ref['sha256']
        cat['symlinks'] = {'link.json': '../outside.json'}
    else:
        link = tmp_path / 'linked-root'
        link.symlink_to(root, target_is_directory=True)
        cat['root'] = str(link)
    with pytest.raises(StateConflictError):
        service_for(tmp_path, catalogs)


@pytest.mark.parametrize('role', ['executor', 'validator'])
@pytest.mark.parametrize('order', ['contract_first', 'contract_last'])
@pytest.mark.parametrize('shape', ['equal', 'contract_inside', 'contract_outside'])
def test_contract_catalog_roots_disjoint(tmp_path, role, order, shape):
    outer, inner = tmp_path / 'outer', tmp_path / 'outer/inner'
    contract_root, other_root = ((outer, outer) if shape == 'equal' else
                                (inner, outer) if shape == 'contract_inside' else (outer, inner))
    contract = catalog(contract_root, 'contract')
    other = catalog(other_root, role)
    catalogs = dict(c=contract, o=other) if order == 'contract_first' else dict(o=other, c=contract)
    with pytest.raises(StateConflictError, match='overlap'):
        service_for(tmp_path, catalogs)
    roots = {'executor': [], 'validator': [], 'contract': [str(contract_root)]}
    roots[role] = [str(other_root)]
    with pytest.raises(StateConflictError, match='overlap'):
        MemoryService(tmp_path / 'roots.sqlite', dev_v2_registration_roots=roots)


def test_contract_online_registration_auth_audit_and_allowlist(tmp_path):
    _, catalogs, _, _ = setup(tmp_path)
    contract = catalogs.pop('contract')
    roots = {'executor': [], 'validator': [], 'contract': [contract['root']]}
    app = create_app(tmp_path / 'store.sqlite', api_key=API, importer_key=IMPORTER,
                     operator_key=OPERATOR, dev_v2_catalogs=catalogs, dev_v2_registration_roots=roots)
    value = dict(catalog_id='contract', catalog=contract)
    for credentials in ({'key': API}, {'key': API, 'operator_key': OPERATOR}):
        assert sync_request(app, '/dev-v2/catalogs', method='POST', payload=value, **credentials)[0] == 403
    for _ in range(2):
        status, result = sync_request(app, '/dev-v2/catalogs', method='POST', payload=value,
                                      key=API, importer_key=IMPORTER)
        assert status == 200 and result['role'] == 'contract'
    service = MemoryService(tmp_path / 'store.sqlite')
    assert validate(service)[5]['checks']['scene_reproduction']
    events = [row for row in service.store.events() if row['event_type'] == 'catalog.registered'
              and row['entity_id'] == 'contract']
    assert len(events) == 1 and events[0]['payload']['actor'] == 'importer'
    outside = catalog(tmp_path / 'outside', 'contract')
    assert sync_request(app, '/dev-v2/catalogs', method='POST', key=API, importer_key=IMPORTER,
                        payload=dict(catalog_id='outside', catalog=outside))[0] == 409


@pytest.mark.parametrize('location', ['executor', 'contract'])
def test_v1_comparator_and_receipt_shape_unchanged(tmp_path, location):
    state = copy.deepcopy(STATE)
    state['robot']['quat'] = [-0.0, -0.0, -0.0, -1.0]
    state['objects']['cup'][0] += 0.0000009  # v1 tolerance, not v2 rounded equality
    _, catalogs, _, _ = setup(tmp_path, location=location, original=state, replay=(1, state), v1=True)
    assert validate(service_for(tmp_path, catalogs))[5]['checks']['scene_reproduction']
