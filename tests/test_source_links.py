"""Pinned source-only link chains; no implicit cross-root traversal."""
import pytest

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.dev_v2 import digest, evidence_operation
from attention_memory_service.memory_service import MemoryService
from tests.test_auto_validation import catalog


def chain(tmp_path, count=2):
    roots = [tmp_path / str(i) for i in range(count + 1)]
    cats = [catalog(root, 'source') for root in roots]
    data = b'controlled interpreter bytes'
    sha = digest(data)
    (roots[-1] / 'python').write_bytes(data)
    cats[-1]['files'] = {'python': sha}
    for i in reversed(range(count)):
        target = roots[i + 1] / 'python'
        (roots[i] / 'python').symlink_to(target)
        cats[i]['files'] = {'python': sha}
        cats[i]['links'] = {'python': dict(target=str(target), target_catalog=f's{i + 1}', target_name='python')}
    return roots, cats, data


def register(service, cats):
    for i in reversed(range(len(cats))): service.dev_v2.register(f's{i}', cats[i])


@pytest.mark.parametrize('count', [2, 8])
def test_source_chain_registered_read_and_online(tmp_path, count):
    roots, cats, data = chain(tmp_path, count)
    service = MemoryService(tmp_path / 'store.sqlite', dev_v2_registration_roots={
        'executor': [], 'validator': [], 'source': [str(r) for r in roots]})
    for i in reversed(range(len(cats))):
        service.register_dev_v2_catalog(catalog_id=f's{i}', catalog=cats[i])
    assert service.dev_v2.read('s0', 'python', raw=True) == data
    restart = MemoryService(service.store.path)
    assert restart.dev_v2.read('s0', 'python', raw=True) == data


def test_same_directory_relative_hop(tmp_path):
    roots, cats, data = chain(tmp_path)
    # Like bin/python -> bin/python3.11: the next hop stays in this catalog.
    (roots[1] / 'python').unlink()
    (roots[1] / 'python3.11').write_bytes(data)
    (roots[1] / 'python').symlink_to('python3.11')
    cats[1]['files']['python3.11'] = digest(data)
    cats[1]['links']['python'] = dict(target=str(roots[1] / 'python3.11'), target_catalog='s1', target_name='python3.11')
    service = MemoryService(tmp_path / 'store.sqlite')
    register(service, cats)
    assert service.dev_v2.read('s0', 'python', raw=True) == data


@pytest.mark.parametrize('fault', ['different_hop', 'missing_catalog', 'wrong_role', 'regular_link',
                                  'outside', 'unsealed_target', 'wrong_digest', 'unlisted_link',
                                  'implicit_hop', 'parent_name', 'absolute_name', 'parent_target', 'cycle', 'too_long'])
def test_source_link_registration_refuses_invalid_chain(tmp_path, fault):
    roots, cats, data = chain(tmp_path, 9 if fault == 'too_long' else 2)
    pin = cats[0]['links']['python']
    if fault == 'different_hop': pin['target'] = str(roots[2] / 'python')
    if fault == 'missing_catalog': pin['target_catalog'] = 'absent'
    if fault == 'wrong_role': cats[2]['role'] = 'contract'
    if fault == 'regular_link':
        (roots[0] / 'python').unlink(); (roots[0] / 'python').write_bytes(data)
    if fault == 'outside': pin['target_name'] = 'other'
    if fault == 'unsealed_target': cats[2]['files'] = {'evidence.json': digest(b'{}')}
    if fault == 'wrong_digest': cats[0]['files']['python'] = '0' * 64
    if fault == 'unlisted_link': cats[0]['files'] = {'evidence.json': digest(b'{}')}
    if fault == 'implicit_hop': cats[1].pop('links')
    if fault in {'parent_name', 'absolute_name'}:
        name = '../0/python' if fault == 'parent_name' else str(roots[0] / 'python')
        cats[0]['files'] = {name: digest(data)}; cats[0]['links'] = {name: pin}
    if fault == 'parent_target': pin['target_name'] = '../1/python'
    if fault == 'cycle':
        (roots[0] / 'python').unlink(); (roots[0] / 'python').symlink_to('python')
        pin.update(target=str(roots[0] / 'python'), target_catalog='s0')
    service = MemoryService(tmp_path / 'store.sqlite')
    with pytest.raises(StateConflictError): register(service, cats)


@pytest.mark.parametrize('fault', ['retarget', 'digest', 'remove_link_pin'])
def test_source_link_changes_after_registration_refused(tmp_path, fault):
    roots, cats, data = chain(tmp_path)
    service = MemoryService(tmp_path / 'store.sqlite')
    register(service, cats)
    if fault == 'retarget':
        other = roots[2] / 'other'; other.write_bytes(data)
        (roots[1] / 'python').unlink(); (roots[1] / 'python').symlink_to(other)
    elif fault == 'digest': (roots[2] / 'python').write_bytes(b'changed')
    else:
        # Removing the symlink itself also invalidates the declaration.
        (roots[0] / 'python').unlink(); (roots[0] / 'python').write_bytes(data)
    with pytest.raises(StateConflictError): service.dev_v2.read('s0', 'python', raw=True)


def test_link_retarget_during_operation_same_final_file_refused(tmp_path):
    roots, cats, data = chain(tmp_path)
    service = MemoryService(tmp_path / 'store.sqlite'); register(service, cats)
    @evidence_operation
    def operation():
        assert service.dev_v2.read('s0', 'python', raw=True) == data
        (roots[0] / 'python').unlink()
        # Same final bytes, but skips a pinned hop.
        (roots[0] / 'python').symlink_to(roots[2] / 'python')
    with pytest.raises(StateConflictError): operation()


@pytest.mark.parametrize('role', ['executor', 'validator', 'contract'])
def test_links_are_source_only(tmp_path, role):
    _, cats, _ = chain(tmp_path)
    service = MemoryService(tmp_path / 'store.sqlite')
    cats[0]['role'] = role
    with pytest.raises(StateConflictError, match='source role'):
        service.dev_v2.register('s0', cats[0])
