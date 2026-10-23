"""External original SDK/model identity remains sealed and complete."""
import json
from pathlib import Path

import pytest

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.dev_v2 import digest
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.memory_service_api import create_app
from tests.dev_v2_fixtures import Fixture
from tests.test_auto_validation import catalog
from tests.test_provisional_notes import API, IMPORTER
from tests.test_sync_legacy_routes import sync_request


def setup(tmp_path):
    f = Fixture(tmp_path)
    lock = json.loads(Path(f.lock['uri']).read_text())
    root = tmp_path / 'originals'
    for old in (f.root / 'openarm365_dev_v2/sdk.py', f.root / 'model.json'):
        new = root / old.relative_to(f.root)
        new.parent.mkdir(parents=True, exist_ok=True)
        new.write_bytes(old.read_bytes())
        sha = lock['files'].pop(str(old))
        lock['files'][str(new)] = sha
        if str(old) in lock['model_sources']:
            lock['model_sources'] = {str(new): sha}
            f.sdk['robot_model_identity'] = lock['model_sources']
    sdk = root / 'openarm365_dev_v2/sdk.py'
    lock['source_root'] = str(root)
    lock['source_sets'] = {str(sdk.parent): [str(sdk)]}
    f.lock = f.write(Path(f.lock['uri']), lock)
    f.sdk_sha = digest(json.dumps(f.sdk, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())
    r = f.make('episode')
    cats = f.seal()
    cats['originals'] = dict(root=str(root), role='source', context=r['context'],
                            files={str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()})
    return f, cats, root


def import_execution(service):
    return service.import_dev_v2_execution(import_id='episode', catalog_id='episode', descriptor='episode/execution.json')


def test_external_source_identity_startup_and_online(tmp_path):
    _, cats, root = setup(tmp_path)
    source = cats.pop('originals')
    service = MemoryService(tmp_path / 'startup.sqlite', dev_v2_catalogs={**cats, 'originals': source})
    assert import_execution(service)['kind'] == 'execution'
    app = create_app(tmp_path / 'online.sqlite', api_key=API, importer_key=IMPORTER, dev_v2_catalogs=cats,
                     dev_v2_registration_roots={'executor': [], 'validator': [], 'source': [str(root)]})
    args = dict(method='POST', payload=dict(catalog_id='originals', catalog=source), key=API)
    assert sync_request(app, '/dev-v2/catalogs', **args)[0] == 403
    assert sync_request(app, '/dev-v2/catalogs', **args, importer_key=IMPORTER)[0] == 200
    assert import_execution(MemoryService(tmp_path / 'online.sqlite'))['kind'] == 'execution'


@pytest.mark.parametrize('fault', ['unregistered', 'wrong_role', 'unsealed', 'digest', 'extra', 'missing', 'symlink', 'startup_digest'])
def test_source_identity_refuses_unsealed_or_changed_files(tmp_path, fault):
    _, cats, root = setup(tmp_path)
    sdk = root / 'openarm365_dev_v2/sdk.py'
    if fault == 'unregistered': cats.pop('originals')
    if fault == 'wrong_role': cats['originals']['role'] = 'contract'
    if fault == 'unsealed': cats['originals']['files'].pop('model.json')
    if fault == 'startup_digest':
        sdk.write_text('changed')
        with pytest.raises(StateConflictError, match='digest'):
            MemoryService(tmp_path / 'store.sqlite', dev_v2_catalogs=cats)
        return
    service = MemoryService(tmp_path / 'store.sqlite', dev_v2_catalogs=cats)
    if fault == 'digest': sdk.write_text('changed')
    if fault == 'extra': (sdk.parent / 'new.py').write_text('# extra implementation')
    if fault == 'missing': sdk.unlink()
    if fault == 'symlink':
        outside = tmp_path / 'outside.py'; outside.write_bytes(sdk.read_bytes())
        sdk.unlink(); sdk.symlink_to(outside)
    with pytest.raises(StateConflictError):
        import_execution(service)


@pytest.mark.parametrize('role', ['executor', 'validator', 'contract'])
@pytest.mark.parametrize('source_first', [False, True])
def test_source_role_roots_disjoint(tmp_path, role, source_first):
    root = tmp_path / 'root'
    source, other = catalog(root, 'source'), catalog(root / 'nested', role)
    cats = {'s': source, 'o': other} if source_first else {'o': other, 's': source}
    with pytest.raises(StateConflictError, match='overlap'):
        MemoryService(tmp_path / 'catalog.sqlite', dev_v2_catalogs=cats)
    roots = {'executor': [], 'validator': [], 'contract': [], 'source': [str(root)]}
    roots[role] = [str(root / 'nested')]
    with pytest.raises(StateConflictError, match='overlap'):
        MemoryService(tmp_path / 'roots.sqlite', dev_v2_registration_roots=roots)
