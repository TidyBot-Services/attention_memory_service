"""Frame exclusions do not weaken non-frame or historical source checks."""
import copy
import json
from pathlib import Path
import shutil
import sqlite3

import pytest
from attention_memory_service.dev_v2 import is_frame_image, digest, encoded
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.core.store import StateConflictError
from tests.dev_v2_fixtures import Fixture


@pytest.mark.parametrize('name, expected', [
    ('live/a.jpg', True), ('frames/a.jpeg', True), ('recording_frames/a.png', True),
    ('recording_views/top/a.jpg', True), ('snapshots/objects/a.png', False),
    ('snapshots/objects/abc', False), ('snapshots/objects/code.py', False),
    ('snapshots/objects/notes.txt', False), ('live/camera_initial.png', False),
    ('frames/camera_final.png', False), ('a.png', False), ('frames/a.json', False)])
def test_rule(name, expected):
    assert is_frame_image(name) is expected


def setup(tmp_path):
    f = Fixture(tmp_path); f.make('failed'); f.add_advice(f.records['failed'])
    catalog = f.seal()['failed']
    frame = Path(catalog['root'])/'live/frame.jpg'; frame.parent.mkdir(); frame.write_bytes(b'RGB')
    catalog['files']['live/frame.jpg'] = digest(b'RGB')
    return MemoryService(tmp_path/'store.sqlite'), catalog, frame


def test_strip_count_idempotence_and_ref(tmp_path):
    service, catalog, frame = setup(tmp_path)
    service.dev_v2.register('frames', catalog)
    assert 'live/frame.jpg' not in service.dev_v2.catalog('frames')['files']
    assert service.dev_v2.catalog('frames')['excluded_frame_images'] == 1
    frame.unlink()
    service.dev_v2.register('frames', catalog)
    with pytest.raises(StateConflictError, match='registered evidence file missing'):
        service.dev_v2.ref('frames', dict(uri=str(frame), sha256=digest(b'RGB')), raw=True)


def test_snapshot_objects_remain_verified_evidence(tmp_path):
    service,catalog,_=setup(tmp_path)
    obj=Path(catalog['root'])/'snapshots/objects/abc'
    obj.parent.mkdir(parents=True);obj.write_bytes(b'code or notes')
    catalog['files']['snapshots/objects/abc']=digest(obj.read_bytes())
    service.dev_v2.register('objects',catalog)
    assert 'snapshots/objects/abc' in service.dev_v2.catalog('objects')['files']
    obj.unlink()
    with pytest.raises(StateConflictError,match='missing'):
        service.dev_v2.register('objects',catalog)


def test_legacy_reregister_and_startup_still_checks_nonframes(tmp_path):
    service, catalog, frame = setup(tmp_path)
    with sqlite3.connect(service.store.path) as db:
        db.execute('insert into dev_v2_catalogs values (?,?)', ('legacy', encoded(catalog)))
    frame.unlink()
    service.dev_v2.register('legacy', catalog)
    MemoryService(service.store.path, dev_v2_catalogs={'legacy': catalog})
    other = next(n for n in catalog['files'] if not is_frame_image(n))
    (Path(catalog['root'])/other).write_bytes(b'changed')
    with pytest.raises(StateConflictError, match='digest changed'):
        service.dev_v2.register('legacy', catalog)


@pytest.mark.parametrize('kind', ['links', 'symlinks'])
def test_frame_links_rejected(tmp_path, kind):
    service, catalog, _ = setup(tmp_path)
    catalog[kind] = {'live/frame.jpg': 'other.jpg'}
    with pytest.raises(StateConflictError, match='frame images cannot'):
        service.dev_v2.register('frames', catalog)


def test_listing_missing_root_is_isolated(tmp_path):
    f = Fixture(tmp_path); f.make('failed'); f.add_advice(f.records['failed'])
    service = MemoryService(tmp_path/'store.sqlite', dev_v2_catalogs=f.seal())
    service.import_dev_v2_execution(import_id='failed', catalog_id='failed', descriptor='failed/execution.json')
    service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')
    service.create_dev_v2_candidate(memory_id='m', source_import_id='source', created_at=1.)
    shutil.rmtree(f.root)
    listing = service.dev_v2_memory_listing()
    assert listing['memories'] == []
    assert listing['unavailable'][0]['memory_id'] == 'm'
    assert listing['unavailable'][0]['reason']


def test_historical_agent_vision_source_never_opens_image(tmp_path, monkeypatch):
    audit = Path('/tmp/frame-cleanup-audit/store.sqlite3')
    if not audit.exists(): pytest.skip('read-only audit copy unavailable')
    dest = tmp_path/'audit.sqlite3'; shutil.copyfile(audit, dest)
    service = MemoryService(dest)
    original = Path.open
    def no_images(path, *args, **kwargs):
        assert path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', no_images)
    source = service.dev_v2.source('origin-ead9d335ffb568c7fd99f79c5439b8df-source')
    assert source['channel'] == 'dev_agent'


def test_validation_queue_isolates_missing_historical_evidence(tmp_path, monkeypatch):
    from tests.test_provisional_notes import payload
    from tests.test_auto_validation import outcome
    service = MemoryService(tmp_path/'queue.sqlite')
    service.create_provisional_note(**payload('lost'))
    outcome(service, 'lost', 0)
    with sqlite3.connect(service.store.path) as db:
        p = json.loads(db.execute('SELECT payload FROM memory_v2_provenance WHERE memory_id=?',('lost',)).fetchone()[0])
        p.update(channel='dev_v2',demoted_at=1,source_import_id='gone')
        db.execute('UPDATE memory_v2_provenance SET payload=? WHERE memory_id=?',(encoded(p),'lost'))
    monkeypatch.setattr(service.dev_v2,'get',lambda key: (_ for _ in ()).throw(StateConflictError('registered evidence file missing')))
    rows = service.validation_queue()
    assert len(rows)==1 and rows[0]['status']=='isolated'
    assert rows[0]['reason']=='registered evidence file missing'


def test_cross_seed_queue_isolates_verification_error(tmp_path, monkeypatch):
    from tests.test_provisional_notes import payload
    service = MemoryService(tmp_path/'cross.sqlite')
    service.create_provisional_note(**payload('lost'))
    with sqlite3.connect(service.store.path) as db:
        m = json.loads(db.execute('SELECT payload FROM memories WHERE id=?',('lost',)).fetchone()[0]);m['status']='trusted'
        db.execute('UPDATE memories SET payload=? WHERE id=?',(encoded(m),'lost'))
        p = json.loads(db.execute('SELECT payload FROM memory_v2_provenance WHERE memory_id=?',('lost',)).fetchone()[0]);p.update(channel='dev_v2',source_import_id='gone')
        db.execute('UPDATE memory_v2_provenance SET payload=? WHERE memory_id=?',(encoded(p),'lost'))
        db.execute('INSERT INTO memory_v2_plans VALUES (?,?)',('lost',encoded(dict(profile='single_scene_v1'))))
        db.execute('INSERT INTO memory_v2_decisions VALUES (?,?)',('lost',encoded(dict(trust_scope='scene_scoped'))))
    monkeypatch.setattr(service.cross_seed,'status',lambda **kw: (_ for _ in ()).throw(StateConflictError('registered evidence file missing')))
    row = service.cross_seed.queue()[0]
    assert row['status']=='isolated' and row['reason']
