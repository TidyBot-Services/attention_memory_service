"""Synthetic source epochs through the real import verifier."""
import copy,json
from pathlib import Path
import pytest
from attention_memory_service.source_policy import dev_tier,drift_record,source_limit
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.core.store import StateConflictError
from tests.dev_v2_fixtures import Fixture

DEV=dict(budget_tier='dev',runner_boundary='dev_loop')

def setup(tmp_path,config,*,record=True):
    f=Fixture(tmp_path);write=f.write
    def configured(path,value):
        if Path(path).name=='config.json':value.update(config)
        return write(path,value)
    f.write=configured
    r=f.make('failed')
    old=json.loads(Path(f.lock['uri']).read_bytes());new=copy.deepcopy(old)
    name=next(iter(old['source_sets'].values()))[0]
    Path(name).write_text('# newer SDK\n')
    from attention_memory_service.dev_v2 import digest
    new['files'][name]=digest(Path(name).read_bytes())
    current=f.write(f.root/'current-lock.json',new)
    r['descriptor']['current_lock']=dict(path=current['uri'],sha256=current['sha256'])
    drift=drift_record(DEV,old,new,f.lock['sha256'],current['sha256'])
    if record:r['descriptor']['source_drift']=drift
    f.write(f.root/'failed/execution.json',r['descriptor'])
    service=MemoryService(tmp_path/'authority.sqlite',dev_v2_catalogs=f.seal())
    return service,f,name,drift


def test_dev_import_records_exact_drift(tmp_path):
    service,f,name,drift=setup(tmp_path,DEV)
    evidence=service.import_dev_v2_execution(import_id='old',catalog_id='failed',descriptor='failed/execution.json')
    assert evidence['source_drift']==drift
    assert drift['files'][0]['path']==name
    Path(name).write_text('# changed after current lock\n')
    with pytest.raises(StateConflictError):service.dev_v2.execution('old')


@pytest.mark.parametrize('config',[{},dict(budget_tier='legacy',runner_boundary='dev_loop'),dict(budget_tier='dev',runner_boundary='formal'),dict(budget_tier='dev',budget_boundary='formal',runner_boundary='dev_loop')])
def test_non_dev_drift_refused(tmp_path,config):
    service,*_=setup(tmp_path,config)
    with pytest.raises(ValueError,match='source evidence digest drift'):
        service.import_dev_v2_execution(import_id='old',catalog_id='failed',descriptor='failed/execution.json')


def test_drift_cannot_be_omitted(tmp_path):
    service,*_=setup(tmp_path,DEV,record=False)
    with pytest.raises(StateConflictError,match='source drift record differs'):
        service.import_dev_v2_execution(import_id='old',catalog_id='failed',descriptor='failed/execution.json')


def test_frozen_size_policy():
    assert source_limit(DEV)==32768
    assert source_limit({})==16384
    assert source_limit(DEV,{'source_size_limit_bytes':16384})==16384


def test_historical_dev_provenance_remains_readable_but_new_import_is_strict(tmp_path):
    f=Fixture(tmp_path);write=f.write
    def configured(path,value):
        if Path(path).name=='config.json':value.update(DEV)
        return write(path,value)
    f.write=configured
    failed=f.make('failed');f.add_advice(failed)
    service=MemoryService(tmp_path/'authority.sqlite',dev_v2_catalogs=f.seal())
    service.import_dev_v2_execution(import_id='failed',catalog_id='failed',descriptor='failed/execution.json')
    source=service.import_dev_v2_source(import_id='source',catalog_id='failed',descriptor='failed/source.json')
    (f.root/'openarm365_dev_v2/sdk.py').write_text('# newer code\n')
    assert service.dev_v2.source('source')==source
    with pytest.raises(StateConflictError):
        service.import_dev_v2_execution(import_id='failed',catalog_id='failed',descriptor='failed/execution.json')
    # Origin policy bytes remain exact even when only historical SDK checks differ.
    Path(failed['record']['source']['uri']).write_text('changed policy')
    with pytest.raises(StateConflictError):service.dev_v2.source('source')
