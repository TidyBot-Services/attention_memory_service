"""Origin-only agent evidence; registered catalogs and a real in-process service."""
import json
from pathlib import Path

import pytest

from attention_memory_service.core.store import StateConflictError
from benchmarks.attention_harness.tests.test_dev_agent_memory_phase2 import agent_origin, run_origin, register
from benchmarks.attention_harness.memory_validation_provision import read


def test_agent_origin_candidate_untrusted_and_validation_arms_refused(agent_origin):
    f=agent_origin;receipt=run_origin(f)
    resolved,source,candidate=register(f,receipt)
    assert candidate.status.value=='candidate'
    with pytest.raises(StateConflictError,match='cannot record results'):
        f.service.dev_v2._result_evidence({},receipt['registration']['execution_import_id'])
    assert f.service.provenance(candidate.memory_id)['channel']=='dev_agent'
    assert f.service.provenance(f.note['memory_id'])['channel']=='dev_agent'
    with pytest.raises(StateConflictError,match='Dev v2 candidate'):
        f.service.authorize_dev_v2_validation(memory_id=candidate.memory_id,cell='arm',version=1,
            seed=101,context=source['context'],policy_sha256='a'*64,config_sha256='b'*64,exposure='none')
    assert f.service.get_memory(candidate.memory_id).status.value=='candidate'


@pytest.mark.parametrize('change',['channel','purpose','replay','help','source'])
def test_registered_agent_origin_tamper_refused(agent_origin,change):
    f=agent_origin;receipt=run_origin(f)
    catalogs={cid:read(ref) for cid,ref in receipt['catalogs'].items()}
    if change=='channel':
        path=f.root/'v1/bundle.json';value=json.loads(path.read_bytes());value['channel']='dev_v2'
    elif change=='purpose':
        path=Path(next(ref['uri'] for ref in receipt['catalogs'].values() if 'executor' in ref['uri']))
        catalog=json.loads(path.read_bytes());name=next(n for n in catalog['files'] if n.endswith('descriptor.json'))
        path=Path(catalog['root'])/name;value=json.loads(path.read_bytes());value['import_purpose']='memory_validation'
    elif change=='replay':
        v=read(read(receipt['validation'])['descriptor']);path=Path(v['agent_policy_replays'][0]['uri']);value=json.loads(path.read_bytes());value['policy_sha256']='f'*64
    elif change=='help':path=f.root/'v2/agent-help-evidence.json';value={}
    else:path=f.root/'v1/policy.py';value='tampered policy'
    path.chmod(0o644);path.write_text(value if isinstance(value,str) else json.dumps(value))
    f.service.dev_v2.register('scene-contract',dict(role='contract',root=str(Path(f.native.contract['uri']).parent),context=dict(f.context,namespace='offline-agent-origin'),files={'scene.json':f.native.contract['sha256']}))
    with pytest.raises((StateConflictError,ValueError)):
        # Bypass harness receipt reuse so the service itself detects the tamper.
        for cid,c in catalogs.items():f.service.dev_v2.register(cid,c)
        executor=next((cid,c) for cid,c in catalogs.items() if c['role']=='executor' and any('/execution/descriptor.json' in n for n in c['files']))
        cid,c=executor;name=next(n for n in c['files'] if n.endswith('/execution/descriptor.json'))
        f.service.import_dev_v2_execution(import_id=receipt['origin_key'],catalog_id=cid,descriptor=name)
        source=next((cid,c) for cid,c in catalogs.items() if c['role']=='executor' and any('/source/descriptor.json' in n for n in c['files']))
        cid,c=source;name=next(n for n in c['files'] if n.endswith('/source/descriptor.json'))
        f.service.import_dev_v2_source(import_id='tampered',catalog_id=cid,descriptor=name)


def test_forged_agent_bundle_channel_refused_after_reading_registered_evidence(agent_origin,monkeypatch):
    f=agent_origin;receipt=run_origin(f);resolved,_,_=register(f,receipt)
    importer=f.service.dev_v2;args=resolved['failed_execution']
    evidence,d,bundle,_,_=importer.execution(args['import_id'])
    original=importer.ref
    def forged(cid,ref,**kw):
        if ref==evidence['bundle']:return dict(bundle,channel='dev_v2')
        return original(cid,ref,**kw)
    monkeypatch.setattr(importer,'ref',forged)
    with pytest.raises(StateConflictError,match='channel differs'):
        importer.verify_execution(args['catalog_id'],args['descriptor'])
