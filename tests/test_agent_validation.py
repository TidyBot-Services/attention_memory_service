"""Offline agent validation authority, semantic exposure checks and invalid arms."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from attention_memory_service import agent_validation
from attention_memory_service.core.store import StateConflictError
from benchmarks.attention_harness.tests.agent_validation_fakes import fixture,synthetic_creation
from benchmarks.attention_harness.tests.test_dev_agent_memory_phase3 import run_arm
from benchmarks.attention_harness.dev_ledger import DevV2Store
from benchmarks.attention_harness import dev_experiment


def test_agent_generation_semantic_receipt_and_lineage(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch)
    try:
        with synthetic_creation(f.native):
            batch,current,store,result=run_arm(f,'treatment')
            assert result['status']=='passed'
            evidence=f.service.dev_v2
            execution,d,*_=evidence.execution(current['arm_id'])
            record=evidence.ref(execution['catalog_id'],d['version_record'])
            envelope=json.loads((Path(store.cell(current['arm_id'])['artifact_dir'])/'v1/memory-validation-envelope.json').read_bytes())
            grant=evidence.validation_grant(json.loads(Path(envelope['grant']['uri']).read_bytes())['grant_id'])
            candidate=f.service.get_memory(batch['memory_id'])
            check=lambda:agent_validation.generation(evidence,execution,d,grant['binding'],batch['plan'],candidate)
            assert check()['origin_source_sha256']==batch['origin']['sha256']
            read=evidence.ref
            def forged(cid,ref,**kw):
                value=read(cid,ref,**kw)
                if ref==record['dev_memory_exposure']:
                    value=deepcopy(value);value['entries'][0]['memory_version']+=1
                return value
            monkeypatch.setattr(evidence,'ref',forged)
            with pytest.raises(StateConflictError,match='exposure receipt'):check()
    finally:f.runner.pool.shutdown()


@pytest.mark.parametrize('fault',['formal','trial','probe','help','wall','mode','advisor','notes','mixed','excluded'])
def test_service_refuses_non_arm_manifests(tmp_path,monkeypatch,fault):
    f=fixture(tmp_path,monkeypatch)
    try:
        with synthetic_creation(f.native):
            batch=f.provision(f.note,'service-operation-001')
            store=DevV2Store(next(iter(batch['cells'].values()))['ledger'])
            manifest=dev_experiment.verify_manifest(dev_experiment.snapshot(store,batch['experiment_id']))
            cell=store.cell(batch['pairs'][0]['control_cell']);config=next(iter(manifest['configs'].values()))
            if fault in {'formal','trial','probe','help','wall'}:
                key=dict(formal='formal_versions',trial='trial_runs',probe='read_only_probes',help='help',wall='active_wall_seconds')[fault]
                manifest['agent_budget'][key]+=1
            elif fault=='mode':manifest['selection']['mode']='interactive'
            elif fault=='advisor':manifest['selection']['advisor_proxy']['enabled']=True
            elif fault=='notes':cell['run_notes']=True
            elif fault=='mixed':manifest['cells'][1]['seed']+=1
            else:cell['excluded_from_results']=False
            with pytest.raises(StateConflictError):agent_validation.shape(manifest,cell,config)
    finally:f.runner.pool.shutdown()


def test_missing_formal_is_inconclusive_never_a_trial(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch,outcome='inconclusive')
    try:
        with synthetic_creation(f.native):
            batch,current,store,result=run_arm(f)
            assert result['status']=='failed' and result['checks']['native_evaluation'] is False
            decision=f.service.record_dev_v2_failed_check(**result['failed_check'])
            assert decision['decision']=='inconclusive'
            assert f.service.record_dev_v2_failed_check(**result['failed_check'])==decision
            assert f.runner.finish(current)==result
            assert f.service.get_memory(batch['memory_id']).status.value=='candidate'
            assert f.service.get_memory(f.note['memory_id']).status.value=='provisional'
            assert f.service.list_pairs(batch['memory_id'])==[]
            from benchmarks.attention_harness.tests.autovalidate_fakes import outcome
            outcome(f.service,f.note['memory_id'],1,ctx=dict(f.context,namespace='offline-agent-origin'))
            assert [n['memory_id'] for n in f.service.validation_queue()]==[f.note['memory_id']]
    finally:f.runner.pool.shutdown()
