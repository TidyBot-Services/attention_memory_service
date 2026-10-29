"""D7.1 provisional agent evidence; no network or simulator."""
import json
import time
from pathlib import Path

import pytest

from attention_memory_service import provisional_notes
from attention_memory_service.core.store import StateConflictError
from attention_memory_service.memory_service import MemoryService
from benchmarks.attention_harness import memory_auto_import as importer
from benchmarks.attention_harness.tests.test_dev_agent_memory_phase1 import fixture, proposal
from tests.test_provisional_notes import payload


def test_agent_provisional_flags_grants_and_idempotency(proposal, tmp_path):
    _, value = proposal
    request = importer.provisional_payload(value, 'test')
    svc = MemoryService(tmp_path/'service.sqlite')
    created = svc.create_provisional_note(**request)
    assert svc.create_provisional_note(**request) == created
    provenance = svc.v2.provenance(created['memory_id'])
    assert provenance['channel'] == provenance['origin']['channel'] == 'dev_agent'
    assert created['status'] == 'provisional' and created['unvalidated'] and created['excluded_from_results']
    grant = svc.authorize_dev_use(memory_id=created['memory_id'], memory_version=1,
        development_id='cell:v3', context=request['context'], source_policy_sha256='c'*64, now=time.time())
    assert grant['unvalidated'] and grant['excluded_from_results'] and grant['trusted_here'] is False
    assert grant['source_seed'] == 102
    with pytest.raises(StateConflictError):
        svc.record_dev_result(grant_id=grant['grant_id'], run_id='r', result_policy_sha256='d'*64,
                              artifact_uri='x', artifact_sha256='e'*64)


@pytest.mark.parametrize('field', ['request_view', 'advisor', 'help_job', 'delivery', 'help', 'help_execution'])
def test_agent_evidence_digest_refusal(proposal, tmp_path, field):
    _, value = proposal
    request=importer.provisional_payload(value,'test')
    texts=request['agent_evidence']['refs' if field == 'help' else 'help']
    texts[field] += '\n'
    with pytest.raises(StateConflictError, match='digest drift'):
        MemoryService(tmp_path/'svc.sqlite').create_provisional_note(**request)


@pytest.mark.parametrize('change', ['no_channel', 'other_channel', 'onecall_digests', 'no_evidence', 'onecall_with_evidence'])
def test_agent_variant_requires_exact_channel_and_evidence(proposal, tmp_path, change):
    _, value = proposal
    request=importer.provisional_payload(value,'test')
    if change == 'no_channel': request['origin'].pop('channel')
    elif change == 'other_channel': request['origin']['channel']='other'
    elif change == 'onecall_digests':
        request['origin']['digests']['help_model_input']=request['origin']['digests'].pop('agent_help_evidence')
    elif change == 'no_evidence': request.pop('agent_evidence')
    else: request=dict(payload(),agent_evidence=request['agent_evidence'])
    with pytest.raises((ValueError, StateConflictError)):
        MemoryService(tmp_path/'svc.sqlite').create_provisional_note(**request)


def test_onecall_service_golden(tmp_path, monkeypatch):
    golden=json.loads((Path(__file__).parent/'fixtures/agent_memory_phase1_onecall.json').read_text())
    monkeypatch.setattr(provisional_notes.time,'time',lambda: 1700000000.)
    svc=MemoryService(tmp_path/'svc.sqlite')
    request=payload()
    memory=provisional_notes.create(svc,**request)
    assert json.dumps(request,sort_keys=True,ensure_ascii=False) == golden['payload']
    assert json.dumps(memory.artifact(),sort_keys=True,ensure_ascii=False) == golden['memory']
    assert json.dumps(svc.v2.provenance('p-1'),sort_keys=True,ensure_ascii=False) == golden['provenance']
