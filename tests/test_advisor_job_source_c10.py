"""Synthetic sealed Advisor-job sources; no HTTP, provider or simulator."""
import json
import sqlite3
from pathlib import Path

import pytest

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.dev_v2 import digest, encoded
from attention_memory_service.memory_service import MemoryService
from tests.dev_v2_fixtures import Fixture


class AskingFixture(Fixture):
    policy = 'reactive_help'
    features_at = None

    def write(self, path, value):
        if isinstance(value, dict):
            if path.name == 'config.json' and self.features_at == 'config':
                value['planner_features'] = {'approval_rule': 'all'}
            if value.get('schema') == 'attentionbench.dev-experiment.v3':
                value['selection'] = {'policies': [self.policy]}
                for cell in value['cells']:
                    cell['policy_id'] = self.policy
                if self.features_at in ('manifest', 'selection'):
                    target = value if self.features_at == 'manifest' else value['selection']
                    target['planner_features'] = {'memory': {'enabled': True}, 'approval_rule': 'all'}
            if value.get('schema_version') == 'attentionbench.formal-runner-result.v1':
                value['policy_id'] = self.policy
        return super().write(path, value)


def source_fixture(tmp_path, *, fault=None, policy='reactive_help', table=False):
    f = AskingFixture(tmp_path); f.policy = policy
    if fault in ('manifest', 'selection', 'config'): f.features_at = fault
    failed = f.make('failed'); d = f.add_advice(failed)
    ledger = Path(d['ledger']['uri'])
    with sqlite3.connect(ledger) as db:
        if fault != 'existing_request':
            db.execute('DELETE FROM v2_planner_requests' if table else 'DROP TABLE v2_planner_requests')
        if fault == 'cell':
            cell = json.loads(db.execute('SELECT payload FROM v2_cells').fetchone()[0])
            cell['planner_features'] = {'approval_rule': 'all'}
            db.execute('UPDATE v2_cells SET payload=?', (encoded(cell),))
        record = json.loads(db.execute('SELECT payload FROM v2_versions WHERE version=2').fetchone()[0])
        jobs = {k: json.loads(p) for k, p in db.execute("SELECT kind,payload FROM v2_jobs WHERE version=2")}
        advisor, model = jobs['advisor'], jobs['model']
        help_value = json.loads(Path(record['help']['uri']).read_bytes())
        if fault == 'unsettled': db.execute("UPDATE v2_jobs SET state='claimed' WHERE kind='advisor'")
        if fault in ('advisor_calls', 'help_count', 'help_credits'): advisor['actual'][fault] = 2
        if fault == 'dev_calls': model['actual']['dev_calls'] = 0
        if fault == 'late': advisor['settled_at'] = model['claimed_at'] + 1
        if fault == 'claim': advisor['claimed_at'] = advisor['settled_at'] + 1
        if fault == 'content': help_value['content'] += '\n'
        if fault == 'rejected': help_value['rejected'] = True
        if fault == 'rejected_false': help_value['rejected'] = False
        if fault in ('request', 'response'): help_value[fault] = model['receipt'][fault]
        if fault in ('guidance', 'guidance_absent', 'previous', 'feedback'):
            job = model if fault in ('guidance', 'guidance_absent') else advisor
            ref = job['receipt']['request']; request = json.loads(Path(ref['uri']).read_bytes())
            packet = json.loads(request['messages'][-1]['content'])
            if fault == 'guidance': packet['attention_control']['help']['content'] = ''
            if fault == 'guidance_absent': packet['attention_control'].pop('help')
            if fault == 'previous': packet['previous_complete_source'] += '\n'
            if fault == 'feedback': packet['public_feedback'] = {}
            request['messages'][-1]['content'] = encoded(packet)
            job['receipt']['request'] = f.write(Path(ref['uri']), request)
            if fault in ('guidance', 'guidance_absent'): record['model_input'] = job['receipt']['request']
            else: help_value['request'] = job['receipt']['request']
        if fault == 'repair': record['source'] = f.write(f.root/'failed/other.py', 'different source\n')
        record['help'] = f.write(Path(record['help']['uri']), help_value)
        d['version_record'] = f.write(Path(d['version_record']['uri']), record)
        db.execute('UPDATE v2_versions SET payload=? WHERE version=2', (encoded(record),))
        for kind, job in jobs.items():
            db.execute('UPDATE v2_jobs SET payload=? WHERE kind=?', (encoded(job), kind))
    d.pop('request_id'); d['ledger']['sha256'] = digest(ledger.read_bytes())
    f.write(f.root/'failed/source.json', d)
    failed['descriptor']['ledger'] = d['ledger']
    f.write(f.root/'failed/execution.json', failed['descriptor'])
    service = MemoryService(tmp_path/'authority.sqlite', dev_v2_catalogs=f.seal())
    service.import_dev_v2_execution(import_id='failed', catalog_id='failed', descriptor='failed/execution.json')
    return f, service


@pytest.mark.parametrize('policy', ['reactive_help', 'retry_k_then_ask',
    'budget_matched_random_escalation', 'trace_aware_hint_only', 'full_trace_aware_attention_planner'])
@pytest.mark.parametrize('table', [False, True])
def test_asking_source_accepted(tmp_path, policy, table):
    _, service = source_fixture(tmp_path, policy=policy, table=table)
    source = service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')
    assert source['advisor_evidence'] == 'advisor_job' and source['request_id'] is None
    assert service.create_dev_v2_candidate(memory_id='m1', source_import_id='source', created_at=1).memory_id == 'm1'


@pytest.mark.parametrize('fault,reason', [
    ('unsettled', 'settled'), ('advisor_calls', 'cost'), ('help_count', 'cost'),
    ('help_credits', 'cost'), ('dev_calls', 'cost'), ('content', 'guidance'),
    ('request', 'guidance'), ('response', 'guidance'), ('late', 'late'), ('claim', 'claim/settlement'),
    ('guidance', 'generation input'), ('guidance_absent', 'generation input'), ('config', 'request row'), ('manifest', 'request row'), ('selection', 'request row'),
    ('cell', 'request row'), ('existing_request', 'request row'),
    ('rejected', 'guidance'), ('rejected_false', 'rejected'),
    ('previous', 'failed trace'), ('feedback', 'failed trace'), ('repair', 'model output')])
def test_job_evidence_refused(tmp_path, fault, reason):
    _, service = source_fixture(tmp_path, fault=fault)
    with pytest.raises(StateConflictError, match=reason):
        service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')


@pytest.mark.parametrize('policy', ['autonomous', 'demo_first', 'unknown'])
def test_nonasking_policy_refused(tmp_path, policy):
    _, service = source_fixture(tmp_path, policy=policy)
    with pytest.raises(StateConflictError, match='asking policy'):
        service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')


def test_planner_evidence_form_unchanged(tmp_path):
    f = Fixture(tmp_path); f.add_advice(f.make('failed'))
    service = MemoryService(tmp_path/'authority.sqlite', dev_v2_catalogs=f.seal())
    service.import_dev_v2_execution(import_id='failed', catalog_id='failed', descriptor='failed/execution.json')
    source = service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')
    assert source['advisor_evidence'] == 'planner_request' and source['request_id'] == 'help-failed'


def test_pre_c10_planner_import_reverified_without_mutation(tmp_path):
    f = Fixture(tmp_path); f.add_advice(f.make('failed'))
    service = MemoryService(tmp_path/'authority.sqlite', dev_v2_catalogs=f.seal())
    service.import_dev_v2_execution(import_id='failed', catalog_id='failed', descriptor='failed/execution.json')
    source = service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')
    legacy = dict(source); legacy.pop('advisor_evidence')
    with sqlite3.connect(service.store.path) as db:
        db.execute("UPDATE dev_v2_imports SET payload=? WHERE id='source'", (encoded(legacy),))
    assert service.dev_v2.source('source') == source
    assert service.create_dev_v2_candidate(memory_id='m1', source_import_id='source', created_at=1).memory_id == 'm1'
    with sqlite3.connect(service.store.path) as db:
        assert json.loads(db.execute("SELECT payload FROM dev_v2_imports WHERE id='source'").fetchone()[0]) == legacy
        changed = dict(legacy, guidance='different')
        db.execute("UPDATE dev_v2_imports SET payload=? WHERE id='source'", (encoded(changed),))
    with pytest.raises(StateConflictError, match='immutable import'):
        service.dev_v2.source('source')
