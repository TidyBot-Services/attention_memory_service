"""C10b sealed generation feedback; all artifacts are synthetic and offline."""
import copy
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.dev_v2 import digest, encoded
from attention_memory_service.feedback_evidence import public_failure, visible_review
from attention_memory_service.memory_service import MemoryService
from tests.dev_v2_fixtures import Fixture
from tests.test_advisor_job_source_c10 import source_fixture

RULE = {'schema': 'attentionbench.public-failure.v1', 'validation_failure': True,
        'execution_statuses': ['failed', 'timeout', 'timed_out'], 'public_errors': True,
        'completion_signal': 'none_registered', 'no_declared_completion_is_failure': True}
REVIEW = {'status': 'completed', 'passed': False, 'reason': 'The cup remained on the counter.',
          'reference_suggestion': 'Move the gripper toward the cup before closing it.'}


def projected_writer(write, fault=None, target='advisor'):
    def wrapped(path, value):
        if isinstance(value, dict):
            if value.get('schema') == 'attentionbench.dev-identity-lock.v2':
                if not any(Path(n).name == 'dev_failure_rules.json' for n in value['files']):
                    ref = write(path.parent/'dev_failure_rules.json',
                                {'schema': 'attentionbench.failure-classification.v1', 'precedence': []})
                    value['files'][ref['uri']] = ref['sha256']
            if value.get('schema') == 'attentionbench.dev-experiment.v3':
                value['public_failure_rule'] = copy.deepcopy(RULE)
            if path.name in ('advisor-input.json', 'model-input.json'):
                packet = json.loads(value['messages'][-1]['content'])
                feedback = packet.setdefault('public_feedback', {})
                # For validation-generation tests, use the failed execution's trace.
                if not feedback:
                    feedback.update(json.loads((path.parent.parent/'failed/feedback.json').read_bytes()))
                feedback.update(eval_review=visible_review(REVIEW, dev=(path.name == 'model-input.json')),
                                **public_failure({'valid': True}, feedback, RULE))
                if path.name.startswith(target):
                    if fault == 'exact':
                        for key in ('eval_review', 'public_failure', 'public_failure_rule'): feedback.pop(key)
                    if fault == 'extra': feedback['unsealed_hint'] = 'extra'
                    if fault == 'reason': feedback['eval_review']['reason'] = 'Changed objective reason'
                    if fault == 'suggestion': feedback['eval_review']['reference_suggestion'] = 'Unsealed suggestion'
                    if fault == 'suggestion_reason': feedback['eval_review']['reason'] = REVIEW['reference_suggestion']
                    if fault == 'classification': feedback['public_failure'] = False
                    if fault == 'numeric_bool': feedback['public_failure'] = 1
                    if fault == 'rule': feedback['public_failure_rule'] = 'other'
                    if fault == 'base': feedback['public_error'] = 'Changed error'
                value['messages'][-1]['content'] = encoded(packet)
        return write(path, value)
    return wrapped


def with_eval(advice, fault=None):
    def wrapped(f, failed, *args, **kwargs):
        d = failed['descriptor']; cell_id = failed['name']
        review = dict(REVIEW)
        if fault == 'incomplete': review['status'] = 'error'
        ref = f.write(f.root/cell_id/'eval-review.json', review)
        job = dict(job_id=cell_id+'-eval', cell=cell_id, version=1, kind='eval',
                   claimed_at=2.0, settled_at=2.5, actual={}, receipt={'eval_review': ref})
        with sqlite3.connect(d['ledger']['uri']) as db:
            cell = json.loads(db.execute('SELECT payload FROM v2_cells WHERE id=?', (cell_id,)).fetchone()[0])
            cell['runtime'] = dict(cell.get('runtime', {}), public_failure_rule=copy.deepcopy(RULE))
            if fault == 'runtime': cell['runtime']['public_failure_rule']['no_declared_completion_is_failure'] = False
            db.execute('UPDATE v2_cells SET payload=? WHERE id=?', (encoded(cell), cell_id))
            if fault != 'missing_eval':
                db.execute('INSERT INTO v2_jobs VALUES (?,?,?,?,?,?)',
                           (job['job_id'], cell_id, 1, 'eval', 'claimed' if fault == 'unsettled_eval' else 'settled', encoded(job)))
        result = advice(f, failed, *args, **kwargs)
        # The existing planner fixture binds its pre-write packet. Rebind to the
        # enriched bytes actually written, preserving the real planner check.
        with sqlite3.connect(d['ledger']['uri']) as db:
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='v2_planner_requests'").fetchone():
                for rid, payload in db.execute('SELECT id,payload FROM v2_planner_requests').fetchall():
                    row = json.loads(payload)
                    request = json.loads((f.root/cell_id/'advisor-input.json').read_bytes())
                    row['binding']['input_sha256'] = digest(json.dumps(request['messages'], sort_keys=True, ensure_ascii=False).encode())
                    db.execute('UPDATE v2_planner_requests SET payload=? WHERE id=?', (encoded(row), rid))
        result['ledger']['sha256'] = d['ledger']['sha256'] = digest(Path(d['ledger']['uri']).read_bytes())
        f.write(f.root/cell_id/'source.json', result)
        f.write(f.root/cell_id/'execution.json', d)
        return result
    return wrapped


def enriched_source(tmp_path, fault=None, target='advisor', planner=False):
    with patch.object(Fixture, 'write', staticmethod(projected_writer(Fixture.write, fault, target))), \
         patch.object(Fixture, 'add_advice', with_eval(Fixture.add_advice, fault)):
        if not planner:
            f, service = source_fixture(tmp_path)
        else:
            f = Fixture(tmp_path); f.add_advice(f.make('failed'))
            service = MemoryService(tmp_path/'authority.sqlite', dev_v2_catalogs=f.seal())
            service.import_dev_v2_execution(import_id='failed', catalog_id='failed', descriptor='failed/execution.json')
    return f, service


@pytest.mark.parametrize('planner', [False, True])
def test_sealed_three_key_source_and_candidate(tmp_path, planner):
    f, service = enriched_source(tmp_path, planner=planner)
    result = service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')
    assert result['feedback_projection'] == 'with_eval_review'
    assert service.dev_v2.source('source') == result
    assert service.create_dev_v2_candidate(memory_id='m1', source_import_id='source', created_at=1)


@pytest.mark.parametrize('target', ['advisor', 'model'])
@pytest.mark.parametrize('fault', ['extra', 'reason', 'suggestion', 'suggestion_reason', 'classification',
                                   'numeric_bool', 'rule', 'base', 'incomplete', 'runtime',
                                   'missing_eval', 'unsettled_eval'])
def test_bad_additions_refused(tmp_path, target, fault):
    _, service = enriched_source(tmp_path, fault, target)
    with pytest.raises(StateConflictError):
        service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')


@pytest.mark.parametrize('artifact', ['eval-review.json', 'dev_failure_rules.json'])
@pytest.mark.parametrize('fault', ['unsealed', 'drift'])
def test_projection_requires_catalog_pins(tmp_path, artifact, fault):
    f, _ = enriched_source(tmp_path)
    catalogs = f.seal()
    name = next(n for n in catalogs['failed']['files'] if Path(n).name == artifact)
    if fault == 'unsealed': catalogs['failed']['files'].pop(name)
    else: (f.root/name).write_text('{}')
    with pytest.raises(StateConflictError):
        service = MemoryService(tmp_path/'different.sqlite', dev_v2_catalogs=catalogs)
        service.import_dev_v2_execution(import_id='failed', catalog_id='failed', descriptor='failed/execution.json')
        service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')


def test_old_exact_import_annotation_is_nonmutating(tmp_path):
    _, service = source_fixture(tmp_path)
    result = service.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')
    assert result['feedback_projection'] == 'exact'
    legacy = {k: v for k, v in result.items() if k != 'feedback_projection'}
    with sqlite3.connect(service.store.path) as db:
        db.execute("UPDATE dev_v2_imports SET payload=? WHERE id='source'", (encoded(legacy),))
    assert service.dev_v2.source('source') == result
    with sqlite3.connect(service.store.path) as db:
        assert json.loads(db.execute("SELECT payload FROM dev_v2_imports WHERE id='source'").fetchone()[0]) == legacy


@pytest.mark.parametrize('fault', [None, 'exact', 'extra', 'reason', 'suggestion', 'classification', 'rule', 'base'])
def test_validation_generation_checks_same_projection(tmp_path, fault):
    from tests.test_arm_generation import setup, prepare, import_trial
    from tests.test_provider_source_c10a import codex_writer
    # The C10a writer supplies complete standalone Codex receipts for this fixture.
    with patch.object(Fixture, 'write', staticmethod(codex_writer(projected_writer(Fixture.write)))), \
         patch.object(Fixture, 'add_advice', with_eval(Fixture.add_advice)):
        f, service, plan = setup(tmp_path)
    from tests import test_arm_generation
    plan['dev_model'] = dict(plan['dev_model'], model='controlled_dev_model')
    service.record_plan('m1', plan)
    with patch.object(test_arm_generation, 'MODEL', plan['dev_model']), patch.object(Fixture, 'write', staticmethod(codex_writer(projected_writer(Fixture.write, fault, 'model')))):
        prepare(f, service, plan, 't0')
    if fault not in (None, 'exact'):
        with pytest.raises(StateConflictError): import_trial(service, 't0')
    else:
        trial = import_trial(service, 't0')
        assert trial['trial']['feedback_projection'] == ('exact' if fault == 'exact' else 'with_eval_review')
