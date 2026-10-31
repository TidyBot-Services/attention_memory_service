"""Strict execution-help origins, including the historical phase-5 failure."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import types
from types import SimpleNamespace

import pytest

from attention_memory_service import agent_help_interval as service_help, agent_origin
from attention_memory_service.core.store import StateConflictError
from benchmarks.attention_harness import dev_agent_help_interval as harness_help
from benchmarks.attention_harness import memory_independent_validator as validator
from benchmarks.attention_harness.memory_agent_origin import references
from benchmarks.attention_harness.tests import agent_origin_fakes as fakes
from benchmarks.attention_harness.tests.test_dev_agent_memory_phase2 import run_origin, register
from benchmarks.attention_harness.tests.test_memory_provisional import ADVICE

V2 = 'agent-policy-v2'
# Optional replay of a real agent-policy-v2 run directory; skipped when unset.
HISTORICAL = Path(os.environ.get('ATTENTION_MEMORY_HISTORICAL_V2_RUN', '/nonexistent'))


def baseline():
    repo = Path(agent_origin.__file__).resolve().parents[1]
    code = subprocess.check_output(['git', 'show', '9387881:attention_memory_service/agent_origin.py'],
                                   cwd=repo, text=True)
    old = types.ModuleType('attention_memory_service._origin_before_v2_help')
    old.__package__ = 'attention_memory_service'
    exec(compile(code, 'agent_origin.py@9387881', 'exec'), old.__dict__)
    return old


def read(ref, raw=False):
    data = Path(ref['uri']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == ref['sha256']
    return data if raw else json.loads(data)


def v2_help(f, number, *, proactive=False, execution='e000004'):
    answer = dict(ADVICE, status='delivered', delivered=True, cache_hit=False,
                  system_injected=proactive, decision=dict(action='request'),
                  version_id=None, origin_version_id=f'v{number:04d}',
                  execution_id=execution, help_id='help:' + execution, agent_policy_version=V2)
    f.ws.publish(f'executions/{execution}/help.json', answer)
    f.ws.publish(f'versions/v{number:04d}/help.json', harness_help.answer_projection(answer, number + 1))
    f.ws.publish(f'rpc/responses/h{number}.json', dict(op='ask_help', status='completed', result=answer))
    from benchmarks.attention_harness.dev_v2_loop import immutable
    immutable(f.ws.private / f'help/{execution}/request-view.json', dict(question='How to lift?', frames=[]))
    immutable(f.ws.private / f'help/{execution}/advisor.json', dict(status='completed', kind='hint',
        feedback=answer['content'], cache_hit=False, usage=dict(input_tokens=1, output_tokens=1)))
    jid = answer['help_id']
    f.ledger.reserve(jid, 'help', dict(help=1))
    f.ledger.claim(jid)
    f.ledger.settle(jid, answer, usage=dict(help=1))


@pytest.fixture(params=[False, True], ids=['ask_help', 'proactive'])
def synthetic(tmp_path, monkeypatch, request):
    monkeypatch.setattr(fakes, 'help_interval', lambda f, n: v2_help(f, n, proactive=request.param))
    monkeypatch.setattr(validator, 'GpuMeter', lambda: SimpleNamespace(start=lambda: None,
        stop=lambda: dict(gpu_seconds=1., wall_seconds=1., samples=1)))
    return fakes.build(tmp_path, monkeypatch, validation=True, policy_version=V2)


def inputs(f):
    cell = f.store.cell('cell')
    version = next(v for v in cell['versions'] if v['version'] == 2)
    return references(cell, 2), cell, version, read(version['manifest']), f.config


def verify(f, module=agent_origin):
    refs, cell, version, frozen, config = inputs(f)
    with f.store.tx() as db:
        return module.verify_source(refs, read, db, cell, version, frozen, config)


def reseal(f, name, mutate):
    """Rehash modified evidence so byte checks cannot mask semantic gaps."""
    path = f.root / 'v2/agent-help-evidence.json'
    evidence = json.loads(path.read_bytes())
    ref = evidence['refs'][name]
    target = Path(ref['uri'])
    value = read(ref)
    mutate(value)
    target.chmod(0o644)
    target.write_text(json.dumps(value, sort_keys=True))
    ref['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    path.chmod(0o644)
    path.write_text(json.dumps(evidence, sort_keys=True))


def test_v2_origin_through_replays_registration_and_candidate(synthetic):
    f = synthetic
    assert verify(f) == dict(feedback_projection='exact')
    with pytest.raises(StateConflictError, match='agent help differs from live ledger'):
        verify(f, baseline())
    receipt = run_origin(f)
    _, source, candidate = register(f, receipt)
    assert source['channel'] == 'dev_agent'
    assert candidate.status.value == 'candidate'
    assert f.service.provenance(candidate.memory_id)['channel'] == 'dev_agent'
    assert len(read(read(receipt['validation'])['descriptor'])['agent_policy_replays']) == 2


@pytest.mark.parametrize('where', ['wrapper', 'execution_job', 'live'])
@pytest.mark.parametrize('field', ['id', 'state', 'kind', 'resources', 'receipt', 'usage',
                                  'reserved_at', 'claimed_at', 'settled_at', 'execution_limits', 'extra'])
def test_every_help_field_tamper_refused(synthetic, where, field):
    f = synthetic
    def change(job):
        value = job.get(field)
        if isinstance(value, dict):
            job[field] = dict(value, tampered=True)
        elif isinstance(value, (int, float)):
            job[field] = value + .001
        else:
            job[field] = 'tampered'
    if where == 'live':
        with f.store.tx() as db:
            state, payload = db.execute("SELECT state,payload FROM v2_agent_jobs WHERE id='help:e000004'").fetchone()
            job = dict(json.loads(payload), id='help:e000004', state=state)
            change(job)
            db.execute("UPDATE v2_agent_jobs SET id=?,state=?,payload=? WHERE id='help:e000004'",
                       (job['id'], job['state'], json.dumps({k:v for k,v in job.items() if k not in ('id', 'state')})))
    else:
        reseal(f, 'help_job', lambda job: change(job if where == 'wrapper' else job['execution_job']))
    with pytest.raises(StateConflictError):
        verify(f)


@pytest.mark.parametrize('change', ['missing', 'unsettled', 'extra_help', 'no_projection', 'answer', 'delivery'])
def test_v2_missing_ambiguous_and_delivery_tamper_refused(synthetic, change):
    f = synthetic
    if change in ('missing', 'unsettled', 'extra_help'):
        with f.store.tx() as db:
            if change == 'missing':
                db.execute("DELETE FROM v2_agent_jobs WHERE id='help:e000004'")
            elif change == 'unsettled':
                db.execute("UPDATE v2_agent_jobs SET state='unknown' WHERE id='help:e000004'")
            else:
                payload = json.loads(db.execute("SELECT payload FROM v2_agent_jobs WHERE id='help:e000004'").fetchone()[0])
                payload['receipt'].update(execution_id='e000005', help_id='help:e000005')
                db.execute('INSERT INTO v2_agent_jobs VALUES (?,?,?,?)',
                           ('cell', 'help:e000005', 'settled', json.dumps(payload)))
    elif change == 'no_projection':
        reseal(f, 'help_job', lambda job: job.pop('execution_job'))
    elif change == 'delivery':
        reseal(f, 'delivery', lambda delivery: delivery.update(extra='tampered'))
    else:
        path = f.root / 'v2/help.json'
        value = json.loads(path.read_bytes()); value['extra'] = 'tampered'
        path.chmod(0o644); path.write_text(json.dumps(value))
    with pytest.raises(StateConflictError, match='multiple_helps_in_interval' if change == 'extra_help' else None):
        verify(f)


@pytest.mark.parametrize('policy_version', [None, 'agent-policy-v1', 'agent-policy-v2'])
def test_version_keyed_origin_baseline_golden(tmp_path, monkeypatch, policy_version):
    f = fakes.build(tmp_path, monkeypatch)
    refs, cell, version, frozen, config = inputs(f)
    if policy_version is not None:
        frozen['agent_policy_version'] = policy_version
    sealed = {p: p.read_bytes() for p in f.root.rglob('*') if p.is_file() and p.suffix != '.sqlite'}
    with f.store.tx() as db:
        args = (refs, read, db, cell, version, frozen, config)
        assert agent_origin.verify_source(*args) == baseline().verify_source(*args) == dict(feedback_projection='exact')
    assert sealed == {p: p.read_bytes() for p in sealed}
    with f.store.tx() as db:
        db.execute("UPDATE v2_agent_jobs SET state='unknown' WHERE id='help:v0001'")
        for module in (baseline(), agent_origin):
            with pytest.raises(StateConflictError, match='agent help differs from live ledger'):
                module.verify_source(refs, read, db, cell, version, frozen, config)


def test_real_v2_origin_before_after_on_temporary_copy(tmp_path):
    if not (HISTORICAL / 'ledger.sqlite3').exists():
        pytest.skip('historical phase-5 run unavailable')
    ledger = tmp_path / 'ledger.sqlite3'
    with sqlite3.connect((HISTORICAL / 'ledger.sqlite3').as_uri() + '?mode=ro', uri=True) as src:
        with sqlite3.connect(ledger) as dst:
            src.backup(dst)
    # Preserve every sealed byte and absolute reference. Resolve reads to copies;
    # never initialize a store or rewrite any file in the historical run.
    copied = {}
    def copy_ref(ref, raw=False):
        uri = ref['uri']
        if uri not in copied:
            source = Path(uri)
            assert source.is_relative_to(HISTORICAL)
            target = tmp_path / f'evidence-{len(copied):03d}.data'
            shutil.copyfile(source, target)
            copied[uri] = target
        return read(dict(ref, uri=str(copied[uri])), raw=raw)
    with sqlite3.connect(ledger) as db:
        cell = json.loads(db.execute('SELECT payload FROM v2_cells').fetchone()[0])
        version = json.loads(db.execute('SELECT payload FROM v2_versions WHERE version=2').fetchone()[0])
        frozen = copy_ref(version['manifest'])
        config_path = Path(cell['artifact_dir']) / 'config.json'
        config = copy_ref(dict(uri=str(config_path), sha256=version['config_sha256']))
        refs = references(cell, 2)
        # Materialize all verify_source inputs before the baseline and fixed calls.
        for ref in refs.values():
            copy_ref(ref, raw=True)
        evidence = copy_ref(refs['agent_help_evidence'])
        for ref in evidence['refs'].values():
            copy_ref(ref, raw=True)
        for number in (1, 2):
            record = copy_ref(evidence['refs']['origin_record' if number == 1 else 'help_record'])
            copy_ref(record['source'], raw=True)
            job = copy_ref(evidence['refs']['origin_export' if number == 1 else 'help_export'])
            for name in ('public_feedback', 'bundle'):
                copy_ref(job['receipt'][name], raw=True)
        before = {uri: Path(uri).read_bytes() for uri in copied}
        args = (refs, copy_ref, db, cell, version, frozen, config)
        with pytest.raises(StateConflictError, match='^agent help differs from live ledger$'):
            baseline().verify_source(*args)
        assert agent_origin.verify_source(*args) == dict(feedback_projection='exact')
        jobs = [dict(json.loads(p), id=jid, state=s) for jid,s,p in db.execute(
            'SELECT id,state,payload FROM v2_agent_jobs WHERE cell=? ORDER BY rowid', (cell['key'],))]
        assert service_help.interval_help(jobs, 2) == harness_help.interval_help(jobs, 2)
        raw = harness_help.interval_help(jobs, 2)[0]
        assert raw['id'] == 'help:e000004'
        assert service_help.job_projection(raw, 2) == harness_help.job_projection(raw, 2) == copy_ref(evidence['refs']['help_job'])
        assert before == {uri: Path(uri).read_bytes() for uri in before}


@pytest.mark.parametrize('version', [2, 3, 10001])
def test_projection_cross_repo_golden(synthetic, version):
    raw = next(j for j in synthetic.ledger.jobs() if j['kind'] == 'help')
    before = deepcopy(raw)
    expected = harness_help.job_projection(raw, version)
    assert service_help.job_projection(raw, version) == expected
    assert json.dumps(service_help.job_projection(raw, version), sort_keys=True) == json.dumps(expected, sort_keys=True)
    delivery = dict(op='ask_help', status='completed', result=raw['receipt'], request_id='help-request', extra=[1, 2])
    assert service_help.delivery_projection(delivery, expected['receipt']) == harness_help.delivery_projection(delivery, expected['receipt'])
    assert raw == before


@pytest.mark.parametrize('case', ['equal_ticks', 'before', 'after', 'reopened', 'cache', 'rejected',
    'provider', 'undelivered', 'status', 'unknown', 'invalid_submit', 'undispatched', 'ambiguous_submit', 'no_submit'])
def test_interval_cross_repo_golden(synthetic, case):
    jobs = deepcopy(synthetic.ledger.jobs())
    help_job = next(j for j in jobs if j['kind'] == 'help')
    submits = [j for j in jobs if j['kind'] == 'request']
    if case == 'equal_ticks':
        for j in jobs:
            for field in ('reserved_at', 'claimed_at', 'settled_at'):
                j[field] = 1000.
    elif case == 'before':
        jobs.remove(help_job); jobs.insert(0, help_job)
        help_job['reserved_at'] = submits[0]['reserved_at']
    elif case == 'after':
        help_job['reserved_at'] = submits[-1]['reserved_at'] + 1
    elif case == 'reopened':
        help_job.update(reserved_at=submits[-1]['reserved_at'] + 1, reservation_epoch=1)
    elif case in ('cache', 'rejected', 'provider', 'undelivered', 'status'):
        help_job['receipt'].update({'cache': dict(cache_hit=True), 'rejected': dict(rejected=True),
            'provider': dict(provider='cache'), 'undelivered': dict(delivered=False), 'status': dict(status='unavailable')}[case])
    elif case == 'unknown':
        help_job['state'] = 'unknown'
    elif case == 'invalid_submit':
        submits[0]['receipt']['result'] = dict(outcome=dict(status='invalid_submission'))
    elif case == 'undispatched':
        submits[0]['receipt']['execution_dispatched'] = False
    elif case == 'ambiguous_submit':
        jobs.append(deepcopy(submits[0]))
    else:
        jobs.remove(submits[0])
    from benchmarks.attention_harness.memory_provisional import NotStaged
    try:
        expected = harness_help.interval_help(jobs, 2)
    except NotStaged as exc:
        with pytest.raises(StateConflictError, match=str(exc)):
            service_help.interval_help(jobs, 2)
    else:
        assert service_help.interval_help(jobs, 2) == expected
