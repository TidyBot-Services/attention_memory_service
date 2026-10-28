"""Cross-repo synthetic exporter -> prepare -> service import; no live processes."""
import copy
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.dev_v2 import digest, encoded
from attention_memory_service.identity import store_id
from attention_memory_service.memory_service import MemoryService
from tests.dev_v2_fixtures import Fixture
from tests.test_provisional_notes import payload

from benchmarks.attention_harness import dev_planner_memory, memory_source_drift
from benchmarks.attention_harness.dev_agent_adapters import Executor
from benchmarks.attention_harness.dev_agent_export import Exporter
from benchmarks.attention_harness.dev_agent_ledger import AgentLedger
from benchmarks.attention_harness.dev_agent_workspace import Workspace, atomic_write
from benchmarks.attention_harness.dev_ledger import DevV2Store
from benchmarks.attention_harness.dev_v2_loop import immutable
from benchmarks.attention_harness.dev_v2_memory_import import prepare

SOURCE = b'from robot_sdk import left_arm\nleft_arm.move_delta(0, 0, .01)\n'


class FakeRunner:
    """Same raw binding and result shape as OpenArm365DevRunnerV2."""
    def __init__(self, *, episode_observer, **kwargs):
        self.observe = episode_observer

    def execute(self, request):
        episode = request.artifact_root / 'episode'
        episode.mkdir()
        self.observe(str(episode))
        ids = dict(run_id=request.run_id, attempt_id=request.attempt_id)
        binding = dict(**ids, suite=request.suite, task_id=request.task_id, seed=request.seed,
                       policy_sha256=request.policy_sha256, config_sha256=request.config_sha256)
        observation = dict(robot={'robot0_base_pos': [0., 0., 0.]}, perception=dict(values={'cup': [1., 2., 3.]}))
        for label in ('initial', 'final'):
            immutable(episode / f'{label}_public_observation.json', observation)
        condition = dict(matched=True)
        immutable(episode / 'scene-condition.json', dict(validation=condition, state={}))
        trace = immutable(episode / 'trace.json', dict(**binding, status='completed', sdk_events=[
            dict(source='base', operation='move_delta', status='completed', arguments=dict(dyaw=.1))]))
        native_value = dict(**ids, status='completed', evaluated=True, native_success=False)
        native = immutable(episode / 'native_result.json', native_value)
        receipt = immutable(episode / 'sandbox_receipt.json', dict(**ids, sdk_call_budget=dict(dispatched=1),
                            control_steps=2, elapsed_seconds=.5))
        safety = immutable(episode / 'safety.json', dict(**ids, violations=[]))
        return dict(**binding, schema_version='attentionbench.formal-runner-result.v1', channel='dev_v2',
                    status='completed', native_success=False, native_evaluator=native_value,
                    formal_eligible=False, runner_boundary='dev_loop', entry_lock=None,
                    artifacts=dict(trace=trace, native_result=native, sandbox_receipt=receipt, safety=safety),
                    scene_condition=condition, infrastructure_failure=False, outcome_unknown=False,
                    sandbox=dict(process_isolated=True, sdk_rpc_only=True, deadline_enforced=True,
                                 action_cancellation_verified=True), artifact_dir=str(episode))


class AgentFixture:
    def __init__(self, tmp_path, monkeypatch, *, source=SOURCE, history=True, version=1):
        self.f = Fixture(tmp_path, interface=True)
        base = self.f.make('scope')
        self.context = dict(base['context'], seed=101,
                            scene_contract_ref=immutable(self.f.root / 'context-scene.json', {'evidence_kind': 'controlled_test'}))
        self.root = self.f.root / 'agent'
        self.root.mkdir()
        self.store = DevV2Store(self.root / 'ledger.sqlite')
        self.config = dict(base['config'], budget_tier='dev', runner_boundary='dev_loop', dev_mode='agent',
                           sdk_call_limit=320, horizon=9600, attempt_wall_seconds=1920,
                           control_freq=20, dev_version_history=history)
        self.manifest = dict(schema='attentionbench.dev-experiment.v3', experiment_id='exp', dev_mode='agent',
            ledger_path=str(self.store.path), identity_lock=dict(path=self.f.lock['uri'], sha256=self.f.lock['sha256']),
            configs={'pick:101': self.config}, cells=[dict(key='cell', task_id='pick', seed=101, policy_id='autonomous')],
            initial_sources={'pick': dict(source=SOURCE.decode())})
        manifest_ref = immutable(self.root / 'manifest.json', self.manifest)
        with self.store.tx() as db:
            self.store.put(db, 'experiments', 'exp', dict(manifest=manifest_ref))
            db.execute('INSERT INTO v2_cells VALUES (?,?,?)', ('cell', 'exp', encoded(dict(
                key='cell', experiment_id='exp', dev_mode='agent', policy_id='autonomous',
                artifact_dir=str(self.root)))))
        self.ws = Workspace(self.root / 'workspace').prepare({'task_prompt': 'Move safely.'}, self.config)
        self.ledger = AgentLedger(self.store.path, 'cell')
        self.service = MemoryService(tmp_path / 'authority.sqlite')
        self.service.create_provisional_note(**payload(context=self.context))
        self.service.create_provisional_note(**payload('trusted-memory-fixture', context=self.context))
        self.grant = self.service.authorize_dev_use(memory_id='p-1', memory_version=1,
            development_id=f'cell:v{version}', context=self.context, source_policy_sha256=digest(SOURCE), now=time.time())
        # A separate disposable trusted grant fixture exercises result imports.
        # Production authorization and provisional exclusions are tested separately.
        self.trusted = dict(self.grant, grant_id='trusted-fixture', memory_id='trusted-memory-fixture', trust_scope='scene_scoped',
                            excluded_from_results=False, trusted_here=True, unvalidated=False, result=None)
        with sqlite3.connect(self.service.store.path) as db:
            db.execute('INSERT INTO memory_dev_exposures VALUES (?,?,?,?)',
                       (self.trusted['grant_id'], self.trusted['memory_id'], self.trusted['development_id'], encoded(self.trusted)))
        entry = dict(grant_id=self.trusted['grant_id'], memory_id=self.trusted['memory_id'], version=1,
                     content=self.trusted['guidance'], context=self.context, development_id=f'cell:v{version}',
                     source_policy_sha256=digest(SOURCE), authority=store_id(self.service.store.path),
                     seeded_for_tests=False, excluded_from_results=False, trusted_here=True)
        packet = dict(entries=[entry], seeded_for_tests=False, excluded_from_results=False)
        packet_ref = immutable(self.ws.private / f'memory/v{version}/memory-input.json', packet)
        with self.store.tx() as db:
            self.store.event(db, 'planner_memory_retrieved', dict(cell='cell', version=version, packet=packet_ref))
        if version > 1:
            prior = immutable(self.root / 'prior.py', SOURCE.decode())
            self.store.version('cell', dict(version=version-1, source=prior, formal_eligible=False))
        self.version, self.source = version, source
        self.response = dict(op='submit', request_id='r', version_id=f'v{version:04d}',
                             execution_id='e000003', status='completed')
        self.exporter = Exporter(self.store, 'cell', self.ws, self.config, self.ledger)
        monkeypatch.setattr(dev_planner_memory, 'context', lambda *a, **kw: self.context)
        monkeypatch.setattr(dev_planner_memory, 'sdk_identity', lambda *a: self.f.sdk)
        monkeypatch.setattr(memory_source_drift, 'current_identity', lambda config, lock, out: (lock, None))
        monkeypatch.setattr(self.store, 'planner_snapshot', lambda cell: {}, raising=False)

    def export(self, *, op='submit', valid=True):
        self.response['op'] = op
        atomic_write(self.ws.private, 'sources/r.py', self.source)
        self.ws.publish(f'versions/v{self.version:04d}/help.json', dict(status='not_requested'))
        self.ledger.reserve('request:r', 'request', dict(formal_versions=int(op == 'submit'), moving_executions=1), execution=True)
        self.ledger.claim('request:r')
        executor = Executor(self.config, sim_python='never-launched', runner_factory=FakeRunner)
        if valid:
            (self.ws.private / 'executions/e000003').mkdir(parents=True)
            executor.bind_execution('cell', self.version if op == 'submit' else None, 'e000003')
            executor.execute(self.source, self.ws.private / 'executions/e000003', deadline=time.monotonic() + 5)
            self.ledger.record_execution('request:r', executor.execution_usage)
        self.ledger.settle('request:r', self.response)
        self.exporter.capture(self.response, dict(version=self.version, valid=valid, execution_id='e000003'), executor)
        self.exporter.settle(self.response)
        return executor

    def prepare(self):
        ref = prepare(self.store, 'cell', self.version, self.context['namespace'], self.root / 'import', self.f.root)
        self.catalog = json.loads(Path(ref['uri']).read_bytes())
        self.descriptor_path = self.root / 'import/descriptor.json'
        self.descriptor = json.loads(self.descriptor_path.read_bytes())
        return self.descriptor

    def seal(self):
        # Re-pin test mutations to exercise semantic checks independently of SHA checks.
        for name in self.catalog['files']:
            self.catalog['files'][name] = digest((self.f.root / name).read_bytes())
        self.service.dev_v2.register('agent', self.catalog)
        return 'agent/import/descriptor.json'

    def verify(self):
        return self.service.dev_v2.verify_execution('agent', self.seal())

    def write(self, path, value):
        if path.exists(): path.chmod(0o600)
        return Fixture.write(path, value)

    def mutate_bundle(self, change):
        d = self.descriptor
        ledger = Path(d['ledger']['uri'])
        with sqlite3.connect(ledger) as db:
            job = json.loads(db.execute('SELECT payload FROM v2_jobs WHERE id=?', (d['execute_job_id'],)).fetchone()[0])
            ref = job['receipt']['bundle']
            bundle = json.loads(Path(ref['uri']).read_bytes())
            change(bundle, job)
            job['receipt']['bundle'] = self.write(Path(ref['uri']), bundle)
            db.execute('UPDATE v2_jobs SET payload=? WHERE id=?', (encoded(job), d['execute_job_id']))
        d['ledger']['sha256'] = digest(ledger.read_bytes())
        self.write(self.descriptor_path, d)


@pytest.mark.parametrize('history,source,version', [(True, SOURCE, 1), (False, SOURCE, 2),
    (True, b'from robot_sdk import sensors\nsensors.get_observation()\n', 1)])
def test_export_prepare_verify_and_import(tmp_path, monkeypatch, history, source, version):
    f = AgentFixture(tmp_path, monkeypatch, source=source, history=history, version=version)
    executor = f.export(); d = f.prepare()
    evidence, _, bundle, artifacts, _ = f.verify()
    assert evidence['channel'] == 'dev_agent'
    assert bundle['run_id'] == 'run:cell' and bundle['attempt_id'] == f'attempt:cell:{version-1}'
    assert artifacts['trace']['run_id'] == bundle['run_id']
    assert bundle['scene_condition'] == executor.result['scene_condition']
    assert bundle['public_base_yaw_visible'] is history
    assert d['permission_scope']['action'] == ('agent_dev_tier_motion' if source == SOURCE else 'sensor_only')
    assert 'actor' not in d['permission_scope'] and 'approval_id' not in d
    assert f.ledger.snapshot()['used']['moving_executions'] == 1
    f.service.import_dev_v2_execution(import_id='execution', catalog_id='agent', descriptor='agent/import/descriptor.json')
    feedback = f.service.dev_v2.ref('agent', evidence['public_feedback'])
    text = encoded(feedback)
    original = f.service.record_provisional_outcome(grant_id=f.grant['grant_id'], development_id=f'cell:v{version}',
        public_feedback_text=text, public_feedback_sha256=digest(text.encode()))
    outcome = f.service.record_provisional_outcome(grant_id=f.grant['grant_id'], development_id=f'cell:v{version}',
        public_feedback_text=text, public_feedback_sha256=digest(text.encode()), execution_import_id='execution')
    assert outcome['channel'] == outcome['provenance']['channel'] == 'dev_agent' and outcome['observational']
    assert outcome['recorded_at'] == original['recorded_at']
    assert f.service.record_provisional_outcome(grant_id=f.grant['grant_id'], development_id=f'cell:v{version}',
        public_feedback_text=text, public_feedback_sha256=digest(text.encode())) == outcome
    result = f.service.record_dev_v2_result(grant_id=f.trusted['grant_id'], execution_import_id='execution')
    assert result['result']['channel'] == result['result']['provenance']['channel'] == 'dev_agent'
    assert f.service.record_dev_v2_result(grant_id=f.trusted['grant_id'], execution_import_id='execution') == result
    f.service.dev_v2.verify_result(result)
    from attention_memory_service.dev_use import _online_grant
    public = _online_grant(result)['result']
    assert public['channel'] == 'dev_agent' and not {'bundle', 'ledger', 'memory_input', 'provenance'} & public.keys()
    with pytest.raises(StateConflictError, match='excluded'):
        f.service.record_dev_v2_result(grant_id=f.grant['grant_id'], execution_import_id='execution')
    with pytest.raises(StateConflictError, match='agent offline validation'):
        f.service.dev_v2.verdict('execution', 'unused-validator', 'unused.json')


@pytest.mark.parametrize('field,value', [('public_base_yaw_visible', None), ('public_base_yaw_visible', 1),
    ('public_base_yaw_visible', 'true'), ('formal_eligible', True), ('run_id', 'run:agent'),
    ('attempt_id', 'attempt:agent:0')])
def test_invalid_bundle(tmp_path, monkeypatch, field, value):
    f = AgentFixture(tmp_path, monkeypatch); f.export(); f.prepare()
    def change(bundle, job):
        if value is None: bundle.pop(field)
        else: bundle[field] = value
    f.mutate_bundle(change)
    with pytest.raises(StateConflictError): f.verify()


@pytest.mark.parametrize('where,key', [('descriptor', 'approval_id'), ('descriptor', 'actor'),
    ('permission_scope', 'approval_id'), ('permission_scope', 'actor'), ('agent_execution', 'actor')])
def test_no_agent_approval_identity(tmp_path, monkeypatch, where, key):
    f = AgentFixture(tmp_path, monkeypatch); f.export(); d = f.prepare()
    (d if where == 'descriptor' else d[where])[key] = 'invented'
    f.write(f.descriptor_path, d)
    with pytest.raises(StateConflictError, match='actor or approval_id'): f.verify()


@pytest.mark.parametrize('artifact', ['trace', 'native_result', 'sandbox_receipt', 'safety'])
def test_raw_artifact_tamper(tmp_path, monkeypatch, artifact):
    f = AgentFixture(tmp_path, monkeypatch); f.export(); f.prepare()
    def change(bundle, job):
        ref = bundle['artifacts'][artifact]
        value = json.loads(Path(ref['uri']).read_bytes())
        value['run_id'] = 'tampered'
        ref = f.write(Path(ref['uri']), value)
        bundle['artifacts'][artifact] = ref
        if artifact in ('trace', 'native_result'): job['receipt'][artifact] = ref
    f.mutate_bundle(change)
    with pytest.raises(StateConflictError, match='raw native artifact ID mismatch'): f.verify()


@pytest.mark.parametrize('mutation', ['marker', 'unsettled', 'no_eval', 'usage', 'timing', 'missing_agent_ledger'])
def test_agent_ledger_must_match(tmp_path, monkeypatch, mutation):
    f = AgentFixture(tmp_path, monkeypatch); f.export(); d = f.prepare()
    ledger = Path(d['ledger']['uri'])
    with sqlite3.connect(ledger) as db:
        if mutation == 'marker':
            cell = json.loads(db.execute('SELECT payload FROM v2_cells').fetchone()[0]); cell.pop('agent_export')
            db.execute('UPDATE v2_cells SET payload=?', (encoded(cell),))
        elif mutation == 'unsettled': db.execute("UPDATE v2_jobs SET state='claimed'")
        elif mutation == 'missing_agent_ledger': db.execute('DROP TABLE v2_agent_jobs')
        else:
            agent = json.loads(db.execute('SELECT payload FROM v2_agent_jobs').fetchone()[0])
            if mutation == 'no_eval': agent['receipt']['op'] = 'run'
            elif mutation == 'usage': agent['execution_usage']['sdk_calls'] += 1
            elif mutation == 'timing': agent['claimed_at'] += .001
            db.execute('UPDATE v2_agent_jobs SET payload=?', (encoded(agent),))
    d['ledger']['sha256'] = digest(ledger.read_bytes()); f.write(f.descriptor_path, d)
    with pytest.raises(StateConflictError): f.verify()


@pytest.mark.parametrize('op,valid', [('run', True), ('submit', False)])
def test_nonformal_or_invalid_exports_not_importable(tmp_path, monkeypatch, op, valid):
    f = AgentFixture(tmp_path, monkeypatch); f.export(op=op, valid=valid)
    assert not f.store.cell('cell')['jobs']
    with pytest.raises(Exception, match='unsupported'): f.prepare()


def test_validation_input_and_grant_stay_refused(tmp_path, monkeypatch):
    f = AgentFixture(tmp_path, monkeypatch); f.export(); d = f.prepare()
    d['validation_input'] = {'uri': 'unused', 'sha256': 'a' * 64}
    f.write(f.descriptor_path, d)
    with pytest.raises(StateConflictError, match='offline validation'): f.verify()
    # A synthetic candidate carrying agent provenance cannot receive a grant.
    with sqlite3.connect(f.service.store.path) as db:
        raw = json.loads(db.execute("SELECT payload FROM memory_v2_provenance WHERE memory_id='p-1'").fetchone()[0])
        raw['channel'] = 'dev_agent'
        memory = json.loads(db.execute("SELECT payload FROM memories WHERE id='p-1'").fetchone()[0])
        memory['status'] = 'candidate'
        db.execute("UPDATE memories SET payload=? WHERE id='p-1'", (encoded(memory),))
        db.execute("UPDATE memory_v2_provenance SET payload=? WHERE memory_id='p-1'", (encoded(raw),))
    with pytest.raises(StateConflictError, match='requires a Dev v2 candidate'):
        f.service.authorize_dev_v2_validation(memory_id='p-1', cell='cell', version=1, seed=101,
            context=f.context, policy_sha256=digest(SOURCE), config_sha256=d['config']['sha256'], exposure='candidate_validation')


@pytest.mark.parametrize('case', ['native_value', 'trace_binding', 'receipt_bounds', 'permission_inventory',
    'permission_action', 'permission_version', 'request_id', 'validation_purpose', 'feedback_projection'])
def test_agent_shared_guards(tmp_path, monkeypatch, case):
    f = AgentFixture(tmp_path, monkeypatch); f.export(); d = f.prepare()
    if case in ('native_value', 'trace_binding', 'receipt_bounds'):
        def change(bundle, job):
            key = {'native_value': 'native_result', 'trace_binding': 'trace', 'receipt_bounds': 'sandbox_receipt'}[case]
            ref = bundle['artifacts'][key]
            value = json.loads(Path(ref['uri']).read_bytes())
            if case == 'native_value': value['native_success'] = True
            elif case == 'trace_binding': value['policy_sha256'] = 'f' * 64
            else: value['control_steps'] = 9601
            bundle['artifacts'][key] = f.write(Path(ref['uri']), value)
            if key in ('trace', 'native_result'): job['receipt'][key] = bundle['artifacts'][key]
        f.mutate_bundle(change)
    elif case == 'feedback_projection':
        def change(bundle, job):
            ref = job['receipt']['public_feedback']
            value = json.loads(Path(ref['uri']).read_bytes()); value['action_trace'] = []
            job['receipt']['public_feedback'] = f.write(Path(ref['uri']), value)
        f.mutate_bundle(change)
    else:
        if case == 'permission_inventory': d['permission_scope']['scope']['operations'] = ['base.move_delta']
        elif case == 'permission_action': d['permission_scope']['action'] = 'sensor_only'
        elif case == 'permission_version': d['permission_scope']['action_version'] = True
        elif case == 'request_id': d['agent_execution']['request_id'] = 'request:other'
        else: d['import_purpose'] = 'memory_validation'
        f.write(f.descriptor_path, d)
    with pytest.raises(StateConflictError): f.verify()


@pytest.mark.parametrize('case', ['packet_sha', 'missing_grant', 'wrong_source', 'late_event'])
def test_agent_result_requires_generation_binding(tmp_path, monkeypatch, case):
    f = AgentFixture(tmp_path, monkeypatch); f.export(); d = f.prepare()
    packet = f.ws.private / 'memory/v1/memory-input.json'
    if case in ('packet_sha', 'missing_grant'):
        value = json.loads(packet.read_bytes()); value['entries'] = []
        ref = f.write(packet, value)
        if case == 'missing_grant':
            f.mutate_bundle(lambda bundle, job: bundle.update(agent_memory_input=ref))
            ledger = Path(d['ledger']['uri'])
            with sqlite3.connect(ledger) as db:
                raw = json.loads(db.execute("SELECT payload FROM v2_events WHERE kind='planner_memory_retrieved'").fetchone()[0])
                raw['packet'] = ref
                db.execute("UPDATE v2_events SET payload=? WHERE kind='planner_memory_retrieved'", (encoded(raw),))
            d['ledger']['sha256'] = digest(ledger.read_bytes()); f.write(f.descriptor_path, d)
    elif case == 'late_event':
        ledger = Path(d['ledger']['uri'])
        with sqlite3.connect(ledger) as db:
            db.execute("UPDATE v2_events SET at=? WHERE kind='planner_memory_retrieved'", (time.time() + 3600,))
        d['ledger']['sha256'] = digest(ledger.read_bytes()); f.write(f.descriptor_path, d)
    else:
        f.trusted['source_policy_sha256'] = 'e' * 64
        with sqlite3.connect(f.service.store.path) as db:
            db.execute('UPDATE memory_dev_exposures SET payload=? WHERE grant_id=?', (encoded(f.trusted), f.trusted['grant_id']))
    f.service.import_dev_v2_execution(import_id='execution', catalog_id='agent', descriptor=f.seal())
    with pytest.raises(StateConflictError):
        f.service.record_dev_v2_result(grant_id=f.trusted['grant_id'], execution_import_id='execution')
