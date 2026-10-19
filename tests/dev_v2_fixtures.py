"""CONTROLLED TEST INPUTS ONLY. No provider/simulator/real acceptance receipts."""
import json
import sqlite3
from pathlib import Path

from attention_memory_service.dev_v2 import digest, encoded, public_trace_projection


class Fixture:
    def __init__(self, tmp):
        self.root = tmp / 'executor'; self.root.mkdir()
        self.validator = tmp / 'validator'; self.validator.mkdir()
        self.catalogs = {}
        sdk_file = self.root / 'openarm365_dev_v2/sdk.py'
        sdk_file.parent.mkdir(); sdk_file.write_text('# controlled_test SDK\n')
        files = {str(sdk_file): digest(sdk_file.read_bytes())}
        planner = self.write(self.root / 'planner.json', {'evidence_kind': 'controlled_test'})
        model = self.write(self.root / 'model.json', {'evidence_kind': 'controlled_test'})
        files.update({planner['uri']: planner['sha256'], model['uri']: model['sha256']})
        self.sdk = {'contract': 'controlled_test', 'prompt': 'controlled_test', 'implementation': {'openarm365_dev_v2/sdk.py': files[str(sdk_file)]}, 'planner_lock': planner['sha256'], 'robot_model_identity': {model['uri']: model['sha256']}}
        self.sdk_sha = digest(json.dumps(self.sdk, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())
        lock = {'files': files, 'source_root': str(self.root), 'planner_lock': {'path': planner['uri'], 'sha256': planner['sha256']}, 'model_sources': {model['uri']: model['sha256']}, 'schema': 'attentionbench.dev-identity-lock.v2', 'formal_eligible': False, 'source_sets': {str(sdk_file.parent): [str(sdk_file)]}}
        self.lock = self.write(self.root / 'lock.json', lock)
        self.records = {}

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((encoded(value) + '\n').encode() if not isinstance(value, str) else value.encode())
        return {'uri': str(path), 'sha256': digest(path.read_bytes())}

    def make(self, name, seed=101, variant=0, success=False, code='def policy(sdk):\n    pass\n', exposure='none', memory_id=None, failure=False, completed_calls=None, visual=False):
        directory = self.root / name; directory.mkdir()
        config = {'task_id': 'pick', 'seed': seed, 'robot': 'OpenArm365', 'controller': 'test_controller', 'perception_mode': 'sim_gt', 'scene_id': f'scene-{variant}', 'object_set_id': f'objects-{variant}', 'camera_names': [f'camera-{variant}'], 'task_prompt': f'pick cup variant {variant}', 'camera_width': 64, 'camera_height': 64, 'randomize_cameras': False, 'official_env': 'PickPlaceCounterToSink', 'layout_id': 11 + variant, 'style_id': 34, 'base_mode': 'fixed', 'evidence_kind': 'controlled_test', 'sdk_call_limit': 80, 'horizon': 2400, 'attempt_wall_seconds': 480}
        if visual:
            config['advisor_proxy'] = {'enabled': True, 'mode': 'vision', 'model': 'controlled_test_visual_advisor'}
        camera = {k: config[k] for k in ('camera_names', 'camera_width', 'camera_height', 'randomize_cameras')}
        ctx = {k: config[k] for k in ('task_id', 'robot', 'controller', 'perception_mode', 'scene_id', 'object_set_id', 'camera_names', 'task_prompt')}
        ctx.update(benchmark='AttentionBench', suite='robocasa365', namespace='controlled_test_only', sdk_sha256=self.sdk_sha, camera_config_id='robocasa365:' + digest(json.dumps(camera, sort_keys=True, ensure_ascii=False).encode()), task_variant_id=f"PickPlaceCounterToSink:layout{11+variant}:style34:fixed")
        budget = {'assistance_credits': 1}
        config_ref = self.write(directory / 'config.json', config)
        source = self.write(directory / 'policy.py', code)
        ledger = directory / 'ledger.sqlite'
        manifest = {'schema': 'attentionbench.dev-experiment.v3', 'experiment_id': name, 'ledger_path': str(ledger), 'identity_lock': {'path': self.lock['uri'], 'sha256': self.lock['sha256']}, 'configs': {f'pick:{seed}': config}, 'cells': [{'key': name, 'task_id': 'pick', 'seed': seed, 'policy_id': 'full_trace_aware_attention_planner'}], 'evidence_kind': 'controlled_test'}
        mref = self.write(directory / 'manifest.json', manifest)
        record = {'version': 1, 'formal_eligible': False, 'valid': True, 'source': source, 'manifest': mref, 'config_sha256': config_ref['sha256']}
        vref = self.write(directory / 'version.json', record)
        binding = {'run_id': 'run:' + name, 'attempt_id': 'attempt:' + name + ':0', 'suite': 'robocasa365', 'task_id': 'pick', 'seed': seed, 'policy_sha256': source['sha256'], 'config_sha256': config_ref['sha256']}
        trace_value = {**binding, 'status': 'completed', 'evidence_kind': 'controlled_test'}
        if failure or completed_calls is not None:
            trace_value['sdk_events'] = [{'operation': 'get_observation', 'source': 'sensors', 'status': 'completed'} for _ in range(1 if completed_calls is None else completed_calls)]
            if failure:
                trace_value['error'] = 'RuntimeError: TimeoutError'
                trace_value['sdk_events'].append({'operation': 'move_delta', 'source': 'left_arm', 'status': 'failed'})
        trace = self.write(directory / 'trace.json', trace_value)
        native_value = {'run_id': binding['run_id'], 'attempt_id': binding['attempt_id'], 'status': 'completed', 'evaluated': True, 'native_success': success, 'evidence_kind': 'controlled_test'}
        native = self.write(directory / 'native.json', native_value)
        receipt = self.write(directory / 'sandbox.json', {**{k: binding[k] for k in ('run_id', 'attempt_id')}, 'sdk_call_budget': {'dispatched': 1}, 'control_steps': 1, 'elapsed_seconds': 1.0, 'evidence_kind': 'controlled_test'})
        safety = self.write(directory / 'safety.json', {**{k: binding[k] for k in ('run_id', 'attempt_id')}, 'evidence_kind': 'controlled_test'})
        bundle = {**binding, 'schema_version': 'attentionbench.formal-runner-result.v1', 'channel': 'dev_v2', 'formal_eligible': False, 'dev_version': 1, 'policy_id': 'full_trace_aware_attention_planner', 'config_lock': config_ref, 'status': 'completed', 'native_success': success, 'native_evaluator': native_value, 'infrastructure_failure': False, 'outcome_unknown': False, 'sandbox': {'process_isolated': True, 'sdk_rpc_only': True, 'deadline_enforced': True, 'action_cancellation_verified': True}, 'scene_condition': {'matched': True}, 'artifacts': {'trace': trace, 'native_result': native, 'sandbox_receipt': receipt, 'safety': safety}, 'evidence_kind': 'controlled_test'}
        if visual:
            bundle['artifact_dir'] = str(directory)
        bref = self.write(directory / 'bundle.json', bundle)
        feedback = self.write(directory / 'feedback.json', {**public_trace_projection(trace_value), 'evidence_kind': 'controlled_test'})
        validation_input = self.write(directory / 'validation-input.json', {'memory_ids': [] if exposure == 'none' else [memory_id], 'exposure': exposure, 'context': ctx, 'policy_sha256': source['sha256'], 'config_sha256': config_ref['sha256']})
        cell = {'experiment_id': name, 'policy_id': bundle['policy_id'], 'budget': budget}
        job = {'job_id': name + '-execute', 'cell': name, 'version': 1, 'kind': 'execute', 'claimed_at': 1.0, 'settled_at': 2.0, 'actual': {'executions': 1, 'sdk_calls': 1, 'control_steps': 1, 'execution_seconds': 1.0}, 'receipt': {'bundle': bref, 'trace': trace, 'native_result': native, 'execution_status': 'completed', 'public_feedback': feedback}}
        with sqlite3.connect(ledger) as db:
            db.executescript('CREATE TABLE v2_cells(id TEXT,exp TEXT,payload TEXT); CREATE TABLE v2_experiments(id TEXT,payload TEXT); CREATE TABLE v2_versions(cell TEXT,version INTEGER,payload TEXT); CREATE TABLE v2_jobs(id TEXT,cell TEXT,version INTEGER,kind TEXT,state TEXT,payload TEXT); CREATE TABLE v2_planner_requests(id TEXT,cell TEXT,payload TEXT);')
            db.execute('INSERT INTO v2_cells VALUES (?,?,?)', (name, name, encoded(cell)))
            db.execute('INSERT INTO v2_experiments VALUES (?,?)', (name, encoded({'manifest': mref})))
            db.execute('INSERT INTO v2_versions VALUES (?,?,?)', (name, 1, encoded(record)))
            db.execute('INSERT INTO v2_jobs VALUES (?,?,?,?,?,?)', (job['job_id'], name, 1, 'execute', 'settled', encoded(job)))
        descriptor = {'schema_version': 'attentionbench.dev-v2-import.v1', 'cell': name, 'version': 1, 'context': ctx, 'manifest': mref, 'config': config_ref, 'version_record': vref, 'sdk_identity': self.sdk, 'ledger': {'uri': str(ledger), 'sha256': digest(ledger.read_bytes())}, 'execute_job_id': job['job_id'], 'validation_input': validation_input, 'permission_scope': {'run_id': binding['run_id'], 'cell': name, 'code_version': 1, 'code_sha256': source['sha256'], 'config_sha256': digest(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()), 'action': 'sensor_only', 'action_version': 1, 'scope': {'mode': 'controlled_test_no_robot'}}}
        self.write(directory / 'execution.json', descriptor)
        vd = self.validator / name
        monitor = self.write(vd / 'monitor.json', {'schema_version': 'attentionbench.safety-monitor.v1', 'source': 'independent_safety_monitor', 'attempt_id': binding['attempt_id'], 'unsafe_attempts': 0, 'events': [{'kind': 'controlled_test_sample'}], 'violations': [], 'evidence_kind': 'controlled_test'})
        obs = [self.write(vd / f'obs{i}.json', {'robot': {'position': [0, 0, 0]}, 'perception': {'values': {'cup': [1, 2, 3]}}, 'evidence_kind': 'controlled_test'}) for i in (1, 2)]
        comparison = {'budget': budget, 'developer_model': 'controlled_test_model', 'execution_config': {k: v for k, v in config.items() if k != 'seed'}}
        comp = self.write(vd / 'comparison.json', comparison)
        costs = {'credits': 0, 'tokens': 0, 'seconds': 1.0, 'gpu_seconds': 0.0}
        verdict = {'schema_version': 'attentionbench.dev-v2-independent-validation.v1', 'purpose': 'offline_memory_validation', 'execution_import_id': name, 'bindings': {k: binding[k] for k in ('run_id', 'attempt_id', 'policy_sha256', 'config_sha256', 'task_id', 'seed')}, 'bundle': bref, 'native_result': native, 'trace': trace, 'native_success': success, 'checks': {'native_evaluation': True, 'safety': True, 'scene_reproduction': True, 'permissions': True, 'source_identity': True}, 'scene_replay_observations': obs, 'scene_replay_bindings': {'task_id': 'pick', 'seed': seed, 'config_sha256': config_ref['sha256']}, 'safety': monitor, 'permission_scope': descriptor['permission_scope'], 'costs': costs, 'validation_config': comparison, 'evidence_kind': 'controlled_test'}
        self.write(vd / 'verdict.json', verdict)
        trial = {'schema_version': 'attentionbench.dev-v2-trial.v1', 'execution_import_id': name, 'validation_descriptor': name + '/verdict.json', 'exposure': exposure, 'memory_id': memory_id, 'costs': costs, 'developer_model': 'controlled_test_model', 'budget': budget, 'validation_config': comp, 'evidence_kind': 'controlled_test'}
        self.write(vd / 'trial.json', trial)
        value = dict(name=name, context=ctx, config=config, descriptor=descriptor, record=record, bundle=bundle, verdict=verdict, trial=trial, cell=cell)
        self.records[name] = value
        return value

    def add_advice(self, failed, failure_signature=None):
        name = failed['name']; directory = self.root / name; d = failed['descriptor']; ledger = Path(d['ledger']['uri'])
        code = 'def policy(sdk):\n    # controlled_test repair\n    pass\n'
        source = self.write(directory / 'repair.py', code)
        previous = Path(failed['record']['source']['uri']).read_text()
        feedback = json.loads((directory / 'feedback.json').read_text())
        usage = {'total_tokens': 7, 'prompt_tokens': 4, 'completion_tokens': 3}
        hint = 'Align the gripper before closing it. controlled_test'
        ar_packet = {'previous_complete_source': previous, 'public_feedback': feedback}
        mr_packet = {**ar_packet, 'attention_control': {'help': {'provider': 'advisor', 'content': hint}}}
        jobs = []
        for kind, packet, output in [('advisor', ar_packet, hint), ('model', mr_packet, json.dumps({'source': code, 'reason': 'controlled_test repair'}))]:
            req = self.write(directory / f'{kind}-input.json', {'messages': [{'role': 'user', 'content': encoded(packet)}]})
            resp = self.write(directory / f'{kind}-output.json', {'content': output, 'usage': usage})
            wire = self.write(directory / f'{kind}-wire.json', {'http_status': 200, 'request_id': 'controlled_test_' + kind, 'body': {'usage': usage, 'choices': [{'message': {'content': output}}]}})
            receipt = {'request': req, 'response': resp, 'provider_response': wire, 'usage': usage}
            job = {'job_id': name + '-' + kind, 'cell': name, 'version': 2, 'kind': kind, 'claimed_at': 3.0 if kind == 'advisor' else 5.0, 'settled_at': 4.0 if kind == 'advisor' else 6.0, 'actual': {'provider_tokens': 7, 'input_tokens': 4, 'output_tokens': 3, **({'advisor_calls': 1, 'help_count': 1, 'help_credits': 1} if kind == 'advisor' else {'dev_calls': 1})}, 'receipt': receipt}
            jobs.append(job)
        ac, mc = [j['receipt'] for j in jobs]
        help_ref = self.write(directory / 'help.json', {'provider': 'advisor', 'content': hint, 'request': ac['request'], 'response': ac['response'], 'usage': usage})
        record = {'version': 2, 'formal_eligible': False, 'valid': True, 'source': source, 'manifest': d['manifest'], 'config_sha256': d['config']['sha256'], 'help': help_ref, 'model_input': mc['request'], 'model_output': mc['response']}
        vref = self.write(directory / 'version2.json', record)
        row = {'created_at': 3.1, 'responded_at': 4.1, 'deadline_epoch': 5.0, 'binding': {'cell': name, 'version': 2, 'input_sha256': digest(json.dumps([{'role': 'user', 'content': encoded(ar_packet)}], sort_keys=True, ensure_ascii=False).encode()), 'source_sha256': failed['record']['source']['sha256'], 'policy_id': failed['cell']['policy_id'], 'config_sha256': d['config']['sha256']}, 'cell': name, 'version': 2, 'request_type': 'advisor', 'state': 'answered', 'accepted': True, 'adopted': True, 'response': ac['response'], 'adoption': {'model_input': mc['request'], 'result_source': source}}
        with sqlite3.connect(ledger) as db:
            db.execute('INSERT INTO v2_versions VALUES (?,?,?)', (name, 2, encoded(record)))
            db.execute('INSERT INTO v2_planner_requests VALUES (?,?,?)', ('help-' + name, name, encoded(row)))
            for j in jobs:
                db.execute('INSERT INTO v2_jobs VALUES (?,?,?,?,?,?)', (j['job_id'], name, 2, j['kind'], 'settled', encoded(j)))
        d['ledger']['sha256'] = digest(ledger.read_bytes())
        self.write(directory / 'execution.json', d)
        source_d = {**d, 'version': 2, 'version_record': vref, 'failed_execution_import_id': name, 'validator_id': 'validator-' + name, 'validation_descriptor': name + '/verdict.json', 'advisor_job_id': jobs[0]['job_id'], 'model_job_id': jobs[1]['job_id'], 'request_id': 'help-' + name}
        if failure_signature is not None:
            source_d['failure_signature'] = failure_signature
        self.write(directory / 'source.json', source_d)
        return source_d

    def seal(self):
        for name, r in self.records.items():
            for role, root, cid in [('executor', self.root, name), ('validator', self.validator, 'validator-' + name)]:
                self.catalogs[cid] = {'root': str(root), 'role': role, 'context': r['context'], 'ledger_origin': r['descriptor']['ledger']['uri'], 'files': {str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()}}
        return self.catalogs

    def add_result_model(self, result, grant, authority):
        """Controlled generation receipt for a previously issued real Service grant."""
        import time
        name = result['name']; directory = self.root / name; d = result['descriptor']
        previous = 'controlled_test previous code\n'
        code = Path(result['record']['source']['uri']).read_text()
        entry = {'memory_id': grant['memory_id'], 'version': grant['memory_version'], 'content': grant['guidance'], 'grant_id': grant['grant_id'], 'context': grant['context'], 'development_id': grant['development_id'], 'source_policy_sha256': grant['source_policy_sha256'], 'authority': authority}
        packet = {'previous_complete_source': previous, 'attention_control': {'memory': [entry]}}
        usage = {'total_tokens': 7, 'prompt_tokens': 4, 'completion_tokens': 3}
        req = self.write(directory / 'model-input.json', {'messages': [{'role': 'system', 'content': 'Do not access private attributes.'}, {'role': 'user', 'content': encoded(packet)}]})
        content = json.dumps({'source': code, 'reason': 'controlled_test'})
        resp = self.write(directory / 'model-output.json', {'content': content, 'usage': usage})
        wire = self.write(directory / 'model-wire.json', {'http_status': 200, 'request_id': 'controlled_test_model', 'body': {'usage': usage, 'choices': [{'message': {'content': content}}]}})
        receipt = {'request': req, 'response': resp, 'provider_response': wire, 'usage': usage}
        now = time.time()
        job = {'job_id': name + '-model', 'cell': name, 'version': 1, 'kind': 'model', 'claimed_at': now, 'settled_at': now + .01, 'actual': {'dev_calls': 1, 'provider_tokens': 7, 'input_tokens': 4, 'output_tokens': 3}, 'receipt': receipt}
        record = {**result['record'], 'model_input': req, 'model_output': resp}
        self.write(directory / 'version.json', record)
        d['version_record'] = {'uri': str(directory / 'version.json'), 'sha256': digest((directory / 'version.json').read_bytes())}
        with sqlite3.connect(d['ledger']['uri']) as db:
            db.execute('UPDATE v2_versions SET payload=? WHERE version=1', (encoded(record),))
            db.execute('INSERT INTO v2_jobs VALUES (?,?,?,?,?,?)', (job['job_id'], name, 1, 'model', 'settled', encoded(job)))
            execute = json.loads(db.execute("SELECT payload FROM v2_jobs WHERE kind='execute'").fetchone()[0])
            execute.update(claimed_at=now + .02, settled_at=now + .03)
            db.execute("UPDATE v2_jobs SET payload=? WHERE kind='execute'", (encoded(execute),))
        d['ledger']['sha256'] = digest(Path(d['ledger']['uri']).read_bytes())
        d['model_job_id'] = job['job_id']
        self.write(directory / 'execution.json', d)

    def bind_validation(self, record, grant, *, consume=True):
        """Synthetic pre-dispatch test event; never a real execution receipt."""
        import time
        d = record['descriptor']; name = record['name']
        ref = d['validation_input']
        value = json.loads(Path(ref['uri']).read_bytes())
        value['validation_grant_id'] = grant['grant_id']
        d['validation_input'] = self.write(Path(ref['uri']), value)
        # Synthetic native trace consumption proof, built only for test fixtures.
        trace_ref = record['bundle']['artifacts']['trace']
        trace = json.loads(Path(trace_ref['uri']).read_bytes())
        binding = grant['binding']
        if consume:
            trace['memory_validation'] = {'validation_grant_id': grant['grant_id'], 'input_sha256': d['validation_input']['sha256'], 'memory_ids': value['memory_ids'], 'exposure': value['exposure'], 'memory_version': binding['memory_version'] if value['memory_ids'] else None, 'guidance_sha256': digest(b'Align the gripper before closing it. controlled_test') if value['memory_ids'] else None, 'consumed_before_policy': True}
        record['bundle']['artifacts']['trace'] = self.write(Path(trace_ref['uri']), trace)
        bundle_ref = self.write(self.root / name / 'bundle.json', record['bundle'])
        record['verdict']['bundle'] = bundle_ref
        record['verdict']['trace'] = record['bundle']['artifacts']['trace']
        self.write(self.validator / name / 'verdict.json', record['verdict'])
        now = time.time()
        event = {'cell': name, 'version': 1, 'validation_input': d['validation_input'], 'validation_grant_id': grant['grant_id'], 'source': record['record']['source'], 'config': d['config']}
        with sqlite3.connect(d['ledger']['uri']) as db:
            db.execute('CREATE TABLE v2_events(seq INTEGER PRIMARY KEY,at REAL,kind TEXT,payload TEXT)')
            db.execute('INSERT INTO v2_events(at,kind,payload) VALUES (?,?,?)', (now, 'memory_validation_dispatch', encoded(event)))
            job = json.loads(db.execute("SELECT payload FROM v2_jobs WHERE kind='execute'").fetchone()[0])
            job.update(claimed_at=now + .01, settled_at=now + .02)
            job['receipt'].update(bundle=bundle_ref, trace=record['bundle']['artifacts']['trace'])
            db.execute("UPDATE v2_jobs SET payload=? WHERE kind='execute'", (encoded(job),))
        d['ledger']['sha256'] = digest(Path(d['ledger']['uri']).read_bytes())
        self.write(self.root / name / 'execution.json', d)
        record['trial']['validation_grant_id'] = grant['grant_id']
        self.write(self.validator / name / 'trial.json', record['trial'])
