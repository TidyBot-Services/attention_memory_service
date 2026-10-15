"""Offline Dev v2 evidence adapter. Never synthesizes legacy run/request rows.

Trust is configured by the service operator, not in an online JSON payload.
Catalogs pin sealed executor/evaluator snapshots; the importer key is held only
by the trusted execution/validation side. All reads recheck the pinned bytes.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import math
import re
import sqlite3
import time
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .core.models import MemoryRecord
from .core.store import StateConflictError

SCOPE = ('benchmark', 'suite', 'task_id', 'robot', 'controller', 'perception_mode',
         'scene_id', 'object_set_id', 'camera_config_id', 'task_variant_id',
         'sdk_sha256', 'namespace')
AXES = ('scene_id', 'object_set_id', 'camera_config_id', 'task_variant_id')
PRIVATE = re.compile(r'reward|evaluator|native_success|_check_success|private|hidden|derived[_ ]diagnos|eval[_ ]diagnos', re.I)
SHA = re.compile(r'[a-f0-9]{64}\Z')
_OPERATION = ContextVar('dev_v2_evidence_operation', default=None)


def evidence_operation(function):
    """Verify each pinned file once per authority call, never across calls.

    Nested source/plan/pair checks share reads; before returning, recheck file
    identity/metadata and canonical paths to reject changes during the call.
    Without this scope, a trusted lookup would rehash the transitive SDK for
    every source/plan/pair recursion. Every later HTTP call starts fresh.
    """
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _OPERATION.get() is not None:
            return function(*args, **kwargs)
        state = {'memo': {}, 'files': {}, 'directories': {}}
        token = _OPERATION.set(state)
        try:
            result = function(*args, **kwargs)
            for catalog, name, canonical, identity in state['files'].values():
                path = DevV2Evidence._path(catalog, name)
                require(path == canonical and _identity(path) == identity, 'evidence changed during authority operation')
            for folder, expected in state['directories'].items():
                require(_source_set(folder) == expected, 'SDK source file set changed during authority operation')
            return result
        finally:
            _OPERATION.reset(token)
    return wrapped


def _identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _source_set(folder):
    return sorted(str(p.absolute()) for p in Path(folder).rglob('*.py') if '__pycache__' not in p.parts)


def memo_evidence(kind):
    def decorate(function):
        @wraps(function)
        def wrapped(self, key, *args, **kwargs):
            state = _OPERATION.get()
            memo_key = (kind, str(self.store.path), key)
            if state is not None and memo_key in state['memo']:
                return state['memo'][memo_key]
            result = function(self, key, *args, **kwargs)
            if state is not None:
                state['memo'][memo_key] = result
            return result
        return wrapped
    return decorate


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise StateConflictError(message)


def public(value):
    require(not PRIVATE.search(encoded(value)), 'private evaluation content cannot enter online guidance')


def scope(value):
    require(isinstance(value, dict) and all(isinstance(value.get(k), str) and value[k].strip() for k in SCOPE), 'complete exact Dev v2 scope required')
    require(SHA.fullmatch(value['sdk_sha256']) is not None, 'SDK digest required')
    require(value['perception_mode'] in {'sim_gt', 'vision'}, 'invalid perception mode')
    require(isinstance(value.get('camera_names'), list) and value['camera_names'] and all(isinstance(x, str) and x for x in value['camera_names']), 'concrete camera_names required')
    require(isinstance(value.get('task_prompt'), str) and value['task_prompt'].strip(), 'concrete task_prompt required')
    public(value)
    return value


def numeric(value, depth=0):
    require(depth <= 4, 'public numeric observation too deep')
    if type(value) in (int, float):
        require(math.isfinite(value), 'nonfinite public observation')
        return value
    if isinstance(value, dict) and set(value) == {'dtype', 'shape'} and value['dtype'] in ('float64', 'float32', 'int64', 'int32'):
        return {'dtype': value['dtype'], 'shape': numeric(value['shape'], depth + 1)}
    require(isinstance(value, (list, tuple)) and len(value) <= 256, 'public numeric observation required')
    return [numeric(v, depth + 1) for v in value]


def scene_matches(reference, actual, path='state'):
    """Frozen scene comparator semantics: 1e-6, including quaternion sign."""
    if isinstance(reference, dict):
        return (isinstance(actual, dict) and set(reference) == set(actual)
                and all(scene_matches(v, actual[k], path + '.' + k) for k, v in reference.items()))
    if isinstance(reference, list):
        if not isinstance(actual, list) or len(reference) != len(actual):
            return False
        if 'quat' in path and len(reference) == 4 and sum(x * y for x, y in zip(reference, actual)) < 0:
            actual = [-x for x in actual]
        return all(scene_matches(x, y, path + f'[{i}]') for i, (x, y) in enumerate(zip(reference, actual)))
    return type(reference) in (int, float) and type(actual) in (int, float) and math.isfinite(reference - actual) and abs(reference - actual) <= 1e-6


def public_trace_projection(trace):
    """The Dev v2 public projection, with no native result as an argument."""
    public(trace)
    events = []
    for event in trace.get('sdk_events', []):
        operation = event.get('operation')
        if operation not in {'get_observation', 'find_objects', 'move_delta', 'open', 'close', 'move_normalized', 'move_joints'}:
            continue
        projected = {'operation': operation, 'status': event.get('status', 'unknown')}
        if event.get('source') in {'left_arm', 'right_arm', 'left_gripper', 'right_gripper', 'bimanual', 'sensors'}:
            projected['source'] = event['source']
        if event.get('physical_arm') in {'left', 'right', 'both'}:
            projected['physical_arm'] = event['physical_arm']
        if projected['status'] not in ('completed', 'failed', 'started', 'succeeded', 'unknown'):
            projected['status'] = 'unknown'
        projected['arguments'] = {k: numeric(v) for k, v in (event.get('arguments') or {}).items()
                                  if k in {'dx', 'dy', 'dz', 'rotation_delta', 'settle_steps', 'values', 'command', 'target_qpos'}}
        error = event.get('error')
        if error:
            projected['error'] = {k: str(error[k])[:1200] for k in ('type', 'message') if k in error} if isinstance(error, dict) else {'message': str(error)[:1200]}
        result = event.get('result')
        if operation == 'get_observation' and isinstance(result, dict):
            projected['observation'] = {k: numeric(v) for k, v in result.items() if re.fullmatch(r'robot0_(?:right_|left_)?(?:eef_pos|eef_quat|joint_pos|gripper_qpos|base_pos|base_quat|base_to_eef_pos|base_to_eef_quat)', k)}
        if operation == 'find_objects':
            objects = result.get('objects', []) if isinstance(result, dict) else result if isinstance(result, list) else []
            projected['objects'] = [{'name': str(o['name'])[:80], 'position': numeric(o['position']), 'frame': 'world'} for o in objects[:16] if {'name', 'position'} <= set(o)]
        events.append(projected)
    return {'schema_version': 'attentionbench.dev-public-feedback.v1', 'execution_status': trace['status'],
            'public_error': str(trace.get('error') or '')[:1200], 'action_trace': events[-40:]}


class DevV2Evidence:
    def __init__(self, store, catalogs=None):
        self.store = store
        with sqlite3.connect(store.path) as db:
            db.executescript('CREATE TABLE IF NOT EXISTS dev_v2_catalogs(id TEXT PRIMARY KEY,payload TEXT NOT NULL);'
                             'CREATE TABLE IF NOT EXISTS dev_v2_imports(id TEXT PRIMARY KEY,kind TEXT NOT NULL,attempt_id TEXT UNIQUE,payload TEXT NOT NULL);'
                             'CREATE TABLE IF NOT EXISTS dev_v2_validation_grants(id TEXT PRIMARY KEY,memory_id TEXT,cell TEXT,version INTEGER,payload TEXT,UNIQUE(memory_id,cell,version));')
        for catalog_id, catalog in (catalogs or {}).items():
            self.register(catalog_id, catalog)

    def register(self, catalog_id, catalog):
        """Startup configuration only: no registration endpoint or online method."""
        require(isinstance(catalog_id, str) and catalog_id, 'catalog ID required')
        require(catalog['role'] in {'executor', 'validator'}, 'catalog role required')
        root = Path(catalog['root'])
        require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'catalog needs a canonical existing root')
        scope(catalog['context'])
        require(isinstance(catalog.get('files'), dict) and catalog['files'], 'sealed file list required')
        for name, sha in catalog['files'].items():
            self._path(catalog, name)
            require(isinstance(sha, str) and SHA.fullmatch(sha), 'catalog digest required')
        self._put('dev_v2_catalogs', catalog_id, catalog)

    @staticmethod
    def _path(catalog, name):
        relative = Path(name)
        require(isinstance(name, str) and name and not relative.is_absolute() and '..' not in relative.parts, 'unsafe evidence path')
        root = Path(catalog['root'])
        path = root / relative
        require(path.resolve().is_relative_to(root), 'evidence escapes registered root')
        canonical = path.resolve()
        if any(p.is_symlink() for p in (*path.parents, path)):
            target = str(canonical.relative_to(root))
            require(catalog.get('symlinks', {}).get(name) == target and catalog['files'].get(target) == catalog['files'].get(name), 'symlink evidence needs an explicit pinned target in the same root')
        require(canonical.is_file(), 'registered evidence file missing')
        return canonical

    def catalog(self, catalog_id, role=None):
        state = _OPERATION.get()
        key = ('catalog', str(self.store.path), catalog_id)
        if state is not None and key in state['memo']:
            value = state['memo'][key]
        else:
            with sqlite3.connect(self.store.path) as db:
                row = db.execute('SELECT payload FROM dev_v2_catalogs WHERE id=?', (catalog_id,)).fetchone()
            require(row is not None, 'unregistered evidence catalog')
            value = json.loads(row[0])
            if state is not None:
                state['memo'][key] = value
        require(role is None or value['role'] == role, 'wrong evidence catalog role')
        return value

    def read(self, catalog_id, name, *, raw=False):
        catalog = self.catalog(catalog_id)
        require(name in catalog['files'], 'file is not in preregistered source list')
        path = self._path(catalog, name)
        state = _OPERATION.get()
        key = ('bytes', str(self.store.path), catalog_id, name)
        if state is not None and key in state['memo']:
            data = state['memo'][key]
            return data if raw else json.loads(data)
        identity = _identity(path)
        data = path.read_bytes()
        require(digest(data) == catalog['files'][name], 'registered evidence digest changed')
        if state is not None:
            state['memo'][key] = data
            state['files'][key] = (catalog, name, path, identity)
        return data if raw else json.loads(data)

    def ref(self, catalog_id, ref, *, raw=False):
        require(isinstance(ref, dict) and set(ref) == {'uri', 'sha256'}, 'exact artifact reference required')
        catalog = self.catalog(catalog_id)
        path = Path(ref['uri'])
        require(path.is_absolute() and path.is_relative_to(Path(catalog['root'])), 'artifact outside registered root')
        name = str(path.relative_to(catalog['root']))
        require(catalog['files'].get(name) == ref['sha256'], 'artifact is not preregistered at exact digest')
        return self.read(catalog_id, name, raw=raw)

    def verify_file(self, catalog_id, ref):
        catalog = self.catalog(catalog_id)
        path = Path(ref['uri'])
        require(path.is_absolute() and path.is_relative_to(catalog['root']), 'locked source outside registered root')
        relative = str(path.relative_to(catalog['root']))
        require(catalog['files'].get(relative) == ref['sha256'], 'locked source not preregistered')
        canonical = self._path(catalog, relative)
        state = _OPERATION.get()
        key = ('file', str(canonical), ref['sha256'])
        if state is not None and key in state['files']:
            return
        identity = _identity(canonical)
        sha = hashlib.sha256()
        with canonical.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                sha.update(block)
        require(sha.hexdigest() == ref['sha256'], 'locked source digest changed')
        if state is not None:
            state['files'][key] = (catalog, relative, canonical, identity)

    def _put(self, table, key, value, *, kind=None, attempt_id=None):
        require(isinstance(key, str) and key.strip(), 'nonempty immutable ID required')
        payload = encoded(value)
        with sqlite3.connect(self.store.path) as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute(f'SELECT payload FROM {table} WHERE id=?', (key,)).fetchone()
            if old:
                require(old[0] == payload, 'immutable import/catalog ID conflict')
                return
            try:
                if table == 'dev_v2_imports':
                    db.execute('INSERT INTO dev_v2_imports VALUES (?,?,?,?)', (key, kind, attempt_id, payload))
                else:
                    db.execute('INSERT INTO dev_v2_catalogs VALUES (?,?)', (key, payload))
            except sqlite3.IntegrityError as exc:
                raise StateConflictError('attempt already belongs to another import') from exc

    def get(self, import_id):
        with sqlite3.connect(self.store.path) as db:
            row = db.execute('SELECT payload FROM dev_v2_imports WHERE id=?', (import_id,)).fetchone()
        require(row is not None, 'unknown Dev v2 import')
        return json.loads(row[0])

    @contextmanager
    def _ledger(self, catalog_id, descriptor):
        # Read an immutable SQLite backup, never the online store or a caller
        # supplied ledger projection. A private copy gives one verified snapshot.
        data = self.ref(catalog_id, descriptor['ledger'], raw=True)
        with tempfile.TemporaryDirectory(prefix='memory-ledger-') as directory:
            path = Path(directory) / 'snapshot.sqlite'
            path.write_bytes(data)
            db = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)
            db.execute('PRAGMA query_only=ON')
            try:
                yield db
            finally:
                db.close()

    @staticmethod
    def _row(db, table, where, args):
        row = db.execute(f'SELECT payload FROM {table} WHERE {where}', args).fetchone()
        require(row is not None, 'original ledger lineage missing: ' + table)
        return json.loads(row[0])

    def _job(self, db, job_id, cell, version, kind):
        row = db.execute('SELECT cell,version,kind,state,payload FROM v2_jobs WHERE id=?', (job_id,)).fetchone()
        require(row is not None and row[:4] == (cell, version, kind, 'settled'), 'job must be exact settled original ledger job')
        job = json.loads(row[4])
        require(job['job_id'] == job_id and job['cell'] == cell and job['version'] == version and job['kind'] == kind, 'job payload/columns disagree')
        require(all(type(job.get(k)) in (int, float) and math.isfinite(job[k]) for k in ('claimed_at', 'settled_at')) and job['settled_at'] >= job['claimed_at'], 'job lacks real claim/settlement')
        require(isinstance(job.get('actual'), dict) and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in job['actual'].values()), 'actual costs unknown/invalid')
        return job

    def _base(self, catalog_id, d):
        catalog = self.catalog(catalog_id, 'executor')
        require(not {'authoritative', 'native_success', 'trusted'} & set(d), 'self-declared authority/outcome/status is forbidden')
        require(d['schema_version'] == 'attentionbench.dev-v2-import.v1', 'unsupported Dev import descriptor')
        require(type(d['version']) is int and d['version'] > 0, 'exact integer version required')
        require(d['context'] == catalog['context'], 'descriptor scope/namespace differs from registered execution scope')
        scope(d['context'])
        manifest = self.ref(catalog_id, d['manifest'])
        config = self.ref(catalog_id, d['config'])
        record = self.ref(catalog_id, d['version_record'])
        source = self.ref(catalog_id, record['source'], raw=True)
        require(type(record['version']) is int and record['version'] == d['version'] and record['formal_eligible'] is False and record['valid'] is True, 'invalid adopted code version')
        require(record['config_sha256'] == d['config']['sha256'] and record['manifest'] == d['manifest'], 'version/config/manifest binding mismatch')
        require(manifest.get('schema') == 'attentionbench.dev-experiment.v3' and manifest['ledger_path'] == catalog['ledger_origin'], 'registered original ledger/manifest binding mismatch')
        require(config == manifest['configs'][f"{config['task_id']}:{config['seed']}"], 'config differs from frozen manifest')
        for key in ('task_id', 'robot', 'controller', 'perception_mode', 'scene_id', 'object_set_id', 'camera_names', 'task_prompt'):
            require(config.get(key) == d['context'][key], 'config scope mismatch: ' + key)
        camera = {k: config[k] for k in ('camera_names', 'camera_width', 'camera_height', 'randomize_cameras')}
        camera_id = 'robocasa365:' + digest(json.dumps(camera, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())
        variant = config['official_env'] + ':layout' + str(config['layout_id']) + ':style' + str(config['style_id']) + ':' + config['base_mode']
        require(d['context']['camera_config_id'] == camera_id and d['context']['task_variant_id'] == variant, 'derived exact camera/task variant mismatch')
        with self._ledger(catalog_id, d) as db:
            cell = self._row(db, 'v2_cells', 'id=?', (d['cell'],))
            experiment = self._row(db, 'v2_experiments', 'id=?', (cell['experiment_id'],))
            version = self._row(db, 'v2_versions', 'cell=? AND version=?', (d['cell'], d['version']))
            require(version == record and experiment['manifest'] == d['manifest'] and manifest['experiment_id'] == cell['experiment_id'], 'ledger version/experiment differs from original artifacts')
            require(any(c['key'] == d['cell'] and c['task_id'] == config['task_id'] and c['seed'] == config['seed'] and c['policy_id'] == cell['policy_id'] for c in manifest['cells']), 'cell absent from frozen manifest')
        # Source/SDK identity bytes must be included in the registered snapshot.
        identity = manifest['identity_lock']
        lock_ref = {'uri': identity['path'], 'sha256': identity['sha256']}
        lock = self.ref(catalog_id, lock_ref)
        require(lock['schema'] == 'attentionbench.dev-identity-lock.v2' and lock['formal_eligible'] is False, 'frozen Dev identity lock required')
        require(lock.get('source_sets'), 'complete SDK source file sets required')
        for folder, expected in lock['source_sets'].items():
            require(Path(folder).is_relative_to(catalog['root']) and Path(folder).resolve() == Path(folder) and _source_set(folder) == expected and all(name in lock['files'] for name in expected), 'SDK source file set drift')
            if _OPERATION.get() is not None:
                _OPERATION.get()['directories'][folder] = expected
        require(lock['files'].get(lock['planner_lock']['path']) == lock['planner_lock']['sha256'] and lock['model_sources'] and all(lock['files'].get(n) == sha for n, sha in lock['model_sources'].items()), 'planner/robot source identity incomplete')
        for name, sha in lock['files'].items():
            self.verify_file(catalog_id, {'uri': name, 'sha256': sha})
        sdk = d['sdk_identity']
        require(digest(json.dumps(sdk, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()) == d['context']['sdk_sha256'], 'SDK identity does not match exact context digest')
        require(sdk['planner_lock'] == lock['planner_lock']['sha256'] and sdk['robot_model_identity'] == lock['model_sources'], 'SDK identity lock mismatch')
        expected_files = {str(Path(n).relative_to(lock['source_root'])): sha for n, sha in lock['files'].items()
                          if Path(n).is_relative_to(lock['source_root']) and any(part in Path(n).parts for part in ('openarm365', 'openarm365_dev', 'openarm365_dev_v2', 'curobo_v2')) and '/tests/' not in n}
        require(expected_files and sdk['implementation'] == expected_files, 'SDK implementation source list mismatch')
        return catalog, manifest, config, record, source, cell

    def _provider(self, catalog_id, job):
        receipt = job['receipt']
        request = self.ref(catalog_id, receipt['request'])
        output = self.ref(catalog_id, receipt['response'])
        wire = self.ref(catalog_id, receipt['provider_response'])
        usage = wire['body']['usage']
        require(wire['http_status'] == 200 and isinstance(wire.get('request_id') or wire['body'].get('id'), str) and (wire.get('request_id') or wire['body'].get('id')), 'successful original provider response/request ID required')
        require(all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('total_tokens', 'prompt_tokens', 'completion_tokens')) and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'complete provider usage required')
        require(receipt['usage'] == usage and output['usage'] == usage and output['content'] == wire['body']['choices'][0]['message']['content'], 'provider output/usage differs from original wire')
        require(all(job['actual'].get(k) == usage[v] for k, v in [('provider_tokens', 'total_tokens'), ('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens')]), 'ledger provider costs disagree')
        for message in request['messages']:
            if message['role'] == 'user':
                public(json.loads(message['content']))
        return request, output, receipt

    def verify_execution(self, catalog_id, descriptor):
        d = self.read(catalog_id, descriptor)
        catalog, manifest, config, record, source, cell = self._base(catalog_id, d)
        with self._ledger(catalog_id, d) as db:
            job = self._job(db, d['execute_job_id'], d['cell'], d['version'], 'execute')
        bundle = self.ref(catalog_id, job['receipt']['bundle'])
        require(bundle['schema_version'] == 'attentionbench.formal-runner-result.v1' and bundle['channel'] == 'dev_v2' and bundle['formal_eligible'] is False and type(bundle['dev_version']) is int and bundle['dev_version'] == d['version'], 'unsupported real Dev v2 bundle')
        require(bundle['policy_sha256'] == record['source']['sha256'] and bundle['config_sha256'] == d['config']['sha256'] and bundle['config_lock'] == d['config'] and bundle['policy_id'] == cell['policy_id'], 'bundle/code/config binding mismatch')
        require(bundle['run_id'] == 'run:' + d['cell'] and bundle['attempt_id'] == f"attempt:{d['cell']}:{d['version']-1}" and bundle['task_id'] == config['task_id'] and bundle['seed'] == config['seed'] and bundle['suite'] == d['context']['suite'], 'bundle identity/scope mismatch')
        require(bundle['status'] in {'completed', 'failed', 'timeout', 'cancelled', 'condition_invalid'}, 'nonterminal native bundle')
        artifacts = {key: self.ref(catalog_id, bundle['artifacts'][key]) for key in ('trace', 'native_result', 'sandbox_receipt', 'safety')}
        for key, value in artifacts.items():
            require(value['run_id'] == bundle['run_id'] and value['attempt_id'] == bundle['attempt_id'], 'raw native artifact ID mismatch: ' + key)
        trace, native, receipt = (artifacts[k] for k in ('trace', 'native_result', 'sandbox_receipt'))
        require(trace['policy_sha256'] == bundle['policy_sha256'] and trace['config_sha256'] == bundle['config_sha256'] and trace['status'] == bundle['status'] and native['status'] == bundle['status'], 'raw trace/native binding mismatch')
        require(native == bundle['native_evaluator'] and native['native_success'] == bundle['native_success'], 'native result differs from original bundle')
        for key in ('trace', 'native_result'):
            require(job['receipt'][key] == bundle['artifacts'][key], 'ledger raw receipt mismatch')
        require(type(receipt['sdk_call_budget']['dispatched']) is int and 0 <= receipt['sdk_call_budget']['dispatched'] <= min(80, config['sdk_call_limit']) and type(receipt['control_steps']) is int and 0 <= receipt['control_steps'] <= min(2400, config['horizon']) and type(receipt['elapsed_seconds']) in (int, float) and math.isfinite(receipt['elapsed_seconds']) and 0 <= receipt['elapsed_seconds'] <= min(480, config['attempt_wall_seconds']) + 60, 'native resource receipt exceeds frozen bounds')
        require(job['receipt']['execution_status'] == bundle['status'] and job['actual'].get('executions') == 1 and job['actual'].get('sdk_calls') == receipt['sdk_call_budget']['dispatched'] and job['actual'].get('control_steps') == receipt['control_steps'] and job['actual'].get('execution_seconds') == receipt['elapsed_seconds'], 'native actual costs differ from original receipt')
        permission = d['permission_scope']
        require(all(permission.get(k) == v for k, v in {'run_id': bundle['run_id'], 'cell': d['cell'], 'code_version': d['version'], 'code_sha256': bundle['policy_sha256'], 'config_sha256': digest(json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())}.items()), 'permission scope not bound to exact run/code/config/version')
        require(type(permission.get('code_version')) is int and type(permission.get('action_version')) is int and permission['action_version'] == 1 and isinstance(permission.get('scope'), dict) and permission['scope'], 'explicit permission action/version/scope required')
        if permission.get('action') == 'execute_bounded_motion_policy':
            with self._ledger(catalog_id, d) as db:
                approval = self._row(db, 'v2_planner_requests', 'id=?', (d['approval_id'],))
            require(approval['request_type'] == 'approval' and approval['state'] == 'consumed' and approval['binding'] == permission and approval['consumed_by'] == job['job_id'] and approval['created_at'] <= approval['consumed_at'] <= job['claimed_at'] and approval['consumed_at'] <= approval['deadline_epoch'], 'native motion lacks exact consumed human approval receipt')
            require(approval['response']['approved'] is True and approval['response']['actor'] == 'authenticated_human' and approval['response']['response_id'] and approval['created_at'] <= approval['responded_at'] <= approval['consumed_at'], 'genuine external human decision receipt required')
        else:
            require(permission.get('action') == 'sensor_only', 'unknown permission action')
        feedback = self.ref(catalog_id, job['receipt']['public_feedback']); public(feedback)
        projected = ({'schema_version': 'attentionbench.dev-public-feedback.v1', 'execution_status': 'condition_invalid',
                      'public_error': 'scene_contract_mismatch', 'action_trace': []}
                     if bundle['status'] == 'condition_invalid' else public_trace_projection(trace))
        for label, ref in d.get('public_observations', {}).items():
            require(label in {'initial', 'final'}, 'unknown public observation label')
            observation = self.ref(catalog_id, ref); public(observation)
            projected[label + '_observation'] = {'robot': {k: numeric(v) for k, v in observation['robot'].items()},
                'objects': {k: numeric(v) for k, v in observation['perception']['values'].items()}}
        require(all(feedback.get(k) == v for k, v in projected.items()) and not set(feedback) - {*projected, 'evidence_kind'}, 'public feedback differs from original trace projection')
        if 'scene_contract' in config:
            contract = self.ref(catalog_id, config['scene_contract'])
            original = self.ref(catalog_id, d['scene_state'])
            condition = bundle['scene_condition']
            require(contract['schema'] == 'attentionbench.public-scene-contract.v1' and contract['tolerance'] == 1e-6, 'frozen scene contract/tolerance required')
            require(original['validation'] == condition and condition['contract'] == config['scene_contract'] and condition['task_id'] == config['task_id'] and condition['tolerance'] == 1e-6, 'original scene receipt/contract binding mismatch')
            require(type(condition['matched']) is bool and condition['matched'] == scene_matches(contract['tasks'][config['task_id']]['state'], original['state']) and condition['raw_sha256'] == digest(encoded(original['state']).encode()), 'original scene verdict differs from raw state')
        evidence = {'catalog_id': catalog_id, 'descriptor': descriptor, 'context': d['context'], 'cell': d['cell'], 'version': d['version'], 'bundle': job['receipt']['bundle'], 'public_feedback': job['receipt']['public_feedback'], 'source': record['source'], 'config': d['config'], 'manifest': d['manifest'], 'ledger': d['ledger'], 'run_id': bundle['run_id'], 'attempt_id': bundle['attempt_id'], 'status': bundle['status']}
        return evidence, d, bundle, artifacts, cell

    def import_execution(self, *, import_id, catalog_id, descriptor):
        evidence, *_ = self.verify_execution(catalog_id, descriptor)
        self._put('dev_v2_imports', import_id, {'kind': 'execution', **evidence}, kind='execution', attempt_id=evidence['attempt_id'])
        return self.get(import_id)

    @memo_evidence('execution')
    def execution(self, import_id):
        value = self.get(import_id)
        require(value['kind'] == 'execution', 'execution import required')
        checked, d, bundle, artifacts, cell = self.verify_execution(value['catalog_id'], value['descriptor'])
        require(value == {'kind': 'execution', **checked}, 'imported execution changed')
        return value, d, bundle, artifacts, cell

    def verdict(self, import_id, validator_id, descriptor, *, for_validation=True):
        execution, d, bundle, artifacts, cell = self.execution(import_id)
        validator = self.catalog(validator_id, 'validator')
        executor = self.catalog(execution['catalog_id'], 'executor')
        require(not Path(validator['root']).is_relative_to(executor['root']) and not Path(executor['root']).is_relative_to(validator['root']), 'validator root must be independent of executor')
        require(validator['context'] == execution['context'], 'validator scope/namespace mismatch')
        v = self.read(validator_id, descriptor)
        require(v['schema_version'] == 'attentionbench.dev-v2-independent-validation.v1' and v['purpose'] == 'offline_memory_validation' and v['execution_import_id'] == import_id, 'independent offline validator receipt required')
        bindings = {k: bundle[k] for k in ('run_id', 'attempt_id', 'policy_sha256', 'config_sha256', 'task_id', 'seed')}
        require(v['bindings'] == bindings and v['bundle'] == execution['bundle'] and v['native_result'] == bundle['artifacts']['native_result'] and v['trace'] == bundle['artifacts']['trace'], 'independent verdict raw SHA/ID mismatch')
        require(not bundle['infrastructure_failure'] and not bundle['outcome_unknown'] and bundle['status'] in {'completed', 'failed'}, 'infrastructure/unknown/non-evaluated outcome cannot validate memory')
        require(bundle['sandbox'] == {'process_isolated': True, 'sdk_rpc_only': True, 'deadline_enforced': True, 'action_cancellation_verified': True}, 'original sandbox/cleanup conditions did not pass')
        require(bundle['scene_condition'] and bundle['scene_condition']['matched'] is True and artifacts['native_result']['evaluated'] is True and type(bundle['native_success']) is bool, 'original native/scene gate did not pass')
        checks = v['checks']
        require(v['native_success'] is bundle['native_success'] and set(checks) == {'native_evaluation', 'safety', 'scene_reproduction', 'permissions', 'source_identity'} and all(type(x) is bool for x in checks.values()) and all(checks[k] for k in ('native_evaluation', 'permissions', 'source_identity')), 'independent checks incomplete')
        if for_validation:
            require(checks['safety'] and checks['scene_reproduction'], 'independent safety/scene reproduction did not pass')
        # Independent replay observations are original validator-owned artifacts.
        observations = [self.ref(validator_id, r) for r in v['scene_replay_observations']]
        require(v['scene_replay_bindings'] == {'task_id': bundle['task_id'], 'seed': bundle['seed'], 'config_sha256': bundle['config_sha256']} and len(observations) == 2 and v['scene_replay_observations'][0]['uri'] != v['scene_replay_observations'][1]['uri'] and all(isinstance(o, dict) and isinstance(o.get('robot'), dict) and isinstance(o.get('perception'), dict) for o in observations), 'scene reproduction original evidence missing')
        if checks['scene_reproduction']:
            require(scene_matches({'robot': observations[0]['robot'], 'objects': observations[0]['perception']['values']}, {'robot': observations[1]['robot'], 'objects': observations[1]['perception']['values']}), 'same-seed scene reproduction evidence failed')
            config = self.ref(execution['catalog_id'], execution['config'])
            if 'scene_contract' in config:
                contract = self.ref(execution['catalog_id'], config['scene_contract'])
                states = [self.ref(validator_id, ref) for ref in v['scene_replay_states']]
                require(len(states) == 2 and v['scene_replay_states'][0]['uri'] != v['scene_replay_states'][1]['uri'] and all(scene_matches(contract['tasks'][bundle['task_id']]['state'], state) for state in states), 'full fixture/robot/object scene reproduction failed')
        safety_path = self._path(validator, str(Path(v['safety']['uri']).relative_to(validator['root'])))
        self.ref(validator_id, v['safety'])
        from .memory_v2 import MemoryV2Manager
        MemoryV2Manager._safety(safety_path, bundle['attempt_id'])
        require(isinstance(v['permission_scope'], dict) and v['permission_scope'] == d['permission_scope'] and v['permission_scope'], 'permission scope not bound to trusted validation')
        return execution, d, bundle, artifacts, cell, v, safety_path

    def import_source(self, *, import_id, catalog_id, descriptor):
        d = self.read(catalog_id, descriptor)
        _, manifest, config, record, repair, cell = self._base(catalog_id, d)
        failed, fd, bundle, artifacts, _, verdict, _ = self.verdict(d['failed_execution_import_id'], d['validator_id'], d['validation_descriptor'], for_validation=False)
        require(bundle['native_success'] is False and failed['cell'] == d['cell'] and failed['version'] == d['version'] - 1 and failed['context'] == d['context'], 'source must be previous independent native-confirmed failed attempt in exact scope')
        with self._ledger(catalog_id, d) as db:
            advisor = self._job(db, d['advisor_job_id'], d['cell'], d['version'], 'advisor')
            model = self._job(db, d['model_job_id'], d['cell'], d['version'], 'model')
            row = self._row(db, 'v2_planner_requests', 'id=?', (d['request_id'],))
        ar, ao, ac = self._provider(catalog_id, advisor)
        mr, mo, mc = self._provider(catalog_id, model)
        help_value = self.ref(catalog_id, record['help'])
        require(help_value['provider'] == 'advisor' and not help_value.get('rejected') and help_value['content'] == ao['content'] and help_value['request'] == ac['request'] and help_value['response'] == ac['response'] and help_value['usage'] == ac['usage'], 'answered original Advisor guidance required')
        require(isinstance(ao['content'], str) and ao['content'].strip(), 'nonempty answered guidance required')
        require(row['binding'] == {'cell': d['cell'], 'version': d['version'], 'input_sha256': digest(json.dumps(ar['messages'], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()), 'source_sha256': failed['source']['sha256'], 'policy_id': cell['policy_id'], 'config_sha256': d['config']['sha256']}, 'Advisor request binding differs from original source/input/scope')
        require(advisor['claimed_at'] <= row['created_at'] <= advisor['settled_at'] <= row['responded_at'] <= row['deadline_epoch'] and row['responded_at'] <= model['claimed_at'], 'Advisor response late or not available before actual adoption')
        require(advisor['actual'].get('advisor_calls') == 1 and advisor['actual'].get('help_count') == 1 and advisor['actual'].get('help_credits') == 1 and model['actual'].get('dev_calls') == 1, 'committed assistance/model cost required')
        require(row['cell'] == d['cell'] and row['version'] == d['version'] and row['request_type'] == 'advisor' and row['state'] == 'answered' and row['accepted'] is True and row['adopted'] is True and row['response'] == ac['response'] and row['adoption'] == {'model_input': mc['request'], 'result_source': record['source']}, 'answered/adopted request must bind actual generation')
        require(record['model_input'] == mc['request'] and record['model_output'] == mc['response'], 'model input/output lineage mismatch')
        packet = json.loads(mr['messages'][-1]['content'])
        advisor_packet = json.loads(ar['messages'][-1]['content'])
        expected_source = self.ref(failed['catalog_id'], failed['source'], raw=True).decode()
        expected_feedback = self.ref(failed['catalog_id'], failed['public_feedback'])
        require(packet['previous_complete_source'] == expected_source and packet['public_feedback'] == expected_feedback and advisor_packet['previous_complete_source'] == expected_source and advisor_packet['public_feedback'] == expected_feedback, 'online help source/public failed trace mismatch')
        require(packet['attention_control']['help']['provider'] == 'advisor' and packet['attention_control']['help']['content'] == ao['content'], 'guidance not actually present in generation input')
        generated = mo['content'].strip()
        if generated.startswith('```'):
            generated = re.sub(r'^```(?:json)?\s*|\s*```$', '', generated)
        require(json.loads(generated)['source'].encode() == repair, 'adopted source differs from actual model output')
        public(ao['content']); public(repair.decode())
        evidence = {'kind': 'source', 'catalog_id': catalog_id, 'descriptor': descriptor, 'failed_execution_import_id': d['failed_execution_import_id'], 'context': d['context'], 'request_id': d['request_id'], 'cell': d['cell'], 'version': d['version'], 'guidance': ao['content'], 'repair': repair.decode(), 'source_policy_sha256': failed['source']['sha256'], 'repair_sha256': record['source']['sha256'], 'source_policy_id': cell['policy_id'], 'source_seed': config['seed'], 'source_run_id': bundle['run_id'], 'source_attempt_id': bundle['attempt_id']}
        self._put('dev_v2_imports', import_id, evidence, kind='source')
        return evidence

    @memo_evidence('source')
    def source(self, import_id):
        old = self.get(import_id)
        require(old['kind'] == 'source', 'source import required')
        return self.import_source(import_id=import_id, catalog_id=old['catalog_id'], descriptor=old['descriptor'])

    def create_candidate(self, *, memory_id, source_import_id, created_at):
        require(type(created_at) in (int, float) and math.isfinite(created_at), 'finite candidate timestamp required')
        s = self.source(source_import_id)
        artifact = {'schema_version': 'attentionbench.memory-artifact.v2', 'kind': 'policy_patch', 'guidance': s['guidance'], 'repair': s['repair'], 'applicability': {k: s['context'][k] for k in SCOPE}}
        provenance = {'schema_version': 'attentionbench.memory-provenance.v2', 'channel': 'dev_v2', 'source_import_id': source_import_id, 'memory_id': memory_id, 'source_kind': 'advisor_proxy', 'human_attention_seconds': 0.0, 'assistance_credits': 1, 'source_trace_id': 'dev-v2-source:' + source_import_id, 'source_raw_trace_id': 'dev-v2-source:' + source_import_id, 'raw_evidence_refs': [], 'artifact': artifact, 'artifact_sha256': digest(encoded(artifact).encode()), **{k: s[k] for k in ('source_policy_id', 'source_policy_sha256', 'source_seed', 'source_run_id', 'source_attempt_id')}}
        candidate = MemoryRecord(memory_id=memory_id, version=1, source_trace_id=provenance['source_trace_id'], guidance=s['guidance'], candidate_repair=s['repair'], applicability=artifact['applicability'], evidence_refs=('dev-v2-source:' + source_import_id,), created_at=created_at)
        # Commit verified external provenance and candidate atomically. No legacy
        # trace row is invented to satisfy AttentionStore.create_memory.
        with self.store._transaction() as db:
            existing = db.execute('SELECT payload FROM memory_v2_provenance WHERE memory_id=?', (memory_id,)).fetchone()
            if existing:
                require(json.loads(existing[0]) == provenance, 'conflicting candidate source')
                old_memory = self.store.get_memory(memory_id)
                require(old_memory.created_at == created_at, 'candidate creation timestamp conflict')
                return old_memory
            inserted = self.store._insert_immutable_tx(db, 'memories', memory_id, candidate.artifact())
            require(inserted, 'memory ID already belongs to another source')
            db.execute('INSERT INTO memory_v2_provenance VALUES (?,?)', (memory_id, encoded(provenance)))
            db.execute('INSERT INTO events(event_key,entity_type,entity_id,event_type,payload) VALUES (?,?,?,?,?)',
                       ('memory.created:' + memory_id, 'memories', memory_id, 'memory.created', encoded(candidate.artifact())))
        return candidate

    def authorize_validation(self, *, memory_id, cell, version, seed, context,
                             policy_sha256, config_sha256, exposure):
        from .memory_v2 import MemoryV2Manager
        manager = MemoryV2Manager(self.store)
        provenance = manager.provenance(memory_id)
        memory = self.store.get_memory(memory_id)
        require(provenance.get('channel') == 'dev_v2' and memory.status.value == 'candidate', 'offline validation grant requires a Dev v2 candidate')
        plan = manager.get_plan(memory_id)
        require(plan is not None, 'freeze a validation plan before dispatch')
        scope(context)
        require(type(version) is int and version > 0 and isinstance(cell, str) and cell and type(seed) is int, 'exact cell/version/seed required')
        require(exposure in {'none', 'candidate_validation'} and all(isinstance(x, str) and SHA.fullmatch(x) for x in (policy_sha256, config_sha256)), 'explicit exposure and code/config digests required')
        require(all(context[k] == memory.applicability[k] for k in SCOPE if k not in AXES), 'validation grant exact SDK/namespace/robot scope mismatch')
        require(any(case['seed'] == seed and all(case[k] == context[k] for k in (*AXES, 'camera_names', 'task_prompt')) for case in plan['cases']), 'validation grant outside frozen concrete case')
        binding = dict(memory_id=memory_id, memory_version=memory.version, cell=cell, version=version, seed=seed,
                       context=context, policy_sha256=policy_sha256, config_sha256=config_sha256,
                       exposure=exposure, plan_sha256=digest(encoded(plan).encode()))
        with sqlite3.connect(self.store.path) as db:
            db.execute('BEGIN IMMEDIATE')
            current = json.loads(db.execute('SELECT payload FROM memories WHERE id=?', (memory_id,)).fetchone()[0])
            require(current['status'] == 'candidate' and current['version'] == memory.version, 'candidate revoked before validation grant')
            old = db.execute('SELECT payload FROM dev_v2_validation_grants WHERE memory_id=? AND cell=? AND version=?', (memory_id, cell, version)).fetchone()
            if old:
                grant = json.loads(old[0]); require(grant['binding'] == binding, 'validation dispatch already binds different evidence')
                return grant
            rows = db.execute('SELECT payload FROM dev_v2_validation_grants WHERE memory_id=?', (memory_id,)).fetchall()
            require(all(json.loads(row[0])['binding']['policy_sha256'] == policy_sha256 for row in rows), 'validation policy version is frozen across all cases')
            grant = {'schema_version': 'attentionbench.dev-v2-validation-grant.v1', 'grant_id': 'validation:' + uuid.uuid4().hex, 'issued_at': time.time(), 'binding': binding}
            db.execute('INSERT INTO dev_v2_validation_grants VALUES (?,?,?,?,?)', (grant['grant_id'], memory_id, cell, version, encoded(grant)))
        return grant

    def validation_grant(self, grant_id):
        with sqlite3.connect(self.store.path) as db:
            row = db.execute('SELECT payload FROM dev_v2_validation_grants WHERE id=?', (grant_id,)).fetchone()
        require(row is not None, 'unknown original offline validation grant')
        return json.loads(row[0])

    def import_trial(self, *, import_id, catalog_id, descriptor):
        self.catalog(catalog_id, 'validator')
        d = self.read(catalog_id, descriptor)
        execution, ed, bundle, artifacts, cell, verdict, safety = self.verdict(d['execution_import_id'], catalog_id, d['validation_descriptor'])
        require(d['schema_version'] == 'attentionbench.dev-v2-trial.v1', 'unsupported offline trial descriptor')
        exposure = d['exposure']
        require(exposure in {'none', 'candidate_validation'}, 'only offline candidate validation exposure supported')
        grant = self.validation_grant(d['validation_grant_id'])
        binding = grant['binding']
        from .memory_v2 import MemoryV2Manager
        manager = MemoryV2Manager(self.store)
        plan = manager.get_plan(binding['memory_id'])
        require(plan and digest(encoded(plan).encode()) == binding['plan_sha256'], 'original validation plan changed')
        require(all(binding[k] == v for k, v in {'cell': ed['cell'], 'version': ed['version'], 'seed': bundle['seed'], 'context': execution['context'], 'policy_sha256': bundle['policy_sha256'], 'config_sha256': bundle['config_sha256'], 'exposure': exposure}.items()), 'offline validation grant/raw run binding mismatch')
        require(exposure == 'none' or binding['memory_id'] == d['memory_id'], 'candidate validation grant memory mismatch')
        ids = [] if exposure == 'none' else [d['memory_id']]
        # Validator must bind exposure to the original executor's frozen input,
        # not infer treatment from the eventual outcome.
        input_value = self.ref(execution['catalog_id'], ed['validation_input'])
        require(input_value == {'memory_ids': ids, 'exposure': exposure, 'context': execution['context'], 'policy_sha256': bundle['policy_sha256'], 'config_sha256': bundle['config_sha256'], 'validation_grant_id': grant['grant_id']}, 'original validation exposure/config mismatch')
        candidate = self.store.get_memory(binding['memory_id'])
        expected_consumption = {'validation_grant_id': grant['grant_id'],
            'input_sha256': ed['validation_input']['sha256'], 'memory_ids': ids, 'exposure': exposure,
            'memory_version': binding['memory_version'] if ids else None,
            'guidance_sha256': digest(candidate.guidance.encode()) if ids else None,
            'consumed_before_policy': True}
        require(artifacts['trace'].get('memory_validation') == expected_consumption,
                'native trace lacks original candidate input consumption receipt')
        with self._ledger(execution['catalog_id'], ed) as db:
            job = self._job(db, ed['execute_job_id'], ed['cell'], ed['version'], 'execute')
            dispatches = [(at, json.loads(payload)) for at, payload in db.execute("SELECT at,payload FROM v2_events WHERE kind='memory_validation_dispatch'")]
        expected_dispatch = {'cell': ed['cell'], 'version': ed['version'], 'validation_input': ed['validation_input'], 'validation_grant_id': grant['grant_id'], 'source': execution['source'], 'config': execution['config']}
        require(len([(at, p) for at, p in dispatches if p == expected_dispatch and grant['issued_at'] <= at <= job['claimed_at']]) == 1, 'trial lacks an original pre-execution validation dispatch event')
        costs = d['costs']
        require(costs == verdict['costs'] and all(type(costs.get(k)) in (int, float) and math.isfinite(costs[k]) and costs[k] >= 0 for k in ('credits', 'seconds', 'gpu_seconds', 'tokens')), 'independent validation costs missing')
        with self._ledger(execution['catalog_id'], ed) as db:
            settled = [json.loads(p) for state, p in db.execute('SELECT state,payload FROM v2_jobs WHERE cell=? AND version<=?', (ed['cell'], ed['version'])) if state == 'settled']
        require(costs['credits'] == sum(j['actual'].get('help_credits', 0) for j in settled) and costs['tokens'] == sum(j['actual'].get('provider_tokens', 0) for j in settled), 'validation cost differs from original settled jobs')
        require(costs['seconds'] == artifacts['sandbox_receipt']['elapsed_seconds'], 'trial time differs from native receipt')
        trial = {'run_id': bundle['run_id'], 'run': {'suite': bundle['suite'], 'task_id': bundle['task_id'], 'seed': bundle['seed'], 'policy_id': cell['policy_id'], 'execution_target': 'simulator', 'developer_model': d['developer_model'], 'budget': d['budget']}, 'success': bundle['native_success'], 'attempt_status': 'succeeded' if bundle['native_success'] else 'failed', 'evaluator_authoritative': True, 'mode': execution['context']['perception_mode'], 'memory_ids': ids, 'exposure': exposure, 'policy_sha256': bundle['policy_sha256'], 'config_sha256': bundle['config_sha256'], 'variation': {k: execution['context'][k] for k in (*AXES, 'camera_names', 'task_prompt')}, 'context': execution['context'], **costs}
        # Shared comparison configuration is an actual preregistered file, with
        # no arm/seed/outcome fields. The per-episode config remains SHA-bound.
        comparison = self.ref(catalog_id, d['validation_config'])
        require(comparison == verdict['validation_config'] and comparison['execution_config'] == {k: v for k, v in self.ref(execution['catalog_id'], execution['config']).items() if k != 'seed'}, 'frozen paired configuration mismatch')
        require(d['budget'] == comparison['budget'] and d['developer_model'] == comparison['developer_model'], 'trial model/budget differs from frozen comparison')
        evidence = {'kind': 'trial', 'catalog_id': catalog_id, 'descriptor': descriptor, 'attempt_id': bundle['attempt_id'], 'trial': trial, 'safety': {'uri': str(safety), 'sha256': digest(safety.read_bytes()), 'attempt_id': bundle['attempt_id']}}
        self._put('dev_v2_imports', import_id, evidence, kind='trial')
        return evidence

    @memo_evidence('trial')
    def trial(self, attempt_id):
        with sqlite3.connect(self.store.path) as db:
            rows = db.execute("SELECT id,payload FROM dev_v2_imports WHERE kind='trial'").fetchall()
        for import_id, payload in rows:
            old = json.loads(payload)
            if old['attempt_id'] == attempt_id:
                return self.import_trial(import_id=import_id, catalog_id=old['catalog_id'], descriptor=old['descriptor'])['trial']
        return None

    def _result_evidence(self, grant, execution_import_id):
        execution, d, bundle, artifacts, cell = self.execution(execution_import_id)
        grant_id = grant['grant_id']
        require(grant['development_id'] == f"{d['cell']}:v{d['version']}" and grant['context'] == d['context'], 'grant belongs to another run/cell/version/scope')
        catalog_id = execution['catalog_id']
        record = self.ref(catalog_id, d['version_record'])
        with self._ledger(catalog_id, d) as db:
            model = self._job(db, d['model_job_id'], d['cell'], d['version'], 'model')
        request, output, receipt = self._provider(catalog_id, model)
        require(model['settled_at'] <= self._execution_claim_time(execution, d), 'native execution preceded settled source generation')
        require(record['model_input'] == receipt['request'] and record['model_output'] == receipt['response'] and model['actual'].get('dev_calls') == 1, 'result model lineage mismatch')
        generated = output['content'].strip()
        if generated.startswith('```'):
            generated = re.sub(r'^```(?:json)?\s*|\s*```$', '', generated)
        require(digest(json.loads(generated)['source'].encode()) == execution['source']['sha256'], 'result source is not actual model output')
        packet = json.loads(request['messages'][-1]['content'])
        require(digest(packet['previous_complete_source'].encode()) == grant['source_policy_sha256'], 'grant source code SHA mismatch')
        entries = packet['attention_control'].get('memory', [])
        require(isinstance(entries, list), 'Memory input must be exact entry list')
        # Some planners place memory on the decision itself; no substring search.
        matching = [entry for entry in entries if entry.get('grant_id') == grant_id]
        require(len(matching) == 1, 'exact grant absent from actual model generation input')
        entry = matching[0]
        require(type(entry.get('version')) is int, 'actual generation Memory version must be an exact integer')
        from .identity import store_id
        require(all(entry.get(k) == v for k, v in {'memory_id': grant['memory_id'], 'version': grant['memory_version'], 'content': grant['guidance'], 'context': grant['context'], 'development_id': grant['development_id'], 'source_policy_sha256': grant['source_policy_sha256'], 'authority': store_id(self.store.path)}.items()), 'actual generation input differs from issued grant')
        require(model['claimed_at'] >= grant['exposed_at'], 'grant was issued after model dispatch')
        # Record history after revocation without re-exposing guidance. Rechecks
        # before dispatch are still required; never count this as causal benefit.
        result = {'schema_version': 'attentionbench.dev-v2-memory-result.v1', 'execution_import_id': execution_import_id, 'run_id': execution['run_id'], 'cell': d['cell'], 'version': d['version'], 'source_policy_sha256': grant['source_policy_sha256'], 'result_policy_sha256': execution['source']['sha256'], 'bundle': execution['bundle'], 'ledger': execution['ledger'], 'model_input': receipt['request'], 'public_feedback': execution['public_feedback'], 'permission_scope': d['permission_scope'], 'claim': 'development_exposure_only_not_causal_impact'}
        return result

    def verify_result(self, grant):
        result = grant['result']
        require(result == self._result_evidence(grant, result['execution_import_id']), 'recorded Dev result evidence changed')

    def record_result(self, *, grant_id, execution_import_id):
        with sqlite3.connect(self.store.path) as db:
            row = db.execute('SELECT payload FROM memory_dev_exposures WHERE grant_id=?', (grant_id,)).fetchone()
        require(row is not None, 'unknown Dev Memory grant')
        grant = json.loads(row[0])
        result = self._result_evidence(grant, execution_import_id)
        with sqlite3.connect(self.store.path) as db:
            db.execute('BEGIN IMMEDIATE')
            current = json.loads(db.execute('SELECT payload FROM memory_dev_exposures WHERE grant_id=?', (grant_id,)).fetchone()[0])
            if current['result'] is not None:
                require(current['result'] == result, 'Dev Memory result is immutable')
                return current
            current['result'] = result
            db.execute('UPDATE memory_dev_exposures SET payload=? WHERE grant_id=?', (encoded(current), grant_id))
        return current

    def _execution_claim_time(self, execution, descriptor):
        with self._ledger(execution['catalog_id'], descriptor) as db:
            return self._job(db, descriptor['execute_job_id'], descriptor['cell'], descriptor['version'], 'execute')['claimed_at']
