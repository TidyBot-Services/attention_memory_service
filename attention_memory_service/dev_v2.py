"""Offline Dev v2 evidence adapter. Never synthesizes legacy run/request rows.

Trust is configured by the service operator, not in an online JSON payload.
Catalogs pin sealed executor/evaluator snapshots; the importer key is held only
by the trusted execution/validation side. All reads recheck the pinned bytes.
"""
from __future__ import annotations

import hashlib
import base64
import binascii
import io
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import math
import os
import re
import sqlite3
import stat
import time
import tempfile
import uuid
from pathlib import Path
from typing import Any

from .provider_evidence import verify_provider, frozen_provider, verify_codex_record
from .feedback_evidence import verify_feedback
from .failure_signature import REASON_CODE, planner_reason_code, public_error_class
from .core.models import MemoryRecord
from .core.store import StateConflictError
from .applicability import (PROFILE_KEY, DEV_LOCAL_KEY, validate_profile, require_profile_match, config_profile, profile_fields)
from .note_title import validate_group, validate_summary, validate_title

SCOPE = ('benchmark', 'suite', 'task_id', 'robot', 'controller', 'perception_mode',
         'scene_id', 'object_set_id', 'camera_config_id', 'task_variant_id',
         'sdk_sha256', 'namespace')
AXES = ('scene_id', 'object_set_id', 'camera_config_id', 'task_variant_id')
PRIVATE = re.compile(r'reward|evaluator|native_success|_check_success|private|hidden|derived[_ ]diagnos|eval[_ ]diagnos', re.I)
SHA = re.compile(r'[a-f0-9]{64}\Z')
_VIEW_WORD = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af\U00020000-\U0002ebef]|[^\s\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af\U00020000-\U0002ebef]+')
_CJK_WORD = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af\U00020000-\U0002ebef]\Z')
_VIEW_ORACLE = re.compile(r'oracle|reward|score|native|simulator|evaluator|_check_success|private|hidden|diagnos', re.I)
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
        state = {'memo': {}, 'files': {}, 'directories': {}, 'links': {}}
        token = _OPERATION.set(state)
        try:
            result = function(*args, **kwargs)
            for resolver, catalog, name, canonical, identity in list(state['files'].values()):
                path = resolver(catalog, name)
                require(path == canonical and _identity(path) == identity, 'evidence changed during authority operation')
            for name, (identity, target) in state['links'].items():
                path = Path(name)
                require(_identity(path, follow=False) == identity and os.readlink(path) == target,
                        'source link changed during authority operation')
            for folder, expected in state['directories'].items():
                require(_source_set(folder) == expected, 'SDK source file set changed during authority operation')
            return result
        finally:
            _OPERATION.reset(token)
    return wrapped


def _identity(path, *, follow=True):
    stat = path.stat() if follow else path.lstat()
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


def view_words(value):
    return sum(1 for part in _VIEW_WORD.findall(value)
               if _CJK_WORD.fullmatch(part) or any(char.isalnum() or char == '_' for char in part))


def view_no_oracle(value):
    if isinstance(value, dict):
        require(all(isinstance(key, str) and not _VIEW_ORACLE.search(key) for key in value),
                'request_view contains native evaluator/simulator oracle field')
        for child in value.values():
            view_no_oracle(child)
    elif isinstance(value, list):
        for child in value:
            view_no_oracle(child)


def public(value):
    require(not PRIVATE.search(encoded(value)), 'private evaluation content cannot enter online guidance')


def scope(value):
    validate_profile(value)
    require(isinstance(value, dict) and all(isinstance(value.get(k), str) and value[k].strip() for k in SCOPE), 'complete exact Dev v2 scope required')
    require(SHA.fullmatch(value['sdk_sha256']) is not None, 'SDK digest required')
    require('sdk_interface_sha256' not in value or isinstance(value['sdk_interface_sha256'], str)
            and SHA.fullmatch(value['sdk_interface_sha256']) is not None, 'SDK interface digest must be SHA-256')
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
    # The writer (dev_loop.event_projection) omits nonfinite event timestamps.
    # Remove only those before the strict JSON/public screen; keep the sealed
    # input untouched and retain validation of every other field.
    trace = {**trace, 'sdk_events': [
        {k: v for k, v in event.items() if k != 'timestamp'}
        if type(event.get('timestamp')) in (int, float) and not math.isfinite(event['timestamp'])
        else event for event in trace.get('sdk_events', [])]}
    public(trace)
    events = []
    for event in trace.get('sdk_events', []):
        operation = event.get('operation')
        if operation not in {'get_observation', 'find_objects', 'move_delta', 'open', 'close', 'move_normalized', 'move_joints'}:
            continue
        projected = {'operation': operation, 'status': event.get('status', 'unknown')}
        # Match the sealed writer: relative/epoch, finite int/float (not bool),
        # preserving the value and type without rounding or a sign restriction.
        if event.get('timestamp_basis') in {'relative', 'epoch'} and type(event.get('timestamp')) in (int, float):
            if math.isfinite(event['timestamp']):
                projected['timestamp_basis'] = event['timestamp_basis']
                projected['timestamp'] = event['timestamp']
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


def signature_present(feedback, signature):
    """Match a frozen failure type against one arm's public feedback only."""
    error_class = signature['public_error_class']
    operation = signature['operation']
    error = feedback.get('public_error', '')
    if 'reason_code' in signature and (planner_reason_code(error) != signature['reason_code']
                                      or public_error_class(error) != error_class):
        return False
    if not error.startswith(error_class) or error[len(error_class):len(error_class) + 1] not in {'', ':', ' ', '('}:
        return False
    return any(e.get('status') == 'failed' and
               '.'.join(x for x in (e.get('source'), e.get('operation')) if x) == operation
               for e in feedback.get('action_trace', []))


def validate_signature(signature, feedback):
    require(isinstance(signature, dict) and set(signature) in (
                {'public_error_class', 'operation'}, {'public_error_class', 'operation', 'reason_code'})
            and all(isinstance(v, str) and v.strip() and len(v) <= 120 for v in signature.values())
            and ('reason_code' not in signature or REASON_CODE.fullmatch(signature['reason_code']) is not None)
            and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*(?:: [A-Za-z_][A-Za-z0-9_]*)*',
                             signature['public_error_class']) is not None
            and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*',
                             signature['operation']) is not None,
            'public failure signature needs error class and operation')
    public(signature)
    require(signature_present(feedback, signature), 'candidate failure signature differs from source feedback')


def stop_position(trace):
    """Successful public SDK calls before termination, from the complete raw trace."""
    operations = {'get_observation', 'find_objects', 'move_delta', 'open', 'close',
                  'move_normalized', 'move_joints'}
    return sum(e.get('operation') in operations and e.get('status') in {'completed', 'succeeded'}
               for e in trace.get('sdk_events', []))


class DevV2Evidence:
    def __init__(self, store, catalogs=None, registration_roots=None):
        self.store = store
        self.registration_roots = self._registration_roots(registration_roots)
        with sqlite3.connect(store.path) as db:
            db.executescript('CREATE TABLE IF NOT EXISTS dev_v2_catalogs(id TEXT PRIMARY KEY,payload TEXT NOT NULL);'
                             'CREATE TABLE IF NOT EXISTS dev_v2_imports(id TEXT PRIMARY KEY,kind TEXT NOT NULL,attempt_id TEXT UNIQUE,payload TEXT NOT NULL);'
                             'CREATE TABLE IF NOT EXISTS dev_v2_validation_grants(id TEXT PRIMARY KEY,memory_id TEXT,cell TEXT,version INTEGER,payload TEXT,UNIQUE(memory_id,cell,version));')
        for catalog_id, catalog in (catalogs or {}).items():
            self.register(catalog_id, catalog)

    @staticmethod
    def _registration_roots(value):
        if value is None:
            return None
        require(isinstance(value, dict) and {'executor', 'validator'} <= set(value)
                and set(value) <= {'executor', 'validator', 'contract', 'source'},
                'registration roots require executor and validator lists, with optional contract/source lists')
        roots = {}
        for role, entries in value.items():
            require(isinstance(entries, list), 'registration roots must be lists')
            roots[role] = []
            for entry in entries:
                require(isinstance(entry, str), 'registration root must be an absolute directory')
                path = Path(entry)
                require(path.is_absolute() and path.resolve() == path and path.is_dir(),
                        'registration root must be a canonical existing absolute directory')
                roots[role].append(path)
        roots.setdefault('contract', [])
        roots.setdefault('source', [])
        require(not any(a.is_relative_to(b) or b.is_relative_to(a)
                        for role, entries in roots.items() for other, others in roots.items() if role != other
                        for a in entries for b in others), 'registration roots for different roles overlap')
        return roots

    def register_online(self, catalog_id, catalog):
        if self.registration_roots is None:
            raise PermissionError('registration not configured')
        role = catalog.get('role')
        require(role in self.registration_roots, 'catalog role required')
        root = Path(catalog['root'])
        require(root.is_absolute() and root.resolve() == root and root.is_dir(),
                'catalog needs a canonical existing root')
        require(any(root.is_relative_to(parent) and parent.resolve() == parent
                    for parent in self.registration_roots[role]),
                'catalog outside configured roots for its role')
        self.register(catalog_id, catalog)
        with sqlite3.connect(self.store.path) as db:
            db.execute("INSERT OR IGNORE INTO events(event_key,entity_type,entity_id,event_type,payload) "
                       "VALUES (?,'catalog',?,'catalog.registered',?)",
                       ('catalog-register:' + catalog_id, catalog_id, encoded(dict(
                           catalog_id=catalog_id, role=role, root=str(root), actor='importer',
                           recorded_at=time.time()))))
        return {'catalog_id': catalog_id, 'role': role, 'root': str(root)}

    @evidence_operation
    def register(self, catalog_id, catalog):
        """One sealed-file verifier for startup and restricted importer registration."""
        require(isinstance(catalog_id, str) and catalog_id, 'catalog ID required')
        require(catalog['role'] in {'executor', 'validator', 'contract', 'source'}, 'catalog role required')
        root = Path(catalog['root'])
        require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'catalog needs a canonical existing root')
        scope(catalog['context'])
        require(isinstance(catalog.get('files'), dict) and catalog['files'], 'sealed file list required')
        links = catalog.get('links', {})
        require(isinstance(links, dict) and (not links or catalog['role'] == 'source')
                and set(links) <= set(catalog['files']) and not set(links) & set(catalog.get('symlinks', {})),
                'links require source role, sealed names and no legacy symlink ambiguity')
        for name, sha in catalog['files'].items():
            require(isinstance(sha, str) and SHA.fullmatch(sha), 'catalog digest required')
            self.verify_file_for_catalog(catalog, name, sha, catalog_id=catalog_id)
        self._put('dev_v2_catalogs', catalog_id, catalog)

    def verify_file_for_catalog(self, catalog, name, sha, *, catalog_id=None):
        path = self._path(catalog, name, pending=(catalog_id, catalog))
        identity = _identity(path)
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(block)
        require(hasher.hexdigest() == sha, 'registered evidence digest changed')
        require(self._path(catalog, name, pending=(catalog_id, catalog)) == path and _identity(path) == identity,
                'evidence changed during registration')
        if _OPERATION.get() is not None:
            _OPERATION.get()['files'][('registration', str(self.store.path), catalog_id, name)] = (
                self._path, catalog, name, path, identity)

    def _path(self, catalog, name, *, pending=None, seen=(), hops=0, chain=False):
        require(isinstance(name, str) and name, 'unsafe evidence path')
        relative = Path(name)
        require(isinstance(name, str) and name and not relative.is_absolute() and '..' not in relative.parts, 'unsafe evidence path')
        root = Path(catalog['root'])
        path = root / relative
        if name in catalog.get('links', {}):
            require(catalog['role'] == 'source', 'only source catalogs may declare links')
            require(str(path) not in seen and hops < 8, 'source link cycle or more than 8 hops')
            # Only the final component may be a declared link; never follow an
            # unpinned directory symlink on the way to it.
            require(path.parent.resolve() == path.parent and not any(p.is_symlink() for p in path.parents),
                    'source link parent must be canonical')
            require(path.is_symlink() and stat.S_ISLNK(path.lstat().st_mode), 'declared source link must be a symlink')
            pin = catalog['links'][name]
            require(isinstance(pin, dict) and set(pin) == {'target', 'target_catalog', 'target_name'}
                    and all(isinstance(v, str) and v for v in pin.values()), 'exact source link pin required')
            target_name = Path(pin['target_name'])
            require(not target_name.is_absolute() and '..' not in target_name.parts, 'unsafe source link target name')
            target = Path(pin['target'])
            require(target.is_absolute() and '..' not in target.parts and str(target) == pin['target'],
                    'source link target must be an exact absolute path')
            raw = os.readlink(path)
            hop = Path(os.path.normpath(raw if os.path.isabs(raw) else str(path.parent / raw)))
            require(hop == target, 'source link hop differs from pin')
            destination = pending[1] if pending and pin['target_catalog'] == pending[0] else self.catalog(pin['target_catalog'], 'source')
            require(destination['role'] == 'source' and target == Path(destination['root']) / target_name,
                    'source link target outside target catalog root')
            sha = catalog['files'].get(name)
            require(isinstance(sha, str) and SHA.fullmatch(sha)
                    and destination['files'].get(pin['target_name']) == sha, 'source link target must be sealed at exact digest')
            if _OPERATION.get() is not None:
                _OPERATION.get()['links'].setdefault(str(path), (_identity(path, follow=False), raw))
            return self._path(destination, pin['target_name'], pending=pending, seen=(*seen, str(path)),
                              hops=hops + 1, chain=True)
        if chain:
            require(path.resolve() == path and not any(p.is_symlink() for p in (*path.parents, path)),
                    'every source link hop must be explicitly pinned')
            require(path.is_file() and stat.S_ISREG(path.lstat().st_mode), 'source link must end at a regular file')
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
            state['files'][key] = (self._path, catalog, name, path, identity)
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
        catalog_id = self._catalog_for_ref(catalog_id, ref, 'source')
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
            state['files'][key] = (self._path, catalog, relative, canonical, identity)

    def scene_contract(self, executor_id, ref):
        """Resolve an exact sealed ref from executor, contract or source evidence."""
        return self.ref(self._catalog_for_ref(executor_id, ref, 'contract', 'source'), ref)

    def source_ref(self, executor_id, ref):
        """Read local evidence or an exact source pin in the import's context."""
        return self.ref(self._catalog_for_ref(executor_id, ref, 'source'), ref)

    def _catalog_for_ref(self, executor_id, ref, *roles):
        require(isinstance(ref, dict) and set(ref) == {'uri', 'sha256'}, 'exact artifact reference required')
        executor = self.catalog(executor_id, 'executor')
        path = Path(ref['uri'])
        require(path.is_absolute(), 'scene contract needs an absolute URI')
        if path.is_relative_to(executor['root']):
            require(executor['files'].get(str(path.relative_to(executor['root']))) == ref['sha256'],
                    'artifact is not preregistered at exact digest')
            return executor_id
        with sqlite3.connect(self.store.path) as db:
            rows = db.execute('SELECT id,payload FROM dev_v2_catalogs ORDER BY id').fetchall()
        for catalog_id, payload in rows:
            catalog = json.loads(payload)
            if (catalog['role'] in roles and path.is_relative_to(catalog['root'])
                    and (catalog['role'] != 'source' or catalog['context'] == executor['context'])):
                name = str(path.relative_to(catalog['root']))
                if catalog['files'].get(name) == ref['sha256']:
                    return catalog_id
        allowed = ' or '.join(role + ' catalogs' for role in roles)
        if 'contract' in roles:
            require(False, f'artifact outside registered {allowed} or not pinned at exact digest')
        require(False, f'artifact outside registered root: no matching {allowed} at exact digest')

    def _source_set_catalog(self, executor_id, folder, expected, files):
        path = Path(folder)
        require(path.is_absolute() and path.resolve() == path and path.is_dir(), 'SDK source set needs canonical directory')
        executor = self.catalog(executor_id, 'executor')
        with sqlite3.connect(self.store.path) as db:
            catalogs = [json.loads(row[0]) for row in db.execute('SELECT payload FROM dev_v2_catalogs')]
        allowed = [executor] if path.is_relative_to(executor['root']) else [
            c for c in catalogs if c['role'] == 'source' and c['context'] == executor['context']]
        require(any(path.is_relative_to(c['root']) and all(
            Path(name).is_relative_to(path) and c['files'].get(str(Path(name).relative_to(c['root']))) == files[name]
            for name in expected) for c in allowed), 'SDK source set outside sealed source catalogs')

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
                    root = Path(value['root'])
                    for row in db.execute('SELECT payload FROM dev_v2_catalogs'):
                        other = json.loads(row[0])
                        if value['role'] != other['role'] and {'contract', 'source'} & {value['role'], other['role']}:
                            require(not root.is_relative_to(other['root']) and not Path(other['root']).is_relative_to(root),
                                    'source/contract catalog roots must not overlap other roles')
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
        require_profile_match(d['context'], config_profile(config))
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
        lock = self.source_ref(catalog_id, lock_ref)
        require(lock['schema'] == 'attentionbench.dev-identity-lock.v2' and lock['formal_eligible'] is False, 'frozen Dev identity lock required')
        require(lock.get('source_sets'), 'complete SDK source file sets required')
        for folder, expected in lock['source_sets'].items():
            require(_source_set(folder) == expected and all(name in lock['files'] for name in expected), 'SDK source file set drift')
            self._source_set_catalog(catalog_id, folder, expected, lock['files'])
            if _OPERATION.get() is not None:
                _OPERATION.get()['directories'][folder] = expected
        require(lock['files'].get(lock['planner_lock']['path']) == lock['planner_lock']['sha256'] and lock['model_sources'] and all(lock['files'].get(n) == sha for n, sha in lock['model_sources'].items()), 'planner/robot source identity incomplete')
        for name, sha in lock['files'].items():
            self.verify_file(catalog_id, {'uri': name, 'sha256': sha})
        sdk = d['sdk_identity']
        require(digest(json.dumps(sdk, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()) == d['context']['sdk_sha256'], 'SDK identity does not match exact context digest')
        from .applicability import interface_digest
        require('sdk_interface_sha256' not in d['context'] or interface_digest(sdk) == d['context']['sdk_interface_sha256'], 'SDK interface identity does not match contract and prompt')
        require(sdk['planner_lock'] == lock['planner_lock']['sha256'] and sdk['robot_model_identity'] == lock['model_sources'], 'SDK identity lock mismatch')
        expected_files = {str(Path(n).relative_to(lock['source_root'])): sha for n, sha in lock['files'].items()
                          if Path(n).is_relative_to(lock['source_root']) and any(part in Path(n).parts for part in ('openarm365', 'openarm365_dev', 'openarm365_dev_v2', 'curobo_v2')) and '/tests/' not in n}
        require(expected_files and sdk['implementation'] == expected_files, 'SDK implementation source list mismatch')
        return catalog, manifest, config, record, source, cell

    def _provider(self, catalog_id, job, manifest, config):
        receipt = job['receipt']
        request = self.ref(catalog_id, receipt['request'])
        output = self.ref(catalog_id, receipt['response'])
        wire = self.ref(catalog_id, receipt['provider_response'])
        identity = verify_provider(request, output, wire, receipt, job['actual'],
                                   manifest, config, job['kind'], require)
        if identity == 'codex_exec':
            receipt = dict(receipt, provider_identity=identity)
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
        require(d.get('import_purpose') in (None, 'memory_validation'), 'unknown execution import purpose')
        require(all(permission.get(k) == v for k, v in {'run_id': bundle['run_id'], 'cell': d['cell'], 'code_version': d['version'], 'code_sha256': bundle['policy_sha256'], 'config_sha256': digest(json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())}.items()), 'permission scope not bound to exact run/code/config/version')
        require(type(permission.get('code_version')) is int and type(permission.get('action_version')) is int and permission['action_version'] == 1 and isinstance(permission.get('scope'), dict) and permission['scope'], 'explicit permission action/version/scope required')
        if permission.get('action') == 'dev_tier_sim_motion':
            from .motion_permission import verify
            verify(permission, d, config, bundle, source, require)
        elif permission.get('action') == 'execute_bounded_motion_policy':
            with self._ledger(catalog_id, d) as db:
                approval = self._row(db, 'v2_planner_requests', 'id=?', (d['approval_id'],))
            require(approval['request_type'] == 'approval' and approval['state'] == 'consumed' and approval['binding'] == permission and approval['consumed_by'] == job['job_id'] and approval['created_at'] <= approval['consumed_at'] <= job['claimed_at'] and approval['consumed_at'] <= approval['deadline_epoch'], 'native motion lacks exact consumed human approval receipt')
            require(approval['response']['approved'] is True and approval['response']['actor'] == 'authenticated_human' and approval['response']['response_id'] and approval['created_at'] <= approval['responded_at'] <= approval['consumed_at'], 'genuine external human decision receipt required')
        else:
            require(permission.get('action') == 'sensor_only', 'unknown permission action')
            from .motion_permission import operations
            require('actor' not in permission and not operations(source, require),
                    'sensor-only evidence cannot claim motion or an approval actor')
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
            from .scene_contract import V1, V2, verify_v2_receipt
            contract = self.scene_contract(catalog_id, config['scene_contract'])
            original = self.ref(catalog_id, d['scene_state'])
            condition = bundle['scene_condition']
            require(contract['schema'] in {V1, V2} and contract['tolerance'] == 1e-6, 'frozen scene contract/tolerance required')
            require(original['validation'] == condition and condition['contract'] == config['scene_contract'] and condition['task_id'] == config['task_id'] and condition['tolerance'] == 1e-6, 'original scene receipt/contract binding mismatch')
            if contract['schema'] == V1:
                require(type(condition['matched']) is bool and condition['matched'] == scene_matches(contract['tasks'][config['task_id']]['state'], original['state']) and condition['raw_sha256'] == digest(encoded(original['state']).encode()), 'original scene verdict differs from raw state')
            else:
                verify_v2_receipt(contract, config, original['state'], condition)
        evidence = {'catalog_id': catalog_id, 'descriptor': descriptor, 'context': d['context'], 'cell': d['cell'], 'version': d['version'], 'bundle': job['receipt']['bundle'], 'public_feedback': job['receipt']['public_feedback'], 'source': record['source'], 'config': d['config'], 'manifest': d['manifest'], 'ledger': d['ledger'], 'run_id': bundle['run_id'], 'attempt_id': bundle['attempt_id'], 'status': bundle['status']}
        if d.get('import_purpose') is not None:
            evidence.update(import_purpose=d['import_purpose'], permission=permission)
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
                from .scene_contract import V1, verify_v2_receipt
                contract = self.scene_contract(execution['catalog_id'], config['scene_contract'])
                states = [self.ref(validator_id, ref) for ref in v['scene_replay_states']]
                require(len(states) == 2 and v['scene_replay_states'][0]['uri'] != v['scene_replay_states'][1]['uri'],
                        'full fixture/robot/object scene reproduction failed')
                if contract['schema'] == V1:
                    matched = [scene_matches(contract['tasks'][bundle['task_id']]['state'], state) for state in states]
                else:
                    matched = [verify_v2_receipt(contract, config, receipt['state'], receipt['validation']) for receipt in states]
                require(all(matched), 'full fixture/robot/object scene reproduction failed')
        safety_path = self._path(validator, str(Path(v['safety']['uri']).relative_to(validator['root'])))
        self.ref(validator_id, v['safety'])
        from .memory_v2 import MemoryV2Manager
        MemoryV2Manager._safety(safety_path, bundle['attempt_id'])
        require(isinstance(v['permission_scope'], dict) and v['permission_scope'] == d['permission_scope'] and v['permission_scope'], 'permission scope not bound to trusted validation')
        return execution, d, bundle, artifacts, cell, v, safety_path

    def _visual_advisor(self, catalog_id, config, bundle, advisor, model, help_value, manifest):
        """Check the original visual request, response, frame bytes and ledger receipt."""
        require(help_value.get('mode') == 'vision' and not help_value.get('rejected'), 'visual Advisor help must be answered in vision mode')
        hint_ref = help_value.get('advisor_proxy')
        require(isinstance(hint_ref, dict) and hint_ref == advisor.get('receipt', {}).get('advisor_answer'), 'visual Advisor help/ledger reference mismatch')
        hint = self.ref(catalog_id, hint_ref)
        provider_source, locked_model = frozen_provider(manifest, config, 'advisor', require)
        if provider_source == 'codex':
            verify_codex_record(hint, locked_model, require)
        view_ref = hint.get('request_view')
        require(isinstance(view_ref, dict) and view_ref == advisor['receipt'].get('request_view'), 'visual Advisor request_view/ledger reference mismatch')
        view = self.ref(catalog_id, view_ref)
        view_no_oracle(view)
        public(view)
        require(hint.get('schema_version') == 'attentionbench.advisor-proxy.v3' and hint.get('kind') == 'hint'
                and hint.get('request_type') == 'hint' and hint.get('status') == 'completed'
                and hint.get('mode') == 'vision' and hint.get('requested') is True
                and hint.get('cache_hit') is False and isinstance(hint.get('feedback'), str)
                and hint['feedback'].strip() and hint['feedback'] == help_value.get('content'),
                'completed original visual Advisor answer required')
        proxy = config.get('advisor_proxy')
        require(isinstance(proxy, dict) and proxy.get('enabled') is True and proxy.get('mode') == 'vision'
                and isinstance(proxy.get('model'), str) and proxy['model'].strip()
                and hint.get('model') == proxy['model'],
                'visual Advisor model differs from configured Advisor model')
        require(help_value.get('selected_moments') == hint.get('selected_moments')
                and help_value.get('localization') == hint.get('localization'),
                'visual Advisor selection/localization differs from help')
        require(advisor['settled_at'] <= model['claimed_at'], 'visual Advisor answer settled after Dev model call')
        require(isinstance(view, dict) and set(view) == {'schema_version', 'task', 'question', 'reasoning', 'request_kind',
                'frames', 'steps', 'diff', 'eval_review', 'shown_video_seconds'}
                and view['schema_version'] == 'attentionbench.request-view.v1', 'complete request_view v1 required')
        require(isinstance(view['task'], dict) and set(view['task']) == {'task_id', 'task_prompt'}
                and view['task'] == {'task_id': config['task_id'], 'task_prompt': config['task_prompt']}
                and all(isinstance(view[k], str) for k in ('question', 'reasoning', 'diff'))
                and view['question'].strip() and view['request_kind'] in {'targeted', 'generic'}
                and isinstance(view['steps'], list)
                and all(isinstance(s, dict) and set(s) == {'index', 'label', 'status', 'count', 'stop_reason'}
                        and type(s['index']) is int and type(s['count']) is int and s['count'] > 0
                        and all(isinstance(s[k], str) for k in ('label', 'status', 'stop_reason')) for s in view['steps'])
                and isinstance(view['eval_review'], dict)
                and set(view['eval_review']) == {'passed', 'reason', 'reference_suggestion'}
                and type(view['eval_review']['passed']) is bool
                and all(isinstance(view['eval_review'][k], str) for k in ('reason', 'reference_suggestion'))
                and type(view['shown_video_seconds']) in (int, float)
                and math.isfinite(view['shown_video_seconds']) and view['shown_video_seconds'] >= 0,
                'request_view task/questions/steps/eval_review/video fields incomplete')
        require(isinstance(view['frames'], list) and 0 < len(view['frames']) <= 10,
                'request_view frames required')
        require(isinstance(bundle.get('artifact_dir'), str) and bundle['artifact_dir'],
                'visual frame episode path missing from failed execution')
        episode = Path(bundle['artifact_dir'])
        require(episode.is_absolute() and episode.is_relative_to(self.catalog(catalog_id)['root']),
                'visual frame episode outside registered catalog')
        frames = []
        for frame in view['frames']:
            require(isinstance(frame, dict) and set(frame) == {'view', 'sha256', 'filename', 'data_url'}
                    and frame['view'] in {'overview', 'side', 'top', 'left_wrist', 'right_wrist'}
                    and isinstance(frame['filename'], str) and frame['filename']
                    and isinstance(frame['sha256'], str) and SHA.fullmatch(frame['sha256'])
                    and isinstance(frame['data_url'], str) and frame['data_url'].startswith('data:image/jpeg;base64,')
                    and len(frame['data_url']) > len('data:image/jpeg;base64,'),
                    'request_view frame manifest incomplete')
            relative = Path(frame['filename'])
            require(not relative.is_absolute() and '..' not in relative.parts
                    and (relative.parts[:1] == ('recording_frames',) and frame['view'] == 'overview'
                         or relative.parts[:2] == ('recording_views', frame['view'])),
                    'request_view frame path/view mismatch')
            raw_frame = self.ref(catalog_id, {'uri': str(episode / relative), 'sha256': frame['sha256']}, raw=True)
            # The archived data URL is a JPEG projection of the pinned frame,
            # not the original frame bytes. Reproduce the producer's transform.
            try:
                from PIL import Image
                with Image.open(io.BytesIO(raw_frame)) as original:
                    picture = original.convert('RGB')
                    if picture.width > 768:
                        picture = picture.resize((768, max(1, round(picture.height * 768 / picture.width))))
                    projected = io.BytesIO()
                    picture.save(projected, format='JPEG', quality=80)
                shown = base64.b64decode(frame['data_url'].split(',', 1)[1], validate=True)
            except (OSError, ValueError, binascii.Error) as exc:
                raise StateConflictError('request_view frame image cannot be decoded') from exc
            require(shown == projected.getvalue(), 'request_view image differs from pinned frame projection')
            frames.append({k: frame[k] for k in ('filename', 'sha256', 'view')})
        require(hint.get('frames') == frames and hint.get('views') == sorted({f['view'] for f in frames}),
                'visual Advisor frames differ from request_view')
        require(hint.get('question') == view['question'] and hint.get('shown_video_seconds') == view['shown_video_seconds'],
                'visual Advisor question/video differs from request_view')
        words = sum(view_words(x) for x in [view['question'], view['reasoning'], view['diff'],
                    *(s['stop_reason'] for s in view['steps']), str(view['eval_review']['passed']),
                    view['eval_review']['reason'], view['eval_review']['reference_suggestion']])
        answer_words = view_words(hint['feedback'])
        receipt = advisor['receipt']
        require(all(type(hint.get(k)) in (int, float) and math.isfinite(hint[k]) for k in
                    ('shown_words', 'answer_words', 'read_wps', 'write_wps', 'hint_human_seconds', 'logical_human_seconds'))
                and hint['shown_words'] == words and hint['answer_words'] == answer_words
                and hint['read_wps'] == 4.0 and hint['write_wps'] == 0.7
                and math.isclose(hint['hint_human_seconds'], view['shown_video_seconds'] + words / 4.0 + answer_words / 0.7, rel_tol=1e-12)
                and hint['logical_human_seconds'] == hint['hint_human_seconds']
                and all(receipt.get(k) == hint[k] for k in ('shown_video_seconds', 'shown_words', 'answer_words', 'read_wps', 'write_wps'))
                and advisor['actual'].get('hint_human_seconds') == hint['hint_human_seconds'],
                'visual Advisor logical human time receipt disagrees with components')
        usage = hint.get('usage')
        require(isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in
                    ('total_tokens', 'prompt_tokens', 'completion_tokens'))
                and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
                and all(advisor['actual'].get(k) == usage[v] for k, v in
                        [('provider_tokens', 'total_tokens'), ('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens')])
                and all(advisor['actual'].get(k) == 1 for k in ('advisor_calls', 'help_count', 'help_credits')),
                'visual Advisor usage/committed cost receipt incomplete')
        return hint['feedback'], view_ref, hint['model']

    def _generation_feedback(self, packets, failed, catalog_id, descriptor):
        """Bind additions to the failed version in the sealed original ledger."""
        expected_source = self.ref(failed['catalog_id'], failed['source'], raw=True).decode()
        feedback = self.ref(failed['catalog_id'], failed['public_feedback'])
        require(descriptor['manifest'] == failed['manifest'], 'feedback origin manifest differs')
        manifest = self.ref(catalog_id, descriptor['manifest'])
        with self._ledger(catalog_id, descriptor) as db:
            cell = self._row(db, 'v2_cells', 'id=?', (failed['cell'],))
            version = self._row(db, 'v2_versions', 'cell=? AND version=?',
                                (failed['cell'], failed['version']))
            require(version['source'] == failed['source'], 'feedback origin version differs')

            def eval_job():
                rows = db.execute("SELECT id FROM v2_jobs WHERE cell=? AND version=? AND kind='eval'",
                                  (failed['cell'], failed['version'])).fetchall()
                require(len(rows) == 1, 'sealed Eval job missing or ambiguous')
                return dict(self._job(db, rows[0][0], failed['cell'], failed['version'], 'eval'), state='settled')

            modes = []
            for audience, packet in packets:
                require(packet.get('previous_complete_source') == expected_source,
                        'online help source/public failed trace mismatch')
                modes.append(verify_feedback(packet.get('public_feedback'), feedback, audience=audience, version=version,
                    cell=cell, manifest=manifest, read=lambda ref: self.source_ref(catalog_id, ref),
                    eval_job=eval_job, require=require))
        return 'with_eval_review' if 'with_eval_review' in modes else 'exact'

    def import_source(self, *, import_id, catalog_id, descriptor):
        d = self.read(catalog_id, descriptor)
        title = validate_title(d['title']) if 'title' in d else None
        summary = validate_summary(d['summary']) if 'summary' in d else None
        group = validate_group(d['group']) if 'group' in d else None
        _, manifest, config, record, repair, cell = self._base(catalog_id, d)
        failed, fd, bundle, artifacts, _, verdict, _ = self.verdict(d['failed_execution_import_id'], d['validator_id'], d['validation_descriptor'], for_validation=False)
        require(bundle['native_success'] is False and failed['cell'] == d['cell'] and failed['version'] == d['version'] - 1 and failed['context'] == d['context'], 'source must be previous independent native-confirmed failed attempt in exact scope')
        with self._ledger(catalog_id, d) as db:
            advisor = self._job(db, d['advisor_job_id'], d['cell'], d['version'], 'advisor')
            model = self._job(db, d['model_job_id'], d['cell'], d['version'], 'model')
            row = self._row(db, 'v2_planner_requests', 'id=?', (d['request_id'],)) if 'request_id' in d else None
            # Omitting the ID must not hide an existing (including rejected) request.
            requests = []
            if row is None and db.execute("SELECT 1 FROM sqlite_master WHERE name='v2_planner_requests'").fetchone():
                requests = [json.loads(p) for p, in db.execute(
                    'SELECT payload FROM v2_planner_requests WHERE cell=?', (d['cell'],))]
                requests = [r for r in requests if r.get('version') == d['version'] and r.get('request_type') == 'advisor']
        mr, mo, mc = self._provider(catalog_id, model, manifest, config)
        help_value = self.ref(catalog_id, record['help'])
        provider = help_value.get('provider')
        require(provider in {'advisor', 'human_proxy'}, 'unsupported Advisor provider')
        if provider == 'advisor':
            ar, ao, ac = self._provider(catalog_id, advisor, manifest, config)
            require(help_value['provider'] == 'advisor' and not help_value.get('rejected') and help_value['content'] == ao['content'] and help_value['request'] == ac['request'] and help_value['response'] == ac['response'] and help_value['usage'] == ac['usage'], 'answered original Advisor guidance required')
            require(isinstance(ao['content'], str) and ao['content'].strip(), 'nonempty answered guidance required')
            require(advisor['actual'].get('advisor_calls') == 1 and advisor['actual'].get('help_count') == 1 and advisor['actual'].get('help_credits') == 1 and model['actual'].get('dev_calls') == 1, 'committed assistance/model cost required')
            if row is not None:
                require(row['binding'] == {'cell': d['cell'], 'version': d['version'], 'input_sha256': digest(json.dumps(ar['messages'], sort_keys=True, ensure_ascii=False, allow_nan=False).encode()), 'source_sha256': failed['source']['sha256'], 'policy_id': cell['policy_id'], 'config_sha256': d['config']['sha256']}, 'Advisor request binding differs from original source/input/scope')
                require(advisor['claimed_at'] <= row['created_at'] <= advisor['settled_at'] <= row['responded_at'] <= row['deadline_epoch'] and row['responded_at'] <= model['claimed_at'], 'Advisor response late or not available before actual adoption')
                require(row['cell'] == d['cell'] and row['version'] == d['version'] and row['request_type'] == 'advisor' and row['state'] == 'answered' and row['accepted'] is True and row['adopted'] is True and row['response'] == ac['response'] and row['adoption'] == {'model_input': mc['request'], 'result_source': record['source']}, 'answered/adopted request must bind actual generation')
            else:
                selection = manifest.get('selection')
                require(isinstance(selection, dict) and not requests
                        and not any(value.get('planner_features') for value in (manifest, selection, config, cell)),
                        'original Advisor request row required for planner features or existing requests')
                require(cell['policy_id'] in {'reactive_help', 'retry_k_then_ask',
                        'budget_matched_random_escalation', 'trace_aware_hint_only',
                        'full_trace_aware_attention_planner'}, 'Advisor source requires an asking policy')
                require('rejected' not in help_value, 'rejected Advisor help cannot be imported')
                require(advisor['claimed_at'] <= advisor['settled_at'] <= model['claimed_at'],
                        'Advisor response late or not available before actual adoption')
                # Adoption is proved below: guidance is in the next version's
                # generation input, and that version's source is the model output.
            guidance, view_ref, advisor_model = ao['content'], None, None
        else:
            guidance, view_ref, advisor_model = self._visual_advisor(catalog_id, config, bundle, advisor, model, help_value, manifest)
            require(model['actual'].get('dev_calls') == 1, 'committed Dev model cost required')
        require(record['model_input'] == mc['request'] and record['model_output'] == mc['response'], 'model input/output lineage mismatch')
        packet = json.loads(mr['messages'][-1]['content'])
        expected_feedback = self.ref(failed['catalog_id'], failed['public_feedback'])
        packets = [('dev', packet)]
        if provider == 'advisor':
            packets.append(('advisor', json.loads(ar['messages'][-1]['content'])))
        projection = self._generation_feedback(packets, failed, catalog_id, d)
        signature = d.get('failure_signature')
        if signature is not None:
            validate_signature(signature, expected_feedback)
        help_input = packet.get('attention_control', {}).get('help', {})
        require(help_input.get('provider') == provider and help_input.get('content') == guidance, 'guidance not actually present in generation input')
        generated = mo['content'].strip()
        if generated.startswith('```'):
            generated = re.sub(r'^```(?:json)?\s*|\s*```$', '', generated)
        require(json.loads(generated)['source'].encode() == repair, 'adopted source differs from actual model output')
        public(guidance); public(repair.decode())
        evidence = {'kind': 'source', 'catalog_id': catalog_id, 'descriptor': descriptor, 'failed_execution_import_id': d['failed_execution_import_id'], 'context': d['context'], 'request_id': d.get('request_id'), 'cell': d['cell'], 'version': d['version'], 'guidance': guidance, 'repair': repair.decode(), 'failure_signature': signature, 'source_policy_sha256': failed['source']['sha256'], 'repair_sha256': record['source']['sha256'], 'source_policy_id': cell['policy_id'], 'source_seed': config['seed'], 'source_run_id': bundle['run_id'], 'source_attempt_id': bundle['attempt_id']}
        evidence['feedback_projection'] = projection
        if mc.get('provider_identity') == 'codex_exec':
            evidence.update(provider_identity='codex_exec', provider_identities=dict(model='codex_exec', advisor='codex_exec'))
        if title is not None:
            evidence['title'] = title
        if summary is not None:
            evidence['summary'] = summary
        if group is not None:
            evidence['group'] = group
        if provider == 'advisor':
            evidence['advisor_evidence'] = 'planner_request' if row is not None else 'advisor_job'
        if provider == 'human_proxy':
            evidence.update(provider=provider, request_view=view_ref, advisor_model=advisor_model, advisor_mode='vision')
        stored = evidence
        # Derived annotations may be absent in immutable pre-C10/C10b imports.
        # Only those omissions are compatible; changed evidence still conflicts.
        with sqlite3.connect(self.store.path) as db:
            old = db.execute('SELECT payload FROM dev_v2_imports WHERE id=?', (import_id,)).fetchone()
        if old:
            legacy = json.loads(old[0])
            optional = {'feedback_projection'}
            if provider == 'advisor' and row is not None:
                optional.add('advisor_evidence')
            if legacy == {k: v for k, v in evidence.items() if k not in optional - set(legacy)}:
                stored = legacy
        self._put('dev_v2_imports', import_id, stored, kind='source')
        return evidence

    @memo_evidence('source')
    def source(self, import_id):
        old = self.get(import_id)
        require(old['kind'] == 'source', 'source import required')
        return self.import_source(import_id=import_id, catalog_id=old['catalog_id'], descriptor=old['descriptor'])

    def create_candidate(self, *, memory_id, source_import_id, created_at, from_provisional=None):
        require(not self.store.count_memory_events(memory_id, 'memory.provisional_created'),
                'provisional notes cannot become candidates; create a new evidence-backed candidate')
        require(type(created_at) in (int, float) and math.isfinite(created_at), 'finite candidate timestamp required')
        s = self.source(source_import_id)
        artifact = {'schema_version': 'attentionbench.memory-artifact.v2', 'kind': 'policy_patch', 'guidance': s['guidance'], 'repair': s['repair'], 'applicability': {k: s['context'][k] for k in (*SCOPE, 'sdk_interface_sha256', PROFILE_KEY, DEV_LOCAL_KEY) if k in s['context']}}
        if s['failure_signature'] is not None:
            artifact['failure_signature'] = s['failure_signature']
        provenance = {'schema_version': 'attentionbench.memory-provenance.v2', 'channel': 'dev_v2', 'source_import_id': source_import_id, 'memory_id': memory_id, 'source_kind': 'advisor_proxy', 'human_attention_seconds': 0.0, 'assistance_credits': 1, 'source_trace_id': 'dev-v2-source:' + source_import_id, 'source_raw_trace_id': 'dev-v2-source:' + source_import_id, 'raw_evidence_refs': [], 'artifact': artifact, 'artifact_sha256': digest(encoded(artifact).encode()), **{k: s[k] for k in ('source_policy_id', 'source_policy_sha256', 'source_seed', 'source_run_id', 'source_attempt_id')}}
        if from_provisional is not None:
            from .seeded_notes import is_provisional
            require(isinstance(from_provisional, str) and from_provisional,
                    'invalid from_provisional id')
            with sqlite3.connect(self.store.path) as db:
                linked = db.execute('SELECT payload FROM memory_v2_provenance WHERE memory_id=?',
                                    (from_provisional,)).fetchone()
            prior = json.loads(linked[0]) if linked else None
            note = self.store.get_memory(from_provisional) if linked else None
            require(is_provisional(prior) and note is not None
                    and note.applicability.get('task_id') == s['context']['task_id']
                    and prior.get('source_seed') == s['source_seed'], 'from_provisional scope mismatch')
            require_profile_match(note.applicability, s['context'])
            require(re.fullmatch(r'p-[0-9a-f]{24}', from_provisional) or prior.get('demoted_at') is not None,
                    'invalid from_provisional id')
            if prior.get('demoted_at') is not None:
                require(prior['source_import_id'] == source_import_id, 'demoted candidate source changed')
                provenance['from_provisional_version'] = note.version
            provenance['from_provisional'] = from_provisional
        if 'title' in s:
            provenance['title'] = s['title']
        if 'summary' in s:
            provenance['summary'] = s['summary']
        if 'group' in s:
            provenance['group'] = s['group']
        if s.get('provider') == 'human_proxy':
            provenance.update(provider='human_proxy', request_view=s['request_view'], advisor_model=s['advisor_model'], advisor_mode=s['advisor_mode'])
        if s['failure_signature'] is not None:
            provenance['failure_signature'] = s['failure_signature']
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
                             policy_sha256, config_sha256, exposure, origin_source_sha256=None, target_seed=None):
        if target_seed is not None:
            from .memory_service import MemoryService
            return MemoryService(self.store.path).cross_seed.authorize(
                memory_id=memory_id, target_seed=target_seed, cell=cell, version=version, seed=seed,
                context=context, policy_sha256=policy_sha256, config_sha256=config_sha256,
                exposure=exposure, origin_source_sha256=origin_source_sha256)
        from .memory_v2 import MemoryV2Manager
        manager = MemoryV2Manager(self.store)
        provenance = manager.provenance(memory_id)
        memory = self.store.get_memory(memory_id)
        require(provenance.get('channel') == 'dev_v2' and memory.status.value == 'candidate', 'offline validation grant requires a Dev v2 candidate')
        plan = manager.get_plan(memory_id)
        require(plan is not None, 'freeze a validation plan before dispatch')
        require(manager.get_decision(memory_id) is None, 'validation batch is closed')
        generated = plan.get('arm_generation') == 'dev_from_origin'
        require(generated or origin_source_sha256 is None, 'origin binding requires dev_from_origin plan')
        scope(context)
        require_profile_match(memory.applicability, context)
        require(type(version) is int and version > 0 and isinstance(cell, str) and cell and type(seed) is int, 'exact cell/version/seed required')
        require(exposure in {'none', 'candidate_validation'} and all(isinstance(x, str) and SHA.fullmatch(x) for x in (policy_sha256, config_sha256)), 'explicit exposure and code/config digests required')
        require(all(context[k] == memory.applicability[k] for k in SCOPE if k not in AXES), 'validation grant exact SDK/namespace/robot scope mismatch')
        require('sdk_interface_sha256' not in memory.applicability or context.get('sdk_interface_sha256') == memory.applicability['sdk_interface_sha256'], 'validation grant exact SDK interface mismatch')
        require(any(case['seed'] == seed and all(case[k] == context[k] for k in (*AXES, 'camera_names', 'task_prompt')) for case in plan['cases']), 'validation grant outside frozen concrete case')
        if plan.get('profile', 'default') == 'single_scene_v1':
            arm = 'control_cell' if exposure == 'none' else 'treatment_cell'
            require(any(pair[arm] == cell for pair in plan['pairs']), 'cell/exposure outside frozen single-scene pairs')
            require(config_sha256 == plan['validation_config_sha256'], 'single-scene policy/config differs from frozen plan')
            if generated:
                require(origin_source_sha256 == plan['origin_source_sha256'], 'validation origin differs from frozen plan')
            else:
                require(policy_sha256 == plan['validation_policy_sha256'], 'single-scene policy/config differs from frozen plan')
        binding = dict(**profile_fields(memory.applicability), memory_id=memory_id, memory_version=memory.version, cell=cell, version=version, seed=seed,
                       context=context, policy_sha256=policy_sha256, config_sha256=config_sha256,
                       exposure=exposure, plan_sha256=digest(encoded(plan).encode()))
        if generated:
            binding.update(origin_source_sha256=origin_source_sha256, dev_model=plan['dev_model'],
                           guidance_sha256=digest(memory.guidance.encode()))
        with sqlite3.connect(self.store.path) as db:
            db.execute('BEGIN IMMEDIATE')
            current = json.loads(db.execute('SELECT payload FROM memories WHERE id=?', (memory_id,)).fetchone()[0])
            require(current['status'] == 'candidate' and current['version'] == memory.version, 'candidate revoked before validation grant')
            old = db.execute('SELECT payload FROM dev_v2_validation_grants WHERE memory_id=? AND cell=? AND version=?', (memory_id, cell, version)).fetchone()
            if old:
                grant = json.loads(old[0]); require(grant['binding'] == binding, 'validation dispatch already binds different evidence')
                return grant
            rows = db.execute('SELECT payload FROM dev_v2_validation_grants WHERE memory_id=?', (memory_id,)).fetchall()
            if not generated:
                require(all(json.loads(row[0])['binding']['policy_sha256'] == policy_sha256 for row in rows), 'validation policy version is frozen across all cases')
            if plan.get('profile', 'default') == 'single_scene_v1':
                require(all(json.loads(row[0])['binding']['config_sha256'] == config_sha256 for row in rows), 'validation configuration is frozen across all single-scene arms')
            grant = {'schema_version': 'attentionbench.dev-v2-validation-grant.v1', 'grant_id': 'validation:' + uuid.uuid4().hex, 'issued_at': time.time(), 'binding': binding}
            db.execute('INSERT INTO dev_v2_validation_grants VALUES (?,?,?,?,?)', (grant['grant_id'], memory_id, cell, version, encoded(grant)))
        return grant

    def validation_manager(self):
        from .memory_v2 import MemoryV2Manager
        return MemoryV2Manager(self.store)

    def validation_candidate(self, memory_id):
        return self.store.get_memory(memory_id)

    def validation_plan(self, memory_id, target_seed=None):
        if target_seed is None:
            return self.validation_manager().get_plan(memory_id)
        with sqlite3.connect(self.store.path) as db:
            row = db.execute('SELECT payload FROM memory_cross_seed_plans WHERE memory_id=? AND seed=?',
                             (memory_id, target_seed)).fetchone()
        return json.loads(row[0]) if row else None

    def _validation_generation(self, execution, ed, binding, plan, candidate):
        """Reconstruct arm lineage from executor files and its settled model job."""
        catalog_id = execution['catalog_id']
        record = self.ref(catalog_id, ed['version_record'])
        require({'parent_source', 'model_input', 'model_output', 'dev_memory_exposure'} <= set(record)
                and 'model_job_id' in ed, 'generation lineage/exposure receipt missing')
        require(binding.get('origin_source_sha256') == plan['origin_source_sha256']
                and binding.get('dev_model') == plan['dev_model'], 'generation grant origin/model changed')
        parent = self.ref(catalog_id, record['parent_source'], raw=True)
        require(record['parent_source']['sha256'] == plan['origin_source_sha256'], 'wrong generation parent/origin')
        with self._ledger(catalog_id, ed) as db:
            job = self._job(db, ed['model_job_id'], ed['cell'], ed['version'], 'model')
        request, output, provider = self._provider(catalog_id, job, self.ref(catalog_id, ed['manifest']), self.ref(catalog_id, ed['config']))
        require(record['model_input'] == provider['request'] and record['model_output'] == provider['response']
                and job['actual'].get('dev_calls') == 1
                and job['settled_at'] <= self._execution_claim_time(execution, ed), 'generation model lineage mismatch')
        model = {'provider': request.get('model_source', 'parcc'), 'model': request.get('model')}
        wire = self.ref(catalog_id, provider['provider_response'])
        require(model == plan['dev_model'] and output.get('model') == model['model']
                and output.get('model_source', 'parcc') == model['provider']
                and wire['body'].get('model', model['model']) == model['model'], 'generation Dev model differs from plan')
        messages = request['messages']
        require(len(messages) == 2 and [m['role'] for m in messages] == ['system', 'user'],
                'generation requires one system and one user packet')
        packet = json.loads(messages[1]['content'])
        require(isinstance(packet, dict) and set(packet) <= {'locked_task', 'previous_complete_source',
                'public_feedback', 'attention_control', 'remaining_budget', 'source_size_limit_bytes'}
                and packet.get('previous_complete_source', '').encode() == parent,
                'generation prompt differs from frozen origin or contains extra sections')
        feedback_evidence = {}
        # Old generation records without a frozen validation_origin omitted this
        # section. Any supplied feedback, and every frozen-origin run, is checked.
        if 'public_feedback' in packet or self.ref(catalog_id, ed['manifest']).get('validation_origin'):
            from .memory_v2 import MemoryV2Manager
            provenance = self.validation_manager().provenance(binding['memory_id'])
            source = self.source(provenance['source_import_id'])
            failed, *_ = self.execution(source['failed_execution_import_id'])
            require(failed['source']['sha256'] == plan['origin_source_sha256'],
                    'generation feedback origin differs from plan')
            feedback_evidence['feedback_projection'] = self._generation_feedback(
                [('dev', packet)], failed, source['catalog_id'], self.read(source['catalog_id'], source['descriptor']))
        require(candidate.version == binding['memory_version']
                and digest(candidate.guidance.encode()) == binding['guidance_sha256'],
                'generation candidate version/guidance changed')
        treatment = binding['exposure'] == 'candidate_validation'
        notes = [{'content': candidate.guidance}] if treatment else []
        require(packet.get('attention_control') == {'memory': notes} and not request.get('memory_truncated'),
                'generation Memory must contain only the granted guidance, or no notes for control')
        exposure = self.ref(catalog_id, record['dev_memory_exposure'])
        entries = [dict(memory_id=binding['memory_id'], memory_version=binding['memory_version'],
                        guidance_sha256=binding['guidance_sha256'])] if treatment else []
        require(exposure == {'schema_version': 'attentionbench.dev-memory-exposure.v1',
                'cell': ed['cell'], 'version': ed['version'], 'model_input': provider['request'], 'entries': entries},
                'sealed Dev memory exposure receipt differs from grant/request')
        require(type(exposure['version']) is int and all(type(e['memory_version']) is int for e in exposure['entries']),
                'exposure versions must be exact integers')
        generated = output['content'].strip()
        if generated.startswith('```'):
            generated = re.sub(r'^```(?:json)?\s*|\s*```$', '', generated)
        require(digest(json.loads(generated)['source'].encode()) == execution['source']['sha256']
                == binding['policy_sha256'], 'generated source differs from executed/granted policy')
        return dict(origin_source_sha256=plan['origin_source_sha256'], dev_model=model, **feedback_evidence)

    def validation_grant(self, grant_id):
        with sqlite3.connect(self.store.path) as db:
            row = db.execute('SELECT payload FROM dev_v2_validation_grants WHERE id=?', (grant_id,)).fetchone()
        require(row is not None, 'unknown original offline validation grant')
        return json.loads(row[0])

    def _verify_validation_arm(self, execution, ed, bundle, artifacts, grant, exposure, memory_id, *, historical=False):
        binding = grant['binding']
        from .memory_v2 import MemoryV2Manager
        manager = self.validation_manager()
        plan = self.validation_plan(binding['memory_id'], binding.get('target_seed'))
        if 'target_seed' in binding:
            require(plan is not None and binding['memory_version'] == plan['memory_version'], 'cross-seed version changed')
            config = self.ref(execution['catalog_id'], execution['config'])
            require(config['seed'] == binding['target_seed'] and config.get('scene_contract') == plan['cases'][0]['scene_contract_ref']
                    and execution['context'] == plan['context'] and bundle['config_sha256'] == plan['validation_config_sha256'],
                    'cross-seed execution differs from frozen case')
        require(plan and digest(encoded(plan).encode()) == binding['plan_sha256'], 'original validation plan changed')
        require(all(binding[k] == v for k, v in {'cell': ed['cell'], 'version': ed['version'], 'seed': bundle['seed'], 'context': execution['context'], 'policy_sha256': bundle['policy_sha256'], 'config_sha256': bundle['config_sha256'], 'exposure': exposure}.items()), 'offline validation grant/raw run binding mismatch')
        require(exposure == 'none' or binding['memory_id'] == memory_id, 'candidate validation grant memory mismatch')
        ids = [] if exposure == 'none' else [memory_id]
        # Validator must bind exposure to the original executor's frozen input,
        # not infer treatment from the eventual outcome.
        input_value = self.ref(execution['catalog_id'], ed['validation_input'])
        require(input_value == {'memory_ids': ids, 'exposure': exposure, 'context': execution['context'], 'policy_sha256': bundle['policy_sha256'], 'config_sha256': bundle['config_sha256'], 'validation_grant_id': grant['grant_id']}, 'original validation exposure/config mismatch')
        candidate = self.validation_candidate(binding['memory_id'])
        require_profile_match(candidate.applicability, execution['context'])
        require_profile_match(candidate.applicability, plan)
        if (historical and plan.get('profile') == 'single_scene_v1' and plan.get('arm_generation') == 'dev_from_origin'
                and candidate.version != binding['memory_version']):
            # A task -> scene transition changes authority/version, not the
            # immutable note validated on the origin. Reverify old accepted
            # trials against the snapshot saved by task promotion, never grant
            # an old version permission for a new execution.
            decision = manager.get_decision(binding['memory_id']) or {}
            frozen = decision.get('cross_seed_validation', {}).get('origin_memory')
            provenance = manager.provenance(binding['memory_id'])
            require(frozen and provenance.get('task_demoted_at') is not None
                    and decision.get('trust_scope') in {'scene_scoped', 'task_scoped'} and candidate.status.value == 'trusted'
                    and frozen['version'] == binding['memory_version'] < candidate.version
                    and candidate.version in provenance.get('scene_trust_versions', [])
                    and all(candidate.artifact()[k] == frozen[k] for k in
                            ('guidance', 'candidate_repair', 'applicability', 'source_trace_id', 'evidence_refs', 'created_at')),
                    'historical origin note differs from the checked version')
            candidate = self.store._decode_memory(frozen)
        generation = (self._validation_generation(execution, ed, binding, plan, candidate)
                      if plan.get('arm_generation') == 'dev_from_origin' else {})
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
        return manager, plan, ids, generation

    def import_trial(self, *, import_id, catalog_id, descriptor, _persist=True):
        self.catalog(catalog_id, 'validator')
        d = self.read(catalog_id, descriptor)
        execution, ed, bundle, artifacts, cell, verdict, safety = self.verdict(d['execution_import_id'], catalog_id, d['validation_descriptor'])
        require(d['schema_version'] == 'attentionbench.dev-v2-trial.v1', 'unsupported offline trial descriptor')
        exposure = d['exposure']
        require(exposure in {'none', 'candidate_validation'}, 'only offline candidate validation exposure supported')
        grant = self.validation_grant(d['validation_grant_id'])
        binding = grant['binding']
        with sqlite3.connect(self.store.path) as db:
            prior = db.execute("SELECT payload FROM dev_v2_imports WHERE id=? AND kind='trial'", (import_id,)).fetchone()
        historical = bool(prior and json.loads(prior[0])['catalog_id'] == catalog_id
                          and json.loads(prior[0])['descriptor'] == descriptor)
        manager, plan, ids, generation = self._verify_validation_arm(
            execution, ed, bundle, artifacts, grant, exposure, d.get('memory_id'), historical=historical)
        decision = manager.get_decision(binding['memory_id'])
        if decision and 'failed_check' in decision:
            with sqlite3.connect(self.store.path) as db:
                old = db.execute("SELECT 1 FROM dev_v2_imports WHERE id=? AND kind='trial'", (import_id,)).fetchone()
            require(old is not None, 'validation batch is closed')
        costs = d['costs']
        require(costs == verdict['costs'] and all(type(costs.get(k)) in (int, float) and math.isfinite(costs[k]) and costs[k] >= 0 for k in ('credits', 'seconds', 'gpu_seconds', 'tokens')), 'independent validation costs missing')
        with self._ledger(execution['catalog_id'], ed) as db:
            settled = [json.loads(p) for state, p in db.execute('SELECT state,payload FROM v2_jobs WHERE cell=? AND version<=?', (ed['cell'], ed['version'])) if state == 'settled']
        require(costs['credits'] == sum(j['actual'].get('help_credits', 0) for j in settled) and costs['tokens'] == sum(j['actual'].get('provider_tokens', 0) for j in settled), 'validation cost differs from original settled jobs')
        require(costs['seconds'] == artifacts['sandbox_receipt']['elapsed_seconds'], 'trial time differs from native receipt')
        trial = {'run_id': bundle['run_id'], 'run': {'suite': bundle['suite'], 'task_id': bundle['task_id'], 'seed': bundle['seed'], 'policy_id': cell['policy_id'], 'execution_target': 'simulator', 'developer_model': d['developer_model'], 'budget': d['budget']}, 'success': bundle['native_success'], 'attempt_status': 'succeeded' if bundle['native_success'] else 'failed', 'evaluator_authoritative': True, 'mode': execution['context']['perception_mode'], 'memory_ids': ids, 'exposure': exposure, 'policy_sha256': bundle['policy_sha256'], 'config_sha256': bundle['config_sha256'], 'variation': {k: execution['context'][k] for k in (*AXES, 'camera_names', 'task_prompt')}, 'context': execution['context'], **costs}
        trial.update(generation)
        if plan.get('profile', 'default') in {'single_scene_v1', 'cross_seed_v1'}:
            signature = manager.provenance(binding['memory_id'])['failure_signature']
            feedback = public_trace_projection(artifacts['trace'])
            present = signature_present(feedback, signature)
            position = stop_position(artifacts['trace'])
            require(verdict.get('failure_signature') == signature and
                    verdict.get('signature_present') is present and
                    type(verdict.get('stop_position')) is int and verdict['stop_position'] == position,
                    'independent signature/stop-position verdict differs from original public trace')
            trial.update(signature_present=present, stop_position=position, validation_cell=ed['cell'])
        # Shared comparison configuration is an actual preregistered file, with
        # no arm/seed/outcome fields. The per-episode config remains SHA-bound.
        comparison = self.ref(catalog_id, d['validation_config'])
        require(comparison == verdict['validation_config'] and comparison['execution_config'] == {k: v for k, v in self.ref(execution['catalog_id'], execution['config']).items() if k != 'seed'}, 'frozen paired configuration mismatch')
        require(d['budget'] == comparison['budget'] and d['developer_model'] == comparison['developer_model'], 'trial model/budget differs from frozen comparison')
        evidence = {'kind': 'trial', 'catalog_id': catalog_id, 'descriptor': descriptor, 'attempt_id': bundle['attempt_id'], 'trial': trial, 'safety': {'uri': str(safety), 'sha256': digest(safety.read_bytes()), 'attempt_id': bundle['attempt_id']}}
        if _persist:
            self._put('dev_v2_imports', import_id, evidence, kind='trial')
        return evidence

    @memo_evidence('trial')
    def trial(self, attempt_id):
        with sqlite3.connect(self.store.path) as db:
            rows = db.execute("SELECT id,payload FROM dev_v2_imports WHERE kind='trial'").fetchall()
        for import_id, payload in rows:
            old = json.loads(payload)
            if old['attempt_id'] == attempt_id:
                trial = self.import_trial(import_id=import_id, catalog_id=old['catalog_id'], descriptor=old['descriptor'])['trial']
                descriptor = self.read(old['catalog_id'], old['descriptor'])
                binding = self.validation_grant(descriptor['validation_grant_id'])['binding']
                return {**trial, 'validation_memory_id': binding['memory_id']}
        return None

    def _result_evidence(self, grant, execution_import_id):
        execution, d, bundle, artifacts, cell = self.execution(execution_import_id)
        from .motion_permission import require_result_permission
        require_result_permission(d, require)
        grant_id = grant['grant_id']
        require(grant['development_id'] == f"{d['cell']}:v{d['version']}" and grant['context'] == d['context'], 'grant belongs to another run/cell/version/scope')
        catalog_id = execution['catalog_id']
        record = self.ref(catalog_id, d['version_record'])
        with self._ledger(catalog_id, d) as db:
            model = self._job(db, d['model_job_id'], d['cell'], d['version'], 'model')
        request, output, receipt = self._provider(catalog_id, model, self.ref(catalog_id, d['manifest']), self.ref(catalog_id, d['config']))
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
        require(grant.get('trust_scope') != 'provisional',
                'provisional exposure is excluded from Dev results')
        require(grant.get('excluded_from_results') is not True,
                'test-seeded exposure is excluded from Dev results')
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
