"""Importer-only hypothetical checks. No Memory, use grant or trust is created.

Control records, plans, grants, pairs and verdicts have dedicated tables. Shared
DevV2Evidence verifies every executor/validator byte and generation receipt;
CrossSeed supplies pairing, probes, transfer/neutral decisions and aggregation.
"""
import json
import re
import sqlite3
import time
from types import SimpleNamespace

from .applicability import require_profile_match, validate_profile, profile_fields
from .cross_seed import CrossSeed, task_conditions
from .dev_v2 import DevV2Evidence, AXES, SCOPE, digest, encoded, require, evidence_operation
from .failure_signature import derive_signature, matches_frozen_derivation
from .memory_v2 import MemoryV2Manager, validate_single_scene_plan
from .seed_guard import classify_seed

LABEL = 'test · not counted in results'


def validate_spec(spec):
    require(isinstance(spec, dict) and set(spec) == {'control_id', 'title', 'guidance', 'origin',
            'context', 'label', 'cross_seed_targets'}, 'complete positive-control spec required')
    require(isinstance(spec['control_id'], str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', spec['control_id']), 'invalid control_id')
    require(all(isinstance(spec[k], str) and spec[k].strip() for k in ('title', 'guidance'))
            and spec['label'] == LABEL, 'positive controls require guidance, title and test label')
    o = spec['origin']
    require(isinstance(o, dict) and set(o) == {'experiment_id', 'cell', 'version'}
            and all(isinstance(o[k], str) and o[k] for k in ('experiment_id', 'cell'))
            and type(o['version']) is int and o['version'] > 0, 'exact origin required')
    ctx = spec['context']; validate_profile(ctx)
    require(type(ctx.get('seed')) is int and classify_seed(ctx['seed']) == 'dev', 'Dev origin seed required')
    targets = spec['cross_seed_targets']
    require(isinstance(targets, list) and all(isinstance(t, dict) and set(t) == {'seed', 'contract'}
            and type(t['seed']) is int and classify_seed(t['seed']) == 'dev' and t['seed'] != ctx['seed']
            and isinstance(t['contract'], str) and t['contract'].startswith('/') for t in targets)
            and len({t['seed'] for t in targets}) == len(targets), 'unique planned Dev targets with absolute contract paths required')
    return digest(encoded(spec).encode())


def origin_evidence(evidence, spec, execution_import_id, catalog_id, descriptor, old=None):
    execution, ed, bundle, artifacts, cell, verdict, _ = evidence.verdict(execution_import_id, catalog_id, descriptor)
    manifest = evidence.ref(execution['catalog_id'], execution['manifest'])
    require(bundle['native_success'] is False and spec['origin'] == dict(experiment_id=manifest['experiment_id'],
            cell=execution['cell'], version=execution['version']), 'positive-control origin must be exact imported failed execution')
    require(spec['context'] == execution['context'] and spec['context']['seed'] == bundle['seed'], 'positive-control applicability must equal origin context')
    feedback = evidence.ref(execution['catalog_id'], execution['public_feedback'])
    signature = old['failure_signature'] if old else derive_signature(feedback)
    require(matches_frozen_derivation(feedback, signature), 'positive-control frozen signature differs from public feedback')
    require(verdict.get('failure_signature') == signature and verdict.get('signature_present') is True,
            'origin validator signature differs from public feedback')
    return execution, cell, signature


class ControlView(SimpleNamespace):
    """Runtime input shape only; never a MemoryRecord or persisted memory status."""
    def artifact(self):
        return vars(self).copy()


class PositiveEvidence(DevV2Evidence):
    def __init__(self, controls):
        self.controls = controls
        super().__init__(controls.store)

    def validation_manager(self): return self.controls
    def validation_candidate(self, memory_id): return self.controls.get_memory(memory_id)
    def validation_plan(self, memory_id, target_seed=None):
        return self.controls.cross_seed.get('plans', memory_id, target_seed) if target_seed is not None else self.controls.get_plan(memory_id)

    def source(self, import_id):
        # A private origin view, never inserted into dev_v2 source imports.
        r = self.controls.get(import_id); e = self.get(r['execution_import_id'])
        return dict(context=e['context'], failed_execution_import_id=r['execution_import_id'],
                    catalog_id=e['catalog_id'], descriptor=e['descriptor'])

    def validation_grant(self, grant_id):
        with sqlite3.connect(self.store.path) as db:
            row = db.execute('SELECT payload FROM positive_control_grants WHERE id=?', (grant_id,)).fetchone()
        require(row is not None, 'unknown positive-control validation grant')
        return json.loads(row[0])


class PositiveCrossSeed(CrossSeed):
    prefix = 'positive_control_cross_seed'
    grants_table = 'positive_control_grants'

    def get(self, kind, memory_id, seed, db=None):
        r = self.service.get(memory_id)
        if kind == 'plans' and seed == r['spec']['context']['seed']:
            p = self.service.get_plan(memory_id)
            return dict(p, context=r['spec']['context'], memory_version=1, role='transfer', target_seed=seed) if p else None
        return super().get(kind, memory_id, seed, db)

    def candidate(self, memory_id): return self.service.get_memory(memory_id)

    def _eligible(self, memory_id, db, *, version=None):
        require(version in (None, 1), 'positive-control version is immutable')
        r = self.service.get(memory_id)
        d = self.service.get_decision(memory_id)
        require(d is None or d['decision'] == 'would_pass', 'positive-control check is closed')
        return self.service.get_memory(memory_id).artifact(), self.service.provenance(memory_id)

    def _target(self, memory_id, seed, execution, contract_ref):
        r = self.service.get(memory_id)
        require((self.service.get_decision(memory_id) or {}).get('decision') == 'would_pass', 'origin must would-pass before cross-seed')
        require(any(t['seed'] == seed and t['contract'] == contract_ref['uri'] for t in r['spec']['cross_seed_targets']), 'target outside frozen spec')
        return super()._target(memory_id, seed, execution, contract_ref)

    def _conditions(self, db, memory_id):
        r = self.service.get(memory_id)
        d = self.service.get_decision(memory_id) or {}
        scenes = [json.loads(row[0]) for row in db.execute(
            f'SELECT payload FROM {self.prefix}_verdicts WHERE memory_id=? AND seed!=? ORDER BY seed',
            (memory_id, r['spec']['context']['seed']))]
        for scene in scenes: require_profile_match(r['spec']['context'], scene)
        # Inputs to the shared hypothetical aggregation, never stored as Memory.
        m = dict(version=1, status='trusted' if d.get('decision') == 'would_pass' else 'candidate')
        result = task_conditions(m, dict(trust_scope='scene_scoped'), scenes)
        result['trust_scope'] = None
        result['verdict'] = 'would be trusted for this task' if not result['missing'] else 'missing checks'
        return result

    def _close(self, db, memory_id, seed, verdict, reason, extra):
        r = self.service.get(memory_id)
        if seed == r['spec']['context']['seed'] and 'failed_check' not in extra:
            return None  # C16 evaluate re-verifies all five pairs before closure.
        value = dict(**profile_fields(r['spec']['context']), memory_id=memory_id, memory_version=1,
                     target_seed=seed, verdict=verdict, reason=reason, decided_at=time.time(), **extra)
        if seed == r['spec']['context']['seed']:
            checks=extra['failed_check']['checks']
            gate='safety_regression' if verdict=='transfer_failed' else ('scene_reproduction_failed' if not checks['scene_reproduction'] else 'control_safety_failed')
            pairs=[json.loads(row[0]) for row in db.execute(
                f'SELECT payload FROM {self.prefix}_pairs WHERE memory_id=? AND seed=?',(memory_id,seed))]
            value.update(record_kind='positive_control',label=LABEL,excluded_from_results=True,
                         decision='would_fail' if verdict == 'transfer_failed' else 'inconclusive',
                         failed_gates=[gate],gates={gate:False},report=self._report(pairs))
            db.execute('INSERT INTO positive_control_decisions VALUES (?,?)', (memory_id, encoded(value)))
        db.execute(f'INSERT INTO {self.prefix}_verdicts VALUES (?,?,?)', (memory_id, seed, encoded(value)))
        if seed != r['spec']['context']['seed'] and not self._conditions(db, memory_id)['missing']:
            self._verify_history(db, memory_id, 1)
        return value

    def grant_binding(self, binding, plan):
        mid=binding['memory_id']
        if binding['target_seed']==self.service.get(mid)['spec']['context']['seed']:
            binding.pop('target_seed')
            binding['plan_sha256']=digest(encoded(self.service.get_plan(mid)).encode())
        return binding

    def pair_target(self,binding):
        return binding.get('target_seed',self.service.get(binding['memory_id'])['spec']['context']['seed'])

    def authorize(self, **args):
        mid = args['memory_id']; seed = args['target_seed']; r = self.service.get(mid)
        if seed != r['spec']['context']['seed']:
            require((self.service.get_decision(mid) or {}).get('decision') == 'would_pass', 'origin must would-pass')
        else:
            require(self.service.get_decision(mid) is None, 'same-scene check is closed')
        return super().authorize(**args)


class PositiveControls:
    _single_scene_failures = MemoryV2Manager._single_scene_failures

    def __init__(self, service):
        self.service, self.store = service, service.store
        with sqlite3.connect(self.store.path) as db:
            for table in ('records', 'plans', 'decisions'):
                db.execute(f'CREATE TABLE IF NOT EXISTS positive_control_{table} (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS positive_control_grants '
                       '(id TEXT PRIMARY KEY, memory_id TEXT, cell TEXT, version INTEGER, payload TEXT NOT NULL, UNIQUE(memory_id,cell,version))')
        self.v2 = self
        self.dev_v2 = PositiveEvidence(self)
        self.cross_seed = PositiveCrossSeed(self)

    def _read(self, table, mid):
        with sqlite3.connect(self.store.path) as db:
            row = db.execute(f'SELECT payload FROM positive_control_{table} WHERE id=?', (mid,)).fetchone()
        return json.loads(row[0]) if row else None

    def get(self, mid):
        r = self._read('records', mid)
        require(r is not None, 'unknown positive control')
        return r

    @evidence_operation
    def create(self, *, spec, spec_sha256, execution_import_id, catalog_id, descriptor):
        require(validate_spec(spec) == spec_sha256, 'positive-control spec digest mismatch')
        execution, cell, signature = origin_evidence(self.dev_v2, spec, execution_import_id, catalog_id, descriptor, self._read('records', spec['control_id']))
        value = dict(record_kind='positive_control', label=LABEL, excluded_from_results=True, spec=spec,
                     spec_sha256=spec_sha256, execution_import_id=execution_import_id, catalog_id=catalog_id,
                     descriptor=descriptor, failure_signature=signature, source_policy_sha256=execution['source']['sha256'],
                     policy_id=cell['policy_id'])
        mid = spec['control_id']
        with self.store._transaction() as db:
            old = db.execute('SELECT payload FROM positive_control_records WHERE id=?', (mid,)).fetchone()
            require(old is None or json.loads(old[0]) == value, 'positive-control spec/origin is immutable')
            db.execute('INSERT OR IGNORE INTO positive_control_records VALUES (?,?)', (mid, encoded(value)))
        return value

    def verify_source_evidence(self, memory_id):
        r = self.get(memory_id)
        self.create(**{k: r[k] for k in ('spec', 'spec_sha256', 'execution_import_id', 'catalog_id', 'descriptor')})

    def get_memory(self, memory_id):
        r = self.get(memory_id)
        return ControlView(memory_id=memory_id, version=1, record_kind='positive_control', status='positive_control',
                           guidance=r['spec']['guidance'], applicability=r['spec']['context'], expires_at=None)

    def provenance(self, memory_id):
        r = self.get(memory_id); o = r['spec']['origin']; e = self.dev_v2.get(r['execution_import_id'])
        return dict(channel='positive_control', record_kind='positive_control', failure_signature=r['failure_signature'],
                    artifact=dict(applicability=r['spec']['context']), source_seed=r['spec']['context']['seed'],
                    source_policy_sha256=r['source_policy_sha256'], source_policy_id=r['policy_id'], source_import_id=memory_id,
                    origin=dict(experiment_id=o['experiment_id'], cell=o['cell'], origin_version=o['version'], policy_id=r['policy_id'],
                                digests=dict(origin_source=r['source_policy_sha256'], origin_public_feedback=e['public_feedback']['sha256'])))

    def get_plan(self, memory_id): return self._read('plans', memory_id)
    def get_decision(self, memory_id): return self._read('decisions', memory_id)
    decision = get_decision

    @evidence_operation
    def record_plan(self, memory_id, plan):
        self.verify_source_evidence(memory_id)
        r = self.get(memory_id); ctx = r['spec']['context']; validate_profile(plan); require_profile_match(ctx, plan)
        require(plan.get('schema_version') == 'attentionbench.memory-validation-plan.v2'
                and plan.get('memory_id') == memory_id and plan.get('policy_id') == r['policy_id']
                and plan.get('profile') == 'single_scene_v1' and plan.get('arm_generation') == 'dev_from_origin'
                and len(plan.get('pairs', [])) == 5, 'exact single_scene_v1 dev_from_origin plan required')
        require(plan['validation_config_sha256'] == self.dev_v2.get(r['execution_import_id'])['config']['sha256'],
                'positive-control plan must use the exact origin config')
        case = dict(seed=ctx['seed'], **{k:ctx[k] for k in (*AXES, 'camera_names', 'task_prompt')})
        validate_single_scene_plan(plan, ctx, ctx['seed'], case, r['failure_signature'], r['source_policy_sha256'])
        with self.store._transaction() as db:
            old = db.execute('SELECT payload FROM positive_control_plans WHERE id=?', (memory_id,)).fetchone()
            require(old is None or json.loads(old[0]) == plan, 'positive-control plan is immutable')
            db.execute('INSERT OR IGNORE INTO positive_control_plans VALUES (?,?)', (memory_id, encoded(plan)))
        return plan

    def authorize_dev_v2_validation(self, **args):
        mid = args['memory_id']; seed = args.pop('target_seed', self.get(mid)['spec']['context']['seed'])
        grant = self.cross_seed.authorize(**args, target_seed=seed)
        return grant

    def record_dev_v2_pair(self, **args):
        return self.cross_seed.pair(**args, target_seed=self.get(args['memory_id'])['spec']['context']['seed'])

    def record_dev_v2_failed_check(self, **args):
        return self.cross_seed.failed_check(**args, target_seed=self.get(args['memory_id'])['spec']['context']['seed'])

    def pairs(self, memory_id):
        with sqlite3.connect(self.store.path) as db:
            return [dict(seed=seed, **json.loads(raw)) for seed, raw in db.execute(
                'SELECT seed,payload FROM positive_control_cross_seed_pairs WHERE memory_id=? ORDER BY seed,pair_id', (memory_id,))]

    @evidence_operation
    def evaluate(self, memory_id):
        self.verify_source_evidence(memory_id)
        old = self.get_decision(memory_id)
        if old: return old
        r = self.get(memory_id); plan = self.get_plan(memory_id)
        require(plan is not None, 'freeze a positive-control plan first')
        pairs = [p for p in self.pairs(memory_id) if p['seed'] == r['spec']['context']['seed']]
        require(len(pairs) == 5, 'five complete positive-control pairs required')
        with self.store._transaction() as db:
            self.cross_seed._verify_history(db, memory_id, 1)
            report = CrossSeed._report(pairs)
            failures = self._single_scene_failures(report, plan)
            value = dict(record_kind='positive_control', label=LABEL, excluded_from_results=True,
                         decision='would_fail' if failures else 'would_pass', failed_gates=failures,
                         gates={k:k not in failures for k in ('insufficient_pairs','mixed_policy_versions','mixed_validation_configuration',
                             'safety_regression','control_signature_not_reproduced','treatment_signature_remains','treatment_stopped_earlier')},
                         report=report, decided_at=time.time(), spec_sha256=r['spec_sha256'],
                         evidence=dict(origin_execution_import_id=r['execution_import_id'],origin_validator_catalog_id=r['catalog_id'],
                             origin_descriptor=r['descriptor'],pairs=[{k:p[k] for k in ('pair_id','control_import_id','treatment_import_id')} for p in pairs]))
            old = db.execute('SELECT payload FROM positive_control_decisions WHERE id=?', (memory_id,)).fetchone()
            if old:return json.loads(old[0])
            db.execute('INSERT INTO positive_control_decisions VALUES (?,?)', (memory_id, encoded(value)))
        return value

    def public(self):
        with sqlite3.connect(self.store.path) as db:
            ids = [r[0] for r in db.execute('SELECT id FROM positive_control_records ORDER BY rowid')]
        return [dict(control_id=mid, title=self.get(mid)['spec']['title'], label=LABEL, record_kind='positive_control',
                     verdict=self.get_decision(mid), pairs=self.pairs(mid),
                     cross_seed=self.cross_seed.status(memory_id=mid)) for mid in ids]

    def read(self, control_id, action):
        require(action in {'record', 'provenance', 'input', 'plan', 'decision', 'cross-status', 'pairs'}, 'unknown positive-control read')
        if action == 'pairs': return self.pairs(control_id)
        if action == 'record': return self._read('records', control_id)
        if action == 'input': return self.get_memory(control_id).artifact()
        if action == 'plan': return self.get_plan(control_id)
        if action == 'decision': return self.get_decision(control_id)
        if action == 'provenance': return self.provenance(control_id)
        return self.cross_seed.status(memory_id=control_id)

    def write(self, action, payload):
        routes = {'create': self.create, 'plans': self.record_plan, 'pairs': self.record_dev_v2_pair,
                  'grants': self.authorize_dev_v2_validation, 'trials': self.dev_v2.import_trial,
                  'failed-checks': self.record_dev_v2_failed_check, 'evaluate': self.evaluate,
                  'cross-probes': self.cross_seed.probe, 'cross-plans': self.cross_seed.plan,
                  'cross-pairs': self.cross_seed.pair, 'cross-failed-checks': self.cross_seed.failed_check}
        require(action in routes, 'unknown positive-control write')
        return routes[action](**payload)
