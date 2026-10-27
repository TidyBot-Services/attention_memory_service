"""Sealed cross-seed checks. Origin plans and their admission rule stay frozen."""
import json
import sqlite3
import time
import uuid
from pathlib import Path

from .applicability import require_profile_match, validate_profile, config_profile, profile_fields
from .auto_validation import lifecycle, rank_validation_queue
from .dev_v2 import AXES, SCOPE, SHA, digest, encoded, evidence_operation, require, public_trace_projection, signature_present
from .scene_contract import V2
from .seed_guard import classify_seed


class CrossSeed:
    prefix = "memory_cross_seed"
    grants_table = "dev_v2_validation_grants"
    def __init__(self, service):
        self.service, self.store, self.evidence = service, service.store, service.dev_v2
        with sqlite3.connect(self.store.path) as db:
            for kind in ('probes', 'plans', 'verdicts'):
                db.execute(f'CREATE TABLE IF NOT EXISTS {self.prefix}_{kind} '
                           '(memory_id TEXT, seed INTEGER, payload TEXT NOT NULL, PRIMARY KEY(memory_id,seed))')
            db.execute(f'CREATE TABLE IF NOT EXISTS {self.prefix}_pairs '
                       '(memory_id TEXT, seed INTEGER, pair_id TEXT, payload TEXT NOT NULL, '
                       'PRIMARY KEY(memory_id,seed,pair_id))')

    def get(self, kind, memory_id, seed, db=None):
        if db is None:
            with sqlite3.connect(self.store.path) as db:
                return self.get(kind, memory_id, seed, db)
        row = db.execute(f'SELECT payload FROM {self.prefix}_{kind} WHERE memory_id=? AND seed=?',
                         (memory_id, seed)).fetchone()
        return json.loads(row[0]) if row else None

    def candidate(self, memory_id): return self.store.get_memory(memory_id)

    def _eligible(self, memory_id, db, *, version=None):
        row = db.execute('SELECT payload FROM memories WHERE id=?', (memory_id,)).fetchone()
        require(row is not None, 'unknown memory')
        m = json.loads(row[0])
        row = db.execute('SELECT payload FROM memory_v2_provenance WHERE memory_id=?', (memory_id,)).fetchone()
        p = json.loads(row[0]) if row else {}
        d = db.execute('SELECT payload FROM memory_v2_decisions WHERE memory_id=?', (memory_id,)).fetchone()
        plan = db.execute('SELECT payload FROM memory_v2_plans WHERE memory_id=?', (memory_id,)).fetchone()
        require(m['status'] == 'trusted' and p.get('channel') == 'dev_v2' and plan and d
                and json.loads(plan[0]).get('profile') == 'single_scene_v1'
                and json.loads(d[0]).get('trust_scope') == 'scene_scoped'
                and (version is None or m['version'] == version)
                and (m.get('expires_at') is None or m['expires_at'] > time.time()),
                'cross-seed checks require a current scene-trusted single_scene_v1 note')
        return m, p

    def _target(self, memory_id, seed, execution, contract_ref):
        require(type(seed) is int and classify_seed(seed) == 'dev', 'target seed must be a development seed')
        p = self.service.v2.provenance(memory_id)
        require(seed != p['source_seed'], 'target cannot be the origin seed')
        origin = self.evidence.source(p['source_import_id'])['context']
        ctx = execution['context']
        require_profile_match(p['artifact']['applicability'], ctx)
        require_profile_match(origin, ctx)
        require(all(ctx.get(k) == origin.get(k) for k in (*SCOPE, 'camera_names', 'task_prompt') if k not in AXES),
                'target belongs to another task or source scope')
        require(isinstance(origin.get('sdk_interface_sha256'), str)
                and ctx.get('sdk_interface_sha256') == origin['sdk_interface_sha256'], 'target SDK interface mismatch')
        config = self.evidence.ref(execution['catalog_id'], execution['config'])
        require_profile_match(ctx, config_profile(config))
        require(config['seed'] == seed and config.get('scene_contract') == contract_ref,
                'target seed/scene contract ref mismatch')
        # Deliberately require the M-8 contract role, even when an executor or
        # source catalog also happens to contain these bytes.
        with sqlite3.connect(self.store.path) as db:
            catalogs = db.execute('SELECT id,payload FROM dev_v2_catalogs').fetchall()
        found = None
        for cid, payload in catalogs:
            catalog = json.loads(payload)
            path = Path(contract_ref['uri'])
            if catalog['role'] == 'contract' and path.is_relative_to(catalog['root']):
                name = str(path.relative_to(catalog['root']))
                if catalog['files'].get(name) == contract_ref['sha256']:
                    found = self.evidence.ref(cid, contract_ref)
                    break
        require(found is not None, 'target needs a registered contract catalog at the exact digest')
        require(found.get('schema') == V2 and str(seed) in found.get('tasks', {}).get(ctx['task_id'], {}).get('seeds', {}),
                'target requires a seed-keyed scene contract')
        return config

    @evidence_operation
    def probe(self, *, memory_id, target_seed, execution_import_id, catalog_id, descriptor, scene_contract_ref):
        args = dict(memory_id=memory_id, target_seed=target_seed, execution_import_id=execution_import_id,
                    catalog_id=catalog_id, descriptor=descriptor, scene_contract_ref=scene_contract_ref)
        with sqlite3.connect(self.store.path) as db:
            old = self.get('probes', memory_id, target_seed, db)
            if old:
                require(old['request'] == args, 'target seed already closed or bound to another probe')
                memory = self.candidate(memory_id).artifact()
                p = self.service.v2.provenance(memory_id)
            else:
                memory, p = self._eligible(memory_id, db)
        self.service.v2.verify_source_evidence(memory_id)
        execution, ed, bundle, artifacts, _, verdict, safety = self.evidence.verdict(
            execution_import_id, catalog_id, descriptor, for_validation=False)
        self._target(memory_id, target_seed, execution, scene_contract_ref)
        for ref in verdict.get('scene_replay_states', []):
            self.evidence.ref(catalog_id, ref)
        require(execution['source']['sha256'] == p['source_policy_sha256'], 'probe must execute frozen origin source')
        clean = self.evidence.ref(execution['catalog_id'], ed['validation_input'])
        require(clean == dict(memory_ids=[], exposure='none', context=execution['context'],
                             policy_sha256=bundle['policy_sha256'], config_sha256=bundle['config_sha256']),
                'probe must read no notes')
        # A probe executes the frozen code directly: no model/advice dispatch.
        with self.evidence._ledger(execution['catalog_id'], ed) as db:
            jobs = [json.loads(r[0]) for r in db.execute('SELECT payload FROM v2_jobs WHERE cell=? AND version<=?',
                                                        (ed['cell'], ed['version']))]
        require(all(j['kind'] == 'execute' for j in jobs)
                and not artifacts['trace'].get('memory_validation') and not artifacts['trace'].get('memory_uses'),
                'probe must execute origin directly with clean Memory')
        present = signature_present(public_trace_projection(artifacts['trace']), p['failure_signature'])
        unsafe = json.loads(safety.read_text())['unsafe_attempts']
        role = ('could_not_finish' if not verdict['checks']['scene_reproduction'] or not verdict['checks']['safety'] or unsafe
                else 'transfer' if present else 'neutral')
        version = old['memory_version'] if old else memory['version']
        value = dict(request=args, memory_id=memory_id, memory_version=version, target_seed=target_seed,
                     role=role, signature_present=present, execution=execution,
                     scene_contract_ref=scene_contract_ref, descriptor_sha256=self.evidence.catalog(catalog_id)['files'][descriptor])
        if old:
            require(old == value, 'probe evidence changed')
            return old
        with self.store._transaction() as db:
            old = self.get('probes', memory_id, target_seed, db)
            if old:
                require(old == value, 'conflicting probe')
                return old
            self._eligible(memory_id, db, version=memory['version'])
            db.execute(f'INSERT INTO {self.prefix}_probes VALUES (?,?,?)', (memory_id, target_seed, encoded(value)))
            if role == 'could_not_finish':
                self._close(db, memory_id, target_seed, role, 'Could not finish: probe safety or scene reproduction failed', {})
        return value

    @evidence_operation
    def plan(self, *, memory_id, target_seed, pairs, dev_model):
        require(isinstance(dev_model, dict) and set(dev_model) == {'provider', 'model'}
                and all(isinstance(v, str) and v.strip() for v in dev_model.values()), 'explicit Dev model required')
        with sqlite3.connect(self.store.path) as db:
            probe = self.get('probes', memory_id, target_seed, db)
            require(probe and probe['role'] in {'transfer', 'neutral'}, 'a successful probe is required')
            count = 5 if probe['role'] == 'transfer' else 3
            require(isinstance(pairs, list) and len(pairs) == count and all(isinstance(p, dict)
                    and set(p) == {'pair_id', 'control_cell', 'treatment_cell'}
                    and all(isinstance(v, str) and v for v in p.values()) for p in pairs)
                    and len({p['pair_id'] for p in pairs}) == count
                    and len({p[k] for p in pairs for k in ('control_cell', 'treatment_cell')}) == 2 * count,
                    f'cross-seed plan requires exactly {count} distinct pairs')
            old = self.get('plans', memory_id, target_seed, db)
            if old:
                require(old['pairs'] == pairs and old['dev_model'] == dev_model, 'cross-seed plan is immutable')
                self.probe(**probe['request'])
                return old
            memory, p = self._eligible(memory_id, db, version=probe['memory_version'])
        self.service.v2.verify_source_evidence(memory_id)
        request = probe['request']
        execution, *_ = self.evidence.verdict(request['execution_import_id'], request['catalog_id'], request['descriptor'])
        self._target(memory_id, target_seed, execution, probe['scene_contract_ref'])
        origin_plan = self.service.v2.get_plan(memory_id)
        ctx = execution['context']
        value = dict(**profile_fields(ctx), profile='cross_seed_v1', memory_id=memory_id, memory_version=memory['version'],
                     role=probe['role'], target_seed=target_seed, pairs=pairs, dev_model=dev_model,
                     arm_generation='dev_from_origin', origin_source_sha256=p['source_policy_sha256'],
                     validation_config_sha256=execution['config']['sha256'], context=ctx,
                     cases=[dict(seed=target_seed, scene_contract_ref=probe['scene_contract_ref'],
                                 **{k: ctx[k] for k in (*AXES, 'camera_names', 'task_prompt')})],
                     seeds=[target_seed], failure_signature=p['failure_signature'],
                     policy_id=origin_plan['policy_id'], assistance_credits=origin_plan['assistance_credits'])
        with self.store._transaction() as db:
            self._eligible(memory_id, db, version=memory['version'])
            require(self.get('verdicts', memory_id, target_seed, db) is None, 'target seed is closed')
            old = self.get('plans', memory_id, target_seed, db)
            require(old is None or old == value, 'conflicting cross-seed plan')
            db.execute(f'INSERT OR IGNORE INTO {self.prefix}_plans VALUES (?,?,?)', (memory_id, target_seed, encoded(value)))
        return value

    def grant_binding(self, binding, plan): return binding

    def pair_target(self, binding): return binding.get('target_seed')

    def authorize(self, *, memory_id, target_seed, cell, version, seed, context,
                  policy_sha256, config_sha256, exposure, origin_source_sha256=None):
        validate_profile(context)
        plan = self.get('plans', memory_id, target_seed)
        if plan is not None:
            require_profile_match(plan['context'], context)
        require(plan and type(seed) is int and seed == target_seed and context == plan['context'], 'grant outside cross-seed case')
        require(type(version) is int and version > 0 and exposure in {'none', 'candidate_validation'}, 'invalid cross-seed dispatch')
        arm = 'control_cell' if exposure == 'none' else 'treatment_cell'
        require(any(p[arm] == cell for p in plan['pairs']) and isinstance(policy_sha256, str) and SHA.fullmatch(policy_sha256)
                and config_sha256 == plan['validation_config_sha256'] and origin_source_sha256 == plan['origin_source_sha256'],
                'grant differs from frozen cross-seed plan')
        with self.store._transaction() as db:
            memory, _ = self._eligible(memory_id, db, version=plan['memory_version'])
            require_profile_match(memory['applicability'], context)
            require(self.get('verdicts', memory_id, target_seed, db) is None, 'target seed is closed')
            binding = dict(**profile_fields(context), memory_id=memory_id, memory_version=memory['version'], target_seed=target_seed,
                cell=cell, version=version, seed=seed, context=context, policy_sha256=policy_sha256,
                config_sha256=config_sha256, exposure=exposure, plan_sha256=digest(encoded(plan).encode()),
                origin_source_sha256=origin_source_sha256, dev_model=plan['dev_model'], guidance_sha256=digest(memory['guidance'].encode()))
            binding = self.grant_binding(binding, plan)
            old = db.execute(f'SELECT payload FROM {self.grants_table} WHERE memory_id=? AND cell=? AND version=?',
                             (memory_id, cell, version)).fetchone()
            if old:
                grant = json.loads(old[0]); require(grant['binding'] == binding, 'conflicting validation dispatch')
                return grant
            grant = dict(schema_version='attentionbench.dev-v2-validation-grant.v1', grant_id='validation:' + uuid.uuid4().hex,
                         issued_at=time.time(), binding=binding)
            db.execute(f'INSERT INTO {self.grants_table} VALUES (?,?,?,?,?)',
                       (grant['grant_id'], memory_id, cell, version, encoded(grant)))
        return grant

    @evidence_operation
    def pair(self, *, memory_id, target_seed, control_import_id, treatment_import_id):
        plan = self.get('plans', memory_id, target_seed)
        require(plan is not None, 'cross-seed plan required')
        self.service.v2.verify_source_evidence(memory_id)
        arms, unsafe = [], []
        for import_id in (control_import_id, treatment_import_id):
            old = self.evidence.get(import_id)
            require(old['kind'] == 'trial', 'offline trial import required')
            result = self.evidence.import_trial(import_id=import_id, catalog_id=old['catalog_id'], descriptor=old['descriptor'])
            d = self.evidence.read(old['catalog_id'], old['descriptor'])
            binding = self.evidence.validation_grant(d['validation_grant_id'])['binding']
            require(self.pair_target(binding) == target_seed and binding['memory_id'] == memory_id
                    and binding['memory_version'] == plan['memory_version'], 'pair belongs to another cross-seed plan')
            arms.append(result['trial'])
            unsafe.append(json.loads(Path(result['safety']['uri']).read_text())['unsafe_attempts'])
        c, t = arms
        match = next((p for p in plan['pairs'] if p['control_cell'] == c['validation_cell']
                      and p['treatment_cell'] == t['validation_cell']), None)
        require(match is not None and c['run_id'] != t['run_id'], 'arms do not match a frozen cross-seed pair')
        require(c['memory_ids'] == [] and t['memory_ids'] == [memory_id]
                and c['exposure'] == 'none' and t['exposure'] == 'candidate_validation', 'pair needs clean Memory')
        require(c['run'] == {**t['run']} and c['run']['seed'] == target_seed
                and c['run']['policy_id'] == plan['policy_id']
                and c['run']['budget']['assistance_credits'] == plan['assistance_credits'], 'paired configuration differs')
        value = dict(pair_id=match['pair_id'], control_import_id=control_import_id, treatment_import_id=treatment_import_id,
                     control=c, treatment=t, control_unsafe=unsafe[0], treatment_unsafe=unsafe[1])
        with self.store._transaction() as db:
            row = db.execute(f'SELECT payload FROM {self.prefix}_pairs WHERE memory_id=? AND seed=? AND pair_id=?',
                             (memory_id, target_seed, match['pair_id'])).fetchone()
            if row:
                require(json.loads(row[0]) == value, 'conflicting cross-seed pair')
                return value
            self._eligible(memory_id, db, version=plan['memory_version'])
            require(self.get('verdicts', memory_id, target_seed, db) is None, 'target seed is closed')
            db.execute(f'INSERT INTO {self.prefix}_pairs VALUES (?,?,?,?)',
                       (memory_id, target_seed, match['pair_id'], encoded(value)))
            pairs = [json.loads(r[0]) for r in db.execute(f'SELECT payload FROM {self.prefix}_pairs WHERE memory_id=? AND seed=?',
                                                        (memory_id, target_seed))]
            if len(pairs) == len(plan['pairs']):
                report = self._report(pairs)
                failures = (self.service.v2._single_scene_failures(report, plan) if plan['role'] == 'transfer'
                            else self._no_harm(report))
                verdict = ('transfer_failed' if failures else 'transfer_passed') if plan['role'] == 'transfer' else (
                    'neutral_harm' if failures else 'neutral_no_harm')
                messages = dict(safety_regression='safety worsened with the note', treatment_stopped_earlier='a run with the note stopped earlier',
                                control_signature_not_reproduced='the target failure did not reproduce enough without the note',
                                treatment_signature_remains='the target failure remained too often with the note')
                reason = '; '.join(messages.get(f, f.replace('_', ' ')) for f in failures) or (
                    'The note worked on this seed' if plan['role'] == 'transfer' else 'No harm on this seed')
                self._close(db, memory_id, target_seed, verdict, reason, dict(report=report, failed_gates=failures))
        return value

    @staticmethod
    def _report(pairs):
        c, t = [p['control'] for p in pairs], [p['treatment'] for p in pairs]
        arms = c + t
        return dict(paired_runs=len(pairs), validation_origin_hashes=sorted({a['origin_source_sha256'] for a in arms}),
                    validation_dev_models=[json.loads(m) for m in sorted({encoded(a['dev_model']) for a in arms})],
                    validation_config_hashes=sorted({a['config_sha256'] for a in arms}),
                    safety_regressions=sum(p['treatment_unsafe'] > p['control_unsafe'] for p in pairs),
                    control_unsafe_attempts=sum(p['control_unsafe'] for p in pairs),
                    treatment_unsafe_attempts=sum(p['treatment_unsafe'] for p in pairs),
                    control_signature_count=sum(a['signature_present'] for a in c),
                    treatment_signature_count=sum(a['signature_present'] for a in t),
                    signature_arms=[dict(treatment_not_earlier=p['treatment']['stop_position'] >= p['control']['stop_position']) for p in pairs])

    @staticmethod
    def _no_harm(report):
        failures = []
        if report['safety_regressions'] or report['treatment_unsafe_attempts'] > report['control_unsafe_attempts']:
            failures.append('safety_regression')
        if any(not a['treatment_not_earlier'] for a in report['signature_arms']):
            failures.append('treatment_stopped_earlier')
        return failures

    @evidence_operation
    def failed_check(self, *, memory_id, target_seed, validation_grant_id, execution_import_id, catalog_id, descriptor):
        grant = self.evidence.validation_grant(validation_grant_id)
        require(grant['binding']['memory_id'] == memory_id and self.pair_target(grant['binding']) == target_seed,
                'failed check belongs to another cross-seed plan')
        execution, ed, bundle, artifacts, _, verdict, _ = self.evidence.verdict(
            execution_import_id, catalog_id, descriptor, for_validation=False)
        _, plan, _, _ = self.evidence._verify_validation_arm(execution, ed, bundle, artifacts, grant,
                                                          grant['binding']['exposure'], memory_id)
        for ref in verdict.get('scene_replay_states', []):
            self.evidence.ref(catalog_id, ref)
        checks = verdict['checks']
        require(not checks['scene_reproduction'] or not checks['safety'], 'all checks passed; use normal trial path')
        self.service.v2.verify_source_evidence(memory_id)
        failed = dict(validation_grant_id=validation_grant_id, execution_import_id=execution_import_id,
                      catalog_id=catalog_id, descriptor=descriptor,
                      descriptor_sha256=self.evidence.catalog(catalog_id)['files'][descriptor], checks=checks)
        # As in M-8, treatment safety failure is harm; a failed replay or a
        # control safety failure is an inconclusive infrastructure exit.
        harm = checks['scene_reproduction'] and grant['binding']['exposure'] == 'candidate_validation'
        role = self.get('plans', memory_id, target_seed)['role']
        status = ('neutral_harm' if role == 'neutral' else 'transfer_failed') if harm else 'could_not_finish'
        reason = ('Safety worsened with the note' if harm else 'Could not finish: scene reproduction or control safety failed')
        with self.store._transaction() as db:
            old = self.get('verdicts', memory_id, target_seed, db)
            if old:
                require(old.get('failed_check') == failed, 'conflicting closed cross-seed check')
                return old
            self._eligible(memory_id, db, version=grant['binding']['memory_version'])
            return self._close(db, memory_id, target_seed, status, reason, dict(failed_check=failed))

    def _conditions(self, db, memory_id):
        m = json.loads(db.execute('SELECT payload FROM memories WHERE id=?', (memory_id,)).fetchone()[0])
        row = db.execute('SELECT payload FROM memory_v2_decisions WHERE memory_id=?', (memory_id,)).fetchone()
        d = json.loads(row[0]) if row else {}
        scenes = [json.loads(r[0]) for r in db.execute(f'SELECT payload FROM {self.prefix}_verdicts WHERE memory_id=? ORDER BY seed', (memory_id,))]
        for scene in scenes:
            require_profile_match(m['applicability'], scene)
        return task_conditions(m, d, scenes)

    def _close(self, db, memory_id, seed, verdict, reason, extra):
        memory, p = self._eligible(memory_id, db)
        value = dict(**profile_fields(memory['applicability']), memory_id=memory_id,
                     memory_version=memory['version'], target_seed=seed, verdict=verdict, reason=reason, decided_at=time.time(), **extra)
        db.execute(f'INSERT INTO {self.prefix}_verdicts VALUES (?,?,?)', (memory_id, seed, encoded(value)))
        conditions = self._conditions(db, memory_id)
        if not conditions['missing']:
            self._verify_history(db, memory_id, memory['version'])
            scenes = [json.loads(r[0]) for r in db.execute(f'SELECT payload FROM {self.prefix}_verdicts WHERE memory_id=? ORDER BY seed', (memory_id,))]
            worked = ', '.join(str(s['target_seed']) for s in scenes if s['verdict'] == 'transfer_passed')
            neutral = ', '.join(str(s['target_seed']) for s in scenes if s['verdict'] == 'neutral_no_harm')
            reason = f'Trusted for this task: worked on seed {worked}' + (f', no harm on seed {neutral}' if neutral else '') + f" ({conditions['scenes']} scenes)"
            decision = json.loads(db.execute('SELECT payload FROM memory_v2_decisions WHERE memory_id=?', (memory_id,)).fetchone()[0])
            origin_memory = decision.get('cross_seed_validation', {}).get('origin_memory', memory)
            decision.update(trust_scope='task_scoped', cross_seed_validation=dict(profile='cross_seed_v1',
                            memory_version=memory['version'], origin_memory=origin_memory, scenes=scenes,
                            transfer_failed=conditions['transfer_failed'], reason=reason))
            p.update(trust_scope='task_scoped')
            db.execute('UPDATE memory_v2_decisions SET payload=? WHERE memory_id=?', (encoded(decision), memory_id))
            db.execute('UPDATE memory_v2_provenance SET payload=? WHERE memory_id=?', (encoded(p), memory_id))
            lifecycle(db, memory, p, 'promoted', reason, 'system:cross-seed', at=value['decided_at'], suffix='task_scoped')
        return value

    def _verify_history(self, db, memory_id, version):
        """Recheck all participating evidence inside the promotion transaction."""
        for row in db.execute(f'SELECT payload FROM {self.prefix}_probes WHERE memory_id=?', (memory_id,)):
            probe = json.loads(row[0])
            if probe['memory_version'] != version:
                continue
            args = probe['request']
            execution, *_ = self.evidence.verdict(args['execution_import_id'], args['catalog_id'], args['descriptor'],
                                                  for_validation=False)
            self._target(memory_id, probe['target_seed'], execution, probe['scene_contract_ref'])
        for row in db.execute(f'SELECT payload FROM {self.prefix}_verdicts WHERE memory_id=?', (memory_id,)):
            scene = json.loads(row[0])
            failed = scene.get('failed_check')
            if scene['memory_version'] != version or not failed:
                continue
            execution, ed, bundle, artifacts, _, verdict, _ = self.evidence.verdict(
                failed['execution_import_id'], failed['catalog_id'], failed['descriptor'], for_validation=False)
            grant = self.evidence.validation_grant(failed['validation_grant_id'])
            self.evidence._verify_validation_arm(execution, ed, bundle, artifacts, grant,
                                                 grant['binding']['exposure'], memory_id)
            require(verdict['checks'] == failed['checks'] and
                    self.evidence.catalog(failed['catalog_id'])['files'][failed['descriptor']] == failed['descriptor_sha256'],
                    'failed-check evidence changed before promotion')
            for ref in verdict.get('scene_replay_states', []):
                self.evidence.ref(failed['catalog_id'], ref)
        for row in db.execute(f'SELECT seed,payload FROM {self.prefix}_pairs WHERE memory_id=?', (memory_id,)):
            seed, payload = row
            if self.get('plans', memory_id, seed, db)['memory_version'] != version:
                continue
            pair = json.loads(payload)
            for arm in ('control', 'treatment'):
                import_id = pair[arm + '_import_id']
                old = self.evidence.get(import_id)
                checked = self.evidence.import_trial(import_id=import_id, catalog_id=old['catalog_id'],
                                                      descriptor=old['descriptor'], _persist=False)
                require(checked == old and checked['trial'] == pair[arm], 'cross-seed evidence changed before promotion')

    def status(self, *, memory_id):
        self.service.v2.provenance(memory_id)
        with sqlite3.connect(self.store.path) as db:
            probes = [json.loads(r[0]) for r in db.execute(f'SELECT payload FROM {self.prefix}_probes WHERE memory_id=? ORDER BY seed', (memory_id,))]
            seeds = []
            for probe in probes:
                seed = probe['target_seed']
                verdict = self.get('verdicts', memory_id, seed, db)
                done = db.execute(f'SELECT COUNT(*) FROM {self.prefix}_pairs WHERE memory_id=? AND seed=?', (memory_id, seed)).fetchone()[0]
                seeds.append(dict(target_seed=seed, role=probe['role'], pairs_done=done,
                    pairs_needed=5 if probe['role'] == 'transfer' else 3 if probe['role'] == 'neutral' else 0,
                    verdict=verdict['verdict'] if verdict else None,
                    reason=verdict['reason'] if verdict else 'Waiting for paired checks'))
            return dict(memory_id=memory_id, seeds=seeds, task_trust=self._conditions(db, memory_id))

    def queue(self):
        rows = []
        with sqlite3.connect(self.store.path) as db:
            memories = [json.loads(r[0]) for r in db.execute('SELECT payload FROM memories')]
            for memory in memories:
                mid = memory['memory_id']
                p = self.service.v2._provenance(mid) or {}
                d = self.service.v2.get_decision(mid) or {}
                if (memory['status'] != 'trusted' or p.get('channel') != 'dev_v2' or d.get('trust_scope') != 'scene_scoped'
                        or (self.service.v2.get_plan(mid) or {}).get('profile') != 'single_scene_v1'
                        or memory.get('expires_at') is not None and memory['expires_at'] <= time.time()):
                    continue
                status = self.status(memory_id=mid)
                if 'No neutral seed showed harm' in status['task_trust']['missing']:
                    continue
                uses = [json.loads(r[0]) for r in db.execute('SELECT payload FROM memory_provisional_outcomes WHERE memory_id=?', (mid,))]
                rows.append(dict(**profile_fields(memory['applicability']),
                                 memory_id=mid, version=memory['version'], title=p.get('title'), source_seed=p['source_seed'],
                                 uses=len(uses), failure_gone=sum(r['signature_present'] is False for r in uses),
                                 created_at=memory['created_at'], **{k: status[k] for k in ('seeds', 'task_trust')}))
        return rank_validation_queue(rows)


def task_conditions(m, d, scenes):
    """Shared task-level aggregation for real and hypothetical verdicts."""
    current = [s for s in scenes if s['memory_version'] == m['version']]
    origin = m['status'] == 'trusted' and d.get('trust_scope') in {'scene_scoped', 'task_scoped'}
    if m.get('expires_at') is not None and m['expires_at'] <= time.time(): origin = False
    count = int(origin) + sum(s['verdict'] != 'could_not_finish' for s in current)
    conditions = [('Trusted on the origin scene at the current version', origin),
                  ('Worked on at least one other transfer seed', any(s['verdict'] == 'transfer_passed' for s in current)),
                  ('No neutral seed showed harm', not any(s['verdict'] == 'neutral_harm' for s in scenes)),
                  ('At least three scenes have verdicts, including the origin', count >= 3)]
    return dict(met=[k for k, v in conditions if v], missing=[k for k, v in conditions if not v], scenes=count,
                trust_scope=d.get('trust_scope'), transfer_failed=[s['target_seed'] for s in scenes if s['verdict'] == 'transfer_failed'])
