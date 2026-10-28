"""Isolated operator fixtures and model rewrites; shared CrossSeed evidence gates.

Only real family-trusted candidates may be references. Fixture candidates are
available solely to their own offline treatment grants, never Memory records.
"""
import json
import math
import re
import sqlite3
import time
from pathlib import Path

from .applicability import require_profile_match, config_profile, profile_fields
from .cross_seed import CrossSeed
from .dev_v2 import DevV2Evidence, SCOPE, digest, encoded, require, evidence_operation
from .families import family_of
from .positive_controls import LABEL, ControlView, origin_evidence, validate_spec as validate_control
from .memory_v2 import MemoryV2Manager
from .seed_guard import classify_seed

STATEMENT = 'Operator preset; same-scene and cross-seed validation were skipped'
STATEMENT_ZH = '操作员预置，跳过了同场景和跨 seed 验证'
TARGET_DIFFERENCES = {'task_id', 'task_prompt', 'task_variant_id', 'seed', 'scene_contract_ref'}
# An independent service has no dependency on simulator/task implementations.
REGISTERED_TASKS = {'pick_place_counter_to_sink', 'pick_place_counter_to_cabinet', 'open_cabinet'}
SHARED_WORDS = set('pick the object from counter and place it in to a an of on with for at by is be as this that its grasp lift move target surface support gripper arm before after then during without ensure use select exact name not substring approach vertical heading orientation stable small increments'.split())
REWRITE_PROMPT = ('Rewrite into task-independent wording for the pick-and-place family. '
                  'Keep the physical idea; drop task-specific names/numbers that only hold for the source scene. '
                  'Return JSON {"title", "guidance"}. Guidance must be nonempty and at most 600 characters.')


def validate_spec(spec):
    require(isinstance(spec, dict) and set(spec) == {'fixture_id','title','guidance','origin','context','label','user_quotes'}, 'complete fixture spec required')
    validate_control(dict(control_id=spec['fixture_id'], **{k:spec[k] for k in ('title','guidance','origin','context','label')}, cross_seed_targets=[]))
    require(isinstance(spec['user_quotes'], list) and bool(spec['user_quotes']) and all(isinstance(q,str) and q.strip() for q in spec['user_quotes']), 'nonempty user quotes required')
    return digest(encoded(spec).encode())


def validate_rewrite(source, rewrite):
    require(isinstance(rewrite,dict) and set(rewrite)=={'title','guidance','receipt'}, 'complete model rewrite required')
    title, text = rewrite['title'], rewrite['guidance']
    require(isinstance(title,str) and 0 < len(title.strip()) <= 100 and isinstance(text,str) and 0 < len(text.strip()) <= 600, 'short title and nonempty guidance <= 600 characters required')
    combined = (title+' '+text).casefold()
    task = source['context']['task_id']
    words = set(re.findall(r'[a-z]+', source['context']['task_prompt'].casefold())) - SHARED_WORDS
    words |= set(task.split('_')) - SHARED_WORDS
    require(task.casefold() not in combined and not words.intersection(re.findall(r'[a-z]+',combined)), 'rewrite contains source task or task-specific target words')
    require(not re.search(r'\bseed\s*[:=#-]?\s*\d+|\b1(?:0\d|1\d|2[0-5])\b',combined), 'rewrite contains seed numbers')
    require(not re.search(r'(?:world|absolute|coordinate|\b[xyz]\s*[=:])[^.;\n]*[-+]?\d|[\[(]\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d',combined), 'rewrite contains absolute world coordinates')
    r=rewrite['receipt']
    require(isinstance(r,dict) and {'provider','model','prompt','prompt_sha256','response_sha256','raw_response','started_at','finished_at'}<=set(r), 'complete model receipt required')
    require(r['provider']=='codex' and isinstance(r['model'],str) and r['model'].strip(), 'Codex model receipt required')
    require(r['prompt']==dict(instruction=REWRITE_PROMPT,title=source['title'],guidance=source['guidance'],task_id=task), 'rewrite prompt differs from fixed source prompt')
    require(r['prompt_sha256']==digest(encoded(r['prompt']).encode()) and isinstance(r['raw_response'],str) and r['response_sha256']==digest(r['raw_response'].encode()), 'rewrite receipt digest mismatch')
    require(all(type(r[k]) in (int,float) and math.isfinite(r[k]) for k in ('started_at','finished_at')) and 0<=r['started_at']<=r['finished_at'], 'invalid model timestamps')
    raw=r['raw_response'].strip()
    if raw.startswith('```'):raw=re.sub(r'^```(?:json)?\s*|\s*```$', '', raw)
    require(json.loads(raw)==dict(title=title,guidance=text), 'rewrite differs from raw model response')


def family_conditions(source_active, scenes):
    tasks={s['target_task_id'] for s in scenes}
    eligible=[]
    for task in sorted(tasks):
        rows=[s for s in scenes if s['target_task_id']==task]
        passed={s['target_seed'] for s in rows if s['verdict'] in {'transfer_passed','neutral_no_harm'}}
        if len(passed)>=2 and not any(s['verdict'] in {'transfer_failed','neutral_harm'} for s in rows):eligible.append(task)
    checks=[('Source is still trusted for its task',source_active),
            ('At least one other family task passed on two scenes without harm',bool(eligible))]
    worked=any(s['target_task_id'] in eligible and s['verdict']=='transfer_passed' for s in scenes)
    return dict(met=[k for k,v in checks if v],missing=[k for k,v in checks if not v],passed_tasks=eligible,
                worked_on=worked,neutral_only=bool(eligible) and not worked,
                evidence_strength='worked on' if worked else 'no harm only' if eligible else 'incomplete')


class FamilyEvidence(DevV2Evidence):
    def __init__(self, family):
        self.family=family
        super().__init__(family.store)
    def validation_manager(self):return self.family
    def validation_candidate(self, mid):return self.family.get_memory(mid)
    def validation_plan(self, mid, target_seed=None):return self.family.cross_seed.get('plans',mid,target_seed)
    def validation_grant(self, grant_id):
        value=self.family._read('grants',grant_id)
        require(value is not None,'unknown family validation grant')
        self.family.active(value['binding']['memory_id'])
        return value
    def _validation_generation(self, execution, ed, binding, plan, candidate):
        result=super()._validation_generation(execution,ed,binding,plan,candidate)
        if binding['exposure']=='candidate_validation':
            record=self.ref(execution['catalog_id'],ed['version_record'])
            request=self.ref(execution['catalog_id'],record['model_input'])
            require(request['messages'][0]['content'].endswith(candidate.treatment_label),'family treatment label missing from model request')
        return result

    def generation_source(self, provenance, binding):
        probe=self.family.cross_seed.get('probes',binding['memory_id'],binding['target_seed'])
        require(probe is not None,'family probe missing')
        e=probe['execution']
        return dict(failed_execution_import_id=probe['request']['execution_import_id'],catalog_id=e['catalog_id'],descriptor=e['descriptor'])


class FamilyCrossSeed(CrossSeed):
    prefix='family_cross_task'
    grants_table='family_grants'
    def candidate(self, mid):return self.service.get_memory(mid)
    def _eligible(self, mid, db, *, version=None):
        c=self.service.active(mid)
        require(version in (None,1),'family candidate version is immutable')
        return self.candidate(mid).artifact(),self.service.provenance(mid)
    def target_provenance(self, mid, seed, provenance):
        target=self.service.target(mid,seed)
        return dict(provenance,source_policy_sha256=target['baseline_source']['sha256'])
    def _target(self, mid, seed, execution, contract_ref):
        c=self.service.active(mid);target=self.service.target(mid,seed);spec=self.service.check(mid)
        require(type(seed) is int and classify_seed(seed)=='dev','Dev target seed required')
        ctx=execution['context'];origin=c['context']
        require(ctx['task_id']==spec['target_task_id'] and ctx['task_id']!=c['source_task_id'] and family_of(ctx['task_id'])==c['family'], 'target must be another registered task of the family')
        require_profile_match(origin,ctx)
        require(all(ctx.get(k)==origin.get(k) for k in (*SCOPE,'camera_names','task_prompt','sdk_interface_sha256','seed','scene_contract_ref') if k not in TARGET_DIFFERENCES), 'family target scope mismatch')
        require(isinstance(origin.get('sdk_interface_sha256'),str),'family source SDK interface missing')
        config=self.evidence.ref(execution['catalog_id'],execution['config'])
        require_profile_match(ctx,config_profile(config))
        require(config['task_id']==ctx['task_id'] and config['seed']==seed and config.get('scene_contract')==contract_ref and contract_ref['uri']==target['contract'], 'family target config/contract mismatch')
        self.require_contract(contract_ref,seed,ctx)
        require(execution['source']['sha256']==target['baseline_source']['sha256'], 'family probe must use frozen target baseline')
        return config
    def grant_binding(self,binding,plan):
        return dict(binding,candidate_id=binding['memory_id'],target_task_id=plan['context']['task_id'])
    def _conditions(self,db,mid):return self.service.conditions(mid,db)
    def _close(self,db,mid,seed,verdict,reason,extra):
        c=self.service.active(mid)
        value=dict(**profile_fields(c['context']),memory_id=mid,memory_version=1,target_task_id=self.service.check(mid)['target_task_id'],target_seed=seed,verdict=verdict,reason=reason,decided_at=time.time(),**extra)
        db.execute(f'INSERT INTO {self.prefix}_verdicts VALUES (?,?,?)',(mid,seed,encoded(value)))
        self.service.event(db,'scene_verdict',mid,reason)
        conditions=self._conditions(db,mid)
        if not conditions['missing']:self._verify_history(db,mid,1)
        self.service.decide(db,mid,conditions)
        return value


class Family:
    _single_scene_failures=MemoryV2Manager._single_scene_failures
    def __init__(self,service):
        self.service,self.store=service,service.store
        with sqlite3.connect(self.store.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS test_fixture_records (id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
            for table in ('candidates','checks','decisions','events','uses','references'):
                db.execute(f'CREATE TABLE IF NOT EXISTS family_{table} (id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS family_grants (id TEXT PRIMARY KEY,memory_id TEXT,cell TEXT,version INTEGER,payload TEXT NOT NULL,UNIQUE(memory_id,cell,version))')
        self.v2=self;self.dev_v2=FamilyEvidence(self);self.cross_seed=FamilyCrossSeed(self)
    def _read(self,table,key,db=None):
        if db is None:
            with sqlite3.connect(self.store.path) as db:return self._read(table,key,db)
        name='test_fixture_records' if table=='fixtures' else 'family_'+table
        row=db.execute(f'SELECT payload FROM {name} WHERE id=?',(key,)).fetchone()
        return json.loads(row[0]) if row else None
    def event(self,db,kind,mid,reason):
        import uuid
        value=dict(kind=kind,candidate_id=mid,reason=reason,at=time.time())
        db.execute('INSERT INTO family_events VALUES (?,?)',(uuid.uuid4().hex,encoded(value)))
    @evidence_operation
    def fixture_create(self,*,spec,spec_sha256,execution_import_id,catalog_id,descriptor):
        require(validate_spec(spec)==spec_sha256,'fixture spec digest mismatch')
        old=self._read('fixtures',spec['fixture_id'])
        execution,cell,signature=origin_evidence(self.service.dev_v2,spec,execution_import_id,catalog_id,descriptor,old)
        value=dict(record_kind='test_fixture',label=LABEL,excluded_from_results=True,operator_preset=True,
                   statement=STATEMENT,statement_zh=STATEMENT_ZH,user_quotes=spec['user_quotes'],title=spec['title'],guidance=spec['guidance'],context=spec['context'],
                   spec=spec,spec_sha256=spec_sha256,preset_levels=dict(scene='trusted',task='trusted'),
                   created_at=old['created_at'] if old else time.time(),revoked_at=old.get('revoked_at') if old else None,
                   execution_import_id=execution_import_id,catalog_id=catalog_id,descriptor=descriptor,failure_signature=signature,
                   source_policy_sha256=execution['source']['sha256'],policy_id=cell['policy_id'],version=1)
        if old and 'revoke_reason' in old:value['revoke_reason']=old['revoke_reason']
        with self.store._transaction() as db:
            current=self._read('fixtures',spec['fixture_id'],db)
            require(current is None or current==value,'fixture is immutable')
            db.execute('INSERT OR IGNORE INTO test_fixture_records VALUES (?,?)',(spec['fixture_id'],encoded(value)))
        return value
    def fixture_revoke(self,*,fixture_id,reason):
        require(isinstance(reason,str) and reason.strip(),'revocation reason required')
        with self.store._transaction() as db:
            r=self._read('fixtures',fixture_id,db);require(r is not None,'unknown fixture')
            if r['revoked_at'] is None:
                r.update(revoked_at=time.time(),revoke_reason=reason)
                db.execute('UPDATE test_fixture_records SET payload=? WHERE id=?',(encoded(r),fixture_id))
                self.event(db,'revoked',fixture_id,reason)
        return r
    def source(self,source,*,eligible=True):
        require(isinstance(source,dict) and set(source)=={'kind','id'} and source['kind'] in {'memory','fixture'},'source kind/id required')
        if source['kind']=='fixture':
            r=self._read('fixtures',source['id']);require(r is not None,'unknown fixture')
            active=r['revoked_at'] is None
            value=dict(r,id=source['id'],kind='fixture',active=active)
        else:
            memory=self.store.get_memory(source['id']);require(memory is not None,'unknown source memory')
            m=memory.artifact();p=self.service.v2._provenance(source['id']) or {};d=self.service.v2.get_decision(source['id']) or {}
            require(p.get('channel')=='dev_v2','real dev_v2 source required')
            require(m['guidance']==p['artifact']['guidance'],'source guidance changed')
            active=(m['status']=='trusted' and p.get('channel')=='dev_v2' and d.get('trust_scope')=='task_scoped' and (m.get('expires_at') is None or m['expires_at']>time.time()))
            value=dict(id=source['id'],kind='memory',version=m['version'],guidance=m['guidance'],title=p.get('title') or 'Untitled note',context=self.service.dev_v2.get(p['source_import_id'])['context'],active=active,failure_signature=p['failure_signature'],provenance=p)
        if eligible:require(active,'source must still be task-trusted and not expired or revoked')
        return value
    @evidence_operation
    def generalize(self,*,source,rewrite):
        r=self.source(source)
        if source['kind']=='memory':self.service.v2.verify_source_evidence(source['id'])
        family=family_of(r['context']['task_id']);require(family is not None,'source has no family')
        key='family-'+digest(encoded([source,r['version']]).encode())[:32]
        old=self._read('candidates',key)
        if old:
            require(old['guidance']==rewrite.get('guidance') and old['title']==rewrite.get('title') and old['model_receipt']==rewrite.get('receipt'),'one immutable rewrite per source/version')
            self.active(key);return old
        validate_rewrite(r,rewrite)
        sha=digest(r['guidance'].encode())
        value=dict(candidate_id=key,version=1,family=family,source_task_id=r['context']['task_id'],
                   from_source=dict(**source,version=r['version'],guidance_sha256=sha,title=r['title']),
                   original_guidance=r['guidance'],guidance=rewrite['guidance'],title=rewrite['title'],context=r['context'],
                   rewrite_kind='model_rewrite',rewrite_label='Model rewrite',model_receipt=rewrite['receipt'],
                   status='candidate',created_at=time.time(),disabled_at=None,
                   treatment_label=f"Family candidate (model rewrite of a {r['context']['task_id']} note)"+(' · '+LABEL if source['kind']=='fixture' else ''))
        if source['kind']=='fixture':value.update(label=LABEL,excluded_from_results=True)
        with self.store._transaction() as db:
            require(digest(self.source(source)['guidance'].encode())==sha,'original guidance changed')
            current=self._read('candidates',key,db)
            if current:
                require(all(current[k]==value[k] for k in ('from_source','guidance','title','model_receipt')),'one immutable rewrite per source/version')
                return current
            db.execute('INSERT INTO family_candidates VALUES (?,?)',(key,encoded(value)))
            self.event(db,'generalized',key,'Model rewrite created; original unchanged')
        return value
    def get(self,mid,*,reconcile=True):
        c=self._read('candidates',mid);require(c is not None,'unknown family candidate')
        r=self.source({k:c['from_source'][k] for k in ('kind','id')},eligible=False)
        revoked=r['kind']=='fixture' and not r['active']
        if revoked:c['status']='revoked'
        elif c['disabled_at'] is not None:c['status']='disabled'
        elif not r['active'] or r['version']!=c['from_source']['version']:
            if c['status']=='family_trusted':
                c['status']='candidate'
                if reconcile:
                    with self.store._transaction() as db:
                        db.execute('UPDATE family_candidates SET payload=? WHERE id=?',(encoded(c),mid))
                        decision=self._read('decisions',mid,db)
                        if decision:
                            decision.update(decision='candidate',reason='Source no longer trusted for its task/version')
                            db.execute('UPDATE family_decisions SET payload=? WHERE id=?',(encoded(decision),mid))
                        self.event(db,'demoted',mid,'Source no longer trusted for its task/version')
        return c
    def active(self,mid):
        c=self.get(mid,reconcile=False);require(c['status'] not in {'revoked','disabled'},'family candidate revoked or disabled')
        r=self.source({k:c['from_source'][k] for k in ('kind','id')})
        require(r['version']==c['from_source']['version'] and digest(r['guidance'].encode())==c['from_source']['guidance_sha256'],'source version/guidance changed')
        return c
    def verify_source_evidence(self,mid):
        c=self.active(mid);source={k:c['from_source'][k] for k in ('kind','id')}
        if source['kind']=='fixture':
            r=self.source(source)
            origin_evidence(self.service.dev_v2,r['spec'],r['execution_import_id'],r['catalog_id'],r['descriptor'],r)
        else:self.service.v2.verify_source_evidence(source['id'])
    def get_memory(self,mid):
        c=self.active(mid)
        return ControlView(memory_id=mid,version=1,record_kind='family_candidate',status=c['status'],guidance=c['guidance'],applicability=c['context'],expires_at=None,treatment_label=c['treatment_label'])
    def provenance(self,mid):
        c=self.get(mid);r=self.source({k:c['from_source'][k] for k in ('kind','id')},eligible=False)
        p=r.get('provenance',{})
        return dict(channel='family',failure_signature=r['failure_signature'],artifact=dict(applicability=c['context']),
                    source_seed=c['context'].get('seed'),source_import_id=mid,source_policy_sha256=r.get('source_policy_sha256',p.get('source_policy_sha256')))
    def get_decision(self,mid):return self._read('decisions',mid)
    def get_plan(self,mid):
        with sqlite3.connect(self.store.path) as db:
            row=db.execute('SELECT payload FROM family_cross_task_probes WHERE memory_id=? ORDER BY seed LIMIT 1',(mid,)).fetchone()
        require(row is not None,'target probe required')
        probe=json.loads(row[0])
        _,_,_,_,cell=self.dev_v2.execution(probe['request']['execution_import_id'])
        return dict(policy_id=cell['policy_id'],assistance_credits=cell['budget']['help_credits'])
    def check(self,mid):
        r=self._read('checks',mid);require(r is not None,'frozen family check required');return r
    def target(self,mid,seed):
        r=next((t for t in self.check(mid)['targets'] if t['seed']==seed),None)
        require(r is not None,'target outside frozen family spec');return r
    def queue_check(self,*,spec):
        require(isinstance(spec,dict) and set(spec)=={'family_check_id','candidate_id','target_task_id','targets','label'},'complete family spec required')
        c=self.active(spec['candidate_id']);task=spec['target_task_id']
        require(isinstance(spec['family_check_id'],str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}',spec['family_check_id']),'invalid family check ID')
        require(task in REGISTERED_TASKS and task!=c['source_task_id'] and family_of(task)==c['family'],'registered different task of the same family required')
        require(spec['label']==(LABEL if c['from_source']['kind']=='fixture' else 'Family check'),'family check label mismatch')
        targets=spec['targets']
        require(isinstance(targets,list) and len(targets)>=2,'at least two target scenes required')
        for t in targets:
            require(isinstance(t,dict) and set(t)=={'seed','contract','baseline_source'} and type(t['seed']) is int and classify_seed(t['seed'])=='dev' and isinstance(t['contract'],str) and Path(t['contract']).is_absolute(),'complete Dev target required')
            b=t['baseline_source'];require(isinstance(b,dict) and set(b)=={'path','sha256'} and Path(b['path']).is_absolute() and re.fullmatch('[a-f0-9]{64}',b['sha256']),'frozen baseline source required')
        require(len({t['seed'] for t in targets})==len(targets),'unique target seeds required')
        with self.store._transaction() as db:
            old=self._read('checks',c['candidate_id'],db);require(old is None or old==spec,'family check is immutable')
            db.execute('INSERT OR IGNORE INTO family_checks VALUES (?,?)',(c['candidate_id'],encoded(spec)))
        return spec
    def conditions(self,mid,db=None):
        if db is None:
            with sqlite3.connect(self.store.path) as db:return self.conditions(mid,db)
        c=self._read('candidates',mid,db);r=self.source({k:c['from_source'][k] for k in ('kind','id')},eligible=False)
        scenes=[json.loads(r[0]) for r in db.execute('SELECT payload FROM family_cross_task_verdicts WHERE memory_id=?',(mid,))]
        return family_conditions(r['active'] and r['version']==c['from_source']['version'] and c['disabled_at'] is None,scenes)
    def decide(self,db,mid,conditions):
        c=self._read('candidates',mid,db);passed=not conditions['missing']
        value=dict(candidate_id=mid,conditions=conditions,decided_at=time.time())
        if c['from_source']['kind']=='fixture':value.update(decision='would be trusted for the family' if passed else 'would not',label=LABEL,excluded_from_results=True)
        else:
            # A later-use demotion/kill is sticky; historical scenes cannot undo it.
            c['status']='family_trusted' if passed and not c.get('demoted_at') and c['disabled_at'] is None else 'candidate'
            value['decision']=c['status'];db.execute('UPDATE family_candidates SET payload=? WHERE id=?',(encoded(c),mid))
        db.execute('INSERT OR REPLACE INTO family_decisions VALUES (?,?)',(mid,encoded(value)))
        self.event(db,'family_verdict',mid,value['decision'])
        return value
    def disable(self,*,candidate_id,reason):
        require(isinstance(reason,str) and reason.strip(),'reason required');c=self.get(candidate_id)
        with self.store._transaction() as db:
            c.update(disabled_at=time.time(),disable_reason=reason,status='candidate')
            db.execute('UPDATE family_candidates SET payload=? WHERE id=?',(encoded(c),candidate_id));self.event(db,'disabled',candidate_id,reason)
        return self.get(candidate_id)
    def record_use(self,*,candidate_id,use_id,signature_present,request_sha256=None):
        c=self.active(candidate_id)
        require(c['from_source']['kind']=='memory' and c['status']=='family_trusted' and type(signature_present) is bool,'real family-trusted use required')
        row=dict(candidate_id=candidate_id,use_id=use_id,signature_present=signature_present,request_sha256=request_sha256)
        with self.store._transaction() as db:
            old=self._read('uses',use_id,db);require(old is None or old==row,'family use is immutable')
            db.execute('INSERT OR IGNORE INTO family_uses VALUES (?,?)',(use_id,encoded(row)))
            uses=[json.loads(r[0]) for r in db.execute('SELECT payload FROM family_uses ORDER BY rowid DESC') if json.loads(r[0])['candidate_id']==candidate_id][:5]
            if len(uses)==5 and sum(u['signature_present'] for u in uses)>=3:
                c.update(status='candidate',demoted_at=time.time());db.execute('UPDATE family_candidates SET payload=? WHERE id=?',(encoded(c),candidate_id));self.event(db,'demoted',candidate_id,'Target failure in at least three of last five uses')
        return self.get(candidate_id)
    def status(self,mid):
        c=self.get(mid);check=self._read('checks',mid)
        cross=self.cross_seed.status(memory_id=mid) if check else dict(seeds=[])
        tested={s['target_seed'] for s in cross['seeds']}
        if check:
            cross['seeds'] += [dict(target_seed=t['seed'],role='pending',pairs_done=0,pairs_needed=None,
                verdict=None,reason='Waiting for clean baseline probe') for t in check['targets'] if t['seed'] not in tested]
            cross['seeds'].sort(key=lambda s:s['target_seed'])
        decision=self._read('decisions',mid)
        if c['status']=='revoked' and decision:decision=dict(decision,status='revoked',decision='Revoked')
        elif c['status']=='disabled' and decision:decision=dict(decision,status='disabled',decision='Disabled')
        elif decision and c['from_source']['kind']=='memory':decision=dict(decision,decision=c['status'])
        return dict(c,check=check,scenes=cross['seeds'],conditions=self.conditions(mid),decision=decision)
    def public(self):
        with sqlite3.connect(self.store.path) as db:
            fixtures=[dict(json.loads(r[1]),fixture_id=r[0]) for r in db.execute('SELECT id,payload FROM test_fixture_records')]
            ids=[r[0] for r in db.execute('SELECT id FROM family_candidates')]
            events=[json.loads(r[0]) for r in db.execute('SELECT payload FROM family_events ORDER BY rowid')]
        for fixture in fixtures:fixture['status']='revoked' if fixture['revoked_at'] else 'operator_preset'
        return dict(fixtures=fixtures,candidates=[self.status(mid) for mid in ids],events=events)
    def references(self,context):
        from .applicability import same_family
        with sqlite3.connect(self.store.path) as db:
            ids=[r[0] for r in db.execute('SELECT id FROM family_candidates')]
        rows=[]
        for mid in ids:
            c=self.get(mid)
            if c['from_source']['kind']!='memory' or c['status']!='family_trusted' or not same_family(c['context'],context):continue
            rows.append(dict(memory_id=mid,version=1,guidance=c['guidance'],title=c['title'],status='trusted',
                record_kind='family_candidate',trust_scope='family_scoped',match_level='same_family',trusted_here=False,
                applicability=c['context'],source_task_id=c['source_task_id'],source_seed=c['context'].get('seed'),
                created_at=c['created_at'],excluded_from_results=True,expires_at=None))
        return rows
    def authorize_reference(self,**args):
        import uuid
        mid=args['memory_id'];c=self.active(mid)
        self.verify_source_evidence(mid)
        require(c['from_source']['kind']=='memory' and c['status']=='family_trusted','real family-trusted candidate required')
        require(type(args.get('memory_version')) is int and args['memory_version']==1 and isinstance(args['development_id'],str) and args['development_id'].strip() and re.fullmatch('[a-f0-9]{64}',args['source_policy_sha256']),'exact family reference binding required')
        require(any(r['memory_id']==mid for r in self.references(args['context'])),'family reference scope mismatch')
        key='familyref:'+digest(encoded([mid,args['development_id']]).encode())
        value=dict(grant_id=key,memory_id=mid,memory_version=1,development_id=args['development_id'],context=args['context'],
                   source_policy_sha256=args['source_policy_sha256'],guidance=c['guidance'],source_trace_id=c['from_source']['id'],evidence_refs=[],
                   source_task_id=c['source_task_id'],match_level='same_family',layer='family',trusted_here=False,
                   excluded_from_results=True,trust_scope='family_scoped',unvalidated=False)
        with self.store._transaction() as db:
            old=self._read('references',key,db);require(old is None or old==value,'family reference binding changed')
            db.execute('INSERT OR IGNORE INTO family_references VALUES (?,?)',(key,encoded(value)))
        return value
    def reference(self,key):
        r=self._read('references',key);require(r is not None,'unknown family reference')
        self.authorize_reference(**{k:r[k] for k in ('memory_id','memory_version','development_id','context','source_policy_sha256')})
        return r
    def outcome(self,*,grant_id,development_id,public_feedback_text,public_feedback_sha256):
        from .provisional_notes import feedback
        from .dev_v2 import signature_present
        observed=feedback(public_feedback_text,public_feedback_sha256)
        r=self._read('references',grant_id);require(r is not None and r['development_id']==development_id,'family exposure binding mismatch')
        request=digest(encoded([grant_id,development_id,public_feedback_sha256]).encode())
        old=self._read('uses',grant_id)
        if old:
            require(old.get('request_sha256')==request,'family outcome is immutable');return old
        p=self.provenance(r['memory_id'])
        result=self.record_use(candidate_id=r['memory_id'],use_id=grant_id,signature_present=signature_present(observed,p['failure_signature']),request_sha256=request)
        return self._read('uses',grant_id)

    def read(self,key,action):
        if action.startswith('probe-'):
            self.active(key)
            return self.cross_seed.get('probes',key,int(action[6:]))
        if action=='source':return self.source(dict(zip(('kind','id'),key.split(':',1))))
        if action=='record':return self.get(key)
        if action=='input':return self.get_memory(key).artifact()
        if action=='status':return self.status(key)
        if action=='check':return self.check(key)
        if action=='provenance':return self.provenance(key)
        if action=='fixture':return self._read('fixtures',key)
        require(False,'unknown family read')
    def write(self,action,payload):
        routes={'fixture-create':self.fixture_create,'fixture-revoke':self.fixture_revoke,'generalize':self.generalize,
                'queue-check':self.queue_check,'disable':self.disable,
                'probes':self.cross_seed.probe,'plans':self.cross_seed.plan,'grants':self.cross_seed.authorize,
                'pairs':self.cross_seed.pair,'failed-checks':self.cross_seed.failed_check,'trials':self.dev_v2.import_trial}
        require(action in routes,'unknown family write')
        if 'memory_id' in payload:self.active(payload['memory_id'])
        return routes[action](**payload)
