"""Original agent generation, budget and exposure evidence for M-8 arms."""
import json,math
from .dev_v2 import digest,require

ARM_BUDGET=dict(formal_versions=1,moving_executions=5,read_only_probes=60,eval_runs=4,help=0,
    trial_runs=4,sdk_calls_per_execution=320,control_steps_per_execution=9600,
    execution_wall_seconds=1920,active_wall_seconds=1920,model_turns=None,tokens=None)


def shape(importer,catalog_id,manifest,cell,config):
    selection=manifest.get('selection',{});origin=manifest.get('validation_origin',{})
    require(manifest.get('schema')=='attentionbench.dev-experiment.v3' and manifest.get('formal_eligible') is False
        and config.get('budget_tier')=='dev' and config.get('runner_boundary')=='dev_loop'
        and manifest.get('dev_mode')==cell.get('dev_mode')==config.get('dev_mode')=='agent'
        and manifest.get('validation_experiment') is cell.get('validation_experiment') is True
        and manifest.get('agent_validation')==selection.get('agent_validation')==origin
        and set(origin)=={'manifest','config','source','cell','version'}
        and manifest.get('agent_budget')==ARM_BUDGET
        and all(type(manifest['agent_budget'].get(k)) is type(v) for k,v in ARM_BUDGET.items()), 'exact agent validation manifest/budget required')
    # Resolve and digest-check the origin independently through registered evidence.
    ref=origin['manifest']
    original=importer.ref(importer._catalog_for_ref(catalog_id,ref,'executor'),ref)
    version=original.get('agent_policy_version','agent-policy-v1')
    original_cell=next((c for c in original['cells'] if c['key']==origin['cell']),None)
    require(original_cell is not None,'agent validation origin cell missing')
    # Independent policy rule: only v2 hint-only hides Memory. Never trust the
    # arm's claimed policy or visibility marker as the authority for this bypass.
    visibility=('validation_override' if version=='agent-policy-v2'
                and original_cell['policy_id']=='trace_aware_hint_only' else None)
    require(all(row.get('memory_visibility')==visibility
        and ('memory_visibility' in row)==(visibility is not None)
        for row in (manifest,cell,*manifest['cells']))
        and all('memory_visibility' not in row for row in (config,*manifest['configs'].values())),
        'agent validation memory visibility differs from origin')
    require(version in ('agent-policy-v1','agent-policy-v2')
        and all(row.get('agent_policy_version','agent-policy-v1')==version
            for row in (manifest,selection,cell,config,*manifest['cells'],*manifest['configs'].values())),
        'agent validation policy version differs from origin')
    models={k:original.get(k) for k in ('dev_model','dev_reasoning_effort','eval_model')}
    allowed={'gpt-6.1-sol','gpt-6-astra','gpt-6-sol'}
    require(original.get('dev_mode')=='agent' and all(type(v) is str for v in models.values())
        and models['dev_model'] in allowed and models['eval_model'] in allowed
        and models['dev_reasoning_effort'] in ('low','medium','high'), 'unsupported agent validation origin models')
    require(all(manifest.get(k)==selection.get(k)==v for k,v in models.items()), 'agent validation models differ from origin')
    expected=dict(mode='diagnostic',help_budget=0,run_notes=False,memory_family=False,target_depth=1,
        target_valid_executions=1,max_versions=1,reproducible_reset=True,attempt_wall_seconds=1920)
    require(all(selection.get(k)==v and type(selection.get(k)) is type(v) for k,v in expected.items()) and config.get('attempt_wall_seconds')==1920
        and selection.get('tasks')==[config['task_id']] and selection.get('seeds')==[config['seed']]
        and selection.get('policies')==[cell['policy_id']] and selection.get('scene_contract')==config.get('scene_contract')
        and len(manifest['cells'])==10 and len({c['key'] for c in manifest['cells']})==10
        and {(c['validation_pair'],c['validation_arm']) for c in manifest['cells']}=={(i,a) for i in range(5) for a in ('control','treatment')}
        and manifest.get('schedule')==[dict(cell=c['key'],through_version=1) for c in manifest['cells']], 'exact single-scene agent arms required')
    require(all(c['task_id']==config['task_id'] and c['seed']==config['seed'] and c['policy_id']==cell['policy_id'] for c in manifest['cells'])
        and all(c.get('excluded_from_results') is True for c in manifest['cells']) and cell.get('excluded_from_results') is True
        and all(manifest['configs'][f"{c['task_id']}:{c['seed']}"].get('dev_mode')=='agent' for c in manifest['cells']), 'mixed agent validation batch')
    require(not any(obj.get(k) for obj in (cell,manifest,selection) for k in ('run_notes','memory_family','memory_authority','planner_features','positive_control','fork'))
        and not any(selection.get(k,{}).get('enabled') for k in ('advisor_proxy','eval_agent')),
        'agent validation cannot acquire extra assistance')


def usage(importer,catalog_id,d,manifest,record,cell,job):
    config=importer.ref(catalog_id,d['config']);shape(importer,catalog_id,manifest,cell,config)
    require(type(d['version']) is int and d['version']==1,'agent validation requires one formal version')
    limits={k:ARM_BUDGET[k] for k in ('formal_versions','moving_executions','read_only_probes','eval_runs','help','trial_runs')}
    with importer._ledger(catalog_id,d) as db:
        row=db.execute('SELECT payload FROM v2_agent_runs WHERE cell=?',(d['cell'],)).fetchone()
        require(row is not None,'agent validation usage missing')
        run=json.loads(row[0])
        jobs=[dict(json.loads(p),state=s) for s,p in db.execute('SELECT state,payload FROM v2_agent_jobs WHERE cell=?',(d['cell'],))]
    require(run['limits']==limits and all(type(v) is int for v in run['limits'].values()) and run['wall_seconds']==1920,'agent validation ledger budget differs')
    require(all(j['state'] in {'settled','released'} for j in jobs),'unresolved agent validation work')
    require(all(type(v) is int and v>=0 for j in jobs for v in j['resources'].values()),'invalid agent resource count')
    used={k:sum(j['resources'].get(k,0) for j in jobs if j['state']!='released') for k in limits}
    require(all(type(v) is int and 0<=v<=limits[k] for k,v in used.items()) and used['formal_versions']==1 and used['eval_runs']<=1,
        'agent validation budget exceeded')
    require(all(type(j.get(t)) in (int,float) and math.isfinite(j[t]) for j in jobs if j['state']=='settled' for t in ('claimed_at','settled_at'))
        and run['deadline'] is not None and all(j['settled_at']<=run['deadline']+60 for j in jobs if j['state']=='settled'), 'agent validation wall exceeded')
    receipt=importer.ref(catalog_id,record['agent_usage'])
    require(receipt==dict(schema_version='attentionbench.agent-validation-usage.v1',cell=d['cell'],version=1,
        request_id=job['receipt']['agent_request_id'],limits=limits,wall_seconds=1920,used=used,execution_usage=job['receipt']['execution_usage']),
        'agent usage receipt differs from original ledger')


def generation(importer,execution,d,binding,plan,candidate):
    cid=execution['catalog_id'];record=importer.ref(cid,d['version_record']);manifest=importer.ref(cid,d['manifest'])
    require(binding.get('origin_source_sha256')==plan['origin_source_sha256'] and binding.get('dev_model')==plan['dev_model']
        and plan['dev_model']==dict(provider='codex',model=manifest['dev_model']), 'agent generation model/origin differs')
    parent=importer.ref(cid,record['parent_source'],raw=True)
    require(digest(parent)==plan['origin_source_sha256']==manifest['validation_origin']['source']['sha256'], 'agent parent differs from frozen origin')
    provenance=importer.validation_manager().provenance(binding['memory_id'])
    require(provenance.get('channel')=='dev_agent','agent arm requires agent candidate')
    source=importer.generation_source(provenance,binding);failed,*_=importer.execution(source['failed_execution_import_id'])
    require(failed.get('channel')=='dev_agent' and failed['source']['sha256']==plan['origin_source_sha256']
        and manifest['validation_origin']['manifest']==failed['manifest'] and manifest['validation_origin']['config']==failed['config']
        and importer.ref(cid,record['origin_feedback'])==importer.ref(failed['catalog_id'],failed['public_feedback']), 'agent origin public feedback differs')
    packet_ref=importer.ref(cid,execution['bundle']).get('agent_memory_input')
    packet=importer.ref(cid,packet_ref)
    treatment=binding['exposure']=='candidate_validation'
    expected=[]
    if treatment:expected=[dict(memory_id=candidate.memory_id,version=binding['memory_version'],content=candidate.guidance,
        development_id=f"{d['cell']}:v1",source_policy_sha256=plan['origin_source_sha256'],
        unvalidated=True,excluded_from_results=True,trusted_here=False)]
    require(candidate.version==binding['memory_version'] and digest(candidate.guidance.encode())==binding['guidance_sha256']
        and packet==dict(schema_version='attentionbench.memory-input.v1',entries=expected,excluded_from_results=True), 'agent Memory must contain only the granted note')
    exposure=importer.ref(cid,record['dev_memory_exposure'])
    entries=[dict(memory_id=candidate.memory_id,memory_version=candidate.version,guidance_sha256=binding['guidance_sha256'])] if treatment else []
    require(exposure==dict(schema_version='attentionbench.dev-memory-exposure.v1',cell=d['cell'],version=1,model_input=packet_ref,entries=entries),
        'agent exposure receipt differs from delivered input')
    turn=importer.ref(cid,record['agent_bootstrap'])
    require('memory/v1.json' in turn['prompt'] and 'origin/public-feedback.json' in turn['prompt']
        and turn['command'][turn['command'].index('-m')+1]==manifest['dev_model'], 'agent bootstrap lacks origin/Memory delivery')
    with importer._ledger(cid,d) as db:
        deliveries=[(at,json.loads(p)) for at,p in db.execute("SELECT at,payload FROM v2_events WHERE kind='agent_validation_memory'")]
        turns=[json.loads(p) for p, in db.execute("SELECT payload FROM v2_agent_jobs WHERE cell=?",(d['cell'],)) if json.loads(p)['kind']=='model_turn']
    event=dict(cell=d['cell'],version=1,memory_input=packet_ref,dev_memory_exposure=record['dev_memory_exposure'])
    require(turns and len([at for at,p in deliveries if p==event and at<=min(t['claimed_at'] for t in turns)])==1,
        'agent Memory lacks pre-generation delivery event')
    return dict(origin_source_sha256=plan['origin_source_sha256'],dev_model=plan['dev_model'],feedback_projection='exact')


def invalid(manager,evidence,*,memory_id,validation_grant_id,execution_import_id,catalog_id,descriptor):
    from pathlib import Path
    import time
    value=evidence.read(catalog_id,descriptor);grant=evidence.validation_grant(validation_grant_id);binding=grant['binding']
    validator=evidence.catalog(catalog_id,'validator');executor=evidence.catalog(value['executor_catalog_id'],'executor')
    require(not Path(validator['root']).is_relative_to(executor['root']) and not Path(executor['root']).is_relative_to(validator['root'])
        and validator['context']==executor['context']==binding['context'], 'independent invalid agent validator required')
    require(value['execution_import_id']==execution_import_id and value['validation_grant_id']==validation_grant_id
        and value['cell']==binding['cell'] and binding['memory_id']==memory_id and binding['version']==1
        and value['reason'].startswith('agent_arm_invalid:'), 'invalid agent arm binding differs')
    cid=value['executor_catalog_id'];raw=evidence.read(cid,value['executor_descriptor'])
    require(raw=={k:value[k] for k in ('cell','ledger','manifest','config','reason')}, 'invalid agent ledger seal differs')
    manifest=evidence.ref(cid,value['manifest']);config=evidence.ref(cid,value['config'])
    with evidence._ledger(cid,raw) as db:
        cell=evidence._row(db,'v2_cells','id=?',(value['cell'],))
        experiment=evidence._row(db,'v2_experiments','id=?',(cell['experiment_id'],))
        require(not db.execute("SELECT 1 FROM v2_jobs WHERE cell=? AND kind='execute' AND state='settled'",(value['cell'],)).fetchone(),
            'executed arm cannot be closed as missing formal evidence')
        if binding['policy_sha256']!=manifest['validation_origin']['source']['sha256']:
            dispatches=[(at,json.loads(p)) for at,p in db.execute("SELECT at,payload FROM v2_events WHERE kind='memory_validation_dispatch'")]
            original=[at for at,p in dispatches if p.get('cell')==value['cell'] and p.get('version')==1
                and p.get('validation_grant_id')==validation_grant_id and p.get('source',{}).get('sha256')==binding['policy_sha256'] and p.get('config',{}).get('sha256')==value['config']['sha256']]
            formal=[json.loads(p) for state,p in db.execute('SELECT state,payload FROM v2_agent_jobs WHERE cell=?',(value['cell'],))
                if json.loads(p)['resources'].get('formal_versions')==1 and state=='settled']
            require(len(original)==1 and len(formal)==1 and formal[0]['receipt'].get('version_id')=='v0001'
                and grant['issued_at']<=original[0]<=formal[0]['claimed_at'],'invalid submission lacks its original dispatch binding')
    shape(evidence,cid,manifest,cell,config)
    require(experiment['manifest']==value['manifest'] and cell.get('state')=='invalid_generation' and cell.get('reason')==value['reason']
        and cell.get('owner') is None and cell.get('experiment_eligible') is False
        and value['config']['sha256']==binding['config_sha256'] and config['seed']==binding['seed'], 'original invalid agent state required')
    provenance=manager.provenance(memory_id);plan=manager.get_plan(memory_id)
    require(provenance.get('channel')=='dev_agent' and plan and plan['arm_generation']=='dev_from_origin'
        and digest(json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())==binding['plan_sha256']
        and plan['origin_source_sha256']==binding.get('origin_source_sha256')==manifest['validation_origin']['source']['sha256'], 'invalid agent plan differs')
    failed=dict(validation_grant_id=validation_grant_id,execution_import_id=execution_import_id,catalog_id=catalog_id,
        descriptor=descriptor,descriptor_sha256=validator['files'][descriptor],checks=dict(native_evaluation=False),reason=value['reason'])
    old=manager.get_decision(memory_id)
    if old:
        require(old.get('failed_check')==failed,'conflicting closed validation decision');return old
    report=manager.impact_report(memory_id)
    decision=dict(schema_version='attentionbench.memory-validation-decision.v1',memory_id=memory_id,profile='single_scene_v1',
        decision='inconclusive',failed_gates=['invalid_agent_arm'],failed_gate_messages=['No valid formal execution was submitted.'],
        reason='Could not finish the check: agent arm has no valid formal execution',failed_check=failed,
        minimum_pairs=5,report=report,decided_at=time.time(),trust_scope=None,cross_scene_eligible=False)
    with manager.store._transaction() as db:
        memory=manager.store.get_memory(memory_id)
        require(memory.status.value=='candidate' and memory.version==binding['memory_version'],'invalid agent candidate changed')
        manager._record_decision_tx(db,memory,decision)
    return decision
