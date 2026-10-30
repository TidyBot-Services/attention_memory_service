"""Agent execution evidence only; origin replay uses a separate import purpose."""
import json

from .dev_v2 import digest, require


def verify(importer, catalog_id, d, bundle, config, manifest, record, cell, job):
    require(bundle.get('schema_version') == 'attentionbench.formal-runner-result.v1'
            and bundle.get('formal_eligible') is False
            and type(bundle.get('dev_version')) is int and bundle['dev_version'] == d['version'],
            'unsupported real agent bundle')
    require(cell.get('dev_mode') == 'agent' and cell.get('agent_export') == 'v1'
            and config.get('dev_mode') == 'agent' and manifest.get('dev_mode') == 'agent', 'agent_export v1 and frozen agent mode required')
    require(type(bundle.get('public_base_yaw_visible')) is bool,
            'agent public_base_yaw_visible must be a sealed bool')
    require(not any(k in obj for obj in (d, d['permission_scope'], d.get('agent_execution', {}), bundle)
                    for k in ('actor', 'approval_id')), 'agent evidence cannot carry actor or approval_id')
    validation=d.get('import_purpose')=='memory_validation' and manifest.get('validation_experiment') is True
    require(validation or d.get('import_purpose') in (None, 'memory_origin') and 'validation_input' not in d
            and not any(obj.get(k) for obj in (cell, config, manifest, manifest.get('selection', {}))
                        for k in ('validation_experiment', 'positive_control', 'origin_dev_lock','agent_validation','validation_origin')),
            'agent offline validation and positive controls are unsupported')
    require(d['permission_scope'].get('action') in ('sensor_only', 'agent_dev_tier_motion'),
            'agent execution requires an agent permission action')
    request_id = job['receipt'].get('agent_request_id')
    require(isinstance(request_id, str) and request_id.startswith('request:')
            and d.get('agent_execution') == dict(request_id=request_id)
            and job['receipt'].get('accounting') == 'agent_ledger_mirror',
            'agent execution request must match the execute job')
    with importer._ledger(catalog_id, d) as db:
        require(db.execute("SELECT 1 FROM sqlite_master WHERE name='v2_agent_jobs'").fetchone(),
                'agent ledger missing')
        row = db.execute('SELECT state,payload FROM v2_agent_jobs WHERE cell=? AND id=?',
                         (d['cell'], request_id)).fetchone()
    require(row is not None and row[0] == 'settled', 'agent request must be settled')
    agent = json.loads(row[1])
    response = agent.get('receipt', {})
    require(agent.get('kind') == 'request' and all(type(agent.get('resources', {}).get(k)) is int and agent['resources'][k] == 1
                for k in ('formal_versions', 'moving_executions'))
            and response.get('op') == 'submit' and response.get('request_id') == request_id[8:]
            and response.get('version_id') == f"v{d['version']:04d}"
            and response.get('execution_id') == record.get('agent', {}).get('execution_id')
            and response.get('execution_id') and response.get('execution_dispatched') is not False
            and record.get('agent', {}).get('valid') is True,
            'agent execution is not a valid dispatched formal submission')
    require(all(agent.get(k) == job.get(k) for k in ('reserved_at', 'claimed_at', 'settled_at')),
            'agent mirror timing differs from ledger')
    usage = agent.get('execution_usage', {})
    require(type(usage.get('sdk_calls')) is int and type(usage.get('control_steps')) is int
            and type(usage.get('wall_seconds')) in (int, float)
            and job['receipt'].get('execution_usage') == usage
            and job['actual'] == dict(executions=1, sdk_calls=usage.get('sdk_calls'),
                control_steps=usage.get('control_steps'), execution_seconds=usage.get('wall_seconds')),
            'agent mirror actual costs differ from ledger')
    if validation:
        from .agent_validation import usage as verify_usage
        verify_usage(importer,catalog_id,d,manifest,record,cell,job)


def result_evidence(importer, grant, execution_id, execution, d, bundle):
    """Attest generation exposure, never model authorship or causal benefit."""
    catalog = execution['catalog_id']
    packet_ref = bundle.get('agent_memory_input')
    require(isinstance(packet_ref, dict), 'agent generation Memory packet missing')
    packet = importer.ref(catalog, packet_ref)
    require(packet.get('excluded_from_results') is not True
            and packet.get('seeded_for_tests') is not True, 'agent packet excluded from results')
    with importer._ledger(catalog, d) as db:
        events = [(at, json.loads(raw)) for at, raw in db.execute(
            "SELECT at,payload FROM v2_events WHERE kind='planner_memory_retrieved'")]
        prior = (importer._row(db, 'v2_versions', 'cell=? AND version=?', (d['cell'], d['version'] - 1))
                 if d['version'] > 1 else None)
    matching = [(at, e) for at, e in events if e.get('cell') == d['cell'] and e.get('version') == d['version']]
    require(len(matching) == 1 and matching[0][1].get('packet') == packet_ref
            and grant['exposed_at'] <= matching[0][0] <= importer._execution_claim_time(execution, d),
            'agent Memory packet lacks original pre-execution generation event')
    manifest = importer.ref(catalog, d['manifest'])
    previous_sha = (prior['source']['sha256'] if prior else
                    digest(manifest['initial_sources'][d['context']['task_id']]['source'].encode()))
    if prior:
        require(bundle.get('agent_memory_previous_source') == prior['source'], 'agent previous source differs from ledger')
        importer.ref(catalog, prior['source'], raw=True)
    require(previous_sha == grant['source_policy_sha256'], 'agent grant source code SHA mismatch')
    entries = packet.get('entries')
    require(isinstance(entries, list), 'agent Memory input must be exact entry list')
    matching = [e for e in entries if e.get('grant_id') == grant['grant_id']]
    require(len(matching) == 1, 'exact grant absent from agent generation input')
    entry = matching[0]
    from .identity import store_id
    require(type(entry.get('version')) is int and all(entry.get(k) == v for k, v in {
        'memory_id': grant['memory_id'], 'version': grant['memory_version'], 'content': grant['guidance'],
        'context': grant['context'], 'development_id': grant['development_id'],
        'source_policy_sha256': previous_sha, 'authority': store_id(importer.store.path)}.items()),
        'agent generation input differs from issued grant')
    require(entry.get('excluded_from_results') is not True and entry.get('seeded_for_tests') is not True
            and entry.get('trusted_here') is not False and entry.get('unvalidated') is not True,
            'agent grant excluded from results')
    return dict(schema_version='attentionbench.dev-v2-memory-result.v1', channel='dev_agent',
                provenance=dict(channel='dev_agent', execution_import_id=execution_id),
                execution_import_id=execution_id, run_id=execution['run_id'], cell=d['cell'], version=d['version'],
                source_policy_sha256=previous_sha, result_policy_sha256=execution['source']['sha256'],
                bundle=execution['bundle'], ledger=execution['ledger'], memory_input=packet_ref,
                public_feedback=execution['public_feedback'], permission_scope=d['permission_scope'],
                claim='development_exposure_only_not_causal_impact')
