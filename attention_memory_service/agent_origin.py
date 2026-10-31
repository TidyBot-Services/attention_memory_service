"""Sealed agent help intervals for origin-only source imports."""
import json

from .agent_provisional import verify
from .dev_v2 import require

KEYS = {'agent_help_evidence', 'help', 'origin_source', 'help_source',
        'origin_public_feedback', 'help_public_feedback'}
PURPOSE = 'memory_origin'


def verify_source(refs, read, db, cell, version, manifest, config):
    require(set(refs) == KEYS and cell.get('dev_mode') == config.get('dev_mode') == manifest.get('dev_mode') == 'agent'
            and cell.get('agent_export') == 'v1', 'frozen exported agent origin required')
    require(not any(obj.get(k) for obj in (cell, config, manifest, manifest.get('selection', {}))
                    for k in ('validation_experiment', 'positive_control', 'validation_origin', 'origin_dev_lock')),
            'agent validation and positive controls are unsupported')
    texts = {name: read(ref, raw=True).decode() for name, ref in refs.items()}
    evidence = json.loads(texts['agent_help_evidence'])
    nested = {name: read(ref, raw=True).decode() for name, ref in evidence['refs'].items()}
    origin = dict(cell=cell['key'], origin_version=version['version'] - 1, help_version=version['version'],
                  provider='advisor', digests={name: ref['sha256'] for name, ref in refs.items()})
    guidance = json.loads(texts['help'])['content']
    verify(origin, guidance, dict(refs=texts, help=nested))
    values = {name: json.loads(text) for name, text in nested.items()}
    for label, number in (('origin', origin['origin_version']), ('help', origin['help_version'])):
        row = db.execute('SELECT payload FROM v2_versions WHERE cell=? AND version=?', (cell['key'], number)).fetchall()
        record = values[label + '_record']
        require(len(row) == 1 and json.loads(row[0][0]) == record
                and record['manifest'] == version['manifest'] and record['config_sha256'] == version['config_sha256']
                and read(record['source'], raw=True).decode() == texts[label + '_source'], 'agent source differs from ledger')
        job = values[label + '_export']
        rows = db.execute('SELECT id,state,payload FROM v2_jobs WHERE cell=? AND version=? AND kind=\'execute\'',
                          (cell['key'], number)).fetchall()
        require(len(rows) == 1 and rows[0][0] == job['job_id'] and rows[0][1] == 'settled'
                and json.loads(rows[0][2]) == {k: v for k, v in job.items() if k != 'state'}, 'agent export differs from ledger')
        require(read(job['receipt']['public_feedback'], raw=True).decode() == texts[label + '_public_feedback'],
                'agent feedback differs from ledger')
        bundle = read(job['receipt']['bundle'])
        require(bundle.get('channel') == 'dev_agent' and bundle.get('dev_version') == number
                and bundle.get('policy_sha256') == record['source']['sha256'], 'agent bundle channel or source differs')
        for name in (label + '_execution',):
            sealed = values[name]
            rows = db.execute('SELECT state,payload FROM v2_agent_jobs WHERE cell=? AND id=?', (cell['key'], sealed['id'])).fetchall()
            require(len(rows) == 1 and rows[0][0] == 'settled' and dict(json.loads(rows[0][1]), id=sealed['id'], state=rows[0][0]) == sealed,
                    'agent execution differs from live ledger')
    job = values['help_job']
    if manifest.get('agent_policy_version') == 'agent-policy-v2' and 'execution_job' in job:
        from .agent_help_interval import verify_help
        verify_help(db, cell['key'], version['version'], job, json.loads(texts['help']), values['delivery'])
    else:
        rows = db.execute('SELECT state,payload FROM v2_agent_jobs WHERE cell=? AND id=?', (cell['key'], job['id'])).fetchall()
        require(len(rows) == 1 and rows[0][0] == 'settled' and dict(json.loads(rows[0][1]), id=job['id'], state=rows[0][0]) == job,
                'agent help differs from live ledger')
    require(values['help_record'] == version, 'agent helped version differs')
    return dict(feedback_projection='exact')


def verify_replays(refs, read, config, source_sha256, root, *, count=2):
    from pathlib import Path
    from .resource_receipt import valid
    require(isinstance(refs, list) and len(refs) == count and len({ref['uri'] for ref in refs}) == count,
            'two independent agent policy replays required')
    root = Path(root)
    def owned(ref,raw=False):
        require(Path(ref['uri']).is_relative_to(root), 'policy replay must be validator-owned')
        return read(ref,raw=raw)
    for ref in refs:
        bundle = owned(ref)
        replay_config = owned(bundle['config_lock'])
        from .dev_v2 import digest
        policy = bundle['policy_source']
        require(policy['sha256'] == source_sha256 and digest(owned(policy,raw=True)) == source_sha256,
                'origin replay policy bytes differ')
        require(replay_config == config and bundle['config_sha256'] == bundle['config_lock']['sha256']
                and bundle['policy_sha256'] == source_sha256 and bundle['task_id'] == config['task_id']
                and bundle['seed'] == config['seed'] and bundle.get('runner_boundary') == 'dev_loop'
                and bundle.get('entry_lock') is None and bundle.get('formal_eligible') is False,
                'origin policy replay source/config differs')
        require(bundle['status'] in {'completed', 'failed'} and not bundle['infrastructure_failure']
                and not bundle['outcome_unknown'] and bundle['sandbox'] == dict(process_isolated=True,
                    sdk_rpc_only=True, deadline_enforced=True, action_cancellation_verified=True)
                and bundle['scene_condition']['matched'] is True, 'origin policy replay native conditions failed')
        artifacts = {k: owned(bundle['artifacts'][k]) for k in ('trace', 'native_result', 'safety', 'sandbox_receipt')}
        require(all(all(value.get(k) == bundle[k] for k in ('run_id', 'attempt_id')) for value in artifacts.values()),
                'origin policy replay artifact binding differs')
        trace, native, safety, receipt = (artifacts[k] for k in ('trace', 'native_result', 'safety', 'sandbox_receipt'))
        require(trace['policy_sha256'] == source_sha256 and trace['config_sha256'] == bundle['config_sha256']
                and trace['status'] == bundle['status'] and native == bundle['native_evaluator']
                and native['status'] == bundle['status'] and native.get('evaluated') is True and type(native.get('native_success')) is bool
                and native['native_success'] is bundle['native_success'] and valid(config, receipt), 'origin policy replay evaluation differs')
        require(safety.get('source') == 'native_per_physics_step_monitor' and safety.get('monitor_available') is True
                and type(safety.get('unsafe_attempts')) is int and safety['unsafe_attempts'] == 0 and not safety.get('violations')
                and not any(not step.get('ok') and any(s in step.get('error', '') for s in
                    ('Unexpected physical', 'Measured native', 'Stationary base', 'grasp contact lost', 'object/hand transform'))
                    for step in trace.get('backend_steps', [])), 'origin policy replay safety failed')
