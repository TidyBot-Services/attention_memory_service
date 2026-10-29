"""Byte-bound agent help intervals attested by the offline importer."""
import json

from .dev_v2 import digest, require

KEYS = frozenset(('delivery', 'request_view', 'advisor', 'help_job',
                  'origin_execution', 'help_execution', 'origin_record', 'help_record',
                  'origin_export', 'help_export'))


def verify(origin, guidance, evidence):
    require(isinstance(evidence, dict) and set(evidence) == {'refs', 'help'}, 'agent evidence required')
    texts, nested = evidence['refs'], evidence['help']
    require(isinstance(texts, dict) and set(texts) == set(origin['digests']), 'agent evidence refs incomplete')
    for name, sha in origin['digests'].items():
        require(isinstance(texts[name], str) and digest(texts[name].encode()) == sha,
                'agent evidence digest drift: ' + name)
    bundle = json.loads(texts['agent_help_evidence'])
    require(bundle.get('schema_version') == 'attentionbench.agent-help-evidence.v1'
            and bundle.get('channel') == 'dev_agent'
            and bundle.get('origin_version') == origin['origin_version']
            and bundle.get('help_version') == origin['help_version']
            and set(bundle.get('refs', {})) == KEYS and isinstance(nested, dict) and set(nested) == KEYS,
            'invalid agent help evidence')
    for name, ref in bundle['refs'].items():
        require(isinstance(nested[name], str) and digest(nested[name].encode()) == ref.get('sha256'),
                'agent help digest drift: ' + name)
    values = {name: json.loads(text) for name, text in nested.items()}
    answer = json.loads(texts['help'])
    vid = f"v{origin['origin_version']:04d}"
    require(origin['provider'] == 'advisor' and answer.get('provider') == 'advisor'
            and answer.get('version_id') == vid and answer.get('status') == 'delivered'
            and answer.get('delivered') is True and not answer.get('rejected') and not answer.get('cache_hit')
            and answer.get('content') == guidance, 'agent help was not a delivered Advisor hint')
    advisor, job = values['advisor'], values['help_job']
    require(advisor.get('feedback') == guidance and not advisor.get('cache_hit')
            and advisor.get('status') == 'completed' and advisor.get('kind') == 'hint',
            'agent Advisor record differs from help')
    require(job.get('id') == 'help:' + vid and job.get('state') == 'settled' and job.get('kind') == 'help'
            and type(job.get('resources', {}).get('help')) is int and job['resources']['help'] == 1, 'agent help job must be settled')
    receipt = job.get('receipt', {})
    require(receipt.get('version_id') == vid and receipt.get('content') == guidance
            and receipt.get('delivered') is True and receipt.get('status') == 'delivered'
            and not receipt.get('cache_hit') and not receipt.get('rejected'), 'agent help job receipt differs')
    delivery = values['delivery']
    if answer.get('system_injected') is True:
        require(delivery == receipt and receipt.get('system_injected') is True
                and answer.get('decision', {}).get('action') == 'request', 'invalid proactive grant')
    else:
        require(delivery.get('op') == 'ask_help' and delivery.get('status') == 'completed'
                and delivery.get('result', {}).get('version_id') == vid
                and delivery['result'].get('content') == guidance and delivery['result'].get('delivered') is True
                and not delivery['result'].get('system_injected'), 'invalid ask_help delivery')
    for label, number in (('origin', origin['origin_version']), ('help', origin['help_version'])):
        record, execution, export = (values[label + suffix] for suffix in ('_record', '_execution', '_export'))
        response = execution.get('receipt', {})
        require(type(record.get('version')) is int and record['version'] == number and record.get('valid') is True
                and record.get('source', {}).get('sha256') == origin['digests'][label + '_source']
                and record.get('agent', {}).get('valid') is True, 'invalid agent formal version')
        require(execution.get('kind') == 'request' and execution.get('state') == 'settled'
                and all(type(execution.get('resources', {}).get(k)) is int and execution['resources'][k] == 1
                        for k in ('formal_versions', 'moving_executions'))
                and response.get('op') == 'submit' and response.get('version_id') == f'v{number:04d}'
                and response.get('execution_id') and response.get('execution_dispatched') is not False
                and response.get('result', {}).get('outcome', {}).get('status') != 'invalid_submission'
                and execution.get('id') == 'request:' + response.get('request_id', '')
                and record['agent'].get('execution_id') == response['execution_id'], 'agent version was not executed')
        require(export.get('version') == number and export.get('cell') == origin['cell']
                and export.get('kind') == 'execute' and export.get('state') == 'settled'
                and export.get('receipt', {}).get('agent_request_id') == execution['id']
                and export['receipt'].get('public_feedback', {}).get('sha256') == origin['digests'][label + '_public_feedback']
                and export['receipt'].get('accounting') == 'agent_ledger_mirror'
                and all(export.get(k) == execution.get(k) for k in ('reserved_at', 'claimed_at', 'settled_at')),
                'agent execution mirror differs')
    before, after = values['origin_execution'], values['help_execution']
    times = (before.get('claimed_at'), job.get('claimed_at'), job.get('settled_at'),
             after.get('claimed_at'), after.get('settled_at'))
    require(all(type(t) in (int, float) for t in times) and all(a <= b for a, b in zip(times, times[1:]))
            and before.get('settled_at', float('inf')) <= after['claimed_at'], 'help outside formal interval')
