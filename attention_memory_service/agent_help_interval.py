"""Service-side attestation of agent-policy-v2 execution help projections.

Mirrors attention_harness.dev_agent_help_interval without making the service
depend on the harness. Cross-repo goldens pin projection and interval semantics.
"""
import json
import re

from .dev_v2 import require


def delivered(answer):
    return (answer.get('provider') == 'advisor' and answer.get('delivered') is True
            and answer.get('status') == 'delivered' and not answer.get('cache_hit')
            and not answer.get('rejected'))


def interval_help(jobs, version):
    """Reservation order breaks equal-clock ties, as in the execution ledger."""
    if version < 2:
        return []
    boundaries = []
    for number in (version - 1, version):
        matches = [(i, j) for i, j in enumerate(jobs) if j['kind'] == 'request'
                   and j['state'] == 'settled' and j.get('receipt', {}).get('op') == 'submit'
                   and j['receipt'].get('version_id') == f'v{number:04d}'
                   and j['receipt'].get('execution_id')
                   and j['receipt'].get('execution_dispatched') is not False
                   and j['receipt'].get('result', {}).get('outcome', {}).get('status') != 'invalid_submission']
        require(len(matches) == 1, 'formal_interval_missing_or_ambiguous')
        i, job = matches[0]
        boundaries.append((job['reserved_at'], i))
    return [j for i, j in enumerate(jobs)
            if j['kind'] == 'help' and j['state'] == 'settled'
            and re.fullmatch(r'help:e\d+', j['id']) and delivered(j.get('receipt', {}))
            and boundaries[0] < (j['reserved_at'], i) < boundaries[1]
            and boundaries[0][0] <= j.get('claimed_at', float('-inf'))
                <= j.get('settled_at', float('inf')) <= boundaries[1][0]]


def answer_projection(raw, version):
    eid = raw.get('execution_id')
    require(isinstance(eid, str) and re.fullmatch(r'e\d+', eid)
            and raw.get('help_id') == 'help:' + eid and delivered(raw), 'invalid_execution_help')
    return dict(raw, version_id=f'v{version - 1:04d}', help_execution_id=eid,
                execution_help=raw)


def job_projection(raw, version):
    require(raw['id'] == raw['receipt'].get('help_id'), 'execution_help_identity_drift')
    return dict(raw, id=f'help:v{version - 1:04d}',
                receipt=answer_projection(raw['receipt'], version), execution_job=raw)


def delivery_projection(raw, answer):
    require(raw.get('op') == 'ask_help' and raw.get('status') == 'completed'
            and raw.get('result') == answer['execution_help'], 'help_delivery_missing_or_ambiguous')
    return dict(raw, result=answer, execution_delivery=raw)


def verify_help(db, cell, version, sealed, answer, delivery):
    execution = sealed.get('execution_job')
    require(isinstance(execution, dict) and isinstance(execution.get('id'), str),
            'agent execution help projection required')
    rows = db.execute('SELECT state,payload FROM v2_agent_jobs WHERE cell=? AND id=?',
                      (cell, execution['id'])).fetchall()
    require(len(rows) == 1 and rows[0][0] == 'settled'
            and dict(json.loads(rows[0][1]), id=execution['id'], state=rows[0][0]) == execution,
            'agent help differs from live ledger')
    jobs = [dict(json.loads(payload), id=jid, state=state) for jid, state, payload in db.execute(
        'SELECT id,state,payload FROM v2_agent_jobs WHERE cell=? ORDER BY rowid', (cell,))]
    matches = interval_help(jobs, version)
    require(len(matches) <= 1, 'multiple_helps_in_interval')
    require(len(matches) == 1 and sealed == job_projection(matches[0], version),
            'execution_help_ledger_drift')
    require(answer == answer_projection(matches[0]['receipt'], version), 'execution_help_receipt_drift')
    expected = (sealed['receipt'] if answer.get('system_injected') is True else
                delivery_projection(delivery.get('execution_delivery', {}), answer))
    require(delivery == expected, 'execution_help_delivery_drift')
