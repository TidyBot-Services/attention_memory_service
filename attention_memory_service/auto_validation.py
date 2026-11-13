"""Observational queue ranking and lifecycle notices; never a promotion gate."""
import json
import sqlite3
import time

from .applicability import profile_fields
from .dev_v2 import encoded, require
from .seeded_notes import is_provisional, is_test_seeded


def rank_validation_queue(items):
    """Pure ordering, with a stable ID tie-break for equal creation times."""
    return sorted(items, key=lambda r: (-r['failure_gone'], -r['uses'], r['created_at'], r['memory_id']))


def validation_queue(service):
    now = time.time()
    with sqlite3.connect(service.store.path) as db:
        memories = [json.loads(r[0]) for r in db.execute('SELECT payload FROM memories')]
        provenance = {r[0]: json.loads(r[1]) for r in db.execute('SELECT memory_id,payload FROM memory_v2_provenance')}
        planned = {r[0] for r in db.execute('SELECT memory_id FROM memory_v2_plans')}
        decisions = {r[0]: json.loads(r[1]) for r in db.execute('SELECT memory_id,payload FROM memory_v2_decisions')}
        decided = set(decisions)
        outcomes = [json.loads(r[0]) for r in db.execute('SELECT payload FROM memory_provisional_outcomes')]
        grants = {r[0]: json.loads(r[1]) for r in db.execute('SELECT grant_id,payload FROM memory_dev_exposures')}
    rows = []
    for memory in memories:
        mid = memory['memory_id']
        p = provenance.get(mid, {})
        if (memory['status'] != 'provisional' or not is_provisional(p) or is_test_seeded(p) or p.get('generalized_from')
                or (p.get('decision_link') or {}).get('decision') not in {None, 'inconclusive'}
                or memory.get('expires_at') is not None and memory['expires_at'] <= now):
            continue
        # A demotion starts a new eligibility cycle. Keep the old immutable
        # plan/decision for audit; revalidation uses a fresh linked candidate.
        if not p.get('demoted_at') and mid in planned | decided:
            continue
        links = [(cid, link) for cid, link in provenance.items() if link.get('from_provisional') == mid
                 and link.get('from_provisional_version', 1) == memory['version']]
        if any(decisions.get(cid, {}).get('decision') != 'inconclusive' for cid, _ in links):
            continue
        uses = [r for r in outcomes if r['memory_id'] == mid]
        if not uses:
            continue
        after = max((decisions[cid]['decided_at'] for cid, _ in links), default=0)
        if after and not any(r['recorded_at'] > after and
                             grants.get(r['grant_id'], {}).get('exposed_at', 0) > after for r in uses):
            continue
        origin = p.get('origin', {})
        if p.get('channel') == 'dev_v2' or p.get('channel')=='dev_agent' and p.get('source_import_id'):
            source = service.dev_v2.get(p['source_import_id'])
            execution = service.dev_v2.get(source['failed_execution_import_id'])
            manifest = service.dev_v2.ref(execution['catalog_id'], execution['manifest'])
            origin = dict(experiment_id=manifest['experiment_id'],
                          cell=execution['cell'], origin_version=execution['version'])
        rows.append(dict(**profile_fields(memory['applicability']),
                         memory_id=mid, version=memory['version'], title=p.get('title'),
                         uses=len(uses), failure_gone=sum(r['signature_present'] is False for r in uses),
                         created_at=memory['created_at'], layer='scene', source_summary=dict(
                             task=memory['applicability']['task_id'], seed=p['source_seed'],
                             experiment_id=origin.get('experiment_id', p.get('source_run_id')),
                             cell=origin.get('cell'), version=origin.get('origin_version'))))
    return rank_validation_queue(rows)


def lifecycle(db, memory, provenance, event, reason, actor, *, at=None, use_ids=None, decision_link=None, suffix=None):
    value = dict(memory_id=memory['memory_id'], title=provenance.get('title'), event=event,
                 reason=reason, time=time.time() if at is None else at, actor=actor,
                 memory=memory)
    if use_ids is not None:
        value['use_ids'] = use_ids
    if decision_link is not None:
        value['decision_link'] = decision_link
    event_key = f"memory-lifecycle:{memory['memory_id']}:{memory['version']}:{event}"
    if suffix is not None:
        event_key += ':' + suffix
    if decision_link is not None:
        event_key += ':' + decision_link['memory_id']
    db.execute("INSERT OR IGNORE INTO events(event_key,entity_type,entity_id,event_type,payload) "
               "VALUES (?,'memory',?,?,?)", (
                   event_key,
                   memory['memory_id'], 'memory.' + event, encoded(value)))


def memory_events(store, *, after=0):
    require(type(after) is int and after >= 0, 'after must be a nonnegative sequence')
    with sqlite3.connect(store.path) as db:
        rows = db.execute("SELECT sequence,payload FROM events WHERE sequence>? "
                          "AND event_key LIKE 'memory-lifecycle:%' ORDER BY sequence", (after,)).fetchall()
    return [dict(seq=seq, **{k: json.loads(payload)[k] for k in
                            ('memory_id', 'title', 'event', 'reason', 'time')}) for seq, payload in rows]


def demote_if_due(db, memory_id):
    """Called in the outcome transaction: concurrent records/retries cross once."""
    memory = json.loads(db.execute('SELECT payload FROM memories WHERE id=?', (memory_id,)).fetchone()[0])
    if memory['status'] != 'trusted':
        return
    p = json.loads(db.execute('SELECT payload FROM memory_v2_provenance WHERE memory_id=?', (memory_id,)).fetchone()[0])
    plan = db.execute('SELECT payload FROM memory_v2_plans WHERE memory_id=?', (memory_id,)).fetchone()
    if p.get('channel') not in {'dev_v2','dev_agent'} or not plan or json.loads(plan[0]).get('profile') != 'single_scene_v1':
        return
    uses = [json.loads(r[0]) for r in db.execute(
        'SELECT payload FROM memory_provisional_outcomes WHERE memory_id=? ORDER BY rowid', (memory_id,))]
    uses = [r for r in uses if r['memory_version'] == memory['version']]
    uses.sort(key=lambda r: r['recorded_at'])  # Stable row order breaks clock ties.
    decision_row = db.execute('SELECT payload FROM memory_v2_decisions WHERE memory_id=?', (memory_id,)).fetchone()
    decision = json.loads(decision_row[0]) if decision_row else {}
    task = decision.get('trust_scope') == 'task_scoped'
    other = False
    if task:
        grants = {r[0]: json.loads(r[1]) for r in db.execute(
            'SELECT grant_id,payload FROM memory_dev_exposures WHERE memory_id=?', (memory_id,))}
        origin_uses = [r for r in uses if grants.get(r['grant_id'], {}).get('context', {}).get('seed') == p['source_seed']]
        other_uses = [r for r in uses if type(grants.get(r['grant_id'], {}).get('context', {}).get('seed')) is int
                      and grants[r['grant_id']]['context']['seed'] != p['source_seed']]
        other = len(other_uses) >= 5 and sum(r['signature_present'] is True for r in other_uses[-5:]) >= 3
        uses = other_uses if other else origin_uses
    last = uses[-5:]
    if len(last) < 5 or sum(r['signature_present'] is True for r in last) < 3:
        return
    now = time.time()
    reason = 'target failure in 3 of last 5 uses' + (' on other seeds' if other else '')
    if task:
        memory.update(version=memory['version'] + 1, status_reason=reason)
        p.update(trust_scope='scene_scoped', task_demoted_at=now,
                 scene_trust_versions=[*p.get('scene_trust_versions', []), memory['version']])
        decision.update(trust_scope='scene_scoped')
        db.execute('UPDATE memories SET payload=? WHERE id=?', (encoded(memory), memory_id))
        db.execute('UPDATE memory_v2_provenance SET payload=? WHERE memory_id=?', (encoded(p), memory_id))
        db.execute('UPDATE memory_v2_decisions SET payload=? WHERE memory_id=?', (encoded(decision), memory_id))
        lifecycle(db, memory, p, 'demoted', reason, 'system:auto-demotion', at=now,
                  use_ids=[r['grant_id'] for r in last])
        return
    expires_at = memory.get('expires_at')
    memory.update(status='provisional', version=memory['version'] + 1, status_reason=reason,
                  expires_at=now + 14 * 86400 if expires_at is None else min(expires_at, now + 14 * 86400))
    p.update(demoted_at=now, trust_scope='provisional', unvalidated=True, excluded_from_results=True)
    if 'cross_seed_validation' in decision:
        decision.update(trust_scope='provisional')
        db.execute('UPDATE memory_v2_decisions SET payload=? WHERE memory_id=?', (encoded(decision), memory_id))
    db.execute('UPDATE memories SET payload=? WHERE id=?', (encoded(memory), memory_id))
    db.execute('UPDATE memory_v2_provenance SET payload=? WHERE memory_id=?', (encoded(p), memory_id))
    lifecycle(db, memory, p, 'demoted', reason, 'system:auto-demotion', at=now,
              use_ids=[r['grant_id'] for r in last])


def validation_reason(decision):
    if 'failed_check' in decision:
        return decision['reason']
    report, failures = decision['report'], decision['failed_gates']
    n = report['paired_runs']
    control, treatment = report['control_signature_count'], report['treatment_signature_count']
    if not failures:
        return (f'Passed: the failure stopped in {n - treatment} of {n} runs with the note '
                f'(it happened in {control} of {n} without)')
    reasons = dict(control_signature_not_reproduced=f'without the note the failure only happened in {control} of {n} runs',
                   treatment_signature_remains=f'with the note the failure still happened in {treatment} of {n} runs',
                   safety_regression='safety worsened with the note',
                   treatment_stopped_earlier='a run with the note stopped earlier than its paired control',
                   insufficient_pairs='fewer than five pairs were completed',
                   mixed_policy_versions='validation runs used different policy versions',
                   mixed_validation_configuration='validation runs used different configurations')
    return 'Rejected: ' + '; '.join(reasons[code] for code in failures)
