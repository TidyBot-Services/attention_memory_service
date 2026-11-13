"""Linked, provisional generalized copies; no change to trust/validation gates."""
import json
import math
import time
from dataclasses import replace

from .applicability import SAME_TASK, SAME_SKILL, GENERAL
from .core.models import MemoryStatus
from .dev_v2 import digest, encoded, require, public, view_no_oracle
from .family import validate_rewrite
from .seeded_notes import is_provisional
from .skill_tags import VERSION, tags_for

REWRITE_PROMPT = ('Rewrite this robot guidance for reuse by the same skill and fixture. '
                  'Keep the physical idea, remove source task names, coordinates, seeds and scene-specific measurements. '
                  'Write a short conditional precondition in applies_when. Give a concrete action title of at most 80 characters, '
                  'never a placeholder. Return JSON {"title", "guidance", "applies_when"}; guidance at most 600 characters, '
                  'applies_when at most 240 characters.')
FIELDS = ('level', 'skill_tags', 'skill_tags_version', 'applies_when', 'source_run', 'source_version',
          'source_verified', 'generalized_from', 'generalized_copy')


def source_label(origin, evidence):
    """Importer-attested version bytes, separately labelled from svc trust."""
    if evidence is None:
        return dict(source_verified=False)
    require(isinstance(evidence, dict) and set(evidence) in ({'raw_version', 'sha256'}, {'raw_version','sha256','outcomes'}), 'version evidence required')
    raw = evidence['raw_version']
    require(isinstance(raw, str) and digest(raw.encode()) == evidence['sha256'], 'source version digest mismatch')
    record = json.loads(raw)
    require(type(record.get('version')) is int and record['version'] == origin['help_version']
            and record.get('source', {}).get('sha256') == origin['digests']['help_source'],
            'source version identity mismatch')
    verdict = record.get('agent', {}) if origin.get('channel') == 'dev_agent' else record
    passed = (record.get('valid') is True and verdict.get('native_success') is True
              and verdict.get('eval_verdict') == 'pass')
    if 'outcomes' in evidence:
        require(origin.get('channel') != 'dev_agent', 'agent verdict must come from sealed version')
        texts = evidence['outcomes']
        require(isinstance(texts, dict) and set(texts) == {'bundle','native','eval','execute_job','eval_job'},
                'complete version outcomes required')
        values = {}
        for name, ref in texts.items():
            require(isinstance(ref, dict) and set(ref) == {'raw','sha256'} and isinstance(ref['raw'],str)
                    and digest(ref['raw'].encode()) == ref['sha256'], 'outcome digest mismatch: '+name)
            values[name] = json.loads(ref['raw'])
        bundle, native, review = (values[k] for k in ('bundle','native','eval'))
        for name, kind in (('execute_job','execute'),('eval_job','eval')):
            job = values[name]
            require(job.get('cell') == origin['cell'] and job.get('version') == origin['help_version']
                    and job.get('kind') == kind and job.get('state') == 'settled', 'outcome job identity mismatch')
        receipt = values['execute_job']['receipt']
        er = values['eval_job']['receipt']
        require(receipt['bundle']['sha256'] == texts['bundle']['sha256']
                and receipt['native_result']['sha256'] == texts['native']['sha256']
                and (er.get('eval_review') or er.get('review'))['sha256'] == texts['eval']['sha256'],
                'outcomes differ from settled job receipts')
        require(bundle.get('dev_version') == origin['help_version'] and bundle.get('policy_id') == origin['policy_id']
                and bundle.get('policy_sha256') == origin['digests']['help_source']
                and bundle.get('config_sha256') == record.get('config_sha256')
                and bundle.get('run_id') == 'run:'+origin['cell']
                and bundle.get('attempt_id') == f"attempt:{origin['cell']}:{origin['help_version']-1}",
                'outcome source/version binding mismatch')
        require(bundle.get('native_evaluator') == native
                and bundle.get('native_success') is native.get('native_success')
                and bundle.get('artifacts',{}).get('native_result',{}).get('sha256') == texts['native']['sha256'],
                'native outcome differs from bundle')
        passed = (record.get('valid') is True and native.get('evaluated') is True
                  and native.get('native_success') is True and review.get('status') == 'completed'
                  and review.get('passed') is True)

    # Contradictory embedded Eval evidence cannot earn the label.
    if isinstance(verdict.get('eval'), dict) and verdict['eval'].get('passed') is not True:
        passed = False
    return dict(source_verified=passed, source_verified_evidence=dict(
        sha256=evidence['sha256'], version=origin['help_version'], raw_version=raw,
        **({'outcomes':evidence['outcomes']} if 'outcomes' in evidence else {})))


def metadata(origin, context, evidence=None):
    return dict(level=SAME_TASK, skill_tags=list(tags_for(context['task_id'])), skill_tags_version=VERSION,
                applies_when='When repeating this task.', source_run=origin['experiment_id'],
                source_version=origin['help_version'], **source_label(origin, evidence))


def source(service, memory_id):
    memory = service.store.get_memory(memory_id)
    require(memory is not None, 'unknown original note')
    provenance = service.v2.provenance(memory_id)
    require(is_provisional(provenance) and not provenance.get('generalized_from'), 'original provisional note required')
    return dict(title=provenance.get('title') or memory.guidance[:80], guidance=memory.guidance,
                context=memory.applicability), memory, provenance


def generalize(service, *, memory_id, rewrite):
    r, memory, provenance = source(service, memory_id)
    validate_rewrite(r, rewrite, instruction=REWRITE_PROMPT, generalized=True)
    public({k:rewrite[k] for k in ('title','guidance','applies_when')})
    view_no_oracle({k:rewrite[k] for k in ('title','guidance','applies_when')})
    require(bool(tags_for(memory.applicability['task_id'])), 'source task has no registered skills')
    child_id = memory_id + '-g'
    old = service.store.get_memory(child_id)
    if old:
        p = service.v2.provenance(child_id)
        require(p['model_rewrite'] == rewrite, 'one immutable model rewrite per original note')
        return detail(service, memory_id)
    require(memory.status is MemoryStatus.PROVISIONAL and (memory.expires_at is None or memory.expires_at > time.time()),
            'original note is unavailable')
    child = replace(memory, memory_id=child_id, version=1, guidance=rewrite['guidance'], created_at=time.time())
    artifact = dict(provenance['artifact'], guidance=child.guidance)
    cp = dict(provenance, artifact=artifact, artifact_sha256=digest(encoded(artifact).encode()),
              generalized_from=memory_id, title=rewrite['title'], level=SAME_SKILL,
              applies_when=rewrite['applies_when'], model_rewrite=rewrite)
    cp.pop('generalized_copy', None)
    cp.pop('summary', None)  # Source-scene summary text is not part of the rewrite.
    with service.store._transaction() as db:
        current = json.loads(db.execute('SELECT payload FROM memory_v2_provenance WHERE memory_id=?', (memory_id,)).fetchone()[0])
        stored_memory = json.loads(db.execute('SELECT payload FROM memories WHERE id=?',(memory_id,)).fetchone()[0])
        require(current == provenance and stored_memory == memory.artifact(), 'original changed during generalization; retry')
        current['generalized_copy'] = child_id
        db.execute('INSERT INTO memories(id,payload) VALUES (?,?)', (child_id, encoded(child.artifact())))
        db.execute('INSERT INTO memory_v2_provenance(memory_id,payload) VALUES (?,?)', (child_id, encoded(cp)))
        db.execute('UPDATE memory_v2_provenance SET payload=? WHERE memory_id=?', (encoded(current), memory_id))
        # Existing lifecycle enable logic must retain provisional status.
        event(db, child_id, 'memory.provisional_created', dict(actor='model-rewrite', original=memory_id))
    return detail(service, memory_id)


def verify(manager, memory, provenance):
    evidence = provenance.get('source_verified_evidence')
    require(source_label(provenance['origin'], {k:v for k,v in evidence.items() if k != 'version'} if evidence else None)['source_verified'] == provenance.get('source_verified', False),
            'source verified label differs from evidence')
    require(provenance.get('skill_tags_version') == VERSION
            and provenance.get('skill_tags') == list(tags_for(memory.applicability['task_id'])), 'skill tags drift')
    parent_id = provenance.get('generalized_from')
    if not parent_id:
        require(provenance['level'] == SAME_TASK, 'original L1 level changed')
        return
    require(parent_id != memory.memory_id, 'cyclic generalized note')
    parent = manager.store.get_memory(parent_id)
    pp = manager.provenance(parent_id)
    require(not pp.get('generalized_from') and pp.get('generalized_copy') == memory.memory_id
            and parent.applicability == memory.applicability and parent.source_trace_id == memory.source_trace_id
            and pp['origin'] == provenance['origin'], 'generalized source binding changed')
    validate_rewrite(dict(title=pp.get('title') or parent.guidance[:80], guidance=parent.guidance,
                         context=parent.applicability), provenance['model_rewrite'],
                     instruction=REWRITE_PROMPT, generalized=True)
    confirmed = provenance.get('level_confirmation')
    content = provenance.get('operator_edit') or provenance['model_rewrite']
    if provenance.get('operator_edit'):
        ec = provenance.get('edit_confirmation', {})
        require(ec.get('edit') == provenance['operator_edit'] and isinstance(ec.get('actor'),str)
                and ec['actor'].strip(), 'operator edit evidence missing')
        validate_edit(dict(title=pp.get('title') or parent.guidance[:80], guidance=parent.guidance,
                           context=parent.applicability), provenance['model_rewrite'], provenance['operator_edit'])
    require(memory.guidance == content['guidance'] and provenance['applies_when'] == content['applies_when'],
            'generalized guidance differs from recorded rewrite/edit')
    if confirmed:
        require(isinstance(confirmed.get('actor'), str) and confirmed['actor'].strip()
                and confirmed['level'] == provenance['level'] and type(confirmed.get('memory_version')) is int and 1 <= confirmed['memory_version'] <= memory.version
                and confirmed['confirmed_at'] >= confirmed['opened_at']
                and confirmed['human_attention_seconds'] == confirmed['confirmed_at'] - confirmed['opened_at'],
                'invalid operator confirmation')
    require(provenance['level'] in (SAME_TASK, SAME_SKILL, GENERAL), 'unknown reuse level')


def event(db, mid, kind, payload):
    import uuid
    db.execute('INSERT INTO events(event_key,entity_type,entity_id,event_type,payload) VALUES (?,\'memory\',?,?,?)',
               (uuid.uuid4().hex, mid, kind, encoded(dict(payload, recorded_at=time.time()))))


def detail(service, memory_id):
    r, memory, p = source(service, memory_id)
    child_id = p.get('generalized_copy')
    child = service.store.get_memory(child_id) if child_id else None
    cp = service.v2.provenance(child_id) if child else None
    return dict(original=dict(memory_id=memory_id, title=r['title'], guidance=memory.guidance,
                              **{k:p[k] for k in FIELDS if k in p}),
                generalized=(dict(memory_id=child_id, title=cp['title'], guidance=child.guidance,
                                  confirmation=cp.get('level_confirmation'),
                                  **{k:cp[k] for k in FIELDS if k in cp}) if cp else None))


def validate_edit(source, model_rewrite, edit):
    require(isinstance(edit, dict) and set(edit) == {'guidance', 'applies_when'}, 'edit guidance and applies_when required')
    # Reuse M-11's content checks without changing the retained model receipt.
    candidate = dict(model_rewrite, **edit)
    raw = encoded({k:candidate[k] for k in ('title','guidance','applies_when')})
    candidate['receipt'] = dict(candidate['receipt'], raw_response=raw, response_sha256=digest(raw.encode()))
    validate_rewrite(source, candidate, instruction=REWRITE_PROMPT, generalized=True)
    public(edit); view_no_oracle(edit)


def confirm(service, *, memory_id, level, actor, opened_at, edit=None):
    require(level in (SAME_TASK, SAME_SKILL, GENERAL), 'unknown reuse level')
    require(isinstance(actor, str) and actor.strip().isprintable() and 0 < len(actor.strip()) <= 200, 'operator actor required')
    now = time.time()
    require(type(opened_at) in (int, float) and math.isfinite(opened_at) and 0 <= opened_at <= now,
            'invalid opening time')
    r, parent, pp = source(service, memory_id)
    child_id = pp.get('generalized_copy'); require(child_id is not None, 'generalized copy unavailable')
    child = service.store.get_memory(child_id); cp = service.v2.provenance(child_id)
    require(child.status is MemoryStatus.PROVISIONAL and child.expires_at > now, 'generalized copy unavailable')
    require(parent.status is MemoryStatus.PROVISIONAL and (parent.expires_at is None or parent.expires_at > now),
            'original note is unavailable')
    if edit is not None:
        validate_edit(r, cp['model_rewrite'], edit)
    record = dict(actor=actor.strip(), confirmed_at=now, opened_at=opened_at,
                  human_attention_seconds=now-opened_at, level=level, edit=edit)
    previous = json.loads(encoded(cp))
    previous_memory = child.artifact()
    child = replace(child, version=child.version+1)
    record['memory_version'] = child.version
    cp.update(level=level, level_confirmation=record)
    if edit:
        cp.update(operator_edit=edit, edit_confirmation=record)
        child = replace(child, guidance=edit['guidance'])
        cp['applies_when'] = edit['applies_when']
        cp['artifact']['guidance'] = child.guidance
        cp['artifact_sha256'] = digest(encoded(cp['artifact']).encode())
    with service.store._transaction() as db:
        current = json.loads(db.execute('SELECT payload FROM memory_v2_provenance WHERE memory_id=?', (child_id,)).fetchone()[0])
        stored_child = json.loads(db.execute('SELECT payload FROM memories WHERE id=?',(child_id,)).fetchone()[0])
        stored_parent = json.loads(db.execute('SELECT payload FROM memories WHERE id=?',(memory_id,)).fetchone()[0])
        require(current == previous and stored_child == previous_memory and stored_parent == parent.artifact(),
                'note changed during confirmation; retry')
        db.execute('UPDATE memories SET payload=? WHERE id=?', (encoded(child.artifact()), child_id))
        db.execute('UPDATE memory_v2_provenance SET payload=? WHERE memory_id=?', (encoded(cp), child_id))
        event(db, child_id, 'memory.level_confirmed', record)
    return detail(service, memory_id)
