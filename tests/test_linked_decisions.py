"""A checked candidate atomically retires its still-visible provisional source."""
import time

import pytest

from attention_memory_service import auto_validation, families
from attention_memory_service.core.store import StateConflictError
from attention_memory_service.memory_service import MemoryService
from attention_memory_service.memory_service_api import create_app
from tests.test_auto_validation import outcome, scene_feedback
from tests.test_provisional_notes import API, CHANGED
from tests.test_single_scene import setup_scene, import_pairs
from tests.test_sync_legacy_routes import sync_request

PID = 'p-' + 'a' * 24


def linked_scene(tmp_path, rejected=False):
    controls = (True, True, False, False, False) if rejected else (True,) * 5
    f, service, plan = setup_scene(tmp_path, controls=controls, interface=True, linked=True)
    ctx = service.store.get_memory(PID).applicability
    outcome(service, PID, 0, CHANGED, ctx)
    service.record_plan('m1', plan)
    import_pairs(f, service)
    return f, service, ctx


@pytest.mark.parametrize('rejected', [False, True])
def test_linked_source_off_visible_and_operator_can_reenable(tmp_path, monkeypatch, rejected):
    _, service, ctx = linked_scene(tmp_path, rejected)
    monkeypatch.setattr(families, 'FAMILIES', {'pick_and_place': ('pick', 'pick_place_*')})
    queries = [ctx, {**ctx, 'seed': 102},
               {**ctx, 'task_id': 'pick_place_plate', 'task_prompt': 'Move a plate'}]
    for query in queries:
        assert PID in [m.memory_id for m, _ in service.retrieve_levels(
            query, now=time.time(), levels=('same_scene', 'same_task', 'same_family'))]
    old_grant = service.authorize_dev_use(memory_id=PID, memory_version=1, development_id='pending:v1',
        context=ctx, source_policy_sha256='c' * 64, now=time.time())
    service.evaluate('m1')
    event = 'rejected' if rejected else 'superseded'
    reason = 'rejected after automatic check' if rejected else 'replaced by its checked version m1'
    candidate_outcome = 'rejected' if rejected else 'promoted'
    note = service.get_memory(PID)
    assert note.status.value == 'disabled' and note.status_reason == reason
    for query in queries:
        assert PID not in [m.memory_id for m, _ in service.retrieve_levels(
            query, now=time.time(), levels=('same_scene', 'same_task', 'same_family'))]
    assert service.validation_queue() == []
    with pytest.raises(StateConflictError): service.get_dev_use(old_grant['grant_id'])
    row = service.get_dev_v2_memory_public(PID)
    assert row['status'] == 'disabled' and row['unvalidated']
    assert row['decision_link'] == dict(memory_id='m1', decision=candidate_outcome, reason=reason,
                                      decided_at=service.decision('m1')['decided_at'])
    assert row in service.list_dev_v2_memories(status='disabled')
    events = [e for e in service.memory_events() if e['memory_id'] == PID]
    assert len(events) == 1 and (events[0]['event'], events[0]['reason']) == (event, reason)
    history = service.history(PID)
    assert [(h['status'], h['actor']) for h in history[-2:]] == [
        ('disabled', 'system:validation'), (event, 'system:validation')]
    audit = service.store.events()
    service.evaluate('m1')
    assert service.store.events() == audit
    reopened = MemoryService(service.store.path)
    assert reopened.get_dev_v2_memory_public(PID) == row
    app = create_app(service.store.path, api_key=API)
    assert sync_request(app, f'/dev-v2/memories/{PID}', key=API) == (200, row)
    enabled = reopened.enable(PID, actor='operator:test', reason='deliberate reread')
    assert enabled.status.value == 'provisional'
    assert PID in [m.memory_id for m in reopened.retrieve(ctx, now=time.time())]
    assert reopened.validation_queue() == []
    assert reopened.get_dev_v2_memory_public(PID)['decision_link'] == row['decision_link']
    assert reopened.history(PID)[-1]['actor'] == 'operator:test'
    assert reopened.history(PID)[-1]['reason'] == 'deliberate reread'
    # Retrying the completed decision must not undo an explicit operator enable.
    reopened.evaluate('m1')
    assert reopened.store.get_memory(PID).status.value == 'provisional'


@pytest.mark.parametrize('rejected', [False, True])
def test_decision_source_shutdown_and_audit_roll_back_together(tmp_path, monkeypatch, rejected):
    _, service, _ = linked_scene(tmp_path, rejected)
    before = service.store.events()
    original = auto_validation.lifecycle
    def fail_after_shutdown(db, memory, provenance, event, reason, actor, **kwargs):
        original(db, memory, provenance, event, reason, actor, **kwargs)
        if memory['memory_id'] == PID:
            assert memory['status'] == 'disabled'
            assert db.execute('SELECT 1 FROM memory_v2_decisions WHERE memory_id=?', ('m1',)).fetchone()
            # Other connections still see the pre-decision state.
            assert service.store.get_memory('m1').status.value == 'candidate'
            assert service.store.get_memory(PID).status.value == 'provisional'
            assert service.v2.get_decision('m1') is None
            raise RuntimeError('controlled failure after source lifecycle write')
    with monkeypatch.context() as patch:
        patch.setattr(auto_validation, 'lifecycle', fail_after_shutdown)
        with pytest.raises(RuntimeError, match='controlled failure'):
            service.evaluate('m1')
    assert service.store.get_memory('m1').status.value == 'candidate'
    assert service.store.get_memory(PID).status.value == 'provisional'
    assert service.v2.get_decision('m1') is None
    assert 'decision_link' not in service.v2._provenance(PID)
    assert service.store.events() == before
    service.evaluate('m1')
    assert service.store.get_memory(PID).status.value == 'disabled'


def test_candidate_demotion_does_not_revive_its_source(tmp_path):
    f, service, ctx = linked_scene(tmp_path)
    service.evaluate('m1')
    before = service.get_dev_v2_memory_public(PID)
    for i in range(5):
        outcome(service, 'm1', i, scene_feedback(f, True), ctx)
    assert service.store.get_memory('m1').status.value == 'provisional'
    assert service.get_dev_v2_memory_public(PID) == before
    assert [m.memory_id for m in service.retrieve(ctx, now=time.time())] == ['m1']
    assert [r['memory_id'] for r in service.validation_queue()] == ['m1']
    assert service.memory_events()[-1]['event'] == 'demoted'


@pytest.mark.parametrize('state', ['disabled', 'expired', 'rolled_back'])
def test_existing_operator_revocation_is_preserved(tmp_path, state):
    _, service, ctx = linked_scene(tmp_path)
    if state == 'disabled': service.disable(PID, actor='op', reason='operator off')
    elif state == 'rolled_back': service.rollback(PID, actor='op', reason='operator withdraw')
    else:
        service.set_expiry(PID, expires_at=0, actor='op', reason='operator expiry')
        service.retrieve(ctx, now=time.time())
    before = service.store.get_memory(PID)
    service.evaluate('m1')
    assert service.store.get_memory(PID) == before
    assert service.get_dev_v2_memory_public(PID)['decision_link']['memory_id'] == 'm1'
    assert service.validation_queue() == []
