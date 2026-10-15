"""Controlled isolated authorities; no real provider/native acceptance or usage."""
import time

import pytest

from attention_memory_service.core.memory import MemoryManager
from attention_memory_service.core.store import StateConflictError
from attention_memory_service.memory_service import MemoryService
from tests.test_dev_v2 import make_promoted
from tests.test_memory import candidate
from tests.test_store import populated_store


def test_future_queries_cannot_expire_other_namespace_or_legacy(tmp_path):
    f, service, trusted = make_promoted(tmp_path)
    ctx = f.records['failed']['context']
    wall = time.time()
    service.set_expiry('m1', expires_at=wall + 3600,
                       actor='controlled_test_operator', reason='controlled_test clock')
    # A legacy record in the same temporary authority is also outside this query.
    store, _ = populated_store(service.store.path)
    legacy = MemoryManager(store)
    legacy.add_candidate(candidate(expires_at=wall + 3600))
    legacy.record_validation('memory-1', successes=3, failures=0)
    legacy.promote('memory-1')
    before = service.store.list_memories()
    events = service.store.events()
    wrong = {**ctx, 'namespace': 'controlled_test_other'}
    for query in (wrong, ctx):
        found = service.retrieve(query, now=wall + 7200)
        assert [m.memory_id for m in found] == ([] if query == wrong else ['m1'])
    assert service.store.list_memories() == before
    assert service.store.events() == events
    # Exercise provenance -> plan -> impact and both grant -> retrieve paths.
    assert service.provenance('m1')['channel'] == 'dev_v2'
    assert service.get_plan('m1')['memory_id'] == 'm1'
    grant = service.authorize_dev_use(
        memory_id='m1', memory_version=trusted.version, development_id='clock:v1',
        context=ctx, source_policy_sha256='a' * 64, now=wall + 7200)
    assert service.get_dev_use(grant['grant_id']) == grant
    use = service.authorize_use(ctx, memory_id='m1', attempt_id='controlled_test_attempt',
                                now=wall + 7200)
    assert use['version'] == trusted.version
    with pytest.raises(StateConflictError):
        service.authorize_use(wrong, memory_id='m1', attempt_id='controlled_test_wrong',
                              now=wall + 7200)
    assert service.store.list_memories() == before
    assert service.store.events() == events
    assert legacy.retrieve({'suite': 'robosuite', 'failure_stage': 'grasp'}, now=wall)


def test_past_query_cannot_revive_actual_expiry_and_preserves_history(tmp_path):
    f, service, trusted = make_promoted(tmp_path)
    ctx = f.records['failed']['context']
    wall = time.time()
    args = dict(memory_id='m1', memory_version=trusted.version,
                development_id='clock:v1', context=ctx,
                source_policy_sha256='a' * 64, now=wall)
    grant = service.authorize_dev_use(**args)
    historical = service.authorize_use(ctx, memory_id='m1',
        attempt_id='controlled_test_attempt', now=wall)
    service.set_expiry('m1', expires_at=wall - 1,
                       actor='controlled_test_operator', reason='controlled_test actual expiry')
    assert service.retrieve(ctx, now=wall - 7200) == []
    expired = service.get_memory('m1')
    assert expired.status.value == 'expired' and expired.status_reason == 'expiry_reached'
    assert service.verify_package('m1')['status'] == 'expired'
    events = service.store.events()
    expiry = [e for e in events if e['event_type'] == 'memory.expired']
    assert len(expiry) == 1 and expiry[0]['payload']['reason'] == 'expiry_reached'
    with pytest.raises(StateConflictError):
        service.authorize_dev_use(**{**args, 'now': wall - 7200})
    with pytest.raises(StateConflictError):
        service.get_dev_use(grant['grant_id'])
    with pytest.raises(StateConflictError):
        service.authorize_use(ctx, memory_id='m1', attempt_id='controlled_test_late',
                              now=wall - 7200)
    reopened = MemoryService(service.store.path)
    assert reopened.retrieve(ctx, now=wall - 7200) == []
    assert reopened.store.events() == events
    assert reopened.v2.get_use_grant(historical['grant_id']) == historical
    assert reopened.get_plan('m1')['memory_id'] == 'm1'
    assert len(reopened.list_pairs('m1')) == 5


@pytest.mark.parametrize('now', [True, float('inf'), float('nan'), '123'])
def test_live_clock_still_rejects_invalid_compatibility_time(tmp_path, now):
    service = MemoryService(tmp_path / 'controlled_test.sqlite')
    with pytest.raises(ValueError, match='finite'):
        service.retrieve({'perception_mode': 'sim_gt'}, now=now)
