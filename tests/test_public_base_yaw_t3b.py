"""T3b: sealed writer choice, exact feedback equality, and all recomputation paths."""
import copy
import inspect
from unittest.mock import patch

import pytest

from attention_memory_service import dev_v2, cross_seed
from attention_memory_service.core.store import StateConflictError
from attention_memory_service.memory_service import MemoryService
from tests.dev_v2_fixtures import Fixture


FIELD = 'public_base_yaw_visible'
ABSENT = 'absent'
H1 = {'budget_tier': 'dev', 'dev_version_history': True}
BASE = {'operation': 'move_delta', 'status': 'completed',
        'arguments': {'dx': .1, 'dy': .2, 'dyaw': .3}}


@pytest.mark.parametrize('source', [ABSENT, None, 'base', 'left_arm', 'right_arm', 'sensors', 'bimanual'])
@pytest.mark.parametrize('operation', ['move_delta', 'move_joints', 'get_observation'])
@pytest.mark.parametrize('visible', [False, True])
def test_base_yaw_is_public_only_for_new_history_runs(source, operation, visible):
    from benchmarks.attention_harness.dev_loop import public_feedback
    event = {**BASE, 'operation': operation, **({} if source == ABSENT else {'source': source})}
    trace = {'status': 'completed', 'sdk_events': [event]}
    before = copy.deepcopy(trace)
    actual = dev_v2.public_trace_projection(trace, base_yaw_visible=visible)
    assert actual == public_feedback(trace, base_yaw_visible=visible)
    assert ('dyaw' in actual['action_trace'][0]['arguments']) is (
        visible and operation == 'move_delta' and source in (ABSENT, None, 'base'))
    assert trace == before
    assert dev_v2.encoded(dev_v2.public_trace_projection(trace)) == dev_v2.encoded(
        public_feedback(trace, base_yaw_visible=False))


@pytest.mark.parametrize('value', [None, 0, 1, 'true', [], {}])
def test_non_boolean_decision_refused(value):
    with pytest.raises(StateConflictError, match='must be a bool'):
        dev_v2.public_base_yaw_visible({FIELD: value}, H1)


@pytest.mark.parametrize('config', [{}, {'budget_tier': 'dev'}, {'dev_version_history': True},
    {**H1, 'budget_tier': 'legacy'}, {**H1, 'dev_version_history': False},
    {**H1, 'dev_version_history': 1}, {**H1, 'dev_version_history': 'true'}])
def test_true_requires_exact_frozen_history_config(config):
    with pytest.raises(StateConflictError, match='requires Dev version history'):
        dev_v2.public_base_yaw_visible({FIELD: True}, config)
    assert dev_v2.public_base_yaw_visible({FIELD: False}, config) is False
    assert dev_v2.public_base_yaw_visible({}, config) is False


@pytest.mark.parametrize('yaw', [True, '0.3', None, float('inf')])
def test_visible_yaw_keeps_numeric_validation(yaw):
    trace = {'status': 'completed', 'sdk_events': [{**BASE, 'arguments': {'dyaw': yaw}}]}
    with pytest.raises((StateConflictError, ValueError)):
        dev_v2.public_trace_projection(trace, base_yaw_visible=True)


def fixture_writer(flag, config, feedback_yaw):
    """Alter only synthetic inputs before any digest/ledger/catalog is produced."""
    original = Fixture.write
    def write(path, value):
        if path.name == 'config.json':
            value.update(config)
        elif path.name == 'bundle.json' and flag != ABSENT:
            value[FIELD] = flag
        elif path.name == 'trace.json':
            if value.get('sdk_events', [])[:1] != [BASE]:
                value['sdk_events'] = [copy.deepcopy(BASE), *value.get('sdk_events', [])]
        elif path.name == 'feedback.json':
            row = copy.deepcopy(BASE)
            if not feedback_yaw:row['arguments'].pop('dyaw')
            value['action_trace'][0] = row
        return original(path, value)
    return staticmethod(write)


@pytest.mark.parametrize('flag,config,yaw,error', [
    (True, H1, True, None), (True, H1, False, 'public feedback differs'),
    (False, H1, False, None), (False, H1, True, 'public feedback differs'),
    (ABSENT, H1, False, None), (ABSENT, H1, True, 'public feedback differs'),
    (False, {}, False, None), (ABSENT, {}, False, None),
    (False, {}, True, 'public feedback differs'), (ABSENT, {}, True, 'public feedback differs'),
    (True, {}, True, 'requires Dev version history'),
    (True, {**H1, 'dev_version_history': 1}, True, 'requires Dev version history'),
    (True, {**H1, 'budget_tier': 'legacy'}, True, 'requires Dev version history'),
    (None, H1, False, 'must be a bool'), (1, H1, True, 'must be a bool'),
    ('true', H1, True, 'must be a bool'),
])
def test_execution_import_requires_exact_sealed_projection(tmp_path, flag, config, yaw, error):
    with patch.object(Fixture, 'write', fixture_writer(flag, config, yaw)):
        f = Fixture(tmp_path)
        f.make('execution')
    service = MemoryService(tmp_path / 'authority.sqlite', dev_v2_catalogs=f.seal())
    args = dict(import_id='execution', catalog_id='execution', descriptor='execution/execution.json')
    if error:
        with pytest.raises(StateConflictError, match=error):service.import_dev_v2_execution(**args)
    else:
        assert service.import_dev_v2_execution(**args)


@pytest.mark.parametrize('flag', [True, False, ABSENT])
def test_trial_and_cross_seed_recompute_with_sealed_flag(tmp_path, monkeypatch, flag):
    from tests.test_single_scene import setup_scene, import_pairs
    from tests.test_cross_seed import target
    visible = flag is True
    with patch.object(Fixture, 'write', fixture_writer(flag, H1, visible)):
        f, service, plan = setup_scene(tmp_path, interface=True)
        service.record_plan('m1', plan)
        original = dev_v2.public_trace_projection
        calls = []
        def projection(trace, *, base_yaw_visible):
            # Required kwarg makes a forgotten call site fail, even for False.
            caller = inspect.currentframe().f_back.f_code.co_name
            assert base_yaw_visible is visible
            result = original(trace, base_yaw_visible=base_yaw_visible)
            assert ('dyaw' in result['action_trace'][0]['arguments']) is visible
            calls.append(caller)
            return result
        monkeypatch.setattr(dev_v2, 'public_trace_projection', projection)
        monkeypatch.setattr(cross_seed, 'public_trace_projection', projection)
        import_pairs(f, service)
        assert service.evaluate('m1').status.value == 'trusted'
        args, _ = target(f, service, 114)
        assert service.cross_seed.probe(**args)['role'] == 'transfer'
    assert {'verify_execution', 'import_trial', 'probe'} <= set(calls)
