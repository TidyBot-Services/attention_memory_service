"""C14: sealed Dev v2 writer timestamp semantics; synthetic, offline inputs."""
import copy
import math
from unittest.mock import patch

import pytest

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.dev_v2 import public_trace_projection
from attention_memory_service.memory_service import MemoryService
from tests.dev_v2_fixtures import Fixture


ROW = {'operation': 'get_observation', 'status': 'completed', 'arguments': {}}


@pytest.mark.parametrize('stamp,basis,kept', [
    (0, 'relative', True), (2, 'relative', True),
    (0.123456789012345, 'relative', True), (-1.25, 'relative', True),
    (-0.0, 'relative', True), (1720000000.1234567, 'epoch', True),
    (-2, 'epoch', True), (1.5, 'absolute', False), (1.5, None, False),
    (float('nan'), 'relative', False), (float('inf'), 'relative', False),
    (float('-inf'), 'epoch', False), (True, 'relative', False),
    ('1.25', 'relative', False), (None, 'relative', False),
])
def test_writer_timestamp_semantics(stamp, basis, kept):
    # dev_loop.event_projection preserves finite relative/epoch values as-is,
    # including negatives; it drops absolute/unknown bases and non-numbers.
    event = {**ROW, 'timestamp': stamp, 'timestamp_basis': basis}
    trace = {'status': 'completed', 'sdk_events': [event]}
    expected = {**ROW, **({'timestamp': stamp, 'timestamp_basis': basis} if kept else {})}
    projected = public_trace_projection(trace)['action_trace']
    assert projected == [expected]
    assert trace['sdk_events'][0] is event and event['timestamp'] is stamp
    if kept:
        assert type(projected[0]['timestamp']) is type(stamp)
        assert math.copysign(1, projected[0]['timestamp']) == math.copysign(1, stamp)


@pytest.mark.parametrize('extra', [{}, {'timestamp_basis': 'relative'}, {'timestamp': 1.25}])
def test_legacy_or_incomplete_timestamp_pair(extra):
    trace = {'status': 'completed', 'sdk_events': [{**ROW, **extra}]}
    before = copy.deepcopy(trace)
    assert public_trace_projection(trace)['action_trace'] == [ROW]
    assert trace == before


def test_event_filter_and_last_40_preserved():
    events = [{**ROW, 'timestamp': i / 10, 'timestamp_basis': 'relative', 'unlisted': 'ignored'}
              for i in range(42)]
    events.append({'operation': 'unsupported', 'timestamp': 99, 'timestamp_basis': 'relative'})
    trace = {'status': 'completed', 'sdk_events': events}
    before = copy.deepcopy(trace)
    assert public_trace_projection(trace)['action_trace'] == [
        {**ROW, 'timestamp': i / 10, 'timestamp_basis': 'relative'} for i in range(2, 42)]
    assert trace == before


def test_nonfinite_timestamp_does_not_bypass_other_checks():
    event = {**ROW, 'timestamp': float('nan'), 'timestamp_basis': 'relative'}
    with pytest.raises(StateConflictError, match='private evaluation content'):
        public_trace_projection({'status': 'completed', 'sdk_events': [{**event, 'hidden': 'secret'}]})
    with pytest.raises(ValueError):
        public_trace_projection({'status': 'completed', 'sdk_events': [{**event, 'other': float('nan')}]})


@pytest.mark.parametrize('timestamped,fault', [
    (False, None), (True, None), (False, 'unknown_key'), (True, 'unknown_key'),
    (False, 'changed_timestamp'), (True, 'changed_timestamp'), (True, 'missing_timestamp'),
])
def test_execution_import_enforces_exact_event_projection(tmp_path, timestamped, fault):
    timing = {'timestamp': 0.123456789012345, 'timestamp_basis': 'relative'} if timestamped else {}
    write = Fixture.write

    def writer(path, value):
        if path.name == 'trace.json':
            value = {**value, 'sdk_events': [{**ROW, **timing}]}
        elif path.name == 'feedback.json':
            # Independent expected feedback, not computed by the service.
            row = {**ROW, **timing}
            if fault == 'unknown_key':
                row['unsealed_hint'] = 'extra'
            elif fault == 'changed_timestamp':
                row.update(timestamp=99.0, timestamp_basis='relative')
            elif fault == 'missing_timestamp':
                row.pop('timestamp')
                row.pop('timestamp_basis')
            value = {**value, 'action_trace': [row]}
        return write(path, value)

    with patch.object(Fixture, 'write', staticmethod(writer)):
        fixture = Fixture(tmp_path)
        fixture.make('execution')
    service = MemoryService(tmp_path / 'authority.sqlite', dev_v2_catalogs=fixture.seal())
    args = dict(import_id='execution', catalog_id='execution', descriptor='execution/execution.json')
    if fault:
        with pytest.raises(StateConflictError, match='public feedback differs from original trace projection'):
            service.import_dev_v2_execution(**args)
    else:
        assert service.import_dev_v2_execution(**args)
