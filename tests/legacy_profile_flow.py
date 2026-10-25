"""Deterministic legacy import/validation/retrieval response capture for C1."""
import json
from unittest.mock import patch

from attention_memory_service.core.memory import MemoryManager
from tests.test_memory import candidate
from tests.test_store import populated_store
from tests.test_provisional_notes import payload, service


def capture(root):
    store, _ = populated_store(root / 'core.sqlite')
    manager = MemoryManager(store)
    values = [manager.add_candidate(candidate()).artifact(),
              manager.record_validation('memory-1', successes=4, failures=1).artifact(),
              manager.promote('memory-1').artifact()]
    values.append([m.artifact() for m in manager.retrieve(
        {'suite': 'robosuite', 'failure_stage': 'grasp'}, now=20)])
    with patch('time.time', return_value=2000000000.0):
        svc = service(root)
        req = payload()
        values.append(svc.create_provisional_note(**req))
        values.append(svc.list_dev_v2_memories(context=req['context']))
        values.append([m.artifact() for m in svc.retrieve(req['context'], now=2000000000.0)])
    return json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
