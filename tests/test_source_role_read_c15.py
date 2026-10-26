"""C15 offline regressions, with failure-rule bytes/pin from the real origin.

Catalog shape: origin-c09a3bb45aefc8f5c8ed952ae6ae38c6-source.json (executor)
and source-ceb768fe5d0681fbfb13d6ce7023fb7da738f6525a830cfcc1469c466b9a95af.json
(source). Paths are relocated into tmp_path; no real services or simulator.
"""
import copy
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.dev_v2 import digest
from attention_memory_service.memory_service import MemoryService
from tests.dev_v2_fixtures import Fixture
from tests import test_feedback_projection_c10b as c10b

REAL_RULES = b'{"schema":"attentionbench.failure-classification.v1","precedence":["code_validation","sdk_usage","planning_execution","task_incomplete","unknown"],"sdk_usage":["attributeerror","typeerror","keyerror","nameerror","has no attribute","missing required","unexpected keyword","list.items"],"planning_execution":["planningrejected","planning rejected","collision","continuous-command","continuous command","4mrad","contact loss","timeout","timed_out"],"task_incomplete":"normal completed process with independently evaluated native false only"}\n'
REAL_RULES_SHA = '36bbdb284606b6d887355e89eb8fd0036ccf81eecf980d67aed5fb31db2c39ed'

def source_root(tmp_path):
    return tmp_path / 'Tidybot-Universe-attention-native/benchmarks/attention_harness'


@contextmanager
def external_rules(tmp_path, *, external_lock=False):
    """Keep the lock in executor evidence, but the frozen rules in source evidence."""
    write, seal = Fixture.write, Fixture.seal
    root = source_root(tmp_path)

    def relocated_write(path, value):
        if path.name == 'dev_failure_rules.json':
            path = root / path.name
            value = REAL_RULES.decode()
        elif external_lock and path.name == 'lock.json':
            path = root / path.name
        return write(path, value)

    def sealed(f):
        catalogs = seal(f)
        for name, record in f.records.items():
            catalogs['source-' + name] = dict(
                role='source', root=str(root), context=copy.deepcopy(record['context']),
                files={p.name: digest(p.read_bytes()) for p in root.iterdir()})
        return catalogs

    with patch.object(Fixture, 'write', staticmethod(relocated_write)), \
         patch.object(Fixture, 'seal', sealed):
        yield


@pytest.mark.parametrize('external_lock', [False, True])
def test_source_import_reads_external_frozen_rules(tmp_path, external_lock):
    assert digest(REAL_RULES) == REAL_RULES_SHA
    with external_rules(tmp_path, external_lock=external_lock):
        f, service = c10b.enriched_source(tmp_path, planner=True)
    source = service.import_dev_v2_source(
        import_id='source', catalog_id='failed', descriptor='failed/source.json')
    assert source['feedback_projection'] == 'with_eval_review'
    assert service.dev_v2.source('source') == source
    assert service.create_dev_v2_candidate(memory_id='m1', source_import_id='source', created_at=1)
    executor = service.dev_v2.catalog('failed')
    assert not Path(source_root(tmp_path)).is_relative_to(executor['root'])
    assert 'dev_failure_rules.json' not in executor['files']


@pytest.mark.parametrize('fault', [None, 'digest', 'context', 'unregistered', 'unsealed', 'role', 'drift'])
def test_source_read_requires_matching_context_role_path_and_digest(tmp_path, fault):
    f = Fixture(tmp_path)
    failed = f.make('failed')
    root = source_root(tmp_path)
    root.mkdir(parents=True)
    path = root / 'dev_failure_rules.json'
    path.write_bytes(REAL_RULES)
    ref = dict(uri=str(path), sha256=REAL_RULES_SHA)
    catalogs = f.seal()
    source = dict(role='source', root=str(root), context=copy.deepcopy(failed['context']),
                  files={'dev_failure_rules.json': REAL_RULES_SHA})
    if fault == 'context':
        source['context']['namespace'] = 'another-context'
    if fault == 'role':
        source['role'] = 'contract'
    if fault == 'unsealed':
        other = f.write(root / 'other.json', {})
        source['files'] = {'other.json': other['sha256']}
    if fault != 'unregistered':
        catalogs['source'] = source
    service = MemoryService(tmp_path / 'authority.sqlite', dev_v2_catalogs=catalogs)
    if fault == 'digest':
        ref['sha256'] = '0' * 64
    if fault == 'drift':
        path.write_text('{}')
    if fault is None:
        assert service.dev_v2.source_ref('failed', ref) == json.loads(REAL_RULES)
        # A matching later catalog must still be found after a wrong-context pin.
        wrong = copy.deepcopy(source)
        wrong['context']['namespace'] = 'wrong-context'
        service.dev_v2.register('aaa-wrong-context', wrong)
        assert service.dev_v2.source_ref('failed', ref) == json.loads(REAL_RULES)
    else:
        message = 'registered evidence digest changed' if fault == 'drift' else 'artifact outside registered root'
        with pytest.raises(StateConflictError, match=message):
            service.dev_v2.source_ref('failed', ref)
        with pytest.raises(StateConflictError):
            service.dev_v2.verify_file('failed', ref)


def test_executor_root_reads_keep_exact_pin_checks(tmp_path):
    f, service = c10b.enriched_source(tmp_path, planner=True)
    ref = f.records['failed']['descriptor']['manifest']
    assert service.dev_v2.source_ref('failed', ref) == service.dev_v2.ref('failed', ref)
    with pytest.raises(StateConflictError, match='artifact is not preregistered at exact digest'):
        service.dev_v2.source_ref('failed', dict(ref, sha256='0' * 64))
    unsealed = f.write(f.root / 'unsealed.json', {})
    with pytest.raises(StateConflictError, match='artifact is not preregistered at exact digest'):
        service.dev_v2.source_ref('failed', unsealed)
    assert service.import_dev_v2_source(
        import_id='source', catalog_id='failed', descriptor='failed/source.json')['kind'] == 'source'


def test_validation_generation_reads_external_frozen_rules(tmp_path):
    from tests import test_arm_generation, test_single_scene
    from tests.test_provider_source_c10a import codex_writer

    def service_with_source(path, *, dev_v2_catalogs):
        # setup_scene deliberately keeps only the origin executor/validator.
        # Register its source catalog as well before importing that origin.
        catalogs = dict(dev_v2_catalogs)
        catalogs['source-failed'] = dict(
            role='source', root=str(source_root(tmp_path)),
            context=copy.deepcopy(catalogs['failed']['context']),
            files={'dev_failure_rules.json': REAL_RULES_SHA})
        return MemoryService(path, dev_v2_catalogs=catalogs)

    with external_rules(tmp_path), \
         patch.object(test_single_scene, 'MemoryService', service_with_source), \
         patch.object(Fixture, 'write', staticmethod(codex_writer(c10b.projected_writer(Fixture.write)))), \
         patch.object(Fixture, 'add_advice', c10b.with_eval(Fixture.add_advice)):
        f, service, plan = test_arm_generation.setup(tmp_path)
    plan['dev_model'] = dict(plan['dev_model'], model='controlled_dev_model')
    service.record_plan('m1', plan)
    with patch.object(test_arm_generation, 'MODEL', plan['dev_model']), \
         patch.object(Fixture, 'write', staticmethod(codex_writer(c10b.projected_writer(Fixture.write)))):
        test_arm_generation.prepare(f, service, plan, 't0')
    trial = test_arm_generation.import_trial(service, 't0')
    assert trial['trial']['feedback_projection'] == 'with_eval_review'
