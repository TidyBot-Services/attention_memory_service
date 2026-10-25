"""Sealed synthetic generation lineage; no simulator/provider execution."""
import json
import sqlite3
from pathlib import Path

import pytest

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.dev_v2 import digest, encoded, public_trace_projection, stop_position
from attention_memory_service.memory_service_api import create_app
from tests.test_single_scene import setup_scene, SIGNATURE
from tests.test_sync_legacy_routes import sync_request

MODEL = {'provider': 'codex', 'model': 'controlled_test_dev'}


def setup(tmp_path, **kwargs):
    f, service, plan = setup_scene(tmp_path, generated=True, **kwargs)
    plan.pop('validation_policy_sha256')
    plan.update(arm_generation='dev_from_origin',
                origin_source_sha256=service.v2.provenance('m1')['source_policy_sha256'], dev_model=MODEL)
    return f, service, plan


def grant_args(r, plan):
    return dict(memory_id='m1', cell=r['name'], version=1, seed=plan.get('target_seed', 101), context=r['context'],
                policy_sha256=r['record']['source']['sha256'], config_sha256=r['descriptor']['config']['sha256'],
                exposure=r['trial']['exposure'], origin_source_sha256=plan['origin_source_sha256'],
                **({'target_seed': plan['target_seed']} if 'target_seed' in plan else {}))


def prepare(f, service, plan, name, *, fault=None, unsafe=False, checks=None):
    r = f.records[name]
    args = grant_args(r, plan)
    if fault == 'granted_sha': args['policy_sha256'] = '0' * 64
    grant = service.authorize_dev_v2_validation(**args)
    f.bind_validation(r, grant)
    d, record = r['descriptor'], r['record']
    directory = f.root / name
    memory = service.store.get_memory('m1')
    treatment = name.startswith('t')
    parent = f.records['failed']['record']['source']
    if fault == 'wrong_parent':
        parent = f.write(directory / 'wrong-parent.py', 'def policy(sdk):\n    return\n')
    notes = [{'content': memory.guidance}] if treatment else []
    if fault in {'control_note', 'second_note'}:
        notes.append({'content': 'other provisional or run note'})
    if fault == 'other_note':
        notes = [{'content': 'other guidance'}]
    if fault == 'old_guidance':
        notes = [{'content': 'guidance from an older version'}]
    packet = {'previous_complete_source': Path(parent['uri']).read_text(), 'attention_control': {'memory': notes}}
    if fault == 'extra_section':
        packet['run_notes'] = [{'content': 'run hint'}]
    if fault == 'help_note':
        packet['attention_control']['help'] = {'content': 'task reference'}
    model = {**MODEL, **({'model': 'another_dev'} if fault == 'model' else {})}
    if fault == 'provider':
        model['provider'] = 'another_provider'
    request = f.write(directory / 'model-input.json', dict(model_source=model['provider'], model=model['model'],
        messages=[{'role': 'system', 'content': 'Write a public SDK policy.'},
                  {'role': 'user', 'content': encoded(packet)}]))
    code = Path(record['source']['uri']).read_text()
    if fault == 'generated_sha': code += '# different output\n'
    content = encoded({'source': code, 'reason': 'controlled_test'})
    usage = {'total_tokens': 7, 'prompt_tokens': 4, 'completion_tokens': 3}
    response = f.write(directory / 'model-output.json', dict(model=model['model'], model_source=model['provider'],
                                                           content=content, usage=usage))
    wire = f.write(directory / 'model-wire.json', dict(http_status=200, request_id='controlled_test',
        body=dict(model=model['model'], usage=usage, choices=[{'message': {'content': content}}])))
    entries = [dict(memory_id='m1', memory_version=memory.version,
                    guidance_sha256=digest(memory.guidance.encode()))] if treatment else []
    if fault == 'old_version': entries[0]['memory_version'] -= 1
    if fault == 'wrong_receipt_hash': entries[0]['guidance_sha256'] = '0' * 64
    exposure = dict(schema_version='attentionbench.dev-memory-exposure.v1', cell=name, version=1,
                    model_input=request, entries=entries)
    exposure_ref = f.write((f.validator / name if fault == 'outside' else directory) / 'dev-memory-exposure.json', exposure)
    record.update(parent_source=parent, model_input=request, model_output=response, dev_memory_exposure=exposure_ref)
    if fault == 'missing': record.pop('dev_memory_exposure')
    d['version_record'] = f.write(directory / 'version.json', record)
    job = dict(job_id=name + '-model', cell=name, version=1, kind='model', claimed_at=1.0, settled_at=2.0,
               actual=dict(dev_calls=1, provider_tokens=7, input_tokens=4, output_tokens=3),
               receipt=dict(request=request, response=response, provider_response=wire, usage=usage))
    with sqlite3.connect(d['ledger']['uri']) as db:
        db.execute('UPDATE v2_versions SET payload=? WHERE version=1', (encoded(record),))
        db.execute('INSERT INTO v2_jobs VALUES (?,?,?,?,?,?)', (job['job_id'], name, 1, 'model', 'settled', encoded(job)))
    d['ledger']['sha256'] = digest(Path(d['ledger']['uri']).read_bytes())
    d['model_job_id'] = job['job_id']
    f.write(directory / 'execution.json', d)
    trace = json.loads(Path(r['bundle']['artifacts']['trace']['uri']).read_text())
    r['verdict'].update(failure_signature=SIGNATURE,
        signature_present=public_trace_projection(trace)['public_error'].startswith(SIGNATURE['public_error_class']),
        stop_position=stop_position(trace))
    r['verdict']['costs']['tokens'] = r['trial']['costs']['tokens'] = 7
    r['verdict']['checks'].update(checks or {})
    if unsafe:
        monitor = Path(r['verdict']['safety']['uri'])
        value = json.loads(monitor.read_text())
        value.update(unsafe_attempts=1, violations=[{'kind': 'controlled_test_sample'}])
        r['verdict']['safety'] = f.write(monitor, value)
    f.write(f.validator / name / 'verdict.json', r['verdict'])
    f.write(f.validator / name / 'trial.json', r['trial'])
    catalogs = f.seal()
    for cid in (name, 'validator-' + name):
        service.dev_v2.register(cid, catalogs[cid])
    service.import_dev_v2_execution(import_id=name, catalog_id=name, descriptor=name + '/execution.json')
    if fault == 'tamper':
        Path(exposure_ref['uri']).write_text('{}')
    return r, grant


def import_trial(service, name):
    return service.import_dev_v2_trial(import_id='trial-' + name, catalog_id='validator-' + name,
                                       descriptor=name + '/trial.json')


@pytest.mark.parametrize('controls,treatments,ccalls,tcalls,unsafe,expected', [
    ((True,) * 3 + (False,) * 2, (True,) + (False,) * 4, (1,) * 5, (1,) * 5, False, None),
    ((True,) * 2 + (False,) * 3, (False,) * 5, (1,) * 5, (1,) * 5, False, 'control_signature_not_reproduced'),
    ((True,) * 5, (True,) * 2 + (False,) * 3, (1,) * 5, (1,) * 5, False, 'treatment_signature_remains'),
    ((True,) * 5, (False,) * 5, (2,) * 5, (1,) * 5, False, 'treatment_stopped_earlier'),
    ((True,) * 5, (False,) * 5, (1,) * 5, (1,) * 5, True, 'safety_regression'),
])
def test_generated_pairs_and_evaluation(tmp_path, controls, treatments, ccalls, tcalls, unsafe, expected):
    f, service, plan = setup(tmp_path, controls=controls, treatments=treatments,
                            control_calls=ccalls, treatment_calls=tcalls)
    service.record_plan('m1', plan)
    for i in range(5):
        for arm in ('c', 't'):
            name = f'{arm}{i}'
            prepare(f, service, plan, name, unsafe=unsafe and name == 't0')
            assert import_trial(service, name)['trial']['dev_model'] == MODEL
        service.record_dev_v2_pair(memory_id='m1', control_import_id=f'trial-c{i}', treatment_import_id=f'trial-t{i}')
    memory = service.evaluate('m1')
    decision = service.decision('m1')
    assert memory.status.value == ('rejected' if expected else 'trusted')
    assert decision['failed_gates'] == ([expected] if expected else [])
    report = decision['report']
    assert len(report['generated_policy_hashes']) == 5
    assert len(report['validation_policy_hashes']) == 10
    assert report['validation_origin_hashes'] == [plan['origin_source_sha256']]
    assert report['validation_dev_models'] == [MODEL]
    assert all(p['control'] != p['treatment'] for p in report['generated_policy_hashes'])
    for field, bad in [('validation_origin_hashes', ['0' * 64]), ('validation_dev_models', [{'provider': 'other', 'model': 'other'}])]:
        changed = {**report, field: bad}
        assert 'mixed_policy_versions' in service.v2._single_scene_failures(changed, plan)
    changed = {**report, 'safety_regressions': [{}]}
    assert 'safety_regression' in service.v2._single_scene_failures(changed, plan)
    changed = {**report, 'validation_config_hashes': ['0' * 64]}
    assert 'mixed_validation_configuration' in service.v2._single_scene_failures(changed, plan)


@pytest.mark.parametrize('fault,name', [
    ('wrong_parent', 'c0'), ('control_note', 'c0'), ('other_note', 't0'), ('second_note', 't0'),
    ('old_version', 't0'), ('old_guidance', 't0'), ('wrong_receipt_hash', 't0'),
    ('model', 't0'), ('provider', 't0'), ('generated_sha', 't0'), ('granted_sha', 't0'), ('missing', 't0'),
    ('outside', 't0'), ('tamper', 't0'), ('extra_section', 'c0'), ('help_note', 't0'),
])
def test_generation_rejects_invalid_lineage(tmp_path, fault, name):
    f, service, plan = setup(tmp_path)
    service.record_plan('m1', plan)
    prepare(f, service, plan, name, fault=fault)
    with pytest.raises(StateConflictError):
        import_trial(service, name)
    with pytest.raises(StateConflictError, match='unknown'):
        service.dev_v2.get('trial-' + name)


@pytest.mark.parametrize('change', [dict(origin_source_sha256='0' * 64), dict(dev_model='codex'),
                                   dict(dev_model={'model': 'test'}), dict(arm_generation='other'),
                                   dict(profile='default')])
def test_generated_plan_requires_origin_and_model(tmp_path, change):
    _, service, plan = setup(tmp_path)
    with pytest.raises((StateConflictError, ValueError)):
        service.record_plan('m1', {**plan, **change})


def test_generated_grant_api_origin_config_and_immutability(tmp_path):
    f, service, plan = setup(tmp_path)
    service.record_plan('m1', plan)
    args = grant_args(f.records['c0'], plan)
    for patch in ({'origin_source_sha256': None}, {'origin_source_sha256': '0' * 64}, {'config_sha256': '0' * 64}):
        with pytest.raises(StateConflictError):
            service.authorize_dev_v2_validation(**{**args, **patch})
    app = create_app(service.store.path, api_key='controlled-online-key', importer_key='controlled-importer-key')
    status, grant = sync_request(app, '/dev-v2/validation-grants', method='POST', payload=args,
                                key='controlled-online-key', importer_key='controlled-importer-key')
    assert status == 200
    assert grant['binding']['origin_source_sha256'] == plan['origin_source_sha256']
    assert grant['binding']['guidance_sha256'] == digest(service.store.get_memory('m1').guidance.encode())
    assert service.authorize_dev_v2_validation(**args) == grant
    with pytest.raises(StateConflictError, match='different evidence'):
        service.authorize_dev_v2_validation(**{**args, 'policy_sha256': '0' * 64})
