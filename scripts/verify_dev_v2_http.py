#!/usr/bin/env python3
"""Isolated real HTTP transport check; --controlled uses synthetic test inputs.

No provider calls, simulator jobs, robot actions, production authorities or keys.
The report labels controlled inputs explicitly; this is not real Memory/native
acceptance. Run from this checkout with the existing test environment.
"""
import argparse
import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention_memory_service import MemoryServiceClient
from attention_memory_service.dev_v2 import SCOPE, digest
from attention_memory_service.memory_service_api import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controlled', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = {'verification_class': 'controlled_test_http' if args.controlled else 'empty_authority_http',
              'real_provider_calls': 0, 'real_native_calls': 0, 'checks': [], 'completed': False}
    server = thread = sock = None
    code = 1
    try:
        # Fail before constructing any controlled trusted state if socket is denied.
        sock = socket.socket(); sock.bind(('127.0.0.1', 0)); sock.listen(128)
        port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory(prefix='dev-v2-http-test-') as temp:
            root = Path(temp); catalogs = {}; f = None; cases = []
            if args.controlled:
                from tests.dev_v2_fixtures import Fixture
                f = Fixture(root); failed = f.make('failed'); f.add_advice(failed)
                for i, seed in enumerate(range(101, 106)):
                    c = f.make('c' + str(seed), seed=seed, variant=i % 2)
                    f.make('t' + str(seed), seed=seed, variant=i % 2, success=True, exposure='candidate_validation', memory_id='http-test-memory')
                    cases.append({'seed': seed, **{k: c['context'][k] for k in ('scene_id', 'object_set_id', 'camera_config_id', 'task_variant_id', 'camera_names', 'task_prompt')}})
                catalogs = {k: v for k, v in f.seal().items() if k in {'failed', 'validator-failed'}}
            import uvicorn

            def restart(new_catalogs):
                nonlocal server, thread, sock
                if server:
                    server.should_exit = True; thread.join(10)
                    if thread.is_alive(): raise RuntimeError('isolated server did not stop before restart')
                    sock.close()
                    sock = socket.socket(); sock.bind(('127.0.0.1', 0)); sock.listen(128)
                app = create_app(root / 'authority.sqlite', api_key='controlled-online-key',
                                 operator_key='controlled-operator-key', importer_key='controlled-importer-key', dev_v2_catalogs=new_catalogs)
                server = uvicorn.Server(uvicorn.Config(app, log_level='error', lifespan='off'))
                thread = threading.Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True); thread.start()
                limit = time.monotonic() + 10
                while not server.started and thread.is_alive() and time.monotonic() < limit: time.sleep(.02)
                if not server.started: raise RuntimeError('isolated server did not start within 10 seconds')
                url = f'http://127.0.0.1:{sock.getsockname()[1]}'
                return (MemoryServiceClient(url, api_key='controlled-online-key', timeout=10),
                        MemoryServiceClient(url, api_key='controlled-online-key', importer_key='controlled-importer-key', timeout=10),
                        MemoryServiceClient(url, api_key='controlled-online-key', operator_key='controlled-operator-key', timeout=10))

            online, offline, operator = restart(catalogs)
            url = online.base_url
            assert online.health()['status'] == 'ok'; report['checks'].append('health')
            wrong = MemoryServiceClient(url, api_key='wrong-controlled-key', timeout=5)
            try: wrong.store_id()
            except RuntimeError as exc: assert '401' in str(exc)
            else: raise AssertionError('unauthenticated identity allowed')
            report['checks'].append('authentication')
            assert online.store_id() == online.store_id(); report['checks'].append('identity')
            ctx = f.records['failed']['context'] if f else {'suite': 'robocasa365', 'task_id': 'pick', 'perception_mode': 'sim_gt'}
            assert online.retrieve(ctx, now=time.time()) == []; report['checks'].append('empty_retrieval')
            if f:
                try: online._call('POST', '/dev-v2/executions', {'import_id': 'failed', 'catalog_id': 'failed', 'descriptor': 'failed/execution.json'})
                except PermissionError: pass
                else: raise AssertionError('online key imported native source')
                report['checks'].append('importer_capability')
                offline.import_dev_v2_execution(import_id='failed', catalog_id='failed', descriptor='failed/execution.json')
                offline.import_dev_v2_source(import_id='source', catalog_id='failed', descriptor='failed/source.json')
                offline.create_dev_v2_candidate(memory_id='http-test-memory', source_import_id='source', created_at=1.0)
                assert online.retrieve(ctx, now=time.time()) == []; report['checks'].append('candidate_not_online')
                offline.record_plan('http-test-memory', {'schema_version': 'attentionbench.memory-validation-plan.v2', 'memory_id': 'http-test-memory', **{k: ctx[k] for k in SCOPE if k not in ('scene_id', 'object_set_id', 'camera_config_id', 'task_variant_id')}, 'policy_id': 'full_trace_aware_attention_planner', 'seeds': list(range(101, 106)), 'cases': cases, 'assistance_credits': 1})
                for seed in range(101, 106):
                    for arm in ('c', 't'):
                        name = arm + str(seed); r = f.records[name]
                        validation_grant = offline.authorize_dev_v2_validation(memory_id='http-test-memory', cell=name, version=1, seed=seed, context=r['context'], policy_sha256=r['record']['source']['sha256'], config_sha256=r['descriptor']['config']['sha256'], exposure=r['trial']['exposure'])
                        f.bind_validation(r, validation_grant)
                catalogs = {k: v for k, v in f.seal().items() if k not in {'failed', 'validator-failed'}}
                online, offline, operator = restart(catalogs)
                report['checks'].append('original_validation_dispatch_grants')
                for seed in range(101, 106):
                    for arm in ('c', 't'):
                        name = arm + str(seed)
                        offline.import_dev_v2_execution(import_id=name, catalog_id=name, descriptor=name + '/execution.json')
                        offline.import_dev_v2_trial(import_id='trial-' + name, catalog_id='validator-' + name, descriptor=name + '/trial.json')
                    offline.record_dev_v2_pair(memory_id='http-test-memory', control_import_id='trial-c' + str(seed), treatment_import_id='trial-t' + str(seed))
                trusted = offline.promote('http-test-memory'); report['checks'].append('controlled_original_promotion_gates')
                # Only the operator sets lifecycle deadlines. The ordinary online
                # key must not advance them via retrieve or a nested grant call.
                wall = time.time()
                expiry_payload = {'expires_at': wall + 3600, 'actor': 'controlled_test_operator',
                                  'reason': 'controlled_test server clock boundary'}
                try: online._call('PUT', '/memories/http-test-memory/expiry', expiry_payload)
                except PermissionError: pass
                else: raise AssertionError('online key changed operator expiry')
                operator.set_expiry('http-test-memory', **expiry_payload)
                expected = offline.get_memory('http-test-memory')
                assert online.retrieve({**ctx, 'namespace': 'controlled_test_other'}, now=wall + 7200) == []
                assert offline.get_memory('http-test-memory') == expected
                assert [m.memory_id for m in online.retrieve(ctx, now=wall + 7200)] == ['http-test-memory']
                assert offline.get_memory('http-test-memory') == expected
                report['checks'].append('online_future_time_cannot_expire_other_namespace')
                report['checks'].append('operator_expiry_role_separation')
                assert [m.memory_id for m in online.retrieve(ctx, now=time.time())] == ['http-test-memory']
                raw = online._call('POST', '/retrieve', {'context': ctx, 'now': time.time()})
                assert not {'validation_successes', 'validation_failures', 'confidence', 'native_success'} & raw[0].keys()
                try: online.impact_report('http-test-memory')
                except PermissionError: pass
                else: raise AssertionError('private evaluation escaped to online key')
                report['checks'].append('private_isolation')
                grant = online.authorize_dev_use(memory_id='http-test-memory', memory_version=trusted.version, development_id='result:v1', context=ctx, source_policy_sha256=digest(b'controlled_test previous code\n'), now=wall + 7200)
                assert online.get_dev_use(grant['grant_id'])['memory_version'] == trusted.version
                assert offline.get_memory('http-test-memory') == expected
                report['checks'].append('nested_grant_uses_server_clock')
                assert online.retrieve({**ctx, 'namespace': 'wrong'}, now=time.time()) == []; report['checks'].append('exact_scope_and_grant')
                # Real HTTP restart with an appended, operator-owned sealed list;
                # there is intentionally no online root-registration endpoint.
                f_result = f.make('result')
                f.add_result_model(f_result, grant, online.store_id())
                result_catalog = f.seal()['result']
                online, offline, operator = restart({'result': result_catalog})
                offline.import_dev_v2_execution(import_id='result', catalog_id='result', descriptor='result/execution.json')
                saved = offline.record_dev_v2_result(grant_id=grant['grant_id'], execution_import_id='result')
                assert saved == offline.record_dev_v2_result(grant_id=grant['grant_id'], execution_import_id='result')
                public_result = online.get_dev_use(grant['grant_id'])['result']
                assert public_result['claim'] == 'development_exposure_only_not_causal_impact' and 'bundle' not in public_result
                report['checks'].append('bundle_result_idempotence_and_restart')
                operator.disable('http-test-memory', actor='controlled_test_operator', reason='HTTP controlled test')
                try: online.get_dev_use(grant['grant_id'])
                except Exception as exc:
                    from attention_memory_service.core.store import StateConflictError
                    assert isinstance(exc, StateConflictError)
                else: raise AssertionError('revoked grant remained readable')
                report['checks'].append('immediate_revocation')
            server.should_exit = True; thread.join(10)
            if thread.is_alive(): raise RuntimeError('isolated server cleanup timed out')
            report['checks'].append('normal_cleanup')
            report['completed'] = True; code = 0
    except Exception as exc:
        report['error'] = {'type': type(exc).__name__, 'errno': getattr(exc, 'errno', None), 'message': str(exc)}
        code = 2 if isinstance(exc, PermissionError) else 1
    finally:
        if server: server.should_exit = True
        if thread and thread.is_alive(): thread.join(10)
        if sock: sock.close()
        report['owned_server_stopped'] = thread is None or not thread.is_alive()
    data = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    if args.report: args.report.write_text(data)
    print(data, end='')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
