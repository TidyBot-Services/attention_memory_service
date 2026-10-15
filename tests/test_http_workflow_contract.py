"""Replay the supervisor script via sync handlers. This test uses NO HTTP/socket.

The real transport script is separately runnable on the supervisor host. This
unit test checks its payload/workflow compatibility without pretending transport.
"""
import json
import time

from attention_memory_service.core.store import StateConflictError
from attention_memory_service.memory_service_client import MemoryServiceClient
from scripts import verify_dev_v2_http
from tests.test_sync_legacy_routes import sync_request


def test_supervisor_workflow_via_synchronous_handlers(tmp_path, monkeypatch, capsys):
    import uvicorn
    active = {}

    class ControlledSocket:
        def bind(self, address): pass
        def listen(self, n): pass
        def getsockname(self): return ('127.0.0.1', 19876)
        def close(self): pass

    class ControlledServer:
        def __init__(self, config):
            self.started = False; self.should_exit = False
            active['app'] = config.app
        def run(self, sockets):
            self.started = True
            while not self.should_exit: time.sleep(.001)

    def call(self, method, path, payload=None, *, operator=False, importer=False):
        if importer and not self.importer_key:
            raise PermissionError('offline importer key is required')
        status, value = sync_request(active['app'], path, method=method, payload=payload,
                                     key=self.api_key, importer_key=self.importer_key,
                                     operator_key=self.operator_key if operator else None)
        if status == 403: raise PermissionError(value['detail'])
        if status == 409: raise StateConflictError(value['detail'])
        if status != 200: raise RuntimeError(f'Memory Service HTTP {status}: {value}')
        return value

    monkeypatch.setattr(verify_dev_v2_http.socket, 'socket', ControlledSocket)
    monkeypatch.setattr(uvicorn, 'Server', ControlledServer)
    monkeypatch.setattr(MemoryServiceClient, '_call', call)
    report = tmp_path / 'synthetic-workflow-no-transport.json'
    monkeypatch.setattr('sys.argv', ['verify_dev_v2_http.py', '--controlled', '--report', str(report)])
    assert verify_dev_v2_http.main() == 0
    value = json.loads(report.read_text())
    assert value['completed'] and value['owned_server_stopped']
    assert 'bundle_result_idempotence_and_restart' in value['checks']
    assert {'online_future_time_cannot_expire_other_namespace',
            'operator_expiry_role_separation',
            'nested_grant_uses_server_clock'} <= set(value['checks'])
    assert value['real_native_calls'] == value['real_provider_calls'] == 0
    capsys.readouterr()
