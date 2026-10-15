"""Legacy assertion replay through sync handlers/dependencies, NOT HTTP/ASGI.

This avoids the sandbox's anyio event-loop wakeup hang. Original ASGI tests
remain intact for a supervisor host. HTTP validation uses verify_dev_v2_http.py.
"""
import inspect

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from starlette.routing import Match

from tests import test_dev_use, test_service


def sync_request(app, path, *, method='GET', payload=None, key=None, importer_key=None, operator_key=None):
    from urllib.parse import unquote
    path = unquote(path)
    headers = {'x-memory-service-key': key, 'x-memory-importer-key': importer_key, 'x-memory-operator-key': operator_key}
    scope = {'type': 'http', 'method': method, 'path': path, 'root_path': '', 'headers': []}
    for route in app.routes:
        matched, child = route.matches(scope)
        if matched is not Match.FULL:
            continue
        scope.update(child)

        def invoke(dependant):
            kwargs = {}
            for field in dependant.header_params:
                kwargs[field.name] = headers.get(field.alias.lower())
            for field in dependant.path_params:
                kwargs[field.name] = scope['path_params'][field.alias]
            for field in dependant.body_params:
                annotation = inspect.signature(dependant.call).parameters[field.name].annotation
                annotation = inspect.get_annotations(dependant.call, eval_str=True)[field.name]
                kwargs[field.name] = annotation.model_validate(payload) if hasattr(annotation, 'model_validate') else payload
            if dependant.request_param_name:
                kwargs[dependant.request_param_name] = Request(scope)
            for dependency in dependant.dependencies:
                result = invoke(dependency)
                if dependency.name:
                    kwargs[dependency.name] = result
            return dependant.call(**kwargs)
        try:
            return 200, invoke(route.dependant)
        except HTTPException as exc:
            return exc.status_code, {'detail': exc.detail}
    return 404, {'detail': 'not found'}


@pytest.mark.parametrize('name', ['test_dev_result_binds_real_native_summary_and_is_immutable',
                                 'test_dev_use_grant_http_requires_auth_and_returns_evidence',
                                 'test_standalone_service_starts_with_empty_store_and_auth'])
def test_legacy_assertions_via_synchronous_handlers(tmp_path, monkeypatch, name):
    def request(app, path, *, key=None):
        return sync_request(app, path, key=key)

    def post(app, path, payload, *, key=None):
        return sync_request(app, path, method='POST', payload=payload, key=key)

    monkeypatch.setattr(test_dev_use, '_request', request)
    monkeypatch.setattr(test_dev_use, '_post', post)
    monkeypatch.setattr(test_service, '_request', request)
    function = getattr(test_dev_use, name, None) or getattr(test_service, name)
    if 'monkeypatch' in inspect.signature(function).parameters:
        function(tmp_path, monkeypatch)
    else:
        function(tmp_path)
