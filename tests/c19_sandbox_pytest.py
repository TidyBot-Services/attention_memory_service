"""Opt-in sandbox ASGI adapter, matching the main tree's C12 test adapter."""
from unittest.mock import patch

import pytest


@pytest.fixture(scope='session',autouse=True)
def synchronous_anyio_handlers():
    async def run_sync(func,*args,**kwargs):
        return func(*args)
    with patch('anyio.to_thread.run_sync',run_sync):
        yield
