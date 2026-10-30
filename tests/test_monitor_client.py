"""Public monitor reads use the ordinary client and service response envelopes."""
import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from attention_memory_service import MemoryServiceClient, memory_service_client


def test_monitor_feeds_with_api_key_only(monkeypatch):
    feeds = json.loads((Path(__file__).parent / 'fixtures/monitor_feeds.json').read_text())
    memory_id = feeds['cross_seed_status']['memory_id']
    calls = []

    def transport(request, *, timeout):
        assert request.method == 'GET' and request.data is None
        assert dict(request.header_items()) == {
            'X-memory-service-key': 'synthetic-read-key', 'Content-type': 'application/json',
        }
        assert timeout == 30.0
        url = urlsplit(request.full_url)
        calls.append(url.path)
        if url.path == '/dev-v2/cross-seed/status':
            assert parse_qs(url.query) == {'memory_id': [memory_id]}
            assert url.query == 'memory_id=note+%2F%2B%26%3F%E4%B8%AD%E6%96%87'
            value = feeds['cross_seed_status']
        else:
            assert not url.query
            value = feeds[{'/dev-v2/family': 'family', '/dev-v2/cross-seed/queue': 'cross_seed_queue'}[url.path]]
        return io.BytesIO(json.dumps(value).encode())

    monkeypatch.setattr(memory_service_client, 'urlopen', transport)
    client = MemoryServiceClient('http://localhost:8878', api_key='synthetic-read-key')
    assert client.family_list() == feeds['family']
    assert client.cross_seed_queue() == feeds['cross_seed_queue']['queue']
    assert client.cross_seed_status(memory_id=memory_id) == feeds['cross_seed_status']
    assert calls == ['/dev-v2/family', '/dev-v2/cross-seed/queue', '/dev-v2/cross-seed/status']
