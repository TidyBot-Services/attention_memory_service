# TidyBot Attention Memory Service

Independent, evidence-gated Memory Service for AttentionBench v2. This package
owns candidate creation, retrieval, validation evidence, promotion, and the
SQLite-backed per-memory artifact package. It does not import Tidybot-Universe,
ASPIRE, a simulator, or a model client.

Install and run with a dedicated Python environment:

```bash
pip install -e .
export ATTENTION_MEMORY_API_KEY='replace-with-a-long-random-secret'
python -m attention_memory_service --store-path /path/to/attention_memory.sqlite3
```

The daemon binds to loopback port 8768 by default. For remote operation, add
TLS and deployment controls; the client rejects non-HTTPS remote URLs. The
`/health` endpoint is unauthenticated; memory operations require the key.

The v2 schema is additive to AttentionBench v1's SQLite trace/store schema.
The v1 files remain frozen in Universe for historical verification; this
repository is the source of truth for v2 Memory Service changes.
