# TidyBot Attention Memory Service

Independent, evidence-gated Memory Service for AttentionBench v2. This package
owns candidate creation, retrieval, validation evidence, promotion, and the
SQLite-backed per-memory artifact package. It does not import Tidybot-Universe,
ASPIRE, a simulator, or a model client.

Install and run with a dedicated Python environment:

```bash
pip install -e .
export ATTENTION_MEMORY_API_KEY='replace-with-a-long-random-secret'
export ATTENTION_MEMORY_OPERATOR_KEY='replace-with-a-different-long-random-secret'
python -m attention_memory_service --store-path /path/to/attention_memory.sqlite3
```

The daemon binds to loopback port 8768 by default. For remote operation, add
TLS and deployment controls; the client rejects non-HTTPS remote URLs. The
`/health` endpoint is unauthenticated; memory operations require the key.
The separate operator key is required for human `disable`, `rollback`, and
`set-expiry` actions. Operator calls require a nonempty actor identifier and
reason, which are retained in the lifecycle audit. For example:

```bash
python -m attention_memory_service.operator_cli --actor operator-1 \
  disable MEMORY_ID --reason 'unsafe in camera variant B'
```

Validation plans freeze at least five paired development-seed cases. Every
case explicitly names a realized scene, object configuration, camera view,
and same-task instruction variant; all four axes must vary. Both arms must
attest the same applied case,
policy and execution configuration, native outcome, and independent safety
artifact. Promotion requires every predefined case. Supporting successes,
counterexamples, empirical confidence, and the conservative exact-match
retrieval scope are exported in the memory package. Any counterexample removes
its full variation tuple from trusted retrieval.
An intervention may qualify through a success gain or through fewer assistance
credits with no success regression; either path still requires the treatment
success and independent safety gates.

The v2 schema is additive to AttentionBench v1's SQLite trace/store schema.
The v1 files remain frozen in Universe for historical verification; this
repository is the source of truth for v2 Memory Service changes.
