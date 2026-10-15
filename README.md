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

The exported `lifecycle.jsonl` includes event order, actor, reason, status,
memory version, and expiry time for each change.

Validation plans freeze at least five paired development-seed cases. Every
case explicitly names a realized scene, object configuration, camera view,
and same-task instruction variant; all four axes must vary. Both arms must
attest the same applied case, policy and execution configuration, native
outcome, and independent safety artifact. Promotion requires every predefined
case.
The validation policy code digest must be identical across all cases and
within every control/treatment pair. Its digest may differ from the failed
source policy (for example, after a repair); both digests and that distinction
are reported rather than silently attributing the policy change to memory.

The Service rejects a safety artifact whose `unsafe_attempts` disagrees with
its violation list, or which has no monitor events; a claimed source label
alone is not sufficient evidence. This is an integrity check on the recorded
monitor output, not cryptographic proof of an independent process. Supporting
successes, counterexamples, empirical confidence, and the conservative exact-match
retrieval scope are exported in the memory package. Any counterexample removes
its full variation tuple from trusted retrieval.
`empirical_confidence` is the observed treatment success fraction, not a
probability that the repair is correct. The report also includes a Wilson 95%
binomial reference interval with sample size; the frozen variation matrix is
not assumed IID, so this interval must not be presented as generalization
confidence or substituted for the promotion gate.
An intervention may qualify through a success gain or through fewer assistance
credits on pairs where both arms succeed, with no success regression. Credits
saved when both arms fail are reported separately but cannot justify promotion.
At least one supported, counterexample-free variation must remain retrievable;
either path still requires the treatment success and independent safety gates.
Safety is checked per pair: one treatment safety regression blocks promotion
even if a different pair makes the aggregate unsafe-attempt totals equal.

The v2 schema is additive to AttentionBench v1's SQLite trace/store schema.
The v1 files remain frozen in Universe for historical verification; this
repository is the source of truth for v2 Memory Service changes.

Candidate provenance retains the source run, attempt, advisor packet, raw
trace ID, response ID, the guidance actually received, response-content and
source policy-code SHA-256 digests, and raw evidence URIs with their SHA-256
digests. Request/response
lineage and response bytes are rechecked on provenance reads; older candidates
derive the missing links from the authoritative source records. A
trusted lookup requires both an exact validated variation tuple and the
concrete camera list and task prompt recorded in its frozen validation plan.
Use records are accepted only when the completed raw policy trace contains an
explicit trusted retrieval event and the reported outcome agrees with the
native evaluator. The event also binds the exact memory version returned to
the policy and a Service-issued, attempt-bound use grant; a use record with
another version, timestamp, attempt, scope context, or missing grant is
rejected. The grant context must agree with the raw attempt's suite, task,
perception mode, and recorded variation context.
Trusted uses that end in timeout or cancellation retain those outcomes in the
use ledger; such incomplete attempts are not eligible as paired promotion
evidence.
If an operator revokes a memory after a policy has already retrieved it, the
historical use and outcome are still recorded at the original retrieval time.
Revocation prevents any later retrieval; it does not erase evidence of use.
The Service signs off on a use grant while the memory is still trusted and
unexpired. The grant is bound to the policy attempt and checked against the
final raw retrieval event. A fabricated event with only a backdated timestamp
cannot substitute for a grant. This is a Service-side authorization record,
not cryptographic protection against an attacker with direct SQLite write access.
Remote runners require the `attempt_bound_use_grants` health capability and
reject an older daemon before resetting a task.
Promotion re-hashes the candidate's original artifact files; missing or
changed raw evidence blocks promotion.
Trusted retrieval also rechecks those source bytes before exposure.
Live retrieval uses at least the Memory Service's own wall clock, so a stale
or backdated client timestamp cannot revive an expired memory.


## Development-agent Memory exposure

`POST /dev-use-grants` is separate from attempt-bound `POST /use-grants`.
It accepts a trusted memory ID, exact validated context, graph-scoped
`development_id`, pre-edit policy source SHA-256, and time. The Service
rechecks retrieval and trusted/expiry state transactionally before returning
guidance. The immutable exposure records memory version, source trace, and raw
evidence references. Reusing the development ID with a different context or
source digest is rejected.

After a native run, `POST /dev-use-results` links the grant to the terminal
Attention store run, post-edit policy SHA-256, and persisted
`attention_run.json` bytes/digest. A Dev exposure is **not** a runtime
`MemoryUseRecord`, a treatment success, or evidence for promotion. The
`development_use_evidence` health capability lets a caller fail closed
before exposing guidance. This boundary assumes the Service can read the
shared Attention store and resulting artifact path; remote deployments must
mount these paths or add an independently verified artifact transport.

The isolated Dev v2 channel supports sealed executor/validator source imports,
original-provider-backed candidates, existing paired validation/promotion and
bundle-bound Dev grant results. See [the Dev v2 contract](docs/dev-v2-import-contract.md)
for startup catalogs, separate importer credentials, exact payloads, privacy
boundaries, main-client follow-up and isolated HTTP verification. This capability
alone is not evidence of a genuine provider/native integration run.
