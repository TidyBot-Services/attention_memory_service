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

### Validation profiles

Plans without `profile` use the unchanged default rule: five **different**
development seeds, real variation on all four scene/object/camera/task axes,
no pairwise or net safety regression, no success regression, a measured success
or successful-pair assistance gain, at least three treatment successes, a
treatment success fraction of at least 0.75, and a supported scope without a
counterexample. These memories support the existing multi-scene report.

`single_scene_v1` is a separate Dev v2 rule. A source descriptor may declare
`failure_signature: {"public_error_class": "RuntimeError: TimeoutError",
"operation": "left_arm.move_delta"}`. Source import checks that this signature
appears in the originating failed attempt's public feedback, then seals it in
the candidate provenance and artifact. It cannot be changed by a plan or later
trial. The optional objective Eval reason class is not used by this version.

The plan declares `profile: "single_scene_v1"`, the exact source `failure_signature`,
one source seed and case, at least five distinct `pairs` with `pair_id`,
`control_cell`, and `treatment_cell`, plus `validation_policy_sha256` and
`validation_config_sha256`. It retains the normal scope, policy ID and credits
fields. Every arm needs a preregistered grant, a fresh isolated process, the
verified scene contract, the same policy and config SHA, and independent
validator evidence. Only treatment gets the candidate. For this profile, each
independent verdict also supplies `failure_signature`, `signature_present`, and
`stop_position`; the service recomputes both values from that arm's sealed
public trace and rejects a disagreement.

For five pairs, promotion requires the signature in at least four control arms
and at most one treatment arm. With more pairs the limits are
`ceil(0.8 * pairs)` and `floor(0.2 * pairs)`. No treatment may have more unsafe
attempts than its paired control, and the treatment total may not exceed the
control total. `stop_position` is the number of successful (`completed` or
`succeeded`) SDK events in the complete public trace before termination; each
treatment count must be at least its paired control count. This pairwise rule
prevents a longer run elsewhere from hiding an early stop. `evaluate` records
the profile, per-arm signature and stop counts, total counts, failed gates and
the full impact report; a failed completed plan is `rejected`.

A passing memory has `trust_scope: "scene_scoped"` (display name
`scene_scoped_trusted`). Retrieval requires the exact task, seed, source scene
and object contract, robot/controller, SDK/namespace, camera configuration and
names, task variant and prompt, and perception mode. Revocation, expiry and
rollback work as for other trusted memories. This rule only supports the claim
that the targeted failure type disappeared in that exact scene. It does not
show task outcome improvement or cross-scene generalization; exclude it from
cross-scene aggregate claims. The decision and impact report mark
`cross_scene_eligible: false` for downstream filtering.

The read-only `GET /dev-v2/memories` and `GET /dev-v2/memories/{memory_id}`
accept the online service key. Their **public allowlist** is memory ID, optional
`title` (or `null` for existing/untitled notes),
version, status, trust scope/display name, profile, the applicability summary
(benchmark, suite, task, seed, scene/object/camera/task IDs, robot, controller,
perception mode, camera names, task prompt, SDK SHA, namespace and planned
case summaries), failure
signature class/operation, validation summary (planned/completed pair counts,
decision, failed gate codes and signature counts), and source label. Guidance,
repair text, grant bodies, import descriptors, sealed evidence, and keys remain
behind importer authorization. The existing generic `/memories` list continues
to omit Dev v2 rows; the UI must call these new read-only routes to show them.
The UI can display `title` as a short, human-readable heading and keep the
technical `memory_id` in the detail view. The title field does not copy
guidance, repair, grants, import descriptors, private evidence, or keys. It is
only read from an explicitly supplied field; guidance is never used to generate one.

An optional `title` can be supplied in the test-seeded JSON body or in the
sealed source descriptor imported by `/dev-v2/sources` before candidate creation.
It is trimmed and must be 1–80 characters of plain text, with no newline,
control character, URL, path, key-like text, or native evaluator/oracle term.
The normalized title is stored in provenance. Invalid titles are rejected with
an explicit error. Omitting it leaves the public title `null`; existing notes
are not assigned one retroactively.

### Importer-only test-seeded notes

`POST /dev-v2/test-seeded-notes` exists to exercise a real Dev Memory read,
disable, expiry, and rollback cycle before a candidate has passed validation.
It requires both the service API key and the distinct offline importer key;
the online key, even paired with the operator key, receives 403. This is **test-only**
guidance, never a validation path or evidence of Memory effectiveness.

The JSON body requires `actor`, `reason`, `memory_id`, `context`, and
`guidance`; `repair` is optional. `context` contains the exact Dev v2 scope
(`benchmark`, `suite`, `task_id`, `robot`, `controller`, `perception_mode`,
`scene_id`, `object_set_id`, `camera_config_id`, `task_variant_id`,
`sdk_sha256`, `namespace`) plus integer `seed`, ordered `camera_names`,
`task_prompt`, and `scene_contract_ref: {"uri": "...", "sha256": "..."}`.
Every field must match at retrieval, including the scene contract reference.
The response contains `memory_id`, `version`, `status`,
`source_kind: "seeded_for_tests"`, `source_label: "seeded for tests"`,
`trust_scope: "test_seeded"`, `seeded_for_tests: true`, and
`excluded_from_results: true`. Repeating the same `memory_id` and note content,
including its normalized title, is idempotent; changing the title or other
content is a conflict. The first write records
actor, reason, and server time in the audit event.

These notes have a retrievable `trusted` lifecycle state but are **not**
`scene_scoped_trusted` or multi-scene trusted. They have no validation plan,
pairs, decision, or impact report. Promotion and evaluation reject them;
default and `single_scene_v1` gates remain unchanged. Retrieval, use grants,
Dev exposure grants, and use records carry the exclusion flags. Downstream
run and report code must exclude every entry with `excluded_from_results=true`
from evaluation, success, and effect statistics. Dev result recording also
rejects these grants. The public Dev v2 catalogue shows the source label,
scope and exclusion flag but only guidance length and SHA-256, never guidance
or repair text. Operator disable/enable, expiry and rollback follow the normal
lifecycle; expired and rolled-back notes cannot be enabled again.

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
