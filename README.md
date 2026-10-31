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

An opt-in plan mode, `arm_generation: "dev_from_origin"`, instead freezes
`origin_source_sha256` (the note's sealed origin source), `validation_config_sha256`
and `dev_model: {"provider": "codex", "model": "<model id>"}`. Each arm has Dev
generate its own next policy from that origin. The grant binds that arm's generated
SHA plus the origin, model and current candidate guidance hash/version. Control's
Dev prompt contains no notes; treatment contains only the exact candidate guidance.
The service checks the executor's sealed parent, model request/response, settled
job, generated source and `attentionbench.dev-memory-exposure.v1` receipt against
the grant and executed bundle. The receipt schema and required version-record
fields are in the [import contract](docs/dev-v2-import-contract.md#dev-generated-arms-from-a-frozen-origin).
Reports list per-pair control/treatment SHAs in `generated_policy_hashes`;
`mixed_policy_versions` in this mode means differing origins or Dev models.
Config, safety, signature and stop-position gates stay unchanged. Plans without
`arm_generation` retain the frozen executed-policy behavior above.

For five pairs, promotion requires the signature in at least three control arms
and at most one treatment arm. With more pairs the limits are
`ceil(0.6 * pairs)` and `floor(0.2 * pairs)`. No treatment may have more unsafe
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


### Cross-seed validation and task trust

`cross_seed_v1` attaches checks to a current, scene-trusted Dev v2 note whose
origin plan is `single_scene_v1`. Candidate seeds come from the worker; none are
hard-coded. The service refuses the origin seed, another task/scope, a changed
SDK interface, and seeds without an exact registered **contract** catalog.
Each target case carries its own `scene_contract_ref`, pointing to a v2 contract
that contains that seed's normalized fingerprint. Original scene matching and
existing plans are unchanged; the target contract never replaces the origin's.
The existing exact executor scope checks, including SDK build identity, remain.

All four writes below require the online **and importer** keys, as `/dev-v2/pairs`
does. Both reads require the online key only.

| Route | Input / result |
|---|---|
| `POST /dev-v2/cross-seed/probes` | `memory_id`, `target_seed`, `execution_import_id`, validator `catalog_id`, `descriptor`, `scene_contract_ref`; returns the service-computed role |
| `POST /dev-v2/cross-seed/plans` | `memory_id`, `target_seed`, `pairs` (`pair_id`, `control_cell`, `treatment_cell`), `dev_model` (`provider`, `model`); freezes the plan from the probe |
| `POST /dev-v2/cross-seed/pairs` | `memory_id`, `target_seed`, `control_import_id`, `treatment_import_id`; closes the scene when all prescribed pairs arrive |
| `POST /dev-v2/cross-seed/failed-checks` | `memory_id`, `target_seed`, `validation_grant_id`, `execution_import_id`, validator `catalog_id`, `descriptor`; closes an independently failed check |
| `GET /dev-v2/cross-seed/status?memory_id=...` | Per-seed role, pairs done/needed, verdict and plain reason; `task_trust.met`, `missing`, scene count and failed transfer seeds |
| `GET /dev-v2/cross-seed/queue` | `queue`: scene-trusted notes with an open need, ordered by the M-8 failure-gone/use-count rank |

First import a sealed direct execution of the note's frozen origin source with
no notes and no model/advice jobs. Its sealed `validation_input` is the normal
`none` input (empty `memory_ids`, context, policy/config hashes), without a grant.
Import its independent validator descriptor through `probes`. The service derives
the signature from the public trace: present means **transfer**, absent means
**neutral**. Failed scene reproduction or probe safety means **could_not_finish**;
that seed is closed and does not count as a tested scene. Client-supplied roles
are refused. Replay twice, independent roots, native evaluation, permissions,
original source identity, and all catalog/digest gates remain required.

The service freezes exactly **five** pairs for transfer or **three** for neutral.
All pairs use `dev_from_origin`: the worker supplies cells and Dev model, while
the service binds the origin SHA, target config, signature, case and pair count.
For each arm, use the existing `POST /dev-v2/validation-grants` with the additional
`target_seed` field, then the existing execution/trial import routes. Grants bind
the current memory version and exact plan. M-8's original Dev request, output,
lineage, exposure receipt, native consumption and dispatch checks enforce empty
Memory for control and only this note for treatment.

Transfer uses the unchanged `_single_scene_failures` rule (control >= 3/5,
treatment <= 1/5, no safety regression, no earlier stop). Neutral checks only
pairwise/net safety and earlier stopping. Verdicts are `transfer_passed`,
`transfer_failed`, `neutral_no_harm`, `neutral_harm`, or `could_not_finish`.
The failed-check exit preserves M-8's distinction: treatment safety failure is
harm/failure; control safety or replay failure is `could_not_finish`.

Every scene verdict evaluates task trust in the same transaction. It requires
current origin-scene trust, at least one other `transfer_passed`, no
`neutral_harm`, and at least three verdict-bearing scenes including the origin.
`transfer_failed` counts and is listed in the decision, but does not itself block
promotion. Evidence from an older memory version cannot satisfy a new crossing;
closed seeds remain closed. A known neutral harm remains a blocker. Promotion
rechecks participating sealed evidence, updates the decision to
`trust_scope: "task_scoped"`, and emits one `promoted` lifecycle event. The original
scene report and `cross_scene_eligible: false` are retained; this does not loosen
the default multi-scene profile or admit a note to the frozen matrix.

A task-scoped note still matches `SAME_TASK` on another seed, but `trusted_here`
becomes true and it joins the trusted block. Origin labels, source-evidence checks,
legacy/test-seeded notes, and family references keep their existing behavior.
Task trust does not change the note version or guidance. Each demotion increments
the version, so old use records cannot cause a second drop: at least three target
failures in the last five non-origin uses demote task → scene. The origin-use
window can also demote task → scene; a later M-8 crossing demotes scene →
provisional. Each event records its triggering use IDs, and retries are idempotent.

The additive tables are `memory_cross_seed_probes`, `memory_cross_seed_plans`,
`memory_cross_seed_pairs`, and `memory_cross_seed_verdicts`. These retain sealed
provenance separately from origin plans and evidence; service restart preserves
all checks and decisions. No feasible target seeds are assumed by this feature.

### Dev listing evidence diagnostics

`GET /dev-v2/memories` and `POST /dev-v2/memories/query` return
`{"memories": [...], "unavailable": [...]}`. Each unavailable entry contains
`memory_id`, `error_class`, and `reason` (the verification error message). A row
whose evidence raises `StateConflictError`, or whose pinned file disappears
(`FileNotFoundError`), is omitted entirely and logged as a warning. Healthy rows
retain their public fields and order. The listing allowlist is unchanged (Dev agent
notes stay out of this projection and are delivered only through retrieval grants).
Each row has a separate evidence verification scope, including context matching.
Invalid query contexts and programming errors still fail the request.

In-process `MemoryService.list_dev_v2_memories` keeps returning a list; callers
that need diagnostics can use `MemoryService.dev_v2_memory_listing`. Diagnostics
are local to each response. Existing clients reading the `memories` key work
unchanged. Retrieval, authorization, trusted use, provisional outcomes and
validation remain strict. A frozen-harness follow-up can display `unavailable`
in its client/UI; consuming that field is optional and needs no service change.

### Automatic validation service hooks

Evidence catalogs can be registered at startup, plus the importer-only registration
endpoint restricted to configured roots. Set `--dev-v2-registration-roots /abs/roots.json`
to a JSON object such as
`{"executor":["/abs/experiments"],"validator":["/abs/validator"],"contract":["/abs/contracts"],"source":["/abs/original-sources"]}`.
The same object is accepted as `dev_v2_registration_roots` by `create_app` and
`MemoryService`. Roots must be canonical existing absolute directories and disjoint
across roles, including between `source` and `contract`. The `contract` and `source`
lists are optional (omitting one permits no online registration for that role).
Registration is disabled (403) when the whole option is omitted.

| Route | Authorization | Result |
| --- | --- | --- |
| `POST /dev-v2/catalogs` | Service key and offline importer key | Accepts `{catalog_id, catalog}` using the startup catalog shape. Root must be equal to or inside a configured root for its role. |
| `POST /dev-v2/failed-checks` | Service key and offline importer key | Records a sealed failed independent check and closes a single-scene batch without accepting a trial. |
| `GET /dev-v2/validation-queue` | Service key | `{items: [...]}` with memory ID, version, title, uses, failure_gone, created_at, source_summary and layer. |
| `GET /dev-v2/memory-events?after=0` | Service key | `{events: [...]}` ordered by `seq`, with memory_id, title, event, plain reason and server time. Use the last sequence as the next cursor. |

Both registration paths verify every sealed file's existence and SHA-256 immediately,
and later evidence reads still recheck bytes. Existing path and pinned-symlink rules
remain enforced. Identical registration is idempotent; conflicting IDs return 409.
Online registration writes an importer audit event with the catalog ID, role, root
and server timestamp. The operator key does not authorize registration.

The `contract` catalog role is supported at startup and through the same endpoint.
It uses the same context, sealed file list and path rules, with a root disjoint from
executor and validator catalogs. A config's `scene_contract` reference must resolve
inside its executor catalog or a registered `contract` or `source` catalog at the
exact sealed digest. A contract inside the SDK source directory can share its
source catalog's sealed list; no overlapping contract catalog is needed or allowed.
Unregistered files and validator-only contract files are refused.

The `source` role separately pins original SDK/model files outside the experiment
root. Identity locks keep their original paths: each locked file and complete SDK
source set must be in the executor catalog or a registered source catalog, with
exact digests and no extra or missing implementation files. Files in a contract or
validator catalog cannot substitute for source evidence. All roles retain the same
path, symlink and digest checks.

Source catalogs can explicitly pin cross-root symlink chains with `links`:
`{"python": {"target": "/ABS/other/bin/python", "target_catalog": "other-source", "target_name": "python"}}`.
Each hop must match the actual one-hop link target and an exact sealed name in a
source catalog, with the same digest through to the final regular file. Register
target catalogs first (a link may also target a name in its own catalog). Chains
are limited to eight hops; cycles, implicit hops and directory symlinks are refused.
Registration and subsequent reads check the pins and bytes. Each source root still
needs its configured allowlist entry; role overlap rules are unchanged. See the
[import contract](docs/dev-v2-import-contract.md) for the complete declaration.

Scene contract v1 retains its 1e-6 comparator and quaternion-sign equivalence.
V2 (`attentionbench.public-scene-contract.v2`) requires an integer config seed and
its `tasks[task_id].seeds[str(seed)]` entry. The service recomputes raw and normalized
SHA-256 from each sealed full state; normalization rounds finite numbers to six
decimal places using Python `round`, then canonical JSON (sorted keys, compact
separators, ASCII escaping). Booleans are refused. Each original/replay receipt's
hashes and `matched` must agree with those computations; matching uses only the
seed entry's normalized hash. Both independent replays must match for validation.
V2 replay artifacts contain `{state, validation}` like the original scene receipt;
v1 replay artifacts retain their bare state shape. See the
[import contract](docs/dev-v2-import-contract.md) and literal cross-repo fingerprint
vectors in `tests/test_scene_contract.py`.

The queue ranks recorded observational outcomes by failure_gone descending, uses
descending, then oldest created_at (memory ID breaks ties). Unused, test-seeded,
non-provisional, expired, disabled and withdrawn notes are excluded, as are notes
already linked to a validation candidate or with a current plan/decision. These
counts choose work; they never promote a note. The worker still imports a separate
sealed-source candidate with `from_provisional` and freezes its validation plan.

A failed independent check uses `record_dev_v2_failed_check` with `memory_id`,
`validation_grant_id`, `execution_import_id`, `catalog_id` and `descriptor` (the
original independent verdict in the validator catalog). All native, permission,
source-identity, binding and digest checks still apply. If only treatment safety
fails, the candidate is immediately rejected with `safety_regression`, even with
missing pairs. Any scene-reproduction failure, or control safety failure, closes
the batch as `inconclusive`: the candidate stays unpromoted and unrejected, and its
linked provisional note remains available. Both outcomes appear in memory-events.
An inconclusive source re-enters the queue only after a new use is exposed and its
outcome recorded after the decision; a delayed report of an earlier use does not
qualify. A fresh candidate/plan handles the next check. Closed candidates cannot
receive new grants, trials or pairs, or be promoted. Failed arms never count as
accepted trials or pairs. Identical retries are idempotent; conflicting records
return 409. All-true checks must use normal trial import.

Completed `single_scene_v1` validation emits `promoted` or `rejected` notices with
plain reasons. Controls must have grant exposure `none`; treatments must have
`candidate_validation` for exactly the memory under test, including the control
grant's memory binding. Thresholds and all safety, permission, scene and digest
gates remain unchanged.

When a checked candidate has `from_provisional`, the candidate status, decision,
source-note shutdown and lifecycle audit commit in one transaction. A passing
candidate turns the source note Off with reason
`replaced by its checked version <candidate id>` and a `superseded` notice; a
rejected candidate turns it Off with reason `rejected after automatic check`
and a `rejected` notice for the source too. The source remains in listings with
`decision_link` (candidate memory ID, decision, plain reason and decision time).
It is excluded from retrieval at every level and from the validation queue.
Existing operator disable, expiry or withdrawal is preserved. Operators can
explicitly turn a disabled source back on as provisional, with the usual audit
and expiry checks; its decision link and queue exclusion remain. Decision retries
and later demotion of the candidate never revive or re-disable the source.

The importer-only `POST /dev-v2/provisional-outcomes` also accepts observational
outcomes for scene-trusted Dev grants. Sealed `/dev-v2/results` records feed this
same outcome path automatically. When three or more of the last five recorded
uses of the current memory version show the target signature, the service
atomically demotes the note to provisional and emits one `demoted` event with
reason `target failure in 3 of last 5 uses`, the five use IDs and actor
`system:auto-demotion`. It waits for five outcomes and orders them by server record
time, with insertion order breaking clock ties. Retries do not duplicate outcomes
or demotion events.

Demotion clears active scene trust, increments the note version to revoke old
grants, and applies the provisional read/exclusion rules and a maximum 14-day
expiry. Old decisions and sealed provenance remain available for audit. The note
re-enters the queue and can be validated through a fresh evidence-backed candidate
linked by `from_provisional`; old validation evidence cannot promote it again.
Disable, expiry and withdrawal still apply. The main-repo worker owns scheduling,
budgets and freeze controls; these endpoints do not launch runs.

The read-only `GET /dev-v2/memories` and `GET /dev-v2/memories/{memory_id}`
accept the online service key. Their **public allowlist** is memory ID, optional
`title`, `summary`, and `group` (each `null` when absent),
version, status, trust scope/display name, profile, the applicability summary
(benchmark, suite, task, seed, scene/object/camera/task IDs, robot, controller,
perception mode, camera names, task prompt, SDK SHA, namespace and planned
case summaries), failure
signature class/operation, validation summary (planned/completed pair counts,
decision, failed gate codes and signature counts), and source label. Guidance,
repair text, grant bodies, import descriptors, sealed evidence, and keys remain
behind importer authorization. The existing generic `/memories` list continues
to omit Dev v2 rows; the UI must call these new read-only routes to show them.
The UI can display `title` as a short, human-readable heading, `summary` as a
short description, and `group` as a label, while keeping the technical
`memory_id` in the detail view. These fields do not copy
guidance, repair, grants, import descriptors, private evidence, or keys. They are
only read from explicitly supplied fields; guidance is never used to generate them.

Optional `title`, `summary`, and `group` can be supplied in the test-seeded JSON body or in the
sealed source descriptor imported by `/dev-v2/sources` before candidate creation.
`title` and `summary` are trimmed and must each be 1–80 characters of plain text, with no newline,
control character, URL, path, key-like text, or native evaluator/oracle term.
`group` is trimmed, has internal spaces collapsed, and must be 1–24 characters
of letters, digits, spaces, or hyphens. Group labels retain their supplied case;
UIs should compare them case-insensitively when grouping. The normalized fields
are stored in provenance. Invalid values are rejected with explicit errors.
Omitted fields read back as `null`; existing notes are not assigned values retroactively.

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
including its normalized title, summary, and group, is idempotent; changing any of those fields or other
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
