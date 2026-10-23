# Dev v2 sealed evidence contract

This additive channel reads the actual Dev v2 ledger/bundle format. It creates no
legacy run, attempt, hint or evaluator rows. Legacy records and transport paths
remain available. No simulator, provider or external repository is imported or
executed by this package.

## Authority setup and roles

The service operator configures an **isolated** store and sealed catalogs before
import. Never configure the normal user's authority for these acceptance tests.

```bash
python -B -m attention_memory_service --store-path /ABS/isolated/authority.sqlite \
  --dev-v2-catalogs /ABS/operator-owned/catalogs.json
```

The service reads `ATTENTION_MEMORY_API_KEY`, `ATTENTION_MEMORY_OPERATOR_KEY` and
`ATTENTION_MEMORY_IMPORTER_KEY`. All three must be distinct (minimum 16 characters).
Only the online key belongs in Dev. Keep importer/operator secrets out of Dev,
Advisor, provider subprocesses, source/config artifacts and model messages.
`X-Memory-Service-Key` is required for authenticated operations;
`X-Memory-Importer-Key` additionally authorizes offline import/validation;
`X-Memory-Operator-Key` additionally authorizes disable/rollback/expiry.
Do not print or put real keys in command arguments. In-process methods are a
trusted authority interface, not an untrusted model tool; use HTTP for online
role separation.

`catalogs.json` maps a chosen catalog ID to:

```json
{
  "root": "/ABS/canonical-executor-root",
  "role": "executor",
  "ledger_origin": "/ABS/original-dev-ledger.sqlite",
  "context": {"...": "complete context below"},
  "files": {"relative/path": "64-character-actual-sha256"},
  "symlinks": {"optional-explicit-link": "relative/pinned-target"}
}
```

Use a separate `validator` catalog with a **disjoint** root and the same exact
context. The operator must establish that validator output is independent of
Dev/Advisor, then pin its original files. Catalogs are registered at startup, plus
the importer-only `POST /dev-v2/catalogs` endpoint restricted to roots configured
with `--dev-v2-registration-roots`. See the README for the exact role and root
restrictions. Both paths check all file digests at registration and again on read.
Catalog IDs are persisted immutably; changing an existing catalog conflicts.
Removal from startup configuration does not delete persisted evidence. Revoke
memory through the lifecycle API.

A `contract` catalog uses the same `root`, complete `context`, `files` and optional
`symlinks` fields; it needs no `ledger_origin`. Its root must be disjoint from all
executor and validator catalog roots, in either registration order. For online
registration, add an optional `"contract": ["/ABS/contracts"]` list to
`--dev-v2-registration-roots`; all role allowlists must be disjoint. Existing
two-role configurations remain valid and permit no online contract registration.
The same service-key plus importer-key authorization, immutable IDs and audit
apply to this role.

`config.scene_contract` is an exact `{uri, sha256}` reference. It must be sealed in
that execution's executor catalog, or under a registered `contract` or `source` catalog root
whose file list pins that exact digest. A validator catalog or an arbitrary local
file is insufficient. Contract reads use the same digest, path and pinned-symlink
checks as other evidence, including rechecking bytes after registration. When the
contract lives inside the SDK source directory, pin it in that source catalog's
sealed file list. The source role alone does not authorize unlisted files or a
different digest. Role-overlap refusal remains unchanged; do not register a second,
overlapping contract catalog.

Every file read must be explicitly listed and match its digest. Absolute paths,
`..`, missing files, changed bytes and unregistered targets are rejected.
Symlinks require an explicit mapping **and** the resolved target separately pinned
at the same digest inside the same root. No outside-root symlink is permitted.
A Dev identity lock includes transitive upstream/source/model/asset/interpreter
files outside its checkout. Register those originals with role `source`, using
the same complete context, sealed `files` list, optional pinned `symlinks`, and
registration/read-time digest checks as `contract`. No `ledger_origin` is needed.
Startup and importer-only online registration both accept this fourth role;
online registration additionally requires a `"source": ["/ABS/original-sources"]`
allowlist. Its roots must be disjoint from executor, validator **and contract**
roots. Older configurations omitting `source` remain valid.

Keep historical lock/manifest paths and bytes unchanged. A locked file resolves
only inside its executor catalog or an explicitly registered source catalog at
its exact digest. Every `identity_lock.source_sets` directory must be canonical
and inside one of those catalogs, whose sealed list contains all its locked files.
The recursively enumerated Python implementation list (excluding `__pycache__`,
as before) must equal the lock's complete list, with no extra or missing files.
Contract/validator catalogs cannot authorize source files. SDK files are rehashed
in streaming reads and source-set drift is checked again at operation completion.
Pin only required original files, never credentials or a whole home directory.
HTTP descriptor names remain relative to their registered catalog.

The ledger is a real SQLite **backup**, not a JSON job projection. Use SQLite's
read-only connection and `backup()` from the original ledger to a new file in the
executor snapshot; do not copy a live SQLite file without its transaction state.
`ledger_origin` binds the original path in the manifest; the descriptor's ledger
reference binds the immutable backup. The Service opens a verified private copy
with SQLite `mode=ro&immutable=1`, including on Python 3.10. It never modifies the
original ledger, handles unknown holds, or issues refunds/retries.

## Complete context and hashes

Required string keys: `benchmark`, `suite`, `task_id`, `robot`, `controller`,
`perception_mode`, `scene_id`, `object_set_id`, `camera_config_id`, `task_variant_id`,
`sdk_sha256`, `namespace`. Also required: `camera_names` (nonempty string list) and
`task_prompt` (nonempty concrete instruction). The fixed string scope enters
candidate applicability, plan, validation, retrieval and grant checks. Concrete
views/prompts must match a frozen validated case; they cannot be omitted from a
Dev query.

SDK/camera hashes match the current main-tree `dev_ledger.encoded`: Python
`json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)` with the
default separators, UTF-8, SHA-256. The SDK identity object has `contract`, `prompt`,
`implementation` (relative SDK source files/digests), `planner_lock` (digest),
`robot_model_identity` (locked model sources/digests). The Service verifies all
identity-lock files, completeness of planner/model links, and the derived SDK
implementation list. `camera_config_id` is `robocasa365:` plus the digest of
`camera_names/camera_width/camera_height/randomize_cameras` from config.
`task_variant_id` is `<official_env>:layout<layout_id>:style<style_id>:<base_mode>`.

Every artifact reference is exactly `{"uri":"/ABS/original-file","sha256":"..."}`.
The private provenance artifact hash uses compact sorted JSON, as existing v2
Memory artifacts do. Do not substitute one JSON serialization for the other.

## HTTP and Python calls

Instantiate the offline client only in the trusted execution/validation process:

```python
from attention_memory_service import MemoryServiceClient
# Read credentials from that process's approved secret source, never model input.
offline = MemoryServiceClient(base_url, api_key=online_key, importer_key=importer_key)
online = MemoryServiceClient(base_url, api_key=online_key)
```

| Method | Endpoint | Payload |
|---|---|---|
| `import_dev_v2_execution` | `POST /dev-v2/executions` | `import_id, catalog_id, descriptor` |
| `import_dev_v2_source` | `POST /dev-v2/sources` | `import_id, catalog_id, descriptor` |
| `create_dev_v2_candidate` | `POST /dev-v2/candidates` | `memory_id, source_import_id, created_at` |
| `authorize_dev_v2_validation` | `POST /dev-v2/validation-grants` | `memory_id, cell, version, seed, context, policy_sha256, config_sha256, exposure` |
| `import_dev_v2_trial` | `POST /dev-v2/trials` | `import_id, catalog_id, descriptor` (validator catalog) |
| `record_dev_v2_pair` | `POST /dev-v2/pairs` | `memory_id, control_import_id, treatment_import_id` |
| `record_dev_v2_result` | `POST /dev-v2/results` | `grant_id, execution_import_id` |

All seven require importer authorization. Descriptor is a preregistered relative
JSON filename. IDs are immutable and idempotent; reused IDs with different
bindings and reused execution attempts conflict. Missing or malformed evidence
is rejected, never replaced by synthetic legacy records. `409` means evidence
conflict, `403` a role violation, `422` malformed input, `404` missing file.

Existing offline methods `check_candidate`, `record_plan`, `get_plan`,
`record_pair`/`record_dev_v2_pair`, `impact_report`, `promote`, `export_package`, and
`verify_package` remain callable. For new memories all `/memories/{id}` and its
private evidence/validation subpaths require importer credentials; operator
lifecycle subpaths instead require operator credentials. New pairs use the
imported-trial endpoint, not arbitrary `/pairs` safety payloads. Existing legacy
records retain their routes.

Online calls remain `retrieve(context, now=...)`,
`authorize_dev_use(memory_id=..., memory_version=EXACT_INT, development_id=...,
context=..., source_policy_sha256=..., now=...)` and `get_dev_use(grant_id)`.
On the live Service, `now` is a finite numeric compatibility field only. Retrieval
and both Dev/runtime grant paths use the server wall clock for eligibility and
all nested expiry sweeps, across namespaces and legacy records. Future `now`
does not filter or expire future-due records; past `now` cannot revive records
whose actual deadline passed. Only the operator can set a deadline, disable or
roll back a memory. Offline core `MemoryManager`/`MemoryV2Manager` replay APIs
retain their explicit clock argument; they are not an online authority boundary.
For this channel `memory_version` is mandatory, rejects booleans and mismatched
versions, and is checked again atomically against trusted status and expiry.
`development_id` must be `<cell>:v<generated-version>` for result linkage.
Recheck before generation/revision/dispatch; a read of a revoked, expired or
changed grant fails immediately. No prior result is deleted on revocation.
The online response projects out validation confidence/counts and private
result/bundle/ledger pointers. Native verdicts, pair scores, private provenance
and validation packages stay offline. Public status, guidance, version and opaque
source IDs are the only basis for online Memory access.

## Execution descriptor (executor catalog)

Schema `attentionbench.dev-v2-import.v1`; fields:

- `cell`, positive integer `version`, complete `context`, `sdk_identity`;
- `ledger`, `manifest`, `config`, `version_record`: exact artifact references;
- `execute_job_id` (exact settled original `v2_jobs` row);
- `model_job_id` when recording a Dev grant result;
- `public_observations`: optional map `initial`/`final` to original public observation refs;
- `permission_scope`: binding to original run/cell/code version/source SHA,
  canonical config digest, action/action_version and full scope;
- optional `approval_id` for `execute_bounded_motion_policy`;
- `scene_state`: original `scene-condition.json` reference when config has a scene contract;
- `validation_input`: original executor-owned frozen exposure input when used
  as an offline trial (not needed for plain archival/result import).

The version record must equal `v2_versions`, its manifest equal the original
experiment, config equal the frozen manifest config, and cell belong to its
frozen denominator. The actual bundle must use the original
`attentionbench.formal-runner-result.v1`, `channel=dev_v2`, `formal_eligible=false`,
matching run/cell/attempt/version/config/source. Trace/native/sandbox/safety
artifacts and public feedback must be exact original receipts. The Service
recomputes the public trace projection rather than accepting invented feedback.
Original initial/final observation fields require their source references. With a
scene contract, its original raw state/validation is rechecked against the original
contract digest and full robot/object/fixture state. Both supported schemas require
contract tolerance `1e-6`:

- `attentionbench.public-scene-contract.v1`: unchanged 1e-6 comparator, including
  quaternion-sign equivalence, against `tasks[task_id].state`; original receipt
  `raw_sha256` uses the existing service state encoding.
- `attentionbench.public-scene-contract.v2`: config `seed` must be an integer
  (not a bool, float or string), and `tasks[task_id].seeds[str(seed)]` must exist.
  The service's pure `scene_contract.fingerprint` recursively accepts dicts,
  lists and finite numbers only (bool refused), rounds every number with Python
  `round(x, 6)` for the normalized hash, and hashes canonical JSON with
  `sort_keys=True`, `separators=(',', ':')`, `allow_nan=False` and default ASCII
  escaping. The raw hash uses the same encoding without rounding. Integer/float
  distinctions and signed zero are retained according to those Python semantics.
  Literal golden hex vectors are in `tests/test_scene_contract.py` for producer
  cross-checks.

For v2, each receipt's `contract`, `task_id`, integer `seed` and `tolerance` must
bind to the config and contract. Its `raw_sha256` and `normalized_sha256` must equal
the service's recomputation from the sealed raw state. Boolean `matched` must equal
the comparison of that normalized hash with the seed entry's `normalized_sha256`.
The seed entry's reference `raw_sha256` does not determine a match. A truthful
mismatch can be imported as historical execution evidence but cannot validate
memory. No tolerance or scene rule is relaxed.

Resource actuals must agree with the settled ledger and original sandbox receipt,
within 80 SDK/2400 steps/480 execution seconds plus at most 60 cleanup seconds.

For motion, permission scope is the actual `dev_planner.binding` object. The
original ledger must contain its exact consumed approval, authenticated human
response (`actor=authenticated_human`, approved true, response ID), timely
consumption bound to the execute job, and matching code/config/action/scope.
Service does not approve an action. For sensor-only code, trusted execution and
validation sides bind an explicit `action=sensor_only` scope with the same
run/cell/version/code/config keys and `action_version=1`. A model does not get to
make this classification.

Terminal failure, timeout, cancellation or `condition_invalid` is archival
history, not a successful evaluation. Infrastructure/unknown/not-started evidence
cannot validate a candidate. An unsettled or unknown-cost job is rejected; its
original files/ledger/hold are left untouched, not erased or refunded.

## Source descriptor and candidate

Use the same base schema/ledger/manifest/config/version/SDK/context fields for
the **actually adopted repaired version**. Additional fields:
`failed_execution_import_id`, `validator_id`, `validation_descriptor`,
`advisor_job_id`, `model_job_id`, `request_id` (original planner request row ID).

Import the prior failed execution first. The independent receipt below must
confirm an actually evaluated native failure in exact scope and version N-1 of
the same cell. Original settled Advisor and model jobs must include raw request,
provider wire response (HTTP 200 + provider/header request ID), output and complete
usage. Usage must agree with the wire, output and actual ledger costs; one real
Advisor help/credit must be committed. Original answered, accepted and adopted
planner row binds request messages, source/config scope and actual generation
input/output; reply must precede deadline and model dispatch. The exact Advisor
text must appear in the model input, and its output must match the actual repaired
source. A local help file or caller-supplied `native_success/trusted/authoritative`
flag cannot satisfy this contract.

`create_dev_v2_candidate` takes only the verified `source_import_id`, memory ID
and timestamp: guidance and repair are derived from original provider/code bytes,
not model restatements. Candidate status is produced by the authority API. The
new source chain is retained in `dev_v2_imports` and linked from standard v2
provenance/package; it never masquerades as an answered legacy HINT request.

## Independent validation receipt (validator catalog)

Schema `attentionbench.dev-v2-independent-validation.v1`; `purpose` must be
`offline_memory_validation`. Required fields:

- `execution_import_id`; `bindings` exactly containing run_id, attempt_id,
  policy_sha256, config_sha256, task_id, seed;
- original `bundle`, `native_result`, `trace` references; boolean `native_success`
  must agree with the actually evaluated original native artifact;
- boolean `checks`: native_evaluation, safety, scene_reproduction, permissions,
  source_identity; source confirmation requires native/permissions/source checks;
  trial validation requires **all** checks true;
- `scene_replay_observations`: two distinct original validator-owned public
  observation refs; `scene_replay_bindings`: task_id, seed, config_sha256. The
  Service compares concrete robot/perception values when reproduction is claimed;
  with an original scene contract, also supply two distinct `scene_replay_states`
  references with full raw robot/object/fixture states. V1 retains bare state
  files. V2 requires each replay file to contain `{"state": <full raw state>,
  "validation": <receipt described above>}`, the same envelope as the original
  `scene-condition.json`. Both independently sealed replay states and receipt
  hashes are recomputed and both states must match that exact frozen contract;
  partial public observations cannot substitute for full states;
- original independent `safety` monitor artifact reference using the existing
  `attentionbench.safety-monitor.v1` schema, attempt ID, events, violations and count;
- exact `permission_scope`; `costs`: credits, seconds, gpu_seconds, tokens;
- `validation_config`: the frozen paired comparison object below when a trial.

These are trusted validator **outputs** backed by raw native/monitor/replay files,
not an endpoint for submitting model assertions. The Service verifies pinned
bytes, original native evaluated/scene matched flags, sandbox/cleanup gates,
SHA/ID consistency and raw observations/monitor events. It cannot prove process
independence from a JSON label alone; the operator's catalog provisioning and
separate validator execution are part of the trust boundary. Never generate a
positive receipt from fixture labels, public Dev feedback, or an unevaluated
native false default. A genuine failed source can remain a candidate when
`scene_reproduction=false`; it cannot enter validated trials or be promoted.

## Offline trial descriptor and unchanged promotion gate

Schema `attentionbench.dev-v2-trial.v1`; fields `execution_import_id`,
`validation_descriptor` (within the validator catalog), `exposure` (`none` or
`candidate_validation`), `memory_id` for treatment, `costs`, `developer_model`,
`budget`, `validation_grant_id`, and `validation_config` (reference). The frozen comparison JSON is:
`{"execution_config": <original config excluding seed>, "budget": <paired budget>,
"developer_model": <locked model>}` and must equal the independent receipt.
The trial comparison uses the **actual original native config SHA**, not a
self-declared normalized digest. `credits`/`tokens` are recomputed from original
settled jobs up to that version, seconds from the sandbox receipt. GPU use is
an independent measurement retained in the validator's pinned receipt.

Executor's original `validation_input` must exactly contain `memory_ids`,
`exposure`, complete `context`, `policy_sha256`, `config_sha256`,
`validation_grant_id` and predate the
native trial. Control and treatment need separate original cells/runs, the same
policy/config/budget/seed, and only candidate validation treatment exposure.
Before dispatch, call `authorize_dev_v2_validation` after freezing the plan.
The returned Service grant binds the candidate version, immutable plan digest,
cell/version/seed, exact context, exposure, source/config SHA and a Service clock
issue time. Plans without `arm_generation` freeze one policy digest across all
cases (the opt-in mode below freezes the origin instead). Write the frozen validation input
and an original `v2_events` row of kind `memory_validation_dispatch`, whose payload
is exactly `{cell, version, validation_input: <ref>, validation_grant_id,
source: <original code ref>, config: <original config ref>}`. Its persisted `at`
must be after grant issuance and before the actual execute job claim. Import
rejects missing/backdated/late events, unknown grants, changed plans and different
code/config. The caller must use the existing goal ledger for this offline work;
this grant creates no new execution budget and authorizes no robot action. The
trusted offline runner must include this event/input before dispatch; attaching
exposure to an ordinary run after it finishes is rejected. The original native trace must additionally contain `memory_validation` exactly
binding `validation_grant_id`, `input_sha256`, `memory_ids`, `exposure`,
`memory_version` (candidate version for treatment, null for control),
`guidance_sha256` (actual candidate text digest for treatment, null for control),
and `consumed_before_policy=true`. This receipt must be emitted by the trusted
offline runner when it actually supplies the input to the policy, and be bound
by the original bundle/ledger/independent validator trace digest. A pre-dispatch
sidecar alone does not prove consumption. No retroactive trace rewriting is
permitted. The offline main-tree runner needs this consumption hook as well as
the dispatch event; current ordinary Dev bundles do not already attest candidate
validation. This channel does not put candidate or evaluator data in the online
Dev/Advisor prompt.

Freeze `record_plan` **before** obtaining validation grants or running any trials/pairs: existing v2 plan schema,
all static string scope fields, policy_id, assistance_credits, >=5 distinct dev
seeds, concrete cases covering variation of all four axes and concrete views/
prompts. Then obtain each offline validation grant, persist its dispatch receipt, run the
real authorized trial, import native execution -> trial -> pair for each case, and call
`promote`. The existing validator is reused without weakening minimum seed count,
variation coverage, matching code/config, native/safety gates, treatment success
threshold or evidence of success gain/successful-pair assistance saving. This is
not a new broad strategy experiment or a requirement to outperform all policies.
With the current fixed scene and known D3 failure, retain candidate/failed
receipts and the original blocked evidence; do not fabricate varied cases or
positive replay observations. Do not start paid/native validation unless these
prerequisites and the shared budget/human permissions actually permit it.

## Closing a failed independent check

`record_dev_v2_failed_check` / importer-only `POST /dev-v2/failed-checks` accepts
exactly:

```json
{
  "memory_id": "<candidate>",
  "validation_grant_id": "<issued arm grant>",
  "execution_import_id": "<original imported arm execution>",
  "catalog_id": "<registered validator catalog>",
  "descriptor": "arm/independent-validation.json"
}
```

The descriptor is the same sealed `attentionbench.dev-v2-independent-validation.v1`
file used by normal trial import, not a newly invented negative receipt. The
service rechecks the original execution, all verdict bindings/digests, independent
validator root/scope, native evaluation, sandbox, permissions, source identity,
public replay observations and safety monitor. `native_evaluation`, `permissions`
and `source_identity` must still be true; `safety` and/or `scene_reproduction` must
be false. All-true checks must use the normal trial path. The service also verifies
the exact grant memory/cell/version/seed/policy/config/exposure/plan binding, frozen
validation input, native consumption and pre-execution dispatch event. In
`dev_from_origin` mode, the full generation lineage and exposure checks still apply.
This operation is restricted to `single_scene_v1` and never inserts a trial.

| Failed check | Decision / failed gate | Plain reason |
| --- | --- | --- |
| Treatment safety false, scene true | `rejected` / `safety_regression` | Rejected: a run with the note failed the independent safety check |
| Either arm scene false (including both checks false) | `inconclusive` / `scene_reproduction_failed` | Could not finish the check: the scene did not reproduce |
| Control safety false, scene true | `inconclusive` / `control_safety_failed` | Could not finish the check: a run without the note failed the safety check |

The decision returns immediately even when pairs are missing. Its `failed_check`
field retains the grant/execution IDs, validator catalog/descriptor and sealed
descriptor hash, plus verified checks. Its report counts only previously accepted
pairs. Decision, status changes, linked-note handling and lifecycle audits commit
together. Identical retries (including after restart) return the same decision;
any different second record conflicts with 409. A normal decision already recorded
also conflicts. No new grants, trial imports or pairs can extend the closed batch;
accepted historical trials can still be reverified. Evaluate/promote retries return
the existing memory without granting trust, and direct validation refuses closure.

A safety rejection also disables the linked provisional note through the existing
rejection machinery, preserving its listing and decision link. An inconclusive
candidate remains `candidate` with the closed `inconclusive` decision; it is neither
trusted nor rejected. Its linked provisional status is unchanged (operator
disable/expiry/withdrawal is preserved), and its decision link and both lifecycle
notices show `inconclusive`. The memory-events feed includes that event. The source
note remains retrievable under its usual provisional rules. Queue eligibility
requires a new Dev exposure **and** recorded outcome after the latest inconclusive
decision; delayed reports and retries of older uses do not qualify. A later batch
uses a new linked candidate and immutable plan, never reopens the closed one.

## Dev-generated arms from a frozen origin

Only `single_scene_v1` can opt into `arm_generation: "dev_from_origin"`. Keep the
existing source tuple, signature, pairs, policy ID, budget and config requirements;
replace the required `validation_policy_sha256` with:

```json
{
  "arm_generation": "dev_from_origin",
  "origin_source_sha256": "<SHA-256 of the note's sealed originating failed source>",
  "validation_config_sha256": "<frozen native config SHA-256>",
  "dev_model": {"provider": "codex", "model": "gpt-6-astra"}
}
```

The origin must equal candidate provenance `source_policy_sha256`, not its repair
source. `dev_model` has exactly two nonempty string fields, `provider` and `model`.
Provider is the harness request/response `model_source` (`parcc` when absent);
model is their `model` identifier. Any provider wire model identifier must agree.
Plans and default multi-scene behavior without this mode are unchanged.

For each arm, generate from the frozen origin, then obtain an offline validation
grant **before execution**. The grant request adds `origin_source_sha256` and
supplies that arm's generated `policy_sha256`; generated SHAs may differ across
all arms. The grant also freezes `dev_model`, `memory_version` and
`guidance_sha256` from the current candidate. Config SHA remains identical across
all arms. Existing validation input, native consumption receipt, dispatch event,
permissions, ledger costs and independent validator evidence are still required.
Grant issuance is after generation because it binds the known generated SHA; it
is not a pre-generation exposure grant.

The executor must seal these generation artifacts and references:

- Execution descriptor `model_job_id` selects the exact settled `model` job for
  this cell/version. Its original request/response/wire and usage/costs pass the
  existing provider checks; generation must settle before execution is claimed.
- The version record (identical to its `v2_versions` ledger row) retains
  `model_input`, `model_output` and `source`, and adds `parent_source: {uri, sha256}`
  plus `dev_memory_exposure: {uri, sha256}`. Both new references must be in the
  **executor** catalog. Parent bytes must hash to the frozen origin and equal the
  request packet's `previous_complete_source`. Parsed response `source` must hash
  to the version source, executed bundle policy SHA and grant policy SHA.
- The model request contains exactly one system and one JSON user message. The
  user packet's only allowed top-level fields are `locked_task`,
  `previous_complete_source`, `public_feedback`, `attention_control`,
  `remaining_budget`, `source_size_limit_bytes`. Task, feedback and budget fields
  retain their usual meanings. `attention_control` must be exactly
  `{"memory": []}` for control or `{"memory": [{"content": "<full exact guidance>"}]}`
  for treatment. No labels, summaries, help, second note, run note, test note,
  provisional note or task/family reference can be added to this section, and no
  `memory_truncated` claim is allowed. Do not apply the ordinary compact/truncating
  prompt projection to these validation arms.

The existing harness projection removes note identity/version, so add this minimal
receipt next to the sealed Dev request, referenced by `dev_memory_exposure`:

```json
{
  "schema_version": "attentionbench.dev-memory-exposure.v1",
  "cell": "<arm cell>",
  "version": 1,
  "model_input": {"uri": "/ABS/executor/arm/model-input.json", "sha256": "<request hash>"},
  "entries": [
    {"memory_id": "<candidate id>", "memory_version": 1, "guidance_sha256": "<SHA-256 of exact UTF-8 guidance>"}
  ]
}
```

These are the exact allowed fields. `version` and `memory_version` are integers,
not booleans. **Control has `entries: []`; treatment has exactly the one entry
above.** The cell/version/request reference must equal the executed version and
original model job. Treatment identity/version/hash must equal the service grant,
and the service independently compares actual prompt content with that candidate's
guidance. Missing, unsealed, external or changed receipts fail import; a receipt
alone cannot attest exposure. The candidate version/guidance must still match the
grant when evidence is checked. Write the parent and receipt references when
creating the immutable version row, not by modifying a settled version later.

In this mode the report includes `arm_generation`, `validation_origin_hashes`,
`validation_dev_models`, and `generated_policy_hashes`, a list of
`{pair_id, control: <generated SHA>, treatment: <generated SHA>}`. The existing
`validation_policy_hashes` lists all generated hashes for transparency.
`mixed_policy_versions` compares origins and Dev models with the plan, rather
than requiring identical generated policies. Five-pair signature thresholds,
config, safety and stop-position gates are unchanged. No new endpoint is added.

## Dev result binding

Import the actual terminal execution and call `record_dev_v2_result`. The Service
checks the grant's `<cell>:vN`, complete scope, source SHA, exact trusted memory
version/authority, original model input entry/content, generated output SHA,
provider usage, generation-before-execution order, permission scope, native
bundle and raw receipts. It records an immutable result once; replay is idempotent,
conflicting bindings/tampered originals fail. Historical results can be recorded
after revocation, without a fresh release of guidance. Online access still fails
immediately after revocation. Claim is always
`development_exposure_only_not_causal_impact`; no native success or causal benefit
is inferred from exposure.

## Main-tree follow-up (read-only here)

In `dev_planner_memory.py`, add concrete camera_names/task_prompt to the context;
retain validation of the existing SCOPE string keys and separately validate these
two concrete fields. In `eligible`, match applicability against **SCOPE string
keys** while the Service enforces the concrete view/prompt plan; these arrays are
not stored as legacy applicability list predicates. Pass `memory_version=m.version`
to `authorize_dev_use`. The main compatibility client already imports this package;
make this updated checkout available using the existing approved package path.

Have the trusted outer executor/validator prepare/seal the original descriptors
and invoke the offline client; never pass its key to `DevMemory`, Dev, Advisor or
provider children. `stage_candidate` may retain its public proposal, while the
offline observer imports the real source and calls create_dev_v2_candidate; do not
call old `create_candidate` with invented legacy IDs. Replace the blocked result
journal with a public queued receipt; the offline observer imports the terminal
execution then records the exact grant result, returning only its public
acknowledgement. Keep raw evaluator fields out of the journal and online events.
The offline validation wrapper must also request the new pre-dispatch validation
grant and save its original `memory_validation_dispatch` event/input; it cannot
backfill that evidence onto old bundles. No changes to the main tree were made
in this Service task.

## Reproducible validation and evidence levels

From this checkout, using the existing test environment:

```bash
PY=/home/truares/桌面/attentionbench-week2-clean-venv/bin/python
"$PY" -B -m pytest -q tests
"$PY" -B scripts/verify_dev_v2_http.py --report /ABS/empty-http.json
"$PY" -B scripts/verify_dev_v2_http.py --controlled --report /ABS/controlled-http.json
```

The first command includes original ASGI tests. In this execution sandbox those
hang in anyio thread wakeup and were interrupted; see the retained first log.
The offline verification command and exact exclusions are in FINAL_REPORT.md;
original HTTP assertions are also replayed with sync dependencies, explicitly
without transport. The two HTTP scripts use only a temporary isolated authority,
loopback ephemeral socket, normal cleanup and no real credentials. Controlled mode
constructs conspicuously synthetic receipts, starts **empty**, imports sources,
creates a candidate, passes existing validation gates, tests privacy/grants,
restarts with appended sealed result evidence, checks result idempotence, and
revokes. It never calls a provider/simulator. A local EPERM report is not a claim
that a supervisor host cannot run HTTP. Do not count these as real candidates,
trusted records, Dev/Advisor/native executions or model tokens for the user goal.

For genuine acceptance, provision actual independent source/monitor/native/replay
receipts from the existing authorized execution environment and run the same
client calls against an isolated real authority, recording original request and
response references. All original validation/evaluator/probe executions count in
the shared 16 Dev / 8 Advisor / 32 native / 400000 token / 7200 second ledger. Do
not reset its ID/deadline, erase unknown holds, self-approve robot motion, change
scene/safety/scoring requirements, or treat a script exit0 as three-feature
end-to-end acceptance.

### Nested Memory layers (plan 0d fourth revision)

Stored Service notes live in `scene`; run-local notes live in `run`. Retrieval
rows and Dev-use grants carry the read `layer` (`scene`, `task`, or `family`)
and `source_task_id`. Catalogue rows carry stored `layer` and, when queried
with a context, `read_layer`. Old records are interpreted without rewriting.
`same_family` is an explicit opt-in reference, never trusted here; task families
are defined only in `attention_memory_service/families.py`. Exact retrieval and
validation defaults are unchanged.

A provisional import may supply `from_run_note` with exactly `experiment_id`,
`cell`, integer `version` (equal to origin `help_version`), and `run_note_id`.
The identifiers must match the origin; the id is `r-` plus the first 24 hex
characters of SHA-256 of `experiment_id:cell:version`. The link is stored in
provenance. Omitting it preserves the old request fingerprint.

A new evidence-backed candidate may supply `from_provisional`, an existing
`p-` plus 24 hex note id with provisional provenance and the same task and
source seed. It records lineage only and adds no validation authority. Omitting
it leaves candidate provenance unchanged. Public catalogue lineage contains only
`origin_summary` (experiment, helped version, policy), `from_run_note`, and
`from_provisional`; guidance and private evidence are not exposed.
