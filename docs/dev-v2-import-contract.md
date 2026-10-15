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
Dev/Advisor, then pin its original files. An arbitrary payload cannot register
roots, files, verdicts or trusted status. There is no registration HTTP endpoint.
Catalog IDs are persisted immutably. Add new IDs on a later isolated service
startup for later sealed snapshots; changing an existing catalog conflicts.
Removal from startup configuration does not delete persisted evidence. Revoke
memory through the lifecycle API.

Every file read must be explicitly listed and match its digest. Absolute paths,
`..`, missing files, changed bytes and unregistered targets are rejected.
Symlinks require an explicit mapping **and** the resolved target separately pinned
at the same digest inside the same root. No outside-root symlink is permitted.
A Dev identity lock includes transitive upstream/source/model/asset/interpreter
files outside its checkout. Choose their existing canonical common execution
root (for example `/home/truares`) and list only the required files; keep the
validator root outside it (for example a dedicated `/tmp/...` isolation root).
This does not grant access to unlisted files or require copying environments.
SDK files are rehashed in streaming reads. Do not pin a whole home directory or
credentials. Paths in HTTP payloads are relative descriptor names, not arbitrary
filesystem read requests.

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
scene contract, its original raw state/validation is rechecked using the frozen
1e-6 comparator (including quaternion-sign equivalence), original contract digest
and full robot/object/fixture state. No tolerance or scene rule is relaxed.
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
  references with full raw robot/object/fixture states; both must pass that exact
  frozen contract at 1e-6, not a partial public observation substitute;
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
issue time. All cases freeze one policy digest. Write the frozen validation input
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
