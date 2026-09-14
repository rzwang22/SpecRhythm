# K3 local verification and pending operator evidence

Baseline execution c0ecc2a405c8b6c9cb3016c7254739bda5bec2b6, continued from clean
0b37d37ff364bcc6533b8805940bcbe572dafc56. Draft PR5 only; no reset, historical output
rewrite, other PR changes, AutoDL connection or GPU execution.

The local retained `pingpong-prepost3-delivery-20260913T160424Z-1632.tar.gz` was
discovered during export-layout review. All102 included logical objects were read
through inventory.logical_paths and verified for byte count/SHA256, without modifying
source results. Its c0ecc2a ordinary/eager figures are55.254760/55.150790 tok/s, with
the original execution/measurement/cleanup PASS and negative performance conclusion.
[Read-only timing derivative](k3-baseline-observations.json) records archive SHA256,
join method and original physical/native array indices. Across77 adjacent measured
ordinary transitions, the next home's entire claimed batch was already READY before
parent feedback. All77 still finished the previous home's seed+three extensions
before the next Target GPU start. The154 owner status queue waits average41.440461ms.
This supports an actual dispatch serialization mechanism, not absence of legal work.
It does not identify all Target pre-forward time or predict new throughput.

CPU tests cover real K3 machine/backend, true owner queue, actual service factory,
resident Target scheduler/adapter, and pinned vLLM source bookkeeping with deterministic
imperfect Draft and independent Target oracle. Both feedback orders, repeated first/
mid rejection, later success, budget1/2, EOS, refill, invalid version/frontier, duplicate
claim/feedback, cancellation and retirement are exercised. Full/runtime allocator
runs retain real ownership/fence checks and reconcile at state boundaries. A held
physical step cannot be read as completed; admission stays on the authoritative owner.

The real drive report→immutable serializer→summarize/qualify→single export→re-read
regression includes all three modes and capacity/correctness/performance. The previous
strict probe boolean, phase equality, device identity and request/proposal/native TP
associations remain enforced. Joint failures at Target-only and each of three modes
retain true mode/run/layer/process0-vs-command1, and exporter failure does not erase
first error. Three-point command and single-package/parent-shell behavior are tested.
Native times in offline arithmetic tests are explicitly synthetic, not GPU evidence.

Development first failures were retained: duplicate target_bonus_is_committed metadata
key, a fixture using the wrong proposal_row keyword, and initial registration calling
super().batch_propose directly instead of the overridden queued K3 seed path. These
were corrected in the implementation/fixture, not by permitting short READY proposals.
The first full suite found two stale test assumptions: the shell failure stand-in still
used the middle mode as the last capacity point, and the resident360 capacity test did
not include the three new grouped modes. The test models now explicitly assert the
new roster and K3 budget/ceiling while retaining every old-mode assertion. No timeout
or budget was enlarged. Later focused tests passed after the concrete corrections.

Prior baseline CI: push jobs all SUCCESS; PR Python3.9/3.12 and source contract SUCCESS,
PR python311 contract failed in the old `pingpong-True` CPU full-pool drain fixture:
`RuntimeError: DataError: diagnostic total drain deadline expired`. The original log
was inspected; it uses a5s deadline across resident360 cleanup. It does not implicate
K3 (which was absent). The log does not isolate the elapsed per-operation cause, and
local pass does not establish why that historical run failed. No retry-to-pass,
assertion relaxation or old scheduler/timeout modification was used.

Final local full suite:2414 passed,3 existing GPU/platform skips,0 failed. Ruff,
compileall, Python3.9 grammar and Bash syntax checks pass. The final counter check
also covers cancelling an incomplete seed+extension, consumed claims, pending eager
tokens, and repeated settlement; strict evidence qualification rejects inconsistent
candidate conservation. New CI status is recorded at delivery. GPU output correctness,
execution/measurement/cleanup, evidence integrity, cross-home pipeline, native overlap
and performance remain PENDING until the new operator package is returned.

Compatibility checks: Python3.9 relevant148 tests passed; Python3.12 relevant136
passed. Both versions cover production reports and K3 paths;3.9 additionally covers
real scheduler/service routing and allocator audit transitions.354 src/test Python
files parse with Python3.9 grammar; compileall passes on3.9 and3.11. Ruff passes;
all20 current repository Bash files pass syntax validation. The pinned entry is
validated again after its execution SHA is fixed.

Execution commit: `b2a2d29210d68357590a567d4819f3552175d179`. The subsequent delivery
adds the pinned child launcher, its three failure/success regressions and docs only
(plus CI syntax coverage); no src change. Launcher+foreground14 tests pass on3.11,
the three new launcher tests pass on3.9, all21 Bash files and355 Python grammar
checks pass. The explicit pinned vLLM CI source job includes the new Target test;
its full local98-test selection passed. Remote CI is checked after ordinary push;
no pending check is labelled PASS.


## Capacity failure follow-up (2026-09-14)

Read-only verification of `pingpong-k3-delivery-20260914T051537Z-2351.tar.gz`
resolved inventory.logical_paths and checked sizes/SHA256 for all17 logical files.
Export was COMPLETE. The first serial-k3 capacity attempt failed at the legacy
minimum4 assertion, before any joint correctness, other capacity or performance point.
Historical labels remain capacity UNKNOWN, execution FAILED, measurement INVALID,
cleanup FAILED and performance PENDING. No historical record was changed.

Coordinator exit1 and Draft exit0 coexist with cleanup_valid=false: the wrapper
observed live descendants and the supervisor subsequently emptied its owned group.
The final empty PID set does not retrospectively qualify normal teardown. The package
shows supervisor SIGTERM/reap and socket removal, not an orderly pre-drive Draft RPC.
The observed descendant's exact role is not inferred from its PID/name alone.

The first new real-run regression failed in all three modes with the original
`speculative capacity cannot be less than the baseline K4 reserve` message. Earlier
report fixtures entered at drive(), supplying a capacity report and bypassing the
model-initialized run() arithmetic. New tests substitute only model/worker execution
and hardware query responses, run the real capacity function and report construction,
serialize, summarize/qualify, export, reread and apply the same capacity/device contract
for all nine mode/stage combinations. Full-run fixtures initially also exposed missing
attention-backend, batch-invariance and native timing fields in the CPU GPU stand-in;
those fixtures now supply the real startup/forward producer inputs. No production
qualifier was weakened to accept those omissions.

New cases exercise exact-fit/one-block shortage across block boundaries, invalid
bool/float/negative/missing values, frozen budget mismatches, all three reservations,
legacy defaults, and actual backend materialized frontiers/correction catch-up for
both early/late feedback and rejection. A real owner + Unix socket server handles
pre-drive shutdown in CPU tests. Constructor, transport, Target shutdown and disk
failures preserve the original capacity error and remain separately visible in the
single archive. Script tests retain first failure over a second export failure and
prove the static entry runs before prepare/model commands. GPU allocation and actual
vLLM descendant cleanup remain PENDING, as do joint outputs, native pipeline overlap
and performance. Local validation results are recorded in project-status.md.

The local Python3.12 environment initially lacked an installed package for child
processes: the new static CLI and an existing supervisor regression each exited1
with `ModuleNotFoundError: specrhythm`. Both original child stderr records identify
that import failure, rather than a drain timeout or capacity assertion. Supplying
PYTHONPATH=src, as the production runner already does, yielded266 passed; Python3.9
also passed the same266 cases. No test, timeout or production branch was changed for
this environment correction. The initial2 failures remain recorded here.

The first full local suite reported2510 passed,3 skipped and2 failures in unchanged
Phase4B.2 shell tests. Their subprocess stderr was `python: command not found` at
phase4b1_gate_helpers.sh:229, with actual shell rc127 instead of the intended2/19.
Invoking the virtualenv interpreter by absolute path had not activated its bin
on PATH. The shell regression passed after supplying that PATH; a complete suite
was then rerun under the correctly activated environment. This environment failure
is separate from the reproduced K3 capacity bug and from historical GPU cleanup.


Final full suite: **2512 passed,3 skipped in343.09s** with the pinned vLLM source
checkout enabled. Existing skips are the opt-in CUDA backend and two Linux-only
owned-process exit-status cases on macOS. Python3.9/3.12 relevant suites each pass266;
the final shell/launcher suites each pass15. Ruff, compileall,360 Python files parsed
with the3.9 grammar,21 tracked Bash scripts, and git diff whitespace checks pass.
Only environment setup changed between the identified import/PATH failures and
revalidation; no assertion or timeout was relaxed. New GPU capacity, joint correctness,
cleanup, native overlap and performance remain operator gates, all PENDING.


## Dispatch follow-up against successful 778d K3 run (2026-09-14)

Continued from clean a5cdcf9f6e81cc9975321e37fc6100a143fa607d; the778d capacity
fix and owner status snapshot remain. All145 logical files in062632Z-1634 were
resolved through inventory.logical_paths and checked for bytes/SHA256. Export COMPLETE,
first code0, no missing records; all three original capacity/execution/measurement/
cleanup and shared joint outputs PASS. Single-window rates46.5544/67.6436/71.3875
remain valid, not a repeatability claim. Request/native rejoining reproduces zero
ordinary/recovery cross-request overlap and6860.926–6904.569ms parent-eager overlap.
Old reports and archives were not rewritten. The new derivative is explicitly
labelled offline analysis and retains source object hashes/indices.

The remaining known bottleneck is Target scheduler's two repeated resident prompt
normalization/JSON/digest passes. Source and111 ordinary measured dispatches place
84.087ms in scheduler within99.055ms claim→Target. Inclusive prefix/block-record
work accounts for37.945ms; full block auditing7.911ms. Other scheduler work remains
38.231ms, and14.969ms lies outside scheduler. Those are same-lane boundary differences,
not a sum with Draft/TP/RPC/wait times. Neither the necessary admission fence nor a
fast mailbox-only verify-start ACK waits for the other request's complete proposal.
The detailed A/version2 rejection → correction/seed → two extensions → A/version3
READY and B claim/Target timeline is in k3-pipeline-778d-observations.json.

The new production-chain regression substitutes only hardware/transport boundaries:
real proposer factory, controller, authoritative owner, actual resident/FixedBatch/
Ping scheduler and verify-start/end adapter. A mixed acceptance/rejection batch
recovers K3 while B is dispatched; both Target/Draft completion orders and eager
on/off are event-controlled. Existing complete-output, EOS, budget, refill, repeated
rejection, stale/duplicate claim/feedback, ownership and drain suites remain enabled.
Both real full pool audits must still execute, with no repeat prompt digest. Restoring
the old PoolScheduler physical_rows method in the same regression yields4 failures
at that structural assertion, not a wall-time threshold. Its execution interleaving
assertions themselves still pass with held GPU work: this rules out inventing an
old whole-proposal wait. Earlier owner-only regressions missed post-claim CPU work.

New report regressions collect actual owner/backend records, attach explicitly
synthetic native GPU endpoints, run strict joins, serialize/export/re-read and obtain
the same results. Missing/duplicated TP rank, missing native request IDs or clock,
wrong proposal/version and Draft parent dependency all make integrity INCOMPLETE and
overlap null. An other-request interval in the same home remains cross-request,
not cross-home; duplicate TP intervals do not double its union. Bounded prompt-proof
spans record real setup/measurement scopes; mutated prompts/types/frontiers/blocks
and resident disappearance remain rejected. Existing run→report→qualify→archive
capacity/correctness/performance regressions still cover all three K3 modes.

Development failures: the new fixture initially used the wrong eager constructor
keyword and work-step helper, and held only the pure extension purpose, missing a
real mixed eager/recovery batch. It now drives the actual interfaces and both physical
purposes. A compatibility command initially named a nonexistent report-test file
(collection exit4, no executed tests); the real device-contract selection replaced
that filename. None required assertion/timeout relaxation or protocol changes.
Fixture logging now uses its absolute temporary point path instead of writing test
receipts in the checkout. No production logging behavior changed.

Full local suite:2535 passed,3 existing platform/GPU skips. The final owner-vs-Target
claim and transport landmark additions then passed27 targeted regressions; both
Python3.9/3.12 passed269 relevant cases and their final27-case selection. Ruff,
compileall on3.9/3.11/3.12,364 Python files parsed as3.9,21 Bash scripts and whitespace
checks pass. The new implementation's remote CI is queried after the ordinary push;
the baseline a5cdcf9's eight checks were all SUCCESS. Separate final CI status is
reported at delivery, never inferred from local or baseline success.
New GPU correctness, cross-request pipeline, native overlap and performance are
PENDING. The same fixed three-mode foreground runner and single verified archive
remain the next gate; it does not retry or extend the experiment.

## Resident normalization follow-up (2026-09-14)

The baseline 082831Z-1891 archive and read-only analysis are reused, not rerun or
rewritten. Implementation begins at the clean PR head8b8a2d5 above execution8a3aca0.
The pinned vLLM0.25.1 CPU source export passes the inventory SHA256 audit for commit
752a3a504485790a2e8491cacbb35c137339ad34; its Scheduler.schedule begins at line396.
The new stock span surrounds that actual base call (including installed predicate
callbacks), not the ResidentSetupScheduler wrapper.

`test_k3_resident_schedule.py` drives actual PingPrePostScheduler/FixedBatch/Pool/
ResidentSetupScheduler/identity binding with a CPU stock allocator. The structural
regression first failed on the old path's repeated fixed_identity int conversions.
After the change every live current token is normalized once in the call-local row;
restoring the old `_binding_input` tuple path makes the assertion fail again. Three
modes compare old/new identity maps, selected IDs, candidates, query positions,
resident decisions and all360 admission records; timestamps are the only removed
record field. The original CheckpointJsonl append still executes in this comparison.

Negative regressions cover current-prompt mutations, table/internal mismatch,
reverse-map alias, absent binding, an invalid generated suffix in an unselected
resident, stale claim, block conflict, illegal frontier and missing resident. They
must fail before stock allocation. Further comparisons cover unbound and
non-prefix-free matching, integer-coercible inputs versus invalid inputs, prefix
growth, finished/removed rows, refill and retained historical bindings. The existing
full-output K3, EOS/budget, feedback-order, repeated rejection, duplicate claim/commit,
resource-release, Serial idle and event-controlled cross-request suites remain active.

New phase fields are collected by the actual production scheduler, serialized,
qualified by the resident phase validator and formal diagnostic qualifier, exported
with the real content-addressed total-pack exporter, reread and qualified again.
This scheduler-only harness deliberately lacks GPU startup evidence: its unrelated
GPU/device qualification stays incomplete, never a fabricated PASS. Existing real
fixed_runtime drive/serialize/qualify/export tests still cover three-mode capacity,
non-scan correctness and scan performance including strict probe and native binding.
Missing phase/work metadata is null/INCOMPLETE and fails the new diagnostic contract;
it does not make execution or output correctness fail. Existing old reports lack
these new subphases and must retain their original qualification, not be overwritten
by replaying a new diagnostic requirement against them.

Native overlap tests preserve exact request/version/TP association, recompute recovery
physical-interval unions and conservative fraction bounds, and count overlapping
Target steps. Mixed roles/TP copies do not double the numerator. Missing native
records yield null fractions/counts, not zero or OBSERVED. Six new trace rows per
call use existing phase budgets; no recorder threshold or timeout was enlarged.

Development failures before final verification: the first new test fixture used an
incomplete Proposal constructor (fixed to production schema); the intended structural
assertion then failed on the unchanged implementation. An unclosed parenthesis in
new report code caused collection failure and was corrected. Ruff found import,
closure binding and line-length issues, corrected without changing test assertions.
These are local development errors, not diagnoses of remote CI failures. Final test
counts and actual remote CI status are appended at delivery.

New GPU correctness, native pipeline/recovery coverage and performance remain PENDING.
READY publication and post_prepost/fence order are unchanged and require a separate
future proposal if the new GPU timeline still shows an exposed recovery bottleneck.

Local verification: the full suite with the pinned source audit completed2556 passed/
3 existing skips. Python3.9 and3.12 each passed the341-case related selection with
`PYTHONPATH=$PWD/src`. Their first invocation passed340 and failed the static-entry
subprocess because those external virtualenvs did not have SpecRhythm installed
(ModuleNotFoundError); only the invocation environment was corrected. First logs
were retained under `/tmp/sr-k3-resident-py39.log` and `...-py312.log`; configured
results are in separate logs. No product/test/timeout workaround was applied.

Final review moved the mode-dependent normalizer selection outside the per-request
loop (once per binding call, without storing token state). The full suite and affected
compatibility cases are rerun on that final source. Ruff, compileall3.9/3.11/3.12,
369 tracked/new Python files parsed with3.9 grammar,21 tracked Bash scripts and
`git diff --check` pass. The server-only GPU stages remain unexecuted locally.

After the final normalizer-factory adjustment, Python3.9 and3.12 each passed38
affected production scheduler/identity/dispatch cases. The final source's full pinned-source run completed2556 passed/3 existing skips
in349.15s. The delivery commit pins execution
`d007dce448bd2a7d510172222ae166d7bb6f299e`, without changing execution code.
Pinned entry/runbook first-error/single-archive regression:15 passed. Git object
checks confirm this exact execution SHA contains the runner, normalizer path and
strict resident diagnostic gate, with the original three-mode budgets.
The implementation's push/PR CI source contracts and Python3.11 contracts have
reported SUCCESS; Python3.9/3.12 full jobs are still running at this documentation
snapshot. Delivery CI is reported separately after its normal push, not assumed PASS.
