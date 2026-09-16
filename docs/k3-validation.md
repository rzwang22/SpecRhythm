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

## Four-mode comparison and framing, 2026-09-15

Read-only source: `pingpong-k3-delivery-20260915T023730Z-1970.tar.gz`, execution
`d007dce448bd2a7d510172222ae166d7bb6f299e`; all145 logical files/106 unique objects
passed byte/SHA256 verification. [Independent derivative](k3-four-mode-baseline-d007.json)
retains exact metrics and interval definitions. All three old output comparisons have
16 requests/512 matching tokens. Old execution/measurement/cleanup PASS remains intact.
Old Serial was B8, not the new Serial B16 control. Its53.9663 tok/s, ordinary
PingPong75.4392 and eager81.4353 are historical single-window observations only.

Normal PingPong cross-request native overlap occurred on77/124 rounds with recovery
coverage3.7555–3.8835%; eager on1/134 with0.14413–0.14701%. The latter's own-parent
eager overlap7748.638–7793.639ms is a separate mechanism. These observations disprove
a blanket claim that B always waits for the entire A proposal. They do not show that
recovery is adequately hidden. No historical window or throughput is recomputed by
subtracting audit/CPU time.

Same-lane measured parent spans give normal claim→Target67.8754ms, scheduler48.1725ms,
binding8.9343ms, admission16.8085ms and actual stock call0.37465ms. The wrapper label
is not vLLM scheduler-body time. Within the admission parent, interval unions for
JSON average5.4446ms and fsync2.0483ms;9.3156ms remains unaccounted within that stage.
The first two360-row rounds each contain720 JSON encodings and360 appends; fsync
counts are2 and1. Across124 rounds the record total is44626 as requests finish.
JSON/checkpoint spans are nested and cannot be added to their parent. The remaining
work is not assigned wholesale to Python/GIL, locking or storage.

New tests exercise four production service factories, queued owner claims and actual
resident Target scheduler output, including one16-request Serial output (root+K3=64
query positions), B16 Draft extensions, and the existing event-controlled A/B and
Serial idle constraints. Pinned stock Target bookkeeping with an imperfect Draft
produces identical complete16-request/32-token outputs in all four modes and both
feedback orders. Other regressions retain rejection/recovery, EOS/tails, stale/duplicate
feedback, claims, resource retirement and candidate/commit conservation.

Actual schedule→installed logger tests require360 logical rows but360 canonical payload
encodings, both full KV audits, and byte-identical native/buffered log output. Replacing
the optimized production factory with the legacy path fails that structural check.
The old/new schedule comparison checks identity, selected candidates/positions,
decisions and all admission fields. Producer→serialize→qualify→single-package→reread
checks now include all four modes; missing fields still fail or stay explicitly missing.
Capacity v2 checks typed geometry plus required/reserved KV positions. v1 is an explicit
historical replay path, not an implicit fallback for new reports. The comparator allows
Serial Target16 versus PingPong8 and requires shared workload/options/models/numerics/
resources/configuration; runtime UUIDs are checked only within each run.

Development failures retained: initial tests assumed three modes/B8 and a two-home
Serial-eager fixture; these inputs were updated to the actual four-mode production
routing. The first full run had40 failures (2558 passes/3 skips) from reading optional
legacy manifest capacity outside qualification error handling in new display code.
It now reads the real runtime capacity inside the existing boundary;103 relevant
regressions then passed. The first Python3.12 related run had297 passes and one existing
supervisor test failure: the child reached its10s execution timeout before any logging
receipt or measurement snapshot, was SIGTERM-reaped, and owned cleanup completed.
No child traceback identified the blocking point. An isolated diagnostic run passed;
its initial cause remains unconfirmed. No assertion or timeout was relaxed.

New GPU capacity, output correctness, cleanup, cross-cohort native overlap and performance
remain PENDING until the operator returns the new four-point package. CPU ordering and
synthetic native intervals are not GPU overlap evidence.

Final local gate for the implementation: **2612 passed, 3 skipped** with the pinned
vLLM source audit enabled (349.13s). Affected Python3.9 and Python3.12 suites each:
**351 passed**. Ruff, compileall(src), Python3.9 grammar for216 source files, all21
tracked Bash scripts and git diff --check pass. The three skips are the opt-in CUDA test and two Linux exit-status/subreaper tests
on macOS, not disabled assertions. Initial failures above remain part
of the validation record; CI is reported separately after push.

The [CPU framing microbenchmark](k3-admission-cpu-benchmark.json) uses360 logical rows
from the real CPU resident fixture,12 alternating-order repetitions, existing observed
JSON wrappers and bounded logger append/finish. Every output byte matches. With actual
fsync replaced by a no-op in that isolated benchmark process, median times are5.4473ms
legacy and4.9817ms prepared. This is supplemental local CPU evidence, not server timing,
GPU overlap, durable-write performance or a projected throughput gain. Structural
regressions assert operation/record conservation rather than millisecond thresholds.


## 2026-09-15 entry ceiling repair (after fca2118)

Continued from clean PR HEAD `6a8072dd2ba4e478447666234fc4864ea8d48477`, retaining
`fca2118b89d21b11f18f5de43d69657e820087ab` and its four-mode configuration parent.
The old measurement heredoc used ceiling8 for every mode. Real CPU-substituted
`fixed_runtime.run/drive -> native timeline hooks -> serialization -> summarize ->
emit_result -> point_reports -> exact shell heredoc` reproduces rejection of both
Serial B16 reports, after all original qualification checks pass. Both Ping B8
reports pass that old gate. The old source heredoc is retained as a read-only
regression fixture, not as rewritten historical evidence.

Previous shell tests replaced the entire Python command and tested orchestration;
they never executed PY_MEASUREMENT. The new tests execute it in a separate actual
Python process. The CPU client fixture now honors the real admission payload's
capacity instead of silently slicing every mode to8. GPU computation remains a
replacement; actual report construction, native observation hooks and qualifiers
are production functions.

`k3_acceptance` validates mode/point, strictly typed geometry, reported active and
Target capacities, native TP request cardinality, and measured batch statistics.
A full-batch fixture checks one genuine native-record forward containing16 distinct
requests for each Serial mode, not a maximum of16. The real joint loop invokes this
check for **each** K3 mode before continuing; the older protocol coverage check on
the eager run remains. Full-batch proof is retained in joint run receipts and the
comparison, and runtime native records survive the existing single archive projection.
Missing receipt evidence fails the four-mode comparison. Partial batches are not
relabelled full, and raw owner underfill/lifecycle reasons remain available.

Audit of remaining paths: K3 runner configuration/capacity already uses geometry;
scan timing's geometry comparison is now strictly typed. The generic joint loop
receives four explicit modes and a K3-only callback. Legacy prepost3 `coverage()`
ceiling8 and the original default three-mode decode grid are not used by this
single-point K3 entry and remain unchanged. Export's `--k3` uses all four MODES;
comparison also rejects a report whose mode differs from its selected filename.
There is no GPU algorithm or measurement boundary change.

Development checks retained: the first test fixture omitted the production
`emit_result` layer and therefore lacked capacity_status; adding that real layer
exposed the intended two Serial assertion failures. The first implementation used
a nonexistent Target forward_id in a compact receipt; the real producer test
caught it, and receipts now reference actual rank/host_start_ns (plus native B/IDs).
An attempted bad Ping B16 performance fixture was correctly stopped even earlier
by K3Window; its negative gate test now injects actual non-scan B16 native records
into a derived test view, explicitly not an original run qualification.

Related Python3.9 and Python3.12 production-chain suites each passed157 tests.
Ruff, compileall on3.9/3.11/3.12,217 source-file Python3.9 AST checks and21 tracked
Bash scripts passed. Full suite:2645 passed,3 skipped in366.87s (opt-in CUDA and two macOS-inapplicable
Linux process cases). No assertion or timeout was relaxed. Pinned execution is
recorded in the runbook/project status. No AutoDL or GPU was invoked; new GPU qualification and performance PENDING.

Fixed execution for this entry repair: `c663cb30f470ed9bf24465d07af7ea3a6991c73e`. The later entry-only
commit points to this source and does not alter tested execution code.


## A100 state/read/report delivery regression scope (2026-09-15)

See k3-design's independent source-package hash/timeline and filesystem limits. New
CPU regressions run real owned subprocesses coordinated by FIFO rendezvous and inject
only file reads: single ENOENT/short JSON recovery, persistent missing/corrupt failure,
worker fatal logs during retry, unchanged absolute deadlines, invalid phase/identity/
types, first-publication total timeout, and pre-signal causal recording. Real delayed
Draft report publication after coordinator exit shares the existing deadline. Actual
K3 and Target-only service factories exercise backend shutdown and report completion.
Socketpair tests prove failed responses are not sent twice and preserve the original
validation error when its error response also fails.

Publication tests inject interrupted JSON serialization, fsync failure and expired
deadline. They inspect absent final files, actual partial bytes and FAILED markers;
then export/re-read the archive, verify hashes, and replay completion qualification
from original source digest plus receipt. Real inner Bash uses only substituted GPU
commands, then the production exporter/copy path handles malformed JSON, first rc23,
copy write failure and digest mismatch, preserves local files and emits one upload.
Geometry/nativeB16, report chain and prior interleaving suites remain in the gates.
No timing threshold proves an optimization, and no CPU test proves DPC consistency.

Development first-failure record: the initial fsync-injection test matched the full
path against IO_CONTEXT.name, which intentionally stores a basename; it therefore did
not inject. Matching the actual basename corrected the test (no production assertion
or budget was relaxed). The first59 affected tests and17 state-recovery tests passed.
Final full/version/source/Bash gates and actual new CI are recorded at delivery.

The first Python3.9 run found three failures in the new test's cross-process time
assertion: this macOS3.9 interpreter reports a per-process monotonic origin (the
parent decision was2.275s while the child's receipt was0.947s). This was a test
clock-domain error, not evidence of signal-before-decision. The regression now
proves ordering by the child's actual read of the decision in its signal handler
and the parent's real decision/signal timestamps; the synthetic downstream error
uses an explicitly injected shared clock. No production timestamp/budget or
assertion was relaxed. Linux server monotonic-domain acceptance remains GPU pending.

The initial full run found two real new attribution regressions: generic nonzero
coordinator-exit cleanup had displaced the already recorded worker failure in legacy
CLI reports. It is now classified as a cleanup reaction, not a competing earlier
cause; the original CLI assertions are kept. Two other failures were invocation
environment errors (`python: command not found`) in unchanged shell-helper tests;
the final gate activates the project's Python environment on PATH. Baseline push CI
34943435521 had a source-contract failure while its paired PR workflow succeeded;
that historical failure is investigated separately, never relabelled by local retries.

Historical CI detail: baseline push34943435521's `phase4b3-draft-source-contract`
failed `test_actual_claim_schedule_verify_enters_before_other_recovery_finishes[False-True]`
on the existing event wait in `test_k3_dispatch_chain.py:208`. The paired baseline PR
workflow34943441578 succeeded. This identifies the assertion, not the reason for
its missed coordination; no budget/timeout was changed and no CI retry was requested.

Local implementation gates: full pytest2677 passed/3 skipped in352.03s with the
project Python on PATH and pinned vLLM source enabled. The three skips are opt-in
CUDA and two Linux-only process cases on macOS. After the final small archive/legacy
identity/error-attribution adjustments: production sealing/pinned tests18 passed;
Python3.9 related suite131 passed plus28 final delta tests; Python3.12 related suite
131 passed plus46 final delta tests. Pinned source-only gate17 passed. Ruff,
compileall3.9/3.11/3.12,220 source-file Python3.9 AST, all21 Bash scripts and diff
checks passed. New GPU/DPC validation remains PENDING.

## A100 local Target-only shutdown omission (2026-09-15)

Read-only source bundle:
`pingpong-k3-delivery-a100-local-20260915T140620Z-761.tar.gz`, SHA256
`430a325a9c9522bebddeaf89aecd1e8f5bd3b4bd18bc0e93df199e52371f4eae`.
All146 logical entries were resolved through inventory.logical_paths and checked against
their byte counts and SHA256. Source execution `ae5be9a6b318b31931808fd1523064b33e2f0975`,
entry `1975061693a72f7ac1880da6b7cff82efc39fe9d`; branch was clean and equal to remote entry
HEAD before this repair. Existing commits/results remain intact; PR5 stays Draft.

Four capacity points: capacity/execution/cleanup PASS, measurement not applicable,
effective_exit_code0. All five included process supervisors report zero state-read errors.
The Target-only run `joint/target/runs/joint-correctness-target-B16-A-20260915T221651-707631036144378`
failed at `drain:draft_shutdown`: start707675077154214ns, deadline707735077154214ns,
end707675459606236ns. Elapsed0.382452022s, remaining59.617547978s. The first exception
was the report publisher's `TimeoutError` from a missing deadline, not actual expiry.
Target-only execution FAILED/measurement INVALID/cleanup FAILED/effective_exit_code1
remain historical. The joint suite and four performance points did not complete/start.
Archive COMPLETE is export integrity, not joint correctness PASS. `/tmp` is overlay
(physical backing unknown); persistence is DPC. The deadline bug does not implicate either
A100 speed or DPC consistency.

`tests/test_k3_shutdown_deadline.py` replaces only hardware and accept-loop orchestration:
local entry environment -> `fixed_draft.serve` actual factory -> `fixed_drain.settle`
payload construction -> real `UnixDraftClient` and AF_UNIX framed exchange -> actual server
`_handle/_dispatch` -> actual state machine (and K3 owner) -> physical CPU KV settlement,
report serialization/publication/read/size/hash receipt. The first test before production
changes failed exactly at `shutdown, payload={}`, returning
`RuntimeError: TimeoutError: final Draft report exceeded original drain deadline`.
No GPU was required. The previous factory test supplied its own correct shutdown deadline,
so it never exercised coordinator payload construction; that coverage gap is now explicit.

New tests cover all five modes, identical/repeated deadlines, conflict/missing/None/bool/
string/float/illegal integers, genuine expiry, pre-shutdown rejection and failure latching,
first-message failure with owner fault release and no new budget. Report build, serialization,
fsync, publication and verification failure/expiry cases retain phase and partial bytes,
and cannot qualify. RPC error -> first/secondary error summary -> real export -> inventory
reread preserves original error context and exit code. Existing runner/local-delivery tests
still exercise first-error stop, copy failure/local fallback, one UPLOAD ONLY, four geometry
and native B16 evidence rules. Hardware substitutes prove the CPU cleanup contract only;
GPU correctness/cleanup/native overlap/performance on the repaired SHA remain PENDING.

Local full suite:2756 passed/3 skipped in413.96s. Python3.9/3.12 related suites:
432 passed each. After making the owner fault notification an internal, non-RPC sentinel,
the final affected suite passed96 tests each on3.9/3.11/3.12. Ruff, three-version compileall,
221 source files parsed as Python3.9,21 Bash scripts, static four-mode interface and diff
checks passed. The three platform/opt-in skips are not GPU acceptance. Baseline entry CI
had all eight GitHub checks SUCCESS; new execution/entry CI is reported separately.

No assertions, timeouts, drain/setup budgets or diagnostic qualification were weakened.

## B64 scale validation (2026-09-16; GPU pending)

Based on verified A100 execution5b50529f3bd3617f29c60ee9de6bf7143f96449e,
retaining delivery6e07617b94b27f5bc9f1ddc126c1bc18a9592bbf. No server connection
or GPU execution in this implementation turn. Historical evidence remains unchanged.

New CPU regressions exercise actual B64 configuration→capacity_for→fixed_runtime.run
and drive→startup/native producer hooks→summarize/device/native qualification→archive
projection→replay. GPU execution is replaced at its hardware boundary; reports and
qualifiers are not mocked. Owner/controller/resident scheduler tests drive64 cached
seeds through two real backend extension calls and one64/32-request Target schedule,
retaining both physical pool audits. This proves CPU configuration/dispatch behavior,
not native GPU geometry or overlap.

Negative cases include implicit B64, inconsistent point/manifest/metadata, loaded
sequence/query deficit, KV deficit, duplicates, missing TP rank, aggregated small
forwards, invalid configuration types and missing geometry. Warmup partial batches
count actual opportunities. Actual fixture construction selects64 original requests,
caps outputs at32, preserves source bytes and seals the new manifest; complete-output
comparison checks IDs, tokens, lengths and finish reasons including EOS. Both entries
exercise first-error stop and one-package local/copy-failure fallback. Existing K3
rolling/rejection/EOS/accounting/deadline tests are retained.

During development the first expanded qualifier run found a missing `require`
import; a subsequent entry-test expansion found an unparameterized test-only `batch`
variable. Both were corrected at their source; no assertion, timeout or GPU budget
was weakened. One initial test command referenced a nonexistent filename and collected
no tests; the corrected affected selection passed216 tests. Validation totals and
commit/CI status are appended after the final checks.

The first full run recorded24 failures,2797 passes and19 skips.20 failures came
from an older hardware-substitution fixture omitting the manifest's real active_limit;
the other4 were its legacy routing error-text contract requiring B16 in the message.
The fixture now supplies the producer field, and mismatch errors name the configured
B16/B64. The focused capacity/startup/deadline suite then passed163 tests. This was
not a server or CI result. Subsequent source-audit-enabled full runs are recorded below.

Report adaptation also separates `cross_home_overlap_steps` and
`recovery_coverage_by_other_homes` from other-request overlap within the same home.
Their denominator is unchanged: unique physical recovery intervals clipped to the
actual window. A synthetic-native regression shows one cross-home step becoming
zero after home labels become equal while other-request overlap stays nonzero;
these injected intervals are CPU evidence tests, not GPU overlap measurements.

Final local validation: source-audit-enabled full pytest2845 passed/3 skipped
(381.09s); the preceding completed full run2843 passed/3 skipped (411.38s).
Python3.9 and3.12 related production-chain suites each369 passed, followed by157
final capacity/evidence/deadline tests each. Ruff, compileall on3.9/3.11/3.12,
221 source-file Python3.9 AST checks,22 Bash scripts and diff --check passed.
The final B64-only warmup metadata label is covered by those157-case checks.
No timeout/retention assertion was loosened. GPU capacity/correctness/cleanup,
physical B64/B32 geometry, overlap and throughput remain PENDING. CI is checked
on the pushed commit; local passes are not represented as remote CI or GPU passes.

Fixed delivery validation: execution `f6f67aa1e1d7aea2a81665ec628d0ae857c148ee`
was read with `git archive` into an isolated directory. Its production static
B64 contract passed all eight mode/role cases and its runner passed Bash syntax;
GPU status stayed NOT_RUN/PENDING. Entry/runner/local-delivery tests passed46;
the foreground entry tests additionally passed6 each on Python3.9/3.12. All23
repository Bash scripts passed syntax checks after adding the pinned B64 entry.

The implementation's [first push CI](https://github.com/rzwang22/SpecRhythm/actions/runs/35000694509/job/104488053926)
failed `test_real_owner_dispatch_initial_work_wait_and_window_drain[pingpong-True]`:
the original five-second deadline expired in `DiagnosticDualController.execute`
while waiting for a `diagnostic_settle` response. This is the legacy PingPong
resident360 scan fixture, not B64 execution. The test, fixed_drain and fixed_settle
have no diff against6e07617. The same SHA's independent PR job passed. Logs establish
the owner-response timeout but do not establish the cost that exhausted its budget;
CPU/IO scheduling is only a hypothesis. No timeout, assertion or implementation was
changed to turn this result green, and the workflow was not rerun. Other CI jobs were
still running at this snapshot; final delivery status is reported from GitHub.

## B64 performance-exploration policy validation (2026-09-16)

The default B64 runner executes all four capacities and all four performance points
without invoking Target-only or independent full-output runs. The explicit strict
option preserves that separate gate. Real runner subprocess tests cover both paths,
first-error stop, secondary export errors and exactly one upload path; B16 still
runs its original strict gate. New tests enter the real manifest/point builder and
CPU-substituted `fixed_runtime.drive`, serialize runtime, invoke scan summarize/emit,
execute the runner's actual measurement heredoc, export and replay native qualification.
They reject hidden smaller forwards, missing ranks, policy conflicts, frontier damage
and missing release evidence. Existing lifecycle/protocol suites remain required.

The comparison/export unit additionally combines those drive-produced geometry proofs
with explicitly synthetic host timing fixtures (not GPU evidence), exercises four-mode
comparison and archive replay, rejects forged equivalence PASS/missing proof/invalid
measurement, and rejects a missing plan falling back to strict. The real strict joint
loop and comparator receive CPU-produced output with an explicit worker-boundary fault;
two Serial modes, not the last eager mode, appear in aggregate failure sources.

Initial new-test failures exposed fixture omissions: a direct summary bypassed the
production `emit_result` cleanup fields, and the direct-drive export fixture lacked
`fixed_cli.run_point`'s `point.json` publication. Tests now execute the former and
serialize the actual selected point at the latter boundary. Production checks correctly
rejected both omissions; no assertion, timeout or runtime budget was relaxed.

The evidence CLI now accepts and checks the B64 runner's geometry argument (previously
passed by the runner but absent from its parser), plus the independent validation
profile. This fixes an entry/report contract mismatch, without changing collection.
CPU passes prove these contracts, not GPU output equivalence, geometry, overlap or
throughput. New server performance remains PENDING; default full-output check NOT_RUN.

Local validation environment incident: the first full run ended20 failed/2478 passed/
3 skipped/477 errors after the volume reached116MiB free; the first storage failures
were `OSError: [Errno 28] No space left on device`. The concurrent Python3.9 run also
failed on ENOSPC. Those logs are retained outside the repository. Only this task's
completed pytest scratch directories were removed, preserving history and raw server
archives. An attempted `tmp_path_retention_policy=failed` rerun was stopped after a
separate reproducible fixture collision: removing successful per-test directories
allows a reused name while a sibling `.tar.gz` still exists. The existing exporter
correctly rejected `new delivery archive required`. The45 deadline cases passed
under the original default retention. Final runs use fresh basetemp directories and
original retention, assertions, deadlines and budgets. These are diagnosed local
environment/test-artifact failures, not GPU evidence or resolved remote CI failures.

Final local suite result: **2873 passed,3 skipped,4 failed**. Failures were the three
`test_shared_shell_accepts_natural_teardown_and_removes_guard` modes (target/serial/dual,
15-second subprocess deadline) and the legacy prepost capacity first-error case
(20-second subprocess deadline). They are retained as failures, not declared fixed by
focused passes. Relevant Python3.9 and3.12 suites each passed270 tests; the new policy/
entry/scan-summary delta passed65 tests. Ruff, Python3.9 syntax (222 source files),
three-version compileall and all23 repository Bash scripts passed. No deadlines,
assertions or GPU budgets were changed. Remote CI is reported separately.

Timeout investigation: an independent public archive of unmodified3ac3752 passed the
three natural-teardown cases; its prepost capacity/failure case also passed when run
from that baseline checkout (0.58s). The current checkout's three natural-teardown
cases passed in isolation (21.46s total). These focused comparisons do not explain
the four original full-suite subprocess timeouts, so the full-suite failure remains
reported. The failed first mixed-working-directory probe is retained in local logs.
No unrelated lifecycle or legacy prepost implementation was modified.

Final fixed-entry/runner/local-delivery suite:46 passed. Additional explicit default/
strict profile forwarding cases and local single-package fallback:24 passed each on
Python3.9 and3.12. The pinned execution files (runner, policy, native acceptance and
evidence CLI) were byte-compared against899b54a6b58c0ea07046582c5a17934f630ac040;
the eight role/mode static capacity contracts passed, with GPU capacity still PENDING.
Implementation CI push35059823704 and PR35059827533: Python3.11 contract and pinned
Draft source-contract jobs passed; Python3.9/3.12 full jobs were in progress at this
documentation snapshot. Completion/failed CI states must be read from those runs.
