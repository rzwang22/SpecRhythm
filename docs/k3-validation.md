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
