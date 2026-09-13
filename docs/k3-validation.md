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
