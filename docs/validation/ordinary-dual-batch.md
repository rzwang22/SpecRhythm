# Ordinary double-batch implementation and evidence boundary

Development starts at clean local/remote `2e43b8b39f889470a3d3ebd35f34ceb7b429c6c8`;
no earlier commits or historical reports are replaced. PR #5 remains Draft.

## Source and dependencies

Reference fixed MineDraft commit
[`272c8556797495fe9c9f6a3008337bcd5cc2d031`](https://github.com/electron-shaders/MineDraft/tree/272c8556797495fe9c9f6a3008337bcd5cc2d031).
Read the ordinary double-batch branch of
[`ParallelSpecDecodeWorker`](https://github.com/electron-shaders/MineDraft/blob/272c8556797495fe9c9f6a3008337bcd5cc2d031/minedraft/plugin/spec_decode/spec_decode_worker.py)
and [`ParallelMQAScorer.start_score_proposals/score_proposals`](https://github.com/electron-shaders/MineDraft/blob/272c8556797495fe9c9f6a3008337bcd5cc2d031/minedraft/plugin/spec_decode/mqa_scorer.py).
MineDraft starts scoring one batch, drafts the other, then collects scoring. Its
vLLM0.9.2 worker/scheduler are not transplanted into pinned vLLM0.25.1.

SpecRhythm already has the independent Draft owner and authoritative token-boundary
claims. `K3Owner.status` already reads a snapshot. Queued feedback is distinguished
from physical settlement; unified normal/recovery work already shares batches.
An event-controlled real owner/controller/scheduler test already permits B Target
while A Draft is unfinished. A synchronous sole `engine.step()` caller therefore
does not itself imply GPU serialization. No new Target thread/future is necessary
for this dependency and none is added. The improvement being tested is reduced
CPU work on that existing overlap-capable path, not a newly invented pipeline.

## Production changes

|Boundary|Reference|Opt-in `dual-batch`|Checks retained|
|---|---|---|---|
|Owner admission RPC|Repeat full model provenance in each claimed proposal|Same `machine.admit`, lossless call-local model table and bounded claims|Owner queue, safe token boundary, prefix/version, exact claim and home/capacity|
|Coordinator control publish|Compare full nested packet and recursively copy it|Compare fresh scalar lifecycle state; retain read-only returned admission identity; compact atomic command|Every lifecycle/phase/deadline transition, publication errors and live population|
|Target prepare consumers|Repeated control file decode in nested schedulers|One fresh read for the synchronous call, nested ContextVar view; worker hook has its own scope|All live KV/prefix/binding/position checks before forward; both complete resident audits|
|Feedback hook|Construct next-proposal request list that `pp_feedback` ignores|Skip only the unused list/hashes via explicit proposer hook|Synchronization rows, acceptance/correction, terminal handling, authoritative feedback and settlement|

The wire table compares provenance **contents** within one message, supports
heterogeneous entries, retains all request fields and rejects invalid references.
It is not a long-lived model/prefix/KV cache. The scope ends on success or exception,
is thread isolated, and never spans Target calls. The coordinator is the sole
control publisher and cannot replace admission while its synchronous Target call
runs. Fresh `ServingClock.control()` rows contain scalars; publisher snapshots do
not reference mutable clock rows. Publication failure does not advance its state.
The Target scheduler and Draft owner remain the authoritative live validators.

Both resident snapshots/block audits, complete prompt comparison, all admission
records and existing deferred diagnostic/online-control policies remain. This
leaves real scheduler, IPC, physical settlement and audit costs; it does not prove
these are small or name residual time GIL. `post_prepost` and READY publication
order are unchanged. No batch reduction, wait-for-both-groups, sleep or extra GPU
sync is introduced. EOS, empty homes, refill and finite drain use existing paths.

## Verification

New tests drive real K3Owner/controller JSON wire, the pinned stock scheduler and
Target proposer feedback chain. They check exactly one command read within a
Target call, two live KV audits, full records, duplicate claim rejection, current
scope invalidation/exception/thread isolation, lossless provenance and ordinary-only
guard. Production feedback tests run repeated rejection then successful K3 cycles
and assert no ignored next-proposal request is built. Event-controlled A/B tests
hold an actual Draft extension while B dispatches; no millisecond speed assertion.

Report tests use real `fixed_runtime.drive` construction/serialization, capacity
and native geometry qualification, archive projection and reread. CPU substitutes
replace model work, not the report/qualifier. The real local Bash entry test fails
on a missing sealed S1 before GPU access, checks first-error stop, archive digest,
unique upload and absence of later runs. Secondary report failure cannot replace
the execution code. Old Serial/eager/protocol and deadline suites remain required.

Local environment observations: macOS Desktop files sometimes returned OS error60
while importing source, including an initial report regression and full collection.
A byte-identical snapshot in `/tmp` avoids that filesystem condition. The new Python
3.12 venv initially lacked an editable package installation, causing subprocess
module lookup failures; it was installed before final validation. These are recorded
separately from code assertions; neither assertions nor timeout budgets were widened.
The first completed full run was 3136 passed / 4 environment failures / 7 skipped;
its failures were subprocess module lookup and a missing shell `python`. After
editable installation and setting the venv PATH, final full Python3.12 validation
was **3142 passed / 7 skipped**. Python3.9 production/protocol/report/deadline
regressions: **169 passed / 4 ordinary-only skips**. All 8 new entry regressions
and 6 command regressions passed. Ruff, compileall, Python3.9 grammar, all entry
Bash syntax and diff checks passed. Actual CI is reported separately at delivery.

GPU unavailable on this Mac. The configured A100 SSH endpoint refused connection;
no GPU or remote environment changes were performed. CPU tests prove dependency
safety, not device intersection or performance. Historical output differences are
unresolved. Eager remains on its existing path until ordinary reference/new GPU
results justify a separate integration decision.

Final delivery additionally tests the immutable launcher (success and error23),
then the missing-comparison export guard: the exporter must retain MISSING/INVALID,
never invent an empty-mode success if runner comparison publication failed.
Post-full-suite delivery/export regressions: **32 passed on Python3.12**, **8 passed
on Python3.9** for the final entry/report chain. These supplement the full suite
above; they are not additional GPU evidence. The first targeted export invocation
had a misspelled test filename (no tests collected), corrected before these results.
