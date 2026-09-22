# B128 Target profile validation

This change introduces full/lean Target forensics controls at one execution commit.
It does not change GPU sampling, K3, Draft scheduling, READY or the validation policy.
The performance entry remains performance-exploration, output equivalence NOT_RUN.
Historical Serial output mismatches are unresolved and not requalified.

The production hook's return is ignored by the pinned runner. Lean prevents full
logits detach/CPU conversion, log-softmax, top-k and duplicate argmax for numerical
forensics. It retains the original small mapping/positions/sequence tensor reads
and CPU online evidence. Workload indexing occurs at actual Target worker startup;
current prompt and suffix normalization still occurs each forward through the
existing identity owner. No model/KV/version cache was added.

CPU tests exercise capture → raw JSONL → compact collection → profile-aware
structural validator; full continues to pass the original full validator and lean
is deliberately rejected as full numerical evidence. A poison logits tensor and
forbidden numeric helper prove the production hook takes the lean path. Tests also
run pinned stock bookkeeping without optional diagnostic fields, alias/prompt/owner
invalidation, missing required mapping/prefix/position evidence, real startup/drive
report generation → scan summary → archive projection → re-read qualification.
Four-mode geometry/TP checks remain enabled. Existing controlled owner/scheduler/
adapter interleaving runs under both profiles, including early/late feedback.

Initial new-test failures were fixture defects: the small fixture contains 5 rows,
not 12; the report fixture omitted the real S2 manifest/workload/diagnostic environment
because it previously never invoked capture initialization. These were repaired at
the fixture/hardware boundary; production acceptance was not weakened.

The only new scheduling fields are sparse observations at select/live eligibility/
claim. No third dispatch configuration is advertised because no further removable
barrier was established. CPU safe interleaving does not prove physical GPU overlap.
Tests and CI results are recorded below before delivery. GPU capacity, runtime
qualification, cleanup, feedback improvement, overlap and performance: PENDING.

## Local verification record

- Initial full suite: 3,029 passed / 14 failed / 3 skipped (3,046 collected).
  Eleven new tests inherited serving-manifest or numerical-plan environment from
  preceding configuration tests. The declared tiny/S2 fixture now explicitly owns
  those inputs. Production workload/profile validation was not weakened. A rerun
  with deliberately poisoned parent S1/plan variables passed all26 profile tests.
- The other three initial failures are unchanged legacy tests:
  `test_execution_pair_runbook` capacity-serial and evidence-serial cases reached
  their existing20s subprocess limits; `test_phase4_process_cleanup` shell-return
  case reached its existing15s limit. No timeout/assertion changes. Their precise
  cause is unresolved; source/entry paths are unchanged by this patch. Passing
  reruns or CI do not retroactively explain those failures.
- Related production-path set:54 passed; Python3.9 and3.12 related sets:60 passed
  each, including pinned bookkeeping, real owner/scheduler, profile report/export
  and repeated entry. These preceded the final fixture-isolation regression above.
- Ruff PASS; Python3.9 AST230 source files PASS; Bash18 scripts PASS; compileall
  and diff checks performed. Final full rerun and remote CI recorded at delivery.
- No A100 connection, GPU run or DPC filesystem experiment was performed.


## Final local implementation run and fixed launcher

- Final full implementation suite: **3039 passed, 5 failed, 3 skipped** (3047
  collected, 1162.604s). All26 new profile/partition tests passed. The five failures
  are `TimeoutExpired`, not a diagnosed GPU/protocol/profile error:
  - `test_execution_pair_runbook` capacity-serial case: existing20s subprocess cap.
  - `test_k3_b64::test_B64_real_run_capacity_summary_export` performance cases for
    serial-k3, serial-eager-k3 and pingpong-k3: existing20s child Python cap.
    These reach the real measurement-entry subprocess after constructing and
    qualifying runtime evidence. The extracted `PY_MEASUREMENT` body is byte-for-byte
    unchanged from35aeb26; the tests and acceptance subprocess helper are unchanged.
  - `test_phase4_dual::test_target_failure_terminates_draft_without_unbounded_wait`:
    existing5s shell subprocess cap; test and cleanup behavior were not modified.
  Exact reasons for these local elapsed-time failures remain unresolved. Their
  stderr/stdout did not identify a profile assertion or import position. Source
  equivalence and passing related tests do not establish their root causes. No
  assertion/budget relaxation, automatic retry or repeated-run success claim.
- Final focused production-chain + fixed-entry tests on **Python3.9:40/40 PASS**,
  **Python3.12:40/40 PASS**. Separate fixed-entry test:6/6 PASS. Tests invoke actual
  Bash, replace only checkout/external GPU runner, check original23 exit code,
  first-stop/single-upload behavior, default lean and explicit baseline, inherited
  flags reset, and unsupported configuration rejection.
- Ruff PASS; Python3.9/3.12 compileall PASS; Python3.9 grammar230 sources PASS;
  Bash19 scripts plus both runbook bootstrap blocks PASS; `git diff --check` PASS.
- Execution commit4ff1170397db0393caa2d8a43c2089a8e602ddfd contains all profile,
  runtime/qualification/report and repeated-run changes. Fixed-entry commit
  7f5d03002d2e24788ec9bedb792d974d8b55ec3f pins exactly that execution commit.
  Verified the execution commit contains `run_k3_b128.sh` and `target_profile.py`.
- No new GPU result. Full/lean GPU capacity, input evidence, cleanup, feedback
  latency, overlap and throughput remain PENDING; output equivalence NOT_RUN.

## Remote CI observation

Observed 2026-09-17T13:55:08.471352+00:00. Both push and pull-request checks are included.

- Execution: 4 ('completed', 'success'), 4 ('in_progress', None).
  Runs: [35229089308](https://github.com/rzwang22/SpecRhythm/actions/runs/35229089308), [35229095059](https://github.com/rzwang22/SpecRhythm/actions/runs/35229095059).
- Pinned entry: 4 ('completed', 'success'), 4 ('in_progress', None).
  Runs: [35229419381](https://github.com/rzwang22/SpecRhythm/actions/runs/35229419381), [35229424248](https://github.com/rzwang22/SpecRhythm/actions/runs/35229424248).

Python3.11 serving-contract and pinned-source-contract checks completed successfully
on both triggers for each commit. Python3.9/3.12 full-suite jobs were still running;
this is not an all-green CI claim. Later CI outcomes require a fresh observation.
The subsequent documentation-only delivery commit has its own checks.
