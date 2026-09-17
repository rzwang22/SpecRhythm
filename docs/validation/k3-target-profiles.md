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
