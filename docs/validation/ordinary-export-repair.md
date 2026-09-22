# Ordinary CPU delivery count-budget repair (2026-09-22)

Source execution: `944328263b02a915f397bcb03c8c743c71e9a56c`.
Source archive: `pingpong-k3-delivery-a100-ordinary-cpu-20260919T124903Z-391.tar.gz`.
SHA256: `6cc3d995847c7308b7f1c1e84695ba4f3e225432d80c2d99e9c1c0bc3759a733`.

## Confirmed failure boundary

All six performance points finished with execution/measurement/cleanup PASS and
per-point diagnostic COMPLETE. Paired bounded smoke PASS; complete Target-only
equivalence remains NOT_RUN. Runner exit0/stage complete is unchanged. Packaging
returned41, and the verified DPC copy succeeded. The old package remains INCOMPLETE.

The original inventory has545 candidate logical files,512 included,33 OMITTED_LIMIT
and six required-file MISSING entries (those six also occur among the33 omissions).
All512 included mappings /438 unique payloads passed local byte/hash verification.
Unique payload bytes2662426258 are below the declared4GiB byte budget. The trigger
was logical index512 in `ping_prepost_delivery.export`, not GPU/worker failure,
DPC corruption, or the unique payload count. Sorted traversal omitted the tail of
window5-P0, including its runtime, lifecycle, audit and evidence-status files.

The preceding implementation increased the experiment from two cases/four windows
to three capacity cases, four smoke runs and six windows, but retained the old512
file-count cap. CPU tests exercised individual report/export chains and first-error
export; they did not construct one complete thirteen-run archive. This repair adds
that aggregate-scale test. No historical results/statuses are rewritten.

## Bounded fix

`delivery_budget.count_budget` recognizes the exact ordinary CPU declaration:
P0/P1/S1, order P0,P1,S1,S1,P1,P0, four declared smokes including S0, B128 and
performance-exploration. It budgets64 orchestration files plus64 per planned run:
`64 + 13*64 = 896` logical files and896 unique payloads. The extra room covers
existing report/detail shards, lifecycle/diagnostic receipts and logs. This is
fixed from the declared plan, not enlarged until the current data fits; invalid
plans fail closed with the old cap. Legacy entries remain512. Single-file2GiB and
aggregate4GiB byte limits for this configuration remain unchanged.

The inventory records the profile, derivation, planned runs, observed candidate,
logical and unique counts. A limit omission now identifies its kind, bound and
observed value. Local delivery prints a small export summary before the final
receipt, so a file-count failure is no longer only an unexplained rc41.

Missing required files, malformed JSON, unfinished publication, changed files,
byte overflow and comparison failures retain strict checks. Corrupt bytes are
retained as before. A checksum-verified tar can still have incomplete evidence;
these statuses remain separate.

## Reexport without GPU

`ordinary_reexport` reads the retained raw directory, checks the original execution
SHA and outcome, and invokes the real exporter with source_read_only=True. It never
starts capacity/smoke/inference, rewrites reports or creates a missing comparison.
It creates a separate local output directory, then uses the existing verified DPC
copy/local fallback. Exactly one upload path is printed if a verified package exists.

The archive's reexport_provenance distinguishes the old GPU execution SHA from the
new exporter SHA, preserves the old delivery receipt (including rc41), and states
that no new GPU qualification occurred. Original execution failure wins over new
export/copy failures. Missing retained raw evidence still fails; the incomplete old
tar is not enough to reconstruct the33 omitted source files.

A fresh tar and copy target are used. Old archives, raw files, timestamps and run
outcomes remain unchanged. Copy failure is recorded in a new local fallback package;
archive write failure emits no fictitious upload path. No budget/deadline, algorithm,
geometry, sampling, READY, CPU execution policy or logging producer changes.

## CPU evidence

The real exporter regression creates545 files across13 declared runs; the old cap
reproduces33 omissions, while the new cap preserves every file, including more than
512 unique payloads, through archive reread and SHA256 validation. Additional cases
cover invalid plans,897-file overflow, single/total byte bounds, corrupt JSON,
required-file absence, old-budget compatibility, copy write/digest failures, original
exit23 preservation, interrupted archive writing, one upload, and read-only sources.
The real CLI parser/checks/export/copy also run with only checkout identity substituted.
These are CPU filesystem tests, not proof of the remote DPC reexport result.

Related Python3.9 report/local-delivery regressions:88 passed; final export and
fixed-entry set:28 passed. Python3.12 export-specific set:23 passed, plus five
fixed-entry cases. Ruff, compileall, Python3.9 grammar and Bash/diff checks pass.
The local auxiliary Python3.11 environment has no pytest; its contract CI installs
the declared test dependencies separately. Full-suite and final CI results are
reported at delivery without changing historical failure records.
