# Local Mac, GitHub, and remote GPU workflow

Use GitHub as the source of truth. The Mac owns code editing and cheap tests; the GPU server owns
profiling and engine experiments.

## Local loop

~~~bash
git switch -c codex/<short-topic>
python -m pytest
ruff check .
git add <explicit paths>
git commit -m "<terse change>"
git push -u origin HEAD
~~~

Keep pure strategy logic independent of engine APIs. Engine adapters should later live in separate
packages such as integrations/vllm/ or integrations/sglang/ and consume the same PolicySnapshot
and StepPlan contract.

## Remote loop

~~~bash
ssh <gpu-host-alias>
git clone https://github.com/rzwang22/SpecRhythm.git
cd SpecRhythm
git fetch origin
git switch <branch>
python -m pip install '.[dev]'
pytest
~~~

Run profilers and serving benchmarks on the remote machine. Store large raw logs outside Git or in
an artifact store. Bring back only compact, reviewed JSON/CSV calibration profiles with a manifest
that records commit SHA, environment, engine version, model revisions, command, and seed.

Never edit the same branch independently on both machines. Create a remote-only branch for urgent
server fixes, push it, and merge or cherry-pick it locally through review.

For Phase 3, use the exact detached-commit, environment-probe, TP-check, external-result, resume,
and manifest commands in [phase3-gpu-runbook.md](phase3-gpu-runbook.md). GPU-only tests must remain
explicitly marked and skipped in the Mac/CPU suite; a CPU dry-run is never a latency result.
