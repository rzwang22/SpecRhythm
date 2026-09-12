# Serial-eager B16：A-only 与 A+B 独立复测

执行此文件交付副本中的完整 Bash 块。A-only 为
`bbf12170118961a88244ae97af949eeee7d6028a`；A+B 的 Git 模板使用
`__AB_FULL_SHA__`，交付副本会填入实际完整 SHA。两提交各建一个全新 root，
分别运行相同 resident360、B16、K4、两轮预热、30 秒连续窗口、samples=None、
buffered-live、bound-prefix、setup 900 秒、drain 60 秒的 Serial 对照及 Serial-eager。
不复用或覆盖旧结果，不比较跨次运行 GPU UUID。

每个提交先运行现有 capacity/prefill 检查和物理 GPU correctness，再依次运行
Serial、Serial-eager。正确性检查使用 3 个请求；A-only 保持原逐请求检查，A+B
把同一七轮 case 集合偏移后组成混合批次，并检验真实批量 admission、step、
父轮修复及独立 KV 参考。这个 correctness 构造在性能测量之外，不贡献 Target
观察计数；自然 EOS/尾部仍可能使构造 NOT_OBSERVED，不强行延长生成。

任一点的执行、正确性、测量或 cleanup 不通过，子 Bash 的 ERR trap 保存失败
证据并退出，阻止后续点/提交；外层 if 捕获退出码，交互终端保持打开。只有两个
指定提交和两个 B16 模式，不自动扩大扫描或追加重跑。日志、模型、采样、identity
和测量策略沿用原设置，所有窗口等待均计入墙钟。

```bash
export SR_EAGER_A_ONLY=bbf12170118961a88244ae97af949eeee7d6028a
export SR_EAGER_AB=__AB_FULL_SHA__
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export SR_FIXED_S1=/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469
export SR_EAGER_RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)-$$"
if bash <<'BASH'
set -Eeuo pipefail
cd /root/autodl-tmp/src/SpecRhythm
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1
unset SR_FIXED_ROOT
on_failure() {
  rc=$?; trap - ERR; set +e
  printf 'Stopped at first error: rc=%s, commit=%s\n' "$rc" "${SR_FIXED_COMMIT:-unset}"
  if [[ -n "${SR_FIXED_ROOT:-}" && -d "$SR_FIXED_ROOT" ]]; then
    "$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check status --root "$SR_FIXED_ROOT"
    "$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check errors --root "$SR_FIXED_ROOT"
    bash scripts/run_decode_scan.sh status
    bash scripts/run_decode_scan.sh errors
    bash scripts/run_decode_scan.sh summary
    bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-failure-small.tar.gz"
    printf 'Preserve failed root and bundle: %s\n' "$SR_FIXED_ROOT"
  fi
  exit "$rc"
}
trap on_failure ERR
[[ "$SR_EAGER_A_ONLY" =~ ^[0-9a-f]{40}$ && "$SR_EAGER_AB" =~ ^[0-9a-f]{40}$ ]]
test -z "$(git -c core.fsmonitor=false status --porcelain)"
git fetch origin codex/rolling-eager-v0.1
git merge-base --is-ancestor "$SR_EAGER_A_ONLY" "$SR_EAGER_AB"
for SR_FIXED_COMMIT in "$SR_EAGER_A_ONLY" "$SR_EAGER_AB"; do
  export SR_FIXED_COMMIT
  export SR_FIXED_ROOT="/root/autodl-tmp/SpecRhythm-data/results/rolling-eager/serial-repair-B16-${SR_FIXED_COMMIT:0:12}-${SR_EAGER_RUN_TAG}"
  test ! -e "$SR_FIXED_ROOT"
  test -z "$(git -c core.fsmonitor=false status --porcelain)"
  git checkout --detach "$SR_FIXED_COMMIT"
  printf 'NEW ROOT: %s\n' "$SR_FIXED_ROOT"
test "$(git rev-parse HEAD)" = "$SR_FIXED_COMMIT"
bash scripts/run_decode_scan.sh prepare --s1 "$SR_FIXED_S1" \
  --observation buffered-live --identity-matching bound-prefix \
  --selection-seed 1666 --warmup-steps 2 --window-seconds 30 --repeats 1 \
  --setup-timeout 900 --drain-timeout 60
"$SR_FIXED_PYTHON" - <<'PY'
import json, os, pathlib
root = pathlib.Path(os.environ['SR_FIXED_ROOT'])
scan = json.loads((root / 'scan-config.json').read_text())
assert scan['execution']['git_commit'] == os.environ['SR_FIXED_COMMIT']
assert scan['workload_sha256'] == 'cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4'
assert scan['pool_size'] == len(set(scan['selection']['request_ids'])) == 360
assert scan['options']['samples'] is None
assert scan['options']['window_seconds'] == 30 and scan['options']['repeats'] == 1
assert scan['options']['warmup_steps'] == 2
assert scan['options']['setup_timeout'] == 900 and scan['options']['drain_timeout'] == 60
assert scan['options']['observation'] == 'buffered-live'
assert scan['options']['identity_matching'] == 'bound-prefix'
print(json.dumps({'root': str(root), 'workload_sha256': scan['workload_sha256'],
                  'options': scan['options']}, indent=2))
PY
# Fresh physical capacity/prefill checks, including eager's extra speculative KV.
bash scripts/run_decode_scan.sh capacity --single-point --batch 16 --mode serial
bash scripts/run_decode_scan.sh capacity --single-point --batch 16 --mode serial-eager
# Actual Draft worker + physical page/frontier checks; no CPU executor and no Target model.
"$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check run \
  --root "$SR_FIXED_ROOT" --request-count 3 --timeout 900 --drain-timeout 60
"$SR_FIXED_PYTHON" - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ['SR_FIXED_ROOT']) / 'rolling-eager-gpu-check'
state = json.loads((p / 'state.json').read_text())
result = json.loads((p / 'result.json').read_text())
assert state['status'] == 'COMPLETE' and state['cleanup_status'] == 'PASS'
assert not state['remaining_owned_pids'] and state['exit_code'] == 0
assert result['valid'] and result['physical_gpu_correctness'] == 'PASS'
assert result['cleanup_status'] == 'PASS'
assert result['backend']['eager_gpu_live_work'] == 0
assert not result['backend']['eager_gpu_write_inflight']
assert result['backend']['worker_resources']['blocks_allocated'] == result['backend']['worker_resources']['blocks_freed']
assert all(row['exact'] for row in result['reference_comparisons'])
print(json.dumps({'controlled_physical_cases': result['constructed_cases'],
                  'real_Target_event_observations': result['target_observed_events'],
                  'GPU_overlap': result['gpu_overlap']}, indent=2))
PY
# Serial must qualify first. Any failure stops before serial-eager decode.
bash scripts/run_decode_scan.sh run --single-point --batch 16 --mode serial
"$SR_FIXED_PYTHON" - <<'PY'
import os, pathlib
from specrhythm.serving.fixed_artifacts import point_reports
root = pathlib.Path(os.environ['SR_FIXED_ROOT'])
rows = [r for r in point_reports(root) if not r['point'].get('probe')]
assert len(rows) == 1 and rows[0]['mode'] == 'serial' and rows[0]['batch'] == 16
r = rows[0]
assert all(r[k] == 'PASS' for k in ('capacity_status','execution_status','measurement_status','cleanup_status'))
assert r['formal_comparison_eligible'] and r['effective_exit_code'] == 0
assert r['stop_reason'] == 'time_budget' and r['measured_window_ms'] >= 30000
assert r['actual_target_batch']['min'] == r['actual_target_batch']['max'] == 16
print('Serial B16 execution, measurement and cleanup qualified; proceed to serial-eager B16.')
PY
bash scripts/run_decode_scan.sh run --single-point --batch 16 --mode serial-eager
"$SR_FIXED_PYTHON" - <<'PY'
import json, os, pathlib
from specrhythm.serving.fixed_artifacts import point_reports
root = pathlib.Path(os.environ['SR_FIXED_ROOT'])
rows = [r for r in point_reports(root) if not r['point'].get('probe')]
assert len(rows) == 2 and {r['mode'] for r in rows} == {'serial', 'serial-eager'}
assert len({r['workload_sha256'] for r in rows}) == 1
assert {r['git_commit'] for r in rows} == {os.environ['SR_FIXED_COMMIT']}
for r in rows:
    assert r['batch'] == 16 and r['effective_exit_code'] == 0
    assert all(r[k] == 'PASS' for k in ('capacity_status','execution_status','measurement_status','cleanup_status'))
    assert r['formal_comparison_eligible'] and r['stop_reason'] == 'time_budget'
    assert r['measured_window_ms'] >= 30000 and r['target_steps'] > 0
    assert r['actual_target_batch']['min'] == r['actual_target_batch']['max'] == 16
    life = json.loads((pathlib.Path(r['artifact']) / 'process-lifecycle.json').read_text())
    assert life['cleanup_valid'] and life['owned_cleanup_completed'] and not life['remaining_owned_pids']
eager = next(r for r in rows if r['mode'] == 'serial-eager')['rolling_eager']
c = eager['window_counters']
assert c['started'] > 0, 'serial-eager window did not actually start eager work'
assert eager['cleanup_status'] == 'PASS' and eager['owner_stopped']
assert eager['pending_work'] in (0, [])
life = eager['lifetime_counters']
assert life['completed'] <= life['started'] <= life['admissions']
assert life['accepted_promoted_candidates'] <= life['verified_promoted_candidates']
print(json.dumps({'acceptance': {
    'execution_and_cleanup': 'PASS', 'actual_eager_start': 'OBSERVED',
    'event_classes': eager['observation'],
    'GPU_overlap': eager['GPU_overlap'],
    'comparison_conditions': 'PASS',
    'window_counters': c, 'window_unhidden_wait_ns': eager['window_unhidden_wait_ns'],
}, 'comparison': [{k:r[k] for k in ('mode','artifact','measured_window_ms',
    'committed_window_tokens','decode_throughput_tok_s')} for r in rows]}, indent=2))
PY
"$SR_FIXED_PYTHON" -m specrhythm.serving.eager_retest_report \
  --root "$SR_FIXED_ROOT" --expected-commit "$SR_FIXED_COMMIT" \
  --output "${SR_FIXED_ROOT}-attribution.json"
bash scripts/run_decode_scan.sh summary
bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-complete-small.tar.gz"
parent=$(dirname "$SR_FIXED_ROOT")
tar -C "$parent" -czf "${SR_FIXED_ROOT}-complete-with-attribution.tar.gz" \
  "$(basename "$SR_FIXED_ROOT")-complete-small.tar.gz" \
  "$(basename "$SR_FIXED_ROOT")-attribution.json"
sha256sum "${SR_FIXED_ROOT}-complete-with-attribution.tar.gz"
printf 'Return: %s\n' "${SR_FIXED_ROOT}-complete-with-attribution.tar.gz"
done
BASH
then
  printf 'A-only and A+B B16 runs complete; no additional tests started.\n'
else
  rc=$?
  printf 'Stopped (rc=%s); later tests were not started and this terminal stays open.\n' "$rc"
fi
```

返回两个新 `*-complete-with-attribution.tar.gz`；若失败，返回该 root 的
`*-failure-small.tar.gz`，保留已经完成的前一提交结果。完整原始报告仍留在各自
新 root，输出包含原资格摘要和独立的窗口归因 JSON。

归因报告包含吞吐、每步产出/墙钟、真实 device overlap、enqueue-to-first-GPU
时间上下界、每步 audit/prefix 次数与 union、按用途 forward/B 分布、修复 GPU
union 和 normal/recovery host union。时间字段注明其边界；JSON/prefix、RPC/wait
及内部成本不相加。CPU 测试通过不代表 GPU 重叠或性能收益已经确认。
