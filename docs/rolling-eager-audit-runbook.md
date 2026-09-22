# 同一最终 SHA：四点 Draft 审计对照

本轮唯一 GPU 入口是仓库中的
[`scripts/run_audit_layers_b16.sh`](../scripts/run_audit_layers_b16.sh)。它接受一个完整最终
SHA，不 checkout 历史 A-only，不运行 grid。交付正文给出填好 SHA 的完整可复制命令；
也可先将该 SHA `export SR_AUDIT_FINAL_SHA=...`，执行下方同一个前台入口。

```bash
if bash <<'BASH'
set -Eeuo pipefail
cd /root/autodl-tmp/src/SpecRhythm
: "${SR_AUDIT_FINAL_SHA:?export the full final SHA from this delivery}"
[[ "$SR_AUDIT_FINAL_SHA" =~ ^[0-9a-f]{40}$ ]]
test -z "$(git -c core.fsmonitor=false status --porcelain)"
git fetch origin codex/rolling-eager-v0.1
git checkout --detach "$SR_AUDIT_FINAL_SHA"
test "$(git rev-parse HEAD)" = "$SR_AUDIT_FINAL_SHA"
bash scripts/run_audit_layers_b16.sh "$SR_AUDIT_FINAL_SHA"
BASH
then
  printf 'Four B16 points complete; return the four evidence packages and comparison JSON.\n'
else
  rc=$?
  printf 'Stopped (rc=%s); later points were not started; interactive terminal remains open.\n' "$rc"
fi
```

顺序固定，且每点都有独立的新 root：

| 点 | 模式 | Draft audit | Target audit |
| --- | --- | --- | --- |
| 1 | Serial | full | 原完整检查 |
| 2 | Serial-eager | full | 原完整检查 |
| 3 | Serial | runtime | 原完整检查 |
| 4 | Serial-eager | runtime | 原完整检查 |

每点先同配置 capacity，再三请求物理 Draft GPU correctness（显式相同 audit mode），
然后唯一一个 B16 性能点。每点均验证冻结 workload SHA
`cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4`，resident360，
normal/eager K4，Draft GPU0、Target GPU1/2 TP2，全部真实 prefill 后两轮 warmup，
连续 30 秒 decode，samples=None，setup 900s/drain 60s，buffered-live/bound-prefix，
light causal。S1 默认源和 Python 环境的完整路径在脚本中固定；不下载新模型。
未改变 Target sampling/EOS、静态资格、feedback priority、batch gate 或窗口内等待。

root 格式为
`/root/autodl-tmp/SpecRhythm-data/results/rolling-eager/serial-audit-B16-SHA12-MODE-AUDIT-UTC-PID`。
`NEW ROOT` 在启动前打印。已有 root、脏工作树、SHA 不符均停止，不覆盖旧文件。
四点始终使用同一 checkout；检查每次运行内 Draft/Target UUID 分离，不比较跨次 UUID。

首个 capacity、correctness、执行、测量、cleanup、证据完整性或导出失败都停止后续点。
ERR handler 保留首个退出码，尝试 status/errors、小包及 `*-failure-evidence.tar.gz`，
导出失败不能覆盖首错。supervisor 保留原错误和 owned PID/start identity；不会把
kill 后无进程当作正常清理。所有严格模式/exit 都在子 Bash，外层 if 使交互终端保持打开。
CPU shell 回归实际执行该控制流，并逐点注入错误验证不会继续后续点。

## 返回文件和解读

成功后返回四个 `*-complete-evidence.tar.gz` 和一个
`audit-four-point-SHA12-UTC-PID.json`，脚本打印每个完整路径。不要只返回 summary。
每点的 `*-audit-report.json` 在 root 外新建，同时被证据包纳入；四点报告可从这四个
文件完全离线生成，不需要重新运行 GPU：

```bash
if bash <<'BASH'
set -Eeuo pipefail
cd /root/autodl-tmp/src/SpecRhythm
: "${SR_AUDIT_FINAL_SHA:?full final SHA required}"
: "${SR_AUDIT_REPORT_SERIAL_FULL:?path to existing point report}"
: "${SR_AUDIT_REPORT_EAGER_FULL:?path to existing point report}"
: "${SR_AUDIT_REPORT_SERIAL_RUNTIME:?path to existing point report}"
: "${SR_AUDIT_REPORT_EAGER_RUNTIME:?path to existing point report}"
: "${SR_AUDIT_COMPARISON_OUTPUT:?new output path outside existing roots}"
export PYTHONPATH="$PWD/src"
/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11 \
  -m specrhythm.serving.audit_layer_report --expected-commit "$SR_AUDIT_FINAL_SHA" \
  --reports "$SR_AUDIT_REPORT_SERIAL_FULL" "$SR_AUDIT_REPORT_EAGER_FULL" \
    "$SR_AUDIT_REPORT_SERIAL_RUNTIME" "$SR_AUDIT_REPORT_EAGER_RUNTIME" \
  --output "$SR_AUDIT_COMPARISON_OUTPUT"
BASH
then :; else rc=$?; printf 'Offline report stopped (rc=%s); terminal remains open.\n' "$rc"; fi
```

单点/四点报告最多 8 MiB，不嵌入重复原始 host/TP 数组。证据包每文件最多 512 MiB、
总纳入 1 GiB、最多 32 文件；原始 native/host/backend 只保存一次。
`evidence-inventory.json` 逐文件记录 INCLUDED/MISSING/OMITTED、大小、hash、稳定性；
report 记录 trace mode/status/dropped、native coverage、audit mode/scope/version/完整边界。
任何缺失或截断保持可见，不以“包存在”代替完整性，不因此扩大测试。

原 qualification 三项 PASS、physical GPU correctness、证据完整性、实验配置资格和性能
结论分开。性能 gate 不要求 speedup、不要求 overlap>0。Serial 与 eager 只在相同
audit/observation 下比较。不能从 full 窗口扣除 audit 时间得到 runtime 吞吐估计。

延迟字段覆盖完整 step mean/P50/P90、Target 两 rank forward 与 TP union、enqueue/
dequeue/admission/首 forward，Target GPU 结束时已完成的 eager steps，GPU purpose/B、
parent repair、normal/recovery mixed batches、promotion/discard 和 batch wait；host
分类、prefix/hash、JSON/log、runtime checks、全池 scan/visit 与 affected checks 分列。
`exclusive_process_threads` 按 PID/thread 分区；各线程预算不能相加，RPC/等待/设备
时间与子 span 重叠。`unaccounted` 不解释为 GIL。观测自身 append 成本未隔离，model
forward hooks 不覆盖全部 GPU kernels。这些限制保留在每点报告中。

## 当前 root 的检查、受控停止和导出

需要检查时，先 `export SR_FIXED_ROOT='NEW ROOT 打印的完整路径'`。以下动作只操作该
root，不启动 GPU 测试。单独重导出时选新输出后缀，避免覆盖已有包。

```bash
if bash <<'BASH'
set -Eeuo pipefail
cd /root/autodl-tmp/src/SpecRhythm
: "${SR_FIXED_ROOT:?export the exact NEW ROOT path}"
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export PYTHONPATH="$PWD/src"
bash scripts/run_decode_scan.sh status
bash scripts/run_decode_scan.sh errors
"$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check status --root "$SR_FIXED_ROOT"
"$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check errors --root "$SR_FIXED_ROOT"
BASH
then :; else rc=$?; printf 'Inspection stopped (rc=%s); terminal remains open.\n' "$rc"; fi
```

人工停止时，同时尝试 correctness、decode 的 bounded stop 和证据导出，保留首错：

```bash
if bash <<'BASH'
set -u
cd /root/autodl-tmp/src/SpecRhythm || exit $?
: "${SR_FIXED_ROOT:?export the exact NEW ROOT path}"
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export PYTHONPATH="$PWD/src"
rc=0
"$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check stop --root "$SR_FIXED_ROOT" || rc=$?
bash scripts/run_decode_scan.sh stop || { next=$?; if (( rc == 0 )); then rc=$next; fi; }
bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-operator-small.tar.gz" || {
  next=$?; if (( rc == 0 )); then rc=$next; fi;
}
"$SR_FIXED_PYTHON" -m specrhythm.serving.eager_latency_bundle \
  --root "$SR_FIXED_ROOT" --output "${SR_FIXED_ROOT}-operator-evidence.tar.gz" || {
  next=$?; if (( rc == 0 )); then rc=$next; fi;
}
exit "$rc"
BASH
then :; else rc=$?; printf 'Stop/export rc=%s; terminal remains open.\n' "$rc"; fi
```

仅导出，沿用同一个 `SR_FIXED_ROOT`：

```bash
if bash <<'BASH'
set -Eeuo pipefail
cd /root/autodl-tmp/src/SpecRhythm
: "${SR_FIXED_ROOT:?export the exact NEW ROOT path}"
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export PYTHONPATH="$PWD/src"
bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-manual-small.tar.gz"
"$SR_FIXED_PYTHON" -m specrhythm.serving.eager_latency_bundle \
  --root "$SR_FIXED_ROOT" --output "${SR_FIXED_ROOT}-manual-evidence.tar.gz"
BASH
then :; else rc=$?; printf 'Export stopped (rc=%s); terminal remains open.\n' "$rc"; fi
```

本轮交付后停止。等待用户返回这四点的证据，再判断收益；不接着改批级恢复调度。
