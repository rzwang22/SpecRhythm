# B16 启动关键路径：仅观测 → 修复

执行仓库中的 `scripts/run_eager_latency_b16.sh OBSERVATION_FULL_SHA FIX_FULL_SHA`。
两个完整 SHA 由本轮交付正文给出；脚本拒绝非 40 位 SHA、脏树、非祖先关系和已有 root。
脚本完整内容随 Git 提交获取，不依赖 Codex 本地路径或 `/tmp` 交付文件。

顺序固定：观测提交的 Serial-eager B16；修复提交的 Serial B16；修复提交的
Serial-eager B16。每个提交先 capacity 和三请求物理 GPU correctness，再性能；
不同提交独立新 root。观测提交保留 A+B 的执行逻辑；不复测 A-only。

所有正式点统一冻结 resident360 workload SHA
`cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4`、normal/eager K4、
同一 Draft GPU0 与 Target GPU1/2 TP=2、两轮预热、连续 30 秒、samples=None、
buffered-live/bound-prefix、setup 900s/drain 60s，并启用同一 bounded light causal。
prefill 在窗口前、drain 在窗口后；窗口内等待及 checkpoint 不扣除。没有详细 profiler。

将交付的两个 SHA 赋给变量后，前台入口如下（交付正文提供已填好版本）：

```bash
if bash <<'BASH'
set -Eeuo pipefail
cd /root/autodl-tmp/src/SpecRhythm
test -z "$(git -c core.fsmonitor=false status --porcelain)"
git fetch origin codex/rolling-eager-v0.1
git checkout --detach "$SR_LATENCY_FIX_SHA"
test "$(git rev-parse HEAD)" = "$SR_LATENCY_FIX_SHA"
bash scripts/run_eager_latency_b16.sh "$SR_LATENCY_OBSERVE_SHA" "$SR_LATENCY_FIX_SHA"
BASH
then
  printf 'Three B16 points complete; no further GPU tests started.\n'
else
  rc=$?
  printf 'Stopped (rc=%s); later GPU points were not started; terminal remains open.\n' "$rc"
fi
```

脚本在 checkout 前读完函数体，两版本分别运行对应代码。首次正确性、执行、测量、
清理或证据导出错误立即停止所有后续点，ERR trap 保留原退出码，尝试输出小包及
`*-failure-evidence.tar.gz`。原始 PASS/失败资格不改写；取证失败是独立诊断失败。

成功返回两个 `*-complete-evidence.tar.gz`。其中有原资格小包、v2 attribution JSON、
SVG 原生时间线及必要原始 runtime/backend/service 文件。每个原始文件最多 512 MiB，
总纳入文件最多 1 GiB、32 文件；`evidence-inventory.json` 标记 MISSING、OMITTED_LIMIT
和读前/读后稳定性。不会静默声称完整，也不会因缺文件自动增加 GPU 点。

脚本打印每个完整 root，形式为：
`/root/autodl-tmp/SpecRhythm-data/results/rolling-eager/serial-latency-B16-SHA12-UTC-TAG`。
需要单独查看或停止时，把**该次打印的完整 root** 赋给 `SR_FIXED_ROOT`，在同一 checkout
执行下列只作用于该 root 的命令。`status/errors/stop/bundle` 不会启动新性能点。

```bash
if bash <<'BASH'
set -Eeuo pipefail
cd /root/autodl-tmp/src/SpecRhythm
: "${SR_FIXED_ROOT:?set the exact root printed by the run}"
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export PYTHONPATH="$PWD/src"
bash scripts/run_decode_scan.sh status
bash scripts/run_decode_scan.sh errors
"$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check status --root "$SR_FIXED_ROOT"
"$SR_FIXED_PYTHON" -m specrhythm.continuation.gpu_check errors --root "$SR_FIXED_ROOT"
BASH
then :; else rc=$?; printf 'Inspection failed (rc=%s); terminal remains open.\n' "$rc"; fi
```

人工停止使用以下独立子 Bash。它先停止 correctness，再停止该 root 的 decode 任务，
即使一个停止命令失败也会尝试另一个及导出；最终保留第一个非零退出码。

```bash
if bash <<'BASH'
set -u
cd /root/autodl-tmp/src/SpecRhythm || exit $?
: "${SR_FIXED_ROOT:?set the exact root printed by the run}"
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
then :; else rc=$?; printf 'Stop/export finished with rc=%s; terminal remains open.\n' "$rc"; fi
```

仅导出（不停止、不启动 GPU），保留该 root 原始文件：

```bash
if bash <<'BASH'
set -Eeuo pipefail
cd /root/autodl-tmp/src/SpecRhythm
: "${SR_FIXED_ROOT:?set the exact root printed by the run}"
export SR_FIXED_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export PYTHONPATH="$PWD/src"
bash scripts/run_decode_scan.sh bundle --output "${SR_FIXED_ROOT}-manual-small.tar.gz"
"$SR_FIXED_PYTHON" -m specrhythm.serving.eager_latency_bundle \
  --root "$SR_FIXED_ROOT" --output "${SR_FIXED_ROOT}-manual-evidence.tar.gz"
BASH
then :; else rc=$?; printf 'Bundle failed (rc=%s); terminal remains open.\n' "$rc"; fi
```

所有 bundle 使用新路径，不覆盖既有包；若重发导出，选择新的后缀。正式测量不以
overlap>0 或速度超过 Serial 作为 PASS 条件。新 GPU 性能与物理重叠保持 PENDING。
