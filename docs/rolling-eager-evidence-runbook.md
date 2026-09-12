# Serial-eager B16：只读导出原运行证据

此流程只读取已结束的两个生产 point。原结果仍属于提交
`4a6725b054990e47b7e9c4cf63f38b995d029856`，继续保留 PASS 和有效负性能结果。
取证工具的提交与被测提交分别记录；本文件在 Git 中使用提交模板，交付副本会把
`__FINAL_DIAG_SHA__` 填成取证代码的完整 SHA。

不需要模型加载或 GPU 运行。输出在原 root 之外，使用全新文件名；输入单文件上限
128 MiB、总读取上限 512 MiB、JSONL 最多 100000 条、最多 8 个周期、每周期每流最多
512 条细节。完整 JSON 中的 measured-window 统计先按窗口选择，再应用记录上限。
MISSING/LIMIT/PARTIAL 描述取证完整性，不修改原结果资格。缺少原日志未记录的
dequeue/proposal ID 时不会补造时间戳或身份。

在服务器交互终端直接复制下面整个块。严格模式、错误处理、退出只位于子 Bash；
外层 `if` 接收首个失败状态并保留交互终端。只运行离线 export 和压缩输出文件，
不运行新的 capacity、decode、summary 或 scan。

```bash
export SR_EAGER_DIAG_COMMIT=__FINAL_DIAG_SHA__
export SR_EAGER_DIAG_PYTHON=/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11
export SR_EAGER_SOURCE_ROOT=/root/autodl-tmp/SpecRhythm-data/results/rolling-eager/serial-B16-4a6725b05499-20260912T113101Z-1469
export SR_EAGER_EXPORT="${SR_EAGER_SOURCE_ROOT}-evidence-${SR_EAGER_DIAG_COMMIT:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$.json"
if bash <<'BASH'
set -Eeuo pipefail
on_failure() {
  rc=$?; trap - ERR; set +e
  printf 'Read-only export stopped at first error (rc=%s). Old root unchanged.\n' "$rc"
  printf 'Source: %s\nNew output: %s\n' "$SR_EAGER_SOURCE_ROOT" "$SR_EAGER_EXPORT"
  exit "$rc"
}
trap on_failure ERR
cd /root/autodl-tmp/src/SpecRhythm
[[ "$SR_EAGER_DIAG_COMMIT" =~ ^[0-9a-f]{40}$ ]]
test -z "$(git -c core.fsmonitor=false status --porcelain)"
git fetch origin codex/rolling-eager-v0.1
git checkout --detach "$SR_EAGER_DIAG_COMMIT"
test "$(git rev-parse HEAD)" = "$SR_EAGER_DIAG_COMMIT"
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1
test -d "$SR_EAGER_SOURCE_ROOT"
test ! -e "$SR_EAGER_EXPORT"
"$SR_EAGER_DIAG_PYTHON" -m specrhythm.serving.eager_evidence_export \
  --root "$SR_EAGER_SOURCE_ROOT" --output "$SR_EAGER_EXPORT" \
  --max-file-bytes 134217728 --max-total-bytes 536870912 \
  --max-jsonl-rows 100000 --selected-cycles 8
"$SR_EAGER_DIAG_PYTHON" - "$SR_EAGER_EXPORT" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
assert d['source_commit'] == '4a6725b054990e47b7e9c4cf63f38b995d029856'
assert d['read_only_source'] and not d['inference_started']
assert d['original_result_statuses_unchanged']
for mode, point in d['points'].items():
    old = point['original_result']
    assert old['valid'] and old['formal_comparison_eligible']
    assert all(old[k] == 'PASS' for k in ('execution_status','measurement_status','cleanup_status'))
    print(json.dumps({
        'mode': mode,
        'original_throughput_tok_s': old['decode_throughput_tok_s'],
        'device_status': {k:v['status'] for k,v in point['devices'].items()},
        'overlap': point['recomputed_eager_overlap'],
        'selected_cycles': len(point['selected_cycles']),
        'missing_files': point['missing_files'],
    }, ensure_ascii=False))
print(json.dumps({'bytes_read': d['read_bytes'], 'limited_or_missing': [
    {'path': r['path'], 'status': r['status']}
    for r in d['inventory'] if r['status'] != 'READ'
]}, ensure_ascii=False))
PY
archive="${SR_EAGER_EXPORT%.json}.tar.gz"
test ! -e "$archive"
tar -C "$(dirname "$SR_EAGER_EXPORT")" -czf "$archive" "$(basename "$SR_EAGER_EXPORT")"
sha256sum "$archive"
printf 'Return this new evidence archive: %s\n' "$archive"
BASH
then
  printf 'Offline evidence export complete; no GPU tests started.\n'
else
  rc=$?
  printf 'Export stopped (rc=%s); this interactive terminal remains open.\n' "$rc"
fi
```

返回新 `*-evidence-*.tar.gz` 即可。若导出中有 LIMIT/PARTIAL/MISSING，也请保留并
返回该文件：现有 inventory、文件大小和读取范围可决定下一次更窄的只读导出，
不需要先重跑 GPU。旧 root 中完整的原始文件继续保留。
