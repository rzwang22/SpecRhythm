# PingPong prepost3：固定 B16 两点与一个上传包

仅由用户在 AutoDL 前台执行。本轮本地没有连接服务器、启动模型或运行 GPU。
新模式的 GPU correctness、native overlap、performance 均为 **PENDING**。
执行基础与源码映射见 [设计](pingpong-prepost3-design.md)，CPU 证据见
[验证记录](pingpong-prepost3-validation.md)。

## 固定资源和顺序

只运行 `pingpong-prepost3/runtime` 与 `pingpong-eager-prepost3/runtime`。
两点使用同一完整执行 SHA；入口 `scripts/run_ping_prepost_b16.sh <full-SHA>`。
交付固定入口使用独立 detached worktree，不 reset、stash 或修改保存的 checkout。

1. 分别创建两个新 root，准备同一冻结360请求，逐点检查真实模型容量。首错停止。
2. 一个共享联合 correctness：Target-only 和两个新模式，通过生产启动器执行16个原始请求、
   单独记录的最多32输出 token fixture。比较每请求完整输出，要求自然结束及资源释放。
   新 eager 必须覆盖实际 P1/P4 同批 Target、连续 A→B→A、拒绝 P1 后重新 P4。
   采样结果正确但所需轨迹未出现时，记作 coverage 不完整，不宣称联合 correctness 通过。
3. 运行控制点，再运行 eager 点；每点 execution、measurement、cleanup 与 diagnostic integrity
   均通过后才继续。记录第一次非零码；不会自动重跑、扩展 grid 或改变模型。
4. 生成两点比较，成功或失败都尝试导出一个总包。原始详细日志留在服务器。

性能配置：resident360，全体真实 prefill 后，total active16、初始 home A/B=8/8、Target ceiling8；
GPU0 Draft、GPU1/2 一个 TP2 Target。固定 K4，lookahead≤3、公共步骤1，seed1666；
两轮预热=4次实际非空 Target admission，连续30秒 decode、无 sample12 截止；setup900s/drain60s。
`runtime` audit、`buffered-live`、`bound-prefix`、light phased trace 相同。
旧 Serial 单窗口87.98/100.59 tok/s 不加入这次配对比值。

冻结 workload SHA256 必须是
`cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4`。
模型、dtype、采样及有效引擎设置沿用 S1 产物与现有运行时检查；不同点不比较 GPU UUID 必须相同。
相邻验证机会可以重复请求并混合 home；实际 Target B 可为1..8，不能虚填到8。
空 poll、owner/RPC、refill、等待及完整输出提交均留在窗口墙钟中；不增加预热/反馈全局 barrier。

## 默认服务器路径

- repository：`/root/autodl-tmp/src/SpecRhythm`，固定启动器可用 `SR_PING_REPO` 指定。
- Python：`/root/autodl-tmp/envs/specrhythm-phase4-vllm-0.25.1/bin/python3.11`，`SR_FIXED_PYTHON`。
- S1：`/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469`，`SR_FIXED_S1`。
- results：`/root/autodl-tmp/SpecRhythm-data/results/rolling-eager`，`SR_PING_RESULTS`。

运行创建 `pingpong-prepost3-delivery-<UTC>-<pid>/`，内部 `points/<mode>/` 是两点独立 root，
`joint/` 只执行并存储一次联合 correctness。`SR_PING_RUN_TAG` 可显式给出新的唯一 tag；已有目录或包会被拒绝，
绝不覆盖历史结果。以后授权重复只需新 tag；本轮不自动重复或交错。

## 唯一上传文件与完整性

本次入口修复首次容量 probe 的 `dual_uuid_query` schema/consumer 不一致；旧运行仍保持
FAILED/INVALID、capacity/cleanup PASS、进程码0及外层码1，未进入 correctness/performance。
各模式证据契约及真实生产链复现见 [验证记录](pingpong-prepost3-validation.md)。
新prepost3模式用已有startup/final worker快照和原生forward/request/proposal关联，Dual专属计数
明确不适用；旧pingpong仍执行原live-query规则。容量阶段要求TP2/Draft隔离、真实准备和释放、
零decode verification；正式点要求非零真实forward关联。必要字段缺失不会默认通过。

只上传终端唯一 `UPLOAD ONLY:` 所指向的：

```text
/root/autodl-tmp/SpecRhythm-data/results/rolling-eager/pingpong-prepost3-delivery-<tag>.tar.gz
```

包内 `inventory.json` 保存逻辑相对路径、原文件/导出对象 SHA256、字节数、缺失项和导出错误。
`logical_paths` 将文件映射到 `objects/<sha256>.json`；共享文件只存一份。
包含 comparison、共享完整输出 correctness、两点结果/metadata/status、生命周期与清理、原生设备事件、
有界 host/owner/协议/Target sampling 记录、first-failure 与包内 export status。
runtime投影保留 `target_final_memory`、`target_requests_final`、`diagnostic_drain`；同包
actual-capacity的startup rows及Draft/Target native identity是设备契约的原始来源。
清单声明保留/省略字段、源hash；缺最终快照时明确required_fields_missing，导出INCOMPLETE，禁止猜填。
可将这三个逻辑文件读取后传入 `device_contract.qualify_prepost()` 复算相同设备契约。
服务器另留 `export-status.json`（含最终包 SHA256）和 `export.log`，不要求另行上传。

单原文件≤512MiB，最多192个允许文件，唯一投影内容总量≤1GiB。大型 runtime/backend 只移除明确列出的
无关顶层摘要，保留完整有界事件，不截取 measurement 前缀；投影字段表及原 SHA 均在清单内。
setup8192/warmup4096/measurement65536/drain8192 的原始预算、保留行和 dropped 数一起导出。
完整性继续严格：丢行、缺失归因、原生事件关联缺失不写 COMPLETE。
损坏 point metadata 以原始字节保留，另外标记导出不完整；不会因解析失败丢掉全部失败包。

`export_status=COMPLETE` 仅表示允许文件按声明导出；未启动的后续点仍标 MISSING，不能据此判运行成功。
运行资格、diagnostic integrity、coverage、performance conclusion 分开。第一次执行/诊断错误不被后续导出错误替换。
`first-failure.json` 分别记录stage、failure_layer、原始effective_exit_code、qualification/cleanup及
mode/rank/field/expected原始错误；进程码0与report_qualification失败可以同时成立。
若磁盘或环境故障使总包本身无法生成，终端明确打印 `EXPORT FAILED` 和保留的源目录，不冒充成功上传。

## 读取结果

| 要核对的内容 | 证据 |
|---|---|
| 原始吞吐、窗口、提交量、steps、tokens/step、step分布 | `comparison.json`、两点 `*-audit-report.json` 和原始 light-summary |
| 实际 B、P1/P4/截短长度、有效 Target 输入位置 | `pingpong.cycles`、Target rows 和 sampling；capacity active16/ceiling8 |
| 普通、lookahead、公共及混合物理 forward | `per_role_window_forwards` 的 count/B/event时间与 raw physical bindings |
| 接受、拒绝、复用、废弃、跨 home、容量/依赖延期 | `pingpong.cycles.requests`、outcomes、generated/retained/discarded、owner admission事件 |
| ready→admission、feedback→ready、暴露lookahead等待 | request/version关联；最后没有下一 ready 时为 null |
| owner queue、enqueue/dequeue→首GPU、反馈发布链 | 每请求 queue/native 锚点，`execution_path` 的采样→发送→接收→owner |
| Target等Draft、owner/反馈与日志成本 | exclusive process/thread lanes、outside_complete_steps；未归因保留 |
| 原生重叠 | `native_overlap` 全体并集、ordinary_extension、eager_lookahead、common 的上下界 |

混合物理 batch 可属于多个 role，role event累计与 overlap **不得相加**；总 overlap 只算设备区间并集。
measured-parent 的后续结算可能落在下一 Target step 或 drain；按 launch 落窗的 GPU forward 集合独立统计。
不同集合不强行逐项相等。`complete_step_wall_ms` 使用原始 `target_steps.start_ns/end_ns`，即本次引擎step及输出提交；
之前的admission/无ready轮询并不在该span中，单列为 `outside_complete_steps_ms`，仍全部计入吞吐窗口。
缺失不是 ZERO，CPU或host并发不是GPU重叠；inclusive时间不能从吞吐中扣掉。
单窗口只检查机制，不足以宣称稳定几个百分点收益。未来更大 Target 的覆盖能力仍为假设。

## 固定前台命令

完整执行 SHA：`429956febe6d731291c1a3c0ee0857337a3c54c2`。包含设备契约及单包投影修复；
两个容量点、联合GPU correctness及两个性能点均执行此SHA。后续交付提交仅固定启动器和文档，
不改变src执行路径。旧bc908be运行及其原FAILED/INVALID记录不复用或覆盖。
不要 source 严格子脚本到交互 shell。父 shell 使用 `if … then … else … fi` 接收失败，始终保留交互终端。

复制到服务器前台：

```bash
if bash <<'SR_PING_CHILD'
set -Eeuo pipefail
FINAL_SHA=429956febe6d731291c1a3c0ee0857337a3c54c2
REPO="${SR_PING_REPO:-/root/autodl-tmp/src/SpecRhythm}"
git -C "$REPO" fetch origin codex/rolling-eager-v0.1
git -C "$REPO" cat-file -e "${FINAL_SHA}^{commit}"
RUN_TREE="${REPO}-ping-prepost3-${FINAL_SHA:0:12}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
git -C "$REPO" worktree add --detach "$RUN_TREE" "$FINAL_SHA"
export SR_EXEC_REPO="$RUN_TREE"
bash "$RUN_TREE/scripts/run_ping_prepost_b16.sh" "$FINAL_SHA"
SR_PING_CHILD
then
  printf 'PingPong pair finished. Return only the single archive printed by the runner.\n'
else
  rc=$?
  printf 'PingPong stopped (original rc=%s); later points stopped. Interactive terminal remains open.\n' "$rc"
  # Preserve the interactive parent; original failure is printed above.
fi
```

若checkout已包含交付提交，也可用 `if bash scripts/run_ping_prepost_b16_pinned.sh; then :; else rc=$?; printf 'stopped rc=%s\n' "$rc"; fi`。
此入口只尝试获取代码并创建新worktree；fetch/worktree尚未成功时没有GPU运行或结果目录，
会保留该setup失败码。runner开始后按上述规则尝试单包导出。
