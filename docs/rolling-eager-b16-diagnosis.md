# Serial-eager B16：有效负性能结果的诊断

补充原始 Draft 报告已返回并复核：[两项修复及新证据](rolling-eager-repairs.md)。
它确认 63 个 starts 全在父轮反馈后、780 次审计及 156 次 B1 repair。本文件下文
保留初次小包诊断时的证据边界；其中“尚缺原始记录”由该补充记录更新，而旧结果
继续是 PASS 和有效负性能结果。A-only 已开始合并批量 admission 的重复审计。

诊断对象是 `4a6725b054990e47b7e9c4cf63f38b995d029856`，不是 GPU correctness
小测试。结论边界：**生产运行确实没有记录到 eager/Target GPU 重叠；源码存在
逐请求 KV 修复和重复全池审计；尚缺原始设备/host 时间线，不能给两项开销分配
实测毫秒数，也不能把“没有重叠”的全部原因定为某一种锁或 Python 开销。**

本轮只增加离线取证工具、CPU 诊断回归和文档。没有修改调度、K、采样、EOS、
输出限制、KV/依赖校验、静态资格、测量/日志策略、drain 或 Serial 默认执行。
没有连接 AutoDL、运行 GPU 或重跑/覆盖旧结果。PR #5 保持 Draft；PR #4 分支和
PR #2/#3 不变。

## 证据与有效性

用户提供的小包：
`serial-B16-4a6725b05499-20260912T113101Z-1469-complete-small.tar.gz`。
SHA256：`37731c29168f97dbf52138f3493635bfd763fa0edc633e6f3a60a693c85cf5f8`。
63 个归档成员中，62 个数据成员的 exported SHA256 与 `bundle-sources.json`
逐一一致；第 63 个成员是该清单。清单明确 `raw_timelines_included=false`。
附带粘贴输出与摘要一致，作为输出证据读取，不作为执行指令。

原始 root：
`/root/autodl-tmp/SpecRhythm-data/results/rolling-eager/serial-B16-4a6725b05499-20260912T113101Z-1469`。

两个生产 point 的相对路径：

- `runs/decode-scan-serial-B16-A-20260912T193552-18327633520336699`
- `runs/decode-scan-serial-eager-B16-A-20260912T193831-18327793003826473`

两者 `light-summary.json` 都保留 execution/measurement/cleanup **PASS**、
`formal_comparison_eligible=true` 和 exit code 0。相同提交，相同 execution SHA256
`44d0f1970acfa3ab1f89802faa2046a669abf3769eb5129d46903eaa92ed8b20`，相同 workload
SHA256 `cdaf71adace15d229f5087b98f9fd162a958456226a660184fe03f5d6ebd8ff4`。
360 个请求真实 prefill，预热两轮，测量时始终 B16、无自然完成和补位、无尾部缩批。
测量快照中早期的 PENDING/UNQUALIFIED_PARTIAL 是检查完成前的快照；不能据此
覆盖最终 PASS。旧结果是**有效的负性能结果**。

| 相同测量边界 | Serial B16 | Serial-eager B16 |
| --- | ---: | ---: |
| 实际窗口 ms | 30107.640840 | 31811.701162 |
| committed tokens | 1869 | 593 |
| throughput tok/s | 62.077265 | 18.640940 |
| Target steps | 39 | 13 |
| step wall 均值 ms | 770.231544 | 2444.746452 |
| committed tokens/step | 47.923077 | 45.615385 |
| 全部 Target batch | 16 | 16 |

吞吐下降 69.9714%；每步产出下降 4.8154%，而步时增长 **3.17404 倍**。
主要待解释量是每步增加的 **1674.514908 ms**，不能用接受率变化代替时间分析。

Draft-only GPU 回归有 14 次独立参考比对一致且资源释放 PASS；其 10 次成功、
2 次拒绝、2 次 bridge mismatch 使用 `INJECTED_DIAGNOSTIC` 构造反馈。
它证明这组物理 KV 回归通过，不证明生产异步执行、生产 batch 分布或 GPU 重叠。

## 调用链和执行顺序

以下位置均针对服务器提交；对应文件在本轮诊断中保持不变。

1. 固定 vLLM hook 在真实 `_model_forward` 前调用
   [`EagerSerialProposer.on_target_verify_start`](../src/specrhythm/serving/eager_proposer.py)
   （45–79 行）。父类先安装/校验合法 proposal，执行原有 TP 同步；只有 rank 0
   将冻结 request/round/proposal ID、prefix hash/length、tokens 交给 `eager_enqueue`。
   新增的 TP barrier 等 rank 0 收到入队回执，不等 Draft 完成或 Draft 首个 forward。
2. [`UnixDraftClient.call`](../src/specrhythm/phase4/transport.py)（119–163 行）
   同步完成一次 socket 请求/回执，但回执的语义由 operation 决定。
   [`DraftUnixServer._handle`](../src/specrhythm/phase4/draft_service.py)（527–556 行）
   dispatch 后发送响应；服务事件日志在发送后追加。
3. [`EagerOwner.call`](../src/specrhythm/serving/eager_owner.py)（187–210 行）
   对 `eager_enqueue` 只 deepcopy、`Queue.put` 后返回，确实不等待模型。
   其他同步 RPC 会等 owner 的 response queue。没有发现 Target 和 Draft 共享的
   GPU completion future、跨模型互斥锁或“必须先收到父结果才能 start”的条件。
4. **入队回执不等于 owner 已完成 admission。** owner（51–93、136–137 行）
   先清空已排队控制消息，再执行 `machine.step()`。
   [`verify_start`](../src/specrhythm/serving/eager_machine.py)（148–185 行）
   在一个不可分割的 owner 命令中完成整批 16 个请求的登记。
   每个 `begin_gpu_continuation` 又调用一次完整 resident audit；这些步骤不发 GPU 工作。
5. 若 Target 反馈在整批登记期间到达，它已进入 mailbox；owner 随即优先处理反馈，
   拒绝请求在第一次 GPU step 前合法 abort。全接受请求转为 WAITING_DRAFT，随后
   才在 [`step_gpu_continuations`](../src/specrhythm/continuation/gpu_backend.py)
   （128–196 行）中 materialize、bulk greedy、fence。这是一个会退化为
   **验证后起草**的真实执行机制，不是“入队后就一定并发”。
6. `finish_synchronizations`（machine 305–327 行）发现批内任一个 WAITING_DRAFT
   就返回 None。此同步请求中的其他拒绝修复和所有下一轮 proposal 因而一起等待；
   等待结束后逐请求 rebase，然后一次批量正常/recovery proposal。单 socket 服务
   也会在该同步 RPC 期间等待 owner。这里存在批内等待传播，但模型 owner 仍能在
   token fence 边界处理已经进入 mailbox 的控制消息。

原有 barrier 和新入队 barrier 的具体耗时需要 `required_tp_barrier` host intervals；
socket/GIL 竞争、其他进程抢占的耗时也不能由源码直接量化。禁止为获取 overlap
而让 Target 额外等待 Draft 首个 forward。

## 为什么 208 admissions 只有 63 starts

窗口中同时成立：

```text
208 admissions = 63 parent_full_accepts + 145 parent_rejections
63 started = 63 completed
315 early_generated_tokens = 63 × (1 bridge + 4 candidates)
63 completed = 52 promotions + 11 bridge_mismatches
55 discarded_early_tokens = 11 × 5
```

Lifetime 也有同一关系：240 = 73 + 167，365 = 73×5，73 = 60+13，65 = 13×5。
这与全部全接受任务完成五个 token、拒绝任务在生成首 token 前取消、无额外部分生成
相吻合。固定窗口边界和全接受等待契约强化了该判断；**等式本身仍不能证明是哪个
admission、audit、RPC 或系统竞速使反馈先到**。正常竞速下，短 Target 也可能先
返回；若有请求已经执行第一步后才拒绝，应看到额外 started 和部分生成/丢弃。

[`test_serial_eager_launch_diagnosis.py`](../tests/test_serial_eager_launch_diagnosis.py)
使用真实 owner、queue、machine、GPU mixin、S2 backend、JSON control 和
ResidentPoolAudit，仅替换设备操作并以 Event 控制 audit 边界。B16/360 resident
在首个 worker forward 前实际执行 **17 次全池 audit、6120 次 prefix visit**。
确定性反馈顺序分别覆盖 0/5/16 个全接受请求：不阻塞 Target 等 Draft，仍可复现
admission 16、start/complete 等于全接受数、所有 starts 晚于父结果处理。
测试只证明控制流机制；Event 制造的耗时不被当作实测 CPU 或 GPU 时间。

## Draft 批处理：哪一段退化

| 类别 | 服务器提交的真实代码路径 | 小包可直接提供的生产 GPU 统计 |
| --- | --- | --- |
| ordinary proposal | `machine.batch_propose` 一次调用父类处理整个 normal 集合；四候选首 token 用 cached logits，完整非 EOS 情形有 3 个批量 materialize | 缺 native forwards；不能给实测调用总数/B histogram |
| eager continuation | 每个 token step 对 active works 一次批量 materialize；bridge+四候选有 5 个 step，实际 B 是该步存活请求数 | 63 个 completed 请求、315 个生成 token；不是 63 或 315 个 GPU forward |
| correction/bonus repair | `finish_synchronizations` 逐请求 rebase，非 promotion 分支明确 `materialize((row,), "commit")`；无 work 时也是 `commit_many((plan,))` | **源码明确 B1**；156 个 recovery receipt 对应应走的单请求修复路径，原生窗口 forward 统计仍待导出 |
| rejection recovery proposal | 所有父结算完成后，将需正常恢复的请求合并到一次 `super().batch_propose(normal)` | 156 个 recovery job、624 个 recovery token；实际各 step 的 B/耗时待导出 |

Baseline 的 [`BatchedDraftStateMachine`](../src/specrhythm/phase4/batched_draft_service.py)
（237 行）将整个 commit batch 一次交给 backend；
[`VllmBatchedDraftBackend.commit_many`](../src/specrhythm/phase4/vllm_draft_backend.py)
（179–241 行）收集修复 rows 后一次 materialize/fence。
Eager 的 machine 313–327 行循环则进入 GPU backend 267 或 313 行的 singleton 路径。
Promotion 不执行修复 forward，但仍有逐请求 `eager_parent_settlement` fence。
所以不是“Draft 全部变 B1”，而是**父轮修复丢失批处理**。

[`test_serial_eager_batch_diagnosis.py`](../tests/test_serial_eager_batch_diagnosis.py)
通过真实 baseline/eager machine/backend 调用链，构造 B16 中 5 个 promotion、8 个
rejection、3 个 bridge mismatch：baseline 一次 B16 修复，而 eager 有 11 次 B1
修复；之后恢复生成仍是 B11，之前 eager 的五个 token step 仍是 B16。两条路径的
最终 prefix、下一批候选、私有 KV、64 个 committed token 和资源释放均一致。
此测试特意让 16 个 continuation 都完成以覆盖混合结果，不冒充服务器拒绝请求
已实际起草的证据，也不将 CPU worker 调用数写成生产 GPU forward 统计。

另外，全池 audit 每次都重新读 control JSON，构建 360 个 prefix hash 和 block
记录、检查所有 block owner：[`S2DraftBackend._audit`](../src/specrhythm/serving/s2_draft.py)
（56–77 行）、[`ResidentPoolAudit.check`](../src/specrhythm/serving/s2_pool.py)
（37–90 行）。普通 baseline 完整 proposal+commit 通常 4 次；eager B16 admission
16 次、结算前后 32 次、正常 proposal 前后 2 次，另有每个 eager token step 前后
各一次。**重复的是同一 owner 的全池扫描，不能通过删除检查解决。**

GPU worker 的 `greedy` 有一次 bulk D2H；`fence` 调用 `torch.cuda.synchronize`
（[`vllm_draft_worker.py`](../src/specrhythm/phase4/vllm_draft_worker.py) 479–499 行）。
eager 每 token step 都 fence，拒绝、逐请求父结算、修复又有额外 fence；它们是
实际同步调用，但很多可能在设备已空闲时返回，次数不能直接换算成耗时。

另一个已找到但尚未计时的 CPU 成本是
[`evaluate_eager_eligibility`](../src/specrhythm/continuation/core.py)（247–250 行）：
为了隔离 provider，每次评估都 deepcopy 完整 RequestState；verification start 和
continuation begin 各评估一次，其中包含已累计的 proposal/continuation 历史。
这可能随轮数增加 admission 成本；本次没有单独 deepcopy timer，不能把剩余差额
都记在这里。保留该项为待测候选，而不是混入本轮批处理修复。

没有发现每轮完整 prefill、promotion 后重复正常起草同一已合法候选，或为了恢复
复制一套 KV owner。错误 bridge 的生成和修复是协议所需工作；多生成的 bridge
不能算第二次 committed 输出。小包的实际额外物化量由会计计数保留；物化 token
数不等于 GPU forward 数。

## 时间分解：已知、重叠与 unaccounted

以下只使用同一运行的统一 monotonic 测量边界，不叠加不同 rank 或请求的累计耗时。
`step wall` 是 [`fixed_runtime.py`](../src/specrhythm/serving/fixed_runtime.py)
（366–424 行）的 `engine.step()` 加 authoritative `output_commit` 边界，包含同步
Draft 往返；它不是单独的 Target `_model_forward` 时长。

| 项目 | 当前可确认的范围 |
| --- | --- |
| 整个 decode 窗口 | Serial 30107.641 ms；eager 31811.701 ms |
| 顺序 Target step wall 并集 | 由 count×mean：Serial 30039.030 ms；eager 31781.704 ms |
| step 外窗口差额 | Serial 68.611 ms；eager 29.997 ms；不是主要回退位置 |
| continuation unhidden wait | eager 8923.837190 ms，28.0521% 窗口，686.449015 ms/step |
| Target forward、normal/eager GPU、repair GPU | 缺各设备原始 interval，尚不能拆分 |
| owner 排队/RPC、barrier/fence、CPU audit/JSON | 缺窗口 host interval；调用存在，毫秒数未知 |
| 从 eager 窗口算术扣除已报 wait 后 | **22887.863972 ms，unaccounted/尚未分解**，包含上列 GPU 和 host 成本 |

等待起点是 socket thread 把 `synchronize_and_batch_propose` 放入 owner 队列前的
`_owner_enqueue_ns`，不是 Target GPU start；终点是本次 WAITING_DRAFT 集合最后一个
任务完成（或 fenced abort）的 host 时间。见 owner 65–73、103–115、207–209 行。
这个区间可以包含 queue、反馈解析/取消和 eager 执行，属于同步 RPC 及 step wall
的子区间；不能再与这些 inclusive 时间相加。parent rebase/普通恢复在它之后执行，
但它也不是纯 eager GPU 时间。导出工具将同时给出 interval union 和交集。

1674.515 ms/step 的差额，算术扣去 686.449 ms/step 的等待后仍有 **988.066 ms/step**。
这既不是已量化的 Python 成本，也不是修复后的预期收益；还未分离 Target、repair、
normal generation、审计和同步等差异。反过来，消除 wait 也不能直接当作加速预测。

额外 coordinator `status` RPC 确实存在（fixed_runtime 272 行），但位于上述 step
边界之前；窗口中所有 step 外间隔合计只有约 30 ms。这项调用不能解释窗口内
每步 1.675 秒的主要增量。内部 owner 排队及 verification-start/synchronization
RPC 则位于 step 内，仍需原始 host 区间分解。

Draft logging 的 lifetime append blocking 是 Serial 33.4 ms、eager 93.2 ms，
两者 3 次 fsync；这一总量不足以单独解释整轮差额。其他 owner 的日志累计包含
setup/prefill/warmup/drain，不能塞进测量窗口的差额。`_event` 本身只做
`asdict`/Counter/list，backend report 在 owner 退出后写盘；每请求日志序列化、
未缓冲协议写盘和全池检查的测量占比仍须看原始 host timers，而不是改日志策略。

## overlap=0 的证据链

本次 `rolling_eager.GPU_overlap` 的 reason 是 `native CUDA clock-bound intersection`，
inner/outer overlap 均 **0 ms**，`physical_overlap_status=ZERO`。它不是
`native eager/Target CUDA bounds unavailable` 的缺证分支。

[`DeviceTimeline`](../src/specrhythm/serving/fixed_observe.py) 125–188 行在模型
forward hook 中记录 CUDA event，以启动时 bracketed monotonic anchor 投影，
在既有最终 fence 后读取；ns 使用整数 anchor，CUDA elapsed ms 乘 1e6。
没有为每个打点新增设备同步。Draft `worker.phase="eager"` 被 metadata 记录；
[`eager_overlap`](../src/specrhythm/serving/eager_results.py) 108–149 行显式选择
该 purpose，校验两个 Target rank、三个运行内独立设备及正 CUDA 时长，再把
两 TP rank 的区间**取并集**，与 Draft union 在测量窗口内相交。

外层 `status` 只有 `lower>0 → OBSERVED`、其余 `UNKNOWN`，所以 ZERO 也落入 UNKNOWN。
应读作：**已记录的设备 forward 区间在该窗口的误差外界内也无交集**；不是忽略
ZERO，更不是所有 GPU 工作/未记录 kernel 的全局证明。小包缺原始区间，尚不能
独立复算 anchor、窗口选择和 event 完整性。生产 PASS 的 `device_batches` 已检查
Target B/IDs 和每 rank 正事件，仍应导出 Draft 各 purpose 及完整投影元数据复核。
不跨 Serial/eager 历史 point 比较 GPU UUID，仅保留各运行内部已有身份检查。

## 必须补的最小证据与轨迹

原始 root 应保留 `runtime.json`、`draft-backend-report.json`、
`round-events.jsonl`、`transport-events.jsonl`、`draft-work-events.jsonl`。
小包没有它们，不能从汇总凭空还原某个 request 的周期。

当前尝试结果：成功提升 52 次、父拒绝 145 次、bridge mismatch 11 次均已观察到；
**这三类事件的 request/round/proposal/continuation ID 与逐轮时间关联缺失**。
同一请求“成功 → 拒绝 → 恢复 → 再次 eager”的生产轨迹也未能从小包恢复。
CPU/GPU 小回归中的构造轨迹不能填这个空白。

离线取证工具读取已结束的两个 B16 point，输出完整窗口的紧凑 GPU/host 证据、
按 purpose 的实际 forward/B 分布、区间并集以及有界 request/round 样本。精确
enqueue/dequeue 或 payload ID 若原日志根本没记录，保留 MISSING；通过时间和
request 推导的关联明确标成 derived，不写成直接观测。工具不导入模型、不增加
运行时打点、不改旧结果，不执行 summary 重算来覆盖旧 PASS。

优先查看：

- 每轮 `eager_enqueue` RPC 回执与 Target 两 rank 原生 forward 起止；
- 最后一个 admission / 父结果事件 / 首个真实 eager forward 的相对顺序；
- 首 forward 前 `resident_block_audit`、prefix hash、control JSON 的 union；
- 原生 `purpose=commit` 的实际 B1 rows、调用数量及其与 request/round 的关联；
- 正常恢复是否一批完成，以及剩余等待是否落在整个同步 RPC 之外。

如果原始 host 证据仍无法区分 dequeue 与 admission 成本，再单独设计默认关闭、
有限请求/周期的 in-memory owner 打点；当前不要求 GPU 重跑。

## 结论分级

| 分级 | 结论与证据边界 |
| --- | --- |
| 已确认的运行现象 | 两点资格 PASS；主要变化在 step wall；已记录设备 forward 的重叠上、下界均为零。依据两份 light-summary 及原生 overlap 分支，不把外层 UNKNOWN 当缺证。 |
| 已确认的实现问题 | 入队后整批逐请求 audit 推迟首次 stepping；反馈优先机制可变成验证后起草；父轮 repair 逐请求 B1。源码和真实 owner/backend CPU 回归均复现，但各项生产毫秒贡献尚未测得。 |
| 待确认的主要因果假设 | 重复 audit/历史 deepcopy 是否耗尽 Target 窗口，进而导致本次 145 个拒绝全部提前取消；B1 修复与每步 fence 各贡献多少墙钟；须用相同周期的 admission、反馈、设备 forward 和 host union 区分。 |
| 已排除或不支持的主因 | Target 尾部缩批、workload/提交不同、prefill 被当窗口内 decode；Draft 全部退化 B1；入队 ACK 直接等待 GPU 完成。Draft lifetime 日志总量及窗口 step 外总量不足以单独解释主要增量。 |
| 仍然缺失 | 原始 owner dequeue 和逐 proposal 事件若未写日志，无法从现有源码或汇总恢复；同请求跨轮轨迹、实测 Draft forward/B 分布和 22.888 秒内部分解等待只读导出。 |

## 最小修复方案（未在本轮执行）

1. **缩短 admission 临界路径：**新增真正的批量 continuation enrollment，在同一
   owner 固定快照下先完整校验所有 request/proposal/version/capacity，再发布任务；
   同一批没有 KV 写入时，只在批量边界做完整 pool audit，保留逐请求依赖检查。
   不能简单关闭 audit、让 Target 等 first GPU start，或取消反馈优先规则。
   首先用旧 trace 确认 admission 是否吃掉全部 Target 窗口，再单独复测这一改变。
2. **独立修复 B1 parent settlement：**验证整个父结果集合，区分 promotion 和需要
   correction/bonus 的 rows；一次 fenced batched materialize 后逐请求发布结果、
   记账、retire。全池校验保留在安全批量边界，不改 `commit_frontier` 或依赖协议。
   与第 1 项分开提交/测量，不能混在一次结果里声称某项贡献。

每 token fence 和批内等待传播是否还值得改动，排在上述证据和
独立修复之后。本轮不给无依据的 speedup 预测，不把“收益经验”当作当前实现证据。
服务器只执行只读取证命令，返回新证据包后再决定最小单点复验。

## 本轮验证与交付

最终 Python 3.11 全库 **2058 passed、3 个既有 skip**，284.470 秒；新增 30 项
回归包含 25 项导出边界、4 项 owner 启动顺序、1 项混合 batch 修复。Python 3.9
Rolling Eager 与固定 vLLM 源码契约 **225 passed、零 skip**。Ruff、两版本
compileall、275 个源/测试文件的 3.9 语法、11 个 Bash 脚本及本轮完整命令块均通过。
既有 skip 仍是 opt-in CUDA 和两个 Linux-only 进程生命周期测试，未放宽检查。
最初沙箱拒绝了 3.9 测试所需的 `ps`/socket bind；允许这些本机操作后的原断言全部通过。

导出器已直接读取用户小包的本地解包副本：全部 63 个输入文件的内容 hash 和 mtime
不变，两个旧 PASS 保留，缺原始时间线标 MISSING。此验证不补造任何生产 GPU 事件。
完整提交 SHA、推送后的实际 CI 和填好 SHA 的只读命令副本随本轮交付提供。
