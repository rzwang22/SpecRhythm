# PingPong prepost3：单 Draft owner、跨验证机会接续

运行类型与资格契约（2026-09-13修复）：probe是实际调用的bool，run/drive在模型工作前核对point；
报告公共部分始终写顶层probe。容量为true，correctness和performance为false。
correctness走非scan：point.prepost_correctness=true，decode_scan不适用；仍要求完整输出/清理及
原生TP forward/request/proposal绑定。scan容量/性能必须带decode_scan，保留原窗口与检查规则。
顶层/point/调用参数/阶段冲突或错误类型显式失败，不默认补false。新规则不改变旧模式的执行协议。

联合验证的首错归属当前子运行，不能来自外层容量循环。模式root、实际run目录、原始资格错误及
进程退出码从run_point的阶段文件和结果读取；外层joint阶段、校验命令码、导出码分开。
真实报告、stage来源、joint结果/失败、比较及first-failure仍在同一个有界总包中。
背景与生产链回归见 [验证记录](pingpong-prepost3-validation.md)。

2026-09-13。Draft PR #5 的显式新协议，GPU correctness、overlap、performance **PENDING**。
实现基础为 `f01e8d037007540209999179357c3dd2ff2335a7`，保留
`71f062c2a5fcef9298300904164a5abbde962a2f` 的后续交付。
用户返回的上一轮 Serial-prepost3 / Serial-eager-prepost3 约为 87.98 / 100.59 tok/s，
单窗口约 +14.34%；这是此前 Serial 协议的已返回结果，不是本轮 PingPong 的验证或配对基线。

## 模式与执行边界

| 显式模式 | 提交协议 | Draft 执行 | Target admission |
|---|---|---|---|
| `pingpong-prepost3` | 全接受仅提交实际 P；拒绝提交接受前缀+correction；无额外 bonus | 公共 seed + 3 个普通扩展；可与另一批 Target 重叠 | 正常 home 填充；该 home 无 ready 时可用另一 home |
| `pingpong-eager-prepost3` | 同上 | 验证中最多 3 个 lookahead；反馈后公共成功扩展/拒绝恢复 | 合法 promoted 优先跨 home 接续，再正常 home 填充 |

共同资源为 resident360、总 active16、初始 home A/B 各8、**每次 Target ceiling8**；
Draft GPU0；GPU1+2 是一个 Target TP2 模型。两种模式使用相同 workload/order/seed、模型、
dtype/sampling、日志缓冲、runtime audit、KV 容量配置规则、两轮预热和连续30秒窗口。
新模式只开放 B16 单点，不加入默认扫描列表。旧模式的 gate、bonus/bridge、batch qualifier 不变。
不增加 CUDA Graph、不延迟 Target/反馈、不要求完全隐藏 Draft，也不以正加速为正确性门槛。

## 源码映射

| 文件/接口 | 已有契约与本轮实际用途 |
|---|---|
| `continuation/scheduling.py::select_target_admissions` | 原为纯 CPU 选择契约。本轮由实际 owner 调用；合法 promoted 优先，输入顺序稳定，正常 home 补足。 |
| `continuation/prepost.py` | 沿用 no-bonus acceptance、依赖版本、Lookahead 与 prefix ledger。 |
| `continuation/prepost_backend.py` | 沿用真正的 KV materialize、批量 post、fence、回退与释放；不改旧 Serial 默认循环。 |
| `serving/prepost_machine.py` | 沿用批量验证/反馈校验与 committed authority；原 Serial 同步等待整批下一 proposal，本轮新子类独立结算父请求子集。 |
| `serving/s2_scheduler.py` / `fixed_scheduler.py` | 继续实际 resident gate、私有 KV 检查、stock allocation、positions/B 证据；新子类安装 owner claim。 |
| `serving/ping_prepost_controller.py` | coordinator 发送机会编号与当前 ACTIVE 集合；非空 admission 才推进 A/B 机会角色。空 poll 也有独立编号。 |
| `serving/ping_prepost_owner.py` | 一条 owner 线程、上限256的 mailbox；反馈 ACK 不等待下一 proposal。 |
| `serving/ping_prepost_machine.py` | 唯一 authoritative ready/claim/normal/parent 状态；跨批次原子 claim→consume→settle→ready。 |
| `continuation/ping_prepost_backend.py` | 一次物理 forward 合并不重叠请求的 lookahead、普通扩展、post 行；每行1输入位置；一次 batch fence 后更新。 |
| `serving/ping_prepost_proposer.py` | 实际 Target adapter，接收 claim，在 verification-start enqueue；真实采样反馈经 `pp_feedback` 返回验证 ACK；物理 Draft frontier 标为 PENDING。 |
| `serving/prepost_target.py` | 在 pinned vLLM 的真实 bookkeeping 中投影掉额外 bonus；保持真实 ragged 长度、输出限制和 CPU sampler结果语义。 |
| `serving/ping_prepost_window.py` | 两次真实 admission 为一轮预热，可重复请求；不引入两组全 idle/GPU barrier。 |

真实执行链为 `fixed_runtime.drive → controller.pp_admit → owner atomic claim → atomic control
snapshot → PingPrePostScheduler → stock SchedulerOutput → Target verification hook → eager_enqueue
→ owner/backend token step → Target bookkeeping/no-bonus projection → adapter.pp_feedback → owner
parent validation/settlement → ready → 下一 admission`。原 stock ragged positions/attention/slot mapping
和采样索引被保留；短请求不是有效 K4 padding，不拆成两次 Target forward。

## token、forward 与 KV

设 authoritative committed prefix 为 C，正在验证的实际候选为 P（长度1或4，EOS/预算处可截短）。
lookahead 单独存储为 L，不能写入 C。父依赖绑定 `(request_id, prefix_version, proposal_id, C, P)`；
continuation ID 还记录创建时的 decision version。一个请求只有一份 authoritative 状态。

| 工作 | forward 输入位置 | 产生候选 | 写入后 materialized frontier |
|---|---|---|---|
| lookahead #1 | P 的最后一个 token | a1 | `len(C+P)` |
| lookahead #2 | a1 | a2 | `len(C+P+a1)` |
| lookahead #3 | a2 | a3 | `len(C+P+a1+a2)` |
| 全接受公共步骤 | a3 | a4；下一 P=`a1,a2,a3,a4` | `len(C+P+a1+a2+a3)` |
| 拒绝公共步骤 | 合法 correction | r1；下一 P=`r1` | `len(C+accepted+correction)` |
| 普通对照公共步骤 | 最后一个已接受 P 或 correction | s1 | 合法新 C 长度 |
| 对照普通扩展 #1/#2/#3 | s1 / s2 / s3 | s2 / s3 / s4 | 新 C 长度+1/+2/+3 |

forward 产生的最后一个候选尚未 materialize；下一 token step 才消费它。
所以「3」是三个提前 token forward，不含公共步骤、setup/refill seed 或 terminal repair。
初次 admission 从真实预填充+bootstrap 的 cached logits 采样 P1，新增 forward 数为0，初始化记录独立。
回退仅在此前写入 fence 后进行；公共 forward 使用正确 valid_length 覆盖失效 suffix。
所有请求的公共步骤尽可能同批；没有逐请求 B1 恢复循环。
lookahead 自身遇 EOS 时只保留到 EOS，不额外扩展；预算不足时裁剪 limit。此类 P2/P3 单独由实际长度报告，
不是把四个候选改名为三个。终止时不发布下一 proposal，必要的最后一个 committed token materialization 如实记账。

## 单 GPU 调度与原子性

owner 先处理已到 mailbox 的控制/反馈，然后执行一个物理 token step。已启动写入必须完成 fence；
之后拒绝、关闭资格、取消/stop 可以使后续 lookahead 失效。队列有界，错误向调用方/服务传播。
同一物理 batch 内，每个 request 只能有一行；不允许 common/background 别名。

待结算父轮中，没有 lookahead 或 lookahead 已完成/取消的请求，可立即组成 post 子批。
它与其他请求的普通扩展/lookahead 合并；否则运行普通/lookahead 子批。不会在一个 owner 工作轮里
连续执行三次普通扩展。全接受但尚未完成的 lookahead 可继续，反馈→ready 的暴露等待照实记录。
已 ready 的另一 home 不必等待这些请求全部完成。唯一不可抢占单位是正在执行的 token forward/fence。
这也意味着 coordinator 的同步 admission/status RPC 可能等待该物理边界；报告保留这一成本。

ready 是完成物理更新后的结果，不是 enqueue ACK。`pp_admit` 在同一 owner 稳定状态中先选择，再原子移到 claims；
Target scheduler 与 proposer 用完整 proposal/prefix 对照，consume 只能一次。反馈 ACK 只表示校验通过，
不能被当成 Draft KV 已结算，所以 Target round 的 `logical_draft_kv_length=null`，必须按 request/version
关联后续 owner settlement。父反馈的精确重复幂等；内容冲突/过期 proposal/重复 claim 或 consume 失败。

`home_cohort` 永不随跨批机会迁移；`target_batch_id`、admission opportunity、claim ID 独立。
拒绝不移除静态 eager 资格。`pp_eligibility` / `eager_switch` 保留未来按请求更新的接口，版本递增，
移除/关闭会废弃失效工作；本轮脚本不改变静态集合。新 admission 使用当前决策版本，已合法 P 的 token 依赖
仍按原父前缀版本检查。ready 的普通结果按集合查找，不受全局 FIFO 头部另一 home 积压遮蔽。

严格 promoted 优先不提供无限长请求下的轮转配额保证。本轮 workload 有固定有限输出上限：
优先请求终止释放后，普通 home/refill 能获得容量；CPU 回归验证有限完成后的进展。
没有加入额外 urgency、公平性阈值或动态预算。容量延期、依赖延期、未填满原因与实际请求集合都记录。

## 两条可发生的时间线

以下为生产状态机 CPU 回归所执行的顺序，GPU 时间仍待实测。`D` 上的不同 forward 顺序执行，
不能把普通扩展和 lookahead 画成两个并行的 Draft 模型。

连续全接受跨机会（r 的 home 始终 A）：

```text
机会 A: Target 验证 r.P1 ───────────────┐
Draft0:   look1 → look2 → look3          │ [与独立普通行可同批]
反馈:                              full(P1)
Draft0:                                 common(a3→a4), fence → r.P4 ready
机会 B:                                                   claim r.P4 → Target
Draft0:                                                     look1→look2→look3
反馈:                                                                     full(P4)
Draft0:                                                                    common → P4 ready
机会 A:                                                                               claim r.P4
```

部分拒绝、短验证恢复、再次 P4（另一请求 q 正常推进）：

```text
Target: A 验证 r.P4 → reject(prefix+correction) → B 验证 ready 的 q
Draft0: look1 [正在写] → fence/处理拒绝 → common(correction→r1) + q的兼容工作
owner:  废弃 r 的全部错误 lookahead；r.P1 ready，静态资格保留
Target: 下一有容量机会验证 r.P1（可与其他请求P4混批）
Draft0: look1→look2→look3；full(P1) → common → r.P4 ready
Target: 后续机会验证 r.P4；重复 full → P4，或 rejection → P1
```

若 full 先到，r 的合法剩余 lookahead 继续；q 的 ready proposal 可以先被 claim。
若 rejection 先到，正在写的 token 完成必要 fence 后取消其余两步。
停止时取消未完成工作；已反馈父轮仍结算 authority，正常扩展中的私有 P2/P3 用仅供 drain 的 descriptor
记账释放，不进入 ready，不发布给 Target。原 bounded drain/错误退出仍有效。

## 验证与证据口径

CPU 测试使用真实 controller、owner、backend、resident scheduler、Target adapter 和 pinned vLLM
bookkeeping；硬件和传输可替换。受控线程事件检验 ACK/反馈优先，不能声称 CPU 并发等于 GPU overlap。
`ping_prepost_gpu_check` 另运行 Target-only 与两个新模式的16请求完整输出 fixture（最多32输出），
逐请求比较整个输出，并要求真实混合 P1/P4、A→B→A、拒绝 P1 后恢复 P4 和自然释放覆盖。
若本次确定性 fixture 未覆盖所需路径，单独失败于 coverage，不能强行写 PASS。

性能点只用冻结 resident360，总active16、Target≤8；两轮预热=四次真实非空验证机会。
相邻机会可重复 request IDs；初始 active16/home8/8 仍须真实满足。空 poll、status/RPC、refill、等待、
输出提交均留在窗口墙钟；最后已发起 step 完整记账，drain 不混入测量。旧模式仍要求原来的整批/分组边界。

`ping_prepost_evidence` 按 request/version/claim 跨 Target step 关联，避免套用 Serial 的「反馈必须在本步结算」假设。
原生 GPU interval 使用两 TP rank 的并集及 CUDA 锚点误差上下界；分别统计普通扩展、lookahead、公共步骤，
混合 forward 可以属于多个用途，**总 overlap 只算区间并集一次**。原生事件缺失时输出 null/MISSING。
每请求在 Target 结束时完成的 lookahead 步数使用 device end 的上下界；另保留 host fence 证据。
GPU event 累计、父子 span、RPC 和不同进程/线程不相加到墙钟；未覆盖时间保留，不称作 Python/GIL。

保留 phased trace 的 setup8192 / warmup4096 / measurement65536 / drain8192 行预算，独立 owner/protocol
记录同样有界。任何丢行或映射缺失保持严格诊断失败；执行、测量、清理的原始资格独立保留。
记录 metadata、全审计边界、实际 P/B/positions、反馈→ready、ready→admission、角色forward计数/时间、
promotion/拒绝/丢弃、容量延期及原生重叠。观测自身成本未隔离，不能从吞吐扣除 inclusive span 得出修正吞吐。

单包交付规则、固定服务器命令见 [runbook](pingpong-prepost3-runbook.md)。较大66B—70B Target 是否更能隐藏
lookahead 只是以后待测假设，本轮没有换模型或自动扩大扫描。
