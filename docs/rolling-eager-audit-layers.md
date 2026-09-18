# Serial-eager：Draft 审计分层与 465c215 实测复核

本轮保留 A+B、资格快照修复以及批级 WAITING_DRAFT。新增显式 `draft_audit=runtime`，
默认 `full` 仍执行原完整审计。该配置只作用于 fixed diagnostic 的 Draft；Target 两个
TP rank 在两个模式中都保留原检查。原 Serial/PingPong 默认算法和旧结果没有改写。
下一步仅为[同 SHA 四点 B16 对照](rolling-eager-audit-runbook.md)，不再维护 A-only。
新路径的 GPU correctness、原生 overlap 和 performance 均为 **PENDING**。

## 原始 GPU 证据：窗口与嵌套关系

输入是用户提供的 `serial-latency-B16-465c2159b4b5-20260912T172326Z-1628-failure-evidence.tar.gz`，
SHA256 `ec6129c4d7b971c1ee4b3f09b3635d4e74af6ddab042d403087e32a56f960823`。
原始提交 `465c2159b4b58d8c0e79fc3f66f38083e88368cb`；从 raw runtime/backend、small 的
light summary 和 process lifecycle 联合复算。[机器可读复核](evidence/serial-eager-465-audit-recheck.json)
不嵌入原始事件，不覆盖原文件。

| 同一实际窗口口径 | Serial | Serial-eager |
| --- | ---: | ---: |
| throughput，tok/s | 60.040693 | 45.305571 |
| 实际窗口，ms | 30262.808408 | 30084.600197 |
| steps / committed tokens | 38 / 1817 | 29 / 1363 |
| tokens/step | 47.815789 | 47.000000 |
| 完整 step mean / P50 / P90，ms | 794.603620 / 795.763583 / 811.844280 | 1035.097978 / 1035.514955 / 1073.071211 |
| Target rank 0 / 1 forward mean，ms | 83.487150 / 83.554585 | 83.731950 / 83.749192 |
| Target TP 原生区间并集下/上界，ms | 3185.644744 / 3192.926532 | 2437.415429 / 2442.813083 |
| Draft pool checks / step | 4 | 15 |
| 全部完整 step 之外的窗口，ms | 67.870849 | 66.758824 |
| 原 execution / measurement / cleanup | PASS / PASS / PASS | PASS / PASS / PASS |

eager 的 owner dequeue 到首个实际 GPU forward 平均下/上界为
102.523808–102.579799 ms。首 forward 前两次全池审计平均 81.696735 ms、720 次
prefix visit；29 轮中 28 轮首次 forward 晚于 Target GPU 结束，1 轮有确定原生重叠。
全窗口 eager/Target TP forward overlap 为 **4.436982–4.585746 ms**。这次是少量
已记录的正重叠，不能沿用前次 ZERO，也不能把 host enqueue 当 GPU 起点。
四条 causal lane 均为 light/COMPLETE、dropped=0；两个原生 Target rank 和 Draft 都存在。

五次 eager token step 的 `machine_step` 区间并集平均 388.623000 ms/Target step；
其中嵌套的 `physical_gpu_audit` 为 269.218938 ms，eager GPU event sum 为
101.706627 ms。整个周期的 physical audit 还包括 admission/settlement，平均
369.938927 ms，**不是**上述 269.2 ms 的另一个可加项。第一次起草前的 81.7 ms
与 token-step 内的审计也有父子包含关系。其余时间继续按线程互斥覆盖或 unaccounted
报告，不把算术差额命名为 Python/GIL、调度或空闲。

Draft 原生 forward：Serial proposal 114 次（B16 111、B15 3），commit 38 次 B16；
eager proposal 87 次、commit 29 次，B7–15；eager forward 145 次，B1–16。
这不是把恢复拆成 B1 的证据：拒绝反馈后部分请求退出 eager，后续 token step 的有效 B
合法缩小。每点按用途的完整 B 分布见机器摘要。实际 Target B 始终 16；当前代码没有
改变请求选择、EOS 或有效 batch 的形成。464 次机会产生 115 promotions、349 recovery
jobs；899 accepted + 322 correction + 142 bonus = 1363 committed，promotion 不是输出。

已确认的热点是同步全池重建及检查；必要设备执行仍约 101.7 ms/轮，批级等待和恢复
依赖仍存在。原生 Target forward 平均只差约 0.25 ms，无法解释完整 step 多出的
240.49 ms；每步产出仅下降约 1.7%。不能从旧窗口扣除审计区间计算“修正吞吐”，也不能
把 full/runtime 差值当作严格可加的因果成本。新四点实测用于区分这些影响。

旧 exporter 的离线复现仅 eager 单点 pretty JSON 就有 72,218,619 bytes（Serial
1,908,279 bytes），超过 `eager_retest_report.report` 的 64 MiB 上限。包中 attribution
缺失与此一致，但包里没有原导出异常，不能补写其原始错误。该取证问题不改变两点的
PASS 资格。新 reporter 单点聚合旧证据约 0.25 MB，保留原始事件一份即可复算。

## 调用链、功能副作用与分类

`TRACE.observe("physical_gpu_audit")` 是**包裹被测函数的 inclusive 计时标签**。
TRACE 关闭时仍执行 `_gpu_audit()`；标签区间包含 control read、hash、pool check 和状态
判断，不等于记录器开销，更不等于 eager 算法必须付出的所有延迟。

| 函数及源码 | 读取 / 修改 | 返回值与消费者 | 不变量、同步点及覆盖 |
| --- | --- | --- | --- |
| [`TRACE.observe/span/event`](../src/specrhythm/continuation/trace.py) | 读开关/线程上下文，写有界内存 rows、dropped counter | wrapper 返回原函数结果；report 供离线因果分析 | 观测记录；不增 CUDA fence，不处理 owner 命令。不能把 span wall 全归为 append 成本。 |
| [`_gpu_audit`](../src/specrhythm/continuation/gpu_backend.py) | 调用 pool/control 检查，读取 admission/settlement 状态 | 无业务返回；begin/step/rebase/finish 依赖其错误传播 | 必要运行时检查 + 原完整实验审计。当前 owner 安全边界同步校验受影响请求即可，不必每 token 扫 resident360。 |
| [`S2DraftBackend._audit`](../src/specrhythm/serving/s2_draft.py) | 读 fresh control、retired、物理行；首次从最终 setup 文件冻结 `pool.initial`；更新 checks/peak | 返回完整 control packet；propose/commit、GPU admission/settlement 用它判断 ACTIVE/FINISHED | 首次 freeze 和全局基准有功能影响；不能删。runtime 热路径保留 fresh packet，避免构造全池 retired overlay。 |
| [`physical_rows`](../src/specrhythm/serving/s2_draft.py) | 遍历 live handles、prefix/frontier 和 allocator 当前 block tables；自身不改 KV | 全池 dict 供 audit、setup publish、诊断 settle、terminal release/unrelated digest | 完整实验审计与终止收据。窗口 token step 只需受影响行；setup、全局 reconciliation、terminal receipt 仍全量。 |
| [`ResidentPoolAudit.check`](../src/specrhythm/serving/s2_pool.py) | 重建 owners；freeze 深拷贝初始行；更新 peak_blocks/checks | 无业务返回；report 的 initial、peak、checks 用于容量/驻留资格 | 私有 block 非空、唯一、不跨请求；STAGED/QUEUED 保持初始行，ACTIVE 保留初始 KV。runtime 用 allocator 边界账本和 touched checks；完整实验检查仍单独计数。 |
| [`prefix_record` / `token_prefix_hash`](../src/specrhythm/serving/s2_pool.py) / [`serial.py`](../src/specrhythm/phase4/serial.py) | 读 canonical token tuple，计算 hash，复制 frontier/block IDs；不读写 KV tensor | 行摘要供上述比较和收据；协议 hash 仍由原协议校验 | 重复 hash 是可替代的派生计算；只在不可变 prefix 对象变化后重算，不缓存“全局 PASS”。协议 identity/hash 校验未删除。 |
| [`control/publish`](../src/specrhythm/serving/s2_pool.py) | 每次读原子 JSON 文件；publish 写临时文件后 replace，无逐 token fsync | 完整 snapshot 包括 barrier、requests 和其他协调字段 | 当前请求 admission 必须 fresh。全池 JSON parse 尚保留，不能假定其版本稳定。没有缓存 control 或丢弃字段。 |

[`fixed_runtime.publish_control`](../src/specrhythm/serving/fixed_runtime.py)、S2 clock 产生
control 状态；[`s2_scheduler`](../src/specrhythm/serving/s2_scheduler.py)、
[`s2_proposer`](../src/specrhythm/serving/s2_proposer.py) 和 Draft audit 是不同消费者。
`_audit` 本身不 dequeue RPC，也不改变 admission；owner 的反馈、取消和 shutdown 命令
仍由 [`EagerOwner._run`](../src/specrhythm/serving/eager_owner.py) 在 fenced token step 之间
优先处理。保留一次当前检查不意味着延迟或忽略已经到达的反馈。

| 层 | 保留或替代的工作 |
| --- | --- |
| 必要执行 | 原 batched proposal、五次 eager materialize/sample、兼容父轮 repair、恢复、必要 fence、实际 release；原等待和调度不变。 |
| 必要运行时检查 | 原 core/adapter proposal、continuation、dependency/version 检查；fresh admission；touched frontier/capacity；每个 allocator mutation 的全局 inverse ownership；受控错误传播。 |
| 完整实验审计 | 默认 full 全部保留；runtime 保留 setup、首次 timing entry、最终 release 前、shutdown 和完整 terminal receipts。在这些边界增量账本与完整结果核对。 |
| 观测记录 | 相同 buffered-live、bound-prefix、light causal、原生设备 hooks；新增 mode/scope/version、scan/visit/check 计数。新 trace 仍默认关闭、内存有界、无新设备同步。 |

## runtime 的失效条件和生命周期覆盖

[`FixedAuditMixin`](../src/specrhythm/serving/fixed_audit.py) 只组合到 FixedDraftBackend。
CLI 冻结 `options.draft_audit` 并向 Draft 子进程传递；runtime CLI 只接受显式
Serial/Serial-eager single point，禁止隐式扩展到其他模式。full 的 pool check 算法不变。
两个模式均记录最终 release 前的完整边界；该额外终止校验在 drain 内，不改测量窗口。

[`RuntimeKVGuard`](../src/specrhythm/serving/runtime_kv_audit.py) 在真实 setup 前安装到
同一个 worker。原 [`VllmDraftWorker`](../src/specrhythm/phase4/vllm_draft_worker.py) 的
private full-attention、无 prefix caching、无 eviction/replay fallback、sole-owner 契约
仍是前提。所有实际 ownership mutation 都经过 `kv.allocate_slots` / `kv.free`；
其余 KV 调用是 get_block_ids、new_step_starts、take_new_block_ids。分配成功后，在 GPU
使用前验证 append-only block table、范围、重复块及 inverse ownership。free 只允许
指定 release scope，并且没有未完成写入；原 release fence 没有移除。
未知 allocator API 在 runtime 直接拒绝，新增 API 必须先审查其 ownership 副作用。
control 的 FINISHED 可能早于权威反馈到达：新 proposal/enrollment 仍需 ACTIVE，但已获准
批次保留原 token/fence 行为，再由 owner 结算或取消。不能把这个合法竞态改成运行失败。

账本不是长期有效的检查结果：allocation/growth 更新新增 ownership、peak 和 mutation；
rollback 只降低验证过的逻辑 frontier，保留 private high-water blocks；commit 的
next_round 只能保持或推进一版，prefix 必须相等或扩展。prefix 对象变化时重算摘要，
相同不可变 tuple 才复用 hash。每次 touched check 都重读 allocator block table 并核对
ledger；free 删除 handles/written/ownership，保留 released ID 防止复活。补位请求已经
在 setup prefill，只从 QUEUED 变 ACTIVE；runtime 禁止 timing 后 initialize。

所有 materialize（含 setup）、allocate/free、fence 均经过边界；因此不能在两个全量
审计之间合法地偷偷驱逐、共享或重做 prefill，再在末尾恢复以蒙混通过。单次 full
检查也不是 KV tensor 内容校验；本契约不声称检测绕过私有 worker/allocator API 的
任意内存破坏。新增 mutation API 时必须扩展 guard，不能依赖旧缓存。

`pool.initial` 保持原 freeze；runtime 增量 peak 覆盖所有成功分配，包括随后 rollback
或释放的高水位。`pool.checks` 继续只表示真正完整 check，runtime 检查单独计数，不能
为了满足旧检查次数而伪造计数。原 terminal receipt 的 unrelated hash、release scope、
materialized 结果和 drain 消费者仍有效。

尚未替代：fresh control 的全池 JSON decode；setup/end 与终止收据的完整扫描/摘要；
受影响请求的整个 block table 检查（不是仅检查新尾块）；原协议 prefix/hash、identity、
feedback 和 KV frontier 验证。完整边界可能执行两次 physical_rows 以独立对照增量表，
有记录且不在稳定 token 热路径。没有为追求低耗时关闭必要检查。

## 回归与报告口径

[`test_fixed_runtime_audit.py`](../tests/test_fixed_runtime_audit.py) 用真实 owner/queue/machine/
backend 配合 CPU allocator substitute 对照 full/runtime：混合七轮成功→拒绝→恢复→
再次 rolling，bridge mismatch，EOS、取消、queued refill、开关版本、迟到 completion 和
重复反馈，accepted/correction/bonus 守恒、完整边界 reconciliation 和最终资源释放。
确定性 Event 决定反馈先后；等待超时仅防测试挂死，不用于性能判断。
结构测试在 resident360/B16 token step 检查 full-prefix-visits 不增加；负例覆盖版本、
frontier、非法状态、handle identity、共享/重复 block、越界 allocator 操作、未 fence
release，以及 worker 失败后原 fence/teardown。普通 eager 关闭路径也对照输出/回退。

[`audit_gpu_check.py`](../src/specrhythm/continuation/audit_gpu_check.py) 让必要物理 correctness
使用相同 audit layer。只有独立诊断 reference ID 可在冻结后 prefill；生产类无此例外。
CPU 版本跑过该入口的参考比对，但不等于真实 GPU correctness 通过。

[`audit_layer_report.py`](../src/specrhythm/serving/audit_layer_report.py) 保留原窗口吞吐、
step mean/P50/P90、两 rank 和 TP union、原生 overlap bounds、每周期队列/admission/首次
GPU 时序、Target 结束时各请求完成的 eager steps、按用途 GPU forward/B、normal/recovery
混批、promotion/discard/recovery/unhidden wait、完整扫描/prefix visit/touched checks。
hash/JSON/log/fence 的 inclusive 分类保留，父子 span 不相加；每 PID/thread 用端点扫描
形成互斥分解。allocator_check_ns 是 worker operation 内检查 CPU 累计子项，不是该
operation（含 forward）的 wall。缺失时保留 MISSING/TRUNCATED/unknown，不把空集当 0。

每文件 512 MiB、每点输入 1 GiB、512 steps、输出 8 MiB，四点聚合只嵌入各紧凑结果一次。
report 错误写独立 error metadata、保留原错误并停止后续点；原资格不变。timer/TRACE
append 自身成本未通过 observer on/off 隔离，报告明确 NOT_ISOLATED；设备区间只覆盖
model forward，不覆盖全部 GPU kernels。四点须使用相同观测，不能用 full Serial 与
runtime eager 声称算法收益。任何加速和重叠改善均等用户复测。
