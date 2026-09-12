# Serial-eager B16：两项独立性能修复

原运行提交 `4a6725b054990e47b7e9c4cf63f38b995d029856` 的 Serial/Serial-eager
结果继续保持 execution/measurement/cleanup PASS 和有效负性能结论。本轮从诊断
提交 `6ee326006394db5ac8b89a5d80de57248219a75d` 继续；CPU 回归不代表 GPU
性能已经修复。只修改 Draft PR #5，不合并，不修改 PR #2/#3/#4 或旧结果。

## 补充原始证据复核

只读检查用户补交的 archive，唯一成员为生产 eager point 的
`draft-backend-report.json`。长度 **149732487 bytes**；SHA256
`b769720b2c0e5520a76f07ba01dbc4c8aefe2d6a2a52a56fed3b2dead78f006a` 与声明一致。
另读已有 `*-evidence-6ee326006394-20260912T125110Z-1473.tar.gz`，其 JSON SHA256 为
`870460cda6f306e0900196c5ab35ab044b84269e9132d8df35531fc491509e10`，保留完整测量
窗口的 Target rank 0/1 各 13 条 native forward，以及有界 transport 样本。
旧导出器的 128 MiB 单文件上限导致这个 149.7 MB Draft 报告标 MISSING/LIMIT；
不能据此认为服务器没有 Draft 设备事件。

所有下列计数均过滤到原 measured window
`[18327897037535989, 18327928849237151] ns`，不含 setup、warmup 或 drain：

- 208 admissions、63 started、145 parent rejections。逐 request 顺序核对，63 个
  started 都有同一父轮在前的全接受反馈。拒绝任务在首个 token 前合法取消。
- 13 个周期各有 17 次全池审计、6120 次 prefix visit 位于首个 eager forward 前。
  首个 eager GPU start 下界比对应 rank 0 Target end 上界晚 **443.312–551.023 ms**。
- Target 两 TP rank 取并集后，与真实 `purpose=eager` 区间相交，inner/outer 都为
  **0 ms**。此前的 ZERO 结论被原始设备记录独立复算确认。
- 全窗口 **780 次 audit = 60/step**，280800 次 prefix visit。普通 Serial 的既有
  导出是 **156 次 audit = 4/step**，Target batch 均保持 16。

| Draft purpose | 原生 forward 数 | B → 次数 | CUDA event 累计 ms |
| --- | ---: | --- | ---: |
| eager | 65 | 1→5, 2→5, 3→20, 4→5, 5→5, 6→15, 10→5, 11→5 | 1297.429688 |
| commit（父轮 correction/bonus 修复） | 156 | 1→156 | 2858.601563 |
| proposal（normal/recovery） | 39 | 8→3, 9→3, 10→3, 11→9, 13→9, 14→9, 15→3 | 774.656250 |

因此退化确实位于父轮修复；eager 每 token step 与 normal/recovery 仍批量执行。
窗口 prefix union **14968.312555 ms**、block audit union **3049.324020 ms**；
JSON union **7743.378579 ms** 与 prefix 大量嵌套，禁止相加。Draft fence 的 host
union 为 **143.213238 ms**，598 次调用，不把同步调用数直接当成等待时长。
各 purpose 的 CUDA sum 不是跨进程关键路径；RPC、unhidden wait 和内部审计/GPU
区间仍有重叠，不能把它们的 inclusive 累计直接相加。

prefix+block audit 的联合覆盖是 **18017.636575 ms**，其中 **5791.445298 ms**
落在已报 unhidden wait 内，另有 **12226.191277 ms** 在该等待之外。因此审计
不仅推迟启动，也产生大量额外非隐藏工作。checks、wait、Draft GPU outer intervals
三者联合覆盖 **24268.943147 ms**；剩余 **7542.758015 ms** 未由这三类覆盖，
包含 Target 及其他 CPU/RPC 成本，不能全归因于调度或 Python。

源码上的整批登记、反馈优先和 singleton repair 已由上述真实运行支持：这是
实际晚启动与额外串行修复，不再只是由 `started=full_accepts` 推测。源码位置见
[`gpu_backend.py`](../src/specrhythm/continuation/gpu_backend.py)、
[`eager_machine.py`](../src/specrhythm/serving/eager_machine.py)、
[`eager_owner.py`](../src/specrhythm/serving/eager_owner.py)。保留旧诊断的边界与
算术差额，任何新 GPU 收益仍必须由新提交实测证明。

## A-only：批量 admission

`begin_gpu_continuations` 在同一 CUDA owner 线程收集并验证整批 immutable work：
request/work 唯一性、owner、父 proposal、prefix/version、物化 frontier/logits、
base KV、context capacity 与已退休/重复身份逐项保留。全部验证成功后，对该稳定
物理状态执行一次 resident/control audit，再统一发布登记；无 GPU write，也不
修改物理 prefix/frontier/allocator。单请求接口委托同一批量入口。

`EagerSerialMachine.verify_start` 仍逐请求调用共享 core 的身份、资格及依赖校验，
随后一次性调用 physical batch enrollment。验证或审计失败时不发布任何新的 physical
work；owner 的既有失败/清理协议不变。没有将多轮历史 deepcopy 混入本轮优化。

没有缓存跨命令审计结果。owner 处理反馈、取消、结束或 shutdown 后，下一次 token
step 会重新审计；每个 GPU token step 的写前/写后审计保留。因此首 forward 前是
**1 次 admission audit + 1 次 step audit**，而不是 16+1，且不随本次 B 增长。
登记后仍优先处理已到达的反馈，Target 不等首 forward，不延迟 Target 制造重叠。

真实 owner/S2 backend/360-resident 回归覆盖 B1/B4/B16，均只访问全池两次；受控
Event 证明无父结果时可执行 eager、父结果先到时拒绝任务不启动。批量入口另外
覆盖末项错误 version/owner/frontier 和重复 request，证明不会留下前项的半批登记。

A-only 验证：Python 3.11 全库 **2068 passed、3 个既有 skip**（247.388 秒）；
Python 3.9 相关协议/owner/源码/报告器 **235 passed、零 skip**。Ruff、两版本
compileall、277 个源/测试文件的 3.9 语法、11 个 Bash 脚本及现有 Rolling Eager
runbook Bash 块、diff 检查均通过。未运行 GPU。

## B：后续独立父轮修复

A-only 保留原父轮逐请求修复路径，用于独立归因。第二提交再合并同一稳定状态的
父轮审计、兼容 correction/bonus rows 的 materialize 和写后 fence；保留 promotion
复用、完整依赖校验、EOS、取消及资源释放语义。两提交分别用新 root 测量，且各自
保留相同配置的 Serial 对照。

## 服务器证据与边界

[`eager_retest_report.py`](../src/specrhythm/serving/eager_retest_report.py) 是纯离线
报告器：只读取已结束的两个 B16 point，检查指定 source SHA、相同 workload、运行内
TP/device 身份，输出窗口内各用途真实 forward/B、每周期审计和 prefix union、
修复 GPU union、normal/recovery proposal host union，以及真实设备重叠。

现有服务日志没有精确 Queue.put 时间或 payload request ID；队列插入位于唯一
`eager_enqueue` RPC 的 `received_ns..completed_ns` 内。报告据此给出 enqueue 到
首个原生 GPU start 的上下界，并明确标为同一 step 的时间关联。缺失或不唯一时
返回 MISSING，不能用 host launch 时间代替 GPU start。原始 RPC 及设备端点保留。

单输入最多 512 MiB、总读取最多 1 GiB、每 point 最多 512 个周期，输出最多 64 MiB；
超过上限停止并保留原 root，不自动扩大扫描。输出是旧 root 外的新文件，不覆盖
原摘要或结果资格。报告的未覆盖时间仅表示现有 Draft host/设备观测未覆盖，不等于
设备空闲，也不强行归因于 CPU 或锁。
