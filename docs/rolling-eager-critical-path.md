# Serial-eager：A+B 后的启动关键路径

最新 `465c2159b4b58d8c0e79fc3f66f38083e88368cb` 服务器证据已返回：原始
Serial/eager 60.040693/45.305571 tok/s，原三项 PASS；原生 overlap 为
4.436982–4.585746 ms。首 forward 前两次全池审计平均 81.696735 ms，五个 eager
token step 内审计平均 269.218938 ms。当前工作是[显式审计分层](rolling-eager-audit-layers.md)，
下一 gate 改为[同 SHA 四点 full/runtime](rolling-eager-audit-runbook.md)，不改批级等待。
下面是 068c40a 的历史诊断记录；其原结果和当时的证据边界保持不变。

本轮以 `068c40a8ead1138568b7f846af348d3f732c28d4` 的用户服务器运行作为有效基准。
A 的批量 admission、B 的批量结算保持；不再维护或复测 A-only，不修改 PR #2/#3/#4。
新 GPU overlap/performance 为 **PENDING**，CPU 复杂度证据不证明服务器加速。

## 实际窗口对账

输入为 `serial-repair-B16-068c40a8ead1-20260912T143705Z-1486-complete-with-attribution.tar.gz`。
其中独立归因报告保留完整 Draft 原生 forward、每轮边界和 host union；小包保留实际
capacity、进程生命周期和原始资格。可复查数值与源归因文件 SHA256 见
[机器可读摘要](evidence/serial-eager-ab068-latency.json)。源结果没有修改。

| 窗口指标 | Serial | Serial-eager |
| --- | ---: | ---: |
| throughput，tok/s | 61.064037 | 38.162742 |
| 实际窗口，ms | 30607.213235 | 30448.545897 |
| 完整 steps | 39 | 25 |
| committed tokens | 1869 | 1162 |
| tokens/step | 47.923077 | 46.480000 |
| 完整 step mean，ms | 783.000175 | 1215.755993 |
| 完整 step P50，ms | 781.011223 | 1215.278544 |
| 完整 step P90，ms | 795.727229 | 1374.221927 |
| 所有完整 step 之外的窗口 wall，ms | 70.206410 | 54.646083 |
| pool audit/step | 4 | 15 |
| execution/measurement/cleanup | PASS/PASS/PASS | PASS/PASS/PASS |

`fixed_runtime.drive` 的 start 在 `engine.step()` 前，end 在 `commit_outputs` 完成后的
finally 中；其中包括引擎、跨进程 RPC、Draft 等待和输出提交，**不是 GPU 验证耗时**。
连续窗口从 ScanWindow measurement boundary 到循环实际退出，包括每步 checkpoint、
控制快照和检查。窗口吞吐保持 tokens / 实际窗口 wall，不扣除任何等待。
完整 step 总和与窗口的差额仅约 70/55 ms，排除了“主要差额在 step 外”的解释。
旧包缺少 coordinator 的原始 host spans，不能把这 70/55 ms 全称为 checkpoint；
新 `coordinator_checkpoint` span 和主线程互斥分解给出其真实覆盖。

生产 Draft 仍独占 physical GPU 0；Target rank 0/1 分别使用 physical GPU 1/2、TP=2。
证据来自每 point 的 `actual-capacity.json` 的 target_worker_ranks/ranks、Draft 原生
identity 与 `process-lifecycle.json`：eager Draft PID 8793，coordinator PID 8906，
Target worker 8956/8957。当前运行内三个 UUID 不同；模型仍为 Qwen3-0.6B/32B，
不增加设备、不进行跨次 UUID 相等比较。进程设备分离排除了两模型共用同一 GPU 的解释。

eager 的 25 次父轮修复保持批量，总 CUDA event 497.960938 ms，约 19.918438 ms/step；
125 次 eager token forward 共 2533.875 ms，约 101.355 ms/step；75 次 normal/recovery
proposal forward 共 1484.898438 ms，约 59.395938 ms/step。它们是按 host launch
选择的设备 event 累计，不能与跨进程 RPC/wait 或 CPU span 直接相加。
各用途真实 B 直方图见机器可读摘要，Target 全程 B16。

Draft prefix union 7210.511833 ms，block audit union 1587.527699 ms，JSON union
3912.408418 ms 与 prefix 嵌套。原 unhidden wait 9580.229982 ms 是 enqueue-feedback
到相关工作完成的既有 inclusive 定义；新 `owner_batch_gate` 另记 prepare 后至批次
可结算的边界，二者不得相加。归因报告的未被 Draft host 或任一已记录 GPU outer
区间覆盖部分为 15151.029191 ms，保留为未归因；不全部归因于 CPU、GIL、锁或空闲。

窗口 400 admissions、353 started、120 completed、280 parent rejection、99 promotion、
21 bridge mismatch、233 accepted promoted candidates、482 discarded eager tokens。
相比修复前“started=full_accepts”，A+B 已有更多任务在反馈处理前启动；这仍不证明
它们在 Target **GPU** 完成前启动。GPU 结束、反馈可用、owner 处理反馈是三个事件。

## 原始设备证据的缺口

服务器归因器当时用两个 Target rank 的 native 区间并集计算 eager overlap，报告
lower/upper = 0 ms；外层 UNKNOWN 保留，内部 ZERO 不忽略。但 v1 归因器只导出了
Draft native 区间，没有把用于计算的 Target native 区间写进包。因此这轮本地不能
独立重算 Target GPU 验证分布或画其精确起止，也不能给出真实 Target 结束时的
逐请求 token-step 进度。这是**导出缺口**，不是“服务器没有设备记录”。

[首、中、末实际轮次图](evidence/serial-eager-ab068-timeline.svg) 只绘制已返回的 Draft
设备上下界和 enqueue RPC bracket；Target 两行明确标 MISSING。没有画出未观测的
“及时但未完成”或“重叠后浪费”样例，也不把 host 区间当成 GPU 区间。新 v2 报告
保留完整 Target 双 rank 窗口区间及误差边界，修复该导出缺口；新证据包另外保留
有界原始 runtime/backend/service 文件，导出遗漏不再静默。

## 已确认与待验证的启动成本

测量轮 0/12/24 的 enqueue→first-GPU 区间分别为 121.892–129.452、238.391–241.120、
448.113–450.617 ms；全窗口 mean 240.811–243.914 ms、P50 225.418–228.173、
P90 390.669–393.208 ms。旧日志只将 Queue.put 包含在 service received..completed 内，
这些数值是界限，不是精确队列等待。

对应首 forward 前 prefix+audit union 约为 52.139、46.788、48.383 ms，次数都是
2 audits/720 prefix visits。启动延迟增长不能单靠这两次稳定审计解释。

基准源码确认 `core.evaluate_eager_eligibility` 在调用 provider 前 deepcopy 整个
RequestState；`start_verification` 与 `begin_continuation` 各调用一次。parent receipt
和 promotion 还分别调用；历史 proposals/continuations 没有在每轮删除，且各自保留
长 prefix tuple，因此复制访问随轮数和 prefix 增长。当前 `EagerRequestState` provider
契约只有 `request_id`，其余被复制历史不影响现有静态资格决策。

真实 core 的 32 轮 CPU 回归保留了登记及每轮四次 provider 调用、记录历史对象数量和
复制 host span，验证复杂度增长。CPU observation on/off 数据单独保存于
[CPU 对照](evidence/eager-eligibility-observation-cpu.json)，不作为服务器耗时或 GPU
重叠证据。服务器上的复制耗时占比，以及减少复制后能否赶上 Target 验证窗口仍待复测。

没有队列/锁/GIL 原始等待证据支持其为主要根因。反馈优先维持：已到达拒绝会合法取消
尚未启动的工作，它可能是晚启动的后果，不能为制造 overlap 延迟反馈。批级
`WAITING_DRAFT` 会让已可修复/复用请求等待同批依赖；新记录列出 dependency work IDs
和已 ready 的 request IDs，但本轮不改变该策略。它影响反馈后的路径，不能单独解释
首次 forward 之前的历史复制成本。

## 观测提交：不改变 A+B 执行逻辑

观测提交为 `33b6588004376e930ebabe1c9c768eca0b025371`，直接继承 A+B。

`SR_EAGER_CAUSAL_TRACE=light` 启用默认关闭的有界内存记录，每进程最多 20000 行；
没有逐 token 写盘、设备同步、跨模型 barrier 或为观测添加的锁。每条记录含 host
monotonic ns、PID/thread lane，request/round/proposal/continuation/batch 信息按实际
来源保存；cap、dropped rows、missing、校准误差显式输出。

- core：provider snapshot/provider 调用、两次准入检查、父结算及 promotion。
- owner：payload copy、queue submit/put bracket、ACK、dequeue/dispatch、response wait、
  batch gate 的阻塞与 ready 请求、候选返回。submit→dequeue 包含 Queue.put 开销，
  不谎称原子插入时刻；真实 socket send/receive 使用既有 transport timestamps。
- machine/backend：批量依赖检查、全池 audit、bridge/token step 0 与四个 candidate
  step 1..4、KV 修复、普通/恢复起草批次、completion/abort/promotion。
- Target：verification-start hook、enqueue RPC、反馈可用 hook 和反馈 RPC；实际 GPU
  时间仍只来自既有 CUDA device recorder，经整数锚点映射，不新增 fence。
- coordinator：engine.step、output commit 和 checkpoint；主线程使用互斥区间分解，
  子进程与同进程其他线程用独立 inclusive lanes。未覆盖时间保留 unaccounted。

`eager_retest_report` v2 支持指定一个或两个 ended B16 mode；保留两 rank 的独立
设备报告和并集、逐轮进度上下界、首 forward 相对 Target 开始/结束的位置、normal/
recovery/mixed 成本、所有 bounded causal spans。缺少 trace 不会被写成“零 CPU 成本”。
混合批 GPU 时间不能任意分摊到 normal/recovery 请求。

## 独立的最小修复及边界

性能提交只把 provider 的整状态 deepcopy 替换为独立、冻结的 request identity view，
满足现有公开 provider 契约；不暴露 RequestState 或可变历史，不缓存决策，不降低版本
检查。仍在原来的调用点调用 provider，动态 enabled 开关及 decision_version 语义不变。
公开 `state()` 防御性 snapshot、全部 KV/prefix/proposal/依赖审计保持。

具体实现为 `policy.EagerRequestView(request_id)` 冻结 dataclass；`core` 每次重新
构造，仅引用不可变字符串，不包含 prefix、proposals、continuations 或 accounting。
provider 即使故意用 `object.__setattr__` 绕过冻结保护，也只能修改自己的新视图。
provider 返回错误类型、版本倒退或同版本改变决策仍失败；更高版本的关闭/重开照常生效。

执行顺序不变：enqueue/ACK → owner dequeue → 逐请求 start/begin 依赖与资格检查 →
批量 enrollment 审计 → 第一次 eager GPU forward；期间已到达反馈仍优先处理。
修复只将上述两处资格检查内部的整状态 deepcopy 换成固定字段视图，父反馈及 promotion
的另两处也使用同一个视图路径。没有让 Target 等待首次起草，也没有改变 owner 队列、
GPU launch、批量 materialize/fence 或批级 `WAITING_DRAFT` 的边界。

32 轮真实 core 回归用 deepcopy guard 验证：历史继续增长、129 次 provider 调用保持，
仅登记的公开返回值复制一次 RequestState；此后热路径不复制 RequestState。每个 snapshot
的 copied_history_objects 都为 0，提交量仍为 160 tokens。独立 provider 隔离、版本及
开关回归覆盖这次接口收窄；既有真实 owner/adapter 回归继续覆盖两种反馈顺序、拒绝恢复、
bridge mismatch、迟到、取消、混合批 KV 修复、记账和清理。

同一 32 轮 helper 的三次 CPU 样本中位数，观测版本 off/light 为
202.799/199.657 ms，视图版本 off/light 为 9.241/11.571 ms，原始样本见
[修复 CPU 记录](evidence/eager-eligibility-view-cpu.json)。这是少量、非交错配对的
本机合成样本；观测版本的轻微倒挂说明噪声，不能宣称 light 免费。视图版本的 light
样本约多 2.330 ms/32 轮，只覆盖这个 core helper，不涵盖生产 RPC、报告序列化或
落盘成本。回归验收使用结构性复制 guard 和事件协调，不用这些耗时作脆弱阈值。
服务器复制占比和 GPU 重叠改善仍为 PENDING。

本轮不再合并更多审计、不新增增量 KV 校验、不移除 fence、不改批级恢复策略。它们是否
仍为主要瓶颈，需要新同口径服务器时间线。正式吞吐点统一 buffered-live/bound-prefix
加 light causal；不使用重度 profiler。观测自身影响由 CPU 对照和仅观测服务器点报告，
不能把旧 off 与新 light 的差异全部视作执行优化。

观测版本本地验证：Python 3.11 全库 **2099 passed、3 个既有 skip**（245.067 秒）；
Python 3.9 相关回归 **248 passed**、另两个既有源码契约文件 **60 passed**，均零 skip。
Ruff、两版本 compileall、287 个源/测试文件的 3.9 语法、12 个仓库 Bash 脚本、
8 个 Rolling Eager runbook Bash 块和 diff 检查通过。最后的批标识及父轮绑定补全
还通过定点 owner/report 回归。首次全库因新建 venv 未放入 PATH 导致两项既有 shell
测试找不到 `python`；补齐 PATH 后原断言通过，没有调整测试预算或放宽断言。

视图修复版本本地验证：Python 3.11 全库 **2102 passed、3 个既有 skip**
（256.13 秒）；Python 3.9 的同组相关回归与五个源码契约文件合并 **311 passed**
（13.763 秒、零 skip）。Ruff 全库、两版本 compileall、287 文件的 3.9 语法、
12 个 Bash 脚本、8 个 runbook Bash 块和 diff 检查通过。服务器 GPU 三点尚未执行。
本地测试与 GitHub CI 分开报告；提交后的实际 CI 链接及状态见 PR 交付记录。
CPU on/off 三次中位数 202.799/199.657 ms 落在此次样本噪声内，不能据此宣称观测
零开销；服务器观测成本仍待仅观测点量化。

提交 SHA 在交付时记录；复测流程见
[仓库 runbook](rolling-eager-latency-runbook.md) 和
[前台脚本](../scripts/run_eager_latency_b16.sh)。用户在 AutoDL 执行，本任务不连接服务器。
