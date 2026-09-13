# PingPong prepost3：CPU 验证与待测结论

2026-09-13，Draft PR #5；保留基线 `71f062c2a5fcef9298300904164a5abbde962a2f` 及之前历史。
没有连接 AutoDL，没有执行 GPU。上轮用户返回 Serial-prepost3 / Serial-eager-prepost3 约
87.98/100.59 tok/s、单窗口约+14.34%，仅作为已返回的 Serial 记录。新 PingPong 正确性、重叠、性能 **PENDING**。

## 源码和已执行证据

| 风险/行为 | 生产实现 | CPU 验证 |
|---|---|---|
| home与机会分离、原子claim、跨A/B/A、重复/过期消费 | PingPrePostMachine + Controller + resident Scheduler | `test_ping_prepost.py`、`test_ping_prepost_scheduler.py` |
| 真实 Target输出→adapter→owner→backend | PingPrePostProposer，原 no-bonus projection | `test_ping_prepost_integration.py` 使用 pinned vLLM 原版 bookkeeping；硬件/IPC替换 |
| 两种先后、反馈抢占下一token、其他home进展 | bounded PingPrePostOwner | `test_ping_prepost_owner.py` 用受控 Event，拒绝后只完成正在写的一步 |
| P1成功→P4、连续拒绝→P1→恢复P4，无三步eager恢复 | shared PrePostBackend + PingPrePostBackend | 真实状态机8周期对照、固定token oracle、逐行frontier与forward计数 |
| common+lookahead/普通扩展物理合批 | `_forward_prepost`、`token_step` | 同一物理CPU worker调用B2，真实allocator callbacks + fence；非逐请求B1 |
| full/runtime一致、resident360不重复扫描 | FixedAuditMixin + runtime allocator guard | `test_ping_prepost_audit.py`：两模式在混合回退/继续写入边界核对完整审计与增量证据，最终无block泄漏 |
| 关闭/资格版本、capacity延期、普通/refill最终进展 | owner决策快照，已有稳定promoted优先契约 | 有限输出优先任务结束后普通和新refill入场；不伪称无限请求公平保证 |
| EOS/截短、迟到反馈、单请求停止、未发布P2 drain | terminal及private-normal drain descriptor | 无未验证token输出，无失效复活，私有候选计数后释放 |
| 实际新模式路由/启动与初始化 | factory + config + initial_work | `test_ping_prepost_routing.py` 调用实际factory，TP2/GPU角色、active16/8及仅新B16门槛 |
| 连续预热机会可重复IDs | PingPrePostWindow + 新模式qualifier | 真实4次非空预热、排除提交量、起点仍要求16活跃；旧模式断言保留 |
| 采集→报告、原生映射与丢行拒绝 | ping_prepost_evidence | 实际owner/backend记录联结；合成native端点只证明区间算法，不是GPU证据 |
| 对照执行与联合correctness覆盖门槛 | control同协议 + GPU check coverage | control 5父轮=5公共+15普通扩展；缺少ABA/恢复覆盖被拒绝 |
| 一个去重包、首错和导出故障 | ping_prepost_delivery + shell EXIT trap | >20000行完整导出、65536预算后显式drop；投影后可重算同样机制；坏metadata原文保留 |
| 前台四阶段首错停止、父终端保留 | run_ping_prepost_b16.sh | shell外部GPU命令替身覆盖capacity/correctness/execution/measurement/evidence/export；保留原退出码 |

Target与Draft联合GPU入口要求真实全输出对Target-only相等、真实ragged P1/P4、ABA及拒绝后恢复轨迹。
CPU测试不会产生其PASS。吞吐或完全隐藏lookahead不是correctness阈值。

## 失败及修复记录

首次全库运行有3项失败和1项相关teardown错误，均保留为该次失败记录：

- 两个既有 owner failure 回归：新 bounded queue 的 timeout 参数误传到原 Serial 无界 queue调用，
  与原调用边界及测试拦截不兼容。修复为仅新有界queue传timeout，旧路径仍原样 `put(command)`。
  未提高timeout、未放宽故障断言。
- capacity manifest 断言仍将所有非旧PingPong模式判作总B Target。新增两种opt-in模式应为总16/Target8，
  已添加显式模式分支；保留所有旧模式容量期望。
- 早期新 scheduler 用例将A机会错误预期为正常B补位；按既有契约修正受控拒绝到B机会，
  验证promoted跨home+当前home普通填充。生产选择规则没有为测试改变。

修复后相关41项通过；随后全库先后2256和2261项通过（各3项既有skip）。
最终执行代码全库 **2263 passed / 3 skipped**（退出码0）；Python3.9相关 **64 passed**。
Ruff、两版本compileall、337文件Python3.9语法和git diff检查通过。
完整执行SHA为 `bc908be95d3ad611dc161a467709431bfe0c082a`；实际远程CI在交付回复记录。后续通过不改写首次失败，CI结果单独读取。
一次递归 Bash 检查误包含本地 `tmp/` 的上游 vLLM AMD CI脚本，该脚本使用Mac Bash不支持的 `|&`；
随后按仓库 tracked及未忽略新增脚本限定检查范围，18个仓库Bash文件通过。没有修改上游脚本或放宽仓库门槛。
本轮未执行的3个skip为 NVIDIA GPU opt-in 及两个Linux进程/zombie/subreaper特有用例。

## 服务器尚需回答

- 两新模式联合完整输出、实际KV/cleanup以及所需跨批恢复覆盖：PENDING。
- 普通跨cohort、lookahead和公共forward各自原生GPU重叠，以及总区间并集：PENDING。
- 真实B/P构成、先后顺序、exposed等待、容量延期、forward次数/耗时：PENDING。
- 吞吐与每步产出/耗时，无扣审计或日志的修正吞吐：PENDING。

成功或失败只需用户返回一个总包；字段、界限和固定入口见 [runbook](pingpong-prepost3-runbook.md)。
交付后停止，不自动扩展参数、换模型或继续优化。

CPU命令（未设置 `SR_RUN_GPU_TESTS`）：

```bash
SR_PHASE4B3_SOURCE_AUDIT=/tmp/rolling-eager-vllm-source python -m pytest -q
python -m ruff check .
python -m compileall -q src
git diff --check
```

完整测试使用Python3.11环境，相关64项另外用Python3.9执行。源码契约引用固定vLLM
`752a3a504485790a2e8491cacbb35c137339ad34`，CI的source job固定同一版本；
没有在本轮下载模型或连接GPU运行环境。

固定入口交付额外验证：脚本回归11项分别在Python3.11和3.9通过（含新增setup失败保码/父终端存活）；
交付不改src执行代码。加入固定入口后仓库Bash文件共19个；runbook独立子Bash同样做语法检查。
