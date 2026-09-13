# PingPong prepost3：CPU 验证与待测结论

## 2026-09-13 非 scan correctness 报告与联合首错修复

从 `e30a338629feb2111377fbad66cd595ec929c1cf` 继续，执行定位基线为
`429956febe6d731291c1a3c0ee0857337a3c54c2`。只读按logical_paths复核
`pingpong-prepost3-delivery-20260913T150541Z-1615.tar.gz`：61个逻辑文件的大小/SHA256全部匹配。
两容量点及设备绑定PASS；Target-only完整输出完成；普通PingPong输出/cleanup完成，进程码0，
报告qualification失败。旧FAILED/INVALID不变。按request ID比较16×32=512 token完全一致，
但eager correctness及两性能点未启动，整体联合correctness未通过，export COMPLETE仅表示导出成功。

实际失败目录：`joint/pingpong-prepost3/runs/joint-correctness-pingpong-prepost3-B16-A-20260913T231055-7702917909212099`。
原错误为mode=pingpong-prepost3、stage=correctness、runtime.json、TP0/1、缺probe，层为report_qualification。
外层误用容量循环末尾eager的POINT/ROOT；joint/failure.json也只保留泛化joint_execution层和命令码1。

根因：`fixed_runtime.drive()` 把顶层probe和decode_scan放在同一个条件字典中；非scan correctness
只生成point.probe=false，没有顶层probe。上轮CPU fixture手工提供了probe，并从scan形状的记录
调用qualifier，没有执行真实非scan报告构造。现在probe在公共报告部分无条件生成；run/drive先检查
真实bool实参与point及阶段冲突，qualify_run_kind严格核对顶层、point、runtime_mode和stage。
fixed_results对新模式及联合Target-only参考使用该规则，不将缺失、0或字符串转换为false。

只读定位中另构造过内存派生视图：**仅加入原point.probe=false作为顶层probe**。其余原数据不变，
设备契约可完整核对213次请求验证、35个TP双rank步骤，未发现下一个缺失字段。
该派生检查不代表原运行通过，没有回写旧报告，更没有把512 token一致判成整个联合PASS。

| 阶段 | 运行类型字段 | 其余必需/不适用证据 |
|---|---|---|
| 新模式capacity | 顶层/point/实参probe=true，scan=true | decode_scan、真实准备/释放、startup/final设备及容量；decode verification与hooks为零 |
| Target-only及新模式correctness | 顶层/point/实参probe=false，prepost_correctness=true，scan缺省false或false | decode_scan不适用；自然完整输出/释放；新模式仍要求TP native/request/proposal/hook关联；最后另做三模式输出及轨迹覆盖 |
| 新模式performance | 顶层/point/实参probe=false，scan=true | decode_scan必须存在；原预热/30秒窗口和设备关联、执行/测量/清理/诊断门槛 |

共同设备来源仍为actual-capacity startup、runtime最终快照/设备/steps、Draft native identity。
旧Serial/Serial-eager/PingPong非联合资格规则保留；不增加GPU查询、fence、热路径全池审计，
不改3+1、P1/P4、A/B admission、反馈、KV、模型、采样或预算。

联合runner现在从当前模式的run_point阶段文件、异常directory、light-summary/result读取真实首错，
记录outer_stage、mode、run_directory、failure_layer、primary_error、effective_exit_code及command_exit_code。
脚本进入joint前清除容量POINT/ROOT，失败heredoc读取joint/failure.json；未启动模式不会被标成失败运行。
读取/写入摘要的异常只作为secondary错误打印，保留首次异常/码。总包增加stage.json来源，保留
joint结果/失败、比较、原始报告/native及first-failure。inventory记录export_validation_exit_code；
终端另打印实际export进程码。包关闭后的sidecar/磁盘失败不能回写已关闭的包，不把校验0当最终进程成功。

新CPU回归实际执行drive、ServingClock、PingPrePostController、commit、settle及startup/snapshot/target_report，
只替换模型/设备/IPC响应；生产write_once序列化后调用qualify_prepost和correctness summarize。
两模式×三路径均导出→总包读取→同一契约复算；原始429956完整drive在同一CPU链中复现缺probe。
联合测试保留真实run_point和阶段发布，三失败位置×进程/报告失败使用真实报告和脚本原heredoc，
故意提供过期容量变量仍定位正确；后续导出拒绝41不替换首错1/23。

开发失败记录：新CPU fixture起初缺drive需要的capacity metadata、复用了arrival-output-events文件，
现用生产capacity构造及新输出目录；heredoc用例预写first-failure触发禁止覆盖，现由真实heredoc首次写入。
Target原缺字段文本含引号，测试保留其原始内容。未放宽断言、不可覆盖约束或timeout。
最终全库 **2324 passed / 3既有skip**、退出码0；Python3.9相关 **71 passed**，Python3.12
报告/真实drive/联合首错相关 **58 passed**。Ruff、3.11/3.9 compileall、342文件Python3.9语法、
19个仓库Bash及runbook语法、diff检查通过。CPU替身不产生GPU正确性/重叠/性能PASS。

另外复核了上一e30a338提交的实际CI：PR的3.12失败是旧复现测试要求`<listcomp>`帧名；3.12实际
将同一异常报告在外层帧。测试现严格检查原始出错源码行和KeyError内容，3.9/3.11/3.12均验证。
旧push的3.12还出现refill请求集合与drain deadline失败，本轮未改这些策略/预算；原因未据此确定，
本地重跑成功也不等于已解释旧失败。旧PR3.9为CANCELLED，push3.9及两个contract任务成功。
本轮提交CI另按推送后的实际状态报告，不沿用旧任务成功。

## 2026-09-13 首次容量 probe 的设备证据契约修复

从当前 `8c24f097db419793063bea898e828fc63a50e801` 继续，没有 reset 或覆盖提交、用户改动。
只读核对 `pingpong-prepost3-delivery-20260913T135137Z-1665.tar.gz`：17个逻辑文件的大小与 SHA256
均匹配清单。唯一启动的是 `bc908be95d3ad611dc161a467709431bfe0c082a` 的 `pingpong-prepost3`
容量 probe。原始 capacity/cleanup PASS、effective_exit_code=0、probe=true、capacity_probe、
target_steps=[]；报告 qualification 失败。原 execution=FAILED、measurement=INVALID、错误
`'dual_uuid_query'` 和外层首错码1原样保留。不是 OOM、模型输出错误或性能下降。
第二容量点、联合 correctness、两个性能点均未启动；GPU correctness/overlap/performance 仍 PENDING。
提及的终端文本没有独立可访问附件；以上复核基于总包，不宣称核对过该独立文本。

根因是生产/消费契约不一致。`fixed_observe.target_startup()` 仅旧 `pingpong` 调用
`initialize_pingpong_worker()`；`s2_runtime._target_capacity_snapshot()` 同样仅旧路径写
`dual_uuid_query`。新类链为 `PingPrePostProposer → PrePostProposer → EagerSerialProposer →
S2SerialProposer → RemoteDraftProposer`，执行 Serial 派生设备检查及 hooks。
旧 `decode_scan_results.summarize()` 却对所有 PING_MODES 读取 Dual 专属字段，原执行提交第403行
列表推导首次抛 KeyError。旧断言本来允许0次验证，根因不是 probe 禁止零计数。

`tests/fixtures/device_contract/bc908be-summarize.py.txt` 固定原生产函数及来源SHA。CPU回归运行真实
startup、snapshot、DeviceTimeline、Target hooks、target_report，再调用原 summarize，捕获首个异常
确在该列表推导；相同生产产物交给修复后的真实 summarize 通过。GPU/时钟/传输/预填充模型是CPU替身，
没有替换报告生产函数或 qualifier。旧包 runtime 投影省略了 `target_final_memory`，不能离线恢复
原始最终快照；本次复现不是给旧包补字段。包内 startup 仍可独立核对 TP0/GPU1、TP1/GPU2、Draft/GPU0。

### 修复后各模式契约

| 模式 | 实际路径 | probe | correctness / performance |
|---|---|---|---|
| `pingpong` | 原Dual initializer / DualBatchRemoteProposer | live、initial=1、cache=0、access=subprocess=0 | 原live计数等式：每rank访问/子进程查询数=实际验证step数；未放宽 |
| `serial-prepost3` | PrePostProposer / Serial hooks | 真实启动/结束TP2绑定、Draft隔离与容量、零decode step及hook；Dual计数不适用 | 同一绑定链 + rank0每请求start/end计数 + 两rank原生forward/request/proposal关联 |
| `serial-eager-prepost3` | 同上，已有eager owner | 同上 | 同上 |
| `pingpong-prepost3` | PingPrePostProposer / Serial hooks | 同上 | 同上，保留PingPong容量/轨迹门槛 |
| `pingpong-eager-prepost3` | 同上，已有eager owner | 同上 | 同上，保留PingPong容量/轨迹门槛 |

新 `device_evidence_contract` v1 在已有快照边界读取实际 proposer MRO 路径、tp_rank、hooks_seen
和 Dual 不适用原因；没有新增 UUID 查询、RPC、fence、全池审计或伪造PASS。rank0 hooks 按请求累计，
rank1 没有该 per-request hook，依靠其原生forward证明；不可混为同种计数。setup/prefill GPU forward
在 probe 合法，要求为零的是 decode verification。正式关联检查覆盖全部decode阶段，性能窗口边界不变。

离线 `device_contract.qualify_prepost()` 将 actual-capacity startup、runtime final和native identity
绑定到同一运行的物理设备；检查参数驻留、正容量、三UUID隔离。复用 `device_batches()` 与
`rounds_by_prefix()` 关联原生TP forward、内部请求、稳定请求/prefix、候选数与实际proposal tokens。
原执行、真实准备、释放及运行时KV/ownership检查保持。缺字段、rank缺失/重复、错设备、声明路径不一致、
缺forward或proposal错配都失败；错误包含mode、stage、artifact、rank、field、expected，missing不当零。

总包新增保留 `runtime.target_final_memory`、`target_requests_final`、`diagnostic_drain`；
actual-capacity、native来源继续保留。清单记录源字节/hash及投影字段，缺最终快照时显式
`required_fields_missing`、export INCOMPLETE，仍导出可用原文，不重建证据。
生产→报告→导出→从总包读取→同一契约校验的回归要求结果一致。
first-failure 保留 report_qualification 层、原始进程码与cleanup状态；终端首行将capacity称为stage，
随后打印结构化真实失败层。导出错误与首错分开，成功/失败只要求一个总包。

没有改动 token/forward协议、A/B admission、反馈优先、调度或运行预算。首错测试最初漏调生产
`retained_report()`，cleanup_status因此为None；补齐调用链后要求同时保留cleanup PASS、进程码0、
qualification FAILED、外层码1。没有放宽断言或超时。第一次全库保留2293通过、1失败、3既有skip；
失败就是上述测试链遗漏。修正后相关93项通过，再补充联合correctness入口负例与无新增查询检查。
最终全库 **2296 passed / 3 skipped**，退出码0；Python3.9相关 **49 passed**。
Ruff、3.11/3.9 compileall、340个Python文件的3.9语法、19个仓库Bash及runbook子Bash语法、
`git diff --check` 均通过。未提高预算或timeout。交付前基线8c24的push/PR两组CI均SUCCESS；
本次提交的CI状态另在交付时按实际结果报告，不沿用基线成功。

本次执行修复SHA：`429956febe6d731291c1a3c0ee0857337a3c54c2`。
其子交付提交只固定该SHA的启动器和文档，没有src差异；服务器仍只运行两个PingPong prepost3点。

以下保留上轮实现当时的验证记录。

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
