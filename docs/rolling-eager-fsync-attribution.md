# 原子 JSON 的 fsync 文件归因修复

本轮只修观测采集与首错交付；不改变同步落盘策略或 eager 执行策略。PR #5 保持 Draft。GPU correctness、overlap、performance 的新版本复验均为 **PENDING**。

## 返回证据与失败层次

只读解包于独立临时目录，未改写旧 root 或旧结果：
`serial-execution-B16-c8165ccb93cc-serial-runtime-20260913T072054Z-1613-control`。

| 输入 | SHA256 |
| --- | --- |
| `…-control-failure-small.tar.gz` | `5dddbbb4e5437ee0748e3ee50f714ae55216795694e541b7494ac7c728aae6d2` |
| `…-control-failure-evidence.tar.gz` | `592c9f774cbd33bcb8e5d435c993b0db8dfc4314e78889f534b1022d18104980` |

原始 `light-summary.json`：capacity、execution、measurement、cleanup 全部 PASS，effective_exit_code=0，errors=[]；独立 GPU correctness result 为 valid=true、physical_gpu_correctness=PASS、cleanup=PASS。窗口为 `[7675073703806681, 7675104346048107]` monotonic ns，共 30642.241426 ms，45 steps、2164 committed tokens、70.62146564003774 tok/s。四个 producer 的 causal trace dropped_rows 均为0，measurement_status 均 COMPLETE。

原始 `evidence-status.json` 的唯一错误是 `fsync file attribution incomplete`，failure_layer=diagnostic_evidence。直接从 runtime 的 Target rank 0 `host.intervals` 按原窗口裁剪，得到45次 log_fsync 未提供 log_name，区间并集123.609629 ms；rank1没有未归因事件。与 compact 报告相符。该运行有效，诊断不完整；这不是 GPU 执行失败、性能优化失败或 trace 溢出。只有第一个控制版 Serial 点的材料；按首错停止脚本及用户终端反馈，后续三点未启动，不为它们生成结果。

本轮没有收到单独路径的新增终端文本；终端事实采用用户消息，并由两个包内的状态、原始事件与脚本停止顺序交叉核对。旧运行未知文件名保持 MISSING_FILENAME：不能把下面的源码/CPU 因果证明反填成历史事件身份。

## 实际调用链与修复边界

`S2SerialProposer` 与 `EagerSerialProposer` 均继承
`phase4/vllm_remote.py::RemoteDraftProposer._write_report()`，该方法在 rank0 调用模块已导入的 `atomic_write_json(self.report_path, …)`。`phase4/manifest.py::atomic_write_json()` 顺序为：创建临时文件、JSON 序列化、末尾换行、flush、fsync、关闭、os.replace 到最终路径。旧 `fixed_observe.wrap(os, "fsync", "log_fsync")` 只读取 `fixed_logging.IO_CONTEXT.name`；只有 JSONL adapter/flush 建立此上下文，原子 JSON 路径没有覆盖。

CPU 回归使用两个真实 proposer 类的继承方法及原始 imported alias，仅省略 GPU 构造。令写入上下文为 nullcontext 可在该实际链路复现 MISSING_FILENAME 并被生产 qualifier 拒绝；恢复上下文后，实际 Timers 原始记录经过 JSON 序列化、execution_path 汇总与 CLI qualify，得到 COMPLETE。该试验确认源码存在且修复了这条记录缺口；旧 GPU 事件没有 fd/路径字段，仍不能逐条证明其文件就是某个 report。

新增底层 `specrhythm/io_context.py`，只依赖标准库。写入处为实际 sync 设置线程局部上下文，观察器读取副本，无反向 serving 依赖、/proc 查询、栈抓取、额外磁盘日志或 GPU 同步。函数体内部使用上下文，因此启动前已存在的 `from … import atomic_write_json` 别名也受覆盖，不依赖重新绑定别名。

| 实际写入处 | write_kind | final / physical 路径 | 同步及发布语义 |
| --- | --- | --- | --- |
| `manifest.atomic_write_json` | atomic_json | 目标 / `.{name}.{pid}.tmp` | 保留 flush→fsync→close→replace |
| `process_lifecycle._atomic_json` | atomic_json | 目标 / mkstemp 实际路径 | 保留失败临时文件清理及异常传播 |
| `transport.CheckpointJsonl.append` | checkpoint_jsonl | 同一 JSONL 路径 | 保留 checksum、逐条 write/flush/fsync |
| `fixed_logging.DiagnosticLogs._write` | buffered_checkpoint_jsonl | 同一 JSONL 路径 | 保留原 buffer 边界和每文件 sync |
| `batched_draft_service.write_immutable_report` | immutable_json | 同一报告路径 | 保留 exclusive create、单次 sync |
| `reference._exclusive_freeze` / `_exclusive_copy` | immutable_json / immutable_copy | 最终冻结文件或复制目标 | 保留独占创建、权限、失败清理 |

这些覆盖当前 serving/continuation/phase4 源码里的所有直接 fsync 写入点。phase3 离线学习/trace 写入不在此生产调用链，本轮不修改；若未来进入观测窗口且没有上下文，仍被严格标为缺失。`s2_pool.publish` 本来没有 fsync，本轮不为它新增同步或虚构 fsync 事件。

记录保留 `log_name`，并新增 `log_path`（调用者实际目标路径）、`physical_path`（真正同步的文件，可为临时文件）、`write_kind`。不增加新的 span；在原有每次 fsync span 中携带字段。聚合按 producer 和完整路径/类别分组，不因同名文件合并身份；时间仍是每 producer 的窗口裁剪并集，不累加为跨进程墙钟。

嵌套和异常用 finally 恢复上下文；线程局部状态不跨线程；无长期 fd 映射和缓存。fsync observer 保留原有幂等安装，CheckpointJsonl adapter 补充幂等标志，重复安装不会再包一层或再次 capture/写入。JSONL 原有外层上下文仍保留兼容，实际 native writer 的内层上下文提供精确类别。

## 首错与验收

`execution_evidence.qualify` 保留严格 MISSING_FILENAME 拒绝，在 status 中附带缺失 producer/count/union_ms，并保留 original_qualification、原运行详情与独立 performance_conclusion。旧具名事件仍可读取，缺少新增类别/物理路径时保持 null，绝不猜填。

`execution_failure` 优先打印诊断层次、原运行资格、缺失明细和所有证据路径，再写新的独立 failure-summary sidecar。对老 status 可从旧 compact 报告读取已明确标缺失的行，不猜文件名。读取有界（每报告≤8MiB、最多16个 light-summary），缺失/截断/解析错误明示。原 root 中的运行状态不变。

pair 脚本保存第一退出码与 stage，先输出首错；不再跟随容易掩盖首错的泛化 status/errors 命令。独立尝试 small、raw 两个导出，次生退出码与输出分别保留在 `failure-export-status.txt`、两个 `failure-*-export.log`，不会替代第一退出码。raw 包包含 failure-summary、small-export 结果和已完成的 export-status 部分；raw 导出自身最终退出码要在包关闭后才能写入外部 status，不能声称包中已经包含它。导出成功与证据完整性是不同判断。

## 对照版本与验证

两个新版本都包含本次相同观测修复；控制辅助 ref 从 c8165ccb93cc07d81f684cb48f270da837d6e75b 派生，主分支保留 2f436a14e1284164f04aa892cca5cdfb8c048301 与 68c30b1340e1e4dda6ba017dbba29f4f214c6595。二者 `src/` 差异应仅为旧 2f436a1 的 Serial admission schema/consumer 有界 buffer 变更，不引入其他执行差异。使用 sibling refs，不重写任何历史；runner 接受有共同祖先的两个明确提交。最终完整 SHA 及四点命令见 [runbook](rolling-eager-execution-runbook.md)。

CPU 覆盖真实 writer→observer→raw report→qualify、旧缺口复现、未知 fsync 拒绝、JSONL 两种落盘方式、嵌套/异常/线程隔离、重复安装、write/fsync/replace 失败不发布新快照、原子内容和 flush/fsync/close/replace 顺序一致。shell 回归覆盖首点诊断失败停止，以及 small/raw 同时导出失败仍保留第一退出码和父 shell。

优化路径 CPU 全量 pytest 2182 passed / 3既有平台或GPU skips；Python3.9 针对性回归53 passed，Ruff、compileall、Python3.9 AST303文件及19个仓库Bash脚本语法、git diff --check 通过。控制路径全量结果及新CI在交付时更新。既有 CI 失败单独记录，不由本轮归因修复解释。旧 c816 控制 push CI 的 refill/owner deadline 失败、2f436a1 push CI 的同一 refill 断言失败，与本次服务器 diagnostic_evidence 失败属于不同证据。68c30b1 的 push/PR CI 已通过，也不等于旧失败根因已确定。
