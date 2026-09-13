# Runtime B16：证据控制与 admission 落盘优化

本轮只比较两个版本，各自运行 Serial/runtime 和 Serial-eager/runtime。所有点使用同一冻结360请求、B16/K4、两轮预热、连续30秒窗口、buffered-live/bound-prefix，setup900秒、drain60秒；每点先容量和三请求物理 correctness，之后才执行性能点。无sample12、无网格、无跨运行UUID相等条件。

`scripts/run_execution_pair_b16.sh FULL_SHA` 在当前指定 checkout（`SR_EXEC_REPO`）运行一对，并强制检查 HEAD、干净工作区和冻结配置。旧的 full/runtime 四点脚本及旧结果保持原样。交付时的两版本入口及完整 SHA 命令见本文末尾。

## 独立控制版本的记录约定

显式 `SR_EAGER_CAUSAL_TRACE=light SR_EAGER_CAUSAL_LAYOUT=phased`；旧默认仍legacy20k/off。
每个进程一次运行的 host trace预算为 setup8192、warmup4096、measurement65536、drain8192，总86016行。阶段来自现有控制读取/发布返回值的 `diagnostic_phase`，不新增RPC或控制文件读取；不会使用它来决定执行。span使用入口阶段，跨界span仍保留完整时间端点；迟到旧snapshot不回退观测阶段。没有新增设备同步、全局锁、后台写线程或逐token落盘。

阶段预算相互独立。每阶段报告 attempted/retained/dropped、保守的 dropped区间、实际观测到阶段切换的时刻。全生命周期一旦有drop仍标TRUNCATED；窗口完整性另按实际monotonic窗口检查所有阶段丢失区间是否与窗口相交，不能只看measurement桶。首尾setup/drain缺失仍报告，绝不写成全历史COMPLETE。phase标记延迟期间的事件由旧阶段预算保留；跨界丢失不能通过窗口资格。rank1没有相应host事件时仍明确保留阶段元数据和原生设备区间，不构造owner事件。

新增 per-file fsync 标记、实际 Target enforce_eager/graph mode，以及 Target完整forward诊断函数、sampled-result入hook、feedback payload就绪边界。其 inclusive span含真实复制、校验和序列化，不等于纯观测计时器开销。事件仍只在结束报告序列化一次。

## 输出及失败层次

每点单独root。原运行状态保留在 `light-summary.json`。新 `ROOT-audit-report.json` 复用有界分析器，加入 `execution_path` 的每轮反馈延迟、promotion结果、恢复forward/B、promotion-ready等待、补位forward和按文件fsync；已有吞吐、完整step分布、两个Target rank/TP并集、native overlap、审计/hash/JSON和线程互斥分解保留。

`ROOT-evidence-status.json` 单独写 `original_qualification`、`diagnostic_integrity`、`failure_layer`、各producer的drop/phase范围及缺失原因。原执行PASS但trace不完整时，这里FAILED并停止后续点；不回写原结果为执行失败。报告不以速度或overlap正值作为正确性门槛。

失败 trap 保存第一退出码，有限status/errors后尝试导出小包及原生证据包；导出失败不会覆盖原错误。包清单按文件标记MISSING/OMITTED_LIMIT/unstable，不把“压缩包存在”当作完整证据。分析单源≤512MiB、每点≤1GiB、输出≤8MiB，原始大事件仅在raw中一次；两/四点比较只合并compact报告。不重复嵌入原始事件。服务器脚本在独立子Bash里运行，由外层if捕获失败，停止后续点而保留交互终端。

## 判读

先分别确认物理correctness、执行/测量/清理、证据完整性，再比较机制与性能。两版本都必须重测Serial：admission日志是公共路径，不能拿旧Serial与新eager宣称算法收益。优先核对coordinator admission fsync次数/时间、feedback端点及每轮3次recovery是否变化。本轮没有修改恢复策略，因此不能预设forward减少。

一次窗口用于排查机制，几个百分点差值不作为稳定收益。后续可在用户指示后指定新 `SR_AUDIT_RUN_TAG` 重复，或显式选择反向版本顺序进行交错；默认不自动重复或扩展参数。GPU correctness、overlap和performance在用户回传本轮证据前均PENDING。
