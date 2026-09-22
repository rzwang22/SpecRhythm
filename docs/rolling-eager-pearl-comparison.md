# Serial-eager 与 nano-PEARL：源码及 987ef3 四点复核

本报告只读复核 `987ef3ec7df647e0850c31f14235b401e11eaa1a` 的四点，不修改历史结果。
参考源码固定为 nano-PEARL **`7d020b6ce70d965c85d1bc08248c8af9c22e81cd`**（2026-09-13 从公开 master 解析并只读 clone，未安装或运行）。

## 参考源码对照

以下链接均为固定提交；结论来自代码而非 README。

| 类别 / 问题 | nano-PEARL 实现及依据 | SpecRhythm 当前实现 / 含义 |
|---|---|---|
| 测量：speedup 分母 | [bench.py](https://github.com/smart-lty/nano-PEARL/blob/7d020b6ce70d965c85d1bc08248c8af9c22e81cd/bench.py) 79–101；[eval_benchmark.py](https://github.com/smart-lty/nano-PEARL/blob/7d020b6ce70d965c85d1bc08248c8af9c22e81cd/benchmark/eval_benchmark.py) 92–169：PEARL tok/s 除 AR tok/s | 当前比较对象是普通 Serial speculative decoding，不能把 PEARL/AR 的比值解释为 eager/Serial 的收益。 |
| 测量：计时及停止 | [runner](https://github.com/smart-lty/nano-PEARL/blob/7d020b6ce70d965c85d1bc08248c8af9c22e81cd/nano_pearl/pearl_engine/pearl_model_runner.py) `parallel_generate` 393、`pearl_generate` 413、`pearl_bench_generate` 440：timer 在 prefill 前；bench 在 prefill 后把 max_tokens 改为 1e8、ignore_eos 改为 True，固定 100 个 PEARL step（eval 可配置）；AR 仍依原 SamplingParams EOS/max_tokens 结束 | 两者都包含 prefill，但 bench PEARL 与 AR 的停止条件/输出长度不一致。SpecRhythm 是 resident360 全部真实 prefill 后、两轮预热后连续 30 秒 decode，保持 EOS/请求输出上限；不能搬用参考计时。 |
| 测量：token 分子 | runner bench 返回 running sequences 的 `completion_token_ids`；[engine](https://github.com/smart-lty/nano-PEARL/blob/7d020b6ce70d965c85d1bc08248c8af9c22e81cd/nano_pearl/pearl_engine/pearl_engine.py) `bench_generate` 汇总长度 | 源码还会把下一轮未验证 lookahead 附加到 Target sequence，bench 返回前没有最终验证/裁掉该 suffix。因而这个分子不能当作 SpecRhythm 的 authoritative committed-token 计数。这里只指出源码口径，不推断原作者某次实验的误差大小。 |
| 算法：并行与通信 | runner Draft `pearl_step` 492：连续 gamma 次 decode/argmax；Target 590：同轮一遍 packed forward；随后 `verify` 以 NCCL 广播交换 Draft 验证 tokens 和最后 gamma 个 next-round-input，再全局广播 4×B 的 accept/rollback/correction/finish | SpecRhythm Target verification-start enqueue 后独立 owner 起草；Unix socket JSON RPC、队列和父依赖结算。反馈先到时优先处理，绝不让 Target 等 Draft 首发以制造重叠。 |
| 算法：pre/post verify | runner 512–694：pre 模式验证一个 token，成功转 post；post 验证 gamma 个旧候选，同时 Draft 生成下一块。全接受时保留下一块；拒绝则回到 pre 并安装 correction | PEARL 没有“父 K4 全接受后，Target bonus 必须等于提前生成的 bridge”这一提升门槛，也没有该处的额外 Target bonus。不是删掉当前 bridge 检查就能等价移植。 |
| 算法：提交与回退 | runner `Draft.postprocess` / `Target.postprocess`：post 拒绝保留连续接受前缀，去除未接受旧 suffix 和 Draft 新 lookahead，添加 correction；[sequence](https://github.com/smart-lty/nano-PEARL/blob/7d020b6ce70d965c85d1bc08248c8af9c22e81cd/nano_pearl/pearl_engine/sequence.py) `rollback_tokens`；[block_manager](https://github.com/smart-lty/nano-PEARL/blob/7d020b6ce70d965c85d1bc08248c8af9c22e81cd/nano_pearl/pearl_engine/block_manager.py) `rollback` 截断 token list，并递减/释放越过新长度的 blocks，同块内旧 slot 后续覆盖 | 当前 core 维护独立 committed prefix、proposal、continuation 和版本依赖；物理 rebase 必须等待写入完成。全接受仍提交 bonus，只有 bridge 匹配才复用 suffix，失败走 A+B 批量修复及普通恢复。 |
| 算法：gamma / 接受长度 | [config](https://github.com/smart-lty/nano-PEARL/blob/7d020b6ce70d965c85d1bc08248c8af9c22e81cd/nano_pearl/pearl_config.py) gamma 默认 -1；runner 346–384 在 B=1/2/4/8/16/32、prefix256 上测 Draft/Target 速度，取 round 比率；第一次 generate 选 gamma 后存入 self.gamma，clear_requests 不复位它 | 显式 gamma 是固定窗口。自动选出的 gamma 也不是逐请求 urgency/长度策略；bench 的 B1 warmup 可以提前固定 self.gamma。`cur_acc_tokens` 跨多个全接受块累计，拒绝时记录含 correction 的 streak，MAT 不是 gamma，也不是当前每轮 accepted/K。SpecRhythm 本轮仍固定 K4。 |
| 执行：CUDA Graph | runner 105–111、246–301：enforce_eager 默认 False 时初始化 capture；decode（Draft 和 Target pre/post packed 行）复用 padded-batch graph，prefill、enforce_eager=True 或输入行数 >512 走普通 forward；capture sizes 受 max_num_seqs/512 限制，logits 计算在 replay 外 | 当前四点 Draft provenance `enforce_eager=True`；Target `runtime.target_final_memory[0/1]`、`actual-capacity.target_worker_ranks[0/1]` 均 True，config 也为 True，构造处 `s2_runtime.make_engine` 明确传入。实际均未启用 CUDA Graph。新记录补入实际 Target graph mode；本轮不打开 Graph。 |
| 执行：调度、KV、IPC | [scheduler](https://github.com/smart-lty/nano-PEARL/blob/7d020b6ce70d965c85d1bc08248c8af9c22e81cd/nano_pearl/pearl_engine/scheduler.py) 支持 waiting 优先 prefill、容量不足 preempt/deallocate/requeue；block_manager 支持前缀共享；engine 用共享内存/pickle/Event 启动 worker 内生成循环，迭代通信主要在 NCCL group 内 | SpecRhythm 的 resident360 私有 KV、禁止测量内驱逐/重 prefill、Target TP2/vLLM scheduler/独立 Draft owner、版本审计与逐轮 RPC 是不同执行约束。参考 blocksize256/max_num_seqs512，当前 Draft blocksize16、max_num_seqs128；不能假定相同。 |
| 模型 / 硬件 / 负载 | 参考 bench/eval 允许命令行模型/TP/B，eval 只执行完整批次并略去不足 B 的尾批；仓库代码没有提供本次同 workload、Qwen3、A800 的配对实测 | 当前 Qwen3-0.6B / Qwen3-32B、Draft TP1 + Target TP2、三张 A800-SXM4-80GB、B16/resident360。参考论文或其他硬件的收益不能由代码推算；模型接受率、Graph 和通信差异对收益的量值仍是待测假设。 |

## 四点原始结果和有效性

| 点 | tok/s | steps | tokens | 窗口 ms | 完整 step mean ms |
|---|---:|---:|---:|---:|---:|
| Serial/full | 61.145265 | 39 | 1869 | 30566.553104 | 781.939822 |
| Serial-eager/full | 45.211187 | 29 | 1363 | 30147.405875 | 1037.257559 |
| Serial/runtime | 69.200389 | 44 | 2121 | 30650.116623 | 693.012（原始精度见报告） |
| Serial-eager/runtime | 66.909556 | 42 | 2017 | 30145.170782 | 713.594343 |

四点 execution/measurement/cleanup 全为 PASS。最后一点的失败是 **diagnostic evidence completeness**：Draft causal-host.v1 仅保留 20000 行，dropped_rows=7897，35/42 轮有完整 owner dequeue 关联。Target host trace、Draft 原生 GPU forward 和独立 rolling counter events 仍在包内；不能把残缺的 host trace 当成零，也不能因此抹掉有效负性能结果。eager/runtime 每轮原生 overlap 均为正，窗口上下界 **1828.333607–1839.080881 ms**。CPU/GPU/进程间累计时间不相加。

四包文件名前缀 `serial-audit-B16-987ef3ec7df6-`，共同运行标签 `20260913T051454Z-1700`，按顺序 SHA256：

- serial-full complete-evidence：`4989f9fec7e6ed73f90062022d0bc5d18d3103b82deb4fa143cc821a1493290e`
- serial-eager-full complete-evidence：`042fd0d5a38b98ba9f682f6094d1aac99de899a3e0f9eba2323875b970bd6df6`
- serial-runtime complete-evidence：`3372521c2d2e443e80af54d55769e5a4bfae29a68c76b39e8f7faf4f964b34ec`
- serial-eager-runtime failure-evidence：`424a996cbd06f36022e48b05872d2dd195d084e69cf193b8db0a9373a99a1b8f`

## 为什么已有重叠仍未获益

每 step 提交量为 Serial/runtime **48.2045**、eager/runtime **48.0238**；主要差异是完整 step 约增加 20.58 ms，而不是输出骤减。观察到的一轮顺序为：coordinator resident admission 审计落盘 → Target enqueue → Target GPU → Target 诊断/采样/verify-end fence → 反馈 RPC → owner 父依赖结算、批量 KV 修复 → 恢复 proposal → RPC 返回 → Target/协调进程的收尾、下轮 scheduler → 下次 Target GPU。

**浪费来源已逐请求关联。** 670 admissions/start/completion 中，449 父拒绝、36 bridge mismatch、185 promotion，没有已 admission 的终止/其他残差。丢弃 2425 = 449×5 + 36×5。3349 early tokens = 2425 discarded + 185 reused bridge + 739 reused candidates；其中一个提升候选因 EOS 少一个 token。窗口父反馈总数672不能硬对齐670 admission：同一请求 round32 的候选仅 EOS 一个 token，被拒绝后正常恢复；round42 三 token 候选含 EOS，全接受终止；两者不 admission。故父拒绝450而 admitted rejection449，全接受222而 bridge outcomes221。

**复用没有消除整批恢复 forward。** eager/runtime 42 个测量 step 各有5次 eager、1次 commit、3次 recovery proposal forward。恢复 B 仍为7–15；185 promotion 只缩小恢复 batch，没有消除这3次串行 autoregressive forward。原生记录额外3次 B1 proposal 带 `causal_context.request_kinds=normal`、round0，是窗口内补位、在完整 step 外，不是批处理退化。完整窗口 Draft 原生统计如下（事件累计，仅供比较用途，不能当作关键路径相加）：

| 模式 | 用途 | forward 次数 | GPU event 累计 ms | B |
|---|---|---:|---:|---|
| Serial/runtime | commit |44|867.109375|16|
| Serial/runtime | proposal |135|2700.453125|完整轮132次，另补位B1三次；末端EOS使部分B15|
| eager/runtime | eager |210|5022.328125|B16×199、B15×11|
| eager/runtime | commit |42|828.406250|7–15|
| eager/runtime | proposal |129|2522.796875|完整轮126次B7–15，补位B1×3|

**父反馈不是因为本轮 coordinator fsync 而迟发。** 两个 runtime 点的 coordinator step 内分别有15837和15119次 fsync，全部在各自 Target native forward 之前；调用栈唯一对应 `ResidentSetupScheduler.schedule → _resident_events.append → CheckpointJsonl.append → os.fsync`。eager 42轮平均115.863 ms，Serial 44轮109.882 ms；窗口互斥总量4866.236 / 4834.815 ms。旧 trace 没有 filename，因此该文件归因是“producer 源码 + 单进程 receipt 总数 + step 内逐请求写盘数和位置”联合证据，非旧记录内显式 filename。新控制版本会直接记录文件名来复核。

Target native GPU end（采用两个 rank 的 end_upper）到旧 verify-end，eager 平均130.776 ms、Serial135.075 ms；旧 verify-end 到 state_sync_start，分别4.939 / 3.748 ms。eager 的 `target_feedback_available` 到 transport span 起点平均3.348 ms；transport 到首个 owner parent_result 平均1.937 ms。这不支持“父结果早已可发但被 coordinator 的4.8秒日志拖住”的说法。

但旧 verify-end marker 是同步/barrier之后，不能代表 acceptance 最早确定时间。`vllm_diagnostics.capture_target_forward` 在 vLLM patch 的采样之前，读取 workload、将完整 logits 转 float CPU、抽取数值/位置并生成实验记录；必要的目标采样尚在其后。因此可以确认有观测工作先于结果确定，**不能把上述130.776 ms全部归给记录器**。新版本新增该函数 inclusive span、sampled-results 入 hook、payload-ready 三个边界，保留 JSON/同步子 span，测量后再分解，不删除验证逻辑。

**promotion 就绪等待是当前批级接口限制。** `EagerSerialMachine.finish_synchronizations` 在必要 fence 后逐请求发布 promotion；`batch_propose` 收集普通/recovery请求，再一次返回整批。185个 promoted 请求从 parent_settled 到 RPC 返回平均106.753 ms（94.667–115.466）；它们不依赖同批其他请求的 token 内容，但 Target 当前按整批proposal推进。本轮保留此调度约束。eager 父结算/恢复 RPC inclusive 平均152.238 ms，Serial135.302 ms；eager 另有 enqueue RPC1.290 ms。这些是分别观测到的串行调用差异，不能与其内部 GPU/审计时间再次相加。

两个 Target rank 的 GPU mean，Serial为80.392/80.460 ms，eager为81.167/81.197 ms。eager额外计算并非全部暴露在关键路径：有部分被Target GPU覆盖，还有部分发生在Target CPU处理期间。因而不能用“5022-1839 ms”推导必然减速。完整线程互斥划分仍保留 unaccounted；Draft旧trace尾部缺失尤其不能被统称为Python/GIL/空闲。约3.31%的单窗口差距也不等于稳定效应。

## 写盘功能分类与本轮执行改动选择

| 文件/操作 | 实际消费者与依赖 | 处理 |
|---|---|---|
| `admission-events.jsonl` | `ResidentSetupScheduler.schedule` 先计算 `_resident_decisions`、实际调用 stock schedule，再为每个非finished resident写证据。准入消费者是内存 `_resident_decisions`、fresh `s2_control`、initial proposal lifecycle。`resident_runner` / `serial_runner` 在运行结束后 read 并验证此 JSONL；无跨进程在线恢复读取此文件 | 可按原 bounded-live 256条/1MiB上限分批写原字节。保留所有记录、checksum、即时 capture、read-your-writes、final receipt、失败传播。旧 allowlist 的保守排除导致逐条 fsync；只修这个文件，不删准入检查。 |
| `initial-proposal-events.jsonl` / decode-ready timing | lifecycle/首次验证与性能边界的原始证据，位于 Target 进程；不是coordinator窗口4.8s的来源 | 本轮不改同步策略。 |
| target diagnostics / round / transport / draft-work | 已有 explicit post-run allowlist；buffered-live 容量满或 read-before-read 时仍同步 flush，最终在bounded drain内 flush | 保持。新 fsync filename 上下文同时覆盖原始 append 和多文件批次，不把触发flush的文件误当作全部被刷文件。 |
| `s2-control.json`、stop request、ready/错误/cleanup receipt | 执行、停止、故障排查依赖其即时可见性。`s2_pool.publish` 原本就是 atomic replace，无逐步fsync | 保持即时发布，不进入 admission buffer。 |
| 输出与 KV ownership | coordinator `clock.commit`、owner/core状态、allocator检查、fence；没有依赖 admission JSONL 重放来恢复 GPU KV | 保持原协议和release约束；不新增崩溃后继续执行的承诺。 |

先提交纯证据改动，再独立提交 admission batching。它是共享路径优化，Serial与eager必须同口径重测。是否降低step耗时、是否使eager获得吞吐收益均 **PENDING**。剩余候选是 Target 采样前完整诊断、批级恢复等待和Graph/协议差异；本轮不混入这些执行改动。

## 后续固定长度协议 variant（仅设计，不接入）

若要评估 PEARL 的提交协议，应另建显式 `serial-prepost-k4`，而不是静默改变 serial-eager。Target 接口需携带 `phase(pre/post)`、验证起点/版本、准确 verified count、correction、terminal、以及单独的 lookahead suffix；输出只计验证后的 authoritative prefix。Draft 需区分正在验证的 block 和新生成的 block、每块独立依赖/KV frontier，允许按 reject index 回退，并在 fence 后释放越界blocks。成功从pre进入post；post全接受保留下一块且不额外套用当前bonus桥接。必须先证明分布/EOS/长度/accounting正确，再同K4、同测量口径对照Serial、当前serial-eager和新variant。接口、测试及GPU接入均为后续算法工作。

## eager/runtime 逐轮批处理证据

依据原始native forwards按host launch落入完整step；计数事件按end_ns归属。这里不含step外补位的3次B1 forward。commit/proposal耗时列是该用途设备event累计，不能加到RPC或整轮wall上。

|step（0起）|提交tokens|promotion|recovery B（3次）|commit GPU ms|recovery GPU ms|完整step ms|
|---:|---:|---:|---|---:|---:|---:|
|0|45|6|10,10,10|19.672|58.320|697.890|
|1|37|1|15,15,15|19.719|59.000|712.502|
|2|51|5|11,11,11|19.562|58.664|701.682|
|3|36|2|14,14,14|19.547|58.758|715.788|
|4|43|3|13,13,13|19.781|58.930|716.259|
|5|64|7|9,9,9|19.602|58.742|698.537|
|6|38|3|13,13,13|19.695|58.234|703.548|
|7|41|2|14,14,14|19.500|58.266|710.281|
|8|41|2|14,14,14|19.828|59.484|720.944|
|9|47|5|11,11,11|19.750|59.609|707.338|
|10|60|8|8,8,8|19.711|58.523|692.155|
|11|40|3|13,13,13|19.719|58.797|710.178|
|12|50|5|11,11,11|19.734|59.070|706.268|
|13|41|1|15,15,15|19.945|58.688|712.662|
|14|49|4|12,12,12|19.711|58.594|709.037|
|15|51|4|12,12,12|19.656|60.078|711.487|
|16|47|3|13,13,13|19.859|58.977|722.125|
|17|44|3|13,13,13|19.859|58.875|709.182|
|18|48|3|13,13,13|20.062|59.117|712.051|
|19|49|5|11,11,11|19.984|58.953|704.279|
|20|46|3|13,13,13|19.648|58.422|721.326|
|21|41|3|13,13,13|19.766|58.859|713.846|
|22|62|9|7,7,7|19.742|58.938|709.278|
|23|50|5|11,11,11|19.664|58.742|717.495|
|24|41|4|12,12,12|19.648|58.703|715.491|
|25|41|3|13,13,13|19.633|58.727|731.964|
|26|59|6|10,10,10|19.742|58.719|708.098|
|27|48|3|13,13,13|19.703|58.547|721.471|
|28|53|4|12,12,12|19.805|58.930|726.446|
|29|43|5|10,10,10|19.758|58.469|713.499|
|30|48|6|10,10,10|19.883|58.758|736.621|
|31|52|7|9,9,9|19.617|58.469|704.615|
|32|51|6|10,10,10|19.711|58.688|734.681|
|33|50|6|10,10,10|19.672|58.602|716.659|
|34|58|7|9,9,9|19.945|58.578|720.144|
|35|52|6|10,10,10|19.789|58.508|711.894|
|36|50|5|11,11,11|19.820|58.688|709.546|
|37|50|5|11,11,11|19.656|58.852|710.153|
|38|52|5|11,11,11|19.750|58.578|719.411|
|39|55|5|11,11,11|19.445|58.000|721.628|
|40|48|3|12,12,12|19.453|58.156|720.823|
|41|45|4|12,12,12|19.656|58.062|711.678|

## 一条真实 rolling 轨迹及源码索引

请求 `sr-55fde55fc6ba133880f6e871a56b81bb626a54bb2769179708354f2da677e8f9` 在round8/9均promotion（各提交5 tokens），round10父拒绝（提交2，实际normal_draft恢复），round11/12再次连续promotion（各5）。由Target rounds的request_id/round_id、该step唯一parent_result/parent_settled以及normal_draft关联；这些轮次位于未截断的causal区间。它说明静态资格拒绝后没有永久失效。

| 结论 | 当前仓库源码入口 | 原运行证据入口 |
|---|---|---|
| 准入记录非执行队列/恢复日志 | [resident_scheduler.py](../src/specrhythm/phase4/resident_scheduler.py) `schedule`、`_request_admissible_for_schedule`；[resident_setup.py](../src/specrhythm/phase4/resident_setup.py) `resident_admission_decision` | coordinator host.intervals 的 log_fsync/checkpoint_log_write；同进程 fixed-logging receipt |
| feedback与下一批proposal是同步RPC | [vllm_remote.py](../src/specrhythm/phase4/vllm_remote.py) `on_target_verify_end`、`_rank_zero_propose`；[eager_owner.py](../src/specrhythm/serving/eager_owner.py) `_dispatch` / 工作循环 | Target rounds.timeline；Target causal transport_exchange；Draft protocol events |
| 混合批保留3次恢复、必要fence | [eager_machine.py](../src/specrhythm/serving/eager_machine.py) `finish_synchronizations`、`batch_propose`；[gpu_backend.py](../src/specrhythm/continuation/gpu_backend.py) `rebase_gpu_parents` | fixed_device.forwards purpose/B/causal_context；rolling_eager.events |
| pre-sampling诊断候选成本 | [vLLM patch](../integrations/vllm/patches/0001-custom-proposer-request-and-verify-hooks.patch)；[vllm_diagnostics.py](../src/specrhythm/phase4/vllm_diagnostics.py) `capture_target_forward` | Target native end到verify_end端点；旧记录缺少完整capture函数span，不能全额归因 |
| 默认Graph关闭 | [s2_runtime.py](../src/specrhythm/serving/s2_runtime.py) `make_engine`；Draft worker provenance | config.enforce_eager、actual-capacity.target_worker_ranks、runtime.target_final_memory、draft-backend-report.provenance |
| 单次对照保留独立证据层 | [execution_evidence.py](../src/specrhythm/serving/execution_evidence.py)；[pair脚本](../scripts/run_execution_pair_b16.sh) | 新测试PENDING；不改上述987ef3历史状态 |
