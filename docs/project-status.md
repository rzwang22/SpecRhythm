# SpecRhythm project status


### PR #5 — B128 full/lean Target diagnostics (2026-09-17)

- Continuing from `35aeb26862923fad919a502ac454feef3b91eed4`; behavior reference
  `f7bfb43152f5655edbad9888c53c0a81e9d11954`. Draft remains open, no merge.
- Execution `4ff1170397db0393caa2d8a43c2089a8e602ddfd`; fixed entry
  `7f5d03002d2e24788ec9bedb792d974d8b55ec3f` pins that exact execution commit.
- Final local full suite:3039 passed /5 subprocess-timeout failures /3 skipped;
  remaining timeout causes unresolved, no budgets/assertions relaxed. Final related
  production/entry sets:40 passed each on Python3.9 and3.12. Ruff, compileall,
  Python3.9 grammar, Bash19 and diff checks PASS. Full details in validation report.
- CI observation: execution and pinned-entry Python3.11 serving and pinned-source
  jobs PASS; Python3.9/3.12 full-suite jobs still running on both triggers.
  No overall CI PASS claim.
- Same-version `baseline` and `lean-target`: lean omits optional full-logits CPU /
  log-softmax / top-k / duplicate argmax, initializes immutable workload index once,
  retains live identity/input/mapping/native/accounting evidence and actual sampling.
- No additional dispatch variant: queued claims already run at the next safe owner
  boundary without waiting for unrelated recovery completion. Added sparse admission
  landmarks and partial READY wait partitions; unknown time stays unaccounted.
- Same k3-b128-v1/unified/deferred-window/performance-exploration; full output
  equivalence NOT_RUN. No new GPU execution or performance claim; operator results
  PENDING. [Design](k3-design.md), [validation](validation/k3-target-profiles.md),
  [fixed comparison runbook](k3-target-runbook.md).

Last updated: 2026-09-17

Maintenance rule: every code-changing PR updates this file with its scope, status, evidence,
known limitations, and next gate before that PR is considered complete.

## Goal

Build a reproducible path for evaluating SpecRhythm's SLO-aware speculative-decoding policy,
first as a dependency-free semantic simulator and workload harness, then with measured GPU
calibration, and only afterward as a vLLM or SGLang integration. Simulator outputs are used to
reject inconsistent policies and design controlled experiments; they are not GPU performance
claims.

## Roadmap

1. **Workload and provenance:** deterministic Mooncake replay, R3 proxy construction,
   validation, manifests, and raw-data hygiene.
2. **Simulator semantics:** persistent proposal lifecycle, deterministic candidate trees,
   tree-aware AdaServe/SpecRhythm allocators, queueing-aware SLO accounting, path-dependent eager
   continuation, diagnostics, and constructive counterexamples.
3. **GPU calibration:** measure context/batch/candidate-dependent `D(B,K,C)` and `V(B,K,C)`,
   acceptance traces, confidence calibration, and candidate roof on a pinned model/engine/GPU.
4. **Engine prototype:** implement the validated control plane as a narrow plugin/prototype in
   the selected serving engine.
5. **Evaluation:** paired workload sweeps, ablations, class-level SLO attainment, goodput, waste,
   confidence intervals, and failure analysis.

## Pull request progress

### PR #5 — explicit A100 K3 B128 scale exploration (2026-09-17)

- Execution `f7bfb43152f5655edbad9888c53c0a81e9d11954`; fixed entry
  `23e748c1a1c093984a9b0abdab92d643f71b2624`.
  Continued from clean local/remote `e2dd4357099880c736bc30ee00a12b1886be5688`.
  Added `k3-b128-v1`: Serial A128/Target128, PingPong A64/B64/Target64,
  Draft ceiling128, resident360. B16/B64 entries/defaults remain available.
- Geometry flows through real planning, engine-capacity checks, owner/controller,
  scheduler, native TP qualification and export. True K3/reservations3→4 or eager6→6,
  models/numerics, diagnostics, unified dispatch, serial gate and READY unchanged.
- B128 default is unified/deferred-window/performance-exploration; four initial
  capacity probes then forward/reverse eight fresh performance points. Warmup
  requires256 actual opportunities and active128 identity coverage; each window30s,
  setup900s/drain60s unchanged. Full output equivalence NOT_RUN, historical mismatch open.
- Existing bounded buffers/exports retained; B128 local free-space floor32GiB.
  Added offline per-thread capture statistics, actual128 selection metadata and
  physical-calls/step reporting without changing capture or GPU work.
- Local full suite: 2989 passed / 6 failed / 3 skipped; failures are unchanged
  child-entry tests exceeding20s. Python3.9 affected206 passed; Python3.12
  affected204 passed / 2 new entry tests hit20s. Cause remains unconfirmed; no
  assertion/timeout changes. Final contract additions12 passed. CI is separate;
  see [B128 validation](validation/k3-b128.md).
  Entry CI had a legacy refill progress assertion failure; a later documentation
  HEAD had a legacy5s owner-settlement timeout. Both logs investigated and retained
  separately from local timeouts; passing companion jobs do not resolve their causes.
  Operator GPU capacity, geometry, cleanup, overlap and performance PENDING.
  [B128 runbook](k3-b128-runbook.md); no server connection or GPU execution here.

### PR #5 — B64 diagnostic persistence and unified dispatch (2026-09-16)

- Continued from `5511f89ec142abf37448ffb2031641dc2964b966`, preserving prior
  performance-exploration and B16 defaults. I/O stage:
  `4006ea7d1aed494dc1c663153bd23a7065bb22b6`; final execution:
  `cc42a501b63623de3e046e6388d0dcab0d2f339c`; fixed entry:
  `2cd2ca4906ce61e5038296a4bce88e29a3802a9d`.
- Explicit `io-only` and `unified` use the same final execution, bounded
  `deferred-window` diagnostics, geometry, workload and budgets. Post-run JSONL
  and repeated plugin-report snapshots are finalized under the original drain
  deadline. Admission/KV/protocol checks and live control/ownership remain online.
  Overflow, partial publication and missing bytes continue to fail qualification.
- Existing mixed recovery/ordinary materialization is retained. Runnable inventories
  and unique physical IDs distinguish recovery-only calls from mixed calls. Unified
  policy validates runnable selection and publishes complete legal promotions at
  their own fenced boundary before unrelated materialization, returning to the
  owner command loop; Serial still has its all-Draft idle gate. No claim of a
  historical missed-batch count or GPU forward reduction is made without evidence.
- Each configuration runs four modes in forward then reverse order, with independent
  roots, all raw values/ranges and one bounded deduplicated archive. Local retention,
  verified DPC delivery/fallback, first-error status and sole UPLOAD ONLY are retained.
  Full output equivalence remains NOT_RUN; historical Serial differences remain open.
- Local full-suite result: **2 failed / 2931 passed / 3 skipped**. The two failures
  are an existing 5 s subprocess timeout and an existing fork logger's 2 s deadline
  expiration. Baseline isolated checks passed but do not identify the timeout cause.
  No assertion or budget was loosened. Final Python 3.9/3.12 affected suites each
  passed 126 cases; fixed-entry suite passed 5. See the
  [validation record](validation/k3-deferred-dispatch.md) for first failures and scope.
- HTTPS push initially failed because the configured credential was rejected.
  The existing SSH identity authenticated successfully; delivery uses ordinary SSH
  push without changing the remote configuration or rewriting history. CI status is
  recorded against the uploaded commit separately. GPU capacity/cleanup/native
  overlap/performance remain PENDING.
  [Independent server commands](k3-b64-runbook.md#independent-io-and-dispatch-comparison).

### PR #5 — Target-only 原始 drain deadline 传递（2026-09-15）

- 执行SHA `5b50529f3bd3617f29c60ee9de6bf7143f96449e`；随后固定入口仅绑定此已验证实现并更新交付文档。固定入口/真实runner/本地单包交付24项回归通过，提交中的必需执行文件已逐一核对。

- 从干净且与远端一致的 `1975061693a72f7ac1880da6b7cff82efc39fe9d` 继续，保留 `ae5be9a6b318b31931808fd1523064b33e2f0975` 的本地运行目录、监督器和单包交付修复。
- 最新 A100 总包SHA256与146逻辑文件逐一核验。四容量及其cleanup通过，Target-only在drain 0.382452022秒处失败，仍剩59.617547978秒；监督器状态读取错误为0。旧FAILED/INVALID及未执行阶段保持，不归因于A100或DPC。
- 真实生产链回归先复现 shutdown `{}` 导致 `None` 被发布器误报超时，再补齐同一绝对deadline。Target-only和四K3接收端严格区分缺失/非法/冲突与实际到期；先校验后执行shutdown，报告发布沿用原值。新路径无 `now+60s` 兜底，失败走既有owner/监督器清理并锁定首错。
- RPC、首错汇总、总包保留模式、阶段、运行目录、收到的值/类型和单调时间；仅合法deadline计算余量。报告构建/写入/fsync/发布/校验错误分阶段保留，部分字节与完成回执校验继续严格。
- 未改K3协议、READY、GPU算法、geometry、模型、测量配置或setup900/drain60。四模式容量→Target-only加四模式联合correctness→四性能点、唯一总包交付保持。新GPU correctness/cleanup/overlap/performance全部PENDING。
- 全量pytest 2756通过/3跳过（413.96秒）；Python3.9/3.12相关各432通过；最终owner故障通知分别在3.9/3.11/3.12各96通过。Ruff、三版本compileall、221文件3.9语法、21脚本Bash、静态四模式接口和diff通过。首次红例是修复前真实链漏传deadline，未放宽任何断言/预算。新CI按提交实际状态单独报告；CPU测试不替代A100验收。


### PR #5 — A100 状态读取、本地运行目录与单包交付（2026-09-15）

- 实现SHA `ae5be9a6b318b31931808fd1523064b33e2f0975`；随后固定入口仅绑定此SHA。

- 从当前HEAD `3b8a8db96563e8cd23951448503b308ba11e3089` 继续；保留 c663cb3 的四模式geometry与原生B16验收。只改基础设施，不改K3/no-bonus/READY/调度/KV或setup900/drain60。
- 只读核验 bitahub-retry 总包18个逻辑对象。监督器运行中读取drain-state ENOENT后提前终止，后续Broken pipe和不完整报告为次生事件；三个物理rank容量通过，不代表完整cleanup通过。原历史标签保留，DPC可见性仍是假设，overlay不等于NVMe。
- 新入口默认 `/tmp/specrhythm-runs/<tag>` 保存所有本轮可变文件；完成后本地封包、哈希，再向DPC唯一临时目标复制/复核/发布。默认保留本地目录，失败包包含损坏JSON原始字节。原运行码、导出和交付错误分列，外层唯一UPLOAD ONLY。
- 状态读取一次/轮、有界重试，保留原绝对deadline并检查worker。监督器终止原因先于信号落盘；完整Draft报告先写临时文件，owner停后完整发布和回执验收。旧默认报告路径不变，本入口Target-only对照同样启用完成契约。
- 本地全量2677通过/3跳过；固定源码17通过；Python3.9/3.12相关各131通过，最后变动分别28/46通过；最终封包/入口18通过。Ruff、三版本compileall、220文件3.9 AST、21脚本Bash和diff通过。新CI按实际新SHA单独查询。新GPU correctness、native overlap/performance、A100/DPC适配验收均PENDING；不连接服务器，交付后等待唯一总包。

### PR #5 — 四模式入口 Target ceiling 修复（2026-09-15）

- 修复执行SHA `c663cb30f470ed9bf24465d07af7ea3a6991c73e`；固定入口提交仅绑定该SHA并更新文档。从 `6a8072dd2ba4e478447666234fc4864ea8d48477` 继续，保留 fca2118 执行优化与四模式配置。确认 `PY_MEASUREMENT` 将所有模式固定为 ceiling8，错误拒绝两个 Serial B16；此前 shell 测试替换整段 Python，未执行该验收。
- 入口统一使用 geometry：Serial/Serial-eager16、PingPong/eager8。严格核对模式、geometry类型、实际batch摘要和原生TP请求集合。联合 correctness 对每个模式要求至少一个满批原生forward；Serial内部仍B8、重复请求、顺序拆forward、缺rank均不能通过。比较和单包保留满批证据；尾部小batch及原有原因保持。
- 未改变GPU算法、K3、no-bonus、调度/READY、采样、KV检查、日志或窗口。旧三模式/旧prepost3默认规则不变。新GPU correctness/overlap/performance PENDING。
- 本轮全量2645通过/3跳过（366.87秒）；Python3.9与3.12相关各157通过；Ruff、三版本compileall、217源文件3.9语法、21脚本Bash、diff检查通过。无放宽断言或超时。基线入口HEAD的GitHub八项检查已实际SUCCESS；新提交CI在交付时单独报告，不混同基线。


### PR #5 — K3 四模式与 admission 编码复用（2026-09-15）

- 保留 `d007dce448bd2a7d510172222ae166d7bb6f299e` 和入口 `3a2ffde4cd4350392b7c6f9e98ebe2cd1d1d1315`。独立矩阵提交 `ed6ff9703765e2c36b9ec4b3d0cb91edc1125d8f` 实现 Serial/Serial-eager 单组16、PingPong/eager A8+B8；实际 Target ceiling16/8，兼容 Draft 物理合批上限16。
- 四模式保持真实K3、no-bonus、需求3/预留4和 eager Draft6/6、runtime审计、buffered-live、resident360与原窗口预算。预热统一32次实际请求验证机会，记录逐请求分布，B16两步/B8四步。旧 Serial B8 数据不重命名。
- CPU执行SHA `fca2118b89d21b11f18f5de43d69657e820087ab`；后续固定入口提交只绑定此执行版本并更新交付文档。独立CPU改动只合并当次 admission 的共享标量字段，并复用一次 canonical 编码完成原校验和行。360逻辑记录仍完整输出；不改变双向身份绑定、当次全prompt/后缀检查、两次全KV审计、fence、owner边界或READY发布顺序。
- 最新旧包145文件/106对象校验通过，普通77/124跨组重叠、恢复覆盖3.76–3.88%；eager1/134、0.144–0.147%，自身父轮重叠7.75–7.79秒另计。不能声称恢复已充分隐藏。原运行与负/单窗口性能结论保留。
- 四点入口：静态契约→四容量→Target-only+四完整输出→四runtime30s，只回传一个总包。源SHA、模型/数值/资源与公共配置一致，允许模式规定的Target16/8差异，不比较跨次UUID。
- 最终本地全量2612通过/3跳过；Python3.9和3.12相关各351通过，Ruff/compileall/216源文件3.9语法/21脚本Bash/diff检查通过。首次失败记录见 `docs/k3-validation.md`；3.12监督器测试首次启动超时原因尚未确定，隔离复查通过不等于已确定根因。最终CI以GitHub实际状态为准。
- 新GPU capacity/correctness/cleanup/native overlap/performance均PENDING。普通推送Draft PR5，不合并；等用户返回总包后再决定下一项优化。

### PR #5 — K3 resident 调度规范化（2026-09-14）

- 执行SHA `d007dce448bd2a7d510172222ae166d7bb6f299e`；后续固定入口提交只绑定此SHA和更新交付文档，执行源码不变。

- 从干净PR HEAD `8b8a2d5fb9ca16f35756429feb19214d544520d6` 继续，保留基线执行 `8a3aca007fb60f9f67d1dcb7bedb5f8b78b8602e` 和后续入口。Draft、不合并、只普通推送，不改其他PR/历史结果。
- 复用082831Z-1891总包的145逻辑文件核验与原生关联；三点52.54/75.72/79.70 tok/s和原PASS保留。独立[基线派生](k3-resident-8a3aca-observations.json)标明来源和缺少旧resident子阶段，不把34.6ms全部归因于token转换。
- 只消除同次调用中的重复整行int/tuple规范化。K3私有不可变输入构造器遍历当前完整token行一次，bound/full matching复用本次数据；完整当前prompt比较、生成后缀转换、双向绑定/历史/别名/歧义拒绝均保留。旧普通sequence接口保持原路径。三个K3模式一致启用，不缓存前缀、KV、版本或decision。
- 两次完整resident快照/block审计、live KV、全部admission记录和buffer策略保留。六个有界子span分开binding/readiness/decisions/真实stock调用/initial finish/record append，计数live行、规范化行、token访问和记录数；原stock_schedule为包装层inclusive，嵌套子耗时不重复相加。
- 新诊断报告同时给window/steps、engine-step与外围时间、重叠轮数、恢复物理区间并集分母与其他请求TP覆盖比例；缺证据仍null/INCOMPLETE。OBSERVED只表明非零，不宣称充分隐藏。全部必要原始记录在一个去重总包，沿用首错停止和唯一UPLOAD ONLY。
- `post_prepost()`、promotion READY发布时间、Serial idle gate、A/B claim/反馈/fence/调度策略均不变；READY提前发布仅为后续候选。真正K3/no-bonus、容量3→4与eager6、active16/home8+8/ceiling8、resident360、workload/模型/seed/设备/预算不变。
- 结构性回归在旧路径确认失败；带pinned源码全量2556 passed/3既有skip，3.9/3.12相关各341项通过，最终factory微调后各38项通过；最终源全量复核2556 passed/3既有skip，固定入口15项通过，Ruff/3版本compileall/369文件3.9语法/21个Bash/diff通过。执行提交CI的source与3.11契约已SUCCESS，3.9/3.12全量job在此文档快照仍运行；交付时单独报告最新CI。既有8b8a2d5远端两次CI运行均SUCCESS，不代表本次CI。
- 新GPU correctness、cross-request恢复覆盖、native overlap、performance **PENDING**。不连接AutoDL。完成固定三模式入口后停止，等待唯一服务器总包。[设计](k3-design.md)、[验证](k3-validation.md)、[runbook](k3-runbook.md)。

### PR #5 — K3 Target dispatch 重复 prompt digest（2026-09-14）

- 执行SHA `8a3aca007fb60f9f67d1dcb7bedb5f8b78b8602e`；随后固定入口提交只绑定此SHA并更新交付文档，不变更src执行代码。

- 从干净 `a5cdcf9f6e81cc9975321e37fc6100a143fa607d` 继续，保留778d容量、异常清理及owner信息快照；Draft不合并，不改其他PR或历史结果。
- 只读核验062632Z-1634总包145逻辑文件大小/SHA256。三容量、联合输出、execution/measurement/cleanup通过；46.55/67.64/71.39单窗口结果保留。跨请求普通/恢复原生重叠均0；eager父轮重叠6860.926–6904.569ms；不宣称稳定收益。
- 普通PingPong的111轮claim→Target平均99.055ms，其中scheduler84.087ms、内部prefix/block-record37.945ms和独立完整block audit7.911ms。剩余scheduler38.231ms、scheduler外14.969ms保留未细分；不与RPC/跨进程/GPU时间相加。首段B已claim且A尚未恢复完成，之后Target CPU双快照/hash消耗重叠机会；没有证据表明verify-start等待A整份proposal。状态查询已快，enqueue只等mailbox ACK，单token必要fence保留。
- 三个显式K3模式共用不可变prompt digest证明：每次仍比对完整当前prompt/整数类型/绑定，读取live frontier和block并执行前后两次完整ResidentPoolAudit；移除的是重复规范化/JSON/hash，不是KV审计或当前请求检查。变更/结束/移除时拒绝或退役证明，不缓存control/claim/KV。旧模式路径、K3候选与容量、Serial有意idle gate、A/B策略、反馈和日志缓冲均保持。
- 实际controller→owner→scheduler→verify adapter四个受控顺序证明B进入执行时A仍有未完成扩展，A可在B验证中READY；恢复旧快照实现四项结构断言失败。不是CPU模拟GPU overlap。原测试止于owner/claim，漏过Target后续双全池hash。
- 新有界三段scheduler span与prompt-proof计数；原生forward先绑定request/proposal/version/TP rank再分home，独立报告跨请求普通/恢复、父轮eager和未覆盖Draft区间。记录物理完成与实际READY发布两种时刻；GPU空闲不等于CPU dispatch空闲。缺关联仍INCOMPLETE/null。拒绝周期最多128行；比较只嵌入汇总，单包保留原始证据。
- 本地全量pytest **2535 passed / 3既有skip**；最后证据绑定补充后定向27项通过，Python3.9/3.12相关各269项及最终27项通过。Ruff、3.9/3.11/3.12 compileall、364文件3.9语法、21个Bash和diff通过。参考a5cdcf9远端8个检查SUCCESS，新提交CI在普通推送后查询。新GPU correctness、cross-request pipeline、native overlap、performance **PENDING**；[设计](k3-design.md)、[验证](k3-validation.md)、[只读原始时间线派生](k3-pipeline-778d-observations.json)、[固定入口](k3-runbook.md)。不连接AutoDL；完成后等待用户唯一总包。

### PR #5 — K3 容量预留与初始化异常清理（2026-09-14）

- 本轮实现 SHA `778d87b5b6fd312ae376d468d356396a7b93b891`；后续固定入口只绑定此实现、更新runbook，不变更执行源码。固定后入口15项通过，已用git对象检查该SHA含执行脚本、静态容量接口、预留与清理修复；新CI状态交付时查询。

- 从干净 `4206ff53a2346f29be599257fd05ae001d73633c` 继续；此前实现和 owner 信息快照优化保留。参考入口的实际 CI 全部 SUCCESS；本轮 CI 单独查询。
- 只读核验 `051537Z-2351` 总包17逻辑文件大小/SHA256。真实首错是 Target 初始化后 `fixed_runtime.run → capacity_for(speculative_tokens=3)` 与旧 minimum4 冲突；不是 OOM 或输出错误。旧 capacity UNKNOWN / execution FAILED / measurement INVALID / cleanup FAILED / performance PENDING 保留，其他点未开始。
- 显式拆分候选3、需求3/6、旧最小预留4和最终预留4/6；实际 Target K=3。block计算使用最终预留，extra为0/2。KV 生命周期核查覆盖当前P3、lookahead3、末token未materialize和correction回退/catch-up；所有安全余量、workspace和resident logits不变。
- 新原始容量报告保留逐请求预算和三rank数据，正式 summarize/device qualify 及总包重读均重新计算；旧模式容量函数、字段默认和执行路径不变。加载模型前静态入口只能证明接口一致，GPU容量仍待测。
- K3进入模型初始化后、drive前失败时，按既有drain预算请求真实空Draft owner关闭，并关闭Target handle；记录原始错误、各释放结果和次生写盘/清理错误。CPU真实owner/socket回归覆盖。未返回handle的部分初始化、vLLM内部未传递timeout及进程后代最终退出仍依赖监督器，不能将API返回当cleanup PASS。
- 此前CPU报告测试从drive开始，漏过run中的真实容量推导。新回归经过run→实际capacity_for→报告→summarize/qualify→总包重读；三模式×容量/非scan correctness/scan性能均覆盖。首轮新回归真实复现旧断言后修复，没有放宽断言或预算。
- Python3.9与3.12相关回归各266通过；3.12首轮两项子进程缺包失败已由stderr定位，补齐服务器已有PYTHONPATH后通过，原失败记录保留。全量pytest **2512 passed / 3既有skip**（343.09s）；首轮因Shell PATH缺python导致旧测试2失败，stderr明确rc127，纠正环境后全量通过，未改断言/预算。最终入口回归两版本各15通过；Ruff、compileall、360文件3.9语法、21个Bash及diff通过。[设计](k3-design.md)、[验证说明](k3-validation.md)、[固定runbook](k3-runbook.md)。不连接AutoDL；新GPU容量、correctness、原生流水线重叠和performance **PENDING**。

### PR #5 — 统一候选 K3 与跨组 owner 流水线（2026-09-14）

- 从干净 `0b37d37ff364bcc6533b8805940bcbe572dafc56` 继续，执行参考 `c0ecc2a405c8b6c9cb3016c7254739bda5bec2b6`；旧模式、后续有效提交和历史结果保留，Draft不合并。
- 新显式 `serial-k3` / `pingpong-k3` / `pingpong-eager-k3`：实际 K=min(3,remaining)，候选EOS可短；seed计入K3，初始化缓存seed+两次扩展；拒绝后correction materialization给seed，再两次扩展；成功复用完整lookahead3不产生第四候选。独立READY、版本/KV/fence/ownership、反馈优先和无bonus提交协议保持。
- 三模式同active16/home8/8/Target ceiling8/单Draft+TP2。Serial有显式Draft idle gate；两个PingPong共用稳定home策略，差别为是否生成未决依赖continuation。新owner以每次状态变化后发布的只读snapshot处理信息性status，claim/control/feedback/release仍由真实owner执行，保留单token写入边界。
- CPU可控顺序复现旧status等待GPU替身写入，并证明新status不等写入、B先claim而A尚未完成恢复、B反馈前A能READY。只读校验本地保留c0ecc2a总包102逻辑对象：普通点77个相邻轮次均有另一组先READY，却等到前组四次Draft完成才开始Target；154次status排队平均41.44ms。原始顺序支持调度串行化；剩余Target前处理不强行归因，也不据此预估加速。
- 真实factory/scheduler/adapter/owner/backend、固定vLLM bookkeeping+独立oracle完整输出、full/runtime allocator，以及报告生产→序列化→qualify→总包重读回归覆盖。仍严格拒绝缺probe、错误阶段/设备/native关联。比较JSON、拒绝恢复时序和native交并集纳入唯一总包。
- 全量pytest2414 passed / 3既有skip；Python3.9相关148、Python3.12相关136通过。Ruff、compileall、354文件3.9语法、20个Bash及diff检查通过；候选生命周期守恒与重复结算回归通过。既有CI的5秒旧drain失败单独保留调查结论，新CI状态交付时查询，不以本地通过覆盖旧失败。
- [设计](k3-design.md)、[验证](k3-validation.md)、[服务器runbook](k3-runbook.md)：容量3点→Target-only+3模式联合correctness→固定runtime3点，首错停止、原始码与导出码分开，只上传一个新总包。未连接AutoDL；新GPU correctness、pipeline/overlap、performance **PENDING**。
- 执行SHA `b2a2d29210d68357590a567d4819f3552175d179`；后续交付只增加固定入口、文档/入口回归及CI检查。入口14项通过（3项新增在Python3.9亦通过），21个Bash、355文件3.9语法通过；普通推送后仍保持Draft，等待服务器唯一总包。

### PR #5 — 非 scan correctness 报告与联合首错修复（2026-09-13）

- 保留e30a338当前分支和429956执行基线。150541Z总包61逻辑文件大小/hash通过；两容量点和设备绑定PASS，普通PingPong完成/cleanup且进程码0。与Target-only逐请求512 token相同，但缺顶层probe导致report_qualification失败；原FAILED/INVALID与联合未通过状态不改写，后续点未启动。
- drive公共报告统一真实bool probe；run/drive、point和离线阶段契约严格一致，correctness不要求decode_scan、不跳过设备/原生关联。原CPU fixture手工补probe遗漏真实非scan构造，现用真实drive/clock/drain/report/序列化→qualifier→单包→同一契约回归覆盖。
- 联合首错从实际模式run目录和原始报告读取，外层阶段、进程码、校验码、导出码分开；脚本不继承容量POINT/ROOT。总包保留stage来源、joint失败、原始报告及native。首错后导出失败也不替换原码。
- 无调度、3+1、P1/P4、KV/fence、模型、审计/日志或预算变更；不连接AutoDL。新GPU correctness、overlap、performance **PENDING**。完成CPU验收、普通推送及[固定入口](pingpong-prepost3-runbook.md)后停止等待单个新总包。
- 全库2324 passed / 3既有skip，Python3.9相关71 passed、Python3.12相关58 passed；Ruff、双版本compileall、342文件3.9语法、19个Bash和diff通过。旧CI复现测试的帧名假设改为严格源码行定位；旧refill/drain失败单独记录，未改调度或timeout，见[验证记录](pingpong-prepost3-validation.md)。
- 执行SHA `c0ecc2a405c8b6c9cb3016c7254739bda5bec2b6`；后续交付只固定入口和文档，不改变src执行路径。

### PR #5 — PingPong prepost3 设备证据契约修复（2026-09-13）

- 从 `8c24f097db419793063bea898e828fc63a50e801` 继续，保留此前执行和交付提交、旧结果，保持Draft；不改PR #2/#3/#4。
- 首个GPU容量probe实际准备/容量/cleanup PASS、进程码0、零Target decode；消费者误读新Serial派生路径没有生成的旧Dual专属字段，导致qualification FAILED和原FAILED/INVALID。总包17个逻辑文件大小/hash通过；后续点未运行。本次不是GPU正确性或性能失败。
- 在已有快照记录真实proposer路径、rank及原有hook计数；新四个prepost3模式严格核对启动/结束/原生forward/request/proposal设备关联，Dual计数明确不适用；旧pingpong live计数规则不变，不增加GPU查询/同步/全池审计。
- 单包投影保留最终TP快照及释放来源；缺字段明确失败，首错保留进程码与report_qualification的区别。实际生产→报告→单包→同一契约离线重放回归覆盖，历史缺失字段不补写。
- 只修复证据和验收，不改变3+1、P1/P4、调度、反馈、模型、runtime/buffered-live或预算。固定两点入口仍capacity→联合完整输出correctness→performance，首错停止并只上传一个总包。
- CPU全库 **2296 passed / 3既有skip**、Python3.9相关 **49 passed**；Ruff、双版本compileall、340文件3.9语法、19个Bash和diff检查通过。首次测试链遗漏及修复保留在[验收记录](pingpong-prepost3-validation.md)。[固定服务器入口](pingpong-prepost3-runbook.md)仍只返回单包；GPU correctness、native overlap和performance **PENDING**，普通推送后停止等待新总包。
- 执行修复SHA `429956febe6d731291c1a3c0ee0857337a3c54c2`；后续交付只固定入口与文档，源码执行完全相同。

### PR #5 — 显式 PingPong prepost3 与跨 A/B rolling（2026-09-13）

- 保留 `f01e8d037007540209999179357c3dd2ff2335a7` 实现及 `71f062c2a5fcef9298300904164a5abbde962a2f` 交付；只修改 Draft PR #5，不修改 PR #2/#3/#4 或历史结果。
- 用户返回上一轮同协议 Serial / Serial-eager 约 **87.98 / 100.59 tok/s**，单次窗口约 **+14.34%**。记为已有 Serial 结果，不外推为 PingPong 收益，不覆盖此前有效负性能记录。
- 新模式 `pingpong-prepost3` / `pingpong-eager-prepost3`：真实 resident Target adapter/scheduler + 单 Draft owner；原子 ready claim/consume、独立父轮结算、兼容普通/lookahead/common 行混合单物理 forward；跨 home A→B→A，拒绝恢复 P1 后继续 eager。
- 沿用 no-bonus、K4、最多3步 lookahead、一次公共成功扩展/拒绝恢复、版本/KV/fence/释放检查；默认旧模式不变。新实验固定 resident360、总 active16、home8/8、Target ceiling8、同一 TP2 Target。预热两轮按4次真实 admission，允许相邻请求重复。
- 新联合 correctness 与 Target-only 比完整输出，要求真实混合长度、跨批接续和拒绝后恢复覆盖。脚本固定两点 runtime；不连接 AutoDL，不扩大 grid。GPU correctness、原生 overlap、performance **PENDING**。
- [设计与源码映射](pingpong-prepost3-design.md)；[服务器单包 runbook](pingpong-prepost3-runbook.md)。成功或失败只返回一个 `pingpong-prepost3-delivery-<tag>.tar.gz`，共享内容按 hash 去重；保留原始退出码、运行资格与诊断/导出错误的区别。
- CPU全库 **2263 passed / 3既有skip**，Python3.9相关 **64 passed**；Ruff、两版本compileall、337文件Python3.9语法、18个仓库Bash及diff检查通过。首次失败与修复见[验证记录](pingpong-prepost3-validation.md)。执行提交 `bc908be95d3ad611dc161a467709431bfe0c082a`；后续固定入口只更新启动器/CPU检查/文档，等待用户上传GPU总包，不自动继续调优。

### PR #5 — 显式 Serial pre/post3 协议（2026-09-13）

- 保留优化基准 `c02ee7dec30ceff21e95eea8f3a42b4909d37202` 和后续 `bd38ae1ba3f2c6acb1de30fa98196f68da8c3ec6`，不改旧模式或 PR #2/#3/#4；PR #5 继续 Draft。
- 用户返回的优化版 Serial 86.64 / Serial-eager 83.39 tok/s 作为本轮动机，保留原负性能解释，不从旧窗口扣成本。本轮不重新运行或改写这些历史结果。
- 新 opt-in `serial-prepost3` / `serial-eager-prepost3` 共用“不额外提交 full-accept bonus”的规则；旧 bridge 协议不变。真实 owner/backend 路径实现三步 lookahead + 一次混合公共 forward，失败请求生成 P1，成功请求生成 P4，静态资格在反复拒绝后保留。
- Target 在同步 bookkeeping 的发布边界统一投影输出及 token/KV frontier；复用固定版本 vLLM 的真实 ragged B16 输入与采样索引。CPU 测试执行原版源码函数及 adapter→owner→backend 链，验证连续成功/拒绝/短恢复/再次成功；不把 CPU 替身当成 GPU 证明。
- [设计与完整时序](serial-prepost3-design.md)；[服务器 runbook](prepost3-runbook.md)。新增完整输出 Target-only 联合 GPU correctness 门槛，独立记录16请求 fixture；性能仍 resident360/B16/两轮预热/30秒、runtime/buffered-live，两点新 root。严格 phased trace/归因/导出门槛保留。
- 全量 pytest 2223 passed / 3 个既有平台或 GPU skips；Python3.9 新协议41项通过，报告最终调整后相关48项通过。Ruff、两版本compileall、src+tests共317文件Python3.9 AST、16个仓库Bash语法及diff检查通过。执行 SHA `f01e8d037007540209999179357c3dd2ff2335a7` 已提交；交付新增固定SHA前台入口（17个Bash文件）与启动器首错检查，普通推送后保持Draft；GPU correctness、native overlap、performance **PENDING**。交付后等待服务器证据，不扩展 PingPong、动态资格、长度搜索或 CUDA Graph。


### PR #5 — 原子 JSON fsync 归因修复（2026-09-13）

- 已保留 68c30b1 及此前提交，分支保持 Draft；不改 PR #2/#3/#4。
- c816 控制版 Serial/runtime 45 steps、2164 tokens、30642.241426 ms、70.62146564003774 tok/s；capacity/correctness/execution/measurement/cleanup PASS，所有 causal trace 无丢行。唯一失败为 Target rank0 45次 / 123.609629ms 的 fsync 缺文件名，属于 diagnostic_evidence；后续三点未启动。
- [归因修复及证据说明](rolling-eager-fsync-attribution.md)：真实 `_write_report` imported alias→atomic JSON→fsync 链缺少上下文；共享底层线程上下文覆盖所有 phase4 sync 写入点，区分目标/临时路径及类别。保持全部同步次数、顺序、原子发布和异常传播，不改 buffer 策略及 eager 调度。
- 优化路径全量CPU回归2182 passed / 3既有skips；Python3.9相关回归53 passed，Ruff、compileall、Python3.9 AST303文件及15个已跟踪Bash脚本语法、diff检查通过。
- 控制路径全量CPU回归2173 passed / 3既有skips；Python3.9相关回归53 passed，Ruff、compileall、AST302文件及同样15个Bash脚本检查通过。
- 新控制辅助ref `codex/rolling-eager-fsync-control`：`298578b9eb7942d7faca807728b03a3784ac4235`；新优化：`c02ee7dec30ceff21e95eea8f3a42b4909d37202`。两版源文件仅保留此前admission buffer优化差异，执行脚本相同，均普通推送。
- 控制、优化版本同时修复；未知事件仍严格失败。首错明细先打印，导出次生错误与原退出码分离。新版本 GPU correctness/overlap/performance **PENDING**；[四点复验 runbook](rolling-eager-execution-runbook.md)。

### PR #5 — 987ef3 四点复核及执行控制（2026-09-13）

- 分支 `codex/rolling-eager-v0.1` 保持 Draft；不修改 PR #2/#3/#4。
- 只读确认四点 execution/measurement/cleanup PASS，eager/runtime 66.909556 vs Serial/runtime 69.200389 tok/s；42轮原生重叠仍没有吞吐收益。最后一包是Draft host trace20k截断7897行导致诊断完整性失败，原负性能结果保持有效。
- [实际PEARL源码对照及因果报告](rolling-eager-pearl-comparison.md)：参考完整SHA固定；449 admitted父拒绝、36 bridge mismatch、185 promotion；每轮恢复仍3 forward。公共admission日志每轮约360次fsync发生在Target forward前，约4.8秒窗口成本。
- 第一项独立提交只修证据：阶段预算、准确完整性层次、文件/反馈/Target诊断打点、compact报告及runtime配对脚本。执行优化单独提交；控制版本全量CPU pytest 2150 passed / 3既有平台或GPU skips，Ruff、compileall、Python3.9 AST301文件及14个Bash语法检查通过。GPU正确性/重叠/性能等待用户复测。
- 控制提交 `c8165ccb93cc07d81f684cb48f270da837d6e75b` 已普通推送。执行优化提交 `2f436a14e1284164f04aa892cca5cdfb8c048301` 接入schema/consumer受限的admission有界落盘；生产scheduler链20880条记录完整保留，original-live同步策略不变。两版本脚本各含Serial/runtime及eager/runtime。优化版本全量CPU pytest 2166 passed / 3既有skips，Python3.9相关回归123 passed，Ruff、compileall、303文件Python3.9语法及15个Bash检查通过。
- [服务器运行说明](rolling-eager-execution-runbook.md)。本轮不接入PingPong-eager、不改批级恢复调度、不维护A-only。

| PR | Status | Scope | Evidence / boundary |
| --- | --- | --- | --- |
| [#1 workload-v0.1](https://github.com/rzwang22/SpecRhythm/pull/1) | merged | strict Mooncake replay, R3 proxy config, validator, manifest, fixture tests and docs | workload plumbing only; proxy payload and illustrative acceptance |
| [#2 simulator-semantics-v0.2](https://github.com/rzwang22/SpecRhythm/pull/2) | frozen draft; Phase 2 complete, not merged | proposal lifecycle, deterministic tree oracle, tree-aware allocators, base-preserving residual controls, Phase-2 nested search pools and common-snapshot oracle replay, path-aware eager and accounting | pure-Python proxy and oracle upper bounds only; no deployable oracle, measured search cost, GPU integration, or performance claim |
| [#3 gpu-integration-v0.1](https://github.com/rzwang22/SpecRhythm/pull/3) | draft; Phase 3B.1 and corrected-20 Phase 3C.2 complete; Phase 3C.3 corrected-100 awaiting server run | hardened multi-rank primitives, corrected R3-real traces, common-prefix replay, request-bootstrap statistics, 2x shell decomposition and diagnostic learned ranker | user-run 3×A800 correctness artifacts plus Mac CPU tests; no packed-tree/serving engine, Dual-Batch, SLO, calibrated latency or speedup claim |
| [#4 vllm-serving-v0.1](https://github.com/rzwang22/SpecRhythm/pull/4) | Draft/Open/unmerged; S0 CLOSED/PASS; S1-P G0–G3 PASS at `5a00049`; S2 G1 at `24b31a9` reported zero process exits/clean cleanup but rejected terminal-tail overlap; S2-only contract refinement awaiting retest | independent prefilled-KV resident pool, dynamic Poisson arrival/admission, Target/Serial/PingPong and engineering SLO/goodput | CPU/source contracts are not GPU qualification; finite-trace ideal PD-delivery boundary only |
| [#5 rolling-eager-v0.1](https://github.com/rzwang22/SpecRhythm/pull/5) | Draft/Open/unmerged; returned 465c215 B16 pair PASS; explicit audit-layer four-point retest pending | shared protocol, A+B, eligibility snapshot fix; default full and opt-in incremental Draft runtime audit, compact report/failure export | original results unchanged; latest Serial/eager 60.040693/45.305571 tok/s, native overlap 4.436982–4.585746 ms; new GPU correctness/overlap/performance PENDING |

## Current Serial-eager gate: Draft audit layers

The returned `465c2159b4b58d8c0e79fc3f66f38083e88368cb` evidence preserves original
execution/measurement/cleanup PASS. Both actual windows and native TP ranks have
been rechecked. The 102.52–102.58 ms dequeue-to-first-GPU delay contains two full
resident360 audits averaging 81.70 ms; five token-step functions average 388.62 ms,
including 269.22 ms of nested physical audit and 101.71 ms GPU forward sum. These
inclusive values are not additive. The old exporter reproduces a >64 MiB output
overflow; this evidence-export issue does not invalidate the performance points.

`draft_audit=full` remains default. Opt-in runtime maintains ownership at actual
allocator allocate/free boundaries and checks only affected requests during stable
token steps, with immutable prefix/version invalidation and fresh control reads.
Initial freeze, peaks, full counters, terminal receipts, fences and final release
audits remain functional. Runtime checks have separate counters, not fabricated
full-check counts. Target full audits and JSON control parsing remain unchanged.
A+B, K4, static eligibility, feedback priority and batch WAITING_DRAFT are unchanged.
Baseline S1/S2 and Serial/PingPong default algorithms remain intact.

The [design/diagnosis](rolling-eager-audit-layers.md) maps side effects and remaining
costs. The [repository runbook](rolling-eager-audit-runbook.md) executes exactly four
same-SHA independent roots (Serial/eager × full/runtime), each capacity → GPU
correctness → B16/360/2-warmup/30s performance with matched observation. Compact
point/four-point JSON keeps no raw arrays; bounded bundles record missing/truncated
sources explicitly. First failure preserves the original error and stops later
points while the parent interactive shell remains open.

Local gates: full Python 3.11 pytest **2133 passed, 3 existing skips** (246.303s);
Python 3.9 related suite and all five source-contract files **225 passed**, zero
skips (14.438s). The final compact-report/boundary-metadata checks also pass all
31 focused tests on Python 3.9. Ruff, both-version compileall, 294-file Python 3.9
grammar, all 13 repository Bash files, 13 Rolling Eager runbook Bash blocks and
git diff checks pass. CI status is reported separately after ordinary push.
GPU correctness,
overlap and performance of this change remain PENDING; no AutoDL connection or GPU
execution occurred. After delivery, wait for operator evidence; do not expand the
grid or change batch recovery scheduling. Historical sections below retain their
original results but no longer define the next test gate.

## Serial-eager startup latency after the returned A+B run

Operator evidence at `068c40a8ead1138568b7f846af348d3f732c28d4` retains all
execution/measurement/cleanup PASS: Serial 61.064037 tok/s versus eager 38.162742,
39/25 full B16 steps, 1869/1162 tokens, native overlap lower/upper ZERO. Mean
complete host step is 783.000/1215.756 ms; only 70.206/54.646 ms of the respective
windows lies outside complete steps. These step durations are not Target GPU time.
A+B batching is physically supported by 15 audits/step and one batched parent
repair per step (~19.918 ms event sum); eager GPU work remains ~101.355 ms/step.

Observation commit `33b6588004376e930ebabe1c9c768eca0b025371` adds bounded default-off
causal host spans and a v2 exporter
that retains both Target ranks' native bounds, explicit request/parent/work joins,
snapshot history counts, queue/feedback/batch-gate boundaries, and a disjoint
coordinator wall-time partition. The returned v1 package omitted raw Target
intervals although its server reporter used them; the ZERO summary stays valid,
but exact local Target timelines remain unavailable. No missing span is invented.

Source and a 32-round real-core CPU regression establish that provider evaluation
deepcopies growing unrelated history four times per successful rolling round;
its declared provider contract needs only request_id. The separate performance
revision uses an isolated immutable identity view, preserving
decision versions and all A+B checks. No further audit, fence, feedback-priority
or batch-recovery change is bundled with it. A-only is now historical, not a
new retest target. [Latency diagnosis](rolling-eager-critical-path.md) and the
[repository runbook](rolling-eager-latency-runbook.md) specify exactly three user-run
B16 points across observation/fix roots, with bounded failure export and an
interactive parent shell. New GPU performance/overlap remain PENDING.
Observation local gates: **2099 passed, 3 existing skips** on Python 3.11
(245.067s); Python 3.9 related suite **248 passed** plus **60 passed** in two
additional existing source-contract files. Ruff, both-version compileall,
287-file Python 3.9 grammar, 12 Bash scripts, eight Rolling Eager runbook blocks
and diff checks pass. The two initial shell-test failures were missing `python`
on PATH after creating a private test venv; the complete suite passes with the
correct PATH and unchanged assertions. GPU execution remains operator-only.

Immutable-view local gates: **2102 passed, 3 existing skips** on Python 3.11
(256.13s), **311 passed** on Python 3.9 across the related regressions and all five
source-contract files (13.763s, zero skips). Full Ruff, both compileall versions,
287-file Python 3.9 grammar, 12 Bash scripts, eight runbook Bash blocks and diff
checks pass. The 32-round guard preserves 129 evaluations and 160 committed tokens
while forbidding RequestState history copies in the eligibility hot path; provider
isolation, version regression and live switches are covered. Small CPU samples
record light-mode overhead separately, without a server performance claim.
Post-push GitHub CI is tracked independently in the PR delivery record.

## Historical Serial-eager B16 batch repairs

The A+B result has now returned; the startup-latency section above supersedes this
earlier delivery's pending gate. A-only is retained as history, not a new test point.

A-only `bbf12170118961a88244ae97af949eeee7d6028a` is committed and normally pushed.
A+B additionally batches compatible parent KV repairs and settlement audits while
preserving promotion, per-request accounting, ordinary fallback, terminal release
checks and failure fencing. The GPU correctness entry now exercises three-request
mixed physical batches outside performance measurement. The [two-commit runbook](rolling-eager-retest-runbook.md)
uses separate new roots and a Serial B16 control at each SHA, with first-failure
export/stop and an interactive parent shell. New GPU evidence is PENDING; original
PASS/negative results remain unchanged.

The next repair iteration has verified the returned 149732487-byte raw Draft
report (SHA256 `b769720b2c0e5520a76f07ba01dbc4c8aefe2d6a2a52a56fed3b2dead78f006a`).
All 63 measured starts follow parent full-accept feedback; 13 cycles each perform
17 pre-forward audits/6120 prefix visits, and all 156 parent-repair forwards are
B1. A-only now batches physical enrollment with one whole-pool audit while
retaining per-request checks and fresh before/after token-step audits. Parent
settlement remains unchanged in A-only. [Repair evidence and scope](rolling-eager-repairs.md)
record the independent commits; new GPU correctness/performance remain pending.
A-only local gates: **2068 passed, 3 existing skips** in the Python 3.11 full
suite (247.388s); **235 passed, zero skips** on Python 3.9 related tests and source
contracts. Ruff, compileall, 277-file Python 3.9 grammar, 11 Bash scripts and
Rolling Eager runbook blocks pass. No inference/measurement default was changed.
A-only GitHub push/pull-request CI is **8/8 SUCCESS**, verified with Draft/Open
unchanged. A+B local gates: Python 3.11 full suite **2083 passed, 3 existing skips**
(329.106s), Python 3.9 related/source suite **250 passed, zero skips** (9.578s).
Ruff, both-version compileall, 279-file Python 3.9 grammar, 11 Bash scripts,
4 Rolling Eager runbook blocks and diff checks pass. The runbook's six tests
execute substituted commands in the real parent/child Bash structure and prove
first-failure export/stop at correctness, Serial, eager or attribution stages.
Post-push A+B CI is reported separately at delivery. No local GPU execution.

## Historical diagnosis at 6ee3260 (before the raw Draft return and repairs)

The record below describes the evidence limits and next gate at the diagnosis
commit. The returned raw evidence and the current repair gate are recorded above;
the original measured result and its PASS status have not been changed.

The user-run result at `4a6725b054990e47b7e9c4cf63f38b995d029856` is retained as
valid: both B16 points pass execution, measurement and cleanup on the same frozen
resident360 workload. Step wall time rises 770.232 → 2444.746 ms while committed
tokens/step change only 47.923 → 45.615. The physical Draft regression has 14 exact
reference comparisons and complete release. It is separate from production
concurrency evidence. CI for that measured commit was verified 8/8 SUCCESS.

The returned small bundle reports native GPU overlap lower/upper = 0 and physical
status ZERO; outer UNKNOWN is the existing conservative label, not missing-bounds
evidence. It omits raw runtime/backend/transport timelines. Reported unhidden wait
is 8.923837 seconds (28.05% of the eager window); the other 22.887864 seconds remain
unpartitioned, not attributed wholesale to CPU, locks or scheduling.

Source and diagnosis-only CPU tests identify a late-launch mechanism: each B16
enrollment repeats a full 360-resident audit; 17 such audits precede first worker
forward. Feedback queued during admission is processed before stepping, cancelling
rejected-parent work without GPU launch. This explains how start/complete can equal
full-accept count but does not prove the actual latency cause without raw intervals.
Separately, eager parent repair executes per-request B1 materialize/fence, whereas
normal/recovery and eager token steps keep batching. A mixed-result regression
compares actual baseline/eager backend calls and conserves prefixes/KV/output.

This revision adds only a bounded offline exporter, characterization tests and
documentation. Runtime inference, sampling, K4, static eligibility, safety checks,
logging/measurement boundaries and the default Serial path remain unchanged.
Original result files are never rewritten. The next gate is the user's read-only
export of existing evidence, followed by separately evaluated minimal admission
and parent-settlement batching fixes if the evidence supports them. No AutoDL/GPU
execution, automatic scan or cross-run GPU UUID comparison is performed here.
See [diagnosis](rolling-eager-b16-diagnosis.md) and
[evidence commands](rolling-eager-evidence-runbook.md).

Final diagnostic-tree validation: Python 3.11 full suite **2058 passed, 3 existing
skips** (284.470 seconds), including **30 new cases** (25 offline-export, 4 owner
launch-order, 1 mixed-batch repair). Python 3.9 Rolling Eager and pinned vLLM source
contracts: **225 passed, zero skips**. Ruff, both-version compileall, Python 3.9
grammar for 275 source/test files, all 11 repository Bash scripts, the new runbook
block and diff checks pass. An initial sandboxed Python 3.9 run denied `ps` and
Unix socket `bind`; the same assertions pass with those required local operations
permitted. No test was relaxed. Exporting the returned small bundle leaves every
source file's hash/mtime unchanged and preserves original PASS while reporting
absent native timelines as MISSING. New-commit CI is reported independently at
delivery; the measured 4a6725b commit's verified CI remains 8/8 SUCCESS.

## Rolling Eager Continuation stage 2 (implementation record before operator retest)

The existing branch and Draft PR #5 now connect the shared stage-1 authority to
the real paged-KV Draft backend. Only Target TP rank 0 enqueues an immutable
continuation before the pinned vLLM model-forward hook. All Target ranks wait for
that enqueue acknowledgement. The existing Draft model runs on one owner thread;
mailbox feedback is processed between fenced batched token steps. Enrollment does
not wait for Draft execution and is counted separately from physical start.

The GPU backend fills the unmaterialized fourth parent candidate, then generates
the bridge and four eager candidates. Full acceptance plus an exact bridge and
dependency match retains KV and promotes the four candidates. Rejection or bridge
mismatch crops the logical frontier, invalidates cached logits and physically
repairs only the missing correction/bonus suffix. Private allocated blocks may
remain at their high-water capacity until fenced release; no per-round full
prefix replay or weakened baseline `commit_frontier` is used. Static eligibility
survives recovery, and the provider switch remains reusable for next-stage
PingPong scheduling without enabling GPU mixing in this revision.

`serial-eager` is an explicit fixed-diagnostic/decode-scan selection with K=4 and
an extra five-token Draft speculative reservation. Existing default modes and
scan points remain unchanged. The runtime passes authoritative terminal prefixes,
allows promoted/recovered/tail requests in the same batch, and extends the common
drain deadline through owner join, physical release and log flush. Summary,
status, errors, stop and bundle retain the eager evidence. Generated, promoted,
verified and accepted candidates are distinct; only Target commits are output.
Wait intervals are clipped to the existing decode window. GPU overlap requires
native CUDA clock bounds; host concurrency alone is `UNKNOWN`.

CPU integration covers the real adapter/owner/socket/backend code with substituted
device operations, including both feedback orders, repeated promotion → rejection
→ recovery → promotion, partial cancellation, TP submission order, tail/refill,
bounded shutdown and producer-to-summary accounting. The operator-only
`python -m specrhythm.continuation.gpu_check` entry compares real physical KV
continuations and recovery against separate ordinary Draft reference allocations.
Its deliberately constructed parent receipts are labelled `INJECTED_DIAGNOSTIC`;
natural Target event observations come from the actual Serial-eager B16 run.

Validation on the delivered stage-2 tree: **2028 passed, 3 pre-existing skips**
in the final Python 3.11.15 full pytest run (225.216 seconds), including **76 new
stage-2 cases**. Python 3.9.6 passes all 178 Rolling Eager cases plus 17 pinned
vLLM source cases (**195 passed, zero skips**). Ruff, both-version compileall,
Python 3.9 grammar for all 271 source/test files, 11 repository Bash scripts,
the runbook Bash blocks and Git diff checks pass. An initial full-suite run hit
the unchanged five-second process-cleanup test timeout; two isolated unchanged
reruns and subsequent full suites passed. No timeout, skip or assertion was
relaxed. The three final skips remain the opt-in CUDA test and two Linux-only
process-lifecycle cases. Stage-1 commit
`e4076628b10ccb5fef712dabae645f712c32cb51` was confirmed **8/8 CI checks SUCCESS**
before stage 2; stage-2 CI is reported for the actual pushed commit on Draft PR #5.

The [GPU runbook](rolling-eager-gpu-runbook.md) freezes resident360, the existing
S1/models/topology, two warmup rotations, a 30-second window, one repeat,
`samples=None`, setup 900 seconds, drain 60 seconds, buffered-live observation
and bound-prefix identity. Its child Bash stops at the first failure and retains
evidence. The delivery includes a copy filled with the actual final commit SHA.
No AutoDL connection or GPU execution is performed in this implementation task.
At that implementation checkpoint GPU evidence was PENDING. The operator has now
returned capacity/correctness and both qualified B16 points; the valid negative
performance result and remaining raw-evidence gate are recorded above.

## Rolling Eager Continuation stage 1 (CPU implementation; GPU integration PENDING)

Branch `codex/rolling-eager-v0.1` starts at verified commit
`5a16d00fd10778189db3addbff558a2260944b32`. Its new dependent Draft PR targets
PR #4's head branch `codex/vllm-serving-v0.1`, inspected at that same SHA.
PR #4 remains unmerged and its branch is unchanged; PR #2/#3 are untouched.

The independent `specrhythm.continuation` package implements fixed four-candidate
normal/eager work, a separate predicted bonus bridge, static stable-ID eligibility,
owner-local versioned request/proposal/continuation state, complete dependency
validation, both asynchronous arrival orders, repeatable promotion, rejection and
bridge-mismatch recovery, cancellation/release and exact-once Target accounting.
Immutable CPU scheduling views and dispatch adapters exercise cross-cohort next-stage
admission, normal/eager deduplication, recovery alongside another Target request,
provider switches and stale-intent rejection. A deterministic CPU token/KV executor
runs the real protocol; provider failure and foreign-owner dispatch are fail-closed.

The multi-round regression executes P0 → E1 → E2 rejection → normal R3 → E4 → E5 → E6
with independent proposal/continuation IDs and both feedback orders. Its six Target
commits total **28 tokens = 22 accepted candidates + 1 correction + 5 bonuses**.
It generates 30 early tokens, promotes 20 reusable candidates and five bridges,
discards five early tokens, and normally drafts four recovery candidates once.
Static membership survives rejection. EOS, short output/tail boundaries, late and
duplicate messages, decision versions, owner isolation and shutdown are exercised.

Validation: **102 new CPU cases pass on Python 3.9.6 and 3.11.15**; focused pinned
vLLM source plus new CPU contracts pass (119 cases). Final Python 3.11 full pytest
with the exact pinned source export: **1952 passed, 3 skipped** in 212.558 seconds.
The existing skips are one opt-in GPU test and two Linux-only process-lifecycle
tests; no new test is skipped. Ruff, compileall, Python 3.9 grammar for all 257
source/test files and git diff checks pass. CI for the delivered commit is reported
with the Draft PR.

No production GPU path, CLI mode or default simulator policy is changed. There is
**no GPU performance result**, no AutoDL connection and no performance scan in this
stage. CPU frontier evidence does not qualify actual GPU block reuse or mixed-batch
capacity. The next gate is explicit user instruction to integrate Serial/PingPong
GPU adapters and their owner/fence/KV/ready-mailbox boundaries; stop after this
Draft PR. See [rolling-eager-design.md](rolling-eager-design.md) for the audited
baseline rules, public API, test map and concrete next-stage module list.

## Resident360 PingPong warmup boundary repair (GPU retest PENDING)

Base f718ea4dd27848a34bc7fff92d7b906acfecec0d; its retained small bundle confirms
PingPong B64, Target B128 and Serial B128 PASS. PingPong B128 remains FAILED/INVALID
at warmup qualification, with zero process exits, clean360-request release,
27 full B64 window steps,5100 committed tokens and30029.949608 ms. Its snapshot
is excluded; the three valid points retain their original commit/server provenance.

The real ABAAB warmup has two complete pairs, one historical extra A and no pending
half at measurement start. Runtime accepted this boundary; qualification conflated
historical partials with current pending state. Unmodified f718 CPU regressions
reproduce the exact validator error for ABAAB/BABBA; ABAB passes. The repair shares
the chronological pairing definition, records compact actual warmup step/time/token
receipts and independently replays them with full population/lifecycle validation.
Two complete rotations, full forwards and the continuous30s window remain required.
The ready collection fix, scheduler, live UUID, KV/prefix/accounting, bounded drain,
primary error and all original fixed/S1/S2 defaults are retained.

The prior test gap was alternating-only result fixtures and terminal/refill after
warmup in integration. New cases cross the real asynchronous release/refill boundary
during warmup, retain historical steps in small bundles, and still reject actual
half-starts, non-full populations and corrupted token/identity evidence.
Local Python3.11 full pytest: **1850 passed,3 skipped** in223.21s; focused scan
regressions:72 passed. Ruff, compileall, Python3.9 grammar (247 files), all Bash
scripts/runbook blocks and git diff --check PASS. Linux CI is linked in the
delivery. No AutoDL/GPU/full retained artifact audit was run. Next gate: only new-root PingPong B128 plus its capacity,
using the foreground outer-if child-Bash runbook; no rerun of the three PASS points.

## Resident360 PingPong completion/refill repair (historical f718 implementation)

The retained e39afc1 scan confirms eight valid points: Target/Serial/PingPong at
B16 and B32, plus Target/Serial B64. PingPong B64 execution and cleanup passed,
but a pre-forward selection of1 instead of32 stopped after3780.603119 ms/462 tokens.
It remains INSUFFICIENT and excluded. The pool was not exhausted: one natural
completion, one B refill,295 queued, all360 settled and no owned process remained.

A CPU regression on unmodified e39afc1 reproduces the global-ready-count versus
single-cohort FIFO quota mismatch. The fixed scan-only hooks collect the finite
published FIFO independently of verification quota and wait before stock allocation
when the selected cohort is incomplete. Real release/refill continues between polls;
all waiting remains inside the original30s window. The final partial guard, old
S1/S2/fixed64 defaults, live UUID and actual token/KV/dependency checks remain.
The exact server ready distribution still needs the optional narrow raw-event export;
the constructed31+1 CPU distribution is not asserted as server fact.

New regressions join actual scheduler, ready claims, asynchronous owner, private KV,
natural terminal/refill, window expiry and360-request shutdown rather than testing
the scheduler/owner only in isolation. Compact bounded wait evidence is retained
in snapshots/light reports. The explicit single-point order override enables a new
root's PingPong B64 without borrowing historical B16 PASS files. Foreground strict
mode/ERR trap lives inside a child Bash, preserving the interactive terminal.

Final local Python3.11 full pytest on byte-identical source/tests: **1836 passed,
3 skipped** (234.34s), including Phase4/S1/S2 and pinned source contracts. Ruff,
compileall, Python3.9 grammar for247 files, Bash/runbook checks and diff checks
pass. Exact-commit Linux CI status is recorded in the delivery. No AutoDL,
GPU or full CPU offline audit was run. PR #4 stays Draft/Open, unmerged; PR #2/#3
are untouched. Next gate: new root, capacity + PingPong B64 single point, then only
on PASS B128 Target → Serial → PingPong. No automatic old-point reruns. All scan
PingPong B values use the repair; old B16/B32 need remeasurement only if a uniform
new-commit performance table is required. Preserve all old roots and labels.
See [design/schema](decode-scan-design.md) and [foreground runbook](decode-scan-runbook.md).

## Resident360 fixed-batch/time decode scan (GPU PENDING)

Independent `specrhythm-decode-scan` / `run_decode_scan.sh` implements the requested
Target/Serial/PingPong × B16/32/64/128 scan on top of the `e78e4c7` alias repair.
One deterministic original360 pool (3:3:2:2), fresh engines, real all-request
prefill before measurement, two full warmup rotations and a30s time-primary
window replace the old sample12 stopping rule only in this new entry.

Per-point physical capacity covers resident360 and its actual B/mode/config,
including B128 query positions and private KV/workspace. Full-forward guards stop
before a partial model dispatch; pool shortage is INSUFFICIENT, with no tail
filtering, altered EOS or automatic configuration/retry. Real commit/prefix,
refill, live UUID and buffered-live/bound-prefix checks remain. Final settlement
uses the existing bounded protocol, first-error/exit retention and owned cleanup.
Light JSON/CSV and a bounded small bundle exclude failed/stopped/insufficient
points; no full CPU audit runs in the default chain.

CPU regressions cover all12 scheduler shapes, all360 prefill for each mode,
real serial/async Draft state-machine shutdown and window/accounting/CLI boundaries.
The identical source/test bytes in a temporary local checkout pass full pytest:
1823 passed / 3 skipped (233.69s), including Phase4/S1/S2 and pinned source checks.
The Desktop checkout encountered an unchanged 5s shell-cleanup test timeout and
observed blocked file reads; the temporary checkout passed that same test without
changing its timeout. Ruff, compileall, all244 Python files with Python3.9 grammar,
Bash/runbook syntax and diff checks pass. Exact SHA and Linux CI outcomes are in
the delivery. GPU capacity,
full-window availability and performance are **PENDING**; no server was contacted.
Old fixed64/S1/S2 defaults and prior artifacts remain unchanged. PR #4 stays
Draft/Open/unmerged; PR #2/#3 are untouched. Next gate: user foreground B16 three
modes, then only after PASS the remaining nine points, using the
[scan runbook](decode-scan-runbook.md) and [definition/schema](decode-scan-design.md).

## Fixed64/32 bound-prefix alias repair (GPU revalidation PENDING)

The supplied failure archive confirms Serial at
`5b7081e0eb692822db18afedf3ce9bd36fcd9bee` failed with an unmapped verify-start
request (effective rc=125, measurement INVALID, owned cleanup PASS). That result
remains FAILED/INVALID. The optimized identity constructor copied binding maps,
leaving the real Serial constructor's diagnostic alias stale after later binds.
Dual lifecycle/reports also retained stale forward/reverse aliases.

The only runtime repair preserves both original owner-local binding container
references while replacing the matching strategy. Existing bindings/history,
future binds, hook readers, reverse aliases and reports now share the same state;
independent schedulers/proposers/TP workers remain isolated. Idempotent install
keeps the same optimized object, counters and lock. No checks or algorithms removed.

New core regression, run before the runtime edit on the failing commit: linear
2 PASS, bound-prefix 2 FAIL with the exact server verify-start error. After repair,
both empty/prebound cases pass through real S2Serial construction, fixed install,
later binds, start/end hooks and three rounds of actual acceptance/commit accounting.
The extended startup/identity suite passes 36 tests, including Target and both
Dual-based modes, report/lifecycle aliases, owner isolation and negative cases.
Earlier tests missed the replaced-object-to-legacy-alias consumer boundary.

Local full pytest: 1772 passed / 3 skipped (178.98 s), including Phase4/S1/S2;
Ruff, compileall, Python 3.9 grammar for all 236 source/test files, related Bash
scripts, extracted runbook Bash syntax and diff checks pass. Actual Linux Python
3.9/3.12, Phase4 Python 3.11 and pinned-source CI status plus the final SHA are
recorded in the delivery. No AutoDL,
GPU or full CPU audit was run. PR #4 stays Draft/Open/unmerged; PR #2/#3 untouched.
Next gate: [new-root foreground runbook](fixed-identity-runtime-runbook.md),
buffered-live + bound-prefix, prepare/capacity → Serial → check PASS/effect →
PingPong → summary/bundle; stop on failure. Old roots stay unchanged. GPU performance
PENDING; no performance benefit is claimed for this repair.

## Fixed64/32 bound-prefix matching experiment (historical implementation)

Serving HEAD was clean at c1dd96d8c86e321d66d52aea31eb7396bf06786b. The new supplied
buffered-live archive's independent JSON and nine export hashes were reviewed;
CPU reanalysis retained PingPong POSITIVE (752.749–758.354 ms, 20/24 positive steps)
and Serial ZERO with clean joins/clocks. Actual complete64 rotations are
1150.924 / 1354.666 ms; no new GPU execution or performance improvement is claimed.

One opt-in `--identity-matching bound-prefix` mechanism reuses validated frozen
prompt bindings after an immutable prefix-free proof, while comparing the current
full prompt on every bind and retaining original alias/change/errors. All fixed
scheduler modes and Target-worker proposers use the same definition; default linear
and S1/S2 remain unchanged. KV audits, proposal/prefix/version checks, required
barriers, live UUID, buffered logs, primary-error and bounded drain are retained.
Runtime/light/export metadata include actual owner counters and measured scheduler
deltas; the old 52.70 ms arithmetic residual is explicitly not a critical path.

CPU regressions exercise real fixed/S2/resident/Dual scheduling, physical-pool
checks and TP startup, and demonstrate fewer actual full identity scans with equal
selection/root/candidate accounting. GPU throughput impact remains unknown. See
[design/evidence](fixed-identity-runtime-design.md) and the
[two-mode runbook](fixed-identity-runtime-runbook.md). Next gate: new root,
prepare/capacity, Serial PASS before PingPong, then lightweight summary/bundle.
No AutoDL, GPU, full audit, stages, other short modes, G2/G3 or parameter grid by the
agent. PR #4 Draft/Open/unmerged; PR #2/#3 untouched. Stop after delivery.

Local validation: full pytest and focused fixed/S1/S2/Phase4 contracts PASS; Ruff,
compileall, Python 3.9 grammar, Bash/runbook syntax and git diff checks PASS.
Linux CI is checked against the pushed final SHA and reported in the delivery.

## Earlier fixed64/32 buffered-live runtime experiment

Based on e7452fc (clean serving HEAD), reviewed the new original-event attribution JSON,
not only its Markdown: PingPong ZERO has complete coverage, join errors=0, 12 rotations,
24/24 opposite-cohort owner operations finished before Target model launch. Draft commit
forward was missing from earlier proposal-only arithmetic: recorded per64 sums are
239.3625 ms Serial / 411.3140 ms PingPong; residual gap is 280.2046 ms, not CPU time.

Fixed-only `prepare --observation buffered-live` batches a reviewed post-run JSONL
whitelist within 256 records/1 MiB, preserves checksums/order/immediate contract validation,
and requires four conserving final-flush receipts within the shared drain deadline.
Default original-live and live UUID checks remain. Protocol sockets/control/ready and
release/error evidence stay immediate. No scheduler, model, precision, K, 64/32, KV or
proposal/accounting changes. No metadata cache, barrier removal or artificial overlap.
Summary now separates Draft proposal/commit, Target, other recorded GPU work, window
clipping, per-rank host costs, and final flush/drain. Proposal-only max formula is no
longer presented as an end-to-end prediction.

Independent integer-anchor projection fixes a CPU-reproduced 3 ns double-rounding
width error without adding tolerance or GPU synchronization. The real Serial-split
failure row is absent locally, so its old UNKNOWN remains; bounded bad-row details are
now emitted by CPU analysis into a new output. Full audit remains separate.

Local CPU validation: 1736 passed, 3 GPU skips; Ruff, compileall, Python 3.9 grammar,
Bash/runbook syntax and diff checks passed. Linux CI final status is in the delivery;
these checks do not qualify GPU performance.
Next: operator prepare/capacity, buffered Serial once, then buffered PingPong once only
after all Serial checks pass; stop and return evidence. No AutoDL connection/GPU run by
agent, no merge, no changes to PR2/3. See [design](fixed-buffered-runtime-design.md),
[schema](fixed-concurrency-diagnostic-schema.md), [runbook](fixed-buffered-runtime-runbook.md).

## Historical fixed64/32 b424 timing attribution (proposal-only first analysis)

The first-pass arithmetic below is retained as history; the new purpose-complete
recorded-forward correction above supersedes its 320.402 ms residual.

The operator's four continuous b424 points report execution/measurement/cleanup PASS:
Target 172.290, Serial 167.218, Serial-split 91.722 and PingPong 119.020 tok/s.
Serial/PingPong mean full64 rotations are 1144.140/1596.297 ms. The quoted forward
means explain 131.754 ms of the 452.156 ms arithmetic gap; 320.402 ms remains unattributed,
not CPU-only overhead. Split's 5606.630 ms explicit pre-step owner wait accounts for most
of its 5701.237 ms aggregate window difference from PingPong, without proving GPU overlap.

The small bundle lacks runtime/backend/raw events. Reported ZERO overlap is independently
UNKNOWN until event/clock coverage is checked. No runtime optimization was applied.
The new CPU-only, externally limited 120-second analyzer/export preserves observations,
unavailable causal timing, partial reports and old results. It uses actual IDs/rounds/
prefix commits, TP-safe interval unions and per-purpose Draft counters. Original-live
logging/UUID, required synchronization, S1/S2 defaults, 64/32/K4 and equality NOT_REQUIRED
remain unchanged. See [evidence](fixed64-b424-timing-attribution.md) and
[CPU-first runbook](fixed-attribution-runbook.md). Next action: analyze/export already
retained Serial/PingPong evidence; do not rent GPU merely for this tool-only revision.
GPU performance validation of any future optimization remains **PENDING**. No AutoDL
connection or GPU execution was performed by the agent.

## Fixed64/32 diagnostic stop settlement repair

The operator's real run at `89a962127f2e9a10a2564736e2124dae0abe51d8` failed in
Serial shutdown with 64 unresolved next-round proposals. Target abort had not settled
Draft state, and final runtime/backend artifacts were absent. The retained failure
bundle was inspected read-only; it cannot recover a complete Serial timing report.
Old results remain unchanged.

This repair adds a fixed-only owner settlement protocol: synchronize the final actual
Target commit without proposing, explicitly discard unused proposals, wait for issued
work/terminal drains, validate private KV and release it before logical cancellation.
Unused resident KV is released without inventing logical initialization. Idempotency
receipts prevent duplicate release. Normal Serial shutdown still rejects unresolved
proposals; S1/S2 algorithms, models/workload/order/K4, live UUID, 64/32 limits and token
length/EOS/round equality `NOT_REQUIRED` remain unchanged.

A single absolute drain deadline is enforced by both remaining RPC budgets and the
existing owned process supervisor, including blocked coordinator/worker teardown.
Atomic compact snapshots retain completed measurement through drain failure; primary
errors survive secondary cleanup/report failures. Light summary/bundle accept missing
final reports, retain partial evidence separately and exclude it from comparisons.
Full CPU audit remains an independent optional command.

Reproduction: before implementation, the CPU fixed `drive` -> real Serial server ->
real `DraftStateMachine.shutdown` path rejected the same 64 pending proposals. After
implementation the same case physically releases the 100-request resident pool,
generates no additional proposal and passes the inherited shutdown guard. Additional
CPU cases cover unsynchronized final commits, initial proposals at time expiry,
operator stop, grouped ready/inflight/tail work, physical-only residents, release/RPC
failure, snapshot retention and real subprocess deadline termination. Validation: 268 related S1/S2/fixed CPU regressions pass, including 88 fixed tests.
Full local pytest with pinned vLLM source audit: 1693 passed, 3 skipped. Ruff,
compileall, Python 3.9 grammar (226 files), runbook Bash syntax (9 blocks) and
`git diff --check` pass. Exact-commit Linux CI is reported with the delivered SHA. See the [design](fixed-concurrency-diagnostic-design.md),
[schema](fixed-concurrency-diagnostic-schema.md) and [foreground runbook](fixed-concurrency-diagnostic-runbook.md).

**GPU revalidation PENDING.** No AutoDL connection or GPU execution by the agent.
PR #4 remains Draft/Open/unmerged. Next operator gate: new root, prepare/capacity,
Serial short first, then Target/Serial-split/PingPong individually, light summary and
small bundle. No S2 GPU PASS or four-mode performance improvement is claimed.

## Independent fixed64/32 timing diagnostic

Based on `50025b734ed02086533f2302ed2b2951c262ec45`, PR #4 adds a separate foreground
`fixed_cli`/`specrhythm-fixed` entry and committed server script/runbook. It reuses the
existing mixed100, real resident prefill/KV and production Draft/Target paths. Main
modes are target64, serial64, serial-split32+32 and pingpong32+32; models/TP/K4 stay
fixed. Explicit per-cohort capacity cannot overfill the other group while one is busy;
held terminal slots remain charged through real release. S1/S2 defaults are unchanged.

Initial-state 32/64 shape samples use fresh processes/prefill and are distinct from
short continuous windows. Serial-split waits for actual owner completion before each
Target step. New in-memory host/CUDA evidence records control, UUID, pool audits,
serialization, sync and logs; per-round fences are not added. Actual TP rank times and
clock-bounded event-union overlap are not summed into fake kernel overlap/time savings.

Light execution/accounting/measurement summaries return without the full S2 qualifier.
The full CPU artifact audit is an explicit separate command and initially PENDING.
Window cancellations are not natural completions or full-request SLO attainment.
Cross-mode token/length/EOS/round equality remains NOT_REQUIRED. Split-cost formulas
require actual union/context/K shape evidence; missing samples remain null with reasons.

Validation: 60 focused CPU owner/coordinator/TP startup, shape/accounting, interval and
foreground-error regressions pass; S1/S2/fixed compatibility tests total 240 passes.
Full local Python 3.11 pytest with pinned source audit: 1665 passed, 3 skipped (GPU
opt-in and two Linux-only process cases). Ruff, compileall, Python 3.9 syntax, nine
Bash scripts and eight runbook blocks pass; exact-commit Linux CI is recorded with
the delivered commit. **No AutoDL connection or GPU execution by
the agent; GPU timing/performance PENDING.** Existing S2 failures/results are retained,
and this diagnostic does not confer S2 G1/G2/G3 PASS. Next action is the operator's
short run using [the committed runbook](fixed-concurrency-diagnostic-runbook.md), then
return the small bundle. [Design and measurement contracts](fixed-concurrency-diagnostic-design.md).

## Phase S: Serving Workloads & Arrival Replay

**S2 terminal-drain qualification correction** is based on
`24b31a9e0125d773697d92ec0ea333384616e539`. The operator reports zero Target/coordinator,
Draft and effective exit codes, valid completed cleanup and no owned PID left. G1 rejected a
13.152235 ms host overlap between cohort-A `finish_tail` and another A request's proposal
verification, with an empty request intersection. The reported conflict and exact intervals
are retained in the S2 design; full server artifacts were not accessed in this coding task.

The old S2 qualifier applied the active Draft opposite-cohort rule to every operation.
The source audit establishes that `finish_tail` validates a one-token terminal prefix,
materializes/fences it, frees only that request's private KV and generates no next proposal.
The S2-only refinement requires native terminal/owner evidence, correct prefix/version,
independent actual GPU bindings, whole-pool private-block validation and unchanged unrelated
prefix/block snapshots before allowing disjoint same-cohort verification overlap. Operation
name or different IDs alone never grant an exemption. Active drafting and all same-request
collisions remain blocked. Receipt/failure details identify the actual artifacts.

Terminal work/host overlap and token/KV/slot completion are reported separately. Proposal
pipeline overlap, real terminal forward/GPU cost, arrival-to-drain timing, held slots, owned
cleanup, primary failures and real exits remain intact. `release_finished` only extracts the
existing coordinator release decision for an actual blocked-owner regression. Scheduler,
arrival/capacity/SLO/model/K/budget/numerical rules, UUID live mode, five patches and S1 defaults
are unchanged; cross-run equality remains NOT_REQUIRED. Historical results are not rewritten.

New CPU regressions use real S2 dispatch and production materialization/release with simulated
hardware, then the full qualifier. The base qualifier rejected the same fixture; the refined
qualifier accepts its cohort-A terminal drain and still rejects true conflicts, invalid
terminal/proposal/prefix/KV evidence and early release/drain. A real asynchronous owner blocked
inside release keeps the actual coordinator slot held. Related S2/S1/Dual/PingPong regressions:
**219 passed** (including **23 new tests**). Full local Python 3.11 pytest with the pinned
source audit: **1605 passed / 3 skipped** (GPU opt-in and two Linux-only process cases).
Ruff, compileall, Bash and diff checks pass; exact-head CI is recorded in the handoff and
PR #4. The next operator run uses the SHA-filled runbook and a
new root, remeasures capacity, repeats calibration/G0 and complete three-mode G1, then permits
G2/G3 only after the preceding gates pass. Old failed directories/seals remain preserved.
No AutoDL connection or GPU execution was performed. S2 GPU/performance remains unqualified.

**S2 PingPong UUID startup correction** follows the operator's run at
`c12b3768eaaaeca3ecde03999440b7b91763b128`, retained at
`/root/autodl-tmp/SpecRhythm-data/results/phase-s2/s2-c12b3768eaaa-20260910T012202Z-1476`.
Reported results: capacity PASS (small100/large390, 3:3:2:2), calibration/G0 PASS, G1
Target and Serial each completed ten and passed; PingPong failed at first verification.
There is no passed G1.json, so G2 remains blocked. These partial operator results do not
establish complete S2 GPU/performance qualification, and the old root is preserved.

S2 had called only the ordinary worker snapshot. The inherited Dual verification end hook
requires the `DualVerificationUuidQuery(worker)` installed by `worker_dual_runtime_snapshot`,
which the legacy Dual runner invoked but S2 omitted. S2 now uses an explicit PingPong startup
RPC on both Target ranks before prefill/verification. Later capacity/memory snapshots only
read the query evidence, preserving its identity and counters; Target/Serial remain on the
ordinary startup path. Live UUID behavior, TP binding, algorithms, capacity selection, trace,
SLO, models/K/numerical settings, measurement boundary, failure reporting and cleanup are
unchanged. Probe counters may be zero without satisfying a nonempty UUID A/B experiment.

The new CPU regression executes real S2 configuration, proposer constructors, startup RPC
callbacks, inherited verification start/end on both simulated ranks and final evidence reads.
It reproduced `AttributeError: 'S2PingProposer' object has no attribute 'uuid_queries'` at the
original `vllm_dual.py:659` before the fix. Earlier tests stopped at LLM construction or used a
canned RPC/step result, missing that integration boundary. The shared legacy UUID hardware
fixture now explicitly clears unrelated serving profiles so focused tests are order-independent.
Historical `24b31a9` local validation: S2 **53 passed**, S1 **104 passed**, legacy Dual UUID **37 passed**
(combined **194 passed**); full Python 3.11 pytest with pinned source audit **1582 passed,
3 skipped** (GPU opt-in and two Linux-only process cases). Ruff, compileall, eight repository
shell files, eight runbook Bash blocks and diff checks pass. Exact-head Linux CI is recorded
in the handoff and PR #4. The next operator run uses a new root and newly measured capacity
(390 is not hard-coded), repeats calibration/G0/G1, and
permits G2/G3 only after G1 succeeds. The agent has not connected to AutoDL or run GPU work.

**S2 — GPU-resident Prefilled-KV Dynamic Decode Serving** continues from
`5a00049e2eabf09f535fdd5f187f77406f6dcfe2`. The operator reported S1-P G0–G3 PASS
in `/root/autodl-tmp/SpecRhythm-data/results/phase-s1/s1p-5a00049-20260909T144802Z-1469`.
That user-supplied baseline status supersedes the earlier S1 GPU-pending entries below;
the coding agent has not independently rerun it. S0/S1 evidence remains read-only.

The independent `specrhythm-s2` entry freezes nested 3:3:2:2 small100/large500 inputs,
common capacity-driven decrements of ten from actual Draft/TP Target rank probes,
independent Poisson/order seeds and a pre-main engineering SLO policy. Every new attempt
recreates private Target/Draft KV before one uninterrupted arrival-to-drain observation.
Synchronous EngineCore steps permit FIFO admission only at safe boundaries; a separate
arrival thread queues arrivals while GPU work is busy. The common active limit is 128,
distinct from 512 resident slots and the 4096 query-token cap. PingPong cohorts are filled
dynamically without waiting for a future empty cohort; Serial can use all active slots.

Native worker commits, KV block isolation, Target structure, sampled-row TP consensus,
within-run budgets/EOS/accounting, real exit and owned cleanup remain mandatory. Cross-run
token/length/EOS/round equality remains NOT_REQUIRED. Sparse singleton execution, zero
overlap, lower speed and zero SLO attainment are observations, not failures. Queue-inclusive
decode average latency is distinct from TPOT, and output work/throughput/makespan are
reported together. Foreground tagged raw logs, precise primary failures, sealed evidence,
fresh-state recovery and a small upload JSON are included.

Initial `c12b376` local Python 3.11 validation with the pinned vLLM source audit: S2 contracts **48 passed**,
S1 compatibility **104 passed**, Phase4 **1182 passed / 2 platform skips**, and full pytest
**1577 passed / 3 skips** (one GPU opt-in and two Linux-only process cases). Ruff, compileall,
eight repository shell files, eight S2 runbook Bash blocks and staged diff checks pass.
Linux CI for the delivered commit is recorded in the final handoff and PR #4. The new
contracts cover real adapter startup order, synthetic private
allocator residency/restoration, independent arrivals, dynamic slots/cohorts, common
capacity and traces, native-shaped accounting, offline requalification and real owned CPU
child failure/cleanup. Fixed-source tests inspect vLLM's synchronous client, retained cached
requests and scheduler allocation hooks. The partial operator results reported above supersede
the initial GPU-pending status. PingPong with the startup correction, complete G1, and G2/G3
measurements/performance remain **PENDING operator validation**.
The coding agent has performed no GPU execution and no AutoDL connection.

See [S2 design](phase-s2-design.md), [S2 schema](phase-s2-schema.md) and
[S2 server runbook template](phase-s2-runbook.md). The handoff includes a separate rendered
runbook with the final full SHA; each commit uses a fresh S2 result root. PR #4 remains
Draft/Open/unmerged; PR #2/#3 are unchanged. Earlier Phase S/4 entries below are historical.

**S0 — Mixed Real-Text Workload Foundation** adds independent versioned serving
requests, source acquisition/locking, deterministic selection, real Qwen3
tokenization, timestamp-only Mooncake composition, full read-only validation and
sealed review artifacts. Work started from verified branch head
`0684a29800519c02a7b3c2558952ad344b17a9bd` on `codex/vllm-serving-v0.1` with a clean
tree; PR #4 remains Draft/Open/unmerged. The Phase 4 entries below record the
earlier resident work; S0 changes none of its execution or historical evidence.

The real main1000 quotas are chat/code/summary/reasoning 300/300/200/200;
calibration200 is 50 per class, with joint deduplication before splitting.
Selection seed is 1664, independent class-slot seed 1665, thinking is disabled,
natural EOS is enabled and full template token counts obey the 4096 context
constraint including a four-token reserve. The source lock pins all original
train files, Mooncake trace and remote tokenizer by commit and file SHA256.
No source answers, synthetic replacements, Mooncake anonymous lengths or prefix
identities become request content.

Local Python 3.11 real-data construction and independent full validation are
**PASS**, with 1000/200 requests and source/artifact hashes unchanged across
validation. Separate-directory rebuilding also produces identical JSONL and
core manifest. Semantic workload SHA256 is
`05b5f2efad0a4ac1771c9a18d847c08d95d360432e1c9cfa55a82ba2f63cba02`.
Server-local Qwen3-0.6B/Qwen3-32B tokenizer alignment is now **PASS**, based on
the returned checksum-bound 1200-prompt server report. Synthetic fixture tests are
separate and default CI downloads no dataset or model. Linux Python 3.9/3.12
retain the full suite; Python 3.11 additionally exercises the S0 CLI/contracts.

S0 local checks: 62 fixture tests pass on Python 3.9.6 and 3.11.15; the final
Python 3.9 full suite passes 1425 tests with 3 skips, and the Phase 4 suite passes
1182 with 2 skips, with the pinned vLLM source audit enabled. Ruff, compileall,
staged diff checks and all eight runbook Bash blocks/embedded Python pass.
Linux CI results for the delivered commit are linked from PR #4.

See [S0 design](phase-s0-workload-design.md) for source mappings, algorithms,
schema/hash contracts and the real-data summary, and [S0 CPU runbook](phase-s0-workload-runbook.md)
for exact server paths, fetch/import, full validation, dual-tokenizer checking,
rebuild and review bundle commands. Real data and generated workloads stay outside
Git. The coding agent performed no GPU execution, model inference or AutoDL
connection. [The appended closure review](phase-s0-closure-review.md) records the
returned archive SHA256, all 35 matching inventory entries, split disjointness,
server reports and the assistant's review of all twenty full prompts. S0 is
**CLOSED/PASS**. No additional human signature is asserted. The old sealed JSON
`manual_sample_review=PENDING`, original checksums and tar remain unchanged.

| Stage | Scope | Status |
| --- | --- | --- |
| S0 | Four-class real-text construction and validation | CLOSED/PASS; server machine/tokenizer/rebuild evidence and assistant content review accepted |
| S1 / S1-P | Resident three-mode performance qualification | Previous GPU gate stopped by old exact-output policy; S1-P cancels cross-run equality, CPU/CI recorded below; fresh GPU G0–G3 PENDING |
| S2 | Target dynamic arrival, queueing and streaming timestamps | Future; not started |
| S3 | Serial and dynamic PingPong join/leave/setup | Future; not started |
| S4 | SLO and load calibration on the independent calibration set | Future; not started |
| S5 | Formal 1000-request serving comparison | Future; not started |

Phase 4C retains its Dual-Eager name. The existing Phase 4 primary evaluation is
resident decode-only; no runner consumes the new arrival field yet. Dynamic
mixed prefill/decode, calibrated SLO, PD/KV handoff and a new GPU performance claim
are outside S0/S1. `slo_policy_ref=null` / `calibration_status=pending` is not an
accepted calibrated experiment.

### S1-P implementation handoff

The Serial startup follow-up starts at `645635d5a54d886ac874a9e1046ae8ad957bef9b`.
The operator reports G0/Target complete and Serial failing on missing
`decode-ready-context.json`. Source audit confirms S1 bypassed the legacy Serial
CLI's context creation. The S1 adapter now creates it exclusively before calling
the real Serial runner/LLM, using the existing builder, real provenance parser and
frozen config/patch/workload/Git/execution bindings. The regression failed at the
simulated LLM construction entrance before the fix and passes afterward without
precreating context. Startup exceptions remain primary; missing later reports are
secondary diagnostics. Foreground logs, actual return codes, owned cleanup and
NOT_REQUIRED token-equality policy are retained. Runtime/scheduler/backend/Target
and five-patch implementation files are unchanged. Current focused CPU suite: 104
passed; full local Python 3.11 suite with the pinned source audit: 1529 passed,
3 skipped. Ruff, compileall, helper syntax and 12 runbook Bash blocks pass.
Linux CI evidence for the delivered commit is recorded in PR #4.
No GPU run or AutoDL connection was performed. Old roots/results stay unchanged;
next operator step is a new root and foreground G0/G1 at the delivered SHA.

The backend/frontend follow-up starts at `08cf93a531dc928e02820b14412398b952592a49`.
S1 qualification now checks the report producer's `vllm-batched-paged-kv-draft`
identity, independently from shutdown complete, zero live requests and execution
failure false. Each check retains field/expected/actual/artifact evidence; failures
print those details and bounded child-log tails directly. Regression artifacts use
the actual backend report producer with a CPU worker. Cross-run equality remains
**NOT_REQUIRED** under the same `s1-performance-v1` policy.

The default runbook uses foreground `gate`. A read-only log mirror labels each
mode and Target/Draft source while both child streams still write to their original
files. Real CPU subprocess tests prove output is visible before exit, preserve
Target exit 7 and Draft startup exit 9, and exercise the existing owned cleanup.
No backend/scheduler/Target/measurement/patch implementation changes are included.
Current local Python 3.11 checks: 100 focused tests, 1525 full-suite passes with 3
skips; Ruff, compileall and runbook syntax checks pass. Linux CI for the delivered
commit is recorded in PR #4. No GPU execution or AutoDL connection was performed.
All old failure artifacts stay unchanged; the next server step is fresh-root
foreground G0/G1 using [the runbook](phase-s1-runbook.md).

S0 is **CLOSED / PASS** with its original sealed inputs and the assistant review
scope preserved. S1-P starts at clean `cd36d18c63ac706548fabccb4e5cf6e0f15e5897` on
`codex/vllm-serving-v0.1`; PR #4 remains Draft/Open/unmerged. PR #2/#3 are untouched.
The operator reports that the previous raw Target attempt-002 completed two real
runs but the old gate rejected `repeated_run_deterministic=false`. That old policy
result stays unchanged at `/root/autodl-tmp/SpecRhythm-data/results/phase-s1/cd36d18-20260909T110103Z-1489`.
It is background evidence, not an automatically completed S1-P gate. S1-D is cancelled.

S1-P uses versioned `s1-performance-v1`. **Cross-run/cross-mode output equality is
not required or performed.** Independent tokens, bootstrap, final prefix, output
length, EOS/finish reason, proposal/acceptance and rounds may differ. All requests
still share frozen input IDs/order, prompt tokens, 512/1024 budgets, model/tokenizer,
sampling, K4 and existing numerical/backend/resource configuration. No new async
sweep, algorithm change, patch change, resampling or ignored EOS is introduced.

Each run still validates completion exactly once; its own bootstrap, final tokens
and committed-event reconstruction; EOS/budget; proposal prefix/round, correction,
bonus, Target/Draft KV/sync and TP mapping; real measured boundaries; nonzero exit
and owned cleanup. Old Phase4 validator defaults remain strict. G0, manifest, result,
gate, seal, resume and offline comparison bind the same policy/schema. Old roots
cannot be silently migrated or reused. G1 now runs Target, Serial, PingPong and one
independent PingPong repeat; G2/G3 sizes and fixed rotations stay unchanged.

Each repeat reports actual full/timed/bootstrap tokens, completed requests,
EOS/cap/setup-terminal counts, task length distributions, makespan, actual tok/s,
completion latency, Target B/Q, Draft/acceptance work and physical overlap evidence.
Mode-level median throughput ratios use each run's actual tokens. Makespan ratios
carry actual output counts and are not equal-work speedups. Equality fields are
null/NOT_REQUIRED; count equality is descriptive only. The label is
`resident decode-only three-mode performance observation`, with no unconditional
end-to-end improvement, pure batching, equal-total-GPU or serving TTFT/SLO claim.
Zero overlap or lower throughput does not fail qualification; invalid/partial runs
do not enter complete-run aggregates. G3 capacity shortage remains BLOCKED without
changing N or budgets. Measurement boundaries and instrumentation are unchanged.

The committed helper sets `OMP_NUM_THREADS=1`, `VLLM_ALLOW_INSECURE_SERIALIZATION=1`
for the existing local callable RPC path, clears USE_TORCH/TF/FLAX and pins GPU Python
and visibility per child. Actual values are recorded. Detached launch, status,
owned cleanup, fresh-attempt resume and immutable review bundles remain available.

S1-P CPU contracts: **87 PASS** (synthetic only). Full local pytest with the pinned
vLLM source audit: **1512 passed, 3 GPU/platform skips**. Ruff, compileall, diff checks
and all 11 runbook Bash blocks/helper syntax pass. Final-SHA Linux CI runs after push;
its actual completion status and links are recorded in PR #4 and the delivery handoff.
Existing Python 3.9/3.12 CI and Python 3.11 Phase4/source contracts remain; the 3.11 job
also runs the S1-P regression file. **S1-P GPU / performance: PENDING**, awaiting a
fresh server G0–G3 run. No GPU inference or AutoDL connection was performed by the
agent; no new GPU correctness/performance result is claimed, and S2/S3 have not started.

See [design and audited call chain](phase-s1-design.md), [versioned result/schema](phase-s1-schema.md)
and [copyable server runbook](phase-s1-runbook.md). Stop after this handoff and await
S1-P server results; recovering cross-run token equality is not a prerequisite.

## Phase 4A.0–4A.1: vLLM freeze and Serial Disaggregated correctness

Phase 4 is stacked on the exact frozen PR #3 head
`34c7ea9836c2595c8a8aeaeb5680709520edd3d8` and does not modify Phase 3 algorithms or results.
The serving integration freezes vLLM `v0.25.1` at commit
`752a3a504485790a2e8491cacbb35c137339ad34` in a separate Python 3.11/PyTorch 2.11.0 environment;
vLLM is not a dependency of the Python 3.9 simulator package.

Phase 4A.0 brings up Qwen3-0.6B TP=1 on physical GPU 0 and Qwen3-32B TP=2 on physical GPUs
1–2 as separate stock V1 offline engines. It validates exact source/install provenance, physical
placement, every TP rank's local parameters and memory, selected attention backend, repeated
greedy output and token-level comparison with the frozen HF trajectory on five corrected R3-real
requests. Startup/prefill/decode/wall timestamps are recorded only for bring-up observability.

The Phase 4A.0 adapters freeze future candidate/verification/request-state semantics without
importing simulator policies or proxy latency. vLLM built-in colocated speculative decoding is
not `serial-disaggregated` or SpecRhythm `dual-batch`; vLLM DBO is an intra-model-executor
microbatch overlap and is explicitly disabled. No GPU experiment was run by the Mac coding agent.
Phase 4A.1 changes the serving correctness reference to immutable stock vLLM Target-only greedy
output. The Phase 3 HF trajectory remains advisory provenance only. Before any patch is applied,
the reference command runs the same five corrected R3-real requests twice, verifies token and
termination determinism, records model/tokenizer/runtime pins, and freezes
`stock-target-reference.json` without overwrite. Patched Target-only and two independent Serial
runs must match this reference exactly.

The Serial path uses one persistent Qwen3-0.6B Draft process on GPU 0 with per-request mutable KV,
and the stock vLLM speculative scheduler, batched Target verification, rejection sampler and KV
accounting on Target TP=2. A local Unix-domain-socket custom proposer carries only committed-token
deltas and proposals. The fixed vLLM custom proposer API lacks request identity and exact verify
boundaries, so one zero-fuzz Python patch adds those observer hooks to
`gpu_model_runner.py`. It changes no scheduler/sampler/KV/attention/C++/CUDA code and is inactive
for Target-only generation. Exact base, patch and installed-file hashes are validated.

Every round proves Draft → transfer → Target verify → state sync → next Draft ordering. Draft KV
is cropped after rejection and appends the Target correction/bonus; full-context replay per round
is forbidden. Proposal, accepted/rejected, correction/bonus, bootstrap/tail and final output
accounting are checked independently. The Mac agent ran CPU tests and applied/restored the patch
against the exact vLLM source but did not run CUDA or produce GPU results.

The user-run default-mode A/B gate completed most lifecycle checks, but one of five Serial
sequences diverged from stock Target-only at generated position 1 after an exact BF16
log-probability tie changed under speculative batch expansion. Exact correctness therefore has
not passed. Phase 4A.1.1 adds a symmetric C/D batch-invariant mode, per-rank effective-mode
evidence, actual proposal/logits/position mappings, rejected-KV rollback validation, and
single-request local/remote fixed-proposal controls. Existing A/B artifacts remain immutable
default-mode provenance.

At the exact pinned vLLM commit, the batch-invariance documentation and FlashAttention backend
set the minimum NVIDIA compute capability to 8.0. A800 therefore passes the hardware preflight,
but that preflight intentionally leaves `batch_invariant_effective=false`. The first C attempt on
`a7fe058d` stopped before engine creation because stock runner verification imported vLLM before
mode configuration; the runner verifier now uses package metadata without importing vLLM. The
subsequent C1/C2/D1/D2 artifacts and corrected immutable-artifact validator completed with
`outcome=A`, `valid=true`, exact Target-only repeats, exact Serial repeats, D==C, valid
diagnostics, and identical semantics for all 24 keyed rounds. Their raw cross-request event order
differs, which is scheduler interleaving rather than a semantic change. This Phase 4A.1.1
conclusion is frozen and is not reinterpreted by Phase 4B. See
[phase4-vllm-integration.md](phase4-vllm-integration.md),
[phase4-vllm-source-audit.md](phase4-vllm-source-audit.md), and
[phase4-vllm-server-runbook.md](phase4-vllm-server-runbook.md).

## Phase 4B.0–4B.1: Dual-Batch contracts and GPU correctness readiness

Phase 4B adds a linear, non-eager serving control plane without changing vLLM rejection,
attention, paged-KV, model, TP or default scheduling semantics. The user's A800 artifact at
`96842c8a1e6ffd70c5c1321eecd7384ad74cf542` proved stable/internal identity mapping, but it also
proved two failures: the cadence-based readiness workaround allowed an unproposed Target decode,
and the shell wrapper did not prove that EngineCore/TP descendants had exited. That artifact is
integration-failure provenance, not a GPU correctness result.

Phase 4B.0a replaces cadence mutation with an explicit request-level predicate in independent
patch `0002`. Waiting/Drafting decode is forbidden; a matching prefix-version/hash/round proposal
or legal Target tail is allowed; setup prefill is separate. A blocked request consumes no stock
token/KV budget and does not prevent later work. Scheduler artifacts record the decision for every
request/cycle. Target launch now owns one session/PGID, propagates the coordinator exit code,
records descendants and TERM/KILL actions, verifies Draft/socket cleanup, and blocks the next run
after invalid cleanup.

Phase 4B.0b establishes `DecodeReadyProvider -> DecodeReadyManifest -> consumer` and implements
only `ResidentWarmStartProvider`. Untimed real-KV setup performs Target prompt prefill plus exactly
one bootstrap, then initializes Draft through the same committed prefix. Manifest validation and
a TP barrier precede `measurement_start_ns`; initial proposal generation is forbidden before it.
The first Target forward must consume `[bootstrap]` for Target-only or
`[bootstrap]+proposal` for Serial at exact contiguous positions. A third observer patch records
actual forward boundaries and inputs.

Phase 4 main evaluation is decode-only. Resident warm start is real-KV decode-stage isolation,
not an end-to-end PD deployment. KVConnector handoff remains a future provider and is not
implemented. CPU contract tests pass locally; the coding agent has not run CUDA.

The real-A800 Gate A.1/A.2/A.3 run passed. Gate B then proved that pinned vLLM may legally deliver
only one frozen request in an initial proposer callback; both resident proposers incorrectly
required the whole workload in that callback. The failed `d6c7aa8` directory is preserved.

The corrected contract accumulates immutable stable-ID observations across callbacks. A dedicated
EngineCore scheduler admits prompt/bootstrap work, freezes requests after their first output token,
and releases them only after full-set validation, one TP barrier, manifest creation, measurement
start, and atomic setup-ready publication. The real-A800 `98ec816` run reached the second
incremental request and `_complete_global_setup`, proving the earlier whole-batch assumption was
removed, but failed because the JSON-compatible observation list was reconstructed without tuple
normalization. `ResidentSetupObservation.to_dict/from_dict` is now the only serialized boundary;
Target and Serial both use it and reject malformed token types. The resulting L2 Target run passed
on A800 and is preserved read-only.

The resident Serial attempt at `5db8657` is diagnostic-only because Draft was accidentally
started twice and its live PID provenance was overwritten. Its manifest/proposal hashes and
timing/admission logs nevertheless showed correct round-zero publication and first scheduling,
followed by an erroneous second installation pass after the live prefix advanced. Resident Serial
now owns each initial proposal through `published -> installed -> consumed`, uses pinned vLLM's
`scheduled_spec_decode_tokens` as consumption evidence, preserves detailed fail-closed diagnostics,
and leaves consumed requests exclusively to normal proposer rounds. The subsequent real-A800
resident Target/Serial reruns passed; Phase 4B.0 correctness infrastructure is frozen and is not
reinterpreted by the new Dual path.

GPU 0 runs one persistent Draft service. Heavy Draft model work is serialized on its background
worker queue; Unix-socket calls only enqueue work or poll completed proposals. The Target
scheduler on GPUs 1–2 injects only ready proposals and delegates actual scheduling to the stock
vLLM scheduler. Every proposal carries a canonical ID, round, prefix version/count/SHA256, Draft
KV lengths and token list. A mismatch fails before verification; one request cannot own two
proposals or be drafted through an unverified prefix.

Phase 4B.1 now starts Dual from the same logical decode-ready state. Draft initialization produces
no proposal; rank-zero manifest validation and the Target TP barrier precede measurement; first
proposals are asynchronous and post-boundary. The scheduler emits per-request decisions and a
separate proposal lifecycle, while the unified validator compares Target, Serial and one or more
Dual runs exactly, checks keyed repeats and all logical invariants, and proves that it did not
mutate its inputs. Local CPU tests pass. A complete two-request run at `3ee1c3e` passed Target,
Serial, both Dual executions, controlled Cases A/B/C, the exact output triangle, keyed
repeatability and all underlying semantic components. Read-only replay resolved the remaining
instrumentation issues. A legal Target tail may retain historical proposal metadata only when a
prior `CONSUMED` lifecycle event proves there is no live proposal; Draft readiness and the ordered
one-token terminal transitions remain mandatory. Dual-1 contains a real 57.989848 ms
cross-request temporal overlap; Dual-2 intentionally has none under two-ready coordination. The
historical per-verify rows alias both TP ranks to GPU1 because they used process rank/device zero,
while authoritative worker snapshots already prove Target placement on GPUs 1 and 2. Current
instrumentation uses the actual active CUDA device and cross-validates rank, physical GPU and UUID.
The preserved result remains a correctness artifact only and carries no performance claim. See
[phase4b-decode-ready.md](phase4b-decode-ready.md) and
[phase4b-dual-batch.md](phase4b-dual-batch.md). The active server procedure is
[phase4b1-dual-correctness-runbook.md](phase4b1-dual-correctness-runbook.md).

The first Gate-1-only preparation at `b9a0d6d` froze a deterministic controlled-2 stock reference
and applied all three pinned patches to their exact final hashes, then stopped before any serving
run because its helper incorrectly reused the stock-only checker for the patched installation.
This was not a Dual correctness failure. Patch-state validation now has mutually exclusive exact
`stock` and `patched` modes, immutable success/failure manifests, and explicit helper calls. A
fresh Gate-1-only A800 rerun is required; the earlier root remains immutable provenance.

That rerun at `7e4f871` validated both explicit state checks on A800 and completed resident Target.
Resident Serial then failed only when the common Target-forward observer accessed the Dual-only
`proposal_id` attribute on the legacy Serial `Proposal`; its scheduler had already submitted both
requests for speculative verification. Dual-1/Dual-2 did not run, so Gate1 remains not evaluated.
The observer is now proposal-protocol-aware: it preserves a real canonical Dual ID and records
`null` for Serial or no pending proposal. The Serial schema, execution semantics and all Dual
scheduler/state/lifecycle logic remain unchanged.

Gate semantics are now separated without weakening evidence. Gate1 is controlled two-request
semantic correctness and reports per-run temporal/hardware-qualified overlap independently.
Gate1.5/Gate2 keeps at least one hardware-qualified positive overlap mandatory on at least five
requests through the unchanged asynchronous path. Default validation remains overlap-required;
only controlled Gate1 opts into `separate-gate`. An explicit legacy read-only authority mode
accepts only the exact `3ee1c3e` source, recomputes semantic plus runner-only invariants, and
supersedes only structurally proven historical errors. The helper continues to preserve run and
validator exit codes while always checking cleanup. Subsequent user-run A800 evidence established
Outcome A for controlled Gate1 and default-asynchronous corrected-5 Gate2, including exact
Target/Serial/Dual output equality, correct GPU1/GPU2 TP identity and hardware-qualified overlap
in both Gate2 Dual runs. At that historical checkpoint, Gate3 recovery was the only authorized
next GPU action and Phase 4B.2 remained blocked. The later numerical qualification and explicit
human progression decision documented below supersede that project-level block without changing
the immutable historical artifacts.

The first corrected-100 Gate3 attempt at commit `eba0df4` completed preparation, froze its one
allowed deterministic stock pair and applied the patch stack, then failed in resident Target
setup before global readiness. With 100 requests, pinned vLLM enabled chunked prefill under the
16,384-token budget and delivered one proposer callback containing requests at different setup
stages. Target, Serial and Dual had all assumed every row contained exactly one bootstrap token;
that assumption happened to hold for 2/5 requests. Serial and Dual were not run and Gate3 was not
evaluated. The failed directory remains immutable infrastructure-failure provenance, not stock,
Target-token, Serial, Dual or overlap failure evidence.

All three resident consumers now use one dependency-free row classifier. The authoritative
sampled-token row distinguishes bootstrap from no-bootstrap, while a minimal pinned worker hook
supplies the actual post-forward materialized position count. Partial prefill never binds opaque
identity or initializes Draft; full prompt without a sample remains pending; exactly one sampled
bootstrap records/initializes once; more than one output before global readiness fails closed.
Each wave is logged so an early-bootstrap request can be shown frozen while later requests keep
prefilling. Exception cleanup now snapshots an owned Unix-socket inode, proves the Draft PID dead
before removing the unchanged stale socket, and keeps the lifecycle guard on any live process,
socket identity change or leaked Target descendant. The earlier deterministic stock-100 reference
is reusable byte-for-byte because its stock/model/tokenizer/sampling/runtime/workload contract is
unchanged; reuse records both file hashes and commits and does not measure another stock pair.

The user-run `32b09a6` recovery proved that scale-safe setup and cleanup work for corrected-100:
all bootstrap observations, global decode readiness, the first Target forward, measurement
boundary and TP2 placement passed. Exact output compatibility still failed for four requests at
generated positions 3, 4, 12 and 2; the other 96 were exact. In every divergence the immutable
stock artifact has a `0.125` top-two log-probability margin while resident execution collapses the
same token pair to equal values. This is not accepted as a harmless tie because the stock
preference is nonzero.

The first attempted pair at `c142fa7` failed in both stock TP workers before any numerical
checkpoint because the observer incorrectly treated speculative-only common attention metadata
as the generic block-table authority. Resident and comparator were not run. Its exact directory
is immutable `diagnostic-infrastructure-failed` provenance. The generic observer at `e73e884`
then completed exactly one stock-style and one resident corrected-100 run. Read-only comparison
proved equal actual pre-divergence output history, computed-token boundaries, logical ownership,
and current-token embeddings. Both TP ranks found request-dependent first-different KV layers
4, 21, 24, and 56; earlier layers were exact and later layers differed. Raw logits already differ
and each sampler follows its own argmax, excluding a sampler tie-breaking explanation.

The e73 stock `InputBatch.token_ids_cpu` rows contain pinned-vLLM async `-1` placeholders, so that
field is explicitly non-authoritative metadata. Semantic comparison uses complete run outputs
against the immutable stock reference. The subsequent immutable 8773 per-logical-token run
classified all four requests as `BOOTSTRAP`: every prompt K/V position is bitwise exact on both
TP ranks, and the first difference is the bootstrap token at `prompt_length` in layers 4, 21, 24,
and 56 respectively. The preceding control layers remain exact. The stock endpoint uses ordinary
Target-only async scheduling; the resident endpoint disables async through custom-class resident
execution. This is a causal hypothesis, not proof.

The immutable matched-bootstrap control under commit `efea5c8` subsequently classified
`ASYNC_OFF_MATCHES_STOCK`: ordinary stock Target with async scheduling disabled reproduced the
stock async-ON K/V, raw logits and output for all four divergent requests, not the resident
endpoint. Async scheduling is therefore ruled out as the root cause and no more async, layer,
token-KV, mantissa, class-only, freeze-only or cohort-only micro-diagnostics are authorized. See
[phase4b1-gate3-numerical-diagnostics.md](phase4b1-gate3-numerical-diagnostics.md) and
[phase4b1-gate3-matched-bootstrap.md](phase4b1-gate3-matched-bootstrap.md).

The explicit human engineering decision is now:

- Gate3 structural, semantic-prefix, prompt-KV and logical-KV-ownership correctness: **PASS**.
- Gate3 exact stock trajectory: **NOT ACHIEVED (96/100)**; no tolerance is introduced and the
  four divergent tokens remain visible in immutable artifacts.
- The residual classification is `cross-execution-regime bootstrap numerical divergence`.
- Gate3 numerical qualification: **COMPLETE**; further micro-diagnostics: **DEFERRED**.
- `gate3_exact_stock_equivalence=false` and `phase4b2_progression_permitted=true` coexist by
  design. Historical artifacts retaining `phase4b2_blocked=true` are not rewritten.

## Phase 4B.2: decode-only performance infrastructure

Phase 4B.2 is a measurement layer around the existing resident Target, Serial and Dual-Batch
paths; it is not a second serving implementation. The historical decode-ready manifest boundary
remains unchanged. Performance mode first atomically publishes setup-ready, performs one final
Target-TP barrier and per-rank CUDA synchronization, broadcasts a later monotonic
`performance_measurement_start_ns`, and only then permits the Serial round-zero proposal or Dual
initial enqueue. Target performs no measured Draft proposals. There is no per-token CUDA
synchronization; after generation, one collective RPC synchronizes every Target rank and records
the final completion evidence.

Rank-zero resident callbacks emit explicit semantic token-commit events. Serial proposal commits
use the existing `state_sync_end_ns`, Dual proposal commits use existing `commit_end_ns`, and
proposal-free Target tails use the new explicit commit event. The measurement layer reconstructs
every request as exactly one setup bootstrap plus measured commits and fails if that identity does
not equal the final generated sequence. Per-request latency is final commit minus the shared
boundary. TPOT is `(last_commit-first_commit)/(measured_tokens-1)` and is null for a one-token
request. Makespan is the latest final commit minus the boundary; aggregate throughput is measured
committed tokens divided by makespan.

Standalone mode artifacts cannot produce a speedup. Comparison v2 gates pair and three-mode
speedups on matched work: equal request sets/counts, prompt hashes/counts, bootstrap, maximum
output and measured counts; valid within-mode token accounting and cleanup; equivalent
measurement boundaries, workload/config/model/patch/execution/topology provenance. Exact
generated sequences remain independent diagnostics. Finish/termination differences at the frozen
output length are diagnostic provenance. A matched-work failure suppresses speedup; sequence
differences do not. Timestamped process output is used only to report
post-boundary JIT warnings and `warmup_clean`; it is never a latency authority. The initial
corrected-100 run is functional bring-up, not a final paper workload or result. The Mac coding
agent implemented and CPU-tested this infrastructure without running CUDA. See
[phase4b2-decode-performance-runbook.md](phase4b2-decode-performance-runbook.md).

The first A800 bring-up under `56bd0a5` completed Target and Serial GPU execution. Target derived
successfully; Serial execution returned zero but its derived artifact failed because two
Phase-4B.2 fields were written only to `runtime-manifest.json["phase4a1"]`, not the top-level raw
Serial result. Future Serial runs now publish one canonical evidence block to both locations. A
strict offline compatibility path can reuse only that exact historical execution commit, only
when both raw fields are absent, and only after exact raw/runtime/decode-ready provenance and
two-rank GPU1/GPU2 synchronization validation. It records execution and measurement-code commits
separately and never rewrites the raw GPU artifacts. Target and Dual have no fallback. The
Serial artifact has now recovered successfully: 100 requests, 1487 measured tokens,
50394.65011 ms makespan, 29.50710039169275 tok/s and mean TPOT 2345.39918652 ms;
`warmup_clean=false`, one post-boundary JIT event. Target has 100 requests, 1487 measured tokens,
5813.059543 ms makespan, 255.8033319632212 tok/s and mean TPOT 382.881551485 ms;
`warmup_clean=true`, zero post-boundary JIT events. These are operator-reported GPU results.
Nine Target/Serial trajectories differ after identical bootstrap states; that evidence is retained
without further per-token investigation. The offline comparator verifies per-request counts from
the immutable artifacts before approving the pair. Dual is next, at the same `56bd0a50...`
execution commit; measurement/comparison uses the new commit. PR #4 stays Draft and unmerged.
The resulting claim is preliminary matched-work decode-only bring-up, with no exact-sequence,
output-quality, steady-state or final paper benchmark equivalence claim. Phase 4B.3 sweeps
follow, then Phase 4C Dual-Eager.

After a successful Phase 4B.2 bring-up, Phase 4B.3 adds fixed-output batch/output/context sweeps;
Phase 4C then adds real-GPU Dual-Eager. Arrival-rate, throughput/goodput/SLO and capacity-knee
evaluation remain later work.

## Phase 3.0: GPU readiness and real-trace runner

Phase 3.0 is stacked on the frozen PR #2 head and does not modify its simulator algorithms or
reported results. The default package remains dependency free. An optional GPU extra pins PyTorch
`>=2.7.1,<2.8` and Transformers `>=4.56.1,<4.57`; Transformers is used as a correctness collector
because stable per-candidate draft logits, entropy, and margin are required. It is not the final
serving engine.

The real trace schema separates selector-visible draft features from target-only labels. Completed
request/cycle records are immutable and independently validatable, making interruption/resume
safe. The Phase 3A runner implements deterministic draft-only, target-only, and serial
draft-then-verify collection. A five-GPU coordinator keeps draft TP=1 on GPU 0 and a persistent
target TP=4 worker group on GPUs 1–4. It does not implement Dual-Batch overlap.

The available server has three NVIDIA A800-SXM4-80GB GPUs with NV8 links between every pair, so
the reviewed fallback is 1D2V: Qwen3-0.6B on GPU 0 at TP=1 and Qwen3-32B on GPUs 1–2 at TP=2.
At commit `80d576912028da2d32cd1d8ba5cb593d10a547ae`, a user-run correctness smoke reported Python
3.9.25, PyTorch 2.7.1+cu128, CUDA runtime 12.8, driver 580.126.09, and NCCL 2.26.2. Both model
configs support TP=1/2/4 and correctly reject TP=3 without model surgery. The two-request serial
trace committed 10 accepted candidate tokens plus 6 target-root tokens, and validation reported
`target_only_semantic_equivalence=true`. These observations validate topology, loading, trace,
resume, accounting, and greedy token semantics only; `gpu_measurement=false` is intentional, and
no latency or serving-throughput conclusion follows from this smoke test.

The first user-run primitive smoke exposed a real evidence gap: the TP=2 JSON reported about
34 GB for rank 0 and zero for rank 1 because each process could only observe its own allocator,
while the rank-0 writer never gathered rank-1 state. Two verify runs were also produced by
different commits and therefore are not repeat runs of one implementation. Those v1 files remain
smoke provenance only; their numerical values are not accepted as a latency surface or a
performance result.

Phase 3B.1 replaces that format with a strict v2 schema. Each TP rank now retains its logical and
physical GPU identity, UUID, local parameter count/bytes and device placement, allocated/reserved
memory, forward shapes/checksum, and all CUDA/host samples. Distributed barriers and CUDA
synchronization surround each iteration; rank 0 gathers every rank, and the global sample for an
iteration is the maximum participating-rank latency. Missing ranks, zero model state/memory,
missing samples, device mismatch, invalid statistics, or non-max aggregation fail validation.
Only rank 0 atomically publishes the report.

The hardened statistics retain every sample and report mean, standard deviation, CV, min,
P50/P90/P95/P99/max and flagged outliers, with defaults of five warmups and thirty measured
iterations. Before/after hardware snapshots record observed clocks, temperature, power, P-state,
ECC, memory, PCIe, topology and peer-access state without attempting clock control. Repeated-run
comparison requires an identical commit, config checksum, model revisions, GPU model, TP layout,
backend and operation semantics; raw samples are never pooled across incompatible runs.

All v2 measurements are explicitly `backend=hf_correctness`, `serving_engine=false`,
`kv_cache_reuse=false`, `packed_tree_verification=false`, and
`simulator_latency_surface_compatible=false`. Draft remains serial greedy full-context replay;
verify remains serial full-context replay for `B_cand+1` target forwards. Selector timing still
labels the existing kernel as synthetic `torch.topk`; a dependency-free five-stage selector
interface exists without fake timings. Transfer now covers both draft↔target-leader directions,
the target-leader→TP-peer path when present, and payloads from 4 KiB to 256 MiB, but remains a bare
device-copy primitive rather than complete Draft→Verify transport.

No NVIDIA GPU was available to the Mac agent that implemented Phase 3B.1. The user subsequently
completed the same-commit three-run A800 validation: TP=2 verify run-to-run CV was about
0.9%–1.3% with maximum reported variation 3.04%; draft variation was about 6.7%–8.0%; the
synthetic top-k primitive was about 0.03 ms; large-payload peer copies were stable; and every v2
validation passed. These are repeatability observations for `hf_correctness` primitives, not a
serving latency surface or a performance claim.

## Phase 3C.1–3C.3: real selector diagnosis

Phase 3C.1 adds an isolated, resumable pipeline without changing the simulator. It builds a fixed
100-request public-text pilot with a 60/20/20 code/chat/summarization mixture, binds the first 100
chronological Mooncake arrivals, and stores token IDs and lengths from the actual configured Qwen3
tokenizer. Missing datasets are fatal and no prompts are synthesized as fallback. The 40/50/150 ms
classes remain metadata only.

The draft stage uses Qwen3-0.6B TP=1 to build one shared real 4× forest per request; 1× and 2× are
strict prefix-closed subsets. Node counts 16/32/64 and verification budget 4 are derived from the
frozen Phase-2 width/depth/speculative-budget configuration. The target stage uses Qwen3-32B TP=2
to generate one immutable greedy continuation per request, independent of ratio. Both remain
full-context Transformers correctness paths with `kv_cache_reuse=false`.

Label join keeps draft runtime features and target-only labels in separate objects. Five
target-blind selectors and a within-request target oracle replay the same forest, target and fixed
budget. Reports cover pool/selected target-path coverage, candidate efficiency, oracle regret,
depth/probability/entropy calibration, sibling hits and pool robustness. Stable request-level
train/validation/test splits prevent candidate-node leakage.

The user completed and validated the 60/20/20 real Qwen3 pilot. At budget four,
Residual-Probability accepted 1.92 tokens/proposal at all three pools; the within-request oracle
accepted 2.23/2.30/2.30 at 1x/2x/4x. These are selector-signal observations, not latency or system
performance. The old `target_path_pool_coverage` fell as the nested pool expanded because it used
each pool's own realized maximum depth as its denominator. It was not density and did not have a
fixed recall denominator.

Phase 3C.2 preserves that legacy field and adds fixed-denominator monotonic target-path recall,
target-node density, selected precision/recall, K=4/8/16 horizon coverage, first-missing depth,
16/16/32-node shell decomposition and selection-set stability. Per-task and overall reports now
use request-level stratified bootstrap intervals and paired oracle/Residual-Probability
comparisons. Headroom separates generator coverage, selector regret and budget limits; zero oracle
expansion gain is explicitly not identifiable.

The prompt audit also found that Phase 3C.1 ShareGPT chat prompts were raw first-user text without
the Qwen chat template. Those 20 old chat traces remain legacy diagnostics and cannot be pooled
with corrected data. The v2 builder applies the Qwen tokenizer chat template with
`enable_thinking=false`, retains native HumanEval completion prefixes and the explicit
CNN/DailyMail summarization instruction, and records deidentified rendering/tokenizer metadata.

The corrected 20-request (12/4/4) multi-round mode freezes each at-most-16-token target once, creates one
shared forest at every target-prefix position, and replays every selector sequentially over those
same snapshots. Immutable checkpoint/resume and final-token equality are enforced. The Mac agent
implemented and tested this path without running a GPU model. The user then completed the
corrected-20 server run: Residual-Probability stayed at 2.407 accepted/proposal across pools,
Entropy-Margin reached 2.421, and the Oracle rose from 2.680 at 1x to 2.877 at 2x with no further
4x gain. These are token-efficiency observations only.

Phase 3C.3 promotes the corrected run to 100 requests (60/20/20), adds a strict cross-artifact
validator, request-stratified bootstrap intervals and paired deltas, and restricts formal analysis
to 1x/2x. It decomposes the 2x shell into generator coverage, budget/prefix reachability and
ranking failure. A diagnostic linear `learned-shell-ranker` uses fixed runtime draft features,
70/15/15 task-stratified request splits, immutable model/replay provenance and a predeclared
held-out A/B gate. Target labels are available only during offline training; inference rejects
labeled nodes. The Mac agent implemented and CPU-tested this path but did not run the corrected
100-request Qwen trace. That server run and artifact review are the next gate.

Packed-tree, vLLM/SGLang, Dual-Batch, Eager, SLO evaluation and simulator calibration remain out of
scope. See [phase3-real-trace.md](phase3-real-trace.md) for definitions and boundaries.

## Phase 1.5: residual selection

Four residual policies freeze the exact same-state Dual-Batch request set, budgets, path nodes,
candidate forest, roof, and deterministic target outcome, then differ only in residual selection:
request round-robin, path probability, current SLO-aware two-stage, or feasibility-gated two-stage.
All runs report zero base-preservation violations, and residual roof utilization is aligned within
0.15 percentage points at 2.75×, 0.06 points at 3.0×, and 0.05 points at 3.25×.

The decisive result is Probability versus Shaping. At 3.0×, probability raises goodput
2146.6→2452.3 tokens/s, raises attainment 0.728→0.816, and lowers mean queueing 4.30→2.43 seconds.
At 3.25×, it raises goodput 1414.4→1587.4, raises attainment 0.457→0.502, and lowers queueing
19.17→15.05 seconds. The two are effectively tied below the knee at 2.75×.

This rejects the current SLO-stage formula as a forward mechanism: filling idle roof helps, while
global path-probability selection is more efficient than the SLO-weighted residual stage under
pressure. The rejected policies remain only for provenance and diagnosis. The next mechanism gate
is candidate selection or Overdraft-and-Prune, not further tuning of these SLO weights.

The scalar candidate roof is explicitly not a GPU capacity claim. The proxy charges both request
root positions and candidate positions, while GPU calibration must measure the joint surface
`T_verify(B_req, B_cand, C)`. Full results and definitions are in
[phase1.5-residual-selection.md](phase1.5-residual-selection.md).

## Phase 2: oracle headroom

Phase 2 leaves every existing policy unchanged and exposes two separate diagnostic commands.
`phase2-replay` performs the primary same-snapshot causal comparison, while `phase2-simulate`
reports secondary end-to-end, fully-hidden-search upper bounds. Neither command is part of the
normal policy order.

The historical target oracle samples its next target child from the tree passed to verification,
so changing pool width would also change ground truth. The isolated Phase-2 oracle instead freezes
the historical target trajectory on the immutable 1× tree, then adds deterministic, prefix-closed
branches. This makes `A_1×` strictly comparable with Residual-Probability and keeps target truth
constant across ratios and selectors. It also means the canonical target is always present in the
1× pool: this proxy can measure selector, cross-request residual-allocation, and full-tree gaps,
but it cannot identify real missing-target or better-drafter pool-coverage headroom.

The primary replay uses 10,000 deterministic, stratified snapshots per load and keeps
`B_verify`, request roots, the target outcome, and the proxy verification surface fixed while
expanding metadata-only `B_search` to 1×/2×/4×/8×. Variants A/B/C/D isolate the current
target-blind selector, within-request target oracle, global residual oracle, and full-tree oracle
ceiling respectively. All are marked diagnostic-only; B/C/D explicitly leak target outcomes and
all ratios assume search is fully hidden. Detailed definitions, sampling coverage, results, and
decision boundaries are in [phase2-oracle-headroom.md](phase2-oracle-headroom.md).

The completion audit reused all valid interrupted-run artifacts and executed only nine missing
3.25× cells. Final coverage is 3/3 common replays with exactly 10,000 corrected-queue snapshots
each, 9/9 references, and 48/48 end-to-end oracle cells. A_1× reproduces Phase-1.5
Residual-Probability at all three loads with exact integer equality and floating-point absolute
tolerance `1e-12`.

On common snapshots at 8×, B−A adds 16.36/17.07/17.55 committed candidates per cycle across
2.75×/3.0×/3.25×; C−B adds 1.69/3.18/4.02; D−C adds 2.49/2.83/3.01. All
360,000 ordered dominance checks pass. A loses 9.26/9.34/9.33 candidates per cycle versus A_1×
because added branches are distractors around a target already fully covered by 1×.

End to end, A's 1×→8× goodput falls 2567.0→1761.1, 2452.3→1114.0, and
1587.4→742.1 tokens/s. B largely removes the selector loss; at 3.25× it reaches
2952.6/.9412 goodput/attainment at 4×. C reaches 3030.2/.9963 and D 3033.7/.9988, with C/D
core outcomes unchanged by ratio because their oracle already finds the 1× target. These are
fully-hidden-search system ceilings, not deployable or GPU-measured performance.

## Phase 1: shaping causal diagnosis

Three opt-in diagnostics were added without changing the default algorithms:
`shaping-feasible`, `shaping-residual`, and `shaping-feasible-residual`. One-cycle feasibility uses
the frozen total progress gap, counts one future root exactly once, and does not label a request
globally unsalvageable. Residual variants freeze the same-state Dual-Batch request set, per-request
budget, selected candidate path, and root opportunities before shaping otherwise-unused roof.

The scoped full-R3 proxy results are:

| Load | Policy | Goodput tok/s | Attainment | Queue s | P90 TPOT ms | Total progress/cycle | Infeasible opportunities | Stage-1 → infeasible |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2.75× | Dual-Batch | 2557.9 | 0.990 | 0.36 | 28.4 | 63.61 | 0.055 | 0 |
| 2.75× | shaping | 2566.5 | 0.996 | 0.25 | 26.3 | 69.74 | 0.035 | 336,357 (85.7% of stage 1) |
| 2.75× | feasible | 2565.7 | 0.996 | 0.25 | 26.3 | 69.74 | 0.035 | 0 |
| 2.75× | residual | 2567.5 | 0.998 | 0.19 | 25.0 | 69.62 | 0.027 | 128,113 (92.0%) |
| 2.75× | feasible + residual | 2566.8 | 0.997 | 0.19 | 24.9 | 69.61 | 0.027 | 0 |
| 3.0× | Dual-Batch | 1968.3 | 0.673 | 5.91 | 119.9 | 69.93 | 0.409 | 0 |
| 3.0× | shaping | 1746.8 | 0.632 | 11.50 | 228.3 | 74.60 | 0.425 | 3,597,597 (99.0%) |
| 3.0× | feasible | 1750.3 | 0.633 | 11.28 | 224.3 | 74.64 | 0.424 | 0 |
| 3.0× | residual | 2146.6 | 0.728 | 4.30 | 93.6 | 76.70 | 0.353 | 1,177,804 (99.4%) |
| 3.0× | feasible + residual | 2144.7 | 0.727 | 4.31 | 93.6 | 76.71 | 0.354 | 0 |
| 3.25× | Dual-Batch | 1291.6 | 0.421 | 22.70 | 373.5 | 73.24 | 0.640 | 0 |
| 3.25× | shaping | 1117.4 | 0.394 | 37.43 | 612.5 | 76.81 | 0.655 | 5,421,587 (99.6%) |
| 3.25× | feasible | 1116.2 | 0.394 | 37.29 | 609.8 | 76.87 | 0.654 | 0 |
| 3.25× | residual | 1414.4 | 0.457 | 19.17 | 321.8 | 80.57 | 0.609 | 1,761,091 (99.8%) |
| 3.25× | feasible + residual | 1415.0 | 0.457 | 19.14 | 321.6 | 80.57 | 0.608 | 0 |

The feasible-only guard removes essentially all stage-1 allocation to one-cycle-infeasible
requests but leaves goodput almost unchanged. Preserving Dual-Batch breadth/base opportunities is
the materially positive intervention at 3.0× and 3.25×. Combining the guards adds no material gain
over residual preservation alone. This supports breadth/root opportunity cost as the dominant
modeled cause at the proxy knee; it does not validate a new default or a GPU performance claim.

All residual runs report zero base-preservation violations. Their base trees come from the exact
Dual-Batch allocator and sequence-path materializer on each variant's same cycle state; residual
nodes are prefix closed and never evict base work. Because earlier scheduling choices can make
independent runs reach different later states, the invariant is a same-state counterfactual, not
a claim that cycle IDs across divergent runs always contain identical active request sets.

Detailed summaries remain outside Git under
`SpecRhythm-data/results/simulator-semantics-v0.2/phase1-shaping-diagnosis/`.
The full table, class-good-token breakdown, progress accounting, and utilization-denominator audit
are in [phase1-shaping-diagnosis.md](phase1-shaping-diagnosis.md).

## Superseded flat-proxy evidence

- The superseded flat revision had 80 passing tests. The current tree-aware/Phase-2 revision has
  115 passing tests and is Ruff clean, including prefix closure, Figure 5 selection, width-1
  degeneration, path-dependent eager, goodput denominator, proposal/token/tree-node conservation,
  and both Phase-2 CLI commands. The recovery baseline had 114 tests; one integration test was
  added to exercise `phase2-replay` and `phase2-simulate` through the CLI.
- Full Mooncake R3 proxy replay validates at 12,031 requests for 1×, 2×, 4×, and 8×
  time scales, corresponding to 3.401, 6.802, 13.605, and 27.210 requests/s.
- 1× and 2× remain mostly below the configured proxy capacity; 4× and 8× show large queueing
  delay and SLO violations, confirming that pending time is included.
- These results used a **legacy flat-sequence shaping proxy**. They diagnose that proxy only and
  cannot compare AdaServe's or SpecRhythm's tree-aware algorithms.
- Eager compute-waste ratios are high (roughly 0.76–0.88 across the reported proxy sweeps).
  Confidence, admission threshold, latency surfaces, and roof calibration remain open gates.

Generated workloads, validation reports, and comparison JSON remain in the external data tree
and are not committed.

## Tree-aware capacity-knee evidence

The full R3 proxy sweep covers 2.0×, 2.25×, 2.5×, 2.75×, 3.0×, 3.25×, 3.5×, 3.75×, and 4.0×.
The proxy knee is between 2.75× and 3.0×: shaping attainment falls from 0.996 to 0.632 and mean
queueing rises from 0.25 s to 11.50 s. At 3.0×, tree-aware shaping reaches 1746.8 good tokens/s
versus 1968.3 for Dual-Batch; SpecRhythm reaches 1895.8 versus 2645.2 for Dual-Eager. These are
negative proxy results for the frozen tree-aware control plane, not evidence about GPU execution.

At 3.0×, Dual-Batch to shaping adds 2,050,577/522,522 candidate nodes to the 40/50 ms classes,
but realized accepted-node gains are only 99,604/29,814. It also adds 148,942 nodes to the 150 ms
class while realized progress falls by 24,880. There are 3,814 tight requests that receive more
candidates but still miss SLO, and 385 previously attained 150 ms requests lose attainment.
Goodput falls through both numerator (-216,670 SLO-good tokens) and denominator (+29.68 s
makespan). The allocator transfers budget, but proxy expected progress does not translate into
sufficient realized progress.

At the same 3.0× point, flat→tree goodput changes are 498.3→485.3 for AdaServe proxy→tree,
1747.6→1746.8 for shaping, and 1791.9→1895.8 for SpecRhythm with eager. Tree semantics therefore
change both allocation and eager outcomes; the legacy flat result is retained for provenance but
cannot substitute for the tree-aware control plane.

The residual-score ablation at 3.0× gives identical shaping output for path probability and
urgency × path probability under this workload/configuration (1746.8 good tokens/s). The frozen
default remains urgency × path probability; equality here is a diagnostic result, not a reason to
change it.

The complete eager grid at 3.0× spans budgets 1/2/4 and dependency thresholds 0.1/0.2/0.3/0.5;
no cell is filtered. At threshold 0.1, Dual-Eager goodput is 2593.0/2645.2/2651.1 for budgets
1/2/4, while SpecRhythm is 2208.5/1895.8/1827.5. Raising the threshold to 0.5 removes all
Dual-Eager admissions and nearly all SpecRhythm admissions. In the detailed final default run,
SpecRhythm's admitted dependency paths have slightly higher mean probability than Dual-Eager
(0.273 versus 0.258), but consume more eager tokens (910,391 versus 636,371). Counterfactual
same-cycle allocation attributes 846,688 displaced normal nodes to SpecRhythm eager versus only
96 to Dual-Eager. This rejects the "lower-probability admission" explanation and instead
identifies admission volume and normal-budget displacement as the leading mechanism to inspect.
The default is not changed after observing results. These are control-plane sensitivity findings,
not GPU performance claims.

## Current simulator contract

The comparison order includes AR, Serial SD, retained flat-proxy diagnostics, tree-aware
AdaServe, Dual-Batch, Dual-Batch + Rolling Eager, tree-aware shaping, and SpecRhythm. Serial SD
and Dual-Batch share allocation and differ only in exposed cycle latency. Tree-aware AdaServe implements the paper's two-stage
node selection with proxy trees/latencies; it is not the complete AdaServe system. Exact formulas
and root accounting are frozen in [tree-aware-design.md](tree-aware-design.md).

Serial SD and Dual-Batch produce identical allocations for the same fixed logical batch. Their
different cycle duration may alter later trace admissions and queueing, which is an intended
system-level consequence of overlap.

Every request reports queueing, service, and end-to-end decode latency. Every proposal is tracked
to exactly one terminal state, with proposal/token promotion, invalidation, EOS discard, and
compute-waste ratios.

## Open work and known limitations

- `input_tokens` is stored but not used by the current latency model.
- Context-dependent draft/verify latency is not implemented.
- `D(B,K,C)`, `V(B,K,C)`, acceptance, confidence, and candidate roof are proxy inputs until GPU
  calibration.
- R3 proxy lengths are sampled and are not HumanEval, Alpaca, or CNN/DailyMail payloads.
- Phase 4B.1 has a decode-only asynchronous linear Dual runner, complete Gate1/Gate2 A800
  correctness evidence, and completed Gate3 numerical qualification. Phase 4B.2 performance
  infrastructure is implemented but has no A800 result yet. It has no SGLang, packed-tree
  verification, Dual-Eager, KVConnector, arrival scheduling, load/SLO or final-paper evaluation.
  The persistent Draft HF adapter is still a narrow serving prototype.
- No current result may be cited as evidence of real GPU speedup or full AdaServe/SpecRhythm
  reproduction.

## 2026-09-16 — A100 K3 B64 scale implementation

Draft PR#5 stays Draft; ordinary continuation from5b50529/6e07617, no other PR or
historical result changes. Explicit `k3-b64-v1`: serial-k3/serial-eager-k3 A64 Target64;
pingpong-k3/pingpong-eager-k3 A32/B32 Target32; Draft physical ceiling64 for all.
B16 entry/default remains. No scheduling, READY, K3, sampling or log strategy change.

Local scope: geometry propagation, actual-rank capacity and query checks,128-opportunity
warmup,64-request complete-output fixture, single-native-forward geometry qualification,
normalized opportunity metrics/four ratios, and explicit bounded B64 offline packaging.
[B64 runbook](k3-b64-runbook.md) describes the fixed four-point operator flow.
Execution SHA `f6f67aa1e1d7aea2a81665ec628d0ae857c148ee`; the separate delivery commit
adds `scripts/run_k3_b64_pinned.sh`, fixed to this execution SHA. An isolated archive
of the committed source passed the eight-role static check and contained the actual
B64 runner. Server capacity,
correctness, cleanup, native overlap and performance remain PENDING.

Implementation validation: full pytest2845 passed/3 skipped; Python3.9/3.12 related
suites369 passed each plus157 final delta tests each; Ruff, three-version compileall,
Python3.9 AST and23 Bash scripts passed. Final entry/runner/local-delivery regressions:
46 passed; fixed foreground entry tests also passed6 each on Python3.9/3.12.

Implementation push CI35000694509 has a failed Python3.11 legacy PingPong scan
drain test: `diagnostic owner settlement deadline expired`. Its five-second test
deadline expired while awaiting an owner receipt for a staged resident. The test,
fixed_drain and fixed_settle are unchanged by B64; the same commit's PR CI35000697681
passed that job. This does not identify why the first run exhausted its deadline;
no retry, assertion relaxation or budget increase was made. Other jobs were still
running at this documentation snapshot. Final delivery CI is reported separately;
neither local passes nor the other job's success erase this recorded failure.
GPU evidence is still PENDING.

## 2026-09-16 — B64 performance exploration, strict diagnostics retained

Continue from3ac3752/f6f67aa on Draft PR#5 without changing other PRs or historical
results. Default B64 policy now explicitly selects `performance-exploration`:
static → four capacities → four warmup/performance points → native geometry/runtime
and evidence gates → comparison/single archive. Target-only and independent complete
output tests are not launched. `strict-output` remains an opt-in path; B16 defaults
and algorithm/execution/storage/budget configuration are unchanged.

Full output equivalence is NOT_RUN, not PASS. The historical first B64 run had
54/64 exact outputs in each Serial mode with the same10 differences, and64/64 in each
PingPong mode; its strict failure and unstarted performance points remain unchanged.
Native64/32 proof now comes from the current exploration warmup/runtime. Required
capacity/device/KV/protocol/cleanup/measurement/trace checks remain strict. Aggregate
strict mismatch reports now identify actual modes/IDs under comparison, preserving
individual process exit codes and unique single-package delivery.

Implementation/entry validation results and CI are recorded in the delivery update.
New GPU performance and geometry/cleanup evidence are PENDING; no server was contacted.

Local implementation validation: policy/entry/report delta65 passed; related Python3.9
and3.12 suites270 passed each. Full suite2873 passed/3 skipped/4 failed: three existing
natural-teardown shell cases exceeded15s and one legacy prepost capacity/failure-shell
case exceeded20s. This is not a fully green local suite; the causes of those timing
failures are not claimed resolved. Earlier ENOSPC and temporary artifact-retention
collision are documented separately in k3-validation. Ruff, compileall, Python3.9
syntax and Bash checks passed. No assertion or timeout changes.

Execution commit `899b54a6b58c0ea07046582c5a17934f630ac040` contains the complete
policy, native geometry gates, strict attribution repair and regressions. The follow-up
fixed-entry commit pins exactly this version. Both keep PR#5 Draft; no other PR changes.

Final entry checks46 passed, then24 default/strict forwarding and local delivery
checks passed on each of Python3.9/3.12. Execution files were byte-verified against
the fixed implementation SHA and all eight static role/mode cases passed. Baseline
and isolated current checks did not reproduce the four full-suite timeouts; their
original causes remain unresolved. Implementation CI35059823704 (push) and35059827533
(PR) passed the Python3.11 and pinned-source contract jobs; full Python3.9/3.12 jobs
were still running at the documentation snapshot. No GPU test was run locally.

### 2026-09-16 — B64 diagnostic I/O / unified dispatch experiment

Draft PR #5 remains Draft. Opt-in bounded deferred diagnostics postpone post-run
JSONL persistence and repeated plugin-report construction under the unchanged
absolute drain deadline. Both `io-only` and `unified` configurations use the same
recording and evidence contracts. Existing mixed correction/ordinary/lookahead
materialize was verified; no claim that all historical recovery was exclusive.
Unified mode validates runnable-set completeness and publishes valid complete K3
promotions before unrelated recovery calls, returning to the owner queue. Serial's
idle gate remains. Historical missed-batching count is unknown without inventory.

Dedicated two-repetition foreground entry (forward then reverse mode order), one
flattened archive, performance-exploration / output NOT_RUN. B16 defaults, hardware,
geometry, protocol and historical results unchanged. See
[validation](validation/k3-deferred-dispatch.md) and [B64 runbook](k3-b64-runbook.md).
Local test/CI outcomes and final fixed SHA are recorded in the delivery update below.
No AutoDL/A100 connection or GPU run performed; new GPU results are PENDING.

### 2026-09-18 — lean B128 dispatch cycle accounting (pending server)

Preserved `bb79d4f` after the verified `4ff1170` lean reference. Observation commit
`b459781964afca88daea7e3331e91855b1aa06fe` adds matched feedback cycles and bounded
control/worker/owner spans. First-repeat ordinary PingPong explains its apparent
negative step residual exactly: claim begins 97.527 ms before complete-step, while
feedback RPC precedes step end by 78.825 ms. The same 52 samples close to 394.764 ms.
Eager receive→dequeue is partly occupied by a recorded Draft physical host call,
with the remaining duration explicitly unaccounted.

A separate switch changes only online control serialization from fragmented
`json.dump` writes to byte-equivalent one-shot `json.dumps` + write + atomic replace.
No scheduling order, READY publication, live KV check, deadline or output protocol
changes. Same-SHA lean-reference/lean-dispatch-opt controls keep all eight windows
and single-bundle delivery. See `docs/validation/k3-dispatch-cycle.md` for boundaries,
evidence limitations and validation. New GPU performance/overlap remain PENDING;
full output equivalence remains NOT_RUN; historical output differences remain open.


Dispatch delivery: execution `c1fca49f3f846ef94f0759b585fddaf091d78b62`, fixed entry
`9feb51a4d36e4acc0f682ea006a6842d327a69d4`. Observation and encoding optimization
remain separate commits/switches; both same-SHA lean configurations run8 windows.
Related suites passed90 (Python3.9) and94 (Python3.12); fixed-entry/profile13 passed
on3.12, entry/cycle15 on3.9. Final local full run3089 passed/3 failed/3 skipped;
three unchanged legacy20s/5s/15s subprocess harness timeouts remain unexplained.
First full run3061 passed/10 failed/3 skipped is retained separately in
`docs/validation/k3-dispatch-local-checks.json`. No timeout/assertion was relaxed.
Ruff, compileall3.9/3.12, Python3.9 syntax and Bash checks passed. CPU interleaving
and report checks do not certify native GPU performance or output equivalence.
Fixed entry ordinary push preserves Draft PR#5 and all prior branch commits.

CI snapshot for entry9feb51a, run35260463228: Python3.11 contract and pinned-vLLM
source contract PASS; Python3.9/3.12 full jobs RUNNING. Their completion is not
claimed. The documentation follow-up may have its own pending CI run. New GPU
capacity/cleanup/overlap/performance PENDING; output equivalence NOT_RUN.

### 2026-09-18 — B128 joined-report size and console repair

Verified the lean-reference465 archive:252 logical paths/230 unique payloads,
all size/SHA256 checks pass. Four capacity points passed. Forward Serial,
Serial-eager and ordinary PingPong performance execution/measurement/cleanup
passed; ordinary PingPong then failed diagnostic report publication at8MiB.
The66-step analysis body is8416255 bytes before source metadata. Eager PingPong
and reverse repeat did not start. Original diagnostic failure/code1 retained;
no GPU execution failure or new output-equivalence claim.

New post-run summary/table separation preserves every joined row with bounded
8MiB shards (at most8), count/order/hash verification and single-bundle export.
The real archived producer chain now writes a513123-byte summary plus7957178-byte
detail and requalifies COMPLETE in a separate offline derivation. This does not
rewrite the original status. K3 terminal output is a small summary, with all
artifacts retained. Publication failure status and primary error context survive.
No scheduling, GPU, protocol, I/O buffering or deadline change. New GPU tests
remain PENDING; output equivalence remains NOT_RUN. See validation/k3-report-size-fix.md.

Delivery implementation `24a5042d8bce503699bb040f293541f853e1e6b1` and fixed entry
`a7d34cdbe5e65606a14eab3cec9b3704a8f53d74` have been ordinary-pushed to Draft PR#5.
Both lean-reference and lean-dispatch-opt select that identical repaired execution.
The new fixed launcher passed all seven CPU tests on both Python3.9 and3.12.
Python3.9 affected regressions:31 passed. Ruff, compileall3.9/3.12, Python3.9
syntax parsing and20 Bash checks passed. Related3.12 run:65 passed/two unchanged
20s subprocess harness timeouts, retained separately from new report tests.

Prior HEAD bbd0481 CI is now known FAILED (run35261230780):3.12 had one old
owner settlement deadline failure;3.9 cancelled; both contract jobs passed.
This predates the report repair, is not this server's diagnostic-size cause,
and remains unresolved. Current fixed entry CI run35312914255 has both contract
jobs PASS and3.9/3.12 full jobs running at this snapshot; no full-CI success claim.

Full local3.12 result:3105 passed/17 failed/3 skipped. Nine new console fixture
failures were traced to an inherited full numerical plan and corrected by test
input isolation; production conflict checks remain. Explicitly contaminated
focused3.12 run:55 passed/2 source skips;3.9 report/console:24 passed. The other
eight full-suite timing/deadline failures remain unresolved, with IDs preserved
in `docs/validation/k3-report-local-checks.json`; no second full-suite PASS claim.
The fixture/documentation follow-up changes no production source, so the fixed
execution and launcher above remain valid. No new GPU run has been performed.

CI update: entry a7d34cd run35312914255 finished FAILED:3.12=3082 passed/9 failed/
36 skipped; all nine failures are the reproduced console-fixture profile conflict,
fixed in b09827f. Both contract jobs PASS;3.9 cancelled. New follow-up CI is
PENDING; no CI-success claim. The server execution code is unchanged by b09827f.
