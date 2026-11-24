# 用官方 vLLM 运行远程压测

容器内可用统一入口 `vllm-suite` 读取当前目录的 `.vllm.conf`，同一次执行共用一个 UUID 结果目录。[运行示例](../README.md#在已启动的容器内运行) · [全参数配置](examples/vllm.full.conf)

所有新压测都使用 vLLM 0.11.0 的官方 serving benchmark。项目只决定每次调用的参数和先后顺序，记录实际请求数，并保存官方终端报告和 JSON。单次 benchmark 和四阶段测试共用同一个镜像。

需要在同一个容器内先完成压测再启动精度评测时，执行镜像内的 `perf-then-eval`。[镜像构建与运行](../README.md#进入同一个容器先跑压测再跑精度)

## 安装与挂载

在 Docker 宿主机的 Bash 中运行。NAS 只读存放与服务匹配的 tokenizer 和可选数据集。tokenizer 路径指向直接包含 ModelScope 下载产物的目录。输出使用宿主机本地磁盘。

```bash
docker build --platform linux/amd64 -t vllm-suite .
export NAS_ROOT=/mnt/nas/vllm-suite
export RUN_ROOT=/var/tmp/vllm-suite-perf
mkdir -p "$RUN_ROOT/config" "$RUN_ROOT/outputs"
cp perf.example.json "$RUN_ROOT/config/perf.json"
mounts=(--mount "type=bind,src=$NAS_ROOT,dst=/nas,readonly"
        --mount "type=bind,src=$RUN_ROOT/config,dst=/config,readonly"
        --mount "type=bind,src=$RUN_ROOT/outputs,dst=/outputs")
```

在 `perf.json` 的 `vllm_bench_serve` 中填写服务暴露的 `model`、容器内的本地 `tokenizer` 和 `base_url`。例如 `base_url` 为 `http://target-server:8000`，聊天接口的 `backend` 为 `openai-chat`，`endpoint` 为 `/v1/chat/completions`。服务忽略 Authorization 时可使用 `export OPENAI_API_KEY=unused`；有鉴权时通过该环境变量传真实密钥。

离线部署时，在构建机执行 `docker save -o vllm-suite.tar vllm-suite`，将 tar 复制到目标机后执行 `docker load -i vllm-suite.tar`。

## 配置与发起

公共官方参数集中在顶层 `vllm_bench_serve`。每个 `profiles[]` 有独立名字和可选的同名参数块，可覆盖数据集、种子、采样参数和长度。示例第一档随机输入 4096 tokens，官方输出长度参数 1536。实际输出受服务的 EOS 行为影响。真实数据按官方数据源的采样和长度规则配置，例如 ShareGPT 使用 `dataset_name: "sharegpt"` 和容器内的 `dataset_path`，同时移除随机数据专用参数。

```bash
export OPENAI_API_KEY=unused
docker run --rm --network none "${mounts[@]}" vllm-suite perf-suite \
  --config /config/perf.json --output /outputs --validate-only

docker run --rm -e OPENAI_API_KEY -e HF_HUB_OFFLINE=1 "${mounts[@]}" vllm-suite perf-suite \
  --config /config/perf.json --output /outputs --stage all
```

`--validate-only` 只做静态配置检查，包含各档官方参数、编排约束以及 `metadata` 的 `KEY=VALUE` 格式。各档合并公共参数后使用同一预检。官方参数 `burstiness` 必须大于零，任一档配置无效都会在首次发流前报错。随机多模态数据集 `random-mm` 需要 `openai-chat` 后端。ShareGPT 需要配置本地 `dataset_path`，分阶段模式要求关闭 `trust_remote_code`。镜像构建使用同一检查。正式运行才固定输入、加载 tokenizer、抽取样本和连接 API。在已启动的 vllm-suite 容器内可直接运行 `perf-suite --config /config/perf.json --output /outputs --stage all`。

| 阶段 | 每个长度档的官方调用 |
|---|---|
| `connectivity` | 1 次，1 个请求，并发 1。所有长度档联通成功后才进入下个阶段。 |
| `scan` | 对 `concurrency` 中每个 C 调用一次，发送 C × `requests_per_concurrency` 个请求，倍率默认 10。 |
| `overload` | 对 `overload_concurrency` 中每个 C 连跑三轮；每轮一次调用，恰好 C 个请求，并发上限 C。单独运行时直接使用过载列表。 |
| `soak` | 先运行本次 scan，然后在选出的并发下做一次官方 benchmark，发 `soak_requests` 个请求，默认 1000。报告记录实际耗时。 |

顺序为所有档的 connectivity → 所有档的 scan → 所有档的 overload → 所有档的 soak；单阶段运行只执行指定阶段及其前置阶段。正式运行在首次探测前检查全部配置中的本地绝对输入路径。

每次调用均设置 `request_rate=inf`、`ready_check_timeout_sec=0`，并关闭 profiler。项目等全部请求结束、官方结果与阶段汇总落盘后，才启动下一次调用。短采样会在该次发流前报错。编排拥有的参数（请求数、并发、速率、就绪检查、ramp-up 和 profile）不要放进官方参数块。每次调用使用配置的 seed。服务端实际能同时执行的请求数取决于其调度。

scan 以官方 P95 TTFT、TPOT 满足 `slo` 且请求全部成功的档位为候选，在输出吞吐达到候选峰值 95% 的档位中选最小并发。soak 同样检查这两个 SLO。没有候选时跳过 soak，返回非零。成功的单 token 响应照常计入吞吐。阶段汇总的 `requests.tpot_samples` 统计官方输出长度大于 1 的样本数，零样本时该档无法通过 TPOT 验收。使用 `prefix_repetition` 时联通探测只用一个前缀，正式计数必须为所配置前缀数的正整数倍，以保证官方采样恰好返回计划请求数。阶段模式在官方指标选项中补充 P95、TTFT、TPOT，适用于文本生成后端。

联通阶段只检查请求是否完成。若官方报告给出微小负 TPOT，保留原值。scan 和 soak 的 `acceptance` 记录具体原因，例如 `unusable_p95_tpot_ms`、`no_tpot_samples` 或 `slo_exceeded`。该 scan 档退出候选，soak 则记录 `failed_acceptance`，请求成功数仍按官方结果统计。

## 输入如何固定

默认将本地 tokenizer 和数据文件复制到时间戳目录的 `inputs/`，保留相对布局，跳过常见模型权重。复制前先为所有 profile 绑定输入版本。同一个配置路径只解析一次；即使 NAS 上的 latest 符号链接随后切换，各档仍复用同一份资源。根 setup.json 的 inputs.resolved_paths 可核对这些绑定。程序核对复制前、复制后和副本的文件清单及 SHA-256，再发布副本。请预留本地磁盘空间。随机数据只固定 tokenizer、配置和 seed，每轮仍由官方生成请求。运行期间可读取各轮自己的 `summary.json`。

大型输入可以使用顶层可选字段 `"immutable_snapshots": ["/nas/.snapshot/2026-11-24"]`。仅填写由存储系统保证不可变、在整个运行期间可访问的版本路径；普通 NAS 的只读挂载仍使用默认复制方式。零复制模式会记录已解析的快照路径，版本保证由存储端提供。需要跨文件事务一致性的输入应使用存储快照。

分阶段运行要求本地 tokenizer；配置里的外部 tokenizer 文件覆盖和自定义 tokenizer 代码需用单次模式。ShareGPT 的外部图片或视频必须是目标服务也能访问的不可变快照绝对路径。HF 文本数据使用本地自包含的数据目录和官方 `hf_name`，支持 InstructCoder、MTBench、AIMO、NextEditPrediction、MLPerf 适配器。需要其他 HF 适配器或远程资源时使用单次官方模式。

## 报告、单次运行与迁移

每次运行创建独立的时间戳目录。完成后直接打开根目录 `vllm-report.txt` 即可依次阅读各轮原文，`vllm-reports.json` 提供对应的文件索引。根目录 `summary.json` 记录每档的阶段、选定并发、请求数、状态和报告路径。例：`4k/overload-c192-r2/vllm-report.txt` 是第二轮官方终端报告，旁边的 `vllm-result.json` 是官方返回的指标，`setup.json` 记录生效参数。启用 `save_detailed` 会在原生 JSON 中保留逐请求明细。中断返回 130，已完成轮次仍可读取。

每个 profile 最后一档 scan 结束后，程序将本次运行标识、profile、scan 序号和选定并发写入 `<profile>/selection.json`，附 SHA-256 用于检查意外改写。soak 使用这次决定；已提交的空值表示没有合格档。缺失或无效的决定会阻止 soak 发流。

首次发流前，根目录 `setup.json` 保存阶段顺序、请求数、选档规则和输入来源。每轮先提交 `measurement_status` 和已有官方产物，再执行可选导出。`export_status` 单独记录导出结果。导出失败会停止后续流量，已经生成的文本和 JSON 仍进入索引。已提交且通过校验的 soak 仍按实测指标填写 `acceptance` 和 `soak_status`，导出错误单独体现在 `export_status` 和整次运行的失败状态中。

根报告损坏、缺失或进程被强杀后，可在容器内执行：

```bash
perf-suite --report-only --output /outputs/<本次时间戳目录>
```

重建按启动配置展开应执行的轮次，再核对保存的计划。缺少必跑轮次会反映在总状态中。SLO 和吞吐量取自已校验的官方 `vllm-result.json`。新运行会将 TPOT 有效样本数等观测计数与本轮官方结果绑定校验。其他运行的错误记录会产生警告；最终汇总发布时收到中断，本次运行会记录为 `interrupted`。

阶段摘要字段损坏时，该轮标为 `invalid_stage_record`，其余轮次继续汇总。启用原生导出后，`export_status` 记录执行结果，`export_integrity` 记录归档校验结果。新运行使用相对路径和 SHA-256，整个目录搬迁后仍可核对；缺失、损坏或非法的导出记录会使总报告标为失败。旧运行缺少摘要时显示 `unverified`，可定位的旧文件缺失时显示 `missing`。

重建只读取本地计划、阶段记录和官方产物，运行时的 NAS、密钥和 API 都无需提供。`not_started_or_unrecorded` 表示计划中有该轮但没有启动证据，`unconfirmed` 表示对应操作没有可确认的提交记录。根报告可重复生成；重建命令返回成功表示报告已生成，压测结论看 `summary.json` 的 `overall_status`。

新运行先按根计划中的 run_id 和 sequence 核对轮次归属，并检查已完成轮次的并发和请求数。错放的轮次显示 `round_mismatch`，可在原目录检查，其结果从汇总报告及索引排除。

排查报告时可查看以下字段：

| 字段 | 如何解读 |
|---|---|
| `request_count_status` | `final` 为最终计数，`partial` 为退出时已观察到的计数，`unknown` 表示只能确认已持久化的部分。数值 `null` 表示无法确认。 |
| `selection_status` | 新运行的 `committed` 表示已提交决定，`unconfirmed` 或 `invalid` 表示决定无法确认。旧运行显示 `not_recorded`，可以直接看 soak 的实际并发。 |
| `selection_evidence` | 已提交决定的 scan 依据全部通过校验时为 `verified`。依据缺失、错放或损坏时为 `unconfirmed`，具体轮次记录给出原因，历史选档值仍保留。 |
| `measurement_status` | 已提交的测量结论。即使后续归档损坏仍可读取。 |
| `artifact_integrity` | 分别给出文本和 JSON 的状态。新测量按文件名、字节数和 SHA-256 校验，`corrupt` 或 `missing` 会影响整体状态。无历史摘要的已有文件显示 `unverified`。 |

运行级错误保存在根目录 `run-error.json`，重建后的 `summary.json` 包含相同的档位、阶段、操作与原因。整个结果盘不可写时，请保留容器 stderr；离线重建将无法确认的停止原因标为 `unconfirmed`。

单次官方 benchmark 继续使用仅含 `vllm_bench_serve` 的 `perf.native.example.json`，可自行指定 `num_prompts`、`request_rate`、`max_concurrency` 等官方参数，启动命令省略 `--stage`。

旧分阶段配置需按新示例迁移：将 `target` 和 `dataset` 改为官方 API 和数据参数，将按时长的 `timing.soak` 改为请求数 `soak_requests`。原来的 `output_target` 是提示词中的建议长度，而 `random_output_len` 是官方随机数据集传给服务的生成预算。旧运行目录仍可用 `perf-suite --report-only --output /outputs/<旧运行目录>` 离线重建；新运行直接保存官方报告。
