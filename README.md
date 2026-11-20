# vllm-suite

远程大模型精度评测与性能压测。

Ubuntu 24.04 / Python 3.12 镜像，以固定版本 OpenCompass 的数据加载器、提示模板和评分器为核心。九项评测统一由 `eval-suite` 串行执行；模型回答和裁判均调用远程 OpenAI 兼容的 `/chat/completions` 接口。

## 安装

在项目根目录执行以下命令。构建机需已安装 Docker 并启动 daemon（可用 `docker info` 检查），且能访问系统软件源、Python 包源、GitHub 和评分资源下载站点。目标部署平台为 Linux amd64；其他架构构建或运行此镜像需要相应仿真支持。宿主机不需要安装 Python、CUDA 或模型权重。

```bash
docker build --platform linux/amd64 -t vllm-suite:opencompass .
```

构建需要联网下载系统包、Python 依赖和官方评分代码。精度环境依赖版本由 `requirements.build.lock`、`requirements.cpu.lock` 和 `requirements.lock` 固定，独立性能环境由 `requirements.perf.lock` 固定；升级依赖时需更新对应锁文件并重新验证。系统包不保证逐字节可复现。镜像包含 CPU 版 PyTorch（精度框架依赖）、评分资源以及基础 shell/网络工具，不包含模型权重。HalluQA、FollowBench 上游仓库自带的小型数据随官方代码保留；其他数据放 NAS。

构建完成后先检查两个 Python 环境及离线评分资源：

```bash
docker run --rm --network none vllm-suite:opencompass python /opt/evaluation/scripts/smoke_check.py
```

还可运行完整精度流程联调。该脚本在容器内创建临时数据与本地 Mock API，验证九项评测、输入冻结、中文 FollowBench 和错误处理，无需真实 API 密钥或挂载 NAS；生成的分数仅用于验证程序流程：

```bash
docker run --rm --network none vllm-suite:opencompass python /opt/evaluation/scripts/integration_check.py
```

导出后可拷贝到离线机器：

```bash
set -o pipefail
docker save vllm-suite:opencompass | gzip > vllm-suite-opencompass.tar.gz
# 目标机器
gzip -dc vllm-suite-opencompass.tar.gz | docker load
```

以上管道命令在 Bash 或 Zsh 中执行，`pipefail` 确保导出或导入任一步失败时返回非零退出码。导出失败的压缩包不能用于部署，应解决错误后重新导出。

目标机器仍需安装并启动 Docker，同时拷贝所需配置文件（精度使用 `suite.local.json`、`datasets.local.json`；性能使用 `perf.local.json`），准备 NAS 数据及性能所需的 tokenizer，并创建本地 `outputs/`、`cache/` 目录。配置可在下文按示例生成。导入镜像后也应执行上面的离线检查。离线指无需下载依赖和数据；正式评测仍需网络连通目标 API 及所需的裁判 API，不要给正式评测添加 `--network none`。

## 准备 NAS 数据

复制 `dataset-map.example.json` 为 `datasets.local.json`。`root` 是容器内 NAS 挂载点；逻辑名称固定，右侧目录或文件名称可任意修改。例如：

```bash
cp dataset-map.example.json datasets.local.json
```

```json
{
  "version": 1,
  "root": "/nas",
  "datasets": {
    "ceval": "uuid-a",
    "cmrc2018": "uuid-b/random.json",
    "halluqa": "uuid-c/questions.json",
    "followbench_en": "uuid-d"
  }
}
```

只检查本次选择的数据集。目录型数据集内部文件名必须符合加载器要求；映射改变的是数据集根路径，不能把内部文件格式随意改掉。NAS 使用只读挂载。FollowBench 每个 `example_id` 必须连续且包含完整、有序的等级：普通类别为 0–5，example 类别为 1–5；不完整分组会在调用 API 前失败。

| 逻辑名称 | 映射目标与内部结构 | 默认评测方式 |
|---|---|---|
| `ceval` | 根目录含 `dev/val/test/<科目>_<split>.csv` | val，5-shot，生成选项＋准确率 |
| `logiqa-zh` | 原版 LogiQA 目录，含 `zh_train.txt`、`zh_eval.txt`、`zh_test.txt` | 中文 v1 test，0-shot，只接受单个 A/B/C/D；不是 PPL 评测 |
| `mathbench` | MathBench 2024 `mathbench_v1` 根目录，内含 `college/`、`high/` 等子目录及 JSONL | 中英文；选择题 CoT＋四次循环评测，填空题使用上游示例 |
| `cmrc2018` | 有答案的 SQuAD 格式 dev JSON 文件 | 0-shot；官方 v5 中文 EM、F1 |
| `longbench` | v1 JSONL 数据目录，例如 `qasper.jsonl`、`multifieldqa_zh.jsonl` | 21 个子任务，使用各自评分器；不是 LongBench v2 |
| `finsafetybench` | 官方 `FinsafetyBenchmark.csv`，文件可改名 | 直接提问基线；官方裁判提示，报告评分均值和 score=10 的 ASR |
| `financeiq` | 目录含 `dev/<中文类别>.csv`、`test/<中文类别>.csv`，每行七列并带表头 | 5-shot，生成选项＋准确率 |
| `halluqa` | 官方 `HalluQA.json` 文件或包含该文件的目录 | 官方 few-shot 问答提示，裁判五票多数判定 |
| `followbench_en` / `followbench_zh` | 官方 `data` / `data_zh` 目录，包含六类 `*_constraints.json` | 官方规则评分＋LLM 裁判，HSR/SSR |

数据来源：[C-Eval](https://github.com/hkust-nlp/ceval)、[LogiQA](https://github.com/lgw863/LogiQA-dataset)、[MathBench](https://github.com/open-compass/MathBench)、[CMRC2018](https://github.com/ymcui/cmrc2018)、[LongBench v1](https://github.com/THUDM/LongBench)、[FinSafetyBench](https://github.com/sustech-nlp/FinSafetyBench)、[FinanceIQ](https://github.com/Duxiaoman-DI/XuanYuan/tree/main/FinanceIQ)、[HalluQA](https://github.com/OpenMOSS/HalluQA)、[FollowBench](https://github.com/YJiangcm/FollowBench)。提前在联网环境下载、解压并核对格式，评测不会自动下载缺失数据。

若使用镜像自带 HalluQA/FollowBench 数据，可以将映射值设置为 `/opt/benchmarks/HalluQA/HalluQA.json`、`/opt/benchmarks/FollowBench/data` 或 `data_zh` 的完整绝对路径。

## 选择范围和配置 API

复制 `suite.example.json` 为 `suite.local.json`。顶层 `datasets` 列表就是测试范围，并决定串行执行顺序。例如只测两项：

```bash
cp suite.example.json suite.local.json
```

```json
{
  "dataset_map": "/config/datasets.local.json",
  "datasets": ["halluqa", "followbench"],
  "target": {
    "base_url": "http://target-server:8000/v1",
    "model": "target-model",
    "api_key_env": "TARGET_API_KEY",
    "temperature": 0,
    "max_tokens": 2048,
    "timeout": 60
  },
  "judge": {
    "base_url": "http://judge-server:8000/v1",
    "model": "judge-model",
    "api_key_env": "JUDGE_API_KEY",
    "temperature": 0,
    "max_tokens": 2048,
    "timeout": 60
  },
  "followbench": {"language": "zh"},
  "continue_on_error": true
}
```

`base_url` 填到 `/v1`，不要包含 `/chat/completions`。密钥只通过环境变量传入。所有任务复用相同目标 API；HalluQA、FollowBench 和 FinSafetyBench 使用 judge API。当前配置校验要求两个角色都填写；只测客观题时可将 judge 配置为 target，并使用相同的密钥变量，不会产生裁判调用。

示例中的 `target-server`、`judge-server` 是占位地址，需换成容器可访问的实际域名或 IP。容器内 `localhost` 指容器自身；访问 Docker Desktop 宿主机上的 API 可使用 `host.docker.internal`。Linux Docker Engine 可在 `docker run` 中添加 `--add-host=host.docker.internal:host-gateway`，并让 API 监听容器可访问的宿主机接口。性能配置遵循同样规则。

七项主测试范围：

```json
"datasets": ["ceval", "logiqa-zh", "mathbench", "cmrc2018", "longbench", "finsafetybench", "financeiq"]
```

通过 `options` 选择部分子集或小样本验证：

```json
"options": {
  "ceval": {"subsets": ["ceval-computer_network"], "limit": 5},
  "longbench": {"subsets": ["LongBench_qasper"], "limit": 2},
  "finsafetybench": {"language": "both", "limit": 10}
}
```

不填写 `subsets` 则跑全部子集，不填写 `limit` 则跑全量。`limit` 按子集生效；MathBench 以原题计数并保留四个循环版本。HalluQA 和 FollowBench 要求全量，不接受 `limit`。HalluQA 必须提供官方提示词可使用的非空参考答案；FollowBench 还会检查完整分组、评分所需字段及提示文本唯一性。子集名称错误会在调用 API 前列出可选值。FinSafetyBench 支持 `en`、`zh`、`both`，默认 `en`；其直接提问基线不能与论文的攻击实验成绩混比。

旧版 `stages` / `evalscope` 配置已移除，需迁移为上述 `datasets` / `options`。

## 校验与运行

先确认宿主机已经挂载 NAS，并将命令中的 `/实际NAS路径` 替换为实际目录；Docker daemon 所在机器必须能访问该目录。NAS 中的数据与 tokenizer 需要对容器运行用户可读，目录需要可遍历；`outputs/`、`cache/` 需要可写。默认容器以 root 运行，启用 root squash 的 NAS 仍需授予相应读取权限。输出放在宿主机本地磁盘，NAS 只读存放输入资源。程序会拒绝部分常见网络文件系统作为新运行的输出根目录。请在宿主机确认输出目录位于本地磁盘。

先校验数据加载、子集和提示词，不调用 API，也不需要真实密钥：

```bash
mkdir -p outputs cache
docker run --rm \
  -v /实际NAS路径:/nas:ro \
  -v "$PWD":/config:ro \
  -v "$PWD/cache":/cache \
  -v "$PWD/outputs":/outputs \
  vllm-suite:opencompass eval-suite \
  --config /config/suite.local.json --output /outputs/check-001 --validate-only
```

正式执行：

```bash
export TARGET_API_KEY='你的目标API密钥'
export JUDGE_API_KEY='你的裁判API密钥'
docker run --rm \
  -e TARGET_API_KEY -e JUDGE_API_KEY \
  -v /实际NAS路径:/nas:ro \
  -v "$PWD":/config:ro \
  -v "$PWD/cache":/cache \
  -v "$PWD/outputs":/outputs \
  vllm-suite:opencompass eval-suite \
  --config /config/suite.local.json --output /outputs/run-001
```

首次运行时，`--output` 指定本地结果根目录，程序自动创建带时间戳和唯一后缀的子目录并打印路径。运行前会加载所有选定数据并生成提示词，缺文件或缺答案会提前失败。选定请求和参考答案保存在输出目录的 `inputs/` 中，设置只读权限并记录 SHA-256；生成与评分均使用这份快照，不再回读 NAS。请为这些输入及模型、裁判输出预留磁盘空间，不要修改运行目录。配置 `continue_on_error: true` 时某数据集运行失败仍继续后续数据集，最终返回非零退出码。SIGTERM / Ctrl-C 会更新已有汇总状态。输出目录创建后，setup 会逐步保存；准备或执行失败时，`summary.json.errors` 记录失败阶段、数据集和脱敏错误，`setup.snapshot_complete` 表示输入准备是否全部完成。配置文件无法解析或输出目录无法创建时，仅能通过退出码和终端错误定位。

使用真实数据的精度和压测任务在写入样本前，会按源文件、上级目录和输出目录的可读范围限制整个运行目录。可以确定共同用户组时保留组访问；目录型数据集、不同私有组或无法可靠换算的 ACL 默认仅运行账户可访问。冻结输入时只移除写权限。Linux bind mount 保留数字 UID/GID；要由宿主机账户读取私有结果，可在 Docker 命令中传入 `--user "$(id -u):$(id -g)"`，并确保该身份有权读取 NAS、写入输出与缓存。复制归档时应保留整个运行目录及其访问权限。

运行只需要连通目标和裁判 API。数据使用绝对本地路径，评分资源预置于镜像；不初始化模型或 tokenizer，不设置 API 地址白名单。所有提示词原样发送，不静默截断；LongBench 必须由足够上下文长度的远程模型承接。截断回答、服务错误和无效裁判输出会使该项失败，不计作正常低分。FinSafetyBench 裁判回复必须以唯一的 `Rating: [[1]]` 至 `Rating: [[10]]` 行结尾；示例中的数字不会被当作评分。

## 查看 JSON 结果

`summary.json` 包含：

- `schema_version: 2`；`overall_status` 为 `preparing`、`validated`、`running`、`complete`、`failed` 或 `interrupted`，独立于各数据集状态。任务间中断也会记录为 `interrupted`；尚未执行的任务为 `not_run`。
- `metrics`：框架、数据集/子集、指标名、数值、原始结果路径。
- `setup`：API 地址、模型名、采样参数、密钥环境变量名、数据映射、有效任务配置、样本序号、冻结输入文件路径及 SHA-256、完整 Python 依赖版本与锁文件快照、官方代码版本和执行器哈希。不会保存密钥值。

```bash
# 假设 Docker 打印 /outputs/run-001/20261120-120000-123456-abcdef12
RUN_NAME=20261120-120000-123456-abcdef12
RUN_DIR="$PWD/outputs/run-001/$RUN_NAME"
jq '.overall_status, .status, .metrics' "$RUN_DIR/summary.json"
```

每个子集还保留 `samples.jsonl`（请求提示、参考答案、模型输出）和 `metrics.json`；裁判类测试保存裁判结果。失败见对应数据集目录的 `error.txt`，已完成子集指标保留于 `metrics.partial.json` 并纳入总汇总，数据集状态仍标为失败。无跨数据集总分，因为各指标口径不同。

基础验证（第一条为源码测试，需要宿主机 Python 3.12；未安装性能依赖时会跳过流式集成测试，完整运行方式见下文独立性能环境或 Mock 指南）：

```bash
python3.12 -m unittest discover -s tests -v
docker run --rm --network none vllm-suite:opencompass python /opt/evaluation/scripts/smoke_check.py
```

## 性能压测（perf-suite）

性能与精度分别运行，不需要 judge。性能客户端使用独立 Python 环境，包含 vLLM **v0.11.0** 的 benchmark serving 请求实现；源码及 Apache-2.0 许可证保存在 `scripts/perf_vendor/`，来源和 SHA-256 见 `PROVENANCE.json`。不安装 GPU 推理引擎。适配层补充真实内容 TTFT、完成状态、自然 EOS、用量及持续时间调度；不是 MLPerf 合规跑分。

复制 `perf.example.json`，设置 API、匹配模型的本地 tokenizer、`context_window` 和 `max_generation_tokens`（必须符合实际服务限制）。`base_url` 包含 `/v1`。可用 `chat_template` 指定本地模板文件；默认使用 tokenizer 自带模板，不执行远程代码。`target.extra_body` 可传思考模式等参数，但不能覆盖消息、长度、EOS 和流式协议。

### 压测交付与验收约定

以下决定作为两种压测模式的实现和发布验收依据。故障场景和退出码由压测回归测试覆盖，镜像交付另按下文在目标环境验收。原生模式以 vLLM 0.11.0 为准，分阶段模式的健康、SLO、恢复和长稳判定沿用下文规则。

**运行结果。** 原生模式在开始计量前记录实际选出的请求数 N，结束时记录实际发出的请求数和上游成功数 S。就绪探测及 profiler 控制请求单独处理。允许数据集按原生规则少取样本，例如启用 `no_oversample` 后实际得到 20 条，即按 20 条验收。`save_detailed` 仅控制明细，开启和关闭时应得到相同的状态。原生运行目录的 `setup.json` 保存生效参数；`summary.json.requests` 中的 `selected`、`dispatched`、`successful` 分别记录入选、派发和成功数。`traffic_status` 记录压测结果，`artifacts` 分别记录文本报告、指标和额外导出状态。

| 条件 | summary.json 中的 overall_status | 进程退出码 |
|---|---|---|
| 参数校验通过且仅执行校验 | validated | 0 |
| 原生计量请求全部发出且全部成功，N > 0，必需产物和显式要求另存的结果均已写完 | complete | 0 |
| 原生运行正常结束，至少一条成功，但有漏发或失败 | incomplete | 1 |
| 没有可测请求、成功数为零、计数无法核对、准备或执行异常、应交付文件写入失败 | failed | 1 |
| Ctrl-C 或 SIGTERM 中断压测 | interrupted | 130 |

进程优先记录中断或执行异常，然后判断产物写入，再判断请求覆盖。分阶段模式按所选阶段的验收规则确定 `complete` 或 `incomplete`。汇总分别记录发流结果、各产物的写入状态及失败原因；另存失败时仍可查到已完成的请求数。两种模式都在发流前写入运行状态，已结束的运行用原子替换更新汇总。

**原始报告。** 原生模式流式接收 vLLM 打印的终端报告。从 `Serving Benchmark Result` 标题行到结尾分隔行完整收到后，立即原样发布 `vllm-report.txt`，包括原有指标、顺序、空格和换行。临时文件与正式文件在同一文件系统。报告完成后，即使后续原生 JSON 导出失败或收到中断，已发布的文本仍可读取。只收到部分报告时，临时内容不得以正式报告文件名发布。vLLM 返回结果后，另存 `vllm-result.json`。`save_result`、`append_result` 和 `result_dir` 控制的上游导出单独记录状态。分阶段模式继续按已完成阶段保存请求日志和原生报告，离线重建使用现有日志。

**输入与配置。**

- 分阶段压测在准备期间为选中的本地数据集建立一份运行私有快照，所有长度档从这份快照取样，`dataset_sha256` 对应快照字节。复制期间发现源文件变化即终止准备。原始快照位于运行目录的 `inputs/source/`，请额外预留一份源数据大小的磁盘空间。各档另存实际输入及 SHA-256。NAS 上的 tokenizer 直接沿用已有 ModelScope 目录布局；同一次运行只加载一次，记录文件指纹。
- 分阶段模式以顶层 `chat_template` 为显式模板入口。配置了模板时，读取一次，用同一份文本本地计数、构造实际请求并写入运行记录；服务端拒绝时报告失败。省略时使用本地 tokenizer 自带模板，部署方负责使服务端默认模板匹配。`target.extra_body.chat_template` 一律在发流前报错。`chat_template_kwargs` 在本地和请求端使用同一组值。服务端省略 usage 时，计数标记为本地估算。
- 分阶段固定结构逐层校验字段名，包括 `target`、`dataset`、`timing`、`slo` 和 `profiles[]`。错误指出完整字段路径。`target.extra_body` 保留服务端扩展字段，并执行受保护字段校验。原生选项集中放在唯一顶层对象 `vllm_bench_serve`，由固定版本的官方解析器校验。
- 两种模式的显式 seed 均须为 JSON 整数，拒绝 `null`、布尔值、小数和字符串。分阶段模式省略时为 42，原生模式省略时为 0，并接受其随机库支持的整数范围（0 至 4294967295）。分阶段复跑保持数据、tokenizer、模板、档位配置和顺序一致，以各档输入哈希判定样本一致。

**运行目录与部署。** 两种模式在首个 API 请求前保存脱敏后的生效配置、模式、版本、seed 和输入来源。原生模式保留官方默认值及用户覆盖值，记录实际使用的模型、tokenizer、数据集、长度、发流与采样选项。认证只记录环境变量名，自定义请求头只记录名称。分阶段模式额外交付输入快照、各阶段日志和报告索引。原生模式交付汇总、原始文本报告及上游返回的指标 JSON。失败或中断时，汇总逐项注明已生成、未生成和写入失败。

配置、数据集和本地模型资源作为只读输入。精度和压测的 `--output` 指向本地结果根目录，程序每次自动创建带时间戳和唯一后缀的子目录。结果目录与输入分离。Linux 同时核对绑定挂载的源路径，即使 NAS 在容器里有多个挂载名称也会检查目录重叠。原生额外导出集中在本次运行的 `exports/` 下。离线交付按所用模式携带镜像及配置。两种模式都部署时，在联网构建机完成下列步骤，将镜像包和两份已编辑的配置带到目标机；NAS 数据和 tokenizer 单独挂载。

```bash
# 构建机，目标平台为 Linux amd64
set -o pipefail
docker build --platform linux/amd64 -t vllm-suite:opencompass .
docker build --platform linux/amd64 --target native-bench -t vllm-suite:native-bench .
docker save vllm-suite:opencompass vllm-suite:native-bench | gzip > vllm-suite-images.tar.gz
# 同步 perf.local.json、perf.native.local.json 和镜像包到目标机
# 目标机
set -o pipefail
gzip -dc vllm-suite-images.tar.gz | docker load
```

导入后用新输出目录执行下文两种模式的断网校验命令。校验与正式运行使用相同的只读 NAS 挂载。

**发布验收。** 使用本地模拟服务，分别验证全成功、部分 503、全部 503、就绪探测失败和实际样本少于配置数。核对退出码、汇总、请求计数及官方报告原文。分别在报告结尾之前和之后中断，再注入额外 JSON 写入失败，检查已完成报告的保留情况。替换 NAS 源文件、替换外置模板、拼错嵌套字段、传入空 seed，分别核对冻结输入或准备失败的结果。检查输入内容保持一致，并验证运行目录外的导出在发流前失败。最后在离线目标机导入镜像，分别校验两份配置。部署验收记录应包含镜像标签、目标架构和两个离线校验的退出码。

### NAS 上的 tokenizer 目录

把已有 ModelScope 下载产物按原目录结构放在 NAS 上。`tokenizer` 填写容器内**直接包含 tokenizer 配置和词表的模型目录**。下面是 Qwen 快照的一个实际示例：

```text
/nas/modelscope/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master/
├── config.json
├── tokenizer_config.json
├── tokenizer.json
├── vocab.json
└── merges.txt
```

```json
{
  "tokenizer": "/nas/modelscope/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master"
}
```

也支持直接导出的模型目录：如果文件已经位于 `/nas/models/Qwen/Qwen2.5-7B-Instruct/`，就把该路径填入配置。沿用实际下载版本的文件名、相对路径和子目录，路径中的 `master` 可换为实际快照版本。不同模型所需文件由 tokenizer 决定，例如 SentencePiece 词表、特殊 token 和聊天模板。压测只读取本地文件，完整模型目录也可直接挂载。

```bash
cp perf.example.json perf.local.json
# 按服务实际配置修改 API、模型名、上下文窗口、生成预算及长度档位。
mkdir -p outputs
docker run --rm --network none \
  -v /实际NAS路径:/nas:ro \
  -v "$PWD":/config:ro -v "$PWD/outputs":/outputs \
  vllm-suite:opencompass perf-suite \
  --config /config/perf.local.json --output /outputs/perf-check --validate-only
```

校验会从 NAS 加载 tokenizer 并准备输入。独立聊天模板可用 `chat_template` 指定容器可读的文件路径。宿主机运行时，`tokenizer`、`chat_template` 和 `dataset.path` 均填写宿主机实际路径。

容器中正式运行：

```bash
export TARGET_API_KEY='你的目标API密钥'
docker run --rm -e TARGET_API_KEY \
  -v /实际NAS路径:/nas:ro \
  -v "$PWD":/config:ro -v "$PWD/outputs":/outputs \
  vllm-suite:opencompass perf-suite \
  --config /config/perf.local.json --output /outputs/perf-run
```

正式运行及校验在结果根目录下各自创建带时间戳的运行目录；`--report-only` 的 `--output` 填写打印出的已有运行目录。`--stage connectivity|scan|overload|soak|all` 选择阶段；overload/soak 会先运行本次并发扫描，避免复用过期甜点。所有实际运行先做联通检查。`--validate-only` 不调用 API、不需要密钥。

### 可选：宿主机独立性能环境

此方式需要宿主机安装 Python 3.12，安装依赖时需要联网。先从 `perf.example.json` 复制出 `perf.host.local.json`，修改 API、模型限制及 tokenizer 等资源为宿主机实际路径；与容器配置分开保存。

```bash
cp perf.example.json perf.host.local.json
# 编辑 perf.host.local.json，tokenizer 指向宿主机实际的模型目录
python3.12 -m venv .venv-perf
.venv-perf/bin/python -m pip install -r requirements.perf.lock
.venv-perf/bin/python -m pip check
.venv-perf/bin/python scripts/run_perf.py --config perf.host.local.json --output outputs/perf-host-check --validate-only
export TARGET_API_KEY='你的目标API密钥'
.venv-perf/bin/python scripts/run_perf.py --config perf.host.local.json --output outputs/perf-host-run
```

### 在 JSON 中配置完整的 vLLM 原生压测

复制 `perf.native.example.json` 为 `perf.native.local.json`。把所有 `vllm bench serve` 参数放进唯一的顶层对象 `vllm_bench_serve`。例如示例中的 `request_rate: 10` 指每秒发起 10 个请求，`max_concurrency: 32` 限制同时进行的请求数。原生模式按 vLLM 的负载模型执行一次压测。要执行本项目的扫描、过载和长稳阶段，使用前面的 `perf.example.json`。

字段名去掉 CLI 选项的开头 `--`，连字符或下划线均可。开关写 JSON 布尔值，多值参数写字符串数组，例如 `"goodput": ["ttft:2000", "tpot:50"]`。字典参数可写 JSON 对象，例如 `"random_mm_limit_mm_per_prompt": {"image": 2}`。无限请求率写 `"request_rate": "inf"`。未填写的参数沿用 vLLM 0.11.0 默认值（原生 seed 默认为 0）。完整选项由该版本的官方 parser 提供，未知参数会报错。OpenAI 兼容服务的 `base_url` 填主机和端口，完整 API 路径填入 `endpoint`。认证通过 `OPENAI_API_KEY` 环境变量传入。

原生模式需要完整 vLLM 客户端环境。可用项目提供的独立镜像目标构建和运行，容器中的 tokenizer 路径与 NAS 挂载位置需对应。

```bash
cp perf.native.example.json perf.native.local.json
# 编辑模型、地址和 NAS 上的实际 tokenizer 路径
mkdir -p outputs
docker build --platform linux/amd64 --target native-bench -t vllm-suite:native-bench .
docker run --rm --network none -v /实际NAS路径:/nas:ro -v "$PWD":/config:ro -v "$PWD/outputs":/outputs \
  vllm-suite:native-bench --config /config/perf.native.local.json --output /outputs/native-check --validate-only
docker run --rm -e OPENAI_API_KEY -e HF_HUB_OFFLINE=1 \
  -v /实际NAS路径:/nas:ro -v "$PWD":/config:ro -v "$PWD/outputs":/outputs \
  vllm-suite:native-bench --config /config/perf.native.local.json --output /outputs/native-run
```

也可在已安装 `vllm[bench]==0.11.0` 的环境中运行 `python scripts/run_perf.py --config perf.native.local.json --output outputs/native-run`。参数清单可用该环境的 `vllm bench serve --help` 查看。`--validate-only` 校验参数名称、类型、官方解析器约束及输出边界，完整数据读取、tokenizer 加载和服务连通性在正式运行时检查。程序会打印本次创建的运行目录。完成后 `vllm-report.txt` 保存官方终端报告原文，`vllm-result.json` 保存官方返回的指标。每请求明细由 `save_detailed` 控制。`save_result`、`append_result` 和 `result_filename` 控制 vLLM 自身另存的结果文件，文件统一位于本次输出目录的 `exports/` 下。可用 `result_dir` 设置相对子目录，`result_filename` 只填文件名。`append_result` 在本次新运行中写入首条记录。

### 长度与数据

输入长度与数据集可独立指定。下面是加入性能配置的字段示例，仍需保留 `target`、`tokenizer`、`context_window`、`max_generation_tokens` 等必需配置：

```json
{
  "dataset": {"name": "sharegpt", "path": "/nas/datasets/sharegpt.json"},
  "profiles": [
    {"name": "8k", "input_tokens": 8192, "variation": 0.1, "output_target": 1024},
    {"name": "32k", "input_tokens": 32768, "variation": 0.1, "output_target": 1024}
  ]
}
```

`profiles` 可以只保留一个档位，也可以设置多个自定义档位；`input_tokens` 为输入 token 数，`variation` 为允许的长度偏差比例。切换为 `"dataset": {"name": "random"}` 可生成指定长度范围的随机输入，无需数据文件。`output_target` 是自然输出长度提示，不保证实际输出长度；`max_generation_tokens` 才是生成硬上限。每次运行选择一个数据源，比较多个数据集时分别运行并使用不同输出目录。

默认 4k、16k、64k 输入，±10%，k=1024 tokens，包含 chat template 和输出目标提示。`output_target` 只引导自然输出，不映射为 max_tokens；可设 null 去掉长度提示。不会 padding、ignore EOS 或按目标长度提前截断。生成硬预算由 `max_generation_tokens` 与剩余上下文决定。若输入上界加完整生成预算超过服务上下文，准备阶段拒绝该档，不自动缩短；64k 输入上界约 70.4k，还需输出空间。

`dataset.name` 支持：

| 名称 | 本地 JSON/JSONL 输入 |
|---|---|
| `random` | 不需要路径；固定种子生成并重新编码验证长度 |
| `sharegpt` | 原生 `conversations: [{from: human/gpt, value: ...}]`；移除最后的参考回答 |
| `livecodebench` | 原生 `question_content`、可选 `starter_code`，或统一 messages |
| `longcodebench` | 使用该 benchmark 的任务预处理输出：`prompt` 或 `messages` |
| `long-code-arena` | 使用官方任务的仓库上下文构建步骤生成 `prompt` 或 `messages` |

统一格式示例：`{"id":"task-1","messages":[{"role":"user","content":"任务说明及真实代码上下文"}]}`，必须以 user 结尾；参考答案不放入 messages。长上下文适配器消费已构建的任务输入，**不会自动下载仓库或运行任务环境**。输入路径用 `dataset.path`；建议固定数据版本，运行记录文件哈希。各数据源独立运行，不混合报告。真实样本仅筛选符合长度的请求，不拼接、截断或填充。默认每档最多 200 个、至少 20 个匹配样本；不足标记不可测。长稳循环复用冻结样本，报告复用率，缓存热度会影响结果。

### 阶段、SLO 与默认值

1. 联通：3 个短请求，加每档 1 个实际请求，必须自然正常完成。
2. 并发扫描：`[1,2,4,8,16,32,64,128]`；每档预热 30 秒；正式至少 120 秒、200 个完成请求，最长发压 600 秒，随后排空。
3. 超载：只有扫描找到饱和点才执行。相邻末档健康吞吐增长不超过 5% 时，将达到实测峰值 95% 的最低并发作为饱和点；发压 1.5/2/3 倍，每级 180 秒，默认客户端上限 256。未找到饱和点时明确跳过，需增加扫描范围。每级撤压后以并发 1 检查恢复：最多 300 秒，连续三个 30 秒窗口全部成功且 P95 TTFT/TPOT 不超过基线 120%。
4. 长稳：每档合格甜点串行运行 7200 秒，持续补充请求，按 60 秒窗口统计。到时停止新增并自然排空。

所有阶段使用 `request-rate=inf` 与有界客户端并发池。发压时间不包含尾部排空；默认单请求超时 600 秒，失败不自动重试。超载/长稳连续失败且 30 秒无成功或有效流式进度，或一分钟内超时/5xx 比例超过 20%，触发保护；尚在首 token 等待预算内且没有失败的请求不会被当作失活；超载恢复失败停止整次运行。429/503 单列统计。

**提交时可以设置 SLO；未设置用 P95 TTFT=2000ms、P95 TPOT=50ms/token。** 顶层 `slo` 设置全局默认，`profiles[].slo` 可按档覆盖，64k 不自动放宽。甜点候选还需成功率至少 99.9%、无截断、样本充分；选输出 tokens/s 最高档，差距不超过 5% 时选择较低并发。无合格档则跳过长稳并返回未完成验收。

可通过 `timing` 覆盖秒数：`warmup, scan, scan_max, overload, recovery, recovery_window, soak, window, timeout, no_success`；`scan_requests` 为最低完成请求数。缩短这些值仅适合流程验证，不是正式性能结果。

### 结果与验证

`summary.json`、`setup.json` 保存阶段结果、选择与快照；每阶段有请求 JSONL、窗口 JSONL、精确指标 SQLite 和 JSON。长稳完整请求指标落盘，避免在 Python 内存保留全程请求。

压测结束后保留 vLLM v0.11.0 的 **Serving Benchmark Result** 文本格式及原生指标口径：

| 文件 | 内容 |
|---|---|
| `vllm-report.txt` | 各输入档位的并发扫描、超载和长稳报告，包含吞吐及 TTFT、TPOT、ITL、E2EL 的均值、中位数和 P50/P95/P99；同时打印到终端 |
| `vllm-reports.json` | 所有已完成阶段（含预热、恢复）的报告路径索引 |
| `report-jobs.json` | 持久化的报告任务清单，记录待生成、生成中、完成或失败的状态 |
| `<档位>/<阶段>.vllm.txt` | 单个阶段的 vLLM 文本报告 |
| `<档位>/<阶段>.vllm.json` | vLLM 指标、模型及并发信息，保留逐请求长度、TTFT、ITL、生成文本和错误等详细数组 |

报告使用随项目固定的官方 `serve.py` 报告语句；指标从请求明细经 SQLite 汇总，并通过官方 `calculate_metrics` 的逐字段对照测试。仅在全部发压及恢复检查结束后生成报告，报表计算不会占用恢复预算或计入服务吞吐分母。请求明细和原生报告均包含生成内容，需要为这两份数据及报表临时文件预留磁盘空间。每个阶段完成后即保存报告任务；运行被中断时，未生成的报告保持待处理状态。`reporting_status` 表示报告是否成功生成，未生成的报告不能视作完整交付。

中断或报表生成失败后，可以只补生成已完成阶段的报告，不重新压测、不调用 API、不需要 API 密钥：

```bash
# 将 RUN_NAME 替换为本次实际打印的时间戳目录名
RUN_NAME=20261120-120000-123456-abcdef12
# 宿主机重建宿主机运行产生的报告
.venv-perf/bin/python scripts/run_perf.py --report-only --output "$PWD/outputs/perf-host-run/$RUN_NAME"

# 容器重建容器运行产生的报告。新格式支持断网、卸载 NAS
docker run --rm --network none \
  -v "$PWD/outputs":/outputs \
  vllm-suite:opencompass perf-suite --report-only --output "/outputs/perf-run/$RUN_NAME"
```

报告重建应使用本地磁盘上的运行目录。此命令使用运行目录的 `config.json`、阶段结果及请求日志，不与 `--config`、`--validate-only`、`--stage` 合用。新格式直接使用每个请求保存的原生报告 token 数及来源。新运行在 `summary.json.report_contract_version` 声明版本。已声明版本或请求行含新格式计数字段的阶段缺少完成记录时，重建会失败。阶段结果的 `report_contract` 保存格式版本、计数规则版本和请求日志 SHA-256。重建流式校验摘要、总数及序号恰好覆盖 0 至 N−1，允许乱序。任何阶段失败都会保留当前索引。旧格式校验数量和序号，报告元数据标明未验证历史摘要。仅当旧日志缺少原生输出计数并需要重新分词时，才需按原路径挂载 tokenizer；其文件指纹必须与 `setup.json` 一致。使用 `fast_tokenizer_files` 时还会核对 Transformers 选中的版本文件；直接配置的 `tokenizer_file`、词表及合并规则文件也须有原始指纹；旧快照缺少该文件指纹时，无法验证的回退计数会导致重建失败。

旧版本没有任务清单时，按已发布索引、摘要中的发布记录和现存阶段结果恢复清单。缺少已发布阶段的完成记录会指出具体路径，保留原索引；恢复文件后可重试。每次生成都会把整批阶段报告、汇总文本和索引写入独立的 `.vllm-reports/run-*/` 目录，全部写完后原子切换当前版本。读一整批报告时先读取一次 `vllm-reports.json`，再按其中相对于运行目录的路径读取文件。根目录的汇总文本和各档位的固定文件名用于查看当前版本。首次迁移旧目录时，先保存旧版报告，再将固定路径逐个接到旧版，最后一次切换到新版。此过程中断可直接重试。

报告重建要求版本存储和阶段结果位于运行目录中的实际路径；发布的报告入口仍使用项目创建的符号链接。修复旧报告的权限时会先分离共享文件的硬链接。版本目录遵循输出目录和创建进程的权限掩码（umask）。阶段目录沿用原档位目录的所有者、属组与权限限制。新版本的阶段文本和详细 JSON 使用请求日志的所有者与属组，权限位同时受日志、已有报告和 umask 限制。离线重建也会按现存请求日志校正已发布 run-* 版本中对应阶段文件的访问权限，历史报告内容保留。重写已有运行元数据时保留其所有者、属组和读写权限。执行账户无法设置所需属主或属组时，命令会报错。

已发布版本保留，因此重复生成时需要为每一版详细报告预留空间。复制或归档运行目录时须包含隐藏目录和符号链接。补生成成功仅将 `reporting_status` 更新为 `complete`，原来的 `overall_status` 和阶段验收结果保持不变；未完成阶段的部分请求不会被冒充为完整阶段。不要在原压测进程仍运行时对同一目录补生成报告。

原生报告保留 vLLM 对成功、TTFT 和 ITL 的定义；本项目的 `summary.json` 按真实内容 TTFT、自然完成、token 数一致性和 SLO 进行验收，两者可能不同。尤其不能把原生报告中的 Successful requests 直接当作本项目验收通过；原生 ITL 仍是流式事件的间隔。`*.requests.jsonl` 的 `native_output` 保留请求实现返回的完整原生结果，错误信息中的密钥会脱敏。适配层主动超时时没有原生返回值，标记为 `native_output_origin=adapter_timeout`，原生报告元数据记录这类请求数量。

请求调度沿用本项目按持续时间运行的有界并发池，JSON 中的 `vllm_suite_metadata.scheduler` 明确记录这一点。报告格式及指标与官方实现对照，不代表本项目直接执行了完整的 `vllm bench serve` CLI。

### 健康判定与保护规则

扫描、最佳并发选择、饱和点、恢复和最终验收共用健康判定：负载时长及请求数充分、没有触发保护、没有客户端瓶颈、成功率至少 99.9%、无截断及输入 token 数不一致。阶段结果中的 `health` 和 `acceptance_with_slo` 分别记录健康判定与包含延迟 SLO 的判定，并给出失败原因；`selection` 保存每个扫描档位参与最佳并发选择的判定。策略版本与阈值保存在 `setup.json.health_policy`。

饱和趋势只比较扫描末尾两个相邻且健康的档位，异常档位会中断趋势判断；恢复基线和每个恢复窗口还要求 100% 成功，连续三个窗口的 P95 TTFT/TPOT 均不超过基线的 120%。超载验收不要求满足正常负载延迟 SLO；长稳验收要求健康且满足配置的延迟 SLO。

失活与错误率使用独立时间线：出现失败后，距最近成功完成或有效流式输出达到 `no_success` 才触发失活保护；有效输出或成功完成重置该时间线。没有失败、仍在等待首 token 的请求由自身超时预算管理。错误率保护按最近 60 秒内完成的请求计算，完整运行 60 秒后，超时/传输错误/5xx 占比严格超过 20% 时触发。429 单独作为拒绝统计，不计为该窗口的服务错误。保护触发后停止新请求，在途请求依各自时限排空，阶段结果的 `guard` 保留触发时的计数与时间依据。

TTFT 从发请求到首个非空 content/reasoning_content；TPOT 为 `(端到端延迟-TTFT)/(输出tokens-1)`，短于 2 tokens 不参与 TPOT。保留 vLLM 原生计时供对照，不把 SSE chunk 间隔当作真实 ITL。吞吐分母包含排空，分钟窗口按完成请求统计。服务端 usage 优先，否则使用本地 tokenizer 估算并标注；隐藏思考 tokens 无 usage 时无法精确恢复。截断/非自然完成不计为成功。

只有 API 时只能验证外部可用性与恢复，不能证明服务内部排队长度或进程是否重启。报告包含压测机 CPU、峰值 RSS 和事件循环延迟；延迟超过 100ms 标记客户端瓶颈嫌疑。不得把存在客户端瓶颈的结果当作服务容量结论。

```bash
.venv-perf/bin/python -m unittest discover -s tests -v
```

测试覆盖配置、长度、样本不足、SLO 选择、UTF-8 分片 SSE、真实 TTFT、usage 缺失、EOS/截断/断流/超时、并发上限、排空、保护及四阶段编排。真实服务的 2 小时验收需另行执行。

### 超载验收与故障边界

超载阶段将“执行完成”和“外部承载验收”分开：每级 `acceptance` 保存 `passed`、失败原因、是否观察到拒绝及 `internal_queue: unknown`。默认外部承载通过要求发压时长充分、无客户端瓶颈、未触发保护、成功率 ≥99.9%、无 429/503 拒绝、无截断且撤压后恢复。拒绝说明观察到了限流/保护行为，不代表排队承载通过；任何一级不通过，整体不能报告验收成功。此判定不要求超载期间满足正常负载延迟 SLO，也不声称证明服务内部排队。

恢复预算为真正的总截止时间，包含发压和在途请求排空。到期取消恢复检查并判失败，保留请求/窗口日志；正常扫描和长稳结束依旧自然排空。HTTP 200 后的流式超时通过 `error_kind=timeout` 纳入故障保护，HTTP 状态和传输错误类型分别记录。

客户端瓶颈档不参与甜点、饱和点或恢复基线选择。仅支持 `n=1`。`target.extra_body.chat_template_kwargs` 同时传入本地 token 计数与服务请求；应确保本地模板与服务模板一致。请求日志 `input_token_delta` 记录服务端 prompt usage 与本地计数差值（缺 usage 为 null），非零时联通检查失败；运行中出现非零差值的档位不得进入甜点、饱和点、恢复基线或通过验收。缺 usage 时保留本地估算并标注，不能声称已校验服务端 token 数。

### 自然长输出的失活检测与有界准备

三个请求时限相互独立，均可在 `timing` 中覆盖：

- `first_token_timeout`：等待首个有效 content/reasoning_content 的上限，默认继承 `timeout`（600 秒），允许长上下文 prefill 和排队；不是 TTFT SLO。
- `stream_idle_timeout`：首个有效输出之后连续没有新内容的上限，默认继承 `no_success`（30 秒）；每次有效输出重置。role、注释、心跳不重置。
- `timeout`：整请求硬时限，默认 600 秒；持续输出也不能无限占用请求。超时记录 `error_kind=timeout`，主动截止还记录 `timeout_phase`。

自然长输出持续推进时，即使超过 30 秒尚未完成也不触发无成功保护。正常测试结束仍自然排空，直到请求完成或其明确时限到期。超时保护不用于凑输出长度。

禁止 `truncate_prompt_tokens`、`truncation`、`max_input_tokens` 等输入截断参数。输入计数不一致汇总为 `input_token_mismatches`，不会静默当成正常长度档位。

JSONL 按行、JSON 数组通过流式解析器读取。每档采用固定种子的蓄水池抽样，最多保留 `sample_count` 个匹配样本；工作内存与样本上限、单条记录大小有关，不随数据集总行数增长。样本不足仍拒绝该档。数据哈希分块读取，tokenizer 快照仅包含词表、tokenizer 配置和模板资源，不递归读取模型权重；每次哈希最多读取 1 MiB 块。

### 客户端阻塞与 usage 校验

事件循环延迟由独立监测任务按预期唤醒时间计算，覆盖发压、窗口统计、同步日志/SQLite 写入及请求排空。超过 100ms 的档位标记客户端瓶颈嫌疑，不进入甜点选择；监测不随每轮统计结束重置，因此统计本身的耗时也可被发现。所有请求结束后的最终报表计算不计入服务负载阶段。

服务端 usage 若存在，必须是对象；其中出现的 `prompt_tokens`、`completion_tokens`、`total_tokens` 必须是非负整数（不接受布尔、字符串、小数或字段值 null）。三个字段齐全时，total 必须等于 prompt 与 completion 之和。非空输出不能报告零 completion tokens，计数不能超过对应上下文/生成预算。整个 usage 为 null 或缺失视为未提供，缺失的 completion 计数使用本地 tokenizer 估算。

格式错误的 usage 标记 `error_kind=protocol`、`token_source=invalid`，请求判失败，失败记录中的 output_tokens 使用零占位并排除出吞吐和延迟统计；不使用估算掩盖协议错误，也不因单条 usage 格式错误终止整个阶段。失败率仍参与阶段验收。原始错误原因保留在请求日志中。

### Host 上的 FastAPI 联调验证

无需 Docker 或真实模型，可运行 [FastAPI mock 验证脚本](tests/host_mock/README.md)。
它启动独立的本机 HTTP/SSE 服务，通过生产压测 CLI 验证 random/ShareGPT、
4k/16k/64k、并发扫描、过载排队、恢复及缩短后的长稳流程，并注入异常响应。
测试报告保留在指定的 artifacts 目录；mock 数据不能作为真实模型性能或精度结果。
