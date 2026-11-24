# vllm-suite

远程大模型精度评测与性能压测。

Ubuntu 24.04 / Python 3.12 镜像，以固定版本 OpenCompass 的数据加载器、提示模板和评分器为核心。九项评测统一由 `eval-suite` 串行执行；模型回答和裁判均调用远程 OpenAI 兼容的 `/chat/completions` 接口。

## 在已启动的容器内运行

在工作目录保存 `.vllm.conf`，然后执行 `vllm-suite`。以下示例先跑四阶段压测，再跑 CEval。填写服务实际返回的模型名、API 地址和已挂载的 tokenizer 目录。使用标准数据布局时，CEval 从 `/nas/ceval-exam` 读取。

```yaml
model:
  name: default
  base_url: http://target-server:8000/v1
  tokenizer: /nas/modelscope/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master
execution:
  perf: {}
  accuracy: {}
```

```bash
vllm-suite
# 指定其他配置和本地结果根目录
vllm-suite --config /config/nightly.yaml --output /outputs
```

写入 `perf: {}` 或 `accuracy: {}` 即按默认值启用对应任务，删除其中一项即可单独运行另一项。仅跑精度可省略 tokenizer。无鉴权服务可省略 `api_key_env`。HalluQA、FollowBench 和 FinSafetyBench 需要在 `execution.accuracy.judge` 填写裁判服务。

默认结果目录是 `/tmp/vllm-suite/<UUID>/`，根目录包含 `effective-config.yaml` 和 `summary.json`，性能与精度分别写入 `perf/` 和 `accuracy/`。启用精度时，启动预检将所选数据集的实际路径写入 UUID 目录的 `dataset-map.json`；精度进程读取这份快照。两项都启用时，压测成功退出后才发起精度请求。单次压测的 `--result-dir` 必须位于本次 `perf/` 内。重复的 YAML 键会报错并显示行号。需要在删除容器后保留结果时，用 `--output` 指向挂载的本地磁盘。

`perf.stage` 支持 `all`、`connectivity`、`scan`、`overload`、`soak`、`single`。默认四阶段使用 4k、16k、64k 三档。scan 每档请求数为并发度的 10 倍，overload 默认在 192、256 各打三轮独立脉冲，soak 按 scan 的选档发 1000 个请求。官方参数按字符串数组写在 `perf.bench`，每个 profile 的 `bench` 追加在后。默认参数、公共 `bench` 和 profile 的 `bench` 按顺序传给官方解析器。只调整 seed 可写 `bench: [--seed, "7"]`。默认聊天请求地址为 `model.base_url` 追加 `/chat/completions`。默认保存官方汇总报告；在 `bench` 加 `--save-detailed` 可保留逐请求明细。每个 profile 的公共参数与专用参数合并后确定连接地址：显式 `--host` 或 `--port` 启用官方 host/port 模式，和显式 `--base-url` 同时出现会在发流前报错。SLO 支持单独覆盖任一阈值。阶段编排管理并发、请求数及发起速率；要自行指定这些参数，请选择 `single`。

完整模板含全部阶段、九项精度配置和 vLLM 0.11.0 的官方参数示例。容器内可从 `/opt/evaluation/docs/examples/vllm.full.conf` 复制。[查看全参数 YAML](docs/examples/vllm.full.conf)

## 安装

在项目根目录执行以下命令。构建机需已安装 Docker 并启动 daemon（可用 `docker info` 检查），且能访问系统软件源、Python 包源、GitHub 和评分资源下载站点。目标部署平台为 Linux amd64；其他架构构建或运行此镜像需要相应仿真支持。宿主机不需要安装 Python、CUDA 或模型权重。

```bash
docker build --platform linux/amd64 -t vllm-suite .
```

构建需要联网下载系统包、Python 依赖和官方评分代码。精度环境依赖版本由 `requirements.build.lock`、`requirements.cpu.lock` 和 `requirements.lock` 固定，历史报告重建环境由 `requirements.perf.lock` 固定；升级依赖时需更新对应锁文件并重新验证。系统包不保证逐字节可复现。镜像包含 CPU 版 PyTorch（精度框架依赖）、评分资源以及基础 shell/网络工具，不包含模型权重。HalluQA、FollowBench 上游仓库自带的小型数据随官方代码保留；其他数据放 NAS。

构建完成后先检查两个 Python 环境及离线评分资源：

```bash
docker run --rm --network none vllm-suite python /opt/evaluation/scripts/smoke_check.py
```

还可运行完整精度流程联调。该脚本在容器内创建临时数据与本地 Mock API，验证九项评测、输入冻结、中文 FollowBench 和错误处理，无需真实 API 密钥或挂载 NAS；生成的分数仅用于验证程序流程：

```bash
docker run --rm --network none vllm-suite python /opt/evaluation/scripts/integration_check.py
```

导出后可拷贝到离线机器：

```bash
set -o pipefail
docker save vllm-suite | gzip > vllm-suite.tar.gz
# 目标机器
gzip -dc vllm-suite.tar.gz | docker load
```

以上管道命令在 Bash 或 Zsh 中执行，`pipefail` 确保导出或导入任一步失败时返回非零退出码。导出失败的压缩包不能用于部署，应解决错误后重新导出。

一个归档包含压测和精度环境。部署时准备配置、NAS 上的 tokenizer 和数据集，以及本地 `outputs/`、`cache/`。配置可在下文按示例生成。导入镜像后也应执行上面的离线检查。离线指无需下载依赖和数据；正式评测仍需网络连通目标 API 及所需的裁判 API，不要给正式评测添加 `--network none`。

## 进入同一个容器，先跑压测再跑精度

完成上面的镜像构建后，在宿主机启动容器。下面假设 `config/` 已包含 `perf.json`、`suite.local.json` 和精度配置引用的数据集映射，NAS 中的 tokenizer 和数据集路径与两份配置一致。启动前在宿主机设置配置所用的 API 密钥环境变量。

```bash
mkdir -p /var/tmp/vllm-suite cache
docker run -it --name vllm-suite \
  -e OPENAI_API_KEY -e TARGET_API_KEY -e JUDGE_API_KEY -e HF_HUB_OFFLINE=1 \
  -v /实际NAS路径:/nas:ro -v "$PWD/config":/config:ro \
  -v /var/tmp/vllm-suite:/tmp/vllm-suite -v "$PWD/cache":/cache \
  vllm-suite bash
```

进入容器后执行一次：

```bash
perf-then-eval
```

它先按 `/config/perf.json` 跑完四阶段压测，退出码为 0 后才按 `/config/suite.local.json` 执行精度测试。任一任务失败时返回该任务的退出码。每次执行创建并打印一个 UUID 目录。压测与精度结果分别直接写入 `/tmp/vllm-suite/<UUID>/perf/` 和 `/tmp/vllm-suite/<UUID>/accuracy/`，上面的挂载让结果保存在宿主机 `/var/tmp/vllm-suite`。可用 `perf-then-eval --output /tmp/my-run` 指定容器内的本地结果根目录；要在删除容器后保留结果，将所选路径挂载到宿主机本地磁盘。自定义配置路径可用 `perf-then-eval /config/my-perf.json /config/my-accuracy.json`。压测失败时，已有压测结果保留在同一 UUID 目录中，精度阶段不会启动。若容器已在运行，可用 `docker exec -it vllm-suite bash` 进入。

## 准备 NAS 数据

从下载全部数据到启动容器和查看结果，请参阅[全量精度评测操作指南](docs/full-accuracy-evaluation.md)。

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

精度配置顶层可设置 `"concurrency": 8` 和 `"judge_concurrency": 4`，分别限制单次运行中目标模型和裁判同时进行的请求数。两个值必须是正整数，默认均为 1。任意请求完成即可补位。运行中的 `*.partial.jsonl` 按完成顺序保存序号和回答，子集完整后才按输入顺序发布正式 JSONL。读取输入或请求失败时，旁边的 `*.failure.json` 记录失败阶段（input/request）、从零开始的序号和错误类型；已派发的请求可继续结束，缺题子集不计分。结果写入失败会终止整次运行；即使启用 `continue_on_error`，也只在当前请求全部收尾且结果可记录后继续。

查看本次运行的退出码和本地运行目录中的 `summary.json`。成功退出码为 0，数据集失败或结果写入失败为 1，Ctrl-C 为 130，SIGTERM 为 143。中断时保留已经写入 `metrics.partial.json` 的子集成绩，尚未开始的数据集标记为 `not_run`。如果磁盘已满导致最终汇总无法更新，以退出码和标准错误为准，磁盘上的汇总可能仍显示运行中。

## 校验与运行

先确认宿主机已经挂载 NAS，并将命令中的 `/实际NAS路径` 替换为实际目录；Docker daemon 所在机器必须能访问该目录。NAS 中的数据与 tokenizer 需要对容器运行用户可读，目录需要可遍历；`outputs/`、`cache/` 需要可写。默认容器以 root 运行，启用 root squash 的 NAS 仍需授予相应读取权限。输出放在宿主机本地磁盘，NAS 只读存放输入资源。在 Linux 或 macOS 上启动新运行时，程序检查输出路径所在的挂载卷，拒绝 NFS、SMB 等常见网络文件系统。请在宿主机确认输出目录位于本地磁盘。

先校验数据加载、子集和提示词，不调用 API，也不需要真实密钥：

```bash
mkdir -p outputs cache
docker run --rm \
  -v /实际NAS路径:/nas:ro \
  -v "$PWD":/config:ro \
  -v "$PWD/cache":/cache \
  -v "$PWD/outputs":/outputs \
  vllm-suite eval-suite \
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
  vllm-suite eval-suite \
  --config /config/suite.local.json --output /outputs/run-001
```

首次运行时，`--output` 指定本地结果根目录，程序自动创建带时间戳和唯一后缀的子目录并打印路径。运行前会加载所有选定数据并生成提示词，缺文件或缺答案会提前失败。选定请求和参考答案保存在输出目录的 `inputs/` 中，设置只读权限并记录 SHA-256；生成与评分均使用这份快照，不再回读 NAS。请为这些输入及模型、裁判输出预留磁盘空间，不要修改运行目录。配置 `continue_on_error: true` 时某数据集运行失败仍继续后续数据集，最终返回非零退出码。SIGTERM / Ctrl-C 会更新已有汇总状态。输出目录创建后，setup 会逐步保存；准备或执行失败时，`summary.json.errors` 记录失败阶段、数据集和脱敏错误，`setup.snapshot_complete` 表示输入准备是否全部完成。配置文件无法解析或输出目录无法创建时，仅能通过退出码和终端错误定位。

使用真实数据的精度和压测任务在写入样本前，会按源文件、上级目录和输出目录的可读范围限制整个运行目录。可以确定共同用户组时保留组访问；目录型数据集、不同私有组或无法可靠换算的 ACL 默认仅运行账户可访问。冻结输入时只移除写权限。Linux bind mount 保留数字 UID/GID；要由宿主机账户读取私有结果，可在 Docker 命令中传入 `--user "$(id -u):$(id -g)"`，并确保该身份有权读取 NAS、写入输出与缓存。复制归档时应保留整个运行目录及其访问权限。

运行只需要连通目标和裁判 API。数据使用绝对本地路径，评分资源预置于镜像；不初始化模型或 tokenizer，不设置 API 地址白名单。所有提示词原样发送，不静默截断；LongBench 必须由足够上下文长度的远程模型承接。服务返回 `finish_reason: "length"` 时按实际生成的文本评分。服务错误和无效裁判输出会使该项失败，不计作正常低分。FinSafetyBench 裁判回复必须以唯一的 `Rating: [[1]]` 至 `Rating: [[10]]` 行结尾；示例中的数字不会被当作评分。

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
docker run --rm --network none vllm-suite python /opt/evaluation/scripts/smoke_check.py
```

## 性能压测（perf-suite）

镜像内的 `perf-suite` 通过官方 vLLM 0.11.0 serving benchmark 采样、发流、计时和计算指标。每次调用生成独立的官方终端报告和 JSON。四阶段运行先固定本次输入。正常退出或被中断后，均可用 `perf-suite --report-only --output /outputs/<运行目录>` 从本地计划与阶段记录重建汇总。重建按启动配置核对必跑轮次，从校验通过的官方 JSON 读取验收指标，按 run_id 识别运行级错误，保留 scan 当时提交的选档决定及运行级失败原因。联通按请求完成情况判定。scan 和 soak 按已提交的测量记录 SLO 验收结论；导出失败时仍可查看验收结果，已导出的文件另行核对归档完整性。

| 配置 | 用途 |
|---|---|
| `perf.example.json` | 四阶段编排。联通探测、扫描（默认每档 10 × C 个请求）、指定过载档三轮独立脉冲，以及按 `soak_requests` 计数的长稳测试。 |
| `perf.native.example.json` | 单次官方 benchmark，所有参数放在 `vllm_bench_serve`。 |

构建命令为 `docker build --platform linux/amd64 -t vllm-suite .`。配置、NAS 挂载、启动命令、选档规则、报告位置和旧配置迁移见[容器压测操作指南](docs/performance-testing.md)。
