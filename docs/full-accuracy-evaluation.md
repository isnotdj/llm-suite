# 全量精度评测操作指南

容器内可用统一入口 `vllm-suite` 读取当前目录的 `.vllm.conf`，同一次执行共用一个 UUID 结果目录。[运行示例](../README.md#在已启动的容器内运行) · [全参数配置](examples/vllm.full.conf)

本文覆盖项目支持的九项精度评测：准备数据、挂载 NAS、配置远程模型、校验和运行。数据下载需联网；正式评测无需下载依赖，但容器必须能访问目标模型和裁判模型的 OpenAI 兼容 API。

镜像内置 OpenCompass 和官方 vLLM 远程压测客户端。在容器内执行 `eval-suite` 可以单独评测精度，执行 `perf-then-eval` 会先压测再评测精度。单独评测精度只需挂载科目数据，压测还需本地 tokenizer。

## 1. 准备 NAS 数据

在联网且能写入 NAS 的机器上操作。以下 `/mnt/nas` 是示例，请换成实际路径；如果评测机器不能联网，可在联网机器下载后复制到 NAS。

```bash
export DATA_ROOT=/mnt/nas/vllm-suite/datasets
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"
```

依次下载。`wget`、`unzip`、`git` 需要在下载机器上可用；下载失败应先解决再继续。以下上游数据可能更新，正式比较时应记录下载版本并固定输入。

### C-Eval

[官方数据与下载说明](https://github.com/hkust-nlp/ceval/blob/main/README.md)。

```bash
wget -O ceval-exam.zip https://huggingface.co/datasets/ceval/ceval-exam/resolve/3923b519fd180e689d0961bf3a032ece929742f3/ceval-exam.zip
unzip ceval-exam.zip -d ceval-exam
ls ceval-exam/{dev,val,test}
```

### LogiQA 中文 v1

```bash
git clone --depth 1 https://github.com/lgw863/LogiQA-dataset.git logiqa-zh
ls logiqa-zh/zh_{train,eval,test}.txt
```

### MathBench 2024

使用 [OpenCompass 提供的 MathBench 数据包](https://github.com/open-compass/opencompass/blob/main/opencompass/utils/datasets_info.py)：

```bash
wget -O mathbench_v1.zip https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mathbench_v1.zip
unzip mathbench_v1.zip -d .
ls mathbench_v1/{college,high}
```

### CMRC2018

使用官方带答案的 SQuAD 格式开发集：

```bash
mkdir -p cmrc2018
wget -O cmrc2018/cmrc2018_dev.json https://raw.githubusercontent.com/ymcui/cmrc2018/master/squad-style-data/cmrc2018_dev.json
```

### LongBench v1

使用 [LongBench v1 官方 JSONL 数据](https://huggingface.co/datasets/THUDM/LongBench)，不是 v2：

```bash
mkdir -p longbench
wget -O longbench/data.zip https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
unzip longbench/data.zip -d longbench
ls longbench/data/*.jsonl
```

### FinSafetyBench

```bash
mkdir -p finsafetybench
wget -O finsafetybench/FinsafetyBenchmark.csv https://raw.githubusercontent.com/sustech-nlp/FinSafetyBench/main/data/FinsafetyBenchmark.csv
```

### FinanceIQ

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/Duxiaoman-DI/XuanYuan.git xuanyuan-download
git -C xuanyuan-download sparse-checkout set FinanceIQ/data
cp -a xuanyuan-download/FinanceIQ/data financeiq
ls financeiq/{dev,test}
```

以上命令从上游拉取数据，不会自动清理下载的压缩包和仓库。解压后如果目录层级与下表不同，应以实际数据根目录为准修改映射，不要改变数据集内部文件结构。

| 逻辑名称 | 相对 `$DATA_ROOT` 的路径 | 必需结构 |
|---|---|---|
| `ceval` | `ceval-exam` | `dev/`、`val/`、`test/` 内为各科 CSV |
| `logiqa-zh` | `logiqa-zh` | `zh_train.txt`、`zh_eval.txt`、`zh_test.txt` |
| `mathbench` | `mathbench_v1` | `college/`、`high/` 等子目录及 JSONL |
| `cmrc2018` | `cmrc2018/cmrc2018_dev.json` | 带答案的 dev JSON 文件 |
| `longbench` | `longbench/data` | 21 个 LongBench v1 子任务 JSONL |
| `finsafetybench` | `finsafetybench/FinsafetyBenchmark.csv` | 官方 CSV 文件 |
| `financeiq` | `financeiq` | `dev/`、`test/` 内为各类别 CSV |

HalluQA 和 FollowBench 的官方数据已随镜像提供，可直接引用，无需放 NAS。使用自备数据时也可把它们放在 NAS 并修改下方映射。

## 2. 启动容器并挂载目录

在 Docker 宿主机上选择一个可写的本地磁盘目录。示例使用 `/var/tmp/vllm-suite`，后续宿主机命令请在同一个 shell 中执行。

```bash
export RUN_ROOT=/var/tmp/vllm-suite
mkdir -p "$RUN_ROOT/config" "$RUN_ROOT/outputs" "$RUN_ROOT/cache"
```

在包含本指南和并发实现的项目根目录构建本地镜像，并用同一标签验收。

```bash
docker build --platform linux/amd64 -t vllm-suite .
docker run --rm --network none vllm-suite python /opt/evaluation/scripts/integration_check.py
```

确认宿主机已挂载 NAS，再启动容器。将 `/mnt/nas/vllm-suite/datasets` 改成实际路径；宿主机 NAS 数据必须对容器运行用户可读。

```bash
docker run -dit \
  --name vllm-suite \
  -v /mnt/nas/vllm-suite/datasets:/nas:ro \
  -v "$RUN_ROOT/config":/config:ro \
  -v "$RUN_ROOT/outputs":/outputs \
  -v "$RUN_ROOT/cache":/cache \
  vllm-suite \
  bash
```

如使用自行构建的镜像，将镜像名替换为实际标签。已有同名容器时请换名或先处理旧容器。NAS 和配置只读挂载；`outputs`、`cache` 必须位于本地磁盘且可写。程序会为每次评测在输出根目录下创建带时间戳及唯一后缀的运行目录，不修改输入。

## 3. 配置数据集

在宿主机工作目录创建 `config/datasets.local.json`：

```bash
cat > "$RUN_ROOT/config/datasets.local.json" <<'JSON'
{
  "version": 1,
  "root": "/nas",
  "datasets": {
    "ceval": "ceval-exam",
    "logiqa-zh": "logiqa-zh",
    "mathbench": "mathbench_v1",
    "cmrc2018": "cmrc2018/cmrc2018_dev.json",
    "longbench": "longbench/data",
    "finsafetybench": "finsafetybench/FinsafetyBenchmark.csv",
    "financeiq": "financeiq",
    "halluqa": "/opt/benchmarks/HalluQA/HalluQA.json",
    "followbench_en": "/opt/benchmarks/FollowBench/data",
    "followbench_zh": "/opt/benchmarks/FollowBench/data_zh"
  }
}
JSON
```

前七项是相对容器内 `/nas` 的路径；后三项为镜像内的绝对路径。按实际目录调整。

## 4. 配置模型和测试范围

在宿主机创建 `config/suite.local.json`，然后修改目标和裁判模型的地址与模型名：

```bash
cat > "$RUN_ROOT/config/suite.local.json" <<'JSON'
{
  "dataset_map": "/config/datasets.local.json",
  "target": {
    "base_url": "http://目标模型地址:8000/v1",
    "model": "目标模型实际名称",
    "api_key_env": "TARGET_API_KEY",
    "temperature": 0,
    "max_tokens": 2048,
    "timeout": 600
  },
  "judge": {
    "base_url": "http://裁判模型地址:8000/v1",
    "model": "裁判模型实际名称",
    "api_key_env": "JUDGE_API_KEY",
    "temperature": 0,
    "max_tokens": 2048,
    "timeout": 600
  },
  "datasets": [
    "ceval", "logiqa-zh", "mathbench", "cmrc2018", "longbench",
    "finsafetybench", "financeiq", "halluqa", "followbench"
  ],
  "options": {
    "finsafetybench": {"language": "both"}
  },
  "followbench": {"language": "zh"},
  "continue_on_error": true
}
JSON
```

`base_url` 填到 `/v1`，模型名称须与服务暴露的名称一致；密钥通过环境变量传入。HalluQA、FollowBench、FinSafetyBench 需要裁判 API。容器内 `localhost` 指容器自身；服务地址需要能从容器访问。

不设置 `limit` 和 `subsets` 即选择所列评测的全部样本和子集。FinSafetyBench 使用中英文直接提问基线；FollowBench 一次只能选 `zh` 或 `en`。若两种语言都要测，可先按上述配置运行，再复制一份配置，将 `datasets` 设为 `["followbench"]`、`followbench.language` 改为 `en`，单独运行第二次。

LongBench 会发送长上下文，目标模型必须支持对应上下文长度；程序不会自动截断。服务因达到生成长度上限返回 `length` 时，精度测试使用已生成文本正常评分，裁判回复仍须满足题目的评分格式。全量评测可能耗时较长并产生大量目标和裁判请求，请提前确认服务容量和配额。

精度配置顶层可设置 `"concurrency": 8` 和 `"judge_concurrency": 4`，分别限制单次运行中目标模型和裁判同时进行的请求数。两个值必须是正整数，默认均为 1。任意请求完成即可补位。运行中的 `*.partial.jsonl` 按完成顺序保存序号和回答，子集完整后才按输入顺序发布正式 JSONL。读取输入或请求失败时，旁边的 `*.failure.json` 记录失败阶段（input/request）、从零开始的序号和错误类型；已派发的请求可继续结束，缺题子集不计分。结果写入失败会终止整次运行；即使启用 `continue_on_error`，也只在当前请求全部收尾且结果可记录后继续。

查看本次运行的退出码和本地运行目录中的 `summary.json`。成功退出码为 0，数据集失败或结果写入失败为 1，Ctrl-C 为 130，SIGTERM 为 143。中断时保留已经写入 `metrics.partial.json` 的子集成绩，尚未开始的数据集标记为 `not_run`。如果磁盘已满导致最终汇总无法更新，以退出码和标准错误为准，磁盘上的汇总可能仍显示运行中。

## 5. 校验并运行

进入容器：

```bash
docker exec -it vllm-suite bash
```

先检查数据加载、子集和提示词；此步骤不调用 API，也不需要真实密钥：

```bash
eval-suite --config /config/suite.local.json \
  --output /outputs/accuracy-check --validate-only
```

校验通过后，在容器中设置密钥并启动正式评测：

```bash
export TARGET_API_KEY='目标模型密钥'
export JUDGE_API_KEY='裁判模型密钥'

eval-suite --config /config/suite.local.json \
  --output /outputs/accuracy-full
```

数据集串行执行；`continue_on_error: true` 时某项失败仍会继续执行后续项，最终返回非零退出码。校验通过仅说明本地准备成功，正式运行还需要 API 连通并正常返回。

HalluQA 裁判运行中收到 Ctrl-C 或 SIGTERM 时，父进程先停止并回收裁判进程组，再将仍处于 `running` 的 `halluqa/score/run.json` 标为 `interrupted`，并更新总报告。若输出盘写入失败，进程仍会被回收，终端会显示写入错误。

## 6. 查看结果

终端会打印新建的运行目录，其路径位于容器内 `/outputs/accuracy-full/`；在宿主机对应 `$RUN_ROOT/outputs/accuracy-full/`。保留完整运行目录：

- `summary.json`：整体状态、指标、配置快照和错误信息。
- 各子集下的 `samples.jsonl`：提示词、参考答案和模型输出。
- 各子集下的 `metrics.json`：该子集指标。
- 失败时查看相应数据集目录的 `error.txt`，以及 `summary.json` 的 `errors`。

运行使用冻结的输入快照；输出包含样本及模型回答，应按数据权限管理。更多结果字段及评分口径见 [README](../README.md)。
