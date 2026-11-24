# vllm-suite 开发约定

项目提供远程精度评测和性能压测。主要入口为 `scripts/run_suite.py` 与 `scripts/run_perf.py`。保持现有配置、报告和历史运行目录的兼容性，涉及行为变化时同步更新 README 与示例配置。

## 输入、输出与部署

- 统一入口 `vllm-suite` 在容器内读当前目录的 YAML `.vllm.conf`。model 共享，judge 属于 execution.accuracy。每个已配置任务启用，perf 成功才进入 accuracy。根目录保存 UUID、状态和脱敏后的有效配置。维护原生 argv 顺序、阶段 barrier、默认数据映射与 docs/examples/vllm.full.conf。

- 数据集、tokenizer 和配置是输入。NAS 使用只读挂载，tokenizer 使用已有 ModelScope 下载产物的目录与文件布局，项目不负责下载。
- 独立入口正常运行的 `--output` 是本地结果根目录，每次自动创建时间戳运行子目录并打印路径。精度、分阶段压测和原生压测都遵循该规则。
- 分阶段压测的 `--report-only --output` 指向已有运行子目录。保留既有报告、请求日志和可恢复的发布流程。
- 默认构建一个 vllm-suite 镜像，同时提供官方 vLLM 0.11.0 压测客户端和 OpenCompass，容器内用 `perf-then-eval` 串行执行两项任务。四阶段用 `perf.example.json`，单次用 `perf.native.example.json`。官方参数集中在 `vllm_bench_serve`。原生导出保存在对应运行目录的 `exports/`。
- 容器运行时 NAS 挂载到 `/nas:ro`，独立入口的示例将本地输出挂载到 `/outputs`。串行入口默认使用 `/tmp/vllm-suite`，支持 `--output DIR`。每次串行执行创建一个 UUID 目录，两阶段直接写入其下的 perf/ 和 accuracy/。配置和密钥通过配置文件与环境变量传入。README 中的宿主机与容器路径必须各自可用。
- 性能报告需保留 vLLM 原生输出。数据集、长度档和 seed 可配置。SQLite 用于旧分阶段报告的离线重建。分阶段在发流前发布输入副本或固定存储快照，再写根 setup.json 计划；阶段 summary.json 先提交测量再提交导出，根报告从磁盘重建。输入按整次运行绑定，重建时校验官方产物摘要，保留运行级错误和请求计数确认状态。轮次按 run_id 与 sequence 核对；scan 完成后提交 selection.json，soak 和报告重建只读取历史选档决定。多档按阶段整体推进，每次官方调用结束并写入结果后再进入下一次。请求完成、SLO 验收及归档完整性分别判定。TPOT 验收需要有效样本和可用的官方指标。导出文件使用相对路径及摘要校验。

## 验证

- 本地性能环境安装 `requirements.perf.lock` 后运行 `python -m unittest discover -s tests -v`。缺少依赖时部分测试会跳过，报告验证范围。
- 离线和 Docker 相关修改应覆盖镜像构建、只读 NAS、时间戳目录、实际运行路径及离线报告重建。资源不足时明确说明未执行的检查。
- 修改目录或配置行为时，检查 `scripts/integration_check.py`、`tests/host_mock/`、README 中的命令和源码包是否一致。
- 当前源码包为 `vllm-suite.zip`，项目文件更新后同步并核对包内文件。不要把运行产物、缓存、密钥或本地配置打入包中。

## 当前已知限制

- 新运行对常见网络文件系统有拒绝检查，尚未覆盖所有远程挂载类型。报告重建入口目前不检查文件系统类型。部署时应自行确认输出所在磁盘为本地存储。
- 原生镜像需完整 vLLM 环境。没有实际构建和运行该镜像时，不应将模拟服务或 Python 测试当作镜像部署验收。
