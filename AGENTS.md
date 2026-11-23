# vllm-suite 开发约定

项目提供远程精度评测和性能压测。主要入口为 `scripts/run_suite.py` 与 `scripts/run_perf.py`。保持现有配置、报告和历史运行目录的兼容性，涉及行为变化时同步更新 README 与示例配置。

## 输入、输出与部署

- 数据集、tokenizer 和配置是输入。NAS 使用只读挂载，tokenizer 使用已有 ModelScope 下载产物的目录与文件布局，项目不负责下载。
- 正常运行的 `--output` 是本地结果根目录，每次自动创建时间戳运行子目录并打印路径。精度、分阶段压测和原生压测都遵循该规则。
- 分阶段压测的 `--report-only --output` 指向已有运行子目录。保留既有报告、请求日志和可恢复的发布流程。
- 分阶段压测使用 `perf.example.json`，原生 vLLM 0.11.0 参数集中在 `perf.native.example.json` 的 `vllm_bench_serve` 内。原生导出保存在对应运行目录的 `exports/`。
- 容器运行时 NAS 挂载到 `/nas:ro`，本地输出挂载到 `/outputs`。配置和密钥通过配置文件与环境变量传入。README 中的宿主机与容器路径必须各自可用。
- 性能报告需保留 vLLM 原生输出。数据集、长度档和 seed 可配置。SQLite 用于分阶段性能指标的磁盘汇总。

## 验证

- 本地性能环境安装 `requirements.perf.lock` 后运行 `python -m unittest discover -s tests -v`。缺少依赖时部分测试会跳过，报告验证范围。
- 离线和 Docker 相关修改应覆盖镜像构建、只读 NAS、时间戳目录、实际运行路径及离线报告重建。资源不足时明确说明未执行的检查。
- 修改目录或配置行为时，检查 `scripts/integration_check.py`、`tests/host_mock/`、README 中的命令和源码包是否一致。
- 当前源码包为 `vllm-suite.zip`，项目文件更新后同步并核对包内文件。离线镜像包由 `scripts/package_offline.sh` 生成，导入前校验摘要。不要把运行产物、缓存、密钥或本地配置打入包中。

## 当前已知限制

- 新运行对常见网络文件系统有拒绝检查，尚未覆盖所有远程挂载类型。报告重建入口目前不检查文件系统类型。部署时应自行确认输出所在磁盘为本地存储。
- Python 锁文件按 2026 年 4 月底发布截止解析，变更依赖时验证 Linux amd64 / Python 3.12 和对应上游约束。原生镜像需完整 vLLM 环境。没有实际构建和运行该镜像时，不应将模拟服务或 Python 测试当作镜像部署验收。
