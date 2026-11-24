# Official benchmark loopback integration

在有 vLLM 0.11.0 benchmark 依赖、FastAPI 和 Uvicorn 的环境中，从项目根目录执行：

```sh
python tests/host_mock/run.py --output /tmp/vllm-native-mock-run
```

输出目录必须尚未创建。脚本创建本地 tokenizer、启动回环 SSE 服务，再从生产 CLI 依次执行联通检查、两档扫描、三轮过载和按请求数的 soak。它核对服务实际收到的请求总数、每轮官方报告和详细 JSON；不连接真实模型。
