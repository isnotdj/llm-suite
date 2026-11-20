# Host FastAPI integration check

Run from the repository root, with Python 3.12:

```sh
python3.12 -m venv /tmp/vllm-suite-host-mock-venv
/tmp/vllm-suite-host-mock-venv/bin/pip install -r tests/host_mock/requirements.txt
/tmp/vllm-suite-host-mock-venv/bin/python -m pip check
/tmp/vllm-suite-host-mock-venv/bin/python -m unittest discover -s tests -v
/tmp/vllm-suite-host-mock-venv/bin/python tests/host_mock/run.py --output artifacts/host-mock-run
```

The output directory must not already exist. The runner creates a local character
tokenizer (no model download), starts a separate Uvicorn/FastAPI process on an
available loopback port, and runs the unmodified production `run_perf.py` CLI.
The server is stopped in `finally`, including assertion failures.

Both random and synthetic ShareGPT fixtures exercise 4k/16k/64k input profiles,
connectivity, concurrency scan, saturation selection, overload, recovery and a
short soak. A one-slot semaphore creates real server-side queueing. SSE responses
contain 16 naturally finished tokens regardless of the requested output target.
The shortened timing and relaxed test SLO are fixture settings only; production
defaults remain unchanged. These checks validate control flow, not real model
performance, semantic accuracy, a 2-hour stability claim or 30-minute drift.

Fault checks use the production transport against HTTP 429/503, first-token and
stream-idle timeouts, length termination, missing DONE, missing finish_reason,
invalid usage, absent usage, input token mismatch, and DONE before HTTP EOF.
A live 503 load verifies the no-success circuit breaker and failed overload
acceptance. `verification.json` records the final result, `faults.json` contains
request-level observations, and each dataset folder contains the usual summary,
SQLite metrics, JSONL request/window traces, and the vLLM text/JSON reports.
`vllm-report.txt` contains the measured stage reports and `vllm-reports.json`
indexes all stage reports. `server.log` captures
the ASGI process output.

To run only the server after creating a fixture tokenizer:

```sh
MOCK_TOKENIZER="$PWD/artifacts/host-mock-run/tokenizer" \
  /tmp/vllm-suite-host-mock-venv/bin/python -m uvicorn server:app \
  --app-dir tests/host_mock --host 127.0.0.1 --port 18080
```

`GET /health` exposes counters; `POST /v1/chat/completions` accepts `mock_mode`
(default `normal`). This fixture has no authentication and must stay on loopback.
