FROM vllm/vllm-openai:v0.11.0 AS native-bench
RUN python3 -m pip install --no-cache-dir 'vllm[bench]==0.11.0'
COPY scripts/ /opt/evaluation/scripts/
COPY perf.native.example.json /opt/evaluation/
RUN HF_HUB_OFFLINE=1 python3 /opt/evaluation/scripts/run_perf.py \
    --config /opt/evaluation/perf.native.example.json --output /tmp/native-parser-check --validate-only
ENTRYPOINT ["python3", "/opt/evaluation/scripts/run_perf.py"]
CMD ["--help"]

FROM ubuntu:24.04

LABEL org.opencontainers.image.title="vllm-suite" \
      org.opencontainers.image.description="Remote LLM accuracy evaluation and performance testing"

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCOMPASS_REV=f9d399363b4e00dbd5ada33096ca270969310274
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG FOLLOWBENCH_REV=6278f4c1377b4eafab737267b8b21acd52ea0e52
ARG HALLUQA_REV=7e5c67398d7fee013c37ec1d6afafa76f9836180

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=Asia/Shanghai \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH \
    MODELSCOPE_CACHE=/cache/modelscope \
    HF_HOME=/cache/huggingface \
    COMPASS_DATA_CACHE=/cache/opencompass \
    DATASET_WORK_CACHE=/cache/dataset-work \
    TIKTOKEN_CACHE_DIR=/opt/data/tiktoken \
    NLTK_DATA=/opt/data/nltk

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev build-essential pkg-config \
      ca-certificates tzdata locales git git-lfs curl wget openssh-client \
      bash bash-completion vim-tiny less jq unzip zip xz-utils tar rsync \
      iproute2 iputils-ping dnsutils net-tools netcat-openbsd socat \
      traceroute mtr-tiny tcpdump iperf3 ethtool procps psmisc lsof \
      htop sysstat strace file tree ripgrep tini libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv

COPY requirements.build.lock requirements.cpu.lock requirements.lock /opt/evaluation/
RUN python -m pip install --no-cache-dir --no-deps -r /opt/evaluation/requirements.build.lock

# Remote-only execution; CPU torch is required by OpenCompass utilities.
RUN mkdir -p /opt/benchmarks/OpenCompass \
    && curl -fL --retry 3 "https://codeload.github.com/open-compass/opencompass/tar.gz/${OPENCOMPASS_REV}" | tar -xz --strip-components=1 -C /opt/benchmarks/OpenCompass \
    && echo "${OPENCOMPASS_REV}" > /opt/benchmarks/OpenCompass/REVISION \
    && sed -i '/^-e git+/c\rouge-chinese==1.0.3' /opt/benchmarks/OpenCompass/requirements/runtime.txt \
    && python -m pip install --no-cache-dir --no-deps -r /opt/evaluation/requirements.cpu.lock --index-url "${TORCH_INDEX_URL}"
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation -r /opt/evaluation/requirements.lock \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation /opt/benchmarks/OpenCompass \
    && python -m pip check

RUN NLTK_ALLOW_PROXIED_URLOPEN=1 python -c "import nltk; [nltk.download(r, download_dir='/opt/data/nltk', raise_on_error=True) for r in ('punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger_eng', 'wordnet', 'omw-1.4')]"

RUN python -c "import tiktoken; [tiktoken.get_encoding(n) for n in ('gpt2','r50k_base','p50k_base','p50k_edit','cl100k_base','o200k_base')]"

WORKDIR /opt/evaluation
COPY datasets.json /opt/evaluation/datasets.json
COPY scripts/ /opt/evaluation/scripts/
COPY dataset-map.example.json suite.example.json README.md /opt/evaluation/
RUN chmod +x scripts/dataset_path.py scripts/run_suite.py \
    && ln -s /opt/evaluation/scripts/dataset_path.py /usr/local/bin/dataset-path \
    && ln -s /opt/evaluation/scripts/run_suite.py /usr/local/bin/eval-suite

# Keep official data, scoring scripts, prompts and licenses at fixed commits.
RUN mkdir -p /opt/benchmarks /opt/data /data /cache /workspace /outputs \
    && mkdir -p /opt/benchmarks/FollowBench /opt/benchmarks/HalluQA \
    && curl -fL --retry 3 "https://codeload.github.com/YJiangcm/FollowBench/tar.gz/${FOLLOWBENCH_REV}" | tar -xz --strip-components=1 -C /opt/benchmarks/FollowBench \
    && echo "${FOLLOWBENCH_REV}" > /opt/benchmarks/FollowBench/REVISION \
    && curl -fL --retry 3 "https://codeload.github.com/OpenMOSS/HalluQA/tar.gz/${HALLUQA_REV}" | tar -xz --strip-components=1 -C /opt/benchmarks/HalluQA \
    && echo "${HALLUQA_REV}" > /opt/benchmarks/HalluQA/REVISION \
    && test -d /opt/benchmarks/FollowBench/data \
    && test -d /opt/benchmarks/FollowBench/data_zh \
    && test -s /opt/benchmarks/HalluQA/HalluQA.json \
    && test -s /opt/benchmarks/HalluQA/HalluQA_mc.json \
    && python scripts/patch_benchmarks.py /opt/benchmarks \
    && eval-suite --help >/dev/null \
    && python scripts/halluqa_eval.py --help >/dev/null \
    && python /opt/benchmarks/FollowBench-api/code/llm_eval.py --help >/dev/null \
    && python /opt/benchmarks/FollowBench-api/code_zh/llm_eval.py --help >/dev/null \
    && mkdir -p /opt/benchmarks/CMRC2018 /opt/benchmarks/FinSafetyBench \
    && curl -fL --retry 3 https://raw.githubusercontent.com/ymcui/cmrc2018/c0eb1b6ba219847457e6af3180da722bbeb656af/squad-style-data/cmrc2018_evaluate.py -o /opt/benchmarks/CMRC2018/official.py \
    && curl -fL --retry 3 https://raw.githubusercontent.com/sustech-nlp/FinSafetyBench/8dcdf791ac2be782d72f31d4f8e8ad44be54b6f5/judge.py -o /opt/benchmarks/FinSafetyBench/judge.py \
    && python -m pip freeze > /opt/evaluation/requirements.resolved.txt

WORKDIR /workspace
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]

# Independent CPU-only benchmark client; no vLLM serving engine/CUDA install.
COPY requirements.perf.lock perf.example.json /opt/evaluation/
RUN python3.12 -m venv /opt/perf-venv \
    && /opt/perf-venv/bin/pip install --no-cache-dir -r /opt/evaluation/requirements.perf.lock \
    && /opt/perf-venv/bin/pip check \
    && chmod +x /opt/evaluation/scripts/run_perf.py \
    && ln -s /opt/evaluation/scripts/run_perf.py /usr/local/bin/perf-suite \
    && /opt/perf-venv/bin/python -c "import sys; sys.path.insert(0,'/opt/evaluation/scripts'); from perf_native_report import upstream_report; upstream_report()" \
    && perf-suite --help >/dev/null
