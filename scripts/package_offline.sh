#!/usr/bin/env bash
# Build self-contained runtime images and export them for offline import.
set -euo pipefail

usage() {
  echo "Usage: $0 OUTPUT_DIRECTORY [suite|native|all]"
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi
output=$1
mode=${2:-all}
case "$mode" in
  suite|native|all) ;;
  *) usage >&2; exit 2 ;;
esac
if [[ -e "$output" ]]; then
  echo "Output already exists: $output" >&2
  exit 1
fi
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_args=()
if [[ -n ${PIP_INDEX_URL:-} ]]; then
  build_args+=(--secret id=pip_index_url,env=PIP_INDEX_URL)
fi
if [[ -n ${TORCH_INDEX_URL:-} ]]; then
  build_args+=(--secret id=torch_index_url,env=TORCH_INDEX_URL)
fi
images=()
if [[ "$mode" == suite || "$mode" == all ]]; then
  if [[ ${VLLM_SUITE_USE_EXISTING:-0} != 1 ]]; then
    DOCKER_BUILDKIT=1 docker build --platform linux/amd64 "${build_args[@]}" -t vllm-suite:opencompass "$repo"
  fi
  images+=(vllm-suite:opencompass)
fi
if [[ "$mode" == native || "$mode" == all ]]; then
  if [[ ${VLLM_SUITE_USE_EXISTING:-0} != 1 ]]; then
    DOCKER_BUILDKIT=1 docker build --platform linux/amd64 "${build_args[@]}" -f "$repo/Dockerfile.native" -t vllm-suite:native-bench "$repo"
  fi
  images+=(vllm-suite:native-bench)
fi
for image in "${images[@]}"; do
  platform=$(docker image inspect --format '{{.Os}}/{{.Architecture}}' "$image")
  if [[ "$platform" != linux/amd64 ]]; then
    echo "Image $image is $platform; expected linux/amd64" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$output")"
mkdir "$output"
output=$(cd "$output" && pwd)
printf '%s\n' "Preparing $mode images in $output"
for image in "${images[@]}"; do
  case "$image" in
    vllm-suite:opencompass) archive=vllm-suite-opencompass.tar.gz ;;
    vllm-suite:native-bench) archive=vllm-suite-native-bench.tar.gz ;;
  esac
  docker save "$image" | gzip > "$output/$archive"
done
cp "$repo"/README.md "$repo"/dataset-map.example.json "$repo"/suite.example.json \
   "$repo"/perf.example.json "$repo"/perf.native.example.json "$output/"
cp "$repo"/scripts/load_offline.sh "$output/"
(
  cd "$output"
  if command -v sha256sum >/dev/null; then
    sha256sum ./*.tar.gz > SHA256SUMS
  else
    shasum -a 256 ./*.tar.gz > SHA256SUMS
  fi
)
echo "Offline bundle prepared: $output"
