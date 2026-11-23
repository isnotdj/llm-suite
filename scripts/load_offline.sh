#!/usr/bin/env bash
# Verify and import every runtime image listed in an offline bundle.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ -s SHA256SUMS ]] || { echo "Missing image checksums" >&2; exit 1; }
images=()
while read -r checksum image extra; do
  [[ -z "${extra:-}" && "$checksum" =~ ^[0-9a-fA-F]{64}$ ]] || { echo "Invalid checksum manifest" >&2; exit 1; }
  case "$image" in
    ./vllm-suite-opencompass.tar.gz|./vllm-suite-native-bench.tar.gz) ;;
    *) echo "Invalid image filename in manifest: $image" >&2; exit 1 ;;
  esac
  [[ -f "$image" ]] || { echo "Missing image: $image" >&2; exit 1; }
  images+=("$image")
done < SHA256SUMS
[[ ${#images[@]} -gt 0 ]] || { echo "No image archives in checksum manifest" >&2; exit 1; }
sha256sum -c SHA256SUMS
for image in "${images[@]}"; do
  gzip -dc "$image" | docker load
done
