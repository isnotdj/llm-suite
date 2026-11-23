#!/bin/sh
# BuildKit mounts optional index URLs for this build step only.
set -eu
secret_used=0
if [ -r /run/secrets/pip_index_url ]; then
    PIP_INDEX_URL=$(cat /run/secrets/pip_index_url)
    export PIP_INDEX_URL
    secret_used=1
fi
if [ -r /run/secrets/torch_index_url ]; then
    secret_used=1
fi
if [ "$secret_used" -eq 0 ]; then
    exec "$@"
fi
# pip can print URLs in errors. Keep credential-bearing output out of build logs.
log=$(mktemp)
trap 'rm -f "$log"' EXIT
if "$@" >"$log" 2>&1; then
    exit 0
else
    status=$?
    echo "Dependency installation failed; build output is suppressed because an index secret was supplied." >&2
    exit "$status"
fi
