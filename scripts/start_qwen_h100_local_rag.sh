#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
lifecycle="$repo_root/scripts/qwen_h100_local_rag.sh"
timeout_seconds=${QWEN_START_TIMEOUT_SECONDS:-1800}
poll_seconds=${QWEN_START_POLL_SECONDS:-10}
skip_pull=${QWEN_SKIP_PULL:-0}

if ! [[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "QWEN_START_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "QWEN_START_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ "$skip_pull" != "0" && "$skip_pull" != "1" ]]; then
  echo "QWEN_SKIP_PULL must be 0 or 1" >&2
  exit 2
fi

wait_for_url() {
  local label=$1
  local url=$2
  local deadline=$3

  until curl --fail --silent --show-error --output /dev/null "$url"; do
    if (( SECONDS >= deadline )); then
      echo "Timed out waiting for $label at $url" >&2
      "$lifecycle" ps >&2 || true
      return 1
    fi
    sleep "$poll_seconds"
  done
  echo "$label is ready: $url"
}

cd "$repo_root"
"$lifecycle" validate
if [[ "$skip_pull" == "0" ]]; then
  "$lifecycle" pull
fi
"$lifecycle" up "$@"

deadline=$((SECONDS + timeout_seconds))
wait_for_url "Qwen" "http://127.0.0.1:8999/v1/models" "$deadline"
wait_for_url \
  "RAG server" \
  "http://127.0.0.1:8081/v1/health?check_dependencies=true" \
  "$deadline"
wait_for_url \
  "Ingestor server" \
  "http://127.0.0.1:8082/v1/health?check_dependencies=true" \
  "$deadline"
wait_for_url "RAG frontend" "http://127.0.0.1:8090/" "$deadline"

"$lifecycle" ps
echo "Qwen H100 Local RAG Deployment is ready on loopback ports 8999, 8081, 8082, and 8090."
