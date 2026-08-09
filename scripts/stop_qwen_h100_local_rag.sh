#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
lifecycle="$repo_root/scripts/qwen_h100_local_rag.sh"
mode=${1:---stop}

case "$mode" in
  --stop)
    "$lifecycle" stop
    echo "Qwen H100 services stopped; containers, named volumes, and caches were retained."
    ;;
  --down)
    "$lifecycle" down
    echo "Qwen H100 containers and Compose network removed; named volumes and caches were retained."
    ;;
  *)
    echo "Usage: scripts/stop_qwen_h100_local_rag.sh [--stop|--down]" >&2
    exit 2
    ;;
esac
