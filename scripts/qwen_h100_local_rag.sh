#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
compose_dir="$repo_root/deploy/compose"
variant=${QWEN_VARIANT:-fp8}

# Compose reads this file too, but exporting it here keeps lifecycle commands and
# their preflight checks aligned with the deployment source of truth.
set -a
# shellcheck disable=SC1091
source "$compose_dir/.env"
set +a

if [[ "$variant" != "fp8" && "$variant" != "nvfp4" ]]; then
  echo "QWEN_VARIANT must be fp8 or nvfp4" >&2
  exit 2
fi

compose_args=(
  docker compose
  --env-file "$compose_dir/.env"
  -f "$compose_dir/nims.yaml"
  -f "$compose_dir/vectordb.yaml"
  -f "$compose_dir/docker-compose-ingestor-server.yaml"
  -f "$compose_dir/docker-compose-rag-server.yaml"
  -f "$compose_dir/docker-compose-qwen-h100.yaml"
)
if [[ "$variant" == "nvfp4" ]]; then
  compose_args+=(
    -f "$compose_dir/docker-compose-qwen-h100-nvfp4.yaml"
  )
fi
compose_args+=(--profile qwen-h100)

require_ngc_key() {
  if [[ -z "${NGC_API_KEY:-}" ]]; then
    echo "NGC_API_KEY is required for NVIDIA NIM services." >&2
    echo "Set it in deploy/compose/.env, then retry; do not commit the secret." >&2
    exit 3
  fi
}

validate_general_config() {
  NGC_API_KEY=${NGC_API_KEY:-config-validation-only} \
    USERID=${USERID:-$(id -u)} \
    "${compose_args[@]}" config --format json |
    uv run python "$repo_root/scripts/qwen_h100_local_rag.py" \
      validate-config -
}

validate_certified_config() {
  NGC_API_KEY=${NGC_API_KEY:-config-validation-only} \
    USERID=${USERID:-$(id -u)} \
    "${compose_args[@]}" config --format json |
    uv run python "$repo_root/scripts/qwen_h100_local_rag.py" \
      validate-certified-config - --profile "$variant"
}

command=${1:-help}
shift || true

case "$command" in
  config)
    NGC_API_KEY=${NGC_API_KEY:-config-validation-only} \
      USERID=${USERID:-$(id -u)} \
      "${compose_args[@]}" config "$@"
    ;;
  validate)
    if (($# == 0)); then
      validate_general_config
    elif [[ "${1:-}" == "--certified" && $# == 1 ]]; then
      validate_general_config
      validate_certified_config
    else
      echo "Usage: scripts/qwen_h100_local_rag.sh validate [--certified]" >&2
      exit 2
    fi
    ;;
  pull)
    require_ngc_key
    validate_general_config
    "${compose_args[@]}" pull "$@"
    ;;
  up)
    require_ngc_key
    validate_general_config
    "${compose_args[@]}" up -d "$@"
    ;;
  ps)
    "${compose_args[@]}" ps "$@"
    ;;
  logs)
    "${compose_args[@]}" logs "$@"
    ;;
  restart)
    require_ngc_key
    "${compose_args[@]}" restart "$@"
    ;;
  stop)
    "${compose_args[@]}" stop "$@"
    ;;
  down)
    "${compose_args[@]}" down "$@"
    ;;
  monitor)
    output=${1:-"$repo_root/docs/research/evidence/qwen-h100-observations.jsonl"}
    uv run python "$repo_root/scripts/qwen_h100_local_rag.py" monitor \
      --repo-root "$repo_root" --output "$output" --duration 300 --interval 15
    uv run python "$repo_root/scripts/qwen_h100_local_rag.py" \
      evaluate-observations "$output" --minimum-duration 300
    ;;
  help | *)
    echo "Usage: scripts/qwen_h100_local_rag.sh COMMAND [ARGS...]"
    echo "Commands: config validate pull up ps logs restart stop down monitor"
    echo "  validate               Check safety, types, ranges, and compatibility."
    echo "  validate --certified   Also require exact FP8/NVFP4 profile conformity."
    echo "Set QWEN_VARIANT=nvfp4 to include the fallback override."
    ;;
esac
