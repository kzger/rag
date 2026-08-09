#!/usr/bin/env python3
"""Validate and observe the Qwen H100 Local RAG Deployment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REQUIRED_SERVICES = {
    "qwen-vllm",
    "nemotron-vlm-embedding-ms",
    "nemotron-ranking-vl-ms",
    "page-elements",
    "graphic-elements",
    "table-structure",
    "nemotron-ocr",
    "elasticsearch",
    "seaweedfs",
    "redis",
    "nv-ingest-ms-runtime",
    "ingestor-server",
    "rag-server",
    "rag-frontend",
}
FORBIDDEN_SERVICES = {
    "nim-llm",
    "vlm-ms",
    "vlm-captioning-ms",
    "paddle",
    "nemotron-parse",
    "audio",
}
EXPECTED_PORTS = {
    ("qwen-vllm", "8999", 8000),
    ("rag-server", "8081", 8081),
    ("ingestor-server", "8082", 8082),
    ("rag-frontend", "8090", 3000),
}
GPU_SERVICES = {
    "qwen-vllm",
    "nemotron-vlm-embedding-ms",
    "nemotron-ranking-vl-ms",
    "page-elements",
    "graphic-elements",
    "table-structure",
    "nemotron-ocr",
}
QWEN_URL = "http://qwen-vllm:8000/v1"
MODEL_VARIANTS = {
    "fp8": {
        "model": "Qwen/Qwen3.6-27B-FP8",
        "revision": "e89b16ebf1988b3d6befa7de50abc2d76f26eb09",
        "memory_utilization": "0.48",
        "extra_arguments": (),
    },
    "nvfp4": {
        "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "revision": "491c2f1ea524c639598bf8fa787a93fed5a6fbce",
        "memory_utilization": "0.40",
        "extra_arguments": (
            "--quantization",
            "modelopt",
            "--kv-cache-dtype",
            "fp8",
            "--moe-backend",
            "marlin",
        ),
    },
}
SAFE_SETTING_KEYS = {
    "AGENTIC_CONCURRENCY_LIMIT",
    "AGENTIC_CONTEXT_MAX_TOKENS",
    "AGENTIC_PLANNER_LLM_MAX_TOKENS",
    "AGENTIC_PLANNER_LLM_MODEL",
    "AGENTIC_PLANNER_LLM_SERVERURL",
    "AGENTIC_SEED_GEN_LLM_MODEL",
    "AGENTIC_SEED_GEN_LLM_MAX_TOKENS",
    "AGENTIC_SEED_GEN_LLM_SERVERURL",
    "AGENTIC_SYNTHESIS_LLM_MODEL",
    "AGENTIC_SYNTHESIS_LLM_MAX_TOKENS",
    "AGENTIC_SYNTHESIS_LLM_SERVERURL",
    "AGENTIC_TASK_LLM_MODEL",
    "AGENTIC_TASK_LLM_MAX_TOKENS",
    "AGENTIC_TASK_LLM_SERVERURL",
    "APP_EMBEDDINGS_DIMENSIONS",
    "APP_EMBEDDINGS_MODELNAME",
    "APP_EMBEDDINGS_SERVERURL",
    "APP_FILTEREXPRESSIONGENERATOR_MODELNAME",
    "APP_FILTEREXPRESSIONGENERATOR_MAXTOKENS",
    "APP_FILTEREXPRESSIONGENERATOR_SERVERURL",
    "APP_LLM_MODELNAME",
    "APP_LLM_SERVERURL",
    "APP_NVINGEST_CAPTIONENDPOINTURL",
    "APP_NVINGEST_CAPTIONMODELNAME",
    "APP_NVINGEST_EXTRACTCHARTS",
    "APP_NVINGEST_EXTRACTIMAGES",
    "APP_NVINGEST_EXTRACTINFOGRAPHICS",
    "APP_NVINGEST_EXTRACTPAGEASIMAGE",
    "APP_NVINGEST_EXTRACTTABLES",
    "APP_NVINGEST_EXTRACTTABLESMETHOD",
    "APP_NVINGEST_EXTRACTTEXT",
    "APP_NVINGEST_IMAGE_ELEMENTS_MODALITY",
    "APP_NVINGEST_PDFEXTRACTMETHOD",
    "APP_NVINGEST_SEGMENTAUDIO",
    "APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY",
    "APP_QUERYREWRITER_MODELNAME",
    "APP_QUERYREWRITER_SERVERURL",
    "APP_RANKING_MODELNAME",
    "APP_RANKING_SERVERURL",
    "APP_VLM_ENABLE_THINKING",
    "APP_VLM_MAX_TOTAL_IMAGES",
    "APP_VLM_MODELNAME",
    "APP_VLM_SERVERURL",
    "CONVERSATION_HISTORY",
    "ENABLE_AGENTIC_RAG",
    "ENABLE_FILTER_GENERATOR",
    "ENABLE_QUERYREWRITER",
    "ENABLE_QUERY_DECOMPOSITION",
    "ENABLE_REFLECTION",
    "ENABLE_VLM_INFERENCE",
    "ENABLE_VLM_RERANKER_IMAGE_INPUT",
    "MAX_RECURSION_DEPTH",
    "MAX_REFLECTION_LOOP",
    "NV_INGEST_CONCURRENT_BATCHES",
    "NV_INGEST_FILES_PER_BATCH",
    "REFLECTION_LLM",
    "REFLECTION_LLM_SERVERURL",
    "SUMMARY_LLM",
    "SUMMARY_LLM_MAX_CHUNK_LENGTH",
    "SUMMARY_LLM_SERVERURL",
    "SUMMARY_MAX_PARALLELIZATION",
    "VLM_CAPTION_ENDPOINT",
    "VLM_CAPTION_MODEL_NAME",
    "VLM_TO_LLM_FALLBACK",
}


def _load_config(path: str) -> dict[str, Any]:
    if path == "-":
        return json.load(sys.stdin)
    with Path(path).open(encoding="utf-8") as config_file:
        return json.load(config_file)


def _command(service: dict[str, Any]) -> list[str]:
    command = service.get("command", [])
    if isinstance(command, str):
        return command.split()
    return [str(item) for item in command]


def validate_config(config: dict[str, Any], variant: str = "fp8") -> list[str]:
    """Return validation errors for a resolved Qwen H100 Compose config."""
    services = config.get("services", {})
    errors: list[str] = []
    model_variant = MODEL_VARIANTS[variant]
    qwen_model = str(model_variant["model"])
    qwen_revision = str(model_variant["revision"])

    missing = REQUIRED_SERVICES - services.keys()
    if missing:
        errors.append(f"missing required services: {', '.join(sorted(missing))}")

    forbidden = FORBIDDEN_SERVICES & services.keys()
    if forbidden:
        errors.append(
            "default/excluded services must be disabled: "
            + ", ".join(sorted(forbidden))
        )

    actual_ports: set[tuple[str, str, int]] = set()
    for service_name, service in services.items():
        for port in service.get("ports", []):
            host_ip = port.get("host_ip")
            if host_ip != "127.0.0.1":
                errors.append(f"{service_name} must bind to 127.0.0.1")
            actual_ports.add(
                (service_name, str(port.get("published")), int(port["target"]))
            )
    if actual_ports != EXPECTED_PORTS:
        errors.append(
            f"published ports must be exactly {sorted(EXPECTED_PORTS)}; "
            f"got {sorted(actual_ports)}"
        )

    qwen = services.get("qwen-vllm", {})
    image = str(qwen.get("image", ""))
    if "@sha256:" not in image:
        errors.append("Qwen image must be pinned by digest")
    command = _command(qwen)
    for required_argument in (
        qwen_model,
        qwen_revision,
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        str(model_variant["memory_utilization"]),
        "--max-num-seqs",
        "1",
        '{"image":2,"video":0}',
        '{"enable_thinking":false}',
        *model_variant["extra_arguments"],
    ):
        if required_argument not in command:
            errors.append(f"Qwen command is missing {required_argument}")

    for service_name in GPU_SERVICES & services.keys():
        devices = (
            services[service_name]
            .get("deploy", {})
            .get("resources", {})
            .get("reservations", {})
            .get("devices", [])
        )
        device_ids = {
            str(device_id)
            for device in devices
            for device_id in device.get("device_ids", [])
        }
        if device_ids != {"0"}:
            errors.append(f"{service_name} must use only GPU 0")

    for service_name in REQUIRED_SERVICES & services.keys():
        service = services[service_name]
        if "healthcheck" not in service:
            errors.append(f"{service_name} must define a healthcheck")
        if service.get("restart") != "unless-stopped":
            errors.append(f"{service_name} must use restart: unless-stopped")

    rag_environment = services.get("rag-server", {}).get("environment", {})
    ingestor_environment = services.get("ingestor-server", {}).get("environment", {})
    runtime_environment = services.get("nv-ingest-ms-runtime", {}).get(
        "environment", {}
    )
    expected_roles = {
        "APP_LLM_MODELNAME": qwen_model,
        "APP_LLM_SERVERURL": QWEN_URL,
        "APP_VLM_MODELNAME": qwen_model,
        "APP_VLM_SERVERURL": QWEN_URL,
        "APP_QUERYREWRITER_MODELNAME": qwen_model,
        "APP_QUERYREWRITER_SERVERURL": QWEN_URL,
        "APP_FILTEREXPRESSIONGENERATOR_MODELNAME": qwen_model,
        "APP_FILTEREXPRESSIONGENERATOR_SERVERURL": QWEN_URL,
        "REFLECTION_LLM": qwen_model,
        "REFLECTION_LLM_SERVERURL": QWEN_URL,
    }
    for key, expected in expected_roles.items():
        if str(rag_environment.get(key)) != expected:
            errors.append(f"rag-server {key} must be {expected}")

    for key in ("SUMMARY_LLM", "SUMMARY_LLM_SERVERURL"):
        expected = qwen_model if key == "SUMMARY_LLM" else QWEN_URL
        if str(ingestor_environment.get(key)) != expected:
            errors.append(f"ingestor-server {key} must be {expected}")

    for role in ("PLANNER", "TASK", "SEED_GEN", "SYNTHESIS"):
        model_key = f"AGENTIC_{role}_LLM_MODEL"
        url_key = f"AGENTIC_{role}_LLM_SERVERURL"
        if str(rag_environment.get(model_key)) != qwen_model:
            errors.append(f"rag-server {model_key} must be {qwen_model}")
        if str(rag_environment.get(url_key)) != QWEN_URL:
            errors.append(f"rag-server {url_key} must be {QWEN_URL}")

    expected_rag_environment = {
        "ENABLE_VLM_INFERENCE": "true",
        "VLM_TO_LLM_FALLBACK": "false",
        "APP_VLM_ENABLE_THINKING": "false",
        "APP_VLM_MAX_TOTAL_IMAGES": "2",
        "APP_EMBEDDINGS_SERVERURL": "http://nemotron-vlm-embedding-ms:8000/v1",
        "APP_EMBEDDINGS_MODELNAME": "nvidia/llama-nemotron-embed-vl-1b-v2",
        "APP_EMBEDDINGS_DIMENSIONS": "2048",
        "APP_RANKING_SERVERURL": "http://nemotron-ranking-vl-ms:8000",
        "APP_RANKING_MODELNAME": "nvidia/llama-nemotron-rerank-vl-1b-v2",
        "ENABLE_VLM_RERANKER_IMAGE_INPUT": "true",
        "ENABLE_QUERYREWRITER": "true",
        "CONVERSATION_HISTORY": "5",
        "ENABLE_QUERY_DECOMPOSITION": "true",
        "MAX_RECURSION_DEPTH": "3",
        "ENABLE_FILTER_GENERATOR": "true",
        "ENABLE_REFLECTION": "true",
        "MAX_REFLECTION_LOOP": "3",
        "ENABLE_AGENTIC_RAG": "true",
        "AGENTIC_CONCURRENCY_LIMIT": "1",
        "AGENTIC_CONTEXT_MAX_TOKENS": "4096",
        "APP_RETRIEVER_TOPK": "4",
        "APP_FILTEREXPRESSIONGENERATOR_MAXTOKENS": "1024",
    }
    for role in ("PLANNER", "TASK", "SEED_GEN", "SYNTHESIS"):
        expected_rag_environment[f"AGENTIC_{role}_LLM_MAX_TOKENS"] = "1024"
    for key, expected in expected_rag_environment.items():
        if str(rag_environment.get(key)).lower() != expected.lower():
            errors.append(f"rag-server {key} must be {expected}")

    expected_ingestor_environment = {
        "APP_EMBEDDINGS_SERVERURL": "http://nemotron-vlm-embedding-ms:8000/v1",
        "APP_EMBEDDINGS_MODELNAME": "nvidia/llama-nemotron-embed-vl-1b-v2",
        "APP_EMBEDDINGS_DIMENSIONS": "2048",
        "APP_NVINGEST_EXTRACTTEXT": "true",
        "APP_NVINGEST_EXTRACTINFOGRAPHICS": "true",
        "APP_NVINGEST_EXTRACTTABLES": "true",
        "APP_NVINGEST_EXTRACTCHARTS": "true",
        "APP_NVINGEST_EXTRACTIMAGES": "true",
        "APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY": "text_image",
        "APP_NVINGEST_IMAGE_ELEMENTS_MODALITY": "image",
        "APP_NVINGEST_EXTRACTTABLESMETHOD": "yolox",
        "APP_NVINGEST_CAPTIONMODELNAME": qwen_model,
        "APP_NVINGEST_CAPTIONENDPOINTURL": (
            "http://qwen-vllm:8000/v1/chat/completions"
        ),
        "NV_INGEST_FILES_PER_BATCH": "1",
        "NV_INGEST_CONCURRENT_BATCHES": "1",
        "SUMMARY_LLM_MAX_CHUNK_LENGTH": "6144",
        "SUMMARY_MAX_PARALLELIZATION": "1",
    }
    for key, expected in expected_ingestor_environment.items():
        if str(ingestor_environment.get(key)).lower() != expected.lower():
            errors.append(f"ingestor-server {key} must be {expected}")

    if str(ingestor_environment.get("APP_NVINGEST_EXTRACTPAGEASIMAGE")).lower() != (
        "false"
    ):
        errors.append("whole-page extraction must remain disabled")
    if str(ingestor_environment.get("APP_NVINGEST_PDFEXTRACTMETHOD")).lower() not in {
        "none",
        "pdfium",
    }:
        errors.append("Nemotron Parse must remain disabled")
    if str(ingestor_environment.get("APP_NVINGEST_SEGMENTAUDIO")).lower() != ("false"):
        errors.append("audio segmentation must remain disabled")

    expected_runtime_environment = {
        "APP_EMBEDDINGS_SERVERURL": "http://nemotron-vlm-embedding-ms:8000/v1",
        "APP_EMBEDDINGS_MODELNAME": "nvidia/llama-nemotron-embed-vl-1b-v2",
        "VLM_CAPTION_ENDPOINT": "http://qwen-vllm:8000/v1/chat/completions",
        "VLM_CAPTION_MODEL_NAME": qwen_model,
    }
    for key, expected in expected_runtime_environment.items():
        if str(runtime_environment.get(key)).lower() != expected.lower():
            errors.append(f"nv-ingest-ms-runtime {key} must be {expected}")

    return errors


def summarize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return reproducibility evidence without copying secrets from Compose."""
    services = config.get("services", {})
    service_summary: dict[str, dict[str, Any]] = {}
    settings: dict[str, str] = {}
    for service_name in sorted(REQUIRED_SERVICES & services.keys()):
        service = services[service_name]
        devices = (
            service.get("deploy", {})
            .get("resources", {})
            .get("reservations", {})
            .get("devices", [])
        )
        device_ids = sorted(
            {
                str(device_id)
                for device in devices
                for device_id in device.get("device_ids", [])
            }
        )
        ports = [
            {
                "host_ip": port.get("host_ip"),
                "published": str(port.get("published")),
                "target": int(port["target"]),
            }
            for port in service.get("ports", [])
        ]
        volumes = [
            {
                "source": volume.get("source"),
                "target": volume.get("target"),
                "type": volume.get("type"),
            }
            for volume in service.get("volumes", [])
        ]
        service_summary[service_name] = {
            "command": _command(service) if service_name == "qwen-vllm" else [],
            "depends_on": service.get("depends_on", {}),
            "gpu_device_ids": device_ids,
            "has_healthcheck": "healthcheck" in service,
            "image": service.get("image"),
            "ports": ports,
            "restart": service.get("restart"),
            "volumes": volumes,
        }
        environment = service.get("environment", {})
        for key in sorted(SAFE_SETTING_KEYS & environment.keys()):
            settings[key] = str(environment[key])

    return {
        "forbidden_services_present": sorted(FORBIDDEN_SERVICES & services.keys()),
        "required_services": sorted(REQUIRED_SERVICES),
        "services": service_summary,
        "settings": settings,
    }


def _load_observations(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as observations_file:
        return [json.loads(line) for line in observations_file if line.strip()]


def evaluate_observations(
    observations: list[dict[str, Any]], minimum_duration: int
) -> tuple[list[str], dict[str, int]]:
    """Evaluate health and restart observations against the co-residency gate."""
    if not observations:
        return ["observation file is empty"], {}

    errors: list[str] = []
    first_timestamp = datetime.fromisoformat(observations[0]["timestamp"])
    last_timestamp = datetime.fromisoformat(observations[-1]["timestamp"])
    duration = int((last_timestamp - first_timestamp).total_seconds())
    if duration < minimum_duration:
        errors.append(
            f"observation duration is {duration}s; requires {minimum_duration}s"
        )

    first_containers = observations[0].get("containers", {})
    for service_name in sorted(REQUIRED_SERVICES):
        if service_name not in first_containers:
            errors.append(f"missing observation for {service_name}")
            continue
        initial_restart_count = first_containers[service_name].get("restart_count")
        for observation in observations:
            service = observation.get("containers", {}).get(service_name)
            if service is None:
                errors.append(f"missing observation for {service_name}")
                break
            if service.get("health") != "healthy":
                errors.append(
                    f"{service_name} was {service.get('health', 'unknown')} "
                    f"at {observation.get('timestamp')}"
                )
                break
            if service.get("restart_count") != initial_restart_count:
                errors.append(f"{service_name} restart count changed")
                break

    memory_used = [
        int(observation.get("gpu", {}).get("memory_used_mib", 0))
        for observation in observations
    ]
    memory_free = [
        int(observation.get("gpu", {}).get("memory_free_mib", 0))
        for observation in observations
    ]
    summary = {
        "duration_seconds": duration,
        "peak_memory_used_mib": max(memory_used, default=0),
        "minimum_memory_free_mib": min(memory_free, default=0),
    }
    return errors, summary


def _compose_command(repo_root: Path) -> list[str]:
    compose_dir = repo_root / "deploy" / "compose"
    files = (
        "nims.yaml",
        "vectordb.yaml",
        "docker-compose-ingestor-server.yaml",
        "docker-compose-rag-server.yaml",
        "docker-compose-qwen-h100.yaml",
    )
    command = ["docker", "compose", "--env-file", str(compose_dir / ".env")]
    for compose_file in files:
        command.extend(["-f", str(compose_dir / compose_file)])
    command.extend(["--profile", "qwen-h100"])
    return command


def _run_json(command: list[str]) -> Any:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "command failed")
    return json.loads(result.stdout)


def collect_snapshot(repo_root: Path) -> dict[str, Any]:
    """Collect one timestamped container-health and GPU-usage snapshot."""
    containers: dict[str, dict[str, Any]] = {}
    compose_command = _compose_command(repo_root)
    for service_name in sorted(REQUIRED_SERVICES):
        result = subprocess.run(
            [*compose_command, "ps", "-q", service_name],
            text=True,
            capture_output=True,
            check=False,
        )
        container_id = result.stdout.strip()
        if result.returncode != 0 or not container_id:
            containers[service_name] = {
                "health": "missing",
                "restart_count": None,
            }
            continue
        inspection = _run_json(["docker", "inspect", container_id])[0]
        state = inspection.get("State", {})
        containers[service_name] = {
            "health": state.get("Health", {}).get(
                "Status", "healthy" if state.get("Running") else "stopped"
            ),
            "restart_count": int(inspection.get("RestartCount", 0)),
        }

    gpu_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    gpu: dict[str, int | None] = {
        "memory_used_mib": None,
        "memory_free_mib": None,
        "utilization_percent": None,
    }
    if gpu_result.returncode == 0 and gpu_result.stdout.strip():
        used, free, utilization = (
            int(value.strip()) for value in gpu_result.stdout.splitlines()[0].split(",")
        )
        gpu = {
            "memory_used_mib": used,
            "memory_free_mib": free,
            "utilization_percent": utilization,
        }

    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "containers": containers,
        "gpu": gpu,
    }


def _remaining_monitor_time(
    first_sample_time: float,
    current_time: float,
    duration: int,
    interval: int,
) -> float:
    """Return the next sleep interval measured from the first complete sample."""
    remaining = duration - (current_time - first_sample_time)
    return min(float(interval), max(0.0, remaining))


def monitor(repo_root: Path, output: Path, duration: int, interval: int) -> None:
    """Write observation snapshots for at least the requested duration."""
    first_sample_time: float | None = None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as output_file:
        while True:
            snapshot = collect_snapshot(repo_root)
            output_file.write(json.dumps(snapshot, sort_keys=True) + "\n")
            output_file.flush()
            current_time = time.monotonic()
            if first_sample_time is None:
                first_sample_time = current_time
            sleep_seconds = _remaining_monitor_time(
                first_sample_time, current_time, duration, interval
            )
            if sleep_seconds == 0:
                break
            time.sleep(sleep_seconds)


def main() -> int:
    """Run the Qwen H100 validation and observation command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate-config")
    validate_parser.add_argument("config", help="Resolved Compose JSON or - for stdin")
    validate_parser.add_argument(
        "--variant", choices=tuple(MODEL_VARIANTS), default="fp8"
    )
    summarize_parser = subparsers.add_parser("summarize-config")
    summarize_parser.add_argument("config", help="Resolved Compose JSON or - for stdin")
    evaluate_parser = subparsers.add_parser("evaluate-observations")
    evaluate_parser.add_argument("observations", type=Path)
    evaluate_parser.add_argument("--minimum-duration", type=int, default=300)
    monitor_parser = subparsers.add_parser("monitor")
    monitor_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    monitor_parser.add_argument("--output", type=Path, required=True)
    monitor_parser.add_argument("--duration", type=int, default=300)
    monitor_parser.add_argument("--interval", type=int, default=15)
    arguments = parser.parse_args()

    if arguments.command == "validate-config":
        errors = validate_config(_load_config(arguments.config), arguments.variant)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("Qwen H100 compose configuration is valid")
        return 0
    if arguments.command == "summarize-config":
        print(json.dumps(summarize_config(_load_config(arguments.config)), indent=2))
        return 0
    if arguments.command == "evaluate-observations":
        errors, summary = evaluate_observations(
            _load_observations(arguments.observations), arguments.minimum_duration
        )
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("Full-Service Co-residency gate passed")
        print(json.dumps(summary, sort_keys=True))
        return 0
    if arguments.command == "monitor":
        monitor(
            arguments.repo_root.resolve(),
            arguments.output,
            arguments.duration,
            arguments.interval,
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
