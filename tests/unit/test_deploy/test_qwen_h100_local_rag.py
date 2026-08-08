import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.qwen_h100_local_rag import _remaining_monitor_time

REPO_ROOT = Path(__file__).parents[3]
COMPOSE_DIR = REPO_ROOT / "deploy" / "compose"
VALIDATOR = REPO_ROOT / "scripts" / "qwen_h100_local_rag.py"
COMPOSE_FILES = [
    COMPOSE_DIR / "nims.yaml",
    COMPOSE_DIR / "vectordb.yaml",
    COMPOSE_DIR / "docker-compose-ingestor-server.yaml",
    COMPOSE_DIR / "docker-compose-rag-server.yaml",
    COMPOSE_DIR / "docker-compose-qwen-h100.yaml",
]


def test_monitor_duration_starts_at_first_completed_sample() -> None:
    assert _remaining_monitor_time(12.5, 312.4, 300, 15) == pytest.approx(0.1)
    assert _remaining_monitor_time(12.5, 312.5, 300, 15) == 0


def run_validator(
    config: dict, variant: str = "fp8"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "validate-config",
            "-",
            "--variant",
            variant,
        ],
        cwd=REPO_ROOT,
        input=json.dumps(config),
        text=True,
        capture_output=True,
        check=False,
    )


def resolved_compose_config(extra_files: tuple[Path, ...] = ()) -> dict:
    if shutil.which("docker") is None:
        pytest.skip("Docker Compose is required to resolve the deployment config")

    command = ["docker", "compose", "--env-file", str(COMPOSE_DIR / ".env")]
    for compose_file in COMPOSE_FILES:
        command.extend(["-f", str(compose_file)])
    for compose_file in extra_files:
        command.extend(["-f", str(compose_file)])
    command.extend(["--profile", "qwen-h100", "config", "--format", "json"])

    environment = os.environ.copy()
    environment.update(
        {
            "NGC_API_KEY": "test-only-not-a-secret",
            "USERID": str(os.getuid()),
        }
    )
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_resolved_qwen_h100_compose_config_is_valid() -> None:
    result = run_validator(resolved_compose_config())

    assert result.returncode == 0, result.stderr
    assert "Qwen H100 compose configuration is valid" in result.stdout


def test_config_summary_is_machine_readable_and_excludes_secrets() -> None:
    config = resolved_compose_config()
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "summarize-config", "-"],
        cwd=REPO_ROOT,
        input=json.dumps(config),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "NGC_API_KEY" not in result.stdout
    assert "APIKEY" not in result.stdout
    summary = json.loads(result.stdout)
    assert summary["services"]["qwen-vllm"]["ports"] == [
        {"host_ip": "127.0.0.1", "published": "8999", "target": 8000}
    ]
    assert summary["services"]["qwen-vllm"]["gpu_device_ids"] == ["0"]
    assert summary["settings"]["APP_LLM_SERVERURL"] == ("http://qwen-vllm:8000/v1")


def test_validator_rejects_non_loopback_port() -> None:
    config = resolved_compose_config()
    config["services"]["rag-server"]["ports"][0]["host_ip"] = "0.0.0.0"

    result = run_validator(config)

    assert result.returncode == 1
    assert "must bind to 127.0.0.1" in result.stderr


def test_resolved_nvfp4_fallback_compose_config_is_valid() -> None:
    config = resolved_compose_config(
        (COMPOSE_DIR / "docker-compose-qwen-h100-nvfp4.yaml",)
    )

    result = run_validator(config, variant="nvfp4")

    assert result.returncode == 0, result.stderr
    assert "Qwen H100 compose configuration is valid" in result.stdout


def test_validator_rejects_excluded_ingestion_modes() -> None:
    config = resolved_compose_config()
    environment = config["services"]["ingestor-server"]["environment"]
    environment["APP_NVINGEST_EXTRACTPAGEASIMAGE"] = "true"
    environment["APP_NVINGEST_PDFEXTRACTMETHOD"] = "nemotron_parse"
    environment["APP_NVINGEST_SEGMENTAUDIO"] = "true"

    result = run_validator(config)

    assert result.returncode == 1
    assert "whole-page extraction must remain disabled" in result.stderr
    assert "Nemotron Parse must remain disabled" in result.stderr
    assert "audio segmentation must remain disabled" in result.stderr


def test_evaluator_accepts_five_minutes_healthy_without_restarts(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 8, 8, tzinfo=UTC)
    observations = tmp_path / "observations.jsonl"
    snapshots = []
    for offset in (0, 150, 300):
        snapshots.append(
            {
                "timestamp": (start + timedelta(seconds=offset)).isoformat(),
                "containers": {
                    service: {"health": "healthy", "restart_count": 0}
                    for service in (
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
                    )
                },
                "gpu": {"memory_used_mib": 70000, "memory_free_mib": 11559},
            }
        )
    observations.write_text(
        "".join(json.dumps(snapshot) + "\n" for snapshot in snapshots),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "evaluate-observations",
            str(observations),
            "--minimum-duration",
            "300",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Full-Service Co-residency gate passed" in result.stdout


def test_evaluator_rejects_restart_count_change(tmp_path: Path) -> None:
    start = datetime(2026, 8, 8, tzinfo=UTC)
    observations = tmp_path / "observations.jsonl"
    snapshots = []
    for offset, restart_count in ((0, 0), (300, 1)):
        snapshots.append(
            {
                "timestamp": (start + timedelta(seconds=offset)).isoformat(),
                "containers": {
                    service: {
                        "health": "healthy",
                        "restart_count": restart_count if service == "qwen-vllm" else 0,
                    }
                    for service in (
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
                    )
                },
                "gpu": {"memory_used_mib": 70000, "memory_free_mib": 11559},
            }
        )
    observations.write_text(
        "".join(json.dumps(snapshot) + "\n" for snapshot in snapshots),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "evaluate-observations",
            str(observations),
            "--minimum-duration",
            "300",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "qwen-vllm restart count changed" in result.stderr
