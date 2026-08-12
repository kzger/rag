import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scripts.qwen_h100_local_rag import (
    _remaining_monitor_time,
    validate_certified_profile,
    validate_general_config,
)

REPO_ROOT = Path(__file__).parents[3]
COMPOSE_DIR = REPO_ROOT / "deploy" / "compose"
COMPOSE_ENV = COMPOSE_DIR / ".env"
VALIDATOR = REPO_ROOT / "scripts" / "qwen_h100_local_rag.py"
START_SCRIPT = REPO_ROOT / "scripts" / "start_qwen_h100_local_rag.sh"
STOP_SCRIPT = REPO_ROOT / "scripts" / "stop_qwen_h100_local_rag.sh"
LIFECYCLE_SCRIPT = REPO_ROOT / "scripts" / "qwen_h100_local_rag.sh"
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


@pytest.mark.parametrize("script", [LIFECYCLE_SCRIPT, START_SCRIPT, STOP_SCRIPT])
def test_lifecycle_wrapper_has_valid_bash_syntax(script: Path) -> None:
    result = subprocess.run(
        ["bash", "-n", str(script)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_lifecycle_help_distinguishes_general_and_certified_validation() -> None:
    result = subprocess.run(
        [str(LIFECYCLE_SCRIPT), "help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "validate               Check safety, types, ranges" in result.stdout
    assert "validate --certified   Also require exact FP8/NVFP4" in result.stdout


def test_start_wrapper_rejects_invalid_timeout_before_deployment() -> None:
    environment = os.environ.copy()
    environment["QWEN_START_TIMEOUT_SECONDS"] = "0"

    result = subprocess.run(
        [str(START_SCRIPT)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "QWEN_START_TIMEOUT_SECONDS must be a positive integer" in result.stderr


def test_stop_wrapper_rejects_unknown_mode_without_stopping() -> None:
    result = subprocess.run(
        [str(STOP_SCRIPT), "--delete"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Usage:" in result.stderr


def run_validator(config: dict) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "validate-config",
            "-",
        ],
        cwd=REPO_ROOT,
        input=json.dumps(config),
        text=True,
        capture_output=True,
        check=False,
    )


def run_certified_validator(
    config: dict, profile: str = "fp8"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "validate-certified-config",
            "-",
            "--profile",
            profile,
        ],
        cwd=REPO_ROOT,
        input=json.dumps(config),
        text=True,
        capture_output=True,
        check=False,
    )


def resolved_compose_config(
    extra_files: tuple[Path, ...] = (),
    environment_overrides: dict[str, str] | None = None,
) -> dict:
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
            "APP_RETRIEVER_TOPK": "4",
            "APP_VLM_MAX_TOTAL_IMAGES": "2",
            "ENABLE_MULTIMODAL_ACCURACY": "false",
            "NGC_API_KEY": "test-only-not-a-secret",
            "QWEN_MAX_IMAGES_PER_PROMPT": "2",
            "USERID": str(os.getuid()),
        }
    )
    environment.update(environment_overrides or {})
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


def run_lifecycle_with_config(
    tmp_path: Path, config: dict, *arguments: str
) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    config_path = tmp_path / "resolved.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    docker = fake_bin / "docker"
    docker.write_text(
        '#!/usr/bin/env bash\ncat "$FAKE_COMPOSE_CONFIG"\n', encoding="utf-8"
    )
    docker.chmod(0o755)
    uv = fake_bin / "uv"
    uv.write_text('#!/usr/bin/env bash\nshift\nexec "$@"\n', encoding="utf-8")
    uv.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "FAKE_COMPOSE_CONFIG": str(config_path),
            "PATH": f"{fake_bin}:{environment['PATH']}",
        }
    )
    return subprocess.run(
        [str(LIFECYCLE_SCRIPT), *arguments],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_resolved_qwen_h100_compose_config_is_valid() -> None:
    result = run_validator(resolved_compose_config())

    assert result.returncode == 0, result.stderr
    assert "Qwen H100 compose configuration is valid" in result.stdout


def test_general_validation_accepts_custom_retrieval_values() -> None:
    config = resolved_compose_config()
    environment = config["services"]["rag-server"]["environment"]
    environment["VECTOR_DB_TOPK"] = "80"
    environment["APP_RETRIEVER_TOPK"] = "6"

    assert validate_general_config(config) == []


def test_resolved_config_preserves_custom_retrieval_and_conversation_values() -> None:
    custom = {
        "VECTOR_DB_TOPK": "80",
        "APP_RETRIEVER_TOPK": "6",
        "CONVERSATION_HISTORY": "8",
        "MAX_RECURSION_DEPTH": "4",
        "MAX_REFLECTION_LOOP": "2",
    }

    config = resolved_compose_config(environment_overrides=custom)
    environment = config["services"]["rag-server"]["environment"]

    assert validate_general_config(config) == []
    assert {key: environment[key] for key in custom} == custom


def test_qwen_h100_profile_uses_32k_shared_context_with_bounded_retrieval() -> None:
    config = resolved_compose_config()

    assert "32768" in config["services"]["qwen-vllm"]["command"]
    assert config["services"]["rag-server"]["environment"]["APP_RETRIEVER_TOPK"] == "4"
    assert config["services"]["rag-server"]["environment"]["VECTOR_DB_TOPK"] == "100"


def test_resolved_config_propagates_shared_image_limits() -> None:
    config = resolved_compose_config(
        environment_overrides={
            "APP_VLM_MAX_TOTAL_IMAGES": "4",
            "QWEN_MAX_IMAGES_PER_PROMPT": "4",
        }
    )

    assert '{"image":4,"video":0}' in config["services"]["qwen-vllm"]["command"]
    assert (
        config["services"]["rag-server"]["environment"]["APP_VLM_MAX_TOTAL_IMAGES"]
        == "4"
    )
    assert validate_general_config(config) == []


def test_validator_rejects_incompatible_or_unparseable_image_limits() -> None:
    incompatible = resolved_compose_config(
        environment_overrides={
            "APP_VLM_MAX_TOTAL_IMAGES": "3",
            "QWEN_MAX_IMAGES_PER_PROMPT": "2",
        }
    )
    malformed = resolved_compose_config()
    command = malformed["services"]["qwen-vllm"]["command"]
    command[command.index("--limit-mm-per-prompt") + 1] = "not-json"

    incompatible_result = run_validator(incompatible)
    malformed_result = run_validator(malformed)

    assert incompatible_result.returncode == 1
    assert "APP_VLM_MAX_TOTAL_IMAGES,QWEN_MAX_IMAGES_PER_PROMPT" in (
        incompatible_result.stderr
    )
    assert '"APP_VLM_MAX_TOTAL_IMAGES": 3' in incompatible_result.stderr
    assert '"QWEN_MAX_IMAGES_PER_PROMPT": 2' in incompatible_result.stderr
    assert malformed_result.returncode == 1
    assert "parameter=QWEN_MAX_IMAGES_PER_PROMPT actual=null" in (
        malformed_result.stderr
    )


def test_general_validation_accepts_resolved_custom_workload_budgets() -> None:
    custom = {
        "QWEN_MAX_MODEL_LEN": "20000",
        "QWEN_GPU_MEMORY_UTILIZATION": "0.50",
        "QWEN_MAX_NUM_SEQS": "2",
        "LLM_MAX_TOKENS": "6000",
        "APP_VLM_MAX_TOKENS": "5000",
        "AGENTIC_CONTEXT_MAX_TOKENS": "5000",
        "APP_FILTEREXPRESSIONGENERATOR_MAXTOKENS": "900",
        "AGENTIC_PLANNER_LLM_MAX_TOKENS": "900",
        "AGENTIC_TASK_LLM_MAX_TOKENS": "900",
        "AGENTIC_SEED_GEN_LLM_MAX_TOKENS": "900",
        "AGENTIC_SYNTHESIS_LLM_MAX_TOKENS": "900",
        "AGENTIC_CONCURRENCY_LIMIT": "2",
        "NV_INGEST_FILES_PER_BATCH": "2",
        "NV_INGEST_CONCURRENT_BATCHES": "2",
        "SUMMARY_LLM_MAX_CHUNK_LENGTH": "5000",
        "SUMMARY_MAX_PARALLELIZATION": "2",
    }

    config = resolved_compose_config(environment_overrides=custom)

    assert validate_general_config(config) == []
    assert (
        config["services"]["qwen-vllm"]["command"][
            config["services"]["qwen-vllm"]["command"].index("--max-num-seqs") + 1
        ]
        == "2"
    )
    assert (
        config["services"]["rag-server"]["environment"]["AGENTIC_CONCURRENCY_LIMIT"]
        == "2"
    )
    assert (
        config["services"]["ingestor-server"]["environment"][
            "NV_INGEST_FILES_PER_BATCH"
        ]
        == "2"
    )


def test_qwen_h100_retrieval_settings_are_explicit_in_compose_env() -> None:
    exports = {}
    for line in COMPOSE_ENV.read_text(encoding="utf-8").splitlines():
        if not line.startswith("export "):
            continue
        key, separator, value = line.removeprefix("export ").partition("=")
        if separator:
            exports[key] = value

    assert int(exports["APP_RETRIEVER_TOPK"]) > 0
    assert int(exports["VECTOR_DB_TOPK"]) >= int(exports["APP_RETRIEVER_TOPK"])


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
    assert summary["profile_status"] == {
        "kind": "certified",
        "profile": "fp8",
    }
    assert summary["certified_evidence"]["model_revision"] == (
        "e89b16ebf1988b3d6befa7de50abc2d76f26eb09"
    )
    assert summary["certified_evidence"]["resource_envelope"] == {
        "gpu_device_ids": ["0"],
        "gpu_memory_utilization": "0.48",
        "max_model_len": "32768",
        "max_num_seqs": "1",
    }
    assert summary["services"]["qwen-vllm"]["ports"] == [
        {"host_ip": "127.0.0.1", "published": "8999", "target": 8000}
    ]
    assert summary["services"]["qwen-vllm"]["gpu_device_ids"] == ["0"]
    assert summary["settings"]["APP_LLM_SERVERURL"] == ("http://qwen-vllm:8000/v1")
    assert summary["settings"]["APP_RETRIEVER_TOPK"] == "4"
    assert summary["settings"]["VECTOR_DB_TOPK"] == "100"


def test_validator_rejects_non_loopback_port() -> None:
    config = resolved_compose_config()
    config["services"]["rag-server"]["ports"][0]["host_ip"] = "0.0.0.0"

    result = run_validator(config)

    assert result.returncode == 1
    assert "published ports must bind only to 127.0.0.1" in result.stderr


def test_validator_rejects_final_top_k_above_candidate_pool() -> None:
    config = resolved_compose_config()
    environment = config["services"]["rag-server"]["environment"]
    environment["VECTOR_DB_TOPK"] = "5"
    environment["APP_RETRIEVER_TOPK"] = "6"

    result = run_validator(config)

    assert result.returncode == 1
    assert "[configuration]" in result.stderr
    assert "service=rag-server" in result.stderr
    assert "APP_RETRIEVER_TOPK,VECTOR_DB_TOPK" in result.stderr
    assert '"APP_RETRIEVER_TOPK": 6' in result.stderr
    assert '"VECTOR_DB_TOPK": 5' in result.stderr


def test_validator_rejects_invalid_resource_and_workload_values() -> None:
    config = resolved_compose_config()
    command = config["services"]["qwen-vllm"]["command"]
    command[command.index("--gpu-memory-utilization") + 1] = "1.0"
    command[command.index("--max-num-seqs") + 1] = "0"
    rag_environment = config["services"]["rag-server"]["environment"]
    rag_environment["VECTOR_DB_TOPK"] = "not-an-integer"
    rag_environment["CONVERSATION_HISTORY"] = "-1"
    rag_environment["MAX_RECURSION_DEPTH"] = "0"
    rag_environment["LLM_MAX_TOKENS"] = "40000"
    ingestor_environment = config["services"]["ingestor-server"]["environment"]
    ingestor_environment["NV_INGEST_CONCURRENT_BATCHES"] = "0"

    result = run_validator(config)

    assert result.returncode == 1
    for parameter in (
        "--gpu-memory-utilization",
        "--max-num-seqs",
        "VECTOR_DB_TOPK",
        "CONVERSATION_HISTORY",
        "MAX_RECURSION_DEPTH",
        "LLM_MAX_TOKENS",
        "NV_INGEST_CONCURRENT_BATCHES",
    ):
        assert f"parameter={parameter}" in result.stderr


def test_general_validation_derives_shared_roles_from_served_model() -> None:
    config = resolved_compose_config()
    model = "example/custom-qwen"
    command = config["services"]["qwen-vllm"]["command"]
    command[0] = model
    command[command.index("--served-model-name") + 1] = model
    command[command.index("--revision") + 1] = "custom-revision"
    rag_environment = config["services"]["rag-server"]["environment"]
    for key in (
        "APP_LLM_MODELNAME",
        "APP_VLM_MODELNAME",
        "APP_QUERYREWRITER_MODELNAME",
        "APP_FILTEREXPRESSIONGENERATOR_MODELNAME",
        "REFLECTION_LLM",
        "AGENTIC_PLANNER_LLM_MODEL",
        "AGENTIC_TASK_LLM_MODEL",
        "AGENTIC_SEED_GEN_LLM_MODEL",
        "AGENTIC_SYNTHESIS_LLM_MODEL",
    ):
        rag_environment[key] = model
    config["services"]["ingestor-server"]["environment"]["SUMMARY_LLM"] = model
    config["services"]["ingestor-server"]["environment"][
        "APP_NVINGEST_CAPTIONMODELNAME"
    ] = model
    config["services"]["nv-ingest-ms-runtime"]["environment"][
        "VLM_CAPTION_MODEL_NAME"
    ] = model

    assert validate_general_config(config) == []


def test_general_validation_rejects_models_incompatible_with_resolved_nims() -> None:
    config = resolved_compose_config()
    rag_environment = config["services"]["rag-server"]["environment"]
    rag_environment["APP_EMBEDDINGS_MODELNAME"] = "incompatible/embedding"
    rag_environment["APP_RANKING_MODELNAME"] = "incompatible/ranking"

    issues = validate_general_config(config)

    assert any(
        issue.parameter == "APP_EMBEDDINGS_MODELNAME"
        and "resolved nemotron-vlm-embedding-ms image" in issue.rule
        for issue in issues
    )
    assert any(
        issue.parameter == "APP_RANKING_MODELNAME"
        and "resolved nemotron-ranking-vl-ms image" in issue.rule
        for issue in issues
    )


def test_general_validation_rejects_unresolved_ingestion_runtime_endpoints() -> None:
    config = resolved_compose_config()
    environment = config["services"]["ingestor-server"]["environment"]
    environment["APP_NVINGEST_MESSAGECLIENTHOSTNAME"] = "missing-runtime"
    environment["APP_NVINGEST_OCRURL"] = "http://missing-ocr:9999/v1/infer"

    issues = validate_general_config(config)

    assert any(
        issue.parameter == "APP_NVINGEST_MESSAGECLIENTHOSTNAME"
        and "resolved service nv-ingest-ms-runtime" in issue.rule
        for issue in issues
    )
    assert any(
        issue.parameter == "APP_NVINGEST_OCRURL"
        and "resolved service nemotron-ocr" in issue.rule
        for issue in issues
    )


def test_resolved_nvfp4_fallback_compose_config_is_valid() -> None:
    config = resolved_compose_config(
        (COMPOSE_DIR / "docker-compose-qwen-h100-nvfp4.yaml",)
    )

    result = run_validator(config)

    assert result.returncode == 0, result.stderr
    assert "Qwen H100 compose configuration is valid" in result.stdout


def test_unmodified_profiles_match_their_certified_baselines() -> None:
    fp8 = resolved_compose_config()
    nvfp4 = resolved_compose_config(
        (COMPOSE_DIR / "docker-compose-qwen-h100-nvfp4.yaml",)
    )

    assert validate_certified_profile(fp8, "fp8") == []
    assert validate_certified_profile(nvfp4, "nvfp4") == []


def test_safe_custom_config_is_reported_only_as_profile_drift() -> None:
    config = resolved_compose_config(
        environment_overrides={
            "APP_RETRIEVER_TOPK": "6",
            "VECTOR_DB_TOPK": "80",
        }
    )

    assert validate_general_config(config) == []
    drift = validate_certified_profile(config, "fp8")

    assert {issue.category for issue in drift} == {"profile_drift"}
    assert any(
        issue.parameter == "APP_RETRIEVER_TOPK"
        and issue.actual == "6"
        and issue.baseline == "4"
        for issue in drift
    )


def test_certified_cli_reports_drift_separately_from_general_errors() -> None:
    baseline = resolved_compose_config()
    custom = resolved_compose_config(environment_overrides={"APP_RETRIEVER_TOPK": "6"})

    certified = run_certified_validator(baseline)
    drifted = run_certified_validator(custom)

    assert certified.returncode == 0, certified.stderr
    assert "matches certified fp8" in certified.stdout
    assert drifted.returncode == 1
    assert "DRIFT: [profile_drift]" in drifted.stderr
    assert "parameter=APP_RETRIEVER_TOPK" in drifted.stderr
    assert 'actual="6"' in drifted.stderr
    assert 'baseline="4"' in drifted.stderr


def test_certified_command_matching_is_exact_and_redacts_unknown_values() -> None:
    fp8 = resolved_compose_config()
    fp8["services"]["qwen-vllm"]["command"].extend(["--api-key", "review-secret"])
    nvfp4 = resolved_compose_config(
        (COMPOSE_DIR / "docker-compose-qwen-h100-nvfp4.yaml",)
    )
    nvfp4_command = nvfp4["services"]["qwen-vllm"]["command"]
    nvfp4_command[nvfp4_command.index("--quantization") + 1] = "wrong"
    nvfp4_command.append("modelopt")

    fp8_result = run_certified_validator(fp8)
    nvfp4_result = run_certified_validator(nvfp4, "nvfp4")
    summary_result = subprocess.run(
        [sys.executable, str(VALIDATOR), "summarize-config", "-"],
        cwd=REPO_ROOT,
        input=json.dumps(fp8),
        text=True,
        capture_output=True,
        check=False,
    )

    assert fp8_result.returncode == 1
    assert "parameter=command_signature" in fp8_result.stderr
    assert "--api-key" in fp8_result.stderr
    assert "review-secret" not in fp8_result.stderr
    assert nvfp4_result.returncode == 1
    assert "parameter=command_signature" in nvfp4_result.stderr
    assert "review-secret" not in summary_result.stdout
    assert "--api-key" not in summary_result.stdout


def test_lifecycle_accepts_custom_by_default_and_requires_opt_in_certification(
    tmp_path: Path,
) -> None:
    custom = resolved_compose_config(environment_overrides={"APP_RETRIEVER_TOPK": "6"})

    general = run_lifecycle_with_config(tmp_path / "general", custom, "validate")
    certified = run_lifecycle_with_config(
        tmp_path / "certified", custom, "validate", "--certified"
    )
    up = run_lifecycle_with_config(tmp_path / "up", custom, "up")

    assert general.returncode == 0, general.stderr
    assert "custom (not certified)" in general.stdout
    assert certified.returncode == 1
    assert "DRIFT: [profile_drift]" in certified.stderr
    assert up.returncode == 0, up.stderr
    assert "custom (not certified)" in up.stdout


def test_validator_rejects_excluded_ingestion_modes() -> None:
    config = resolved_compose_config()
    environment = config["services"]["ingestor-server"]["environment"]
    environment["APP_NVINGEST_EXTRACTPAGEASIMAGE"] = "true"
    environment["APP_NVINGEST_PDFEXTRACTMETHOD"] = "nemotron_parse"
    environment["APP_NVINGEST_SEGMENTAUDIO"] = "true"

    result = run_validator(config)

    assert result.returncode == 1
    assert "APP_NVINGEST_EXTRACTPAGEASIMAGE" in result.stderr
    assert "APP_NVINGEST_SEGMENTAUDIO" in result.stderr
    assert "Nemotron Parse is excluded" in result.stderr


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
