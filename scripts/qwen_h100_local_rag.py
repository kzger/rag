#!/usr/bin/env python3
"""Validate and observe the Qwen H100 Local RAG Deployment."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
QWEN_VALUE_ARGUMENTS = {
    "--default-chat-template-kwargs",
    "--gpu-memory-utilization",
    "--host",
    "--kv-cache-dtype",
    "--limit-mm-per-prompt",
    "--max-model-len",
    "--max-num-seqs",
    "--moe-backend",
    "--port",
    "--quantization",
    "--reasoning-parser",
    "--revision",
    "--served-model-name",
    "--tensor-parallel-size",
}
QWEN_FLAG_ARGUMENTS = {"--trust-remote-code"}
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
CERTIFIED_RAG_SETTINGS = {
    "ENABLE_VLM_INFERENCE": "true",
    "VLM_TO_LLM_FALLBACK": "false",
    "APP_VLM_ENABLE_THINKING": "false",
    "APP_VLM_MAX_TOTAL_IMAGES": "2",
    "LLM_MAX_TOKENS": "8192",
    "APP_VLM_MAX_TOKENS": "8192",
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
    "VECTOR_DB_TOPK": "100",
    "APP_FILTEREXPRESSIONGENERATOR_MAXTOKENS": "1024",
    "AGENTIC_PLANNER_LLM_MAX_TOKENS": "1024",
    "AGENTIC_TASK_LLM_MAX_TOKENS": "1024",
    "AGENTIC_SEED_GEN_LLM_MAX_TOKENS": "1024",
    "AGENTIC_SYNTHESIS_LLM_MAX_TOKENS": "1024",
}
CERTIFIED_INGESTOR_SETTINGS = {
    "NV_INGEST_FILES_PER_BATCH": "1",
    "NV_INGEST_CONCURRENT_BATCHES": "1",
    "SUMMARY_LLM_MAX_CHUNK_LENGTH": "6144",
    "SUMMARY_MAX_PARALLELIZATION": "1",
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
    "APP_NVINGEST_GRAPHICELEMENTSURL",
    "APP_NVINGEST_MESSAGECLIENTHOSTNAME",
    "APP_NVINGEST_MESSAGECLIENTPORT",
    "APP_NVINGEST_OCRURL",
    "APP_NVINGEST_PAGEELEMENTSURL",
    "APP_NVINGEST_PDFEXTRACTMETHOD",
    "APP_NVINGEST_SEGMENTAUDIO",
    "APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY",
    "APP_NVINGEST_TABLESTRUCTUREURL",
    "APP_QUERYREWRITER_MODELNAME",
    "APP_QUERYREWRITER_SERVERURL",
    "APP_RANKING_MODELNAME",
    "APP_RANKING_SERVERURL",
    "APP_RETRIEVER_TOPK",
    "APP_VLM_ENABLE_THINKING",
    "APP_VLM_MAX_TOTAL_IMAGES",
    "APP_VLM_MAX_TOKENS",
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
    "LLM_MAX_TOKENS",
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
    "VECTOR_DB_TOPK",
    "HF_TOKEN",
    "TOKEN",
    "GITHUB_TOKEN",
    "client_secret",
}


@dataclass(frozen=True)
class ValidationIssue:
    """One actionable, secret-free configuration diagnostic."""

    category: str
    service: str
    parameter: str
    actual: Any
    rule: str
    baseline: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "actual", _redact_secrets(self.actual))
        object.__setattr__(self, "rule", _redact_secrets(self.rule))
        object.__setattr__(self, "baseline", _redact_secrets(self.baseline))

    def __str__(self) -> str:
        fields = (
            f"[{self.category}] service={self.service} parameter={self.parameter} "
            f"actual={json.dumps(self.actual, sort_keys=True)} rule={self.rule}"
        )
        if self.baseline is not None:
            fields += f" baseline={json.dumps(self.baseline, sort_keys=True)}"
        return fields


_SECRET_COMPONENT_PATTERN = (
    r"(?:api[-_]?(?:key|token)|password|secret|(?:access|auth|bearer)[-_]?token)"
)
_SECRET_KEY_PATTERN = re.compile(
    rf"^(?:token|hf_token|{_SECRET_COMPONENT_PATTERN})$|"
    rf"(?:^|[-_])(?:{_SECRET_COMPONENT_PATTERN}|token|secret)$",
    re.IGNORECASE,
)
_KEY_VALUE_SECRET_PATTERN = re.compile(
    rf"(?i)(?:^|[?&\s,])(?:token|hf_token|{_SECRET_COMPONENT_PATTERN}|"
    rf"[A-Za-z0-9]+[-_]token|[A-Za-z0-9]+[-_]secret)"
    rf"\s*[=:]\s*[^&\s,}}]+"
)
_BEARER_SECRET_PATTERN = re.compile(r"(?i)(\bBearer\s+)[^\s,}]+")
_URL_USERINFO_PATTERN = re.compile(r"(://)[^/@\s]+(@)")
_URL_QUERY_SECRET_PATTERN = re.compile(
    r"(?i)([?&](?:token|api[_-]?key|password|secret|(?:access|auth|bearer)[_-]?token|"
    r"[A-Za-z0-9]+[-_]token|[A-Za-z0-9]+[-_]secret)=)[^&#\s]+"
)


def _is_secret_key(key: str | None) -> bool:
    return bool(key and _SECRET_KEY_PATTERN.search(key.replace(" ", "")))


def _redact_secret_string(value: str) -> str:
    value = _KEY_VALUE_SECRET_PATTERN.sub("<redacted>", value)
    value = _BEARER_SECRET_PATTERN.sub(r"\1<redacted>", value)
    value = _URL_QUERY_SECRET_PATTERN.sub(r"\1<redacted>", value)
    return _URL_USERINFO_PATTERN.sub(r"\1<redacted>\2", value)


def _redact_secrets(value: Any, key: str | None = None) -> Any:
    """Keep diagnostics and summaries safe without masking ordinary token limits."""
    if _is_secret_key(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            item_key: _redact_secrets(item, str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_secrets(item) for item in value)
    if isinstance(value, str):
        return _redact_secret_string(value)
    return value


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


def _safe_qwen_command(
    command: list[str], *, include_unknown_options: bool = True
) -> list[str]:
    """Return a reproducible command signature without unknown argument values."""
    if not command:
        return []
    signature = [command[0]]
    index = 1
    while index < len(command):
        argument = command[index]
        option = argument.split("=", maxsplit=1)[0]
        if option in QWEN_FLAG_ARGUMENTS:
            signature.append(option)
            index += 1
        elif option in QWEN_VALUE_ARGUMENTS:
            signature.append(option)
            if "=" in argument:
                signature.append(argument.split("=", maxsplit=1)[1])
                index += 1
            elif index + 1 < len(command):
                signature.append(command[index + 1])
                index += 2
            else:
                signature.append("<missing-value>")
                index += 1
        elif option.startswith("--"):
            secret_option = _is_secret_key(option.removeprefix("--"))
            if include_unknown_options and not secret_option:
                signature.append(option)
            elif include_unknown_options and secret_option:
                signature.extend([option, "<redacted>"])
            index += 1
            if index < len(command) and not command[index].startswith("--"):
                index += 1
        else:
            signature.append("<unexpected-positional>")
            index += 1
    return [_redact_secret_string(item) for item in signature]


def _certified_qwen_command(profile: str) -> list[str]:
    model_variant = MODEL_VARIANTS[profile]
    model = str(model_variant["model"])
    command = [
        model,
        "--revision",
        str(model_variant["revision"]),
        "--served-model-name",
        model,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        "1",
    ]
    if profile == "nvfp4":
        command.append("--trust-remote-code")
        command.extend(str(value) for value in model_variant["extra_arguments"])
    command.extend(
        [
            "--max-model-len",
            "32768",
            "--gpu-memory-utilization",
            str(model_variant["memory_utilization"]),
            "--max-num-seqs",
            "1",
            "--reasoning-parser",
            "qwen3",
            "--limit-mm-per-prompt",
            '{"image":2,"video":0}',
            "--default-chat-template-kwargs",
            '{"enable_thinking":false}',
        ]
    )
    return command


def _nim_model_from_image(image: Any) -> str | None:
    image_name = str(image).split("@", maxsplit=1)[0]
    if "/nim/" not in image_name:
        return None
    repository = image_name.split("/nim/", maxsplit=1)[1]
    return repository.rsplit(":", maxsplit=1)[0]


def _argument_value(command: list[str], parameter: str) -> str | None:
    try:
        return command[command.index(parameter) + 1]
    except (ValueError, IndexError):
        return None


def _add_issue(
    issues: list[ValidationIssue],
    category: str,
    service: str,
    parameter: str,
    actual: Any,
    rule: str,
    baseline: Any | None = None,
) -> None:
    issues.append(
        ValidationIssue(
            category,
            service,
            parameter,
            actual,
            rule,
            baseline,
        )
    )


def _validate_integer(
    issues: list[ValidationIssue],
    service: str,
    parameter: str,
    actual: Any,
    minimum: int,
    maximum: int,
) -> int | None:
    text = str(actual) if actual is not None else ""
    if isinstance(actual, bool) or re.fullmatch(r"[0-9]+", text) is None:
        _add_issue(
            issues,
            "configuration",
            service,
            parameter,
            actual,
            f"must be an integer in range {minimum}..{maximum}",
        )
        return None
    value = int(text)
    if not minimum <= value <= maximum:
        _add_issue(
            issues,
            "configuration",
            service,
            parameter,
            actual,
            f"must be an integer in range {minimum}..{maximum}",
        )
        return None
    return value


def _validate_float(
    issues: list[ValidationIssue],
    service: str,
    parameter: str,
    actual: Any,
    minimum_exclusive: float,
    maximum: float,
) -> float | None:
    try:
        value = float(actual)
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value) or not minimum_exclusive < value <= maximum:
        _add_issue(
            issues,
            "configuration",
            service,
            parameter,
            actual,
            f"must be a number greater than {minimum_exclusive} and at most {maximum}",
        )
        return None
    return value


def _validate_boolean(
    issues: list[ValidationIssue],
    service: str,
    environment: dict[str, Any],
    parameter: str,
) -> bool | None:
    actual = environment.get(parameter)
    normalized = str(actual).lower()
    if normalized not in {"true", "false"}:
        _add_issue(
            issues,
            "configuration",
            service,
            parameter,
            actual,
            "must be true or false",
        )
        return None
    return normalized == "true"


def _validate_exact(
    issues: list[ValidationIssue],
    service: str,
    parameter: str,
    actual: Any,
    expected: Any,
    rule: str,
) -> None:
    if str(actual).lower() != str(expected).lower():
        _add_issue(issues, "safety", service, parameter, actual, rule)


def _validate_internal_endpoint(
    issues: list[ValidationIssue],
    service: str,
    parameter: str,
    actual: Any,
    expected_service: str,
    expected_port: int,
    expected_path: str,
) -> None:
    parsed = urlparse(str(actual))
    try:
        parsed_port = parsed.port
    except ValueError:
        parsed_port = None
    if (
        parsed.scheme != "http"
        or parsed.hostname != expected_service
        or parsed_port != expected_port
        or parsed.path.rstrip("/") != expected_path.rstrip("/")
    ):
        _add_issue(
            issues,
            "safety",
            service,
            parameter,
            actual,
            f"must target resolved service {expected_service}:{expected_port}{expected_path}",
        )


def _resolved_internal_port(services: dict[str, Any], service_name: str) -> int | None:
    """Read a service's internal port from resolved Compose metadata."""
    service = services.get(service_name, {})
    if not isinstance(service, dict):
        return None
    raw_expose = service.get("expose")
    raw_ports = service.get("ports")
    if raw_expose == [] and raw_ports == []:
        return None
    if raw_ports is not None and not isinstance(raw_ports, list):
        return None
    if raw_expose is not None and not isinstance(raw_expose, list | str | int):
        return None
    metadata_values: list[Any] = []
    if isinstance(raw_expose, list):
        metadata_values.extend(
            value for value in raw_expose if "/udp" not in str(value).lower()
        )
    elif raw_expose is not None:
        if "/udp" not in str(raw_expose).lower():
            metadata_values.append(raw_expose)
    if isinstance(raw_ports, list):
        for port in raw_ports:
            if not isinstance(port, dict):
                return None
            if str(port.get("protocol", "tcp")).lower() != "tcp":
                continue
            target = port.get("target")
            if isinstance(target, bool | float) or not isinstance(target, str | int):
                return None
            metadata_values.append(target)
    for value in metadata_values:
        if isinstance(value, bool | float):
            return None
        port_text = str(value).split("/", maxsplit=1)[0]
        if re.fullmatch(r"[0-9]+", port_text) is None or not (
            1 <= int(port_text) <= 65535
        ):
            return None
    healthcheck_text = json.dumps(service.get("healthcheck", {}), sort_keys=True)
    listener_match = re.search(
        r"(?:localhost|127\.0\.0\.1|%s):([0-9]+)" % re.escape(service_name),
        healthcheck_text,
    )
    healthcheck_port = (
        int(listener_match.group(1))
        if listener_match and 1 <= int(listener_match.group(1)) <= 65535
        else None
    )
    if healthcheck_port is not None:
        listener_values = metadata_values
        parsed_values = {
            int(str(value).split("/", maxsplit=1)[0])
            for value in listener_values
            if isinstance(value, str | int)
            and not isinstance(value, bool)
            and re.fullmatch(r"[0-9]+(?:/tcp)?", str(value), re.IGNORECASE)
            and 1 <= int(str(value).split("/", maxsplit=1)[0]) <= 65535
        }
        if (raw_expose or raw_ports) and not parsed_values:
            return None
        if parsed_values and healthcheck_port not in parsed_values:
            return None
        return healthcheck_port
    exposed = raw_expose or []
    if isinstance(exposed, str | int):
        exposed = [exposed]
    candidates: set[int] = set()
    for value in exposed:
        match = re.fullmatch(r"([0-9]+)(?:/(tcp|udp))?", str(value), re.IGNORECASE)
        if match and (match.group(2) or "tcp").lower() != "tcp":
            continue
        if match:
            port = int(match.group(1))
            if 1 <= port <= 65535:
                candidates.add(port)
    ports = raw_ports or []
    for port in ports:
        if isinstance(port, dict):
            if str(port.get("protocol", "tcp")).lower() != "tcp":
                continue
            target = port.get("target")
            if isinstance(target, bool | float) or not isinstance(target, str | int):
                continue
            if re.fullmatch(r"[0-9]+", str(target)) is None:
                continue
            target = int(target)
            if 1 <= target <= 65535:
                candidates.add(target)
    if len(candidates) == 1:
        return candidates.pop()
    if raw_expose is not None or raw_ports is not None:
        return None
    return None


def _require_internal_port(
    issues: list[ValidationIssue], services: dict[str, Any], service_name: str
) -> int | None:
    port = _resolved_internal_port(services, service_name)
    if port is None:
        service = services.get(service_name, {})
        rule = (
            "must declare a valid TCP internal port in range 1..65535 "
            "and agree with the healthcheck listener"
        )
        if isinstance(service, dict):
            expose = service.get("expose")
            tcp_values = {
                int(str(value).split("/", maxsplit=1)[0])
                for value in (expose if isinstance(expose, list) else [expose])
                if isinstance(value, str | int)
                and not isinstance(value, bool)
                and "/udp" not in str(value).lower()
                and re.fullmatch(r"[0-9]+", str(value).split("/", maxsplit=1)[0])
                and 1 <= int(str(value).split("/", maxsplit=1)[0]) <= 65535
            }
            if len(tcp_values) > 1 and not re.search(
                r"(?:localhost|127\.0\.0\.1|%s):[0-9]+" % re.escape(service_name),
                json.dumps(service.get("healthcheck", {})),
            ):
                rule = "internal TCP port topology is ambiguous; healthcheck must identify one listener"
        _add_issue(
            issues,
            "safety",
            service_name,
            "internal_port",
            {
                "expose": service.get("expose") if isinstance(service, dict) else None,
                "ports": service.get("ports") if isinstance(service, dict) else None,
            },
            rule,
        )
    return port


def _validate_qwen_role(
    issues: list[ValidationIssue],
    service: str,
    environment: dict[str, Any],
    model_parameter: str,
    endpoint_parameter: str,
    served_model: str | None,
    qwen_port: int,
) -> None:
    if environment.get(model_parameter) != served_model:
        _add_issue(
            issues,
            "safety",
            service,
            model_parameter,
            environment.get(model_parameter),
            f"must match qwen-vllm --served-model-name ({served_model})",
        )
    _validate_internal_endpoint(
        issues,
        service,
        endpoint_parameter,
        environment.get(endpoint_parameter),
        "qwen-vllm",
        qwen_port,
        "/v1",
    )


def validate_general_config(config: dict[str, Any]) -> list[ValidationIssue]:
    """Validate safety, topology, types, ranges, and cross-field relationships."""
    services = config.get("services", {})
    issues: list[ValidationIssue] = []
    if not isinstance(services, dict):
        _add_issue(
            issues,
            "safety",
            "compose",
            "services",
            type(services).__name__,
            "must be a resolved Compose service mapping",
        )
        return issues

    missing = REQUIRED_SERVICES - services.keys()
    if missing:
        _add_issue(
            issues,
            "safety",
            "compose",
            "services",
            sorted(missing),
            "must include every required Qwen H100 service",
        )
    forbidden = FORBIDDEN_SERVICES & services.keys()
    if forbidden:
        _add_issue(
            issues,
            "safety",
            "compose",
            "services",
            sorted(forbidden),
            "must exclude default or incompatible services",
        )

    actual_ports: set[tuple[str, str, int]] = set()
    published_bindings: set[tuple[str, str]] = set()
    for service_name, service in services.items():
        if not isinstance(service, dict):
            continue
        service_ports = service.get("ports", [])
        if not isinstance(service_ports, list):
            _add_issue(
                issues,
                "safety",
                service_name,
                "ports",
                service_ports,
                "must be a resolved list of port mappings",
            )
            continue
        for port in service_ports:
            if not isinstance(port, dict):
                _add_issue(
                    issues,
                    "safety",
                    service_name,
                    "ports",
                    port,
                    "must be a resolved port mapping",
                )
                continue
            host_ip = port.get("host_ip")
            published = str(port.get("published", ""))
            target = port.get("target")
            if host_ip != "127.0.0.1":
                _add_issue(
                    issues,
                    "safety",
                    service_name,
                    "ports.host_ip",
                    host_ip,
                    "published ports must bind only to 127.0.0.1",
                )
            published_port = int(published) if re.fullmatch(r"[0-9]+", published) else 0
            if not 1 <= published_port <= 65535:
                _add_issue(
                    issues,
                    "safety",
                    service_name,
                    "ports.published",
                    published,
                    "must be an integer port in range 1..65535",
                )
                continue
            target_port = (
                int(target)
                if isinstance(target, str | int)
                and not isinstance(target, bool)
                and re.fullmatch(r"[0-9]+", str(target))
                else 0
            )
            if not 1 <= target_port <= 65535:
                _add_issue(
                    issues,
                    "safety",
                    service_name,
                    "ports.target",
                    target,
                    "must be an integer port in range 1..65535",
                )
                continue
            binding = (str(host_ip), published)
            if binding in published_bindings:
                _add_issue(
                    issues,
                    "safety",
                    service_name,
                    "ports.published",
                    published,
                    "published host ports must be unique",
                )
            published_bindings.add(binding)
            actual_ports.add((service_name, published, target_port))
    if actual_ports != EXPECTED_PORTS:
        _add_issue(
            issues,
            "safety",
            "compose",
            "published_ports",
            sorted(actual_ports),
            f"must match the fixed loopback contract {sorted(EXPECTED_PORTS)}",
        )

    qwen = services.get("qwen-vllm", {})
    image = str(qwen.get("image", ""))
    if re.search(r"@sha256:[0-9a-f]{64}$", image) is None:
        _add_issue(
            issues,
            "safety",
            "qwen-vllm",
            "image",
            image,
            "must be pinned by a sha256 digest",
        )
    command = _command(qwen)
    qwen_model = command[0] if command and not command[0].startswith("--") else None
    served_model = _argument_value(command, "--served-model-name")
    revision = _argument_value(command, "--revision")
    qwen_port = _require_internal_port(issues, services, "qwen-vllm")
    for parameter, actual in (
        ("model", qwen_model),
        ("--served-model-name", served_model),
        ("--revision", revision),
    ):
        if not actual:
            _add_issue(
                issues,
                "configuration",
                "qwen-vllm",
                parameter,
                actual,
                "must be a non-empty string",
            )

    _validate_exact(
        issues,
        "qwen-vllm",
        "--host",
        _argument_value(command, "--host"),
        "0.0.0.0",
        "must listen on the container network interface",
    )
    _validate_exact(
        issues,
        "qwen-vllm",
        "--port",
        _argument_value(command, "--port"),
        str(qwen_port) if qwen_port is not None else None,
        f"must use the resolved internal endpoint port {qwen_port}",
    )
    _validate_exact(
        issues,
        "qwen-vllm",
        "--tensor-parallel-size",
        _argument_value(command, "--tensor-parallel-size"),
        "1",
        "single-H100 topology requires tensor parallel size 1",
    )
    context_length = _validate_integer(
        issues,
        "qwen-vllm",
        "--max-model-len",
        _argument_value(command, "--max-model-len"),
        1024,
        131072,
    )
    _validate_float(
        issues,
        "qwen-vllm",
        "--gpu-memory-utilization",
        _argument_value(command, "--gpu-memory-utilization"),
        0.0,
        0.95,
    )
    _validate_integer(
        issues,
        "qwen-vllm",
        "--max-num-seqs",
        _argument_value(command, "--max-num-seqs"),
        1,
        8,
    )
    raw_image_limit = _argument_value(command, "--limit-mm-per-prompt")
    try:
        multimodal_limit = json.loads(raw_image_limit or "")
    except (TypeError, json.JSONDecodeError):
        multimodal_limit = {}
    qwen_image_limit = _validate_integer(
        issues,
        "qwen-vllm",
        "QWEN_MAX_IMAGES_PER_PROMPT",
        multimodal_limit.get("image"),
        1,
        32,
    )
    if multimodal_limit.get("video") != 0:
        _add_issue(
            issues,
            "safety",
            "qwen-vllm",
            "--limit-mm-per-prompt.video",
            multimodal_limit.get("video"),
            "video input must remain disabled on the certified topology",
        )

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
            _add_issue(
                issues,
                "safety",
                service_name,
                "deploy.resources.reservations.devices.device_ids",
                sorted(device_ids),
                "all GPU services must use only GPU 0",
            )

    for service_name in REQUIRED_SERVICES & services.keys():
        service = services[service_name]
        if "healthcheck" not in service:
            _add_issue(
                issues,
                "safety",
                service_name,
                "healthcheck",
                None,
                "required services must define a healthcheck",
            )
        if service.get("restart") != "unless-stopped":
            _add_issue(
                issues,
                "safety",
                service_name,
                "restart",
                service.get("restart"),
                "required services must use restart: unless-stopped",
            )

    rag_environment = services.get("rag-server", {}).get("environment", {})
    ingestor_environment = services.get("ingestor-server", {}).get("environment", {})
    runtime_environment = services.get("nv-ingest-ms-runtime", {}).get(
        "environment", {}
    )
    runtime_port = _require_internal_port(issues, services, "nv-ingest-ms-runtime")
    qwen_roles = [
        ("APP_LLM_MODELNAME", "APP_LLM_SERVERURL"),
        ("APP_VLM_MODELNAME", "APP_VLM_SERVERURL"),
        ("APP_QUERYREWRITER_MODELNAME", "APP_QUERYREWRITER_SERVERURL"),
        (
            "APP_FILTEREXPRESSIONGENERATOR_MODELNAME",
            "APP_FILTEREXPRESSIONGENERATOR_SERVERURL",
        ),
        ("REFLECTION_LLM", "REFLECTION_LLM_SERVERURL"),
    ]
    for role in ("PLANNER", "TASK", "SEED_GEN", "SYNTHESIS"):
        qwen_roles.append(
            (f"AGENTIC_{role}_LLM_MODEL", f"AGENTIC_{role}_LLM_SERVERURL")
        )
    for model_parameter, endpoint_parameter in qwen_roles:
        _validate_qwen_role(
            issues,
            "rag-server",
            rag_environment,
            model_parameter,
            endpoint_parameter,
            served_model,
            qwen_port or 0,
        )
    if ingestor_environment.get("SUMMARY_LLM") != served_model:
        _add_issue(
            issues,
            "safety",
            "ingestor-server",
            "SUMMARY_LLM",
            ingestor_environment.get("SUMMARY_LLM"),
            f"must match qwen-vllm --served-model-name ({served_model})",
        )
    _validate_internal_endpoint(
        issues,
        "ingestor-server",
        "SUMMARY_LLM_SERVERURL",
        ingestor_environment.get("SUMMARY_LLM_SERVERURL"),
        "qwen-vllm",
        qwen_port or 0,
        "/v1",
    )
    runtime_hostname = str(
        ingestor_environment.get("APP_NVINGEST_MESSAGECLIENTHOSTNAME", "")
    ).strip("\"'")
    if runtime_hostname != "nv-ingest-ms-runtime" or runtime_hostname not in services:
        _add_issue(
            issues,
            "safety",
            "ingestor-server",
            "APP_NVINGEST_MESSAGECLIENTHOSTNAME",
            ingestor_environment.get("APP_NVINGEST_MESSAGECLIENTHOSTNAME"),
            "must target resolved service nv-ingest-ms-runtime",
        )
    _validate_exact(
        issues,
        "ingestor-server",
        "APP_NVINGEST_MESSAGECLIENTPORT",
        ingestor_environment.get("APP_NVINGEST_MESSAGECLIENTPORT"),
        str(runtime_port) if runtime_port is not None else None,
        "must use the resolved nv-ingest-ms-runtime internal messaging port",
    )
    for parameter, expected_service in (
        ("APP_NVINGEST_OCRURL", "nemotron-ocr"),
        ("APP_NVINGEST_PAGEELEMENTSURL", "page-elements"),
        ("APP_NVINGEST_GRAPHICELEMENTSURL", "graphic-elements"),
        ("APP_NVINGEST_TABLESTRUCTUREURL", "table-structure"),
    ):
        _validate_internal_endpoint(
            issues,
            "ingestor-server",
            parameter,
            ingestor_environment.get(parameter),
            expected_service,
            _require_internal_port(issues, services, expected_service) or 0,
            "/v1/infer",
        )

    embedding_service_model = _nim_model_from_image(
        services.get("nemotron-vlm-embedding-ms", {}).get("image")
    )
    embedding_port = _require_internal_port(
        issues, services, "nemotron-vlm-embedding-ms"
    )
    for service_name, environment in (
        ("rag-server", rag_environment),
        ("ingestor-server", ingestor_environment),
        ("nv-ingest-ms-runtime", runtime_environment),
    ):
        if (
            not embedding_service_model
            or environment.get("APP_EMBEDDINGS_MODELNAME") != embedding_service_model
        ):
            _add_issue(
                issues,
                "safety",
                service_name,
                "APP_EMBEDDINGS_MODELNAME",
                environment.get("APP_EMBEDDINGS_MODELNAME"),
                "must match the model identity derived from the resolved "
                f"nemotron-vlm-embedding-ms image ({embedding_service_model})",
            )
        _validate_internal_endpoint(
            issues,
            service_name,
            "APP_EMBEDDINGS_SERVERURL",
            environment.get("APP_EMBEDDINGS_SERVERURL"),
            "nemotron-vlm-embedding-ms",
            embedding_port or 0,
            "/v1",
        )
    embedding_dimensions = _validate_integer(
        issues,
        "rag-server",
        "APP_EMBEDDINGS_DIMENSIONS",
        rag_environment.get("APP_EMBEDDINGS_DIMENSIONS"),
        1,
        65536,
    )
    if embedding_dimensions is not None and str(
        ingestor_environment.get("APP_EMBEDDINGS_DIMENSIONS")
    ) != str(embedding_dimensions):
        _add_issue(
            issues,
            "safety",
            "ingestor-server",
            "APP_EMBEDDINGS_DIMENSIONS",
            ingestor_environment.get("APP_EMBEDDINGS_DIMENSIONS"),
            f"must match rag-server APP_EMBEDDINGS_DIMENSIONS ({embedding_dimensions})",
        )
    ranking_service_model = _nim_model_from_image(
        services.get("nemotron-ranking-vl-ms", {}).get("image")
    )
    if rag_environment.get("APP_RANKING_MODELNAME") != ranking_service_model:
        _add_issue(
            issues,
            "safety",
            "rag-server",
            "APP_RANKING_MODELNAME",
            rag_environment.get("APP_RANKING_MODELNAME"),
            "must match the model identity derived from the resolved "
            f"nemotron-ranking-vl-ms image ({ranking_service_model})",
        )
    _validate_internal_endpoint(
        issues,
        "rag-server",
        "APP_RANKING_SERVERURL",
        rag_environment.get("APP_RANKING_SERVERURL"),
        "nemotron-ranking-vl-ms",
        _require_internal_port(issues, services, "nemotron-ranking-vl-ms") or 0,
        "",
    )
    for service_name, environment, model_key, endpoint_key in (
        (
            "ingestor-server",
            ingestor_environment,
            "APP_NVINGEST_CAPTIONMODELNAME",
            "APP_NVINGEST_CAPTIONENDPOINTURL",
        ),
        (
            "nv-ingest-ms-runtime",
            runtime_environment,
            "VLM_CAPTION_MODEL_NAME",
            "VLM_CAPTION_ENDPOINT",
        ),
    ):
        if environment.get(model_key) != served_model:
            _add_issue(
                issues,
                "safety",
                service_name,
                model_key,
                environment.get(model_key),
                f"must match qwen-vllm --served-model-name ({served_model})",
            )
        _validate_internal_endpoint(
            issues,
            service_name,
            endpoint_key,
            environment.get(endpoint_key),
            "qwen-vllm",
            qwen_port or 0,
            "/v1/chat/completions",
        )

    candidate_pool = _validate_integer(
        issues,
        "rag-server",
        "VECTOR_DB_TOPK",
        rag_environment.get("VECTOR_DB_TOPK"),
        1,
        400,
    )
    final_top_k = _validate_integer(
        issues,
        "rag-server",
        "APP_RETRIEVER_TOPK",
        rag_environment.get("APP_RETRIEVER_TOPK"),
        1,
        400,
    )
    if (
        candidate_pool is not None
        and final_top_k is not None
        and final_top_k > candidate_pool
    ):
        _add_issue(
            issues,
            "configuration",
            "rag-server",
            "APP_RETRIEVER_TOPK,VECTOR_DB_TOPK",
            {
                "APP_RETRIEVER_TOPK": final_top_k,
                "VECTOR_DB_TOPK": candidate_pool,
            },
            "APP_RETRIEVER_TOPK must not exceed VECTOR_DB_TOPK",
        )
    conversation_history = _validate_integer(
        issues,
        "rag-server",
        "CONVERSATION_HISTORY",
        rag_environment.get("CONVERSATION_HISTORY"),
        0,
        100,
    )
    query_rewriter = _validate_boolean(
        issues, "rag-server", rag_environment, "ENABLE_QUERYREWRITER"
    )
    if query_rewriter and conversation_history == 0:
        _add_issue(
            issues,
            "configuration",
            "rag-server",
            "ENABLE_QUERYREWRITER,CONVERSATION_HISTORY",
            {
                "ENABLE_QUERYREWRITER": True,
                "CONVERSATION_HISTORY": conversation_history,
            },
            "query rewriting requires CONVERSATION_HISTORY greater than 0",
        )
    _validate_integer(
        issues,
        "rag-server",
        "MAX_RECURSION_DEPTH",
        rag_environment.get("MAX_RECURSION_DEPTH"),
        1,
        20,
    )
    _validate_integer(
        issues,
        "rag-server",
        "MAX_REFLECTION_LOOP",
        rag_environment.get("MAX_REFLECTION_LOOP"),
        1,
        20,
    )
    _validate_integer(
        issues,
        "rag-server",
        "AGENTIC_CONCURRENCY_LIMIT",
        rag_environment.get("AGENTIC_CONCURRENCY_LIMIT"),
        1,
        16,
    )

    rag_image_budget = _validate_integer(
        issues,
        "rag-server",
        "APP_VLM_MAX_TOTAL_IMAGES",
        rag_environment.get("APP_VLM_MAX_TOTAL_IMAGES"),
        1,
        32,
    )
    if (
        rag_image_budget is not None
        and qwen_image_limit is not None
        and rag_image_budget > qwen_image_limit
    ):
        _add_issue(
            issues,
            "configuration",
            "rag-server,qwen-vllm",
            "APP_VLM_MAX_TOTAL_IMAGES,QWEN_MAX_IMAGES_PER_PROMPT",
            {
                "APP_VLM_MAX_TOTAL_IMAGES": rag_image_budget,
                "QWEN_MAX_IMAGES_PER_PROMPT": qwen_image_limit,
            },
            "RAG image budget must not exceed the Qwen image-per-prompt limit",
        )

    token_settings = (
        ("rag-server", rag_environment, "LLM_MAX_TOKENS"),
        ("rag-server", rag_environment, "APP_VLM_MAX_TOKENS"),
        ("rag-server", rag_environment, "AGENTIC_CONTEXT_MAX_TOKENS"),
        (
            "rag-server",
            rag_environment,
            "APP_FILTEREXPRESSIONGENERATOR_MAXTOKENS",
        ),
        ("rag-server", rag_environment, "AGENTIC_PLANNER_LLM_MAX_TOKENS"),
        ("rag-server", rag_environment, "AGENTIC_TASK_LLM_MAX_TOKENS"),
        ("rag-server", rag_environment, "AGENTIC_SEED_GEN_LLM_MAX_TOKENS"),
        ("rag-server", rag_environment, "AGENTIC_SYNTHESIS_LLM_MAX_TOKENS"),
        (
            "ingestor-server",
            ingestor_environment,
            "SUMMARY_LLM_MAX_CHUNK_LENGTH",
        ),
    )
    if context_length is not None:
        for service_name, environment, parameter in token_settings:
            _validate_integer(
                issues,
                service_name,
                parameter,
                environment.get(parameter),
                1,
                context_length,
            )
    _validate_integer(
        issues,
        "ingestor-server",
        "NV_INGEST_FILES_PER_BATCH",
        ingestor_environment.get("NV_INGEST_FILES_PER_BATCH"),
        1,
        250,
    )
    _validate_integer(
        issues,
        "ingestor-server",
        "NV_INGEST_CONCURRENT_BATCHES",
        ingestor_environment.get("NV_INGEST_CONCURRENT_BATCHES"),
        1,
        16,
    )
    _validate_integer(
        issues,
        "ingestor-server",
        "SUMMARY_MAX_PARALLELIZATION",
        ingestor_environment.get("SUMMARY_MAX_PARALLELIZATION"),
        1,
        64,
    )

    for parameter in (
        "ENABLE_QUERY_DECOMPOSITION",
        "ENABLE_FILTER_GENERATOR",
        "ENABLE_REFLECTION",
        "ENABLE_AGENTIC_RAG",
    ):
        _validate_boolean(issues, "rag-server", rag_environment, parameter)
    for parameter, expected in (
        ("ENABLE_VLM_INFERENCE", "true"),
        ("VLM_TO_LLM_FALLBACK", "false"),
        ("ENABLE_VLM_RERANKER_IMAGE_INPUT", "true"),
    ):
        _validate_exact(
            issues,
            "rag-server",
            parameter,
            rag_environment.get(parameter),
            expected,
            f"Qwen H100 multimodal topology requires {parameter}={expected}",
        )

    fixed_ingestion_settings = {
        "APP_NVINGEST_EXTRACTTEXT": "true",
        "APP_NVINGEST_EXTRACTINFOGRAPHICS": "true",
        "APP_NVINGEST_EXTRACTTABLES": "true",
        "APP_NVINGEST_EXTRACTCHARTS": "true",
        "APP_NVINGEST_EXTRACTIMAGES": "true",
        "APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY": "text_image",
        "APP_NVINGEST_IMAGE_ELEMENTS_MODALITY": "image",
        "APP_NVINGEST_EXTRACTTABLESMETHOD": "yolox",
        "APP_NVINGEST_EXTRACTPAGEASIMAGE": "false",
        "APP_NVINGEST_SEGMENTAUDIO": "false",
    }
    for parameter, expected in fixed_ingestion_settings.items():
        _validate_exact(
            issues,
            "ingestor-server",
            parameter,
            ingestor_environment.get(parameter),
            expected,
            f"single-H100 ingestion topology requires {parameter}={expected}",
        )
    pdf_method = str(ingestor_environment.get("APP_NVINGEST_PDFEXTRACTMETHOD")).lower()
    if pdf_method not in {"none", "pdfium"}:
        _add_issue(
            issues,
            "safety",
            "ingestor-server",
            "APP_NVINGEST_PDFEXTRACTMETHOD",
            ingestor_environment.get("APP_NVINGEST_PDFEXTRACTMETHOD"),
            "must be None or pdfium; Nemotron Parse is excluded from this topology",
        )
    return issues


def validate_certified_profile(
    config: dict[str, Any], profile: str
) -> list[ValidationIssue]:
    """Report drift from an evidence-backed FP8 or NVFP4 H100 profile."""
    if profile not in MODEL_VARIANTS:
        raise ValueError(f"unknown certified profile: {profile}")
    services = config.get("services", {})
    command = _command(services.get("qwen-vllm", {}))
    model_variant = MODEL_VARIANTS[profile]
    issues: list[ValidationIssue] = []

    def record_profile_drift_if_changed(
        service: str, parameter: str, actual: Any, baseline: Any
    ) -> None:
        if str(actual) != str(baseline):
            _add_issue(
                issues,
                "profile_drift",
                service,
                parameter,
                actual,
                f"must equal the certified {profile} baseline",
                baseline,
            )

    record_profile_drift_if_changed(
        "qwen-vllm",
        "model",
        command[0] if command else None,
        model_variant["model"],
    )
    record_profile_drift_if_changed(
        "qwen-vllm",
        "--served-model-name",
        _argument_value(command, "--served-model-name"),
        model_variant["model"],
    )
    record_profile_drift_if_changed(
        "qwen-vllm",
        "--revision",
        _argument_value(command, "--revision"),
        model_variant["revision"],
    )
    record_profile_drift_if_changed(
        "qwen-vllm",
        "command_signature",
        _safe_qwen_command(command),
        _certified_qwen_command(profile),
    )
    for parameter, baseline in (
        ("--max-model-len", "32768"),
        ("--gpu-memory-utilization", model_variant["memory_utilization"]),
        ("--max-num-seqs", "1"),
        ("--limit-mm-per-prompt", '{"image":2,"video":0}'),
        ("--default-chat-template-kwargs", '{"enable_thinking":false}'),
    ):
        record_profile_drift_if_changed(
            "qwen-vllm", parameter, _argument_value(command, parameter), baseline
        )
    rag_environment = services.get("rag-server", {}).get("environment", {})
    ingestor_environment = services.get("ingestor-server", {}).get("environment", {})
    for parameter, baseline in CERTIFIED_RAG_SETTINGS.items():
        record_profile_drift_if_changed(
            "rag-server", parameter, rag_environment.get(parameter), baseline
        )
    for parameter, baseline in CERTIFIED_INGESTOR_SETTINGS.items():
        record_profile_drift_if_changed(
            "ingestor-server", parameter, ingestor_environment.get(parameter), baseline
        )
    return issues


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
            "command": (
                _safe_qwen_command(_command(service), include_unknown_options=False)
                if service_name == "qwen-vllm"
                else []
            ),
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
            settings[key] = str(_redact_secrets(environment[key], key))

    general_issues = validate_general_config(config)
    profile_status: dict[str, str] = {"kind": "invalid" if general_issues else "custom"}
    certified_evidence: dict[str, Any] | None = None
    for profile in MODEL_VARIANTS if not general_issues else ():
        if not validate_certified_profile(config, profile):
            command = _command(services.get("qwen-vllm", {}))
            profile_status = {"kind": "certified", "profile": profile}
            certified_evidence = {
                "model": command[0] if command else None,
                "model_revision": _argument_value(command, "--revision"),
                "resource_envelope": {
                    "gpu_device_ids": service_summary["qwen-vllm"]["gpu_device_ids"],
                    "gpu_memory_utilization": _argument_value(
                        command, "--gpu-memory-utilization"
                    ),
                    "max_model_len": _argument_value(command, "--max-model-len"),
                    "max_num_seqs": _argument_value(command, "--max-num-seqs"),
                },
                "tuning": {
                    **CERTIFIED_RAG_SETTINGS,
                    **CERTIFIED_INGESTOR_SETTINGS,
                    "QWEN_MAX_IMAGES_PER_PROMPT": "2",
                },
            }
            break

    return _redact_secrets(
        {
            "certified_evidence": certified_evidence,
            "forbidden_services_present": sorted(FORBIDDEN_SERVICES & services.keys()),
            "profile_status": profile_status,
            "required_services": sorted(REQUIRED_SERVICES),
            "services": service_summary,
            "settings": settings,
        }
    )


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
    certified_parser = subparsers.add_parser(
        "validate-certified-config",
        help="Check exact conformity with an evidence-backed H100 profile",
    )
    certified_parser.add_argument("config", help="Resolved Compose JSON or - for stdin")
    certified_parser.add_argument(
        "--profile", choices=tuple(MODEL_VARIANTS), required=True
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
        config = _load_config(arguments.config)
        errors = validate_general_config(config)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        status = summarize_config(config)["profile_status"]
        label = (
            f"certified {status['profile']}"
            if status["kind"] == "certified"
            else "custom (not certified)"
        )
        print(f"Qwen H100 compose configuration is valid: {label}")
        return 0
    if arguments.command == "validate-certified-config":
        drift = validate_certified_profile(
            _load_config(arguments.config), arguments.profile
        )
        if drift:
            for issue in drift:
                print(f"DRIFT: {issue}", file=sys.stderr)
            return 1
        print(f"Qwen H100 configuration matches certified {arguments.profile}")
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
