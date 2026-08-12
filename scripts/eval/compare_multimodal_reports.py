# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Compare two compatible public multimodal evaluation reports."""

import argparse
import json
from pathlib import Path
from typing import Any, Literal

QUALITY_METRICS = (
    "candidate_retrieval_hit_at_k",
    "verified_identification_accuracy",
    "unsupported_claim_rate",
    "low_confidence_rejection_precision",
    "low_confidence_rejection_recall",
    "rejection_citation_leak_rate",
)
LATENCY_METRICS = ("ttft", "total")
PERCENTILES = ("p50", "p95")
Selection = Literal["baseline", "candidate", "neither"]


def _identity(report: dict[str, Any]) -> tuple[Any, ...]:
    dataset = report.get("dataset", {})
    configuration = report.get("configuration", {})
    assets = sorted(
        (asset.get("path"), asset.get("sha256"))
        for asset in dataset.get("assets", [])
    )
    return (
        dataset.get("version"),
        dataset.get("manifest"),
        tuple(assets),
        configuration.get("model"),
        configuration.get("collection"),
    )


def _validate_compatible(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    baseline_dataset = baseline.get("dataset", {})
    candidate_dataset = candidate.get("dataset", {})
    if baseline_dataset.get("version") != candidate_dataset.get("version"):
        raise ValueError("dataset version differs")
    if baseline_dataset.get("manifest") != candidate_dataset.get("manifest"):
        raise ValueError("dataset manifest differs")
    baseline_manifest_hash = baseline_dataset.get("manifest_sha256")
    candidate_manifest_hash = candidate_dataset.get("manifest_sha256")
    if not baseline_manifest_hash or not candidate_manifest_hash:
        raise ValueError("dataset manifest hash unavailable")
    if baseline_manifest_hash != candidate_manifest_hash:
        raise ValueError("dataset manifest hash differs")
    if _identity(baseline)[2] != _identity(candidate)[2]:
        raise ValueError("assets differ")
    baseline_config = baseline.get("configuration", {})
    candidate_config = candidate.get("configuration", {})
    if baseline_config.get("model") != candidate_config.get("model"):
        raise ValueError("model differs")
    if baseline_config.get("collection") != candidate_config.get("collection"):
        raise ValueError("collection differs")
    baseline_case_ids = [case.get("case_id") for case in baseline.get("cases", [])]
    candidate_case_ids = [case.get("case_id") for case in candidate.get("cases", [])]
    if (
        len(set(baseline_case_ids)) != len(baseline_case_ids)
        or len(set(candidate_case_ids)) != len(candidate_case_ids)
        or sorted(baseline_case_ids) != sorted(candidate_case_ids)
    ):
        raise ValueError("case ids differ")


def _server_provenance_complete(report: dict[str, Any]) -> bool:
    deployment = report.get("configuration", {}).get("server_deployment", {})
    return deployment.get("provenance") == "explicit-file-complete" and not deployment.get(
        "unavailable_keys"
    )


def _changed_calibration_keys(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[str]:
    baseline_settings = baseline.get("configuration", {}).get("server_deployment", {}).get("settings", {})
    candidate_settings = candidate.get("configuration", {}).get("server_deployment", {}).get("settings", {})
    baseline_request = baseline.get("configuration", {}).get("request_fields", {})
    candidate_request = candidate.get("configuration", {}).get("request_fields", {})
    keys = sorted(set(baseline_settings) | set(candidate_settings))
    changed = [key for key in keys if baseline_settings.get(key) != candidate_settings.get(key)]
    request_keys = sorted(set(baseline_request) | set(candidate_request))
    changed.extend(
        f"request.{key}"
        for key in request_keys
        if baseline_request.get(key) != candidate_request.get(key)
    )
    return changed


def _metric(report: dict[str, Any], name: str) -> float | None:
    value = report.get("metrics", {}).get(name)
    return float(value) if isinstance(value, int | float) else None


def _latency_metric(
    report: dict[str, Any], stage: str, percentile: str
) -> float | None:
    value = report.get("metrics", {}).get("latency_seconds", {}).get(stage, {}).get(
        percentile
    )
    return float(value) if isinstance(value, int | float) else None


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    selection_rationale: str = "",
    selected: Selection = "neither",
) -> dict[str, Any]:
    """Return candidate-minus-baseline deltas, refusing incompatible reports."""
    if selected not in {"baseline", "candidate", "neither"}:
        raise ValueError(f"invalid selection: {selected}")
    _validate_compatible(baseline, candidate)
    changed_keys = _changed_calibration_keys(baseline, candidate)
    if not changed_keys:
        raise ValueError("calibration configurations do not differ")
    if selected in {"baseline", "candidate"} and not (
        _server_provenance_complete(baseline) and _server_provenance_complete(candidate)
    ):
        raise ValueError("complete server provenance is required for selection")
    if selected in {"baseline", "candidate"} and not selection_rationale.strip():
        raise ValueError("selection rationale is required for a selected report")
    deltas: dict[str, Any] = {"quality": {}, "latency_seconds": {}}
    unmet_cases: list[str] = []
    for report in (baseline, candidate):
        coverage = report.get("dataset", {}).get("coverage", {})
        required = set(coverage.get("required", []))
        present = set(coverage.get("present", []))
        for item in sorted(required - present):
            unmet_cases.append(f"dataset coverage is unmet: {item}")
        ambiguity = report.get("dataset", {}).get("ambiguity", {})
        if ambiguity.get("required") and not ambiguity.get("present"):
            reason = ambiguity.get("blocked") or "no declaration supplied"
            unmet_cases.append(f"dataset ambiguity coverage is unmet: {reason}")
    for name in QUALITY_METRICS:
        baseline_value = _metric(baseline, name)
        candidate_value = _metric(candidate, name)
        if baseline_value is None or candidate_value is None:
            if baseline_value is None:
                unmet_cases.append(f"baseline metric {name} is unavailable")
            if candidate_value is None:
                unmet_cases.append(f"candidate metric {name} is unavailable")
        else:
            deltas["quality"][name] = candidate_value - baseline_value
    for stage in LATENCY_METRICS:
        deltas["latency_seconds"][stage] = {}
        for percentile in PERCENTILES:
            baseline_value = _latency_metric(baseline, stage, percentile)
            candidate_value = _latency_metric(candidate, stage, percentile)
            if baseline_value is None or candidate_value is None:
                if baseline_value is None:
                    unmet_cases.append(
                        f"baseline latency_seconds.{stage}.{percentile} is unavailable"
                    )
                if candidate_value is None:
                    unmet_cases.append(
                        f"candidate latency_seconds.{stage}.{percentile} is unavailable"
                    )
            else:
                deltas["latency_seconds"][stage][percentile] = (
                    candidate_value - baseline_value
                )
    baseline_cases = {case.get("case_id"): case for case in baseline.get("cases", [])}
    candidate_cases = {
        case.get("case_id"): case for case in candidate.get("cases", [])
    }
    for label, cases in (("baseline", baseline_cases), ("candidate", candidate_cases)):
        for case_id, case in cases.items():
            if case.get("error"):
                unmet_cases.append(f"{label} case {case_id} failed: {case['error']}")
    unmet_cases = list(dict.fromkeys(unmet_cases))
    return {
        "comparison_version": "multimodal-report-comparison-v1",
        "baseline_variant": baseline.get("variant"),
        "candidate_variant": candidate.get("variant"),
        "identity": {
            "dataset_version": baseline.get("dataset", {}).get("version"),
            "manifest": baseline.get("dataset", {}).get("manifest"),
            "asset_count": len(baseline.get("dataset", {}).get("assets", [])),
            "model": baseline.get("configuration", {}).get("model"),
            "collection": baseline.get("configuration", {}).get("collection"),
        },
        "configurations": {
            "baseline": baseline.get("configuration", {}),
            "candidate": candidate.get("configuration", {}),
        },
        "changed_calibration_keys": changed_keys,
        "deltas": deltas,
        "metric_semantics": {
            "verified_identification_accuracy": (
                "lexical diagnostic; manual adjudication is authoritative"
            ),
            "unsupported_claim_rate": (
                "lexical diagnostic unless replaced by manual adjudication"
            ),
        },
        "selection": {"selected": selected, "rationale": selection_rationale},
        "unmet_cases": unmet_cases,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selected",
        choices=("baseline", "candidate", "neither"),
        default="neither",
    )
    parser.add_argument("--selection-rationale", default="")
    return parser.parse_args()


def main() -> int:
    """Compare two compatible reports and write a sanitized comparison report."""
    args = _parse_args()
    result = compare_reports(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
        selection_rationale=args.selection_rationale,
        selected=args.selected,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
