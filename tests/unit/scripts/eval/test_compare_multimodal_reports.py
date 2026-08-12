# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[4] / "scripts" / "eval" / "compare_multimodal_reports.py"
)
SPEC = importlib.util.spec_from_file_location("compare_multimodal_reports", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def _report(**overrides: object) -> dict:
    report = {
        "variant": "baseline",
        "dataset": {
            "version": "dataset-v1",
            "manifest": "cases.json",
            "manifest_sha256": "manifest-hash",
            "assets": [{"path": "image.png", "sha256": "asset-hash"}],
        },
        "configuration": {
            "model": "qwen",
            "collection": "products",
            "request_fields": {"vlm_temperature": 0.0},
            "server_deployment": {
                "provenance": "explicit-file-complete",
                "settings": {"APP_VLM_TEMPERATURE": 0.0},
                "unavailable_keys": [],
            },
        },
        "metrics": {
            "case_count": 2,
            "verified_identification_accuracy": 0.5,
            "unsupported_claim_rate": 0.5,
            "rejection_citation_leak_rate": 0.5,
            "latency_seconds": {
                "ttft": {"p50": 1.0, "p95": 2.0},
                "total": {"p50": 4.0, "p95": 8.0},
            },
        },
        "cases": [{"case_id": "one"}, {"case_id": "two"}],
    }
    report.update(overrides)
    return report


def test_compare_reports_returns_quality_and_latency_deltas() -> None:
    result = comparison.compare_reports(
        _report(),
        _report(
            variant="candidate",
            configuration={
                "model": "qwen",
                "collection": "products",
                "request_fields": {"vlm_temperature": 0.1},
                "server_deployment": {
                    "provenance": "explicit-file-complete",
                    "settings": {"APP_VLM_TEMPERATURE": 0.0},
                    "unavailable_keys": [],
                },
            },
            metrics={
                "case_count": 2,
                "verified_identification_accuracy": 0.75,
                "unsupported_claim_rate": 0.25,
                "rejection_citation_leak_rate": 0.5,
                "latency_seconds": {
                    "ttft": {"p50": 1.5, "p95": 3.0},
                    "total": {"p50": 3.0, "p95": 7.0},
                },
            },
        ),
        selection_rationale="Prefer the candidate because accuracy improved.",
        selected="candidate",
    )

    assert result["deltas"]["quality"]["verified_identification_accuracy"] == 0.25
    assert result["deltas"]["quality"]["unsupported_claim_rate"] == -0.25
    assert result["deltas"]["quality"]["rejection_citation_leak_rate"] == 0.0
    assert result["deltas"]["latency_seconds"]["ttft"]["p95"] == 1.0
    assert result["selection"] == {
        "selected": "candidate",
        "rationale": "Prefer the candidate because accuracy improved.",
    }
    assert result["metric_semantics"]["verified_identification_accuracy"] == (
        "lexical diagnostic; manual adjudication is authoritative"
    )
    assert result["changed_calibration_keys"] == ["request.vlm_temperature"]


def test_compare_reports_fails_closed_for_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="model differs"):
        comparison.compare_reports(
            _report(),
            _report(configuration={"model": "other-qwen", "collection": "products"}),
        )


def test_compare_reports_rejects_changed_manifest_with_same_filename() -> None:
    with pytest.raises(ValueError, match="manifest hash differs"):
        comparison.compare_reports(
            _report(),
            _report(
                dataset={
                    "version": "dataset-v1",
                    "manifest": "cases.json",
                    "manifest_sha256": "other-manifest-hash",
                    "assets": [{"path": "image.png", "sha256": "asset-hash"}],
                }
            ),
        )


def test_compare_reports_lists_unmet_cases_and_missing_latency() -> None:
    candidate = _report(
        configuration={
            "model": "qwen",
            "collection": "products",
            "request_fields": {"vlm_temperature": 0.1},
            "server_deployment": {
                "provenance": "explicit-file-complete",
                "settings": {"APP_VLM_TEMPERATURE": 0.0},
                "unavailable_keys": [],
            },
        },
        metrics={
            "case_count": 2,
            "verified_identification_accuracy": 0.75,
            "unsupported_claim_rate": 0.25,
        },
        cases=[{"case_id": "one", "error": "timeout"}, {"case_id": "two"}],
    )

    result = comparison.compare_reports(_report(), candidate)

    assert (
        "baseline metric candidate_retrieval_hit_at_k is unavailable"
        in result["unmet_cases"]
    )
    assert (
        "candidate metric candidate_retrieval_hit_at_k is unavailable"
        in result["unmet_cases"]
    )
    assert "candidate latency_seconds.ttft.p50 is unavailable" in result["unmet_cases"]
    assert "candidate case one failed: timeout" in result["unmet_cases"]


def test_compare_reports_fails_closed_for_different_case_ids() -> None:
    with pytest.raises(ValueError, match="case ids differ"):
        comparison.compare_reports(
            _report(), _report(cases=[{"case_id": "one"}, {"case_id": "other"}])
        )


def test_selected_report_requires_non_empty_rationale() -> None:
    candidate = _report(
        configuration={
            "model": "qwen",
            "collection": "products",
            "request_fields": {"vlm_temperature": 0.1},
            "server_deployment": {
                "provenance": "explicit-file-complete",
                "settings": {"APP_VLM_TEMPERATURE": 0.0},
                "unavailable_keys": [],
            },
        }
    )
    with pytest.raises(ValueError, match="selection rationale"):
        comparison.compare_reports(_report(), candidate, selected="candidate")


def test_compare_reports_rejects_invalid_selection_direct_call() -> None:
    with pytest.raises(ValueError, match="invalid selection"):
        comparison.compare_reports(_report(), _report(), selected="unknown")


def test_compare_reports_lists_both_missing_metric_sides() -> None:
    baseline = _report(metrics={})
    candidate = _report(
        configuration={
            "model": "qwen",
            "collection": "products",
            "request_fields": {"vlm_temperature": 0.1},
            "server_deployment": {
                "provenance": "unavailable",
                "settings": {"APP_VLM_TEMPERATURE": None},
                "unavailable_keys": ["APP_VLM_TEMPERATURE"],
            },
        },
        metrics={},
    )

    result = comparison.compare_reports(baseline, candidate)

    assert (
        "baseline metric verified_identification_accuracy is unavailable"
        in result["unmet_cases"]
    )
    assert (
        "candidate metric verified_identification_accuracy is unavailable"
        in result["unmet_cases"]
    )
    assert "baseline latency_seconds.ttft.p50 is unavailable" in result["unmet_cases"]
    assert "candidate latency_seconds.ttft.p50 is unavailable" in result["unmet_cases"]


def test_compare_reports_requires_manifest_hash() -> None:
    report = _report()
    del report["dataset"]["manifest_sha256"]
    with pytest.raises(ValueError, match="manifest hash unavailable"):
        comparison.compare_reports(_report(), report)


def test_selected_report_requires_complete_server_provenance() -> None:
    with pytest.raises(ValueError, match="complete server provenance"):
        comparison.compare_reports(
            _report(),
            _report(
                configuration={
                    "model": "qwen",
                    "collection": "products",
                    "request_fields": {"vlm_temperature": 0.1},
                    "server_deployment": {
                        "provenance": "unavailable",
                        "settings": {"APP_VLM_TEMPERATURE": None},
                        "unavailable_keys": ["APP_VLM_TEMPERATURE"],
                    },
                }
            ),
            selected="candidate",
            selection_rationale="test",
        )


def test_compare_reports_surfaces_blocked_manifest_ambiguity() -> None:
    manifest_metadata = {
        "required": True,
        "present": False,
        "blocked": "No suitable ground-truth asset.",
    }
    result = comparison.compare_reports(
        _report(
            dataset={
                "version": "dataset-v1",
                "manifest": "cases.json",
                "manifest_sha256": "manifest-hash",
                "assets": [{"path": "image.png", "sha256": "asset-hash"}],
                "ambiguity": manifest_metadata,
            }
        ),
        _report(
            variant="candidate",
            configuration={
                "model": "qwen",
                "collection": "products",
                "request_fields": {"vlm_temperature": 0.1},
                "server_deployment": {
                    "provenance": "explicit-file-complete",
                    "settings": {"APP_VLM_TEMPERATURE": 0.0},
                    "unavailable_keys": [],
                },
            },
            dataset={
                "version": "dataset-v1",
                "manifest": "cases.json",
                "manifest_sha256": "manifest-hash",
                "assets": [{"path": "image.png", "sha256": "asset-hash"}],
                "ambiguity": manifest_metadata,
            },
        ),
    )

    assert (
        "dataset ambiguity coverage is unmet: No suitable ground-truth asset."
        in result["unmet_cases"]
    )


def test_compare_reports_rejects_different_assets() -> None:
    with pytest.raises(ValueError, match="assets differ"):
        comparison.compare_reports(
            _report(),
            _report(
                dataset={
                    "version": "dataset-v1",
                    "manifest": "cases.json",
                    "manifest_sha256": "manifest-hash",
                    "assets": [{"path": "image.png", "sha256": "other"}],
                }
            ),
        )


def test_cli_writes_json_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "comparison.json"
    baseline.write_text(json.dumps(_report()), encoding="utf-8")
    candidate.write_text(
        json.dumps(
            _report(
                variant="candidate",
                configuration={
                    "model": "qwen",
                    "collection": "products",
                    "request_fields": {"vlm_temperature": 0.1},
                    "server_deployment": {
                        "provenance": "explicit-file-complete",
                        "settings": {"APP_VLM_TEMPERATURE": 0.0},
                        "unavailable_keys": [],
                    },
                },
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        comparison,
        "_parse_args",
        lambda: comparison.argparse.Namespace(
            baseline=baseline,
            candidate=candidate,
            output=output,
            selection_rationale="No supported selection; calibration is incomplete.",
            selected="neither",
        ),
    )

    assert comparison.main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["selection"]["selected"] == (
        "neither"
    )


def test_quality_metrics_match_the_evaluator_output() -> None:
    """QUALITY_METRICS duplicates the evaluator's metric names; catch any drift."""
    evaluator_path = MODULE_PATH.with_name("evaluate_multimodal_accuracy.py")
    spec = importlib.util.spec_from_file_location(
        "evaluate_multimodal_accuracy", evaluator_path
    )
    assert spec is not None and spec.loader is not None
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)

    produced = evaluator.compute_metrics(
        [
            evaluator.CaseObservation(
                case_id="only",
                expected_sources=["manual.pdf#page=1"],
                accepted_answer_terms=["J7EF Plus"],
                expected_rejection=False,
                answer="This is the J7EF Plus.",
                candidate_sources=["manual.pdf#page=1"],
                citations=["manual.pdf#page=1"],
                ttft_seconds=0.1,
                total_seconds=0.2,
                search_seconds=0.05,
            )
        ]
    )

    assert set(comparison.QUALITY_METRICS) <= set(produced)
