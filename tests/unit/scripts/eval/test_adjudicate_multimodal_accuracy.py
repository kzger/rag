# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[4] / "scripts" / "eval" / "adjudicate_multimodal_accuracy.py"
)
SPEC = importlib.util.spec_from_file_location(
    "adjudicate_multimodal_accuracy", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
adjudicator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adjudicator)


def _report() -> dict:
    return {
        "metrics": {"candidate_retrieval_hit_at_k": 1.0},
        "metric_definitions": {},
        "cases": [
            {
                "case_id": "match",
                "answer": "J7EF",
                "expected_rejection": False,
                "expected_sources": ["manual.pdf#page=3"],
                "citations": ["manual.pdf#page=3"],
            },
            {
                "case_id": "none",
                "answer": "I cannot confirm",
                "expected_rejection": True,
            },
        ],
    }


def _reviews() -> dict:
    return {
        "version": "review-v1",
        "methodology": "Manual atomic-claim review.",
        "cases": {
            "match": {
                "answer_sha256": adjudicator.answer_sha256("J7EF"),
                "identification_correct": True,
                "coverage_attested": True,
                "claims": [
                    {
                        "text": "J7EF",
                        "answer_quote": "J7EF",
                        "supported": True,
                        "evidence": "manual p3",
                    }
                ],
                "rejection_observed": False,
            },
            "none": {
                "answer_sha256": adjudicator.answer_sha256("I cannot confirm"),
                "identification_correct": None,
                "coverage_attested": True,
                "claims": [
                    {
                        "text": "No KB match",
                        "answer_quote": "cannot confirm",
                        "supported": True,
                        "evidence": "empty candidates",
                    }
                ],
                "rejection_observed": True,
            },
        },
    }


def test_adjudication_computes_manually_reviewed_semantic_metrics() -> None:
    result = adjudicator.adjudicate_report(_report(), _reviews(), "review.json")

    assert result["metrics"]["verified_identification_accuracy"] == 1.0
    assert result["metrics"]["unsupported_claim_rate"] == 0.0
    assert result["metrics"]["low_confidence_rejection_precision"] == 1.0
    assert result["metrics"]["low_confidence_rejection_recall"] == 1.0
    assert result["adjudication"]["answer_hash_bound"] is True


def test_adjudication_rejects_stale_answer_hash() -> None:
    reviews = _reviews()
    reviews["cases"]["match"]["answer_sha256"] = "stale"

    with pytest.raises(ValueError, match="hash mismatch for match"):
        adjudicator.adjudicate_report(_report(), reviews, "review.json")


def test_positive_identification_requires_expected_generate_citation() -> None:
    report = _report()
    report["cases"][0]["citations"] = []

    with pytest.raises(ValueError, match="requires expected citation for match"):
        adjudicator.adjudicate_report(report, _reviews(), "review.json")
