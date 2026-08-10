# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Apply answer-bound manual adjudication to a multimodal baseline report."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def answer_sha256(answer: str) -> str:
    """Return the stable digest used to bind a review to one exact answer."""
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def adjudicate_report(
    report: dict[str, Any],
    adjudications: dict[str, Any],
    source_name: str,
) -> dict[str, Any]:
    """Validate manual reviews and replace lexical semantic metrics."""
    report_cases = {item["case_id"]: item for item in report.get("cases", [])}
    reviews = adjudications.get("cases", {})
    if set(report_cases) != set(reviews):
        raise ValueError("Adjudication case ids must exactly match report case ids")

    identification_total = 0
    identification_correct = 0
    supported_claims = 0
    unsupported_claims = 0
    true_positive = false_positive = false_negative = 0
    for case_id, case in report_cases.items():
        review = reviews[case_id]
        expected_hash = answer_sha256(case.get("answer", ""))
        if review.get("answer_sha256") != expected_hash:
            raise ValueError(f"Adjudication answer hash mismatch for {case_id}")
        claims = review.get("claims")
        if not isinstance(claims, list) or not claims:
            raise ValueError(f"At least one audited claim is required for {case_id}")
        if review.get("coverage_attested") is not True:
            raise ValueError(
                f"Exhaustive claim coverage attestation required for {case_id}"
            )
        for claim in claims:
            if (
                not claim.get("text")
                or not claim.get("answer_quote")
                or not claim.get("evidence")
            ):
                raise ValueError(
                    f"Claim text, answer quote, and evidence are required for {case_id}"
                )
            if claim["answer_quote"] not in case.get("answer", ""):
                raise ValueError(f"Claim quote is not present in answer for {case_id}")
            if not isinstance(claim.get("supported"), bool):
                raise ValueError(f"Claim support verdict must be boolean for {case_id}")
            supported_claims += claim["supported"]
            unsupported_claims += not claim["supported"]
        observed_rejection = bool(review["rejection_observed"])
        expected_rejection = bool(case["expected_rejection"])
        true_positive += observed_rejection and expected_rejection
        false_positive += observed_rejection and not expected_rejection
        false_negative += not observed_rejection and expected_rejection
        if not expected_rejection:
            verdict = review.get("identification_correct")
            if not isinstance(verdict, bool):
                raise ValueError(f"Identification verdict required for {case_id}")
            if verdict and not any(
                expected.casefold() == citation.casefold()
                for expected in case.get("expected_sources", [])
                for citation in case.get("citations", [])
            ):
                raise ValueError(
                    f"Positive identification requires expected citation for {case_id}"
                )
            identification_total += 1
            identification_correct += verdict

    claim_total = supported_claims + unsupported_claims
    metrics = report.setdefault("metrics", {})
    metrics["verified_identification_accuracy"] = (
        identification_correct / identification_total if identification_total else 0.0
    )
    metrics["unsupported_claim_rate"] = (
        unsupported_claims / claim_total if claim_total else 0.0
    )
    metrics["low_confidence_rejection_precision"] = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    metrics["low_confidence_rejection_recall"] = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    report["adjudication"] = {
        "version": adjudications.get("version"),
        "source": source_name,
        "methodology": adjudications.get("methodology"),
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "answer_hash_bound": True,
    }
    report["metric_definitions"]["verified_identification_accuracy"] = (
        "Manual verdict accuracy across identifiable cases, bound to exact answer hashes; "
        "the reviewer checks the query image and cited knowledge-base evidence."
    )
    report["metric_definitions"]["unsupported_claim_rate"] = (
        "Manually unsupported atomic factual claims divided by all manually reviewed "
        "atomic factual claims, with reviews bound to exact answer hashes."
    )
    report["metric_definitions"][
        "low_confidence_rejection_precision_recall"
    ] = "Reviewer-observed rejection precision and recall against expected_rejection labels."
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--adjudications", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Validate adjudications and write the final semantic baseline report."""
    args = _parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    adjudications = json.loads(args.adjudications.read_text(encoding="utf-8"))
    result = adjudicate_report(report, adjudications, args.adjudications.name)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
