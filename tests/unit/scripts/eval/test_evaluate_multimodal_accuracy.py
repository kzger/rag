# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[4] / "scripts" / "eval" / "evaluate_multimodal_accuracy.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_multimodal_accuracy", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
multimodal_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(multimodal_eval)


def _config() -> multimodal_eval.EvaluationConfig:
    return multimodal_eval.EvaluationConfig(
        collection="products",
        model="shared-qwen",
        vdb_top_k=20,
        reranker_top_k=4,
    )


def _write_manifest(tmp_path: Path, image_name: str = "query.png") -> Path:
    manifest = {
        "version": "test-v1",
        "cases": [
            {
                "id": "weak-query",
                "prompt": "這是什麼？",
                "image": image_name,
                "expected_sources": ["manual.pdf#page=3"],
                "accepted_answer_terms": ["J7EF Plus"],
                "expected_rejection": False,
            }
        ],
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_load_dataset_fails_when_required_image_is_unreadable(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, image_name="missing.png")

    with pytest.raises(ValueError, match="weak-query.*missing.png"):
        multimodal_eval.load_dataset(manifest)


def test_build_generate_payload_uses_selected_collection_and_current_image(
    tmp_path: Path,
) -> None:
    (tmp_path / "query.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    dataset = multimodal_eval.load_dataset(_write_manifest(tmp_path))

    payload = multimodal_eval.build_generate_payload(dataset.cases[0], _config())

    assert payload["use_knowledge_base"] is True
    assert payload["agentic"] is False
    assert payload["collection_names"] == ["products"]
    assert payload["vdb_top_k"] == 20
    assert payload["reranker_top_k"] == 4
    content = payload["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "這是什麼？"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_search_candidates_preserve_public_result_rank() -> None:
    payload = {
        "results": [
            {"document_name": "wrong.pdf", "metadata": {"page_number": 1}},
            {"document_name": "manual.pdf", "metadata": {"page_number": 3}},
        ]
    }

    candidates = multimodal_eval.parse_search_candidates(payload)

    assert candidates == ["wrong.pdf#page=1", "manual.pdf#page=3"]


def test_search_probe_uses_only_current_image_for_history_case() -> None:
    dataset = multimodal_eval.load_dataset(
        MODULE_PATH.with_name("multimodal_accuracy_cases.json")
    )
    case = next(item for item in dataset.cases if item.history)

    payload = multimodal_eval.build_search_payload(case, _config())

    assert payload["messages"] == []
    assert payload["query"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_parse_generate_stream_records_citations_and_latency() -> None:
    lines = [
        b'data: {"choices":[{"message":{"content":"J7EF "}}],'
        b'"citations":{"results":[{"document_name":"manual.pdf",'
        b'"metadata":{"page_number":3}}]}}',
        b'data: {"choices":[{"message":{"content":"Plus"}}]}',
    ]
    clock = iter([10.25, 10.7])

    result = multimodal_eval.parse_generate_stream(
        lines, started_at=10.0, now=clock.__next__
    )

    assert result.answer == "J7EF Plus"
    assert result.citations == ["manual.pdf#page=3"]
    assert result.ttft_seconds == pytest.approx(0.25)
    assert result.total_seconds == pytest.approx(0.7)


def test_compute_metrics_covers_retrieval_identification_and_rejection() -> None:
    observations = [
        multimodal_eval.CaseObservation(
            case_id="match",
            expected_sources=["manual.pdf#page=3"],
            accepted_answer_terms=["J7EF Plus"],
            expected_rejection=False,
            answer="This is the J7EF Plus.",
            candidate_sources=["manual.pdf#page=3"],
            citations=["manual.pdf#page=3"],
            ttft_seconds=0.2,
            total_seconds=0.8,
            search_seconds=0.12,
        ),
        multimodal_eval.CaseObservation(
            case_id="no-match",
            expected_sources=[],
            accepted_answer_terms=[],
            expected_rejection=True,
            answer="The knowledge base does not contain enough evidence to identify it.",
            candidate_sources=[],
            citations=[],
            ttft_seconds=0.4,
            total_seconds=1.2,
            search_seconds=0.2,
        ),
        multimodal_eval.CaseObservation(
            case_id="negated-match",
            expected_sources=["manual.pdf#page=3"],
            accepted_answer_terms=["J7EF Plus"],
            expected_rejection=False,
            answer="This is not J7EF Plus.",
            candidate_sources=["manual.pdf#page=3"],
            citations=["manual.pdf#page=3"],
            ttft_seconds=0.3,
            total_seconds=1.0,
            search_seconds=0.16,
        ),
    ]

    metrics = multimodal_eval.compute_metrics(observations)

    assert metrics["candidate_retrieval_hit_at_k"] == 1.0
    assert metrics["verified_identification_accuracy"] == 0.5
    assert metrics["unsupported_claim_rate"] == pytest.approx(1 / 3)
    assert metrics["low_confidence_rejection_precision"] == 1.0
    assert metrics["low_confidence_rejection_recall"] == 1.0
    assert metrics["latency_seconds"]["ttft"]["p50"] == pytest.approx(0.3)
    assert metrics["latency_seconds"]["total"]["p95"] == pytest.approx(1.18)
    assert metrics["latency_seconds"]["retrieval"]["p50"] == pytest.approx(0.16)


def test_identification_requires_generate_citation_not_only_search_hit() -> None:
    observation = multimodal_eval.CaseObservation(
        case_id="direct-guess",
        expected_sources=["manual.pdf#page=3"],
        accepted_answer_terms=["J7EF Plus"],
        expected_rejection=False,
        answer="This is J7EF Plus.",
        candidate_sources=["manual.pdf#page=3"],
        citations=[],
        ttft_seconds=0.1,
        total_seconds=0.2,
    )

    metrics = multimodal_eval.compute_metrics([observation])

    assert metrics["candidate_retrieval_hit_at_k"] == 1.0
    assert metrics["verified_identification_accuracy"] == 0.0


def test_report_contains_hashes_but_never_image_data(tmp_path: Path) -> None:
    image = tmp_path / "query.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    dataset = multimodal_eval.load_dataset(_write_manifest(tmp_path))
    observation = multimodal_eval.CaseObservation(
        case_id="weak-query",
        expected_sources=["manual.pdf#page=3"],
        accepted_answer_terms=["J7EF Plus"],
        expected_rejection=False,
        answer="J7EF Plus data:image/png;base64,secret-image-payload",
        candidate_sources=["manual.pdf#page=3"],
        citations=["manual.pdf#page=3"],
        ttft_seconds=0.1,
        total_seconds=0.2,
    )

    report = multimodal_eval.build_report(
        dataset=dataset,
        observations=[observation],
        config=_config(),
    )
    encoded = json.dumps(report)

    assert report["dataset"]["assets"][0]["sha256"]
    assert report["configuration"]["collection"] == "products"
    assert "data:image" not in encoded
    assert "base64" not in encoded


def test_versioned_baseline_manifest_covers_required_weak_query_cases() -> None:
    manifest_path = MODULE_PATH.with_name("multimodal_accuracy_cases.json")
    dataset = multimodal_eval.load_dataset(manifest_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))

    prompts = {case.prompt for case in dataset.cases}
    coverage = {tag for case in raw["cases"] for tag in case["coverage"]}
    assets = {case.image_path.name for case in dataset.cases}

    assert {"這是什麼？", "是這個嗎？", "這又是什麼？"} <= prompts
    assert {
        "ocr-model",
        "no-text-product",
        "similar-products",
        "consecutive-images",
        "knowledge-base-no-match",
    } <= coverage
    assert assets == {
        "Creme_clutch_purse1-small.jpg",
        "j10a-ocr-fixture.png",
        "j7ef-plus-ocr-fixture.png",
        "j7ef-plus.jpg",
        "logo_max.png",
    }

    report = multimodal_eval.build_report(
        dataset=dataset,
        observations=[],
        config=multimodal_eval.EvaluationConfig(
            collection="jorjin_glasses",
            model="shared-qwen",
            vdb_top_k=100,
            reranker_top_k=5,
        ),
    )
    assert {asset["path"] for asset in report["dataset"]["assets"]} == {
        "../../data/multimodal/Creme_clutch_purse1-small.jpg",
        "../../image/j10a-ocr-fixture.png",
        "../../image/j7ef-plus-ocr-fixture.png",
        "../../image/j7ef-plus.jpg",
        "../../image/logo_max.png",
    }
