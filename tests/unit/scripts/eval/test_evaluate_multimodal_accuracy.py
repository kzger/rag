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


def test_build_generate_payload_records_request_vlm_temperature_without_schema_change(
    tmp_path: Path,
) -> None:
    (tmp_path / "query.png").write_bytes(b"fixture")
    dataset = multimodal_eval.load_dataset(_write_manifest(tmp_path))
    config = multimodal_eval.EvaluationConfig(
        collection="products",
        model="shared-qwen",
        vdb_top_k=20,
        reranker_top_k=4,
        request_vlm_temperature=0.0,
    )

    assert (
        multimodal_eval.build_generate_payload(dataset.cases[0], config)[
            "vlm_temperature"
        ]
        == 0.0
    )


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


def test_ambiguous_abstention_phrases_count_as_rejection() -> None:
    observations = [
        multimodal_eval.CaseObservation(
            case_id="ambiguous",
            expected_sources=[],
            accepted_answer_terms=[],
            expected_rejection=True,
            answer="無法唯一判定這是 J7EF Plus 還是 J10A。",
            candidate_sources=[],
            citations=[],
            ttft_seconds=0.1,
            total_seconds=0.2,
        ),
        multimodal_eval.CaseObservation(
            case_id="ambiguous-en",
            expected_sources=[],
            accepted_answer_terms=[],
            expected_rejection=True,
            answer="I cannot uniquely identify the product from this evidence.",
            candidate_sources=[],
            citations=[],
            ttft_seconds=0.1,
            total_seconds=0.2,
        ),
    ]

    metrics = multimodal_eval.compute_metrics(observations)

    assert metrics["low_confidence_rejection_recall"] == 1.0


def test_ambiguous_rejection_terms_do_not_change_identification() -> None:
    observation = multimodal_eval.CaseObservation(
        case_id="identified",
        expected_sources=["manual.pdf#page=3"],
        accepted_answer_terms=["J7EF Plus"],
        expected_rejection=False,
        answer=(
            "This is J7EF Plus. I cannot uniquely identify unrelated details "
            "not established by the page."
        ),
        candidate_sources=["manual.pdf#page=3"],
        citations=["manual.pdf#page=3"],
        ttft_seconds=0.1,
        total_seconds=0.2,
    )

    metrics = multimodal_eval.compute_metrics([observation])

    assert metrics["verified_identification_accuracy"] == 1.0


def test_rescore_report_preserves_identity_and_marks_harness_version(
    tmp_path: Path,
) -> None:
    image = tmp_path / "query.png"
    image.write_bytes(b"fixture")
    dataset = multimodal_eval.load_dataset(_write_manifest(tmp_path))
    report = multimodal_eval.build_report(dataset, [], _config())
    report["cases"] = [
        {
            "case_id": "ambiguous",
            "expected_sources": [],
            "accepted_answer_terms": [],
            "expected_rejection": True,
            "answer": "I cannot uniquely identify the product.",
            "candidate_sources": [],
            "citations": [],
            "ttft_seconds": 0.1,
            "total_seconds": 0.2,
            "search_seconds": 0.0,
            "forbidden_answer_terms": [],
            "error": None,
        }
    ]

    rescored = multimodal_eval.rescore_report(report)

    assert rescored["dataset"] == report["dataset"]
    assert rescored["configuration"] == report["configuration"]
    assert rescored["metrics"]["low_confidence_rejection_recall"] == 1.0
    assert rescored["scoring"]["status"] == "rescored"
    assert rescored["scoring"]["harness_version"] == (
        multimodal_eval.SCORING_HARNESS_VERSION
    )


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


@pytest.mark.parametrize(
    "answer",
    [
        "根據圖片比對，這不是圖片中產品所屬的型號。",
        "Therefore, the image is not the device described in the text J7EF Plus.",
    ],
)
def test_identity_denial_patterns_catch_live_pronoun_answers(answer: str) -> None:
    assert multimodal_eval._denies_identity(answer, ["J7EF Plus"])


@pytest.mark.parametrize(
    "answer",
    [
        "The device has a cable, which is not evidence of its power source.",
        "這不是紅黑配色的耳機，而是圖片中的頭戴式顯示器。",
    ],
)
def test_identity_denial_patterns_preserve_passing_negation(answer: str) -> None:
    assert not multimodal_eval._denies_identity(answer, ["J7EF Plus"])


def test_live_pronoun_answer_is_not_verified_identification() -> None:
    observation = multimodal_eval.CaseObservation(
        case_id="j7ef-pronoun-confirmation",
        expected_sources=["manual.pdf#page=3"],
        accepted_answer_terms=["J7EF Plus"],
        expected_rejection=False,
        answer="根據比對，不是圖片中產品所屬的型號，也不是 J7EF Plus。",
        candidate_sources=["manual.pdf#page=3"],
        citations=["manual.pdf#page=3"],
        ttft_seconds=0.1,
        total_seconds=0.2,
    )

    assert (
        multimodal_eval.compute_metrics([observation])[
            "verified_identification_accuracy"
        ]
        == 0.0
    )


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
    assert len(report["dataset"]["manifest_sha256"]) == 64
    assert report["configuration"]["collection"] == "products"
    assert "data:image" not in encoded
    assert "base64" not in encoded


def test_report_distinguishes_request_and_server_configuration(tmp_path: Path) -> None:
    image = tmp_path / "query.png"
    image.write_bytes(b"fixture")
    dataset = multimodal_eval.load_dataset(_write_manifest(tmp_path))
    config = multimodal_eval.EvaluationConfig(
        collection="products",
        model="shared-qwen",
        vdb_top_k=20,
        reranker_top_k=4,
        endpoint="http://rag.example",
        request_vlm_temperature=0.0,
    )

    report = multimodal_eval.build_report(dataset, [], config)

    assert report["configuration"]["request_fields"]["vlm_temperature"] == 0.0
    assert (
        report["configuration"]["server_deployment"]["settings"]["APP_VLM_TEMPERATURE"]
        is None
    )
    settings = report["configuration"]["server_deployment"]["settings"]
    assert set(settings) == set(multimodal_eval.SERVER_SETTING_TYPES)
    assert all(value is None for value in settings.values())


def test_report_does_not_claim_host_environment_is_remote_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "query.png"
    image.write_bytes(b"fixture")
    monkeypatch.setenv("APP_VLM_TEMPERATURE", "0.1")
    dataset = multimodal_eval.load_dataset(_write_manifest(tmp_path))

    report = multimodal_eval.build_report(dataset, [], _config())

    deployment = report["configuration"]["server_deployment"]
    assert deployment["provenance"] == "unavailable"
    assert deployment["settings"]["APP_VLM_TEMPERATURE"] is None
    assert "APP_VLM_TEMPERATURE" in deployment["unavailable_keys"]


def test_load_server_settings_requires_known_typed_values(tmp_path: Path) -> None:
    settings_file = tmp_path / "server-settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "schema_version": "multimodal-server-settings-v1",
                "settings": {
                    "APP_VLM_TEMPERATURE": 0.1,
                    "ENABLE_QUERY_UNDERSTANDING": True,
                },
            }
        ),
        encoding="utf-8",
    )

    metadata = multimodal_eval.load_server_settings(settings_file)

    assert metadata["provenance"] == "explicit-file-partial"
    assert metadata["settings"]["APP_VLM_TEMPERATURE"] == 0.1
    assert metadata["settings"]["APP_VLM_TOP_P"] is None

    settings_file.write_text(
        json.dumps(
            {
                "schema_version": "multimodal-server-settings-v1",
                "settings": {"NVIDIA_API_KEY": "must-not-be-accepted"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown server setting"):
        multimodal_eval.load_server_settings(settings_file)


def test_load_server_settings_marks_complete_file_provenance(tmp_path: Path) -> None:
    settings_file = tmp_path / "complete-server-settings.json"
    settings_file.write_text(
        json.dumps(
            {
                "schema_version": "multimodal-server-settings-v1",
                "settings": {
                    key: expected_type()
                    for key, expected_type in (
                        multimodal_eval.SERVER_SETTING_TYPES.items()
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    metadata = multimodal_eval.load_server_settings(settings_file)

    assert metadata["provenance"] == "explicit-file-complete"
    assert metadata["unavailable_keys"] == []


def test_checked_in_server_settings_example_is_complete() -> None:
    metadata = multimodal_eval.load_server_settings(
        MODULE_PATH.with_name("multimodal_server_settings.example.json")
    )

    assert metadata["provenance"] == "explicit-file-complete"
    assert metadata["unavailable_keys"] == []


def test_local_environment_capture_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APP_VLM_TEMPERATURE", "0.1")

    metadata = multimodal_eval.load_server_settings(
        None, capture_local_environment=True
    )

    assert metadata["provenance"] == "local-process-environment"
    assert metadata["settings"]["APP_VLM_TEMPERATURE"] == 0.1


def test_request_vlm_temperature_report_metadata_is_neutral(tmp_path: Path) -> None:
    (tmp_path / "query.png").write_bytes(b"fixture")
    dataset = multimodal_eval.load_dataset(_write_manifest(tmp_path))
    config = multimodal_eval.EvaluationConfig(
        collection="products",
        model="shared-qwen",
        vdb_top_k=20,
        reranker_top_k=4,
        request_vlm_temperature=0.0,
    )

    report = multimodal_eval.build_report(dataset, [], config)

    assert report["configuration"]["request_fields"]["vlm_temperature"] == 0.0
    assert report["configuration"]["calibration"]["status"] == "run-metadata-only"
    assert report["configuration"]["calibration"]["raw_and_rrf_thresholds"] == (
        "telemetry-only"
    )


def test_report_carries_manifest_coverage_and_ambiguity_metadata(
    tmp_path: Path,
) -> None:
    image = tmp_path / "query.png"
    image.write_bytes(b"fixture")
    manifest = _write_manifest(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["coverage"] = {"required": ["deliberate-ambiguity"], "present": []}
    raw["ambiguity"] = {
        "required": True,
        "present": False,
        "blocked": "No suitable ground-truth asset.",
    }
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    report = multimodal_eval.build_report(
        multimodal_eval.load_dataset(manifest), [], _config()
    )

    assert report["dataset"]["coverage"] == raw["coverage"]
    assert report["dataset"]["ambiguity"] == raw["ambiguity"]


def test_versioned_baseline_manifest_covers_required_weak_query_cases() -> None:
    manifest_path = MODULE_PATH.with_name("multimodal_accuracy_cases.json")
    dataset = multimodal_eval.load_dataset(manifest_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert raw["ambiguity"] == {
        "required": True,
        "present": True,
        "reason": "The manifest reuses the checked-in company logo asset for a deliberately ambiguous model-choice prompt. A logo identifies the company but cannot uniquely support either J7EF Plus or J10A.",
        "safe_outcome": "abstain_or_reject",
        "ground_truth_asset": "../../image/logo_max.png",
    }

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
        "deliberate-ambiguity",
    } <= coverage
    assert assets == {
        "Creme_clutch_purse1-small.jpg",
        "j10a-ocr-fixture.png",
        "j7ef-plus-ocr-fixture.png",
        "j7ef-plus.jpg",
        "logo_max.png",
    }
    ambiguity_case = next(
        case for case in raw["cases"] if case["id"] == "logo-deliberate-ambiguity"
    )
    assert ambiguity_case["expected_rejection"] is True
    assert ambiguity_case["expected_sources"] == []
    assert set(ambiguity_case["forbidden_answer_terms"]) == {
        "J7EF Plus",
        "J7EF+",
        "J10A",
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
