"""Unit tests for multimodal query understanding and page fusion."""

import json

from langchain_core.documents import Document
from nvidia_rag.rag_server.multimodal_accuracy import (
    PRODUCT_LINE_ALIASES,
    CandidateVerification,
    QueryUnderstanding,
    build_enriched_text_query,
    canonicalize_product_line,
    decide_multimodal_outcome,
    dedupe_by_page,
    derive_product_line_matches,
    derive_query_model_matches,
    fuse_visual_text_candidates,
    parse_query_understanding,
)


def document(
    source: str = "manual.pdf",
    page: int = 1,
    content: str = "content",
    **metadata: object,
) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "content_metadata": {"page_number": page},
            **metadata,
        },
    )


def test_parse_query_understanding_plain_and_fenced_json() -> None:
    payload = {
        "ocr_text": "NVIDIA",
        "brand": "NVIDIA",
        "model": "A100",
        "object_category": "GPU",
        "color": "black",
        "shape": "card",
        "logo": "eye",
        "distinguishing_features": "fan",
        "unknown": "ignored",
    }
    parsed = parse_query_understanding(json.dumps(payload))
    fenced = parse_query_understanding(f"```json\n{json.dumps(payload)}\n```")

    assert parsed is not None
    assert parsed.model == "A100"
    assert parsed.distinguishing_features == "fan"
    assert fenced == parsed
    assert not hasattr(parsed, "unknown")


def test_parse_query_understanding_invalid_and_non_string_values() -> None:
    assert parse_query_understanding("not json") is None
    parsed = parse_query_understanding(
        '{"brand": 123, "model": null, "color": true, "logo": "  mark  "}'
    )

    assert parsed is not None
    assert parsed.brand == ""
    assert parsed.model == ""
    assert parsed.color == ""
    assert parsed.logo == "mark"


def test_build_enriched_text_query_and_determinism() -> None:
    understanding = QueryUnderstanding(ocr_text="NVIDIA", model="A100")
    result = build_enriched_text_query(understanding, "What is this?", "Prior turn")

    assert "OCR text: NVIDIA" in result
    assert "user question: What is this?" in result
    assert "conversation summary: Prior turn" in result
    assert result == build_enriched_text_query(
        understanding, "What is this?", "Prior turn"
    )
    assert build_enriched_text_query(
        QueryUnderstanding(raw_text="ignored"), "question"
    ) == ("user question: question")
    assert build_enriched_text_query(QueryUnderstanding(), "", "") == ""


def test_dedupe_by_page_skips_missing_metadata_and_caps(caplog) -> None:
    docs = [
        document(page=1, content="first"),
        document(page=1, content="duplicate"),
        Document(page_content="missing", metadata={}),
        document(page=2, content="second"),
    ]

    result = dedupe_by_page(docs, limit=2)

    assert [item.page_content for item in result] == ["first", "second"]
    assert "Skipping candidate without source/page metadata" in caplog.text


def test_fuse_only_visual_candidates() -> None:
    docs = [document(page=1), document(page=2)]
    result = fuse_visual_text_candidates(
        visual_documents=docs,
        text_documents=[],
        visual_weight=0.5,
        rrf_k=10,
    )

    assert [item.metadata["multimodal_fusion"]["visual_rank"] for item in result] == [
        1,
        2,
    ]
    assert result[0].metadata["multimodal_fusion"]["fusion_score"] == 0.5 / 11


def test_fuse_same_page_prefers_visual_document_and_preserves_raw_scores() -> None:
    visual = document(
        page=1,
        content="visual content",
        image_retrieval={"raw_score": "0.9"},
    )
    text = document(page=1, content="text content", relevance_score="0.8")

    result = fuse_visual_text_candidates(
        visual_documents=[visual], text_documents=[text], rrf_k=60
    )
    fusion = result[0].metadata["multimodal_fusion"]

    assert result[0].page_content == "visual content"
    assert fusion["visual_rank"] == 1
    assert fusion["text_rank"] == 1
    assert fusion["visual_raw_score"] == 0.9
    assert fusion["text_raw_score"] == 0.8
    assert "multimodal_fusion" not in visual.metadata
    assert "multimodal_fusion" not in text.metadata


def test_fuse_weighting_changes_order() -> None:
    visual = [document(page=1)] + [document(page=i) for i in range(2, 6)]
    text = (
        [document(page=5)]
        + [document(page=i) for i in range(6, 9)]
        + [document(page=1)]
    )

    visual_heavy = fuse_visual_text_candidates(
        visual_documents=visual,
        text_documents=text,
        visual_weight=0.9,
        text_weight=0.1,
        rrf_k=1,
    )
    text_heavy = fuse_visual_text_candidates(
        visual_documents=visual,
        text_documents=text,
        visual_weight=0.1,
        text_weight=0.9,
        rrf_k=1,
    )

    assert visual_heavy[0].metadata["content_metadata"]["page_number"] == 1
    assert text_heavy[0].metadata["content_metadata"]["page_number"] == 5


def test_fuse_tie_is_deterministic_and_honors_cap() -> None:
    docs = [document(source="b.pdf", page=2), document(source="a.pdf", page=1)]
    result = fuse_visual_text_candidates(
        visual_documents=docs,
        text_documents=[],
        visual_weight=1.0,
        rrf_k=1,
        max_candidates=1,
    )
    assert result[0].metadata["source"] == "b.pdf"

    ties = fuse_visual_text_candidates(
        visual_documents=[document(source="b.pdf", page=1)],
        text_documents=[document(source="a.pdf", page=1)],
        visual_weight=0.5,
        text_weight=0.5,
        rrf_k=1,
    )
    assert ties[0].metadata["source"] == "b.pdf"


def test_fuse_skips_documents_without_page_identity(caplog) -> None:
    result = fuse_visual_text_candidates(
        visual_documents=[Document(page_content="bad", metadata={})],
        text_documents=[document()],
    )

    assert len(result) == 1
    assert "Skipping candidate without source/page metadata" in caplog.text


def test_brand_only_identity_does_not_verify() -> None:
    result = decide_multimodal_outcome(
        [
            CandidateVerification(
                candidate_id="C1",
                decision="match",
                confidence=0.99,
                evidence_type="exact_visible_text",
                supporting_evidence="Visible text: JORJIN",
                resolved_identity="JORJIN",
            )
        ]
    )

    assert result.outcome == "ambiguous"


def test_query_model_bridge_matches_page_and_source_tokens() -> None:
    understanding = QueryUnderstanding(model="J7EF Plus", ocr_text="MODEL J10A")
    matches = derive_query_model_matches(
        understanding,
        [
            document(source="J7EF Sales kit.pdf", content="specification"),
            document(source="J10A Sales kit.pdf", content="specification"),
        ],
    )

    assert matches["C1"].matched_token == "J7EF"
    assert matches["C1"].provenance == "source_name"
    assert matches["C2"].matched_token == "J10A"


def test_query_model_bridge_requires_boundaries_and_strong_tokens() -> None:
    understanding = QueryUnderstanding(model="J10A", ocr_text="JORJIN logo 1080")
    matches = derive_query_model_matches(
        understanding,
        [document(source="J10AB.pdf", content="JORJIN logo 1080")],
    )

    assert matches == {}


def test_query_model_bridge_overrides_photo_layout_mismatch() -> None:
    understanding = QueryUnderstanding(model="J7EF Plus")
    result = decide_multimodal_outcome(
        [
            CandidateVerification(
                candidate_id="C1",
                decision="mismatch",
                confidence=0.99,
                evidence_type="generic_similarity",
            )
        ],
        model_text_matches=derive_query_model_matches(
            understanding,
            [document(source="J7EF Sales kit.pdf")],
        ),
    )

    assert result.outcome == "verified"


def test_product_line_alias_matches_punctuation_and_boundaries() -> None:
    understanding = QueryUnderstanding(ocr_text="JREALITY JORJIN")
    assert canonicalize_product_line("J-Reality") == "jreality"
    matches = derive_product_line_matches(
        understanding,
        [document(source="J7EF PULS 產品規格_v3.pdf", content="J7EF Plus")],
        {"jreality"},
        {"jreality": "J7EF PULS 產品規格_v3.pdf"},
    )
    assert matches["C1"].tier == "product_line"
    assert derive_product_line_matches(
        QueryUnderstanding(ocr_text="JREALITYX"),
        [document(source="J7EF PULS 產品規格_v3.pdf", content="J7EF Plus")],
        {"jreality"},
        {"jreality": "J7EF PULS 產品規格_v3.pdf"},
    ) == {}


def test_product_line_alias_requires_registry_and_global_eligibility() -> None:
    assert "jreality" in PRODUCT_LINE_ALIASES
    understanding = QueryUnderstanding(ocr_text="JREALITY")
    assert derive_product_line_matches(
        understanding, [document(content="J-Reality")], set()
    ) == {}


def test_product_line_alias_matches_even_with_brand_observation() -> None:
    matches = derive_product_line_matches(
        QueryUnderstanding(ocr_text="JORJIN JREALITY"),
        [document(source="J7EF PULS 產品規格_v3.pdf", content="J7EF Plus")],
        {"jreality"},
        {"jreality": "J7EF PULS 產品規格_v3.pdf"},
    )
    result = decide_multimodal_outcome(
        [
            CandidateVerification(
                "C1", "mismatch", 0.99, "generic_similarity"
            )
        ],
        product_line_matches=matches,
    )
    assert result.outcome == "verified"
