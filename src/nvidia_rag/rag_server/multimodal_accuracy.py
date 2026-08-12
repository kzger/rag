"""Pure helpers for multimodal query understanding and retrieval fusion."""

import json
import logging
import math
import re
from dataclasses import dataclass, replace
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

QUERY_UNDERSTANDING_FIELDS: tuple[tuple[str, str], ...] = (
    ("ocr_text", "OCR text"),
    ("brand", "brand"),
    ("model", "model"),
    ("object_category", "object category"),
    ("color", "color"),
    ("shape", "shape"),
    ("logo", "logo"),
    ("distinguishing_features", "distinguishing features"),
)


@dataclass(frozen=True)
class QueryUnderstanding:
    """Structured visual/text features extracted from the current query image.

    Used ONLY for retrieval enrichment, never as a directly citable fact.
    """

    ocr_text: str = ""
    brand: str = ""
    model: str = ""
    object_category: str = ""
    color: str = ""
    shape: str = ""
    logo: str = ""
    distinguishing_features: str = ""
    raw_text: str = ""

    @property
    def usable(self) -> bool:
        """True when at least one extracted field (excluding raw_text) is non-empty."""
        return any(
            bool(getattr(self, field).strip())
            for field, _label in QUERY_UNDERSTANDING_FIELDS
        )

    def enriched_text_query(self) -> str:
        """Return non-empty fields in a deterministic, embedding-friendly format."""
        return "\n".join(
            f"{label}: {getattr(self, field).strip()}"
            for field, label in QUERY_UNDERSTANDING_FIELDS
            if getattr(self, field).strip()
        )


def build_query_understanding_prompt() -> str:
    """Build the system prompt used to extract visual retrieval features."""
    return """Examine the single user-supplied image and the accompanying short text.
Extract visual and textual features only for retrieval enrichment.
Reply with ONLY one JSON object, with exactly these keys:
ocr_text, brand, model, object_category, color, shape, logo, distinguishing_features
Every value must be a short string.
Use an empty string ("") when a value is unknown, absent, or illegible.
Do not invent a brand, model, specifications, or any other detail.
Do not answer the user's question.
Your output is used for search only, never as a directly citable fact.
Keep each extracted value concise and useful for semantic search.
Use only evidence visible in the supplied image or short text.
If no image content is readable, return all eight keys with empty strings.
Do not include Markdown fences, explanations, or additional keys."""


def parse_query_understanding(content: str) -> QueryUnderstanding | None:
    """Parse a model response into structured query-understanding fields."""
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()

    try:
        parsed: Any = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match is None:
            return None
        try:
            parsed = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return None

    if not isinstance(parsed, dict):
        return None

    fields = {
        field: value.strip()[:500]
        if isinstance((value := parsed.get(field)), str)
        else ""
        for field, _label in QUERY_UNDERSTANDING_FIELDS
    }
    return QueryUnderstanding(**fields)


def build_enriched_text_query(
    understanding: QueryUnderstanding | None,
    raw_text: str,
    history_summary: str | None = None,
) -> str:
    """Combine structured visual features, the user question, and history."""
    parts: list[str] = []
    if understanding is not None and understanding.usable:
        parts.append(understanding.enriched_text_query())
    raw = (raw_text or "").strip()
    if raw:
        parts.append(f"user question: {raw}")
    if history_summary and history_summary.strip():
        parts.append(f"conversation summary: {history_summary.strip()}")
    return "\n".join(parts)


def _page_identity(document: Document) -> tuple[str, int] | None:
    """Return a document's stable source and page identity, if available."""
    source = document.metadata.get("source")
    source_name = source.get("source_name") if isinstance(source, dict) else source
    content_metadata = document.metadata.get("content_metadata")
    page_number = (
        content_metadata.get("page_number")
        if isinstance(content_metadata, dict)
        else None
    )
    if (
        not isinstance(source_name, str)
        or not source_name
        or not isinstance(page_number, int)
        or isinstance(page_number, bool)
    ):
        return None
    return source_name, page_number


def dedupe_by_page(documents: list[Document], limit: int) -> list[Document]:
    """Keep the first ranked document for each source/page identity."""
    if limit <= 0:
        return []
    result: list[Document] = []
    seen: set[tuple[str, int]] = set()
    for document in documents:
        identity = _page_identity(document)
        if identity is None:
            logger.warning(
                "Skipping candidate without source/page metadata for multimodal fusion"
            )
            continue
        if identity in seen:
            continue
        seen.add(identity)
        result.append(document)
        if len(result) == limit:
            break
    return result


def _raw_score(document: Document, *metadata_keys: str) -> float | None:
    """Read and coerce a raw retrieval score from document metadata."""
    value: Any = document.metadata
    for key in metadata_keys:
        value = value.get(key) if isinstance(value, dict) else None
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _text_raw_score(document: Document) -> float | None:
    """Read the preferred text score, falling back to the generic score."""
    score = _raw_score(document, "relevance_score")
    return score if score is not None else _raw_score(document, "score")


def fuse_visual_text_candidates(
    *,
    visual_documents: list[Document],
    text_documents: list[Document],
    visual_weight: float = 0.5,
    text_weight: float = 0.5,
    rrf_k: int = 60,
    max_candidates: int = 5,
) -> list[Document]:
    """Fuse ranked visual and text page candidates using weighted RRF."""
    visual_by_identity: dict[tuple[str, int], tuple[int, Document]] = {}
    text_by_identity: dict[tuple[str, int], tuple[int, Document]] = {}

    for rank, document in enumerate(visual_documents, start=1):
        identity = _page_identity(document)
        if identity is None:
            logger.warning(
                "Skipping candidate without source/page metadata for multimodal fusion"
            )
            continue
        visual_by_identity.setdefault(identity, (rank, document))

    for rank, document in enumerate(text_documents, start=1):
        identity = _page_identity(document)
        if identity is None:
            logger.warning(
                "Skipping candidate without source/page metadata for multimodal fusion"
            )
            continue
        text_by_identity.setdefault(identity, (rank, document))

    identities = set(visual_by_identity) | set(text_by_identity)
    fused: list[tuple[float, int | None, int | None, str, int, Document]] = []
    for identity in identities:
        visual_entry = visual_by_identity.get(identity)
        text_entry = text_by_identity.get(identity)
        visual_rank, visual_document = visual_entry or (None, None)
        text_rank, text_document = text_entry or (None, None)
        fusion_score = (
            visual_weight / (rrf_k + visual_rank) if visual_rank is not None else 0.0
        ) + (text_weight / (rrf_k + text_rank) if text_rank is not None else 0.0)
        base = visual_document if visual_document is not None else text_document
        assert base is not None
        metadata = dict(base.metadata)
        metadata["multimodal_fusion"] = {
            "fusion_score": float(fusion_score),
            "visual_rank": visual_rank,
            "text_rank": text_rank,
            "visual_raw_score": (
                _raw_score(visual_document, "image_retrieval", "raw_score")
                if visual_document is not None
                else None
            ),
            "text_raw_score": (
                _text_raw_score(text_document) if text_document is not None else None
            ),
            "visual_weight": float(visual_weight),
            "text_weight": float(text_weight),
            "rrf_k": rrf_k,
        }
        candidate = Document(page_content=base.page_content, metadata=metadata)
        source_name, page_number = identity
        logger.info(
            "Fused multimodal candidate source=%s page=%s visual_rank=%s "
            "text_rank=%s fusion_score=%.4f",
            source_name,
            page_number,
            visual_rank,
            text_rank,
            fusion_score,
        )
        fused.append(
            (
                fusion_score,
                visual_rank,
                text_rank,
                source_name,
                page_number,
                candidate,
            )
        )

    fused.sort(
        key=lambda item: (
            -item[0],
            item[1] if item[1] is not None else float("inf"),
            item[2] if item[2] is not None else float("inf"),
            item[3],
            item[4],
        )
    )
    return [item[-1] for item in fused[:max_candidates]]
