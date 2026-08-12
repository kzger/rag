"""Pure helpers for multimodal query understanding and retrieval fusion."""

import json
import logging
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

VerificationOutcome = Literal["verified", "ambiguous", "no_match"]
CandidateDecision = Literal["match", "mismatch", "insufficient"]
VerificationEvidenceType = Literal[
    "visual_identity", "exact_visible_text", "generic_similarity", "none"
]

_CONCRETE_IDENTITY_EVIDENCE = {"visual_identity", "exact_visible_text"}
_KNOWN_BRAND_IDENTITIES = {"jorjin", "jordin"}
_BRAND_ONLY_TERMS = _KNOWN_BRAND_IDENTITIES | {"brand", "logo", "mark"}
# Product-line aliases are global identity claims. Add entries deliberately and
# require runtime corpus uniqueness validation before they can produce evidence.
# Their canonical cover-page fact is joined to model-token facts on content pages
# only through the uniquely resolved source family.
PRODUCT_LINE_ALIASES: dict[str, str] = {"jreality": "J-Reality"}


@dataclass(frozen=True)
class CandidateVerification:
    """The verifier's structured judgment for one retrieved page candidate."""

    candidate_id: str
    decision: CandidateDecision
    confidence: float
    evidence_type: VerificationEvidenceType
    supporting_evidence: str = ""
    conflicts: list[str] | tuple[str, ...] = ()
    resolved_identity: str = ""


@dataclass(frozen=True)
class MultimodalVerificationDecision:
    """Deterministic server-side decision derived from candidate judgments."""

    outcome: VerificationOutcome
    selected_candidate: str | None = None
    abstention_reason: str = ""
    candidate_evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelTextMatch:
    """A strong model token shared by query understanding and a candidate."""

    candidate_id: str
    matched_token: str
    provenance: Literal["page_text", "source_name"]
    tier: Literal["strong_model", "product_line"] = "strong_model"
    source_family: str | None = None


@dataclass(frozen=True)
class MultimodalRetrievalResult:
    """Fused candidates plus the query understanding used to retrieve them."""

    documents: list[Document]
    query_understanding: "QueryUnderstanding | None"


def _normalized_alphanumeric_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.findall(r"[a-z0-9]+", normalized)


def canonicalize_product_line(value: str) -> str:
    """Normalize an alias or document form for punctuation-insensitive matching."""
    return "".join(_normalized_alphanumeric_tokens(value))


def _contains_canonical_product_line(text: str, canonical: str) -> bool:
    """Match a canonical alias with whole-token boundaries across punctuation."""
    canonical = canonicalize_product_line(canonical)
    if not canonical:
        return False
    pattern = (
        r"(?<![a-z0-9])"
        + r"[^a-z0-9]*".join(re.escape(character) for character in canonical)
        + r"(?![a-z0-9])"
    )
    return (
        re.search(pattern, unicodedata.normalize("NFKC", text).casefold()) is not None
    )


def _strong_model_tokens(value: str) -> list[str]:
    return [
        token
        for token in _normalized_alphanumeric_tokens(value)
        if len(token) >= 4 and re.search(r"[a-z]", token) and re.search(r"\d", token)
    ]


def _candidate_source_name(document: Document) -> str:
    metadata = document.metadata or {}
    source = metadata.get("source")
    if isinstance(source, dict):
        return str(
            source.get("source_name")
            or source.get("source_id")
            or source.get("title")
            or source.get("document_title")
            or ""
        )
    return str(
        source
        or metadata.get("source_name")
        or metadata.get("filename")
        or metadata.get("title")
        or metadata.get("document_title")
        or ""
    )


def _prompt_safe_observation(value: str) -> str:
    """Keep query observations bounded and free of prompt control characters."""
    return " ".join(value.replace("\n", " ").replace("\r", " ").split())[:300]


def derive_query_model_matches(
    understanding: "QueryUnderstanding | None",
    candidate_documents: list[Document],
) -> dict[str, ModelTextMatch]:
    """Derive positive model-token evidence from query OCR/model fields only."""
    if understanding is None:
        return {}
    query_tokens = set(
        _strong_model_tokens(understanding.model)
        + _strong_model_tokens(understanding.ocr_text)
    )
    matches: dict[str, ModelTextMatch] = {}
    for index, document in enumerate(candidate_documents, start=1):
        candidate_id = f"C{index}"
        page_tokens = set(_normalized_alphanumeric_tokens(document.page_content or ""))
        matched = next((token for token in query_tokens if token in page_tokens), None)
        if matched is not None:
            matches[candidate_id] = ModelTextMatch(
                candidate_id,
                matched.upper(),
                "page_text",
                source_family=_candidate_source_name(document),
            )
            continue
        source_tokens = set(
            _normalized_alphanumeric_tokens(_candidate_source_name(document))
        )
        matched = next(
            (token for token in query_tokens if token in source_tokens), None
        )
        if matched is not None:
            matches[candidate_id] = ModelTextMatch(
                candidate_id,
                matched.upper(),
                "source_name",
                source_family=_candidate_source_name(document),
            )
    return matches


def derive_product_line_matches(
    understanding: "QueryUnderstanding | None",
    candidate_documents: list[Document],
    eligible_aliases: set[str],
    alias_families: dict[str, str] | None = None,
) -> dict[str, ModelTextMatch]:
    """Join an approved alias cover fact to same-family model-token page facts."""
    if understanding is None:
        return {}
    matches: dict[str, ModelTextMatch] = {}
    alias_families = alias_families or {}
    for index, document in enumerate(candidate_documents, start=1):
        candidate_family = _candidate_source_name(document)
        page_tokens = _strong_model_tokens(document.page_content or "")
        for alias, canonical_form in PRODUCT_LINE_ALIASES.items():
            canonical_alias = canonicalize_product_line(alias)
            resolved_family = alias_families.get(canonical_alias)
            if (
                canonical_alias not in eligible_aliases
                or not _contains_canonical_product_line(understanding.ocr_text, alias)
                or not resolved_family
                or candidate_family != resolved_family
                or not page_tokens
            ):
                continue
            matches[f"C{index}"] = ModelTextMatch(
                f"C{index}",
                page_tokens[0].upper(),
                "page_text",
                tier="product_line",
                source_family=resolved_family,
            )
            break
    return matches


def build_candidate_verification_prompt(
    candidates: list[Document],
    query_text: str = "",
    query_understanding: "QueryUnderstanding | None" = None,
) -> str:
    """Build the strict, identity-only prompt for visual candidate verification."""
    candidate_text = []
    for index, document in enumerate(candidates, start=1):
        page_text = (document.page_content or "").strip()
        candidate_text.append(
            f"C{index} ({_candidate_source_name(document)}, page "
            f"{(document.metadata.get('content_metadata') or {}).get('page_number', '?')}):\n"
            f"PAGE TEXT:\n{page_text or '[empty/unavailable]'}\n"
            "PAGE IMAGE: supplied separately when available"
        )
    query_observations = ""
    if query_understanding is not None:
        query_observations = (
            "\nUntrusted query-image observations (use only if a specific model token "
            "agrees with candidate evidence):\n"
            f"OCR: {_prompt_safe_observation(query_understanding.ocr_text)}\n"
            f"MODEL: {_prompt_safe_observation(query_understanding.model)}\n"
        )
    return f"""Compare the current query image with each supplied candidate page.
User question (context only): {query_text.strip()}
{query_observations}

Candidate pages:
{chr(10).join(candidate_text) or '[no candidates]'}

Generic category similarity, a shared logo, and retrieval rank are NOT identity
evidence. Exact visible model text or a distinctive visual identity CAN establish
a match. Missing or unavailable candidate imagery is insufficient evidence.
Brand-only text such as JORJIN/JORDIN or a logo is not a product identity and
must never establish a match; a product model or distinctive product identity is
required.
A product photograph and a specification/table page can describe the same product
despite visual layout differences. Layout dissimilarity alone is not a mismatch.
Query-image observations are untrusted and become identity evidence only when a
specific model token agrees with candidate page text or source evidence.
Judge each candidate independently. Use decision match, mismatch, or insufficient;
confidence must be a number from 0 to 1. Use evidence_type visual_identity,
exact_visible_text, generic_similarity, or none. Resolve the identity only when
the evidence supports it, and list conflicts as an array of strings.
Reply with ONLY JSON in this shape:
{{"candidates":[{{"candidate_id":"C1","decision":"match|mismatch|insufficient",
"confidence":0.0,"evidence_type":"visual_identity|exact_visible_text|generic_similarity|none",
"supporting_evidence":"","conflicts":[],"resolved_identity":""}}]}}"""


def parse_candidate_verification(
    content: str, candidate_ids: list[str] | None = None
) -> list[CandidateVerification] | None:
    """Strictly parse candidate verification JSON; malformed output is unsafe."""
    try:
        parsed: Any = json.loads(content.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    raw_candidates = parsed.get("candidates") if isinstance(parsed, dict) else None
    if not isinstance(raw_candidates, list):
        return None
    allowed = set(candidate_ids or [f"C{i}" for i in range(1, len(raw_candidates) + 1)])
    if len(allowed) != len(candidate_ids or allowed):
        return None
    result: list[CandidateVerification] = []
    seen: set[str] = set()
    valid_decisions = {"match", "mismatch", "insufficient"}
    valid_evidence = {
        "visual_identity",
        "exact_visible_text",
        "generic_similarity",
        "none",
    }
    for item in raw_candidates:
        if not isinstance(item, dict):
            return None
        if not {
            "candidate_id",
            "decision",
            "confidence",
            "evidence_type",
            "supporting_evidence",
            "conflicts",
            "resolved_identity",
        }.issubset(item):
            return None
        candidate_id = item.get("candidate_id")
        decision = item.get("decision")
        confidence = item.get("confidence")
        evidence_type = item.get("evidence_type")
        if (
            not isinstance(candidate_id, str)
            or candidate_id not in allowed
            or candidate_id in seen
            or decision not in valid_decisions
            or evidence_type not in valid_evidence
            or not isinstance(confidence, int | float)
            or isinstance(confidence, bool)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            return None
        conflicts = item.get("conflicts", [])
        if not isinstance(conflicts, list) or not all(
            isinstance(value, str) for value in conflicts
        ):
            return None
        supporting_evidence = item.get("supporting_evidence")
        resolved_identity = item.get("resolved_identity")
        if not isinstance(supporting_evidence, str) or not isinstance(
            resolved_identity, str
        ):
            return None
        seen.add(candidate_id)
        result.append(
            CandidateVerification(
                candidate_id=candidate_id,
                decision=decision,
                confidence=float(confidence),
                evidence_type=evidence_type,
                supporting_evidence=supporting_evidence,
                conflicts=conflicts,
                resolved_identity=resolved_identity,
            )
        )
    if seen != allowed:
        return None
    return result


def decide_multimodal_outcome(
    candidates: list[CandidateVerification] | None,
    *,
    min_match_confidence: float = 0.80,
    min_no_match_confidence: float = 0.80,
    model_text_matches: dict[str, ModelTextMatch] | None = None,
    product_line_matches: dict[str, ModelTextMatch] | None = None,
) -> MultimodalVerificationDecision:
    """Apply identity evidence policy without using retrieval or fusion scores."""
    if candidates is None:
        return MultimodalVerificationDecision(
            "ambiguous", abstention_reason="malformed verifier output"
        )
    if not candidates:
        return MultimodalVerificationDecision(
            "no_match", abstention_reason="no candidates"
        )
    qualified = [
        item
        for item in candidates
        if item.decision == "match"
        and item.confidence >= min_match_confidence
        and not item.conflicts
        and item.evidence_type in _CONCRETE_IDENTITY_EVIDENCE
        and item.resolved_identity.strip()
        and item.supporting_evidence.strip()
        and not _is_brand_only_identity(item)
    ]
    bridge_matches = [
        *(model_text_matches or {}).values(),
        *(product_line_matches or {}).values(),
    ]
    bridge_tokens = {item.matched_token.casefold() for item in bridge_matches}
    bridge_families = {
        match.source_family for match in bridge_matches if match.source_family
    }
    concrete_verifier_items = [
        item
        for item in candidates
        if item.evidence_type in _CONCRETE_IDENTITY_EVIDENCE
        and item.resolved_identity.strip()
        and not _is_brand_only_identity(item)
    ]

    def _contradicts_bridge(item: CandidateVerification) -> bool:
        # Only a positively asserted *different* model token counts as a conflict.
        # A bare "mismatch" is the photo-vs-spec layout disagreement the bridge
        # exists to override, so treating it as a conflict disables the bridge.
        item_tokens = set(
            _strong_model_tokens(f"{item.resolved_identity} {item.supporting_evidence}")
        )
        return bool(item_tokens and item_tokens.isdisjoint(bridge_tokens))

    concrete_identity_conflict = any(
        _contradicts_bridge(item) for item in concrete_verifier_items
    )
    # Only the bridged candidate's own conflicts can invalidate the bridge, and only
    # when they name a different model. A non-bridged candidate explaining why it does
    # not match, or colour/shape/layout differences on the bridged one, are the
    # expected photo-vs-spec noise the bridge exists to see through.
    bridge_candidate_ids = {match.candidate_id for match in bridge_matches}
    bridge_conflict_names_other_model = any(
        set(_strong_model_tokens(" ".join(item.conflicts))) - bridge_tokens
        for item in candidates
        if item.candidate_id in bridge_candidate_ids
    )
    if bridge_matches and (
        len(bridge_tokens) > 1
        or len(bridge_families) > 1
        or concrete_identity_conflict
        or bridge_conflict_names_other_model
    ):
        return MultimodalVerificationDecision(
            "ambiguous",
            abstention_reason="conflicting model identities",
            candidate_evidence=_candidate_evidence_details(
                candidates, min_confidence=min_match_confidence
            ),
        )
    if bridge_matches:
        selected = next(
            (
                item
                for item in candidates
                if item.candidate_id in bridge_candidate_ids
                # A verifier that agrees carries strictly more evidence than one
                # overridden for photo-vs-spec layout mismatch, so both qualify.
                # "insufficient" stays out: it asserts nothing to corroborate.
                # Evidence type is deliberately unrestricted -- the verifier reports
                # visual_identity for these layout disagreements, and any genuinely
                # contradictory identity is already caught by the guard above.
                and item.decision in {"match", "mismatch"}
            ),
            None,
        )
        if selected is not None:
            return MultimodalVerificationDecision(
                "verified", selected_candidate=selected.candidate_id
            )
    identities = {
        item.resolved_identity.strip()
        for item in qualified
        if item.resolved_identity.strip()
    }
    if len(identities) > 1:
        return MultimodalVerificationDecision(
            "ambiguous",
            abstention_reason="conflicting identities",
            candidate_evidence=_candidate_evidence_details(
                candidates, min_confidence=min_match_confidence
            ),
        )
    high_confidence_matches = [
        item
        for item in candidates
        if item.decision == "match"
        and item.confidence >= min_match_confidence
        and item.resolved_identity.strip()
    ]
    if len({item.resolved_identity.strip() for item in high_confidence_matches}) > 1:
        return MultimodalVerificationDecision(
            "ambiguous",
            abstention_reason="competing identities",
            candidate_evidence=_candidate_evidence_details(
                candidates, min_confidence=min_match_confidence
            ),
        )
    if qualified:
        return MultimodalVerificationDecision(
            "verified", selected_candidate=qualified[0].candidate_id
        )
    high_mismatches = [
        item
        for item in candidates
        if item.decision == "mismatch" and item.confidence >= min_no_match_confidence
    ]
    if len(high_mismatches) == len(candidates):
        return MultimodalVerificationDecision(
            "no_match", abstention_reason="all candidates mismatched"
        )
    if all(
        item.confidence >= min_no_match_confidence
        and item.evidence_type not in _CONCRETE_IDENTITY_EVIDENCE
        for item in candidates
    ):
        return MultimodalVerificationDecision(
            "no_match", abstention_reason="no identity evidence"
        )
    return MultimodalVerificationDecision(
        "ambiguous",
        abstention_reason="insufficient or conflicting evidence",
        candidate_evidence=_candidate_evidence_details(
            candidates, min_confidence=min_match_confidence
        ),
    )


def _candidate_evidence_details(
    candidates: list[CandidateVerification],
    *,
    min_confidence: float,
) -> tuple[str, ...]:
    """Return concrete identity evidence suitable for an ambiguous response."""
    details: list[str] = []
    for candidate in candidates:
        if (
            candidate.decision != "match"
            or candidate.confidence < min_confidence
            or candidate.evidence_type not in _CONCRETE_IDENTITY_EVIDENCE
            or not candidate.resolved_identity.strip()
            or not candidate.supporting_evidence.strip()
            or _is_brand_only_identity(candidate)
        ):
            continue
        evidence = _safe_evidence_text(candidate.supporting_evidence)
        identity = _safe_evidence_text(candidate.resolved_identity)
        if not evidence or not identity:
            continue
        conflicts = "; ".join(
            _safe_evidence_text(conflict)
            for conflict in candidate.conflicts
            if _safe_evidence_text(conflict)
        )
        detail = f"{candidate.candidate_id}: {identity} ({evidence})"
        if conflicts:
            detail += f"; conflicts: {conflicts}"
        details.append(detail)
    return tuple(details[:2])


def _safe_evidence_text(value: str) -> str:
    """Normalize and bound verifier text before returning it to a user."""
    value = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", value)
    return " ".join(value.split())[:300]


def _is_brand_only_identity(candidate: CandidateVerification) -> bool:
    """Reject exact text that identifies only the known brand/logo, not a model."""
    identity_tokens = set(
        re.findall(r"[a-z0-9]+", candidate.resolved_identity.casefold())
    )
    evidence_tokens = set(
        re.findall(r"[a-z0-9]+", candidate.supporting_evidence.casefold())
    )
    if identity_tokens and identity_tokens <= _BRAND_ONLY_TERMS:
        return True
    return bool(identity_tokens & _KNOWN_BRAND_IDENTITIES) and not (
        evidence_tokens - _BRAND_ONLY_TERMS
    )


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

    Used for retrieval enrichment and candidate selection, never as directly
    citable evidence or final-answer context.
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


def page_identity(document: Document) -> tuple[str, int] | None:
    """Return a document's stable source/page identity across nv-ingest and NRL layouts."""
    metadata = document.metadata or {}
    content_metadata = metadata.get("content_metadata")
    if not isinstance(content_metadata, dict):
        content_metadata = {}
    source = metadata.get("source")
    if isinstance(source, dict):
        source_name = source.get("source_name") or source.get("source_id")
    else:
        source_name = source or metadata.get("path") or metadata.get("filename")
    page_number = content_metadata.get("page_number")
    if page_number is None:
        page_number = metadata.get("page_number")
    if page_number is None:
        page_number = content_metadata.get("page_num") or content_metadata.get("page")
    if not isinstance(source_name, str) or not source_name or page_number is None:
        return None
    if isinstance(page_number, bool):
        return None
    if isinstance(page_number, int):
        return source_name, page_number
    if isinstance(page_number, str) and re.fullmatch(r"\d+", page_number):
        return source_name, int(page_number)
    return None


def dedupe_by_page(documents: list[Document], limit: int) -> list[Document]:
    """Keep the first ranked document for each source/page identity."""
    if limit <= 0:
        return []
    result: list[Document] = []
    seen: set[tuple[str, int]] = set()
    for document in documents:
        identity = page_identity(document)
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
        identity = page_identity(document)
        if identity is None:
            logger.warning(
                "Skipping candidate without source/page metadata for multimodal fusion"
            )
            continue
        visual_by_identity.setdefault(identity, (rank, document))

    for rank, document in enumerate(text_documents, start=1):
        identity = page_identity(document)
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
