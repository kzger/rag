# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate weak-text multimodal queries through the public generate API."""

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import statistics
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, NamedTuple

import requests

DEFAULT_REJECTION_TERMS = (
    "cannot confirm",
    "cannot identify",
    "does not contain enough evidence",
    "not enough evidence",
    "unable to determine",
    "無法確認",
    "無法判定",
    "沒有相關資料",
    "證據不足",
)
# A correct identification often uses a bare negator in passing ("...也不是紅黑相間的耳機"),
# so these must match a denial *of the identity*, not any negation anywhere in the answer.
IDENTIFICATION_NEGATION_PATTERNS = (
    re.compile(
        r"(不是|並非|不屬於|並不是|不太可能是|應該不是|恐怕不是)"
        r"[^。！？；\n]{0,15}?(同一|同款|這個產品|這款|該產品|同型號)"
    ),
    re.compile(
        r"\b(is|are)\s+not\b[^.!?\n]{0,25}?\b(the same|this product|that product)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(isn't|aren't)\b[^.!?\n]{0,25}?\b(the same|this product|that product)\b",
        re.IGNORECASE,
    ),
)
# A negator immediately preceding the expected model name ("這不是 J7EF Plus") is also a
# denial. Anchored to the end of the window so it must sit just before that mention.
NEGATOR_BEFORE_TERM = re.compile(
    r"(不是|並非|不屬於|並不是|不太可能是|應該不是|恐怕不是"
    r"|\bis\s+not\b|\bare\s+not\b|\bisn't\b|\baren't\b|\bnot\b)"
    r"[^。！？；.!?\n]{0,10}$",
    re.IGNORECASE,
)
DATA_URI_PATTERN = re.compile(r"data:[^;,\s]+;base64,[A-Za-z0-9+/=_-]+")


class HistoryMessage(NamedTuple):
    """One prior conversation message used by an evaluation case."""

    role: str
    text: str
    image_path: Path | None = None


class EvaluationCase(NamedTuple):
    """One annotated multimodal quality case."""

    case_id: str
    prompt: str
    image_path: Path
    expected_sources: list[str]
    accepted_answer_terms: list[str]
    expected_rejection: bool
    forbidden_answer_terms: list[str]
    history: list[HistoryMessage]


class EvaluationDataset(NamedTuple):
    """A validated, versioned set of multimodal evaluation cases."""

    version: str
    manifest_path: Path
    cases: list[EvaluationCase]


class StreamResult(NamedTuple):
    """Parsed answer, citations, and user-visible streaming latency."""

    answer: str
    citations: list[str]
    ttft_seconds: float
    total_seconds: float


class CaseObservation(NamedTuple):
    """Observed search and generation behavior for one case."""

    case_id: str
    expected_sources: list[str]
    accepted_answer_terms: list[str]
    expected_rejection: bool
    answer: str
    candidate_sources: list[str]
    citations: list[str]
    ttft_seconds: float
    total_seconds: float
    search_seconds: float = 0.0
    forbidden_answer_terms: list[str] = []
    error: str | None = None


class EvaluationConfig(NamedTuple):
    """Non-sensitive deployment and retrieval settings for an evaluation run."""

    collection: str
    model: str
    vdb_top_k: int
    reranker_top_k: int


def _require_readable_asset(path: Path, case_id: str) -> Path:
    try:
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError("not a non-empty file")
        with path.open("rb") as stream:
            stream.read(16)
    except OSError as error:
        raise ValueError(
            f"Case {case_id!r} requires unreadable image asset {str(path)!r}: {error}"
        ) from error
    return path


def load_dataset(manifest_path: str | Path) -> EvaluationDataset:
    """Load and validate a versioned evaluation manifest and every image asset."""
    manifest = Path(manifest_path).resolve()
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Unable to read evaluation manifest {manifest}: {error}"
        ) from error

    version = raw.get("version")
    raw_cases = raw.get("cases")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("Evaluation manifest requires a non-empty version")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Evaluation manifest requires at least one case")

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for raw_case in raw_cases:
        case_id = str(raw_case.get("id", "")).strip()
        prompt = str(raw_case.get("prompt", "")).strip()
        if not case_id or case_id in seen_ids:
            raise ValueError(f"Case id must be non-empty and unique: {case_id!r}")
        if not prompt:
            raise ValueError(f"Case {case_id!r} requires a prompt")
        seen_ids.add(case_id)
        image_path = _require_readable_asset(
            (manifest.parent / str(raw_case.get("image", ""))).resolve(), case_id
        )

        history: list[HistoryMessage] = []
        for index, raw_message in enumerate(raw_case.get("history", [])):
            history_image = raw_message.get("image")
            history_path = None
            if history_image:
                history_path = _require_readable_asset(
                    (manifest.parent / str(history_image)).resolve(),
                    f"{case_id} history[{index}]",
                )
            history.append(
                HistoryMessage(
                    role=str(raw_message.get("role", "user")),
                    text=str(raw_message.get("text", "")),
                    image_path=history_path,
                )
            )

        cases.append(
            EvaluationCase(
                case_id=case_id,
                prompt=prompt,
                image_path=image_path,
                expected_sources=[
                    str(item) for item in raw_case.get("expected_sources", [])
                ],
                accepted_answer_terms=[
                    str(item) for item in raw_case.get("accepted_answer_terms", [])
                ],
                expected_rejection=bool(raw_case.get("expected_rejection", False)),
                forbidden_answer_terms=[
                    str(item) for item in raw_case.get("forbidden_answer_terms", [])
                ],
                history=history,
            )
        )
    return EvaluationDataset(version=version, manifest_path=manifest, cases=cases)


def _image_content(path: Path) -> dict[str, Any]:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{encoded}"},
    }


def _message_content(text: str, image_path: Path | None) -> str | list[dict[str, Any]]:
    if image_path is None:
        return text
    return [{"type": "text", "text": text}, _image_content(image_path)]


def build_generate_payload(
    case: EvaluationCase,
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Build a public /generate request with an explicit knowledge-base scope."""
    messages = [
        {
            "role": message.role,
            "content": _message_content(message.text, message.image_path),
        }
        for message in case.history
    ]
    messages.append(
        {"role": "user", "content": _message_content(case.prompt, case.image_path)}
    )
    payload: dict[str, Any] = {
        "messages": messages,
        "use_knowledge_base": True,
        "agentic": False,
        "collection_names": [config.collection],
        "vdb_top_k": config.vdb_top_k,
        "reranker_top_k": config.reranker_top_k,
        "enable_citations": True,
    }
    if config.model:
        payload["model"] = config.model
    return payload


def build_search_payload(
    case: EvaluationCase, config: EvaluationConfig
) -> dict[str, Any]:
    """Build a public /search request for the candidate Hit@K probe."""
    query = _message_content(case.prompt, case.image_path)
    return {
        "query": query,
        "messages": [],
        "collection_names": [config.collection],
        "vdb_top_k": config.vdb_top_k,
        "reranker_top_k": config.reranker_top_k,
        "enable_query_rewriting": False,
        "enable_reranker": True,
        "enable_citations": True,
    }


def _citation_identity(result: dict[str, Any]) -> str | None:
    document_name = result.get("document_name")
    if not document_name:
        return None
    metadata = result.get("metadata") or {}
    page_number = metadata.get("page_number")
    if page_number in (None, -1):
        return str(document_name)
    return f"{document_name}#page={page_number}"


def parse_search_candidates(payload: dict[str, Any]) -> list[str]:
    """Preserve the public /search result order as candidate ranks."""
    candidates: list[str] = []
    for result in payload.get("results", []):
        identity = _citation_identity(result)
        if identity and identity not in candidates:
            candidates.append(identity)
    return candidates


def parse_generate_stream(
    lines: Iterable[bytes | str],
    started_at: float,
    now: Callable[[], float] = time.perf_counter,
) -> StreamResult:
    """Parse the server's SSE stream without retaining request image data."""
    answer_parts: list[str] = []
    citations: list[str] = []
    first_token_at: float | None = None
    for line in lines:
        raw_line = line.decode("utf-8") if isinstance(line, bytes) else line
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        if raw_line.startswith("data:"):
            raw_line = raw_line[5:].strip()
        chunk = json.loads(raw_line)
        for result in (chunk.get("citations") or {}).get("results", []):
            identity = _citation_identity(result)
            if identity and identity not in citations:
                citations.append(identity)
        choices = chunk.get("choices") or []
        content = choices[0].get("message", {}).get("content", "") if choices else ""
        if content:
            if first_token_at is None:
                first_token_at = now()
            answer_parts.append(content)
    finished_at = now()
    ttft = (first_token_at or finished_at) - started_at
    return StreamResult(
        answer="".join(answer_parts),
        citations=citations,
        ttft_seconds=max(ttft, 0.0),
        total_seconds=max(finished_at - started_at, 0.0),
    )


def _contains_any(text: str, terms: list[str] | tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(term.casefold() in lowered for term in terms)


def _denies_identity(
    text: str, accepted_terms: list[str] | tuple[str, ...] = ()
) -> bool:
    """Detect an answer that denies the identification, not any passing negation."""
    if any(pattern.search(text) for pattern in IDENTIFICATION_NEGATION_PATTERNS):
        return True
    lowered = text.casefold()
    for term in accepted_terms:
        for match in re.finditer(re.escape(term.casefold()), lowered):
            preceding = lowered[max(0, match.start() - 30) : match.start()]
            if NEGATOR_BEFORE_TERM.search(preceding):
                return True
    return False


def _candidate_hit(observation: CaseObservation) -> bool:
    return any(
        expected.casefold() == candidate.casefold()
        for expected in observation.expected_sources
        for candidate in observation.candidate_sources
    )


def _generation_citation_hit(observation: CaseObservation) -> bool:
    return any(
        expected.casefold() == citation.casefold()
        for expected in observation.expected_sources
        for citation in observation.citations
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "mean": statistics.fmean(values) if values else 0.0,
    }


def compute_metrics(observations: list[CaseObservation]) -> dict[str, Any]:
    """Compute deterministic baseline metrics from annotated observations."""
    retrieval_cases = [item for item in observations if item.expected_sources]
    identification_cases = [
        item for item in observations if not item.expected_rejection
    ]
    retrieval_hits = sum(_candidate_hit(item) for item in retrieval_cases)
    identification_results = {
        item.case_id: _candidate_hit(item)
        and _generation_citation_hit(item)
        and _contains_any(item.answer, item.accepted_answer_terms)
        and not _contains_any(item.answer, item.forbidden_answer_terms)
        and not _denies_identity(item.answer, item.accepted_answer_terms)
        for item in identification_cases
    }
    identification_hits = sum(identification_results.values())

    predicted_rejections = [
        _contains_any(item.answer, DEFAULT_REJECTION_TERMS) for item in observations
    ]
    true_positive = sum(
        predicted and item.expected_rejection
        for predicted, item in zip(predicted_rejections, observations, strict=True)
    )
    false_positive = sum(
        predicted and not item.expected_rejection
        for predicted, item in zip(predicted_rejections, observations, strict=True)
    )
    false_negative = sum(
        not predicted and item.expected_rejection
        for predicted, item in zip(predicted_rejections, observations, strict=True)
    )

    unsupported = sum(
        _contains_any(item.answer, item.forbidden_answer_terms)
        or (item.expected_rejection and not predicted)
        or (
            not item.expected_rejection
            and bool(item.accepted_answer_terms)
            and not identification_results[item.case_id]
        )
        for item, predicted in zip(observations, predicted_rejections, strict=True)
    )
    rejection_cases = [item for item in observations if item.expected_rejection]
    return {
        "case_count": len(observations),
        "candidate_retrieval_hit_at_k": retrieval_hits / len(retrieval_cases)
        if retrieval_cases
        else 0.0,
        "verified_identification_accuracy": identification_hits
        / len(identification_cases)
        if identification_cases
        else 0.0,
        "unsupported_claim_rate": unsupported / len(observations)
        if observations
        else 0.0,
        "low_confidence_rejection_precision": true_positive
        / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0,
        "low_confidence_rejection_recall": true_positive
        / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0,
        "rejection_citation_leak_rate": (
            sum(bool(item.citations) for item in rejection_cases) / len(rejection_cases)
            if rejection_cases
            else 0.0
        ),
        "latency_seconds": {
            "retrieval": _latency_summary(
                [item.search_seconds for item in observations]
            ),
            "ttft": _latency_summary([item.ttft_seconds for item in observations]),
            "total": _latency_summary([item.total_seconds for item in observations]),
        },
    }


def _asset_record(path: Path, manifest_dir: Path) -> dict[str, str]:
    return {
        "path": os.path.relpath(path, manifest_dir),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_report(
    dataset: EvaluationDataset,
    observations: list[CaseObservation],
    config: EvaluationConfig,
    variant: str = "gate-off",
) -> dict[str, Any]:
    """Create a reproducibility report containing no image bytes or credentials."""
    assets = sorted(
        {case.image_path for case in dataset.cases}
        | {
            message.image_path
            for case in dataset.cases
            for message in case.history
            if message.image_path is not None
        }
    )
    return {
        "variant": variant,
        "dataset": {
            "version": dataset.version,
            "manifest": dataset.manifest_path.name,
            "assets": [
                _asset_record(path, dataset.manifest_path.parent) for path in assets
            ],
        },
        "configuration": {
            "model": config.model or "server-default",
            "collection": config.collection,
            "vdb_top_k": config.vdb_top_k,
            "reranker_top_k": config.reranker_top_k,
        },
        "metric_definitions": {
            "candidate_retrieval_hit_at_k": (
                "Fraction of cases with an expected source among the ordered "
                "candidates returned by the public /v1/search probe; K is "
                "configuration.reranker_top_k."
            ),
            "verified_identification_accuracy": (
                "Fraction of identifiable cases whose answer contains an annotated "
                "accepted identity, contains no annotated forbidden term, does not "
                "deny the identity, and whose /v1/search candidates and /v1/generate "
                "citations both contain an expected source. Denial is matched as a "
                "phrase (a negator bound to 'the same product' or to an accepted "
                "model term), not as a bare substring, so a negation used in passing "
                "in an otherwise correct answer does not count. This lexical value is "
                "a diagnostic; the adjudicated report replaces it with manual verdicts."
            ),
            "unsupported_claim_rate": (
                "Case-level lexical diagnostic: fraction of outputs that fail an "
                "annotated identity, contain an annotated forbidden claim, or fail to "
                "reject a case annotated as no-match. The adjudicated report replaces "
                "it with manually reviewed atomic-claim counts."
            ),
            "low_confidence_rejection_precision_recall": (
                "Precision and recall of configured rejection phrases against cases "
                "annotated expected_rejection=true."
            ),
            "rejection_citation_leak_rate": (
                "Fraction of expected-rejection cases that emitted any citation; "
                "expected value is zero."
            ),
            "retrieval_latency": (
                "Wall-clock latency of the public /v1/search candidate probe."
            ),
        },
        "metrics": compute_metrics(observations),
        "cases": [
            {
                **item._asdict(),
                "answer": DATA_URI_PATTERN.sub("[REDACTED_IMAGE_DATA]", item.answer),
                "expected_rejection_has_zero_citations": (
                    not item.citations if item.expected_rejection else None
                ),
            }
            for item in observations
        ],
    }


def _observation_from_case(
    case: EvaluationCase,
    *,
    answer: str,
    candidate_sources: list[str],
    citations: list[str],
    ttft_seconds: float,
    total_seconds: float,
    search_seconds: float = 0.0,
    error: str | None = None,
) -> CaseObservation:
    """Combine case annotations with request-specific observed values."""
    return CaseObservation(
        case_id=case.case_id,
        expected_sources=case.expected_sources,
        accepted_answer_terms=case.accepted_answer_terms,
        expected_rejection=case.expected_rejection,
        answer=answer,
        candidate_sources=candidate_sources,
        citations=citations,
        ttft_seconds=ttft_seconds,
        total_seconds=total_seconds,
        search_seconds=search_seconds,
        forbidden_answer_terms=case.forbidden_answer_terms,
        error=error,
    )


def evaluate(
    dataset: EvaluationDataset,
    endpoint: str,
    config: EvaluationConfig,
    timeout: float,
) -> list[CaseObservation]:
    """Run all cases against a live public multimodal /generate endpoint."""
    observations: list[CaseObservation] = []
    base_url = endpoint.rstrip("/")
    for case in dataset.cases:
        generate_payload = build_generate_payload(case, config)
        search_payload = build_search_payload(case, config)
        started_at = time.perf_counter()
        try:
            search_response = requests.post(
                base_url + "/v1/search", json=search_payload, timeout=timeout
            )
            search_response.raise_for_status()
            candidate_sources = parse_search_candidates(search_response.json())
            search_seconds = time.perf_counter() - started_at
            generate_started_at = time.perf_counter()
            with requests.post(
                base_url + "/v1/generate",
                json=generate_payload,
                stream=True,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                stream = parse_generate_stream(
                    response.iter_lines(), generate_started_at
                )
            observations.append(
                _observation_from_case(
                    case,
                    answer=stream.answer,
                    candidate_sources=candidate_sources,
                    citations=stream.citations,
                    ttft_seconds=stream.ttft_seconds,
                    total_seconds=stream.total_seconds,
                    search_seconds=search_seconds,
                )
            )
        except (requests.RequestException, json.JSONDecodeError) as error:
            elapsed = time.perf_counter() - started_at
            observations.append(
                _observation_from_case(
                    case,
                    answer="",
                    candidate_sources=[],
                    citations=[],
                    ttft_seconds=elapsed,
                    total_seconds=elapsed,
                    search_seconds=elapsed,
                    error=f"{type(error).__name__}: {error}",
                )
            )
    return observations


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("multimodal_accuracy_cases.json"),
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8081")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--vdb-top-k", type=int, default=100)
    parser.add_argument("--reranker-top-k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="gate-off")
    return parser.parse_args()


def main() -> int:
    """Run the CLI and return nonzero when any live case fails."""
    args = _parse_args()
    if args.vdb_top_k < 1 or args.reranker_top_k < 1:
        raise SystemExit("candidate counts must be positive integers")
    if args.reranker_top_k > args.vdb_top_k:
        raise SystemExit("--reranker-top-k cannot exceed --vdb-top-k")
    dataset = load_dataset(args.dataset)
    config = EvaluationConfig(
        collection=args.collection,
        model=args.model,
        vdb_top_k=args.vdb_top_k,
        reranker_top_k=args.reranker_top_k,
    )
    observations = evaluate(
        dataset=dataset,
        endpoint=args.endpoint,
        config=config,
        timeout=args.timeout,
    )
    report = build_report(
        dataset=dataset,
        observations=observations,
        config=config,
        variant=args.variant,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))
    return 1 if any(item.error for item in observations) else 0


if __name__ == "__main__":
    raise SystemExit(main())
