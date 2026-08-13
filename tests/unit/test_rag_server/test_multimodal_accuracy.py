"""Public generate seam tests for multimodal accuracy retrieval wiring."""

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document
from nvidia_rag.rag_server import main
from nvidia_rag.rag_server.main import NvidiaRAG
from nvidia_rag.rag_server.multimodal_accuracy import (
    CandidateVerification,
    QueryUnderstanding,
)
from nvidia_rag.rag_server.response_generator import generate_answer_async


class DummyPrompt:
    """Minimal prompt/chain stub used by the public generate seam."""

    def __or__(self, other: object) -> "DummyPrompt":
        return self

    def stream(self, inputs: object, config: object = None) -> list[str]:
        return ["ok"]

    async def astream(
        self, inputs: object, config: object = None
    ) -> AsyncIterator[str]:
        yield "ok"


class RecordingVDB:
    """VDB seam that records visual and text retrieval calls."""

    last_visual_query: str | None = None
    last_text_query: str | None = None
    visual_diverse_pages: bool | None = None
    visual_reranker_top_k: int | None = None
    calls: list[tuple[str, str]] = []
    visual_result: list[Document] = []
    text_result: list[Document] = []

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        type(self).last_visual_query = None
        type(self).last_text_query = None
        type(self).visual_diverse_pages = None
        type(self).visual_reranker_top_k = None
        type(self).calls = []
        type(self).visual_result = []
        type(self).text_result = []

    def check_collection_exists(self, collection_name: str) -> bool:
        return True

    def get_langchain_vectorstore(self, collection_name: str) -> object:
        return object()

    def get_metadata_schema(self, collection_name: str) -> list[dict[str, object]]:
        return []

    def retrieval_image_langchain(
        self,
        query: str,
        collection_name: str,
        vectorstore: object | None = None,
        top_k: int | None = None,
        reranker_top_k: int | None = None,
        diverse_pages: bool = False,
    ) -> list[Document]:
        type(self).last_visual_query = query
        type(self).visual_diverse_pages = diverse_pages
        type(self).visual_reranker_top_k = reranker_top_k
        type(self).calls.append(("visual", query))
        return self.visual_result

    def retrieval_langchain(
        self,
        query: str,
        collection_name: str,
        vectorstore: object | None = None,
        top_k: int | None = None,
        filter_expr: str = "",
        otel_ctx: object | None = None,
    ) -> list[Document]:
        type(self).last_text_query = query
        type(self).calls.append(("text", query))
        return self.text_result


def page_doc(
    source: str,
    page: int,
    content: str = "",
    visual_rank: int | None = None,
    visual_score: float | None = None,
    text_score: float | None = None,
) -> Document:
    metadata: dict[str, object] = {
        "source": {"source_name": source, "source_id": source},
        "content_metadata": {"page_number": page, "type": "text"},
        "collection_name": "test",
    }
    if visual_rank is not None:
        metadata["image_retrieval"] = {
            "raw_score": visual_score,
            "rank": visual_rank,
            "chunk_count": 1,
            "chunks": [],
        }
    if text_score is not None:
        metadata["relevance_score"] = text_score
    return Document(page_content=content, metadata=metadata)


@pytest.fixture(autouse=True)
def stub_chat_prompt(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    monkeypatch.setenv("ENABLE_REFLECTION", "false")

    class DummyChatPromptTemplate:
        @staticmethod
        def from_messages(messages: object) -> DummyPrompt:
            return DummyPrompt()

    monkeypatch.setattr(main, "ChatPromptTemplate", DummyChatPromptTemplate)
    monkeypatch.setattr(main, "StrOutputParser", lambda: DummyPrompt())
    monkeypatch.setattr(main, "get_llm", lambda **kwargs: DummyPrompt())
    monkeypatch.setattr(main, "get_ranking_model", lambda **kwargs: None)

    async def mock_generate_answer_async(
        generator: object, contexts: object, **kwargs: object
    ) -> AsyncIterator[str]:
        yield "ok"

    real_response_tests = {
        "test_direct_disclosure_closes_underlying_stream_on_early_close",
        "test_public_generate_verified_keeps_selected_context_and_citation",
        "test_public_multimodal_generate_closes_prefetched_stream_before_iteration",
    }
    if request.node.name not in real_response_tests:
        monkeypatch.setattr(main, "generate_answer_async", mock_generate_answer_async)


def multimodal_messages() -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "這是什麼？"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,j7ef-current"},
                },
            ],
        }
    ]


def configure_vdb(monkeypatch: pytest.MonkeyPatch, vdb: RecordingVDB) -> None:
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kwargs: vdb)


async def generate_multimodal(rag: NvidiaRAG) -> None:
    await rag.generate(
        messages=multimodal_messages(),
        use_knowledge_base=True,
        collection_names=["test"],
        enable_reranker=False,
        enable_vlm_inference=True,
        vlm_max_total_images=2,
    )


async def generate_direct_multimodal(rag: NvidiaRAG, *, accuracy_enabled: bool) -> Any:
    rag.config.enable_multimodal_accuracy = accuracy_enabled
    return await rag.generate(
        messages=multimodal_messages(),
        use_knowledge_base=False,
        enable_vlm_inference=True,
        vlm_max_total_images=2,
    )


def sse_payloads(chunks: list[str]) -> list[dict[str, object]]:
    """Decode the JSON objects emitted by the public streaming response."""
    return [
        json.loads(chunk.removeprefix("data: ").strip())
        for chunk in chunks
        if chunk.startswith("data: ")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "accuracy_enabled, disclosure_expected", [(True, True), (False, False)]
)
async def test_direct_multimodal_no_kb_disclosure_is_feature_gated(
    monkeypatch: pytest.MonkeyPatch,
    accuracy_enabled: bool,
    disclosure_expected: bool,
) -> None:
    rag = NvidiaRAG()
    original_chunks = ["legacy direct answer"]
    monkeypatch.setattr(main, "generate_answer_async", generate_answer_async)

    async def stream(**kwargs: object) -> AsyncIterator[str]:
        yield original_chunks[0]

    with patch("nvidia_rag.rag_server.main.VLM") as vlm_class:
        vlm_class.return_value.stream_with_messages = stream
        response = await generate_direct_multimodal(
            rag, accuracy_enabled=accuracy_enabled
        )

    chunks = [chunk async for chunk in response.generator]
    disclosure = "Knowledge base was not used for this answer."
    assert any(disclosure in chunk for chunk in chunks) is disclosure_expected
    assert any(original_chunks[0] in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_direct_disclosure_closes_underlying_stream_on_early_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rag = NvidiaRAG()
    rag.config.enable_multimodal_accuracy = True

    class TrackedStream:
        def __init__(self) -> None:
            self.closed = False
            self.close_count = 0
            self.yielded = False

        def __aiter__(self) -> "TrackedStream":
            return self

        async def __anext__(self) -> str:
            if self.yielded:
                await asyncio.sleep(10)
            self.yielded = True
            return "legacy"

        async def aclose(self) -> None:
            self.close_count += 1
            self.closed = True

    tracked = TrackedStream()

    with patch("nvidia_rag.rag_server.main.VLM") as vlm_class:
        vlm_class.return_value.stream_with_messages = lambda **kwargs: tracked
        response = await generate_direct_multimodal(rag, accuracy_enabled=True)
    await response.generator.__anext__()
    await response.generator.aclose()
    assert tracked.closed
    await response.generator.aclose()
    assert tracked.close_count == 1


@pytest.mark.asyncio
async def test_public_generate_gate_off_keeps_legacy_final_vlm_and_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "true")
    vdb = RecordingVDB()
    vdb.visual_result = [page_doc("legacy.pdf", 2, "legacy")]
    rag = NvidiaRAG()
    rag.config.multimodal_accuracy.enable_verification_gate = False
    configure_vdb(monkeypatch, vdb)
    final_called = False

    async def stream(**kwargs: object) -> AsyncIterator[str]:
        nonlocal final_called
        final_called = True
        yield "legacy final answer"

    monkeypatch.setattr(main, "generate_answer_async", generate_answer_async)
    with patch("nvidia_rag.rag_server.main.VLM") as vlm_class:
        vlm = vlm_class.return_value
        vlm.understand_query_async = AsyncMock(return_value=None)
        vlm.stream_with_messages = stream
        response = await rag.generate(
            messages=multimodal_messages(),
            use_knowledge_base=True,
            collection_names=["test"],
            enable_reranker=False,
            enable_vlm_inference=True,
        )

    chunks = [chunk async for chunk in response.generator]
    assert final_called
    assert any("legacy final answer" in chunk for chunk in chunks)
    assert not any("knowledge base cannot confirm" in chunk.lower() for chunk in chunks)
    payloads = sse_payloads(chunks)
    citations = [
        payload["citations"] for payload in payloads if payload.get("citations")
    ]
    assert citations[0]["total_results"] == 1


@pytest.mark.asyncio
async def test_generate_uses_query_understanding_and_dual_retrieval(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "true")
    vdb = RecordingVDB()
    vdb.visual_result = [
        page_doc(
            "J7EF Sales kit.pdf",
            3,
            "J7EF Plus spec sheet",
            visual_rank=1,
            visual_score=0.8,
        )
    ]
    vdb.text_result = [
        page_doc("J7EF Sales kit.pdf", 3, "J7EF Plus spec sheet", text_score=0.9),
        page_doc("J10A Sales kit.pdf", 9, "J10A page", text_score=0.7),
    ]
    rag = NvidiaRAG()
    configure_vdb(monkeypatch, vdb)
    stream_kwargs: dict[str, object] = {}

    async def stream(**kwargs: object) -> AsyncIterator[str]:
        stream_kwargs.update(kwargs)
        yield "ok"

    with patch("nvidia_rag.rag_server.main.VLM") as vlm_class:
        vlm = vlm_class.return_value
        vlm.understand_query_async = AsyncMock(
            return_value=QueryUnderstanding(ocr_text="J7EF Plus", model="J7EF Plus")
        )
        vlm.stream_with_messages = stream
        with caplog.at_level("INFO"):
            await generate_multimodal(rag)

    assert vdb.visual_diverse_pages is True
    assert vdb.visual_reranker_top_k == 5
    assert "這是什麼？" in (vdb.last_visual_query or "")
    assert "j7ef-current" in (vdb.last_visual_query or "")
    assert "OCR text: J7EF Plus" in (vdb.last_text_query or "")
    assert "user question: 這是什麼？" in (vdb.last_text_query or "")
    assert {call[0] for call in vdb.calls} == {"visual", "text"}
    vlm.understand_query_async.assert_awaited_once()
    docs = stream_kwargs["docs"]  # type: ignore[index]
    assert len(docs) == 2
    assert docs[0].page_content == "J7EF Plus spec sheet"
    assert docs[0].metadata["multimodal_fusion"]["fusion_score"] == pytest.approx(
        0.5 / 61 + 0.5 / 61
    )
    assert docs[1].metadata["multimodal_fusion"]["fusion_score"] == pytest.approx(
        0.5 / 62
    )
    assert "data:image" not in caplog.text
    assert "base64" not in caplog.text


@pytest.mark.asyncio
async def test_generate_dual_retrieval_falls_back_to_raw_text_when_understanding_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "true")
    vdb = RecordingVDB()
    vdb.visual_result = [page_doc("visual.pdf", 1, "visual")]
    vdb.text_result = [page_doc("text.pdf", 2, "text")]
    rag = NvidiaRAG()
    configure_vdb(monkeypatch, vdb)

    stream_called = False

    async def stream(**kwargs: object) -> AsyncIterator[str]:
        nonlocal stream_called
        stream_called = True
        yield "ok"

    with patch("nvidia_rag.rag_server.main.VLM") as vlm_class:
        vlm = vlm_class.return_value
        vlm.understand_query_async = AsyncMock(return_value=None)
        vlm.stream_with_messages = stream
        with caplog.at_level("WARNING"):
            await generate_multimodal(rag)

    assert "user question: 這是什麼？" in (vdb.last_text_query or "")
    assert "OCR text:" not in (vdb.last_text_query or "")
    assert "falling back to raw multimodal query" in caplog.text
    assert stream_called


@pytest.mark.asyncio
async def test_generate_dual_retrieval_skips_text_path_when_accuracy_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "false")
    vdb = RecordingVDB()
    vdb.visual_result = [page_doc("visual.pdf", 1, "visual")]
    rag = NvidiaRAG()
    configure_vdb(monkeypatch, vdb)

    with patch("nvidia_rag.rag_server.main.VLM") as vlm_class:
        vlm = vlm_class.return_value
        vlm.understand_query_async = AsyncMock()

        async def stream(**kwargs: object) -> AsyncIterator[str]:
            yield "ok"

        vlm.stream_with_messages = stream
        await generate_multimodal(rag)

    assert [call[0] for call in vdb.calls] == ["visual"]
    vlm.understand_query_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_includes_conversation_summary_in_text_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "true")
    monkeypatch.setenv("CONVERSATION_HISTORY", "5")
    vdb = RecordingVDB()
    vdb.visual_result = [page_doc("visual.pdf", 1, "visual")]
    vdb.text_result = [page_doc("text.pdf", 2, "text")]
    rag = NvidiaRAG()
    configure_vdb(monkeypatch, vdb)
    messages = [
        {"role": "user", "content": "先前問題"},
        {"role": "assistant", "content": "先前圖片辨識為 J7EF Plus。"},
        {"role": "user", "content": "先前文字"},
        *multimodal_messages(),
    ]

    with patch("nvidia_rag.rag_server.main.VLM") as vlm_class:
        vlm = vlm_class.return_value
        vlm.understand_query_async = AsyncMock(return_value=QueryUnderstanding())

        async def stream(**kwargs: object) -> AsyncIterator[str]:
            yield "ok"

        vlm.stream_with_messages = stream
        await rag.generate(
            messages=messages,
            use_knowledge_base=True,
            collection_names=["test"],
            enable_reranker=False,
            enable_vlm_inference=True,
        )

    assert "conversation summary: 先前圖片辨識為 J7EF Plus。" in (
        vdb.last_text_query or ""
    )


@pytest.mark.asyncio
async def test_generate_logs_multimodal_accuracy_stage_timings(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "true")
    vdb = RecordingVDB()
    vdb.visual_result = [page_doc("visual.pdf", 1, "visual")]
    vdb.text_result = [page_doc("text.pdf", 2, "text")]
    rag = NvidiaRAG()
    configure_vdb(monkeypatch, vdb)

    with patch("nvidia_rag.rag_server.main.VLM") as vlm_class:
        vlm = vlm_class.return_value
        vlm.understand_query_async = AsyncMock(return_value=None)

        async def stream(**kwargs: object) -> AsyncIterator[str]:
            yield "ok"

        vlm.stream_with_messages = stream
        with caplog.at_level("INFO"):
            await generate_multimodal(rag)

    assert "Multimodal accuracy stages" in caplog.text


@pytest.mark.asyncio
async def test_public_multimodal_generate_logs_sanitized_latency_stages(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The public image+text seam exposes each retrieval and generation latency."""
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "true")
    vdb = RecordingVDB()
    vdb.visual_result = [page_doc("visual.pdf", 1, "visual")]
    vdb.text_result = [page_doc("text.pdf", 2, "text")]
    rag = NvidiaRAG()
    configure_vdb(monkeypatch, vdb)
    # Frozen clock: this test pins the log *shape* and its sanitization. The elapsed
    # value itself is asserted directly against _GenerationLatencyStream below, so it
    # does not have to depend on how many perf_counter calls the pipeline happens to make.
    monkeypatch.setattr(main.time, "perf_counter", lambda: 100.0)

    async def stream(**kwargs: object) -> AsyncIterator[str]:
        yield "safe answer"

    async def consume_generate_answer_async(
        generator: object, contexts: object, **kwargs: object
    ) -> AsyncIterator[str]:
        async for _ in generator:  # type: ignore[union-attr]
            yield "safe sse chunk"

    monkeypatch.setattr(main, "generate_answer_async", consume_generate_answer_async)
    with patch("nvidia_rag.rag_server.main.VLM") as vlm_class:
        vlm = vlm_class.return_value
        vlm.understand_query_async = AsyncMock(return_value=None)
        vlm.stream_with_messages = stream
        with caplog.at_level("INFO"):
            response = await rag.generate(
                messages=multimodal_messages(),
                use_knowledge_base=True,
                collection_names=["test"],
                enable_reranker=False,
                enable_vlm_inference=True,
            )
            async for _ in response.generator:
                pass

    assert "visual_retrieval_ms=" in caplog.text
    assert "text_retrieval_ms=" in caplog.text
    generation_latency = re.search(
        r"elapsed_ms=(?P<latency>[0-9]+(?:\.[0-9]+)?)", caplog.text
    )
    assert "Generation latency: status=completed" in caplog.text
    assert generation_latency is not None
    assert "data:image" not in caplog.text
    assert "j7ef-current" not in caplog.text
    assert "safe answer" not in caplog.text


@pytest.mark.asyncio
async def test_public_multimodal_generate_closes_prefetched_stream_before_iteration(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Closing the returned SSE before iteration closes the started model stream."""
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "true")
    monkeypatch.setattr(main.time, "perf_counter", lambda: 100.0)
    vdb = RecordingVDB()
    vdb.visual_result = [page_doc("visual.pdf", 1, "visual")]
    vdb.text_result = [page_doc("text.pdf", 2, "text")]
    rag = NvidiaRAG()
    configure_vdb(monkeypatch, vdb)
    monkeypatch.setattr(rag, "_is_shared_vdb_op", lambda _: True)
    upstream_closed = False

    async def stream(**kwargs: object) -> AsyncIterator[str]:
        nonlocal upstream_closed
        try:
            yield "safe answer"
            await asyncio.Future()
        finally:
            upstream_closed = True

    with patch("nvidia_rag.rag_server.main.VLM") as vlm_class:
        vlm = vlm_class.return_value
        vlm.understand_query_async = AsyncMock(return_value=None)
        vlm.stream_with_messages = stream
        with caplog.at_level("INFO"):
            response = await rag.generate(
                messages=multimodal_messages(),
                use_knowledge_base=True,
                collection_names=["test"],
                enable_reranker=False,
                enable_vlm_inference=True,
            )
            await response.generator.aclose()

    assert upstream_closed
    assert caplog.text.count("Generation latency: status=cancelled") == 1


class _LifecycleStream:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.closed = False

    def __aiter__(self) -> "_LifecycleStream":
        return self

    async def __anext__(self) -> str:
        if self.error is not None:
            raise self.error
        self.error = StopAsyncIteration()
        return "chunk"

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_generation_latency_stream_logs_cancelled_on_early_close(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(main.time, "perf_counter", iter((10.0, 10.01)).__next__)
    upstream = _LifecycleStream()
    stream = main._GenerationLatencyStream(upstream)

    await stream.aclose()

    assert upstream.closed
    assert "Generation latency: status=cancelled elapsed_ms=10.0" in caplog.text


@pytest.mark.asyncio
async def test_generation_latency_stream_preserves_upstream_error_when_close_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(main.time, "perf_counter", iter((20.0, 20.01)).__next__)
    original = ValueError("upstream failure")
    upstream = _LifecycleStream(original)

    async def close_with_error() -> None:
        raise RuntimeError("close failure")

    upstream.aclose = close_with_error
    stream = main._GenerationLatencyStream(upstream)

    with pytest.raises(ValueError, match="upstream failure"):
        await stream.__anext__()

    assert "Generation latency: status=error elapsed_ms=10.0" in caplog.text
    assert "close failure" not in caplog.text


@pytest.mark.asyncio
async def test_generation_latency_stream_preserves_cancellation_when_close_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(main.time, "perf_counter", iter((30.0, 30.01)).__next__)
    upstream = _LifecycleStream(asyncio.CancelledError())

    async def close_with_error() -> None:
        raise RuntimeError("close failure")

    upstream.aclose = close_with_error
    stream = main._GenerationLatencyStream(upstream)

    with pytest.raises(asyncio.CancelledError):
        await stream.__anext__()

    assert "Generation latency: status=cancelled elapsed_ms=10.0" in caplog.text


@pytest.mark.asyncio
async def test_generate_fused_candidate_prefers_visual_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "true")
    vdb = RecordingVDB()
    vdb.visual_result = [
        page_doc(
            "same.pdf",
            4,
            "visual page content",
            visual_rank=1,
            visual_score=0.8,
        )
    ]
    vdb.text_result = [page_doc("same.pdf", 4, "text page content", text_score=0.9)]
    rag = NvidiaRAG()
    configure_vdb(monkeypatch, vdb)
    stream_kwargs: dict[str, object] = {}

    async def stream(**kwargs: object) -> AsyncIterator[str]:
        stream_kwargs.update(kwargs)
        yield "ok"

    with patch("nvidia_rag.rag_server.main.VLM") as vlm_class:
        vlm = vlm_class.return_value
        vlm.understand_query_async = AsyncMock(return_value=None)
        vlm.stream_with_messages = stream
        await generate_multimodal(rag)

    fused_doc = stream_kwargs["docs"][0]
    assert fused_doc.page_content == "visual page content"
    assert "image_retrieval" in fused_doc.metadata
    assert "multimodal_fusion" in fused_doc.metadata


@pytest.mark.asyncio
async def test_verification_candidate_building_preserves_page_image_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The expanded page image shape used by legacy VLM assembly reaches verification."""
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "true")
    rag = NvidiaRAG()
    source = {"source_name": "J7EF Sales kit.pdf"}
    candidate = Document(
        page_content="retrieval caption",
        metadata={
            "source": source,
            "content_metadata": {"page_number": 3},
        },
    )
    expanded_image = Document(
        page_content="page image caption",
        metadata={
            "source": {
                "source_name": "J7EF Sales kit.pdf",
                "source_location": "s3://bucket/page-3.png",
            },
            "content_metadata": {"type": "image", "page_number": 3},
        },
    )
    captured: dict[str, object] = {}

    class RecordingVLM:
        def __init__(self, **_: object) -> None:
            pass

        async def verify_candidates_async(self, **kwargs: object) -> None:
            captured.update(kwargs)
            return None

    monkeypatch.setattr("nvidia_rag.rag_server.main.VLM", RecordingVLM)
    await rag._verify_multimodal_candidates(
        query=multimodal_messages()[0]["content"],
        candidates=[candidate],
        expanded_context=[expanded_image],
        question_text="這是什麼？",
        vlm_settings={},
    )

    verification_candidate = captured["candidates"][0]
    assert verification_candidate.metadata["source"]["source_location"] == (
        "s3://bucket/page-3.png"
    )


@pytest.mark.asyncio
async def test_public_generate_verified_keeps_selected_context_and_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "true")
    monkeypatch.setenv("ENABLE_MULTIMODAL_VERIFICATION_GATE", "true")
    vdb = RecordingVDB()
    vdb.visual_result = [page_doc("J7EF.pdf", 3, "J7EF page")]
    vdb.text_result = [
        page_doc("J7EF.pdf", 3, "J7EF page"),
        page_doc("J10A.pdf", 4, "J10A page"),
    ]
    rag = NvidiaRAG()
    rag.config.multimodal_accuracy.enable_verification_gate = True
    configure_vdb(monkeypatch, vdb)
    monkeypatch.setattr(main, "generate_answer_async", generate_answer_async)
    stream_kwargs: dict[str, object] = {}

    async def stream(**kwargs: object) -> AsyncIterator[str]:
        stream_kwargs.update(kwargs)
        yield "verified"

    with patch("nvidia_rag.rag_server.main.VLM") as vlm_class:
        vlm = vlm_class.return_value
        vlm.understand_query_async = AsyncMock(
            return_value=QueryUnderstanding(model="J7EF")
        )
        vlm.verify_candidates_async = AsyncMock(
            return_value=[
                CandidateVerification(
                    "C1",
                    "match",
                    0.95,
                    "exact_visible_text",
                    supporting_evidence="visible J7EF model token",
                    resolved_identity="J7EF",
                ),
                CandidateVerification("C2", "insufficient", 0.4, "generic_similarity"),
            ]
        )
        vlm.stream_with_messages = stream
        response = await rag.generate(
            messages=multimodal_messages(),
            use_knowledge_base=True,
            collection_names=["test"],
            enable_reranker=False,
            enable_vlm_inference=True,
            vlm_max_total_images=2,
        )

    docs = stream_kwargs["docs"]
    assert len(docs) == 1
    assert docs[0].metadata["source"]["source_name"] == "J7EF.pdf"
    chunks = [chunk async for chunk in response.generator]
    payloads = sse_payloads(chunks)
    citation_payloads = [
        payload["citations"]
        for payload in payloads
        if payload.get("citations") is not None
    ]
    assert citation_payloads
    citations = citation_payloads[0]
    assert citations["total_results"] == 1
    assert citations["results"][0]["document_name"] == "J7EF.pdf"
    assert all(result["document_name"] != "J10A.pdf" for result in citations["results"])


@pytest.mark.asyncio
async def test_public_generate_ambiguous_lists_identity_evidence_without_final_vlm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "true")
    monkeypatch.setenv("ENABLE_MULTIMODAL_VERIFICATION_GATE", "true")
    vdb = RecordingVDB()
    vdb.visual_result = [page_doc("J7EF.pdf", 3, "J7EF page")]
    vdb.text_result = [
        page_doc("J7EF.pdf", 3, "J7EF page"),
        page_doc("J10A.pdf", 4, "J10A page"),
        page_doc("generic.pdf", 5, "generic similarity page"),
    ]
    rag = NvidiaRAG()
    rag.config.multimodal_accuracy.enable_verification_gate = True
    configure_vdb(monkeypatch, vdb)
    monkeypatch.setattr(main, "generate_answer_async", generate_answer_async)
    final_stream_called = False

    async def final_stream(**kwargs: object) -> AsyncIterator[str]:
        nonlocal final_stream_called
        final_stream_called = True
        yield "must not be called"

    with patch("nvidia_rag.rag_server.main.VLM") as vlm_class:
        vlm = vlm_class.return_value
        vlm.understand_query_async = AsyncMock(return_value=None)
        vlm.verify_candidates_async = AsyncMock(
            return_value=[
                CandidateVerification(
                    "C1",
                    "match",
                    0.91,
                    "exact_visible_text",
                    supporting_evidence="visible J7EF token",
                    resolved_identity="J7EF Plus",
                ),
                CandidateVerification(
                    "C2",
                    "match",
                    0.90,
                    "visual_identity",
                    supporting_evidence="distinctive J10A label",
                    conflicts=("different model label",),
                    resolved_identity="J10A",
                ),
                CandidateVerification(
                    "C3",
                    "match",
                    0.99,
                    "generic_similarity",
                    resolved_identity="generic",
                ),
            ]
        )
        vlm.stream_with_messages = final_stream
        response = await rag.generate(
            messages=multimodal_messages(),
            use_knowledge_base=True,
            collection_names=["test"],
            enable_reranker=False,
            enable_vlm_inference=True,
        )

    chunks = [chunk async for chunk in response.generator]
    assert not final_stream_called
    assert any(
        "J7EF Plus" in chunk and "visible J7EF token" in chunk for chunk in chunks
    )
    assert any("J10A" in chunk and "different model label" in chunk for chunk in chunks)
    assert not any("generic" in chunk for chunk in chunks)
    payloads = sse_payloads(chunks)
    citation_payloads = [
        payload["citations"]
        for payload in payloads
        if payload.get("citations") is not None
    ]
    assert citation_payloads[0]["total_results"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verification_result, expected_text",
    [
        (None, "cannot uniquely identify"),
        (
            [CandidateVerification("C1", "mismatch", 0.99, "exact_visible_text")],
            "knowledge base cannot confirm",
        ),
    ],
)
async def test_public_generate_abstains_without_final_vlm(
    monkeypatch: pytest.MonkeyPatch,
    verification_result: list[CandidateVerification] | None,
    expected_text: str,
) -> None:
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "true")
    monkeypatch.setenv("ENABLE_MULTIMODAL_VERIFICATION_GATE", "true")
    vdb = RecordingVDB()
    vdb.visual_result = [page_doc("wrong.pdf", 1, "wrong")]
    vdb.text_result = [page_doc("wrong.pdf", 1, "wrong")]
    rag = NvidiaRAG()
    rag.config.multimodal_accuracy.enable_verification_gate = True
    configure_vdb(monkeypatch, vdb)
    monkeypatch.setattr(main, "generate_answer_async", generate_answer_async)
    final_stream_called = False

    async def stream(**kwargs: object) -> AsyncIterator[str]:
        nonlocal final_stream_called
        final_stream_called = True
        yield "unsafe final answer"

    with patch("nvidia_rag.rag_server.main.VLM") as vlm_class:
        vlm = vlm_class.return_value
        vlm.understand_query_async = AsyncMock(return_value=None)
        vlm.verify_candidates_async = AsyncMock(return_value=verification_result)
        vlm.stream_with_messages = stream
        response = await rag.generate(
            messages=multimodal_messages(),
            use_knowledge_base=True,
            collection_names=["test"],
            enable_reranker=False,
            enable_vlm_inference=True,
            vlm_max_total_images=2,
        )

    assert not final_stream_called
    assert response.generator is not None
    chunks = [chunk async for chunk in response.generator]
    response_text = "".join(
        payload["choices"][0]["message"]["content"]
        for payload in sse_payloads(chunks)
        if payload.get("choices") and payload["choices"][0].get("message")
    ).lower()
    assert expected_text in response_text
    payloads = sse_payloads(chunks)
    citation_payloads = [
        payload["citations"]
        for payload in payloads
        if payload.get("citations") is not None
    ]
    assert citation_payloads
    assert citation_payloads[0]["total_results"] == 0
    if expected_text == "malformed":
        assert vlm.verify_candidates_async.await_count == 1


@pytest.mark.asyncio
async def test_public_generate_verification_service_error_keeps_error_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "true")
    monkeypatch.setenv("ENABLE_MULTIMODAL_VERIFICATION_GATE", "true")
    vdb = RecordingVDB()
    vdb.visual_result = [page_doc("candidate.pdf", 1, "candidate")]
    rag = NvidiaRAG()
    rag.config.multimodal_accuracy.enable_verification_gate = True
    configure_vdb(monkeypatch, vdb)

    with patch("nvidia_rag.rag_server.main.VLM") as vlm_class:
        vlm = vlm_class.return_value
        vlm.understand_query_async = AsyncMock(return_value=None)
        vlm.verify_candidates_async = AsyncMock(
            side_effect=RuntimeError("verifier down")
        )
        response = await rag.generate(
            messages=multimodal_messages(),
            use_knowledge_base=True,
            collection_names=["test"],
            enable_reranker=False,
            enable_vlm_inference=True,
        )

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_generation_latency_stream_reports_elapsed_time(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Elapsed time spans construction to termination, and is logged exactly once."""
    clock = iter([100.0, 100.025, 100.999])
    monkeypatch.setattr(main.time, "perf_counter", lambda: next(clock, 100.999))

    class Upstream:
        def __init__(self) -> None:
            self.closed = 0

        def __aiter__(self) -> "Upstream":
            return self

        async def __anext__(self) -> str:
            raise StopAsyncIteration

        async def aclose(self) -> None:
            self.closed += 1

    upstream = Upstream()
    stream = main._GenerationLatencyStream(upstream)

    with caplog.at_level("INFO"):
        async for _ in stream:
            pass
        # A close after natural exhaustion must not log or close a second time.
        await stream.aclose()

    assert caplog.text.count("Generation latency:") == 1
    assert "status=completed elapsed_ms=25.0" in caplog.text
    assert upstream.closed == 1


@pytest.mark.asyncio
async def test_visual_and_text_retrieval_are_timed_independently(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The two retrieval paths report their own durations, not a shared total."""
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "true")
    vdb = RecordingVDB()
    vdb.visual_result = [page_doc("visual.pdf", 1, "visual")]
    vdb.text_result = [page_doc("text.pdf", 2, "text")]

    # Make the visual path measurably slower than the text path.
    original_visual = vdb.retrieval_image_langchain

    def slow_visual(*args: object, **kwargs: object) -> list[Document]:
        time.sleep(0.08)
        return original_visual(*args, **kwargs)

    monkeypatch.setattr(vdb, "retrieval_image_langchain", slow_visual)
    rag = NvidiaRAG()
    configure_vdb(monkeypatch, vdb)

    async def stream(**kwargs: object) -> AsyncIterator[str]:
        yield "answer"

    with patch("nvidia_rag.rag_server.main.VLM") as vlm_class:
        vlm = vlm_class.return_value
        vlm.understand_query_async = AsyncMock(return_value=None)
        vlm.stream_with_messages = stream
        with caplog.at_level("INFO"):
            await generate_multimodal(rag)

    visual = float(re.search(r"visual_retrieval_ms=([0-9.]+)", caplog.text).group(1))
    text = float(re.search(r"text_retrieval_ms=([0-9.]+)", caplog.text).group(1))
    assert (
        visual > text
    ), f"visual={visual} text={text} — paths are not timed separately"
    assert visual >= 80.0
