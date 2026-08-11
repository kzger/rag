# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test suite for query rewriting functionality in the RAG server.
"""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from nvidia_rag.rag_server.main import NvidiaRAG


class DummyPrompt:
    """A minimal LCEL-like object that supports piping and invoke/ainvoke/format_messages."""

    def __init__(self, rewritten_prefix: str = "REWRITTEN"):
        self.rewritten_prefix = rewritten_prefix

    def __or__(self, other):  # support chaining: prompt | llm | parser | output
        return self

    def invoke(self, inputs, config=None):
        # Mimic rewriter returning a transformed query from the provided input
        value = inputs.get("input") or inputs.get("question") or ""
        return f"{self.rewritten_prefix}({value})"

    async def ainvoke(self, inputs, config=None):
        # Async version of invoke
        value = inputs.get("input") or inputs.get("question") or ""
        return f"{self.rewritten_prefix}({value})"

    def stream(self, inputs, config=None):
        # Minimal streaming generator to satisfy generate() call path
        yield "ok"

    async def astream(self, inputs, config=None):
        # Minimal async streaming generator
        yield "ok"

    def format_messages(self, **kwargs):
        # Return a list-like structure for logging compatibility
        class Msg:
            def __init__(self, type_, content):
                self.type = type_
                self.content = content

        return [
            Msg("system", "dummy-system"),
            Msg("human", f"{kwargs}"),
        ]


class DummyVDB:
    """A minimal VDB stub used via monkeypatch on __prepare_vdb_op."""

    last_query = None
    last_retrieval_method = None
    last_diverse_pages = None

    def check_collection_exists(self, collection_name: str) -> bool:
        return True

    def get_langchain_vectorstore(self, collection_name: str):
        return object()

    def get_metadata_schema(self, collection_name: str):
        return []

    def retrieval_langchain(
        self,
        query,
        collection_name,
        vectorstore=None,
        top_k=None,
        filter_expr="",
        otel_ctx=None,
    ):
        """Sync method - called in ThreadPoolExecutor or directly."""
        DummyVDB.last_query = query
        DummyVDB.last_retrieval_method = "langchain"
        return []

    def retrieval_image_langchain(
        self,
        query: str,
        collection_name: str,
        vectorstore: object | None = None,
        top_k: int | None = None,
        reranker_top_k: int | None = None,
        diverse_pages: bool = False,
    ) -> list[object]:
        """Called when query contains images (multimodal)."""
        DummyVDB.last_query = query
        DummyVDB.last_retrieval_method = "image"
        DummyVDB.last_diverse_pages = diverse_pages
        return []


@pytest.fixture(autouse=True)
def stub_chat_prompt(monkeypatch):
    # Disable reflection for these tests so search/generate follow the
    # non-reflection code paths (reflection behaviour is covered elsewhere).
    monkeypatch.setenv("ENABLE_REFLECTION", "false")

    # Replace ChatPromptTemplate.from_messages to avoid real LCEL graph
    import nvidia_rag.rag_server.main as main

    class DummyChatPromptTemplate:
        @staticmethod
        def from_messages(messages):
            return DummyPrompt()

    monkeypatch.setattr(main, "ChatPromptTemplate", DummyChatPromptTemplate)

    # Ensure StreamingFilterThinkParser and StrOutputParser are no-ops in the chain
    class NoOpParser:
        def __ror__(self, other):
            return other

    # StreamingFilterThinkParser is now an instance attribute, not module-level
    # monkeypatch.setattr(main, "StreamingFilterThinkParser", NoOpParser())

    class NoOpStrOutputParser:
        def __ror__(self, other):
            return other

    monkeypatch.setattr(main, "StrOutputParser", lambda: NoOpStrOutputParser())

    # Stub LLM and ranker to avoid external calls during generate()
    monkeypatch.setattr(main, "get_llm", lambda **kwargs: DummyPrompt())
    monkeypatch.setattr(main, "get_ranking_model", lambda **kwargs: None)

    # Mock generate_answer_async to return a simple async generator
    async def mock_generate_answer_async(generator, contexts, **kw):
        yield "ok"

    monkeypatch.setattr(main, "generate_answer_async", mock_generate_answer_async)


@pytest.mark.asyncio
async def test_search_uses_query_rewriter_when_enabled(monkeypatch):
    """Test that query rewriting is used when enabled and messages are provided."""
    from nvidia_rag.rag_server.main import NvidiaRAG

    # Set CONVERSATION_HISTORY > 0, required for query rewriting to work
    monkeypatch.setenv("CONVERSATION_HISTORY", "5")

    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    # Force using our stubbed vdb_op inside generate/search path that may call __prepare_vdb_op
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
    ]

    # Act
    await rag.search(
        query="How does it work?",
        messages=messages,
        collection_names=["test"],
        enable_query_rewriting=True,
        enable_reranker=False,
        filter_expr="",
    )

    # Assert: rewritten query should be used for retrieval
    assert fake_vdb.last_query == "REWRITTEN(How does it work?)"


@pytest.mark.asyncio
async def test_search_skips_query_rewriter_when_history_is_zero(monkeypatch, caplog):
    """Test that query rewriting is skipped with a warning when CONVERSATION_HISTORY=0."""
    import logging

    from nvidia_rag.rag_server.main import NvidiaRAG

    # Set CONVERSATION_HISTORY to 0 (default)
    monkeypatch.setenv("CONVERSATION_HISTORY", "0")

    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    # Force using our stubbed vdb_op inside generate/search path that may call __prepare_vdb_op
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
    ]

    # Act
    with caplog.at_level(logging.WARNING):
        await rag.search(
            query="How does it work?",
            messages=messages,
            collection_names=["test"],
            enable_query_rewriting=True,
            enable_reranker=False,
            filter_expr="",
        )

    # Assert: query rewriting should be skipped and original query used
    assert fake_vdb.last_query == "How does it work?"
    # Assert: a warning should be logged
    assert any(
        "Query rewriting is enabled but CONVERSATION_HISTORY is set to 0"
        in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_search_uses_only_current_query_when_history_disabled(monkeypatch):
    """Test that when multiturn_retrieval_simple is False (default), only current query is used."""
    from nvidia_rag.rag_server.main import NvidiaRAG

    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
    ]

    # Act: multiturn_retrieval_simple defaults to False
    await rag.search(
        query="How does it work?",
        messages=messages,
        collection_names=["test"],
        enable_query_rewriting=False,
        enable_reranker=False,
        filter_expr="",
    )

    # Assert: only current query is used (no history concatenation)
    assert fake_vdb.last_query == "How does it work?"


@pytest.mark.asyncio
async def test_search_combines_history_when_multiturn_enabled(monkeypatch):
    """Test that when multiturn_retrieval_simple is True, history is concatenated."""
    from nvidia_rag.rag_server.main import NvidiaRAG

    # Enable multiturn retrieval via environment variable BEFORE creating NvidiaRAG instance
    monkeypatch.setenv("MULTITURN_RETRIEVER_SIMPLE", "True")

    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
    ]

    # Act
    await rag.search(
        query="How does it work?",
        messages=messages,
        collection_names=["test"],
        enable_query_rewriting=False,
        enable_reranker=False,
        filter_expr="",
    )

    # Assert: history + current query should be concatenated with '. '
    assert fake_vdb.last_query == "What is RAG?. How does it work?"


@pytest.mark.asyncio
async def test_search_skips_query_rewriter_for_image_query(monkeypatch):
    """When query is multimodal with image, query rewriting is skipped and retrieval_image_langchain is used."""
    from nvidia_rag.rag_server.main import NvidiaRAG

    monkeypatch.setenv("CONVERSATION_HISTORY", "5")
    monkeypatch.setenv("ENABLE_REFLECTION", "false")

    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    multimodal_query = [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
    ]
    messages = [
        {"role": "user", "content": "Previous question"},
        {"role": "assistant", "content": "Previous answer"},
    ]

    await rag.search(
        query=multimodal_query,
        messages=messages,
        collection_names=["test"],
        enable_query_rewriting=True,
        enable_reranker=False,
        filter_expr="",
    )

    # Assert: query rewriting skipped - last_query is text + image URL (no "REWRITTEN(...)")
    assert fake_vdb.last_query == "What is in this image? data:image/png;base64,x"
    assert "REWRITTEN" not in str(fake_vdb.last_query)
    # Assert: retrieval_image_langchain was used (not retrieval_langchain)
    assert fake_vdb.last_retrieval_method == "image"


@pytest.mark.asyncio
async def test_search_skips_reflection_for_image_query(monkeypatch):
    """Image search must not send a base64 data URL through text reflection."""
    from nvidia_rag.rag_server.main import NvidiaRAG

    monkeypatch.setenv("ENABLE_REFLECTION", "true")

    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    multimodal_query = [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
    ]

    with patch(
        "nvidia_rag.rag_server.main.check_context_relevance",
        new_callable=AsyncMock,
        return_value=([], True),
    ) as mock_check_relevance:
        await rag.search(
            query=multimodal_query,
            messages=[],
            collection_names=["test"],
            enable_reranker=True,
            filter_expr="",
        )

    mock_check_relevance.assert_not_awaited()
    assert fake_vdb.last_retrieval_method == "image"


@pytest.mark.asyncio
async def test_generate_uses_query_rewriter_when_enabled(monkeypatch):
    """Test that query rewriting is used in generate when enabled with conversation history."""
    from nvidia_rag.rag_server.main import NvidiaRAG

    fake_vdb = DummyVDB()
    # Set CONVERSATION_HISTORY > 0 so chat_history is not empty (query rewriting requires chat history)
    monkeypatch.setenv("CONVERSATION_HISTORY", "5")
    # Ensure multiturn simple retrieval is disabled (test relies on query rewriting)
    monkeypatch.setenv("MULTITURN_RETRIEVER_SIMPLE", "False")
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
        {"role": "user", "content": "How does it work?"},
    ]

    # Act: Calling generate() triggers retrieval before returning the stream
    await rag.generate(
        messages=messages,
        use_knowledge_base=True,
        collection_names=["test"],
        enable_query_rewriting=True,
        enable_reranker=False,
        enable_vlm_inference=False,
        filter_expr="",
    )

    # Assert: rewritten query is used for retrieval inside RAG flow
    assert fake_vdb.last_query == "REWRITTEN(How does it work?)"


@pytest.mark.asyncio
async def test_generate_uses_only_current_query_when_history_disabled(monkeypatch):
    """Test that when multiturn_retrieval_simple is False (default), only current query is used."""
    from nvidia_rag.rag_server.main import NvidiaRAG

    fake_vdb = DummyVDB()
    # Explicitly ensure multiturn simple retrieval is disabled
    monkeypatch.setenv("MULTITURN_RETRIEVER_SIMPLE", "False")
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
        {"role": "user", "content": "How does it work?"},
    ]

    await rag.generate(
        messages=messages,
        use_knowledge_base=True,
        collection_names=["test"],
        enable_query_rewriting=False,
        enable_reranker=False,
        enable_vlm_inference=False,
        filter_expr="",
    )

    # Assert: only current query is used (no history concatenation)
    assert fake_vdb.last_query == "How does it work?"


@pytest.mark.asyncio
async def test_generate_skips_query_rewriter_for_image_query(monkeypatch):
    """When messages contain multimodal content with image, query rewriting is skipped."""
    from nvidia_rag.rag_server.main import NvidiaRAG

    monkeypatch.setenv("CONVERSATION_HISTORY", "5")
    monkeypatch.setenv("ENABLE_REFLECTION", "false")
    monkeypatch.setenv("MULTITURN_RETRIEVER_SIMPLE", "False")

    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
            ],
        },
    ]

    async def _stream(*a, **k):
        yield "ok"

    with patch("nvidia_rag.rag_server.main.VLM") as mock_vlm_class:
        mock_vlm_instance = mock_vlm_class.return_value
        mock_vlm_instance.stream_with_messages = _stream

        await rag.generate(
            messages=messages,
            use_knowledge_base=True,
            collection_names=["test"],
            enable_query_rewriting=True,
            enable_reranker=False,
            enable_vlm_inference=True,
            filter_expr="",
        )

    # Assert: query rewriting skipped - last_query is text + image URL (no "REWRITTEN(...)")
    assert fake_vdb.last_query == "What is in this image? data:image/png;base64,x"
    assert "REWRITTEN" not in str(fake_vdb.last_query)
    # Assert: retrieval_image_langchain was used
    assert fake_vdb.last_retrieval_method == "image"


@pytest.mark.asyncio
async def test_generate_skips_reflection_for_image_query(monkeypatch):
    """Image generation must keep base64 data out of text reflection and reranking."""
    from nvidia_rag.rag_server.main import NvidiaRAG

    monkeypatch.setenv("ENABLE_REFLECTION", "true")
    monkeypatch.setenv("MULTITURN_RETRIEVER_SIMPLE", "False")

    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,x"},
                },
            ],
        },
    ]

    async def _stream(*args, **kwargs):
        yield "ok"

    with (
        patch(
            "nvidia_rag.rag_server.main.check_context_relevance",
            new_callable=AsyncMock,
            return_value=([], True),
        ) as mock_check_relevance,
        patch("nvidia_rag.rag_server.main.VLM") as mock_vlm_class,
    ):
        mock_vlm_class.return_value.stream_with_messages = _stream
        await rag.generate(
            messages=messages,
            use_knowledge_base=True,
            collection_names=["test"],
            enable_query_rewriting=True,
            enable_reranker=True,
            enable_vlm_inference=True,
            filter_expr="",
        )

    mock_check_relevance.assert_not_awaited()
    assert fake_vdb.last_retrieval_method == "image"


@pytest.mark.asyncio
async def test_generate_enables_diverse_image_page_candidates(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The public generate seam passes the feature flag to image retrieval."""
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "true")
    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    assert "base64" not in rag._safe_log_text(
        "這是什麼？ data:image/png;base64,current"
    )
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "這是什麼？"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,current"},
                },
            ],
        }
    ]

    async def stream(*args: object, **kwargs: object) -> AsyncIterator[str]:
        yield "ok"

    with patch("nvidia_rag.rag_server.main.VLM") as mock_vlm_class:
        mock_vlm_class.return_value.stream_with_messages = stream
        await rag.generate(
            messages=messages,
            use_knowledge_base=True,
            collection_names=["test"],
            enable_reranker=False,
            enable_vlm_inference=True,
        )

    assert fake_vdb.last_retrieval_method == "image"
    assert fake_vdb.last_diverse_pages is True
    assert "data:image" not in caplog.text
    assert "base64,current" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("assistant_summary", "expects_summary"),
    [
        ("先前圖片辨識為 J7EF Plus。", True),
        (None, False),
    ],
)
async def test_generate_isolates_previous_image_from_current_multimodal_turn(
    monkeypatch: pytest.MonkeyPatch,
    assistant_summary: str | None,
    expects_summary: bool,
) -> None:
    """Only the current image crosses retrieval and VLM service boundaries."""
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "true")
    monkeypatch.setenv("CONVERSATION_HISTORY", "5")
    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "這是什麼？"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,j7ef-previous"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,j7ef-previous-2"},
                },
            ],
        },
    ]
    if assistant_summary is not None:
        messages.append({"role": "assistant", "content": assistant_summary})
    else:
        messages[0]["content"] = messages[0]["content"][1:]
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "這又是什麼？"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,logo-current"},
                },
            ],
        }
    )
    vlm_call: dict[str, object] = {}

    async def stream(*args: object, **kwargs: object) -> AsyncIterator[str]:
        vlm_call.update(kwargs)
        yield "這是目前上傳的 Logo。"

    with patch("nvidia_rag.rag_server.main.VLM") as mock_vlm_class:
        mock_vlm_class.return_value.stream_with_messages = stream
        await rag.generate(
            messages=messages,
            use_knowledge_base=True,
            collection_names=["test"],
            enable_reranker=False,
            enable_vlm_inference=True,
        )

    assert fake_vdb.last_query == "這又是什麼？ data:image/png;base64,logo-current"
    vlm_messages = str(vlm_call["messages"])
    assert "j7ef-previous" not in vlm_messages
    assert ("先前圖片辨識為 J7EF Plus。" in vlm_messages) is expects_summary
    assert "logo-current" in vlm_messages
    assert {"role": "user", "content": []} not in vlm_call["messages"]


@pytest.mark.asyncio
async def test_generate_reserves_current_and_top_retrieved_images_after_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public generation sends current and top-page images despite image history."""
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "true")
    monkeypatch.setenv("CONVERSATION_HISTORY", "5")
    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)
    top_page = SimpleNamespace(
        metadata={
            "content_metadata": {
                "type": "image",
                "page_number": 9,
                "location": [0, 0, 1, 1],
            },
            "collection_name": "test",
            "source": {
                "source_name": "top.pdf",
                "source_location": "s3://bucket/top-page",
            },
        },
        page_content="",
    )
    monkeypatch.setattr(
        fake_vdb,
        "retrieval_image_langchain",
        lambda *args, **kwargs: [top_page],
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,old-one"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,old-two"},
                },
            ],
        },
        {"role": "assistant", "content": "Prior image summary."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "current question"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,current-image"},
                },
            ],
        },
    ]
    final_vlm_call: dict[str, object] = {}

    async def completion_stream() -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(delta=SimpleNamespace(content="ok", reasoning=None))
            ],
        )

    async def create_completion(**kwargs: object) -> AsyncIterator[SimpleNamespace]:
        final_vlm_call.update(kwargs)
        return completion_stream()

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_completion),
        )
    )
    object_store = SimpleNamespace(
        get_object_from_uri=lambda uri: b"top-page-image",
    )

    with (
        patch(
            "nvidia_rag.rag_server.vlm.VLM._create_async_client",
            return_value=client,
        ),
        patch(
            "nvidia_rag.rag_server.vlm.get_object_store_operator",
            return_value=object_store,
        ),
        patch(
            "nvidia_rag.rag_server.vlm.VLM._convert_image_url_to_png_b64",
            side_effect=lambda value: value.split(",", maxsplit=1)[-1],
        ),
    ):
        await rag.generate(
            messages=messages,
            use_knowledge_base=True,
            collection_names=["test"],
            enable_reranker=False,
            enable_vlm_inference=True,
            fetch_full_page_context=False,
            vlm_max_total_images=2,
        )

    final_messages = final_vlm_call["messages"]
    final_prompt = str(final_messages)
    assert "old-one" not in final_prompt
    assert "old-two" not in final_prompt
    assert "current-image" in final_prompt
    assert "dG9wLXBhZ2UtaW1hZ2U=" in final_prompt


@pytest.mark.asyncio
async def test_generate_preserves_legacy_image_history_when_accuracy_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-in isolation does not change feature-off VLM conversations."""
    monkeypatch.setenv("ENABLE_MULTIMODAL_ACCURACY", "false")
    monkeypatch.setenv("CONVERSATION_HISTORY", "5")
    rag = NvidiaRAG()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,legacy-previous"},
                },
            ],
        },
        {"role": "assistant", "content": "previous answer"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "second"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,current"},
                },
            ],
        },
    ]
    vlm_call: dict[str, object] = {}

    async def stream(*args: object, **kwargs: object) -> AsyncIterator[str]:
        vlm_call.update(kwargs)
        yield "ok"

    with patch("nvidia_rag.rag_server.main.VLM") as mock_vlm_class:
        mock_vlm_class.return_value.stream_with_messages = stream
        await rag.generate(
            messages=messages,
            use_knowledge_base=False,
            enable_vlm_inference=True,
        )

    vlm_messages = str(vlm_call["messages"])
    assert "legacy-previous" in vlm_messages
    assert "current" in vlm_messages


@pytest.mark.asyncio
async def test_generate_combines_history_when_multiturn_enabled(monkeypatch):
    """Test that when multiturn_retrieval_simple is True, history is concatenated."""
    from nvidia_rag.rag_server.main import NvidiaRAG

    # Enable multiturn retrieval via environment variable BEFORE creating NvidiaRAG instance
    # Also set CONVERSATION_HISTORY > 0 so chat_history is not empty
    monkeypatch.setenv("MULTITURN_RETRIEVER_SIMPLE", "True")
    monkeypatch.setenv("CONVERSATION_HISTORY", "5")

    fake_vdb = DummyVDB()
    rag = NvidiaRAG()
    monkeypatch.setattr(NvidiaRAG, "_prepare_vdb_op", lambda self, **kw: fake_vdb)

    messages = [
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "A retrieval-augmented framework."},
        {"role": "user", "content": "How does it work?"},
    ]

    await rag.generate(
        messages=messages,
        use_knowledge_base=True,
        collection_names=["test"],
        enable_query_rewriting=False,
        enable_reranker=False,
        enable_vlm_inference=False,
        filter_expr="",
    )

    # In _rag_chain when multiturn_retrieval_simple is enabled,
    # last previous user query is combined with current retriever_query
    # Expected concatenation: "What is RAG?. How does it work?"
    assert fake_vdb.last_query == "What is RAG?. How does it work?"
