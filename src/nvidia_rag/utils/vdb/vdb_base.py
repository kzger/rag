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
This module contains the implementation of the VDBRag class,
which provides an abstract base class for vector database operations in RAG applications.

This is a pure abstract base class with NO nv_ingest dependencies, allowing the RAG server
to operate independently without requiring nv-ingest-client packages.

Collection Management:
1. create_collection: Create a new collection with specified dimensions and type
2. check_collection_exists: Check if the specified collection exists
3. get_collection: Retrieve all collections with their metadata schemas
4. delete_collections: Delete multiple collections and their associated metadata

Document Management:
5. get_documents: Retrieve all unique documents from the specified collection
6. delete_documents: Remove documents matching the specified source values

Metadata Schema Management:
7. create_metadata_schema_collection: Initialize the metadata schema storage collection
8. add_metadata_schema: Store metadata schema configuration for the collection
9. get_metadata_schema: Retrieve the metadata schema for the specified collection

Retrieval Operations:
10. retrieval_langchain: Perform semantic search and return top-k relevant documents
11. retrieve_chunks_by_filter: Retrieve chunks by metadata filter (source, page_numbers)
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


class VDBRag(ABC):
    """
    VDBRag is a pure abstract base class for vector database operations in RAG applications.

    This class defines the interface for VDB operations without any nv_ingest dependencies,
    allowing the RAG server to work independently. For ingestion operations that require
    nv_ingest support, use VDBRagIngest from vdb_ingest_base.py instead.
    """

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Get the collection name."""
        pass

    @abstractmethod
    async def check_health(self) -> dict[str, Any]:
        """Check the health of the VDB."""
        pass

    # ----------------------------------------------------------------------------------------------
    # Abstract methods for the VDBRag class for ingestion
    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        dimension: int = 2048,
        collection_type: str = "text",
    ) -> None:
        """Create a new collection with specified dimensions and type."""
        pass

    @abstractmethod
    def check_collection_exists(
        self,
        collection_name: str,
    ) -> bool:
        """Check if the specified collection exists."""
        pass

    @abstractmethod
    def get_collection(self) -> list[dict[str, Any]]:
        """Retrieve all collections with their metadata schemas."""
        pass

    @abstractmethod
    def delete_collections(
        self,
        collection_names: list[str],
    ) -> None:
        """Delete multiple collections and their associated metadata."""
        pass

    @abstractmethod
    def get_documents(
        self,
        collection_name: str,
        *,
        force_get_metadata: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve all unique documents from the specified collection.

        Args:
            collection_name: Name of the collection.
            force_get_metadata: When True, run the full metadata scan even when
                the document count is above the fast-path threshold (otherwise
                per-document metadata may be omitted for large collections).
        """
        pass

    @abstractmethod
    def delete_documents(
        self,
        collection_name: str,
        source_values: list[str],
        result_dict: dict[str, list[str]] | None = None,
    ) -> bool:
        """Remove documents matching the specified source values.

        Args:
            collection_name: Name of the collection to delete from
            source_values: List of source values to match for deletion
            result_dict: Optional dict to populate with deletion results.
                        Should contain "deleted" and "not_found" lists.
        """
        pass

    @abstractmethod
    def create_metadata_schema_collection(
        self,
    ) -> None:
        """Initialize the metadata schema storage collection."""
        pass

    @abstractmethod
    def add_metadata_schema(
        self,
        collection_name: str,
        metadata_schema: list[dict[str, Any]],
    ) -> None:
        """Store metadata schema configuration for the collection."""
        pass

    @abstractmethod
    def get_metadata_schema(
        self,
        collection_name: str,
    ) -> list[dict[str, Any]]:
        """Retrieve the metadata schema for the specified collection."""
        pass

    # ----------------------------------------------------------------------------------------------
    # Methods for document info management
    def create_document_info_collection(
        self,
    ) -> None:
        """Create a document info collection."""
        pass

    def add_document_info(
        self,
        info_type: str,
        collection_name: str,
        document_name: str,
        info_value: dict[str, Any],
    ) -> None:
        """Add document info to a collection."""
        pass

    def get_document_info(
        self,
        info_type: str,
        collection_name: str,
        document_name: str,
    ) -> dict[str, Any]:
        """Get document info from a collection."""
        return {}

    @abstractmethod
    def get_catalog_metadata(
        self,
        collection_name: str,
    ) -> dict[str, Any]:
        """Get catalog metadata for a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary containing catalog metadata fields
        """
        pass

    @abstractmethod
    def update_catalog_metadata(
        self,
        collection_name: str,
        updates: dict[str, Any],
    ) -> None:
        """Update catalog metadata for a collection.

        Args:
            collection_name: Name of the collection
            updates: Dictionary of fields to update
        """
        pass

    @abstractmethod
    def get_document_catalog_metadata(
        self,
        collection_name: str,
        document_name: str,
    ) -> dict[str, Any]:
        """Get catalog metadata for a document.

        Args:
            collection_name: Name of the collection
            document_name: Name of the document

        Returns:
            Dictionary containing document catalog metadata
        """
        pass

    @abstractmethod
    def update_document_catalog_metadata(
        self,
        collection_name: str,
        document_name: str,
        updates: dict[str, Any],
    ) -> None:
        """Update catalog metadata for a document.

        Args:
            collection_name: Name of the collection
            document_name: Name of the document
            updates: Dictionary of fields to update
        """
        pass

    # ----------------------------------------------------------------------------------------------
    # Abstract methods for the VDBRag class for retrieval
    @abstractmethod
    def get_langchain_vectorstore(
        self,
        collection_name: str,
    ) -> VectorStore:
        """Get the vectorstore for a collection."""
        pass

    @abstractmethod
    def retrieval_langchain(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        filter_expr: str | list[dict[str, Any]] = "",
    ) -> list[dict[str, Any]]:
        """Perform semantic search and return top-k relevant documents."""
        pass

    def retrieve_chunks_by_filter(
        self,
        collection_name: str,
        source_name: str,
        page_numbers: list[int],
        limit: int = 1000,
    ) -> list[Document]:
        """Retrieve ALL chunks matching (source, page_numbers) via filter-only query.

        No semantic search - used for page context expansion when
        fetch_full_page_context is enabled.

        Args:
            collection_name: Name of the collection to query
            source_name: Source identifier (e.g., file path) to match
            page_numbers: List of page numbers to fetch
            limit: Maximum number of chunks to return (default 1000)

        Returns:
            List of LangChain Document objects with page_content and metadata
        """
        raise NotImplementedError(
            "retrieve_chunks_by_filter is not implemented for this vector store"
        )

    def retrieval_image_langchain(
        self,
        query: str,
        collection_name: str,
        vectorstore: VectorStore | None = None,
        top_k: int = 10,
        reranker_top_k: int | None = None,
        diverse_pages: bool = False,
    ) -> list[Document]:
        """Retrieve documents from a collection using an image query.

        Embeds the image, finds the most similar document page, then returns
        all chunks from that page for multimodal context.

        Args:
            query: The image query (base64 encoded)
            collection_name: Name of the collection to search
            vectorstore: Optional pre-initialized vectorstore
            top_k: Number of results for initial similarity search (VDB top_k)
            reranker_top_k: Final number of documents to return (smaller value).
                           If None, defaults to top_k.
            diverse_pages: Return aggregated candidates from multiple unique pages.

        Returns:
            List of LangChain Document objects with page_content and metadata
        """
        raise NotImplementedError(
            "retrieval_image_langchain is not implemented for this vector store"
        )

    def _image_page_identity(self, document: Document) -> tuple[str, int] | None:
        """Return the source/page identity required by diverse image retrieval."""
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
        ):
            return None
        return source_name, page_number

    def _aggregate_image_page_candidates(
        self,
        scored: list[tuple[Document, Any]],
        collection_name: str,
        max_pages: int,
    ) -> list[Document]:
        """Build ranked, chunk-preserving candidates from image-search hits."""
        candidates: list[Document] = []
        seen_pages: set[tuple[str, int]] = set()
        for rank, (document, score) in enumerate(scored, start=1):
            identity = self._image_page_identity(document)
            if identity is None:
                logger.warning(
                    "Skipping image retrieval result at rank %d: required source/page metadata is missing",
                    rank,
                )
                continue
            try:
                raw_score = float(score)
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping image retrieval result at rank %d: raw similarity score is missing or invalid",
                    rank,
                )
                continue
            if isinstance(score, bool) or not math.isfinite(raw_score):
                logger.warning(
                    "Skipping image retrieval result at rank %d: raw similarity score is missing or invalid",
                    rank,
                )
                continue
            if identity in seen_pages:
                continue
            source_name, page_number = identity
            chunks = self.retrieve_chunks_by_filter(
                collection_name=collection_name,
                source_name=source_name,
                page_numbers=[page_number],
                limit=1000,
            )
            if not chunks:
                chunks = [document]
            valid_chunks: list[Document] = []
            for chunk in chunks:
                chunk_metadata = chunk.metadata
                content_metadata = chunk_metadata.get("content_metadata")
                if content_metadata is not None and not isinstance(
                    content_metadata, dict
                ):
                    logger.warning(
                        "Skipping malformed expanded chunk for image retrieval result at rank %d",
                        rank,
                    )
                    continue
                valid_chunks.append(chunk)
            if not valid_chunks:
                logger.warning(
                    "Skipping image retrieval result at rank %d: no chunks have valid metadata",
                    rank,
                )
                continue
            chunks = valid_chunks
            content = "\n\n".join(
                chunk.page_content for chunk in chunks if chunk.page_content
            )
            metadata = dict(chunks[0].metadata)
            metadata.setdefault("source", document.metadata.get("source", source_name))
            page_metadata = dict(metadata.get("content_metadata") or {})
            page_metadata["page_number"] = page_number
            metadata["content_metadata"] = page_metadata
            metadata["collection_name"] = collection_name
            metadata["image_retrieval"] = {
                "raw_score": raw_score,
                "rank": rank,
                "chunk_count": len(chunks),
                "chunks": [
                    {
                        "page_content": chunk.page_content,
                        "metadata": chunk.metadata,
                    }
                    for chunk in chunks
                ],
            }
            candidates.append(Document(page_content=content, metadata=metadata))
            seen_pages.add(identity)
            if len(candidates) >= max_pages:
                break
        return candidates
