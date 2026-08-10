# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical identity for grouping multimodal retrieval results by page."""

from typing import Any

IMAGE_PAGE_IDENTITY_FIELD = "image_page_identity"


def build_image_page_identity(source_name: Any, page_number: Any) -> str | None:
    """Return a collision-resistant source/page identity, or ``None`` if invalid."""
    if (
        not isinstance(source_name, str)
        or not source_name
        or isinstance(page_number, bool)
        or not isinstance(page_number, int)
    ):
        return None
    return f"{source_name}#page={page_number}"


def extract_nv_ingest_page_identity(record: dict[str, Any]) -> str | None:
    """Extract the canonical identity from an NV-Ingest record or metadata dict."""
    source = record.get("source")
    source_name = source.get("source_name") if isinstance(source, dict) else source
    content_metadata = record.get("content_metadata")
    page_number = (
        content_metadata.get("page_number")
        if isinstance(content_metadata, dict)
        else None
    )
    return build_image_page_identity(source_name, page_number)
