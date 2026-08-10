# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nvidia_rag.utils.vdb.image_page_identity import (
    build_image_page_identity,
    extract_nv_ingest_page_identity,
)


def test_build_image_page_identity_is_unambiguous_and_keeps_page_zero() -> None:
    assert build_image_page_identity("a:1", 0) == "a:1#page=0"
    assert build_image_page_identity("a", 10) == "a#page=10"


def test_extract_nv_ingest_page_identity_rejects_malformed_metadata() -> None:
    assert (
        extract_nv_ingest_page_identity(
            {
                "source": {"source_name": "/doc.pdf"},
                "content_metadata": {"page_number": 4},
            }
        )
        == "/doc.pdf#page=4"
    )
    assert extract_nv_ingest_page_identity({"source": "/doc.pdf"}) is None
    assert build_image_page_identity("/doc.pdf", True) is None
