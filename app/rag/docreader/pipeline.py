"""Reusable text ingestion pipeline for uploaded novels."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .models import ParsedDocument
from .novel_parser import parse_novel_file
from .splitter import DocReaderTextSplitter, SplitterConfig


def build_text_documents_from_parsed(
    parsed: ParsedDocument,
    *,
    file_path: str | Path,
    source_id: str,
    source_order: int = 0,
    work: str = "",
    chunk_size: int = 1800,
    chunk_overlap: int = 180,
) -> list[Document]:
    path = Path(file_path)
    splitter = DocReaderTextSplitter(SplitterConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    chunks = splitter.split_sections(list(parsed.sections))
    documents: list[Document] = []
    for chunk in chunks:
        if not chunk.text.strip():
            continue
        content_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:16]
        section_index = int(chunk.metadata.get("section_index", 0) or 0)
        metadata: dict[str, Any] = {
            # chunk.index restarts for every parsed section. Include the
            # section and source offsets so repeated chapter openings cannot
            # collide when they contain identical text.
            "chunk_id": (
                f"{source_id}_{source_order:04d}_s{section_index:06d}_"
                f"{chunk.index:06d}_{chunk.start:09d}_{chunk.end:09d}_{content_hash}"
            ),
            "source_id": source_id,
            "source_order": source_order,
            "source": str(path),
            "source_file": path.name,
            "source_type": "novel_text",
            "source_format": path.suffix.lower().lstrip("."),
            "work": work,
            "parser_name": parsed.parser_name,
            "parser_encoding": parsed.source.encoding,
            "parser_warnings": "|".join(parsed.warnings),
            "parser_stats": parsed.stats,
            "chunk_kind": "novel_structural_chunk",
            "chunk_start_offset": chunk.start,
            "chunk_end_offset": chunk.end,
            **chunk.metadata,
        }
        documents.append(Document(page_content=chunk.text, metadata=metadata))
    return documents


def build_text_documents(
    file_path: str | Path,
    *,
    source_id: str,
    source_order: int = 0,
    work: str = "",
    chunk_size: int = 1800,
    chunk_overlap: int = 180,
) -> list[Document]:
    path = Path(file_path)
    return build_text_documents_from_parsed(
        parse_novel_file(path),
        file_path=path,
        source_id=source_id,
        source_order=source_order,
        work=work,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

