"""Schemas for novel import jobs."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ImportStatus = Literal["queued", "processing", "completed", "failed", "cancelled"]
ImportStage = Literal["uploaded", "parsing", "embedding", "vector_indexing", "keyword_indexing", "chunks_ready", "ready", "failed", "cancelled"]


class NovelSourceOut(BaseModel):
    source_id: str
    file_name: str
    size_bytes: int
    sha256: str
    order: int
    status: str = "queued"
    chapter_count: int = 0
    paragraph_count: int = 0
    dialogue_paragraph_count: int = 0
    chunk_count: int = 0
    quality_score: float | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class NovelImportAccepted(BaseModel):
    import_id: str
    novel_name: str
    status: ImportStatus
    total_files: int
    total_bytes: int
    status_url: str


class NovelImportOut(BaseModel):
    import_id: str
    novel_name: str
    status: ImportStatus
    stage: ImportStage
    total_files: int
    processed_files: int
    total_bytes: int
    chunk_count: int
    index: dict[str, object] | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    sources: list[NovelSourceOut] = Field(default_factory=list)
    chunks_path: str | None = None
    report_path: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class NovelListItem(BaseModel):
    import_id: str
    novel_name: str
    status: ImportStatus
    stage: ImportStage
    total_files: int
    processed_files: int
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class NovelRenameRequest(BaseModel):
    novel_name: str = Field(min_length=1, max_length=255)

