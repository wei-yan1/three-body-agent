"""Stable intermediate models for text ingestion and auditability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TextSource:
    source: str
    text: str
    encoding: str
    byte_length: int
    normalized: bool = True
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedParagraph:
    index: int
    text: str
    start_offset: int
    end_offset: int
    line_start: int
    line_end: int
    kind: str = "narrative"
    speaker: str | None = None


@dataclass(frozen=True)
class ParsedSection:
    index: int
    title: str
    content: str
    kind: str
    book: str | None = None
    part: str | None = None
    chapter_number: int | None = None
    start_offset: int = 0
    end_offset: int = 0
    line_start: int = 1
    line_end: int = 1
    header_path: tuple[str, ...] = ()
    paragraphs: tuple[ParsedParagraph, ...] = ()

    @property
    def is_chapter(self) -> bool:
        return self.kind == "chapter"


@dataclass(frozen=True)
class ParsedDocument:
    source: TextSource
    sections: tuple[ParsedSection, ...]
    parser_name: str
    warnings: tuple[str, ...] = ()
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def chapters(self) -> tuple[ParsedSection, ...]:
        return tuple(section for section in self.sections if section.is_chapter)

    @property
    def paragraphs(self) -> tuple[ParsedParagraph, ...]:
        return tuple(paragraph for section in self.sections for paragraph in section.paragraphs)
