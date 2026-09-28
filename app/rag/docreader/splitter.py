"""Structure-aware recursive splitter inspired by WeKnora's DocReader.

This module intentionally stays text-only. It keeps the useful DocReader ideas:
ordered separators, protected spans, header context, overlap, and source offsets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SplitterConfig:
    chunk_size: int = 1800
    chunk_overlap: int = 180
    separators: tuple[str, ...] = ("\n\n", "\n", "。", "！", "？", "；", "，", " ", "")
    include_header_context: bool = True
    protected_patterns: tuple[str, ...] = (
        r"(?s)```.*?```",
        r"\$\$.*?\$\$",
        r"!\[[^\]\n]{0,200}\]\([^\)\n]{1,500}\)",
        r"\[[^\]\n]{1,200}\]\([^\)\n]{1,500}\)",
    )

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")


@dataclass(frozen=True)
class TextChunk:
    text: str
    start: int
    end: int
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _protected_spans(text: str, patterns: tuple[str, ...]) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    for pattern in patterns:
        matches.extend((match.start(), match.end()) for match in re.finditer(pattern, text))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    result: list[tuple[int, int]] = []
    last_end = -1
    for start, end in matches:
        if start >= last_end:
            result.append((start, end))
            last_end = end
    return result


def _inside(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < position < end for start, end in spans)


def _relative_spans(spans: list[tuple[int, int]], base: int, end: int) -> list[tuple[int, int]]:
    return [(max(0, start - base), min(end - base, stop - base)) for start, stop in spans if stop > base and start < end]


class DocReaderTextSplitter:
    """Recursively split text while preserving structure and auditable offsets."""

    def __init__(self, config: SplitterConfig | None = None) -> None:
        self.config = config or SplitterConfig()

    def _split_recursive(
        self,
        text: str,
        base_offset: int,
        separators: tuple[str, ...],
        absolute_spans: list[tuple[int, int]],
    ) -> list[tuple[str, int, int]]:
        if not text:
            return []
        if len(text) <= self.config.chunk_size or not separators:
            return [(text, base_offset, base_offset + len(text))]

        separator = separators[0]
        relative_spans = _relative_spans(absolute_spans, base_offset, base_offset + len(text))
        if separator == "":
            pieces: list[tuple[str, int, int]] = []
            start = 0
            while start < len(text):
                end = min(len(text), start + self.config.chunk_size)
                # Never cut through a protected span. Extend to its end instead.
                for span_start, span_end in relative_spans:
                    if span_start < end < span_end and span_start >= start:
                        end = span_end
                        break
                pieces.append((text[start:end], base_offset + start, base_offset + end))
                start = end
            return pieces

        boundaries = [
            match.start() + len(separator)
            for match in re.finditer(re.escape(separator), text)
            if not _inside(match.start() + len(separator), relative_spans)
        ]
        if not boundaries:
            return self._split_recursive(text, base_offset, separators[1:], absolute_spans)

        pieces: list[tuple[str, int, int]] = []
        start = 0
        for boundary in boundaries + [len(text)]:
            if boundary <= start:
                continue
            piece = text[start:boundary]
            if len(piece) > self.config.chunk_size:
                pieces.extend(
                    self._split_recursive(
                        piece,
                        base_offset + start,
                        separators[1:],
                        absolute_spans,
                    )
                )
            else:
                pieces.append((piece, base_offset + start, base_offset + boundary))
            start = boundary
        return pieces

    def _safe_overlap_start(self, start: int, end: int, spans: list[tuple[int, int]]) -> int:
        overlap_start = max(start, end - self.config.chunk_overlap)
        for span_start, span_end in spans:
            if span_start < overlap_start < span_end:
                return span_start
        return overlap_start

    def split_text(self, text: str, *, metadata: dict[str, Any] | None = None) -> list[TextChunk]:
        if not text.strip():
            return []
        spans = _protected_spans(text, self.config.protected_patterns)
        atomic = self._split_recursive(text, 0, self.config.separators, spans)
        chunks: list[TextChunk] = []
        current = ""
        current_start = 0
        current_end = 0
        index = 0

        for piece, start, end in atomic:
            if not current:
                current, current_start, current_end = piece, start, end
                continue
            candidate = current + piece
            if len(candidate) <= self.config.chunk_size:
                current, current_end = candidate, end
                continue

            if current.strip():
                chunks.append(TextChunk(current.strip(), current_start, current_end, index, dict(metadata or {})))
                index += 1
            overlap_start = self._safe_overlap_start(current_start, current_end, spans)
            overlap = text[overlap_start:current_end]
            current = overlap + piece
            current_start = overlap_start
            current_end = end

            # A single protected block can exceed the configured budget. Keep it intact.
            if len(current) > self.config.chunk_size and len(piece) > self.config.chunk_size:
                chunks.append(TextChunk(current.strip(), current_start, current_end, index, dict(metadata or {})))
                index += 1
                current = ""

        if current.strip():
            chunks.append(TextChunk(current.strip(), current_start, current_end, index, dict(metadata or {})))
        return chunks

    def split_sections(self, sections: list[Any]) -> list[TextChunk]:
        """Split ParsedSection-like objects while carrying structural metadata."""
        result: list[TextChunk] = []
        for section in sections:
            header_path = " > ".join(section.header_path)
            prefix = f"{header_path}\n\n" if self.config.include_header_context and header_path else ""
            metadata = {
                "section_index": section.index,
                "section_title": section.title,
                "section_kind": section.kind,
                "book": section.book or "",
                "part": section.part or "",
                "chapter_number": section.chapter_number or "",
                "header_path": header_path,
                "source_start_offset": section.start_offset,
                "source_end_offset": section.end_offset,
                "source_line_start": section.line_start,
                "source_line_end": section.line_end,
                "header_prefix_length": len(prefix),
            }
            for chunk in self.split_text(prefix + section.content, metadata=metadata):
                adjusted = dict(chunk.metadata)
                adjusted["content_start_offset"] = max(0, chunk.start - len(prefix))
                adjusted["content_end_offset"] = max(0, chunk.end - len(prefix))
                adjusted["source_chunk_start_offset"] = section.start_offset + adjusted["content_start_offset"]
                adjusted["source_chunk_end_offset"] = section.start_offset + adjusted["content_end_offset"]
                result.append(TextChunk(chunk.text, chunk.start, chunk.end, chunk.index, adjusted))
        return result
