"""Novel text parser with encoding fallback, structure detection and diagnostics."""

from __future__ import annotations

import re
from pathlib import Path

from .models import ParsedDocument, ParsedParagraph, ParsedSection, TextSource
from .normalize import normalize_text

_BOOK_RE = re.compile(r"^(?P<title>三体\s*(?:I{1,3}|[123]|II[·.]黑暗森林|III[·.]死神永生))\s*$", re.I)
_PART_RE = re.compile(r"^(?P<title>(?:上部|中部|下部|第[一二三四五六七八九十百零〇0-9]+部))\s*$")
_PROLOGUE_RE = re.compile(r"^(?P<title>序\s*章|序言|前言|引子|楔子)\s*$")
_CHAPTER_RE = re.compile(
    r"^(?P<title>(?:第[一二三四五六七八九十百千万零〇0-9]+章(?:\s*[-.:：]?\s*.*)?|(?:chapter|chap\.)\s*[0-9一二三四五六七八九十百千万]+(?:\s*[-.:：]?\s*.*)?))$",
    re.I,
)
_VOLUME_RE = re.compile(r"^(?P<title>(?:卷\s*[0-9一二三四五六七八九十百千万]+|book\s+[0-9a-z]+))$", re.I)
_SPEAKER_RE = re.compile(r"^([^“”\"「」『』：:]{1,24})(?:说|道|问|答|喊|叫|低声道|轻声道)[：:]", re.S)


def _chapter_number(title: str) -> int | None:
    match = re.search(r"第([一二三四五六七八九十百千万零〇0-9]+)章", title)
    if not match:
        match = re.search(r"(?:chapter|chap\.)\s*([0-9]+)", title, re.I)
    if not match:
        return None
    raw = match.group(1)
    if raw.isdigit():
        return int(raw)
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    total = 0
    current = 0
    for char in raw:
        if char in digits:
            current = digits[char]
        elif char in units:
            total += (current or 1) * units[char]
            current = 0
    return total + current


def decode_text_file(path: str | Path) -> TextSource:
    source_path = Path(path)
    data = source_path.read_bytes()
    candidates: list[str] = []
    if data.startswith(b"\xef\xbb\xbf"):
        candidates.append("utf-8-sig")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        candidates.extend(["utf-16", "utf-16-le", "utf-16-be"])
    candidates.extend(["utf-8", "gb18030", "big5", "utf-16", "latin-1"])
    warnings: list[str] = []
    for encoding in dict.fromkeys(candidates):
        try:
            text = data.decode(encoding, errors="strict")
            if encoding == "latin-1":
                warnings.append("decoded_with_latin1_fallback")
            return TextSource(str(source_path), normalize_text(text), encoding, len(data), warnings=tuple(warnings))
        except UnicodeDecodeError:
            continue
    warnings.append("decoded_with_replacement_characters")
    return TextSource(
        str(source_path),
        normalize_text(data.decode("utf-8", errors="replace")),
        "utf-8-replace",
        len(data),
        warnings=tuple(warnings),
    )


def _heading_kind(line: str) -> tuple[str, str] | None:
    for kind, pattern in (("book", _BOOK_RE), ("part", _PART_RE), ("prologue", _PROLOGUE_RE), ("chapter", _CHAPTER_RE), ("volume", _VOLUME_RE)):
        match = pattern.match(line)
        if match:
            return kind, match.group("title").strip()
    return None


def _paragraphs_for_section(content: str, section_start: int, section_line: int) -> tuple[ParsedParagraph, ...]:
    paragraphs: list[ParsedParagraph] = []
    for index, match in enumerate(re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|$)", content, re.S)):
        text = match.group(0).strip()
        if not text:
            continue
        start = section_start + match.start()
        end = section_start + match.end()
        speaker_match = _SPEAKER_RE.match(text)
        has_dialogue = bool(speaker_match or re.search(r"[\"“”‘’「」『』]", text))
        paragraphs.append(
            ParsedParagraph(
                index=index,
                text=text,
                start_offset=start,
                end_offset=end,
                line_start=section_line + content[: match.start()].count("\n"),
                line_end=section_line + content[: match.end()].count("\n"),
                kind="dialogue" if has_dialogue else "narrative",
                speaker=speaker_match.group(1).strip() if speaker_match else None,
            )
        )
    return tuple(paragraphs)


def _parse_quality_score(text: str, sections: list[ParsedSection]) -> float:
    if not text.strip():
        return 0.0
    score = 0.55
    if any(section.is_chapter for section in sections):
        score += 0.25
    if len(sections) > 1:
        score += 0.1
    if "�" in text:
        score -= 0.25
    if len(text) < 100:
        score -= 0.1
    return max(0.0, min(1.0, round(score, 3)))


class NovelTextParser:
    name = "novel-text-parser-v2"

    def supports(self, path: Path | None, text: str | None = None) -> bool:
        return path is None or path.suffix.lower() in {".txt", ".md", ".markdown"}

    def parse(self, path: Path, *, text: str | None = None) -> ParsedDocument:
        source = decode_text_file(path) if text is None else TextSource(str(path), normalize_text(text), "provided", len(text.encode("utf-8")))
        return self.parse_text(source)

    def parse_text(self, source: TextSource) -> ParsedDocument:
        lines = source.text.splitlines(keepends=True)
        sections: list[ParsedSection] = []
        current_title, current_kind = "front_matter", "front_matter"
        current_book: str | None = None
        current_part: str | None = None
        current_chapter: int | None = None
        current_lines: list[str] = []
        current_start, current_line, offset = 0, 1, 0
        warnings = list(source.warnings)

        def flush(end_offset: int, end_line: int) -> None:
            nonlocal current_lines
            content = "".join(current_lines).strip()
            if not content and current_kind == "front_matter":
                current_lines = []
                return
            section_index = len(sections) + 1
            header_path = tuple(item for item in (current_book, current_part, current_title) if item and item != "front_matter")
            sections.append(
                ParsedSection(
                    index=section_index,
                    title=current_title,
                    content=content,
                    kind=current_kind,
                    book=current_book,
                    part=current_part,
                    chapter_number=current_chapter,
                    start_offset=current_start,
                    end_offset=end_offset,
                    line_start=current_line,
                    line_end=end_line,
                    header_path=header_path,
                    paragraphs=_paragraphs_for_section(content, current_start, current_line),
                )
            )
            current_lines = []

        for line_number, raw_line in enumerate(lines, start=1):
            heading = _heading_kind(raw_line.strip()) if raw_line.strip() else None
            if heading:
                flush(offset, line_number - 1)
                current_kind, current_title = heading
                if current_kind == "book":
                    current_book, current_part, current_chapter = current_title, None, None
                elif current_kind == "part":
                    current_part, current_chapter = current_title, None
                elif current_kind == "chapter":
                    current_chapter = _chapter_number(current_title)
                current_start, current_line = offset + len(raw_line), line_number + 1
            else:
                current_lines.append(raw_line)
            offset += len(raw_line)
        flush(offset, len(lines))

        if not any(section.is_chapter for section in sections) and sections:
            warnings.append("no_explicit_chapter_heading_found")
        stats = {
            "section_count": len(sections),
            "chapter_count": sum(section.is_chapter for section in sections),
            "paragraph_count": sum(len(section.paragraphs) for section in sections),
            "dialogue_paragraph_count": sum(sum(p.kind == "dialogue" for p in section.paragraphs) for section in sections),
            "line_count": len(lines),
            "quality_score": _parse_quality_score(source.text, sections),
        }
        return ParsedDocument(source, tuple(sections), self.name, tuple(warnings), stats)


def parse_novel_file(path: str | Path) -> ParsedDocument:
    return NovelTextParser().parse(Path(path))


def parse_novel_text(text: str, *, source: str = "<memory>") -> ParsedDocument:
    source_model = TextSource(source, normalize_text(text), "provided", len(text.encode("utf-8")))
    return NovelTextParser().parse_text(source_model)

