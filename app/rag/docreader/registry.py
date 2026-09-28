"""Parser registry with deterministic fallback, mirroring DocReader's parser chain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import ParsedDocument


class TextParser(Protocol):
    name: str

    def supports(self, path: Path | None, text: str | None = None) -> bool: ...

    def parse(self, path: Path, *, text: str | None = None) -> ParsedDocument: ...


@dataclass
class TextParserRegistry:
    parsers: list[TextParser]

    def register(self, parser: TextParser, *, prepend: bool = False) -> None:
        if prepend:
            self.parsers.insert(0, parser)
        else:
            self.parsers.append(parser)

    def resolve(self, path: Path, *, text: str | None = None) -> TextParser:
        for parser in self.parsers:
            if parser.supports(path, text):
                return parser
        raise ValueError(f"No text parser registered for {path}")

    def parse(self, path: str | Path) -> ParsedDocument:
        source_path = Path(path)
        parser = self.resolve(source_path)
        return parser.parse(source_path)


def default_text_parser_registry() -> TextParserRegistry:
    from .novel_parser import NovelTextParser

    return TextParserRegistry([NovelTextParser()])
