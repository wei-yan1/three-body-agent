"""Text-only document ingestion primitives inspired by WeKnora DocReader."""

from .models import ParsedDocument, ParsedParagraph, ParsedSection, TextSource
from .novel_parser import NovelTextParser, parse_novel_file, parse_novel_text
from .pipeline import build_text_documents, build_text_documents_from_parsed
from .registry import TextParserRegistry, default_text_parser_registry
from .splitter import DocReaderTextSplitter, SplitterConfig, TextChunk

__all__ = [
    "DocReaderTextSplitter",
    "NovelTextParser",
    "ParsedDocument",
    "ParsedParagraph",
    "ParsedSection",
    "SplitterConfig",
    "TextChunk",
    "TextParserRegistry",
    "TextSource",
    "build_text_documents",
    "build_text_documents_from_parsed",
    "default_text_parser_registry",
    "parse_novel_file",
    "parse_novel_text",
]
