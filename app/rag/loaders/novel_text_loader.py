"""Load and normalize local novel text sources for RAG preprocessing."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.rag.docreader.novel_parser import decode_text_file


DEFAULT_THREE_BODY_TEXT_PATH = Path("data/raw/three_body_characters/three_body.txt")


def load_text_file(file_path: str | Path) -> str:
    """Load a text source using DocReader-compatible encoding fallback."""
    return decode_text_file(file_path).text


def load_three_body_text(file_path: str | Path = DEFAULT_THREE_BODY_TEXT_PATH) -> str:
    """Load the imported Three-Body novel text."""
    return load_text_file(file_path)


def load_three_body_text_document(
    file_path: str | Path = DEFAULT_THREE_BODY_TEXT_PATH,
) -> Document:
    """Load the imported Three-Body novel text as one LangChain document."""
    path = Path(file_path)
    source = decode_text_file(path)
    return Document(
        page_content=source.text,
        metadata={
            "source_id": "three_body_txt_local",
            "source": str(path),
            "source_type": "novel_text",
            "work": "《三体》三部曲",
            "source_format": "txt",
            "encoding": source.encoding,
            "byte_length": source.byte_length,
            "decode_warnings": "|".join(source.warnings),
        },
    )
