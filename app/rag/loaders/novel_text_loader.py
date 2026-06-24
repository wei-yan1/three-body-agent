"""Load local novel text sources for RAG preprocessing."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document


DEFAULT_THREE_BODY_TEXT_PATH = Path("data/raw/three_body_characters/three_body.txt")


def load_text_file(file_path: str | Path) -> str:
    """Load a UTF-8 text file."""
    return Path(file_path).read_text(encoding="utf-8")


def load_three_body_text(file_path: str | Path = DEFAULT_THREE_BODY_TEXT_PATH) -> str:
    """Load the imported Three-Body novel text."""
    return load_text_file(file_path)


def load_three_body_text_document(
    file_path: str | Path = DEFAULT_THREE_BODY_TEXT_PATH,
) -> Document:
    """Load the imported Three-Body novel text as one LangChain document."""
    path = Path(file_path)
    return Document(
        page_content=load_three_body_text(path),
        metadata={
            "source_id": "three_body_txt_local",
            "source": str(path),
            "source_type": "novel_text",
            "work": "《三体》三部曲",
            "source_format": "txt",
        },
    )
