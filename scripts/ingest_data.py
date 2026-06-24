"""Data ingestion entrypoint for RAG preprocessing."""

from app.rag.embeddings.text_splitter import split_documents_recursively, split_text_recursively
from app.rag.embeddings.structure_aware_chunker import (
    build_novel_structure_documents,
    build_persona_structure_documents,
)
from app.rag.loaders.novel_text_loader import (
    load_three_body_text,
    load_three_body_text_document,
)
from app.rag.loaders.persona_skill_loader import load_luoji_temporal_persona_skill_documents
from app.rag.loaders.persona_skill_loader import load_all_temporal_persona_skill_documents


__all__ = [
    "load_all_temporal_persona_skill_documents",
    "load_luoji_temporal_persona_skill_documents",
    "load_three_body_text",
    "load_three_body_text_document",
    "build_novel_structure_documents",
    "build_persona_structure_documents",
    "split_text_recursively",
    "split_documents_recursively",
]
