"""Chroma retriever for temporal persona chunks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.rag.embeddings.embedding_model import create_dashscope_embeddings
from app.rag.retrievers.hybrid_fusion_retriever import HybridFusionRetriever


if TYPE_CHECKING:
    from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data/indexes/chroma"
PERSONA_COLLECTION = "three_body_persona_profiles"


class PersonaVectorRetriever:
    """Retrieve persona chunks with strict character and timeline filters."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_CHROMA_DIR,
        collection_name: str = PERSONA_COLLECTION,
        embedding_function: Any | None = None,
    ) -> None:
        from langchain_chroma import Chroma

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function or create_dashscope_embeddings(),
            persist_directory=str(persist_directory),
        )
        self.hybrid_retriever = HybridFusionRetriever(self.vectorstore)

    def retrieve(
        self,
        query: str,
        character: str,
        timeline_stage: str,
        k: int = 5,
    ) -> list[Document]:
        """Retrieve documents for exactly one character and timeline stage."""
        filter_query = {
            "$and": [
                {"character": {"$eq": character}},
                {"timeline_stage": {"$eq": timeline_stage}},
            ]
        }
        sparse_candidates = self.vectorstore.similarity_search(
            query,
            k=max(k * 8, 20),
            filter=filter_query,
        )
        return self.hybrid_retriever.retrieve(
            query,
            k=k,
            filter=filter_query,
            sparse_candidates=sparse_candidates,
        )
