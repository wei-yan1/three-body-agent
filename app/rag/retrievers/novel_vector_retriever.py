"""Chroma retriever for timeline-scoped novel chunks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.rag.embeddings.embedding_model import create_dashscope_embeddings
from app.rag.retrievers.hybrid_fusion_retriever import HybridFusionRetriever


if TYPE_CHECKING:
    from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data/indexes/chroma"
NOVEL_COLLECTION = "three_body_novel_chunks"
TIMELINE_STAGE_ORDER = {
    "T0": 0,
    "T1": 1,
    "T2": 2,
    "T3": 3,
    "T4": 4,
    "T5": 5,
    "T6": 6,
}


class NovelVectorRetriever:
    """Retrieve novel chunks visible from the current timeline stage.

    Novel memory is cumulative: an agent at T4 may retrieve T0-T4 novel chunks,
    but never T5 or T6 chunks.
    """

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_CHROMA_DIR,
        collection_name: str = NOVEL_COLLECTION,
        embedding_function: Any | None = None,
    ) -> None:
        from langchain_chroma import Chroma

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function or create_dashscope_embeddings(),
            persist_directory=str(persist_directory),
        )
        self.hybrid_retriever = HybridFusionRetriever(
            self.vectorstore,
            dense_weight=0.65,
            sparse_weight=0.35,
        )

    def retrieve(
        self,
        query: str,
        timeline_stage: str,
        character: str | None = None,
        k: int = 5,
    ) -> list[Document]:
        """Retrieve novel chunks from the current and earlier T stages."""
        stage_order = self._stage_order(timeline_stage)
        candidate_k = max(k * 4, 12)
        filter_query = {"stage_order": {"$lte": stage_order}}
        candidates = self.vectorstore.similarity_search(
            query,
            k=candidate_k,
            filter=filter_query,
        )
        sparse_candidates = candidates
        if character:
            character_marker = f"|{character}|"
            character_docs = [
                doc
                for doc in candidates
                if character_marker in str(doc.metadata.get("character_mentions", ""))
            ]
            seen = {doc.metadata.get("chunk_id") for doc in character_docs}
            fallback_docs = [
                doc for doc in candidates if doc.metadata.get("chunk_id") not in seen
            ]
            sparse_candidates = character_docs + fallback_docs

        return self.hybrid_retriever.retrieve(
            query,
            k=k,
            filter=filter_query,
            sparse_candidates=sparse_candidates,
        )

    @staticmethod
    def _stage_order(timeline_stage: str) -> int:
        try:
            return TIMELINE_STAGE_ORDER[timeline_stage]
        except KeyError as error:
            valid = ", ".join(TIMELINE_STAGE_ORDER)
            message = f"Unsupported timeline_stage={timeline_stage!r}; valid stages: {valid}"
            raise ValueError(message) from error
