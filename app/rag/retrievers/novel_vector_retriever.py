"""Chroma + persistent BM25 retriever for timeline-scoped novel chunks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.rag.embeddings.embedding_model import create_dashscope_embeddings
from app.rag.indexing.bm25_index import PersistentBM25Index
from app.rag.retrievers.hybrid_fusion_retriever import HybridFusionRetriever

if TYPE_CHECKING:
    from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data/indexes/chroma"
NOVEL_COLLECTION = "three_body_novel_chunks"
TIMELINE_STAGE_ORDER = {f"T{i}": i for i in range(7)}


class NovelVectorRetriever:
    """Retrieve dense and independently persisted sparse novel evidence."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_CHROMA_DIR,
        collection_name: str = NOVEL_COLLECTION,
        embedding_function: Any | None = None,
        trace_id: str | None = None,
    ) -> None:
        from langchain_chroma import Chroma

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function or create_dashscope_embeddings(trace_id=trace_id),
            persist_directory=str(persist_directory),
        )
        sparse_index = PersistentBM25Index(
            Path(persist_directory).parent / "bm25" / f"{collection_name}.json"
        )
        self.hybrid_retriever = HybridFusionRetriever(
            self.vectorstore,
            dense_weight=0.65,
            sparse_weight=0.35,
            sparse_index=sparse_index,
            trace_id=trace_id,
        )

    def retrieve(
        self,
        query: str,
        timeline_stage: str,
        character: str | None = None,
        k: int = 5,
    ) -> list[Document]:
        stage_order = self._stage_order(timeline_stage)
        filter_query = {"stage_order": {"$lte": stage_order}}
        return self.hybrid_retriever.retrieve(query, k=k, filter=filter_query)

    @staticmethod
    def _stage_order(timeline_stage: str) -> int:
        try:
            return TIMELINE_STAGE_ORDER[timeline_stage]
        except KeyError as error:
            valid = ", ".join(TIMELINE_STAGE_ORDER)
            raise ValueError(f"Unsupported timeline_stage={timeline_stage!r}; valid stages: {valid}") from error
