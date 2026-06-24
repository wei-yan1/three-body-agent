"""Embedding model factories."""

from __future__ import annotations

import os
from typing import Any


DEFAULT_DASHSCOPE_DOCUMENT_EMBEDDING_MODEL = "text-embedding-v2"
DEFAULT_DASHSCOPE_QUERY_EMBEDDING_MODEL = "text-embedding-v2"
MULTIMODAL_EMBEDDING_MODELS = {"qwen3-vl-embedding", "qwen2.5-vl-embedding"}


class DashScopeMultiModalTextEmbeddings:
    """LangChain-compatible wrapper for DashScope multimodal text embeddings."""

    def __init__(
        self,
        model: str = "qwen3-vl-embedding",
        dashscope_api_key: str | None = None,
        dimension: int | None = None,
        batch_size: int = 1,
    ) -> None:
        self.model = model
        self.dashscope_api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY")
        self.dimension = dimension
        self.batch_size = batch_size
        if not self.dashscope_api_key:
            message = "DASHSCOPE_API_KEY is required to build DashScope embeddings."
            raise RuntimeError(message)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> list[float]:
        import dashscope

        item = dashscope.MultiModalEmbeddingItemText(text=text, factor=1.0)
        response = dashscope.MultiModalEmbedding.call(
            model=self.model,
            input=[item],
            api_key=self.dashscope_api_key,
            dimension=self.dimension,
            output_type="dense",
            enable_fusion=True,
            auto_truncation=True,
        )
        if response.status_code != 200:
            message = (
                f"DashScope multimodal embedding failed: "
                f"code={response.code}, message={response.message}"
            )
            raise RuntimeError(message)
        return _extract_multimodal_embedding(response.output)


def _extract_multimodal_embedding(output: Any) -> list[float]:
    """Extract a dense vector from DashScope multimodal embedding output."""
    if isinstance(output, dict):
        if isinstance(output.get("embedding"), list):
            return output["embedding"]
        if isinstance(output.get("embeddings"), list) and output["embeddings"]:
            first = output["embeddings"][0]
            if isinstance(first, dict) and isinstance(first.get("embedding"), list):
                return first["embedding"]
            if isinstance(first, list):
                return first

    embeddings = getattr(output, "embeddings", None)
    if embeddings:
        first = embeddings[0]
        if isinstance(first, dict) and isinstance(first.get("embedding"), list):
            return first["embedding"]
        embedding = getattr(first, "embedding", None)
        if isinstance(embedding, list):
            return embedding

    embedding = getattr(output, "embedding", None)
    if isinstance(embedding, list):
        return embedding

    message = f"Cannot extract embedding from DashScope output: {output!r}"
    raise RuntimeError(message)


def create_dashscope_embeddings(
    model: str | None = None,
    dashscope_api_key: str | None = None,
):
    """Create DashScope/Bailian embeddings for Chroma."""
    api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        message = "DASHSCOPE_API_KEY is required to build DashScope embeddings."
        raise RuntimeError(message)
    selected_model = (
        model
        or os.getenv("DASHSCOPE_QUERY_EMBEDDING_MODEL")
        or os.getenv("DASHSCOPE_EMBEDDING_MODEL")
        or DEFAULT_DASHSCOPE_QUERY_EMBEDDING_MODEL
    )

    if selected_model in MULTIMODAL_EMBEDDING_MODELS:
        return DashScopeMultiModalTextEmbeddings(
            model=selected_model,
            dashscope_api_key=api_key,
        )

    try:
        from langchain_community.embeddings import DashScopeEmbeddings
    except ImportError as error:
        message = (
            "langchain-community is required for DashScope embeddings. "
            "Install project dependencies before building Chroma indexes."
        )
        raise RuntimeError(message) from error

    return DashScopeEmbeddings(
        model=selected_model,
        dashscope_api_key=api_key,
    )


def get_dashscope_document_embedding_model() -> str:
    """Return the offline document embedding model used for batch indexing."""
    return (
        os.getenv("DASHSCOPE_DOCUMENT_EMBEDDING_MODEL")
        or DEFAULT_DASHSCOPE_DOCUMENT_EMBEDDING_MODEL
    )
