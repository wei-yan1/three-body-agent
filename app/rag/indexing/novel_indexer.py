"""Incremental embedding + Chroma + BM25 indexing for imported novels."""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langchain_core.documents import Document

from app.rag.embeddings.embedding_model import create_dashscope_embeddings
from app.rag.indexing.bm25_index import PersistentBM25Index


DEFAULT_BATCH_SIZE = 40
DEFAULT_MAX_WORKERS = 5
DEFAULT_MAX_RETRIES = 5
DEFAULT_MAX_EMBED_CHARS = 20_000


@dataclass(frozen=True)
class NovelIndexResult:
    collection_name: str
    vector_index_path: str
    bm25_index_path: str
    input_documents: int
    embedded_documents: int
    skipped_documents: int
    deleted_documents: int
    batch_count: int


class NovelIndexer:
    """Idempotent indexer with bounded embedding concurrency and retry backoff."""

    def __init__(
        self,
        *,
        persist_directory: str | Path = "data/indexes/chroma",
        collection_name: str = "storyrole_novel_chunks",
        bm25_directory: str | Path | None = None,
        embedding_function: Any | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_workers: int = DEFAULT_MAX_WORKERS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_embed_chars: int = DEFAULT_MAX_EMBED_CHARS,
        trace_id: str | None = None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.bm25_path = Path(bm25_directory or self.persist_directory.parent / "bm25") / f"{collection_name}.json"
        self.embedding_function = embedding_function or create_dashscope_embeddings(use_cache=True, trace_id=trace_id)
        self.batch_size = max(1, batch_size)
        self.max_workers = max(1, max_workers)
        self.max_retries = max(1, max_retries)
        self.max_embed_chars = max(1, max_embed_chars)
        self.trace_id = trace_id

    @staticmethod
    def _retry(operation: Callable[[], Any], *, max_retries: int, label: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                return operation()
            except Exception as error:  # noqa: BLE001
                last_error = error
                if attempt == max_retries - 1:
                    break
                delay = min(30.0, 2**attempt) + random.random() * 0.25
                time.sleep(delay)
        raise RuntimeError(f"{label} failed after {max_retries} attempts: {last_error}") from last_error

    def _load_documents(self, chunks_path: str | Path) -> list[Document]:
        documents: list[Document] = []
        first_seen: dict[str, int] = {}
        duplicates: list[tuple[str, int, int]] = []
        with Path(chunks_path).open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("text")
                metadata = record.get("metadata")
                if not isinstance(text, str) or not isinstance(metadata, dict):
                    raise ValueError(f"Invalid chunk record at {chunks_path}:{line_number}")
                if not text.strip():
                    continue
                chunk_id = str(metadata.get("chunk_id") or record.get("chunk_id") or "")
                if not chunk_id:
                    raise ValueError(f"Missing chunk_id at {chunks_path}:{line_number}")
                if chunk_id in first_seen:
                    duplicates.append((chunk_id, first_seen[chunk_id], line_number))
                else:
                    first_seen[chunk_id] = line_number
                documents.append(Document(page_content=text, metadata=metadata))
        if duplicates:
            examples = "; ".join(
                f"{chunk_id} (lines {first_line}, {duplicate_line})"
                for chunk_id, first_line, duplicate_line in duplicates[:5]
            )
            raise ValueError(
                f"Duplicate chunk_id values in {chunks_path}: "
                f"{len(duplicates)} duplicate records; {examples}"
            )
        return documents

    def _collection(self):
        import chromadb

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_directory))
        return client.get_or_create_collection(name=self.collection_name), client

    def _existing_ids(self, collection: Any) -> set[str]:
        result = collection.get(include=[])
        return {str(doc_id) for doc_id in result.get("ids", [])}

    def _source_ids(self, collection: Any, source_ids: set[str]) -> list[str]:
        if not source_ids:
            return []
        result = collection.get(include=["metadatas"])
        stale: list[str] = []
        for doc_id, metadata in zip(result.get("ids", []), result.get("metadatas", []), strict=False):
            if isinstance(metadata, dict) and metadata.get("source_id") in source_ids:
                stale.append(str(doc_id))
        return stale

    def _sanitize_text(self, text: str) -> str:
        if len(text) <= self.max_embed_chars:
            return text
        return text[: self.max_embed_chars] + "\n[embedding input truncated]"

    @staticmethod
    def _chroma_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Convert nested JSON metadata into Chroma's scalar metadata shape."""
        normalized: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                normalized[str(key)] = value
            elif isinstance(value, (dict, list, tuple)):
                normalized[str(key)] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            else:
                normalized[str(key)] = str(value)
        return normalized

    def _embed_batch(self, documents: list[Document]) -> list[list[float]]:
        texts = [self._sanitize_text(document.page_content) for document in documents]
        embeddings = self._retry(
            lambda: self.embedding_function.embed_documents(texts),
            max_retries=self.max_retries,
            label="embedding batch",
        )
        if len(embeddings) != len(documents):
            raise ValueError(f"Embedding count mismatch: expected {len(documents)}, got {len(embeddings)}")
        return embeddings

    def index_chunks(self, chunks_path: str | Path) -> NovelIndexResult:
        documents = self._load_documents(chunks_path)
        collection, _client = self._collection()
        existing_ids = self._existing_ids(collection)
        incoming_ids = {str(document.metadata["chunk_id"]) for document in documents}
        source_ids = {str(document.metadata.get("source_id")) for document in documents}
        stale_ids = [doc_id for doc_id in self._source_ids(collection, source_ids) if doc_id not in incoming_ids]
        if stale_ids:
            self._retry(lambda: collection.delete(ids=stale_ids), max_retries=self.max_retries, label="vector stale delete")

        new_documents = [document for document in documents if str(document.metadata["chunk_id"]) not in existing_ids]
        batches = [new_documents[i : i + self.batch_size] for i in range(0, len(new_documents), self.batch_size)]
        embedded_batches: dict[int, list[list[float]]] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(batches)))) as executor:
            futures = {executor.submit(self._embed_batch, batch): index for index, batch in enumerate(batches)}
            for future in as_completed(futures):
                embedded_batches[futures[future]] = future.result()

        for index, batch in enumerate(batches):
            vectors = embedded_batches[index]
            self._retry(
                lambda batch=batch, vectors=vectors: collection.upsert(
                    ids=[str(document.metadata["chunk_id"]) for document in batch],
                    documents=[document.page_content for document in batch],
                    metadatas=[self._chroma_metadata(document.metadata) for document in batch],
                    embeddings=vectors,
                ),
                max_retries=self.max_retries,
                label="vector upsert",
            )

        bm25 = PersistentBM25Index(self.bm25_path)
        if stale_ids:
            bm25.delete_ids(stale_ids)
        bm25.upsert_documents(documents)
        return NovelIndexResult(
            collection_name=self.collection_name,
            vector_index_path=str(self.persist_directory),
            bm25_index_path=str(self.bm25_path),
            input_documents=len(documents),
            embedded_documents=len(new_documents),
            skipped_documents=len(documents) - len(new_documents),
            deleted_documents=len(stale_ids),
            batch_count=len(batches),
        )

