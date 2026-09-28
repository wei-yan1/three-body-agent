"""Small persistent BM25 index used as an independent sparse retrieval channel."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize_text(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _matches_filter(metadata: dict[str, Any], filter_query: dict[str, Any] | None) -> bool:
    if not filter_query:
        return True
    if "$and" in filter_query:
        return all(_matches_filter(metadata, item) for item in filter_query["$and"])
    for key, condition in filter_query.items():
        if not isinstance(condition, dict):
            if metadata.get(key) != condition:
                return False
            continue
        if "$eq" in condition and metadata.get(key) != condition["$eq"]:
            return False
        if "$lte" in condition:
            try:
                if float(metadata.get(key, 0)) > float(condition["$lte"]):
                    return False
            except (TypeError, ValueError):
                return False
        if "$in" in condition and metadata.get(key) not in condition["$in"]:
            return False
    return True


class PersistentBM25Index:
    """JSON-backed sparse index; writes are atomic and updates are idempotent."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.records = payload.get("records", {})

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(
            json.dumps({"schema_version": 1, "records": self.records}, ensure_ascii=False),
            encoding="utf-8",
        )
        temp.replace(self.path)

    def replace_documents(self, documents: list[Document]) -> None:
        self.records = {}
        for document in documents:
            doc_id = str(document.metadata["chunk_id"])
            self.records[doc_id] = {
                "text": document.page_content,
                "metadata": document.metadata,
                "tokens": tokenize_text(document.page_content),
            }
        self._save()

    def upsert_documents(self, documents: list[Document]) -> None:
        for document in documents:
            doc_id = str(document.metadata["chunk_id"])
            self.records[doc_id] = {
                "text": document.page_content,
                "metadata": document.metadata,
                "tokens": tokenize_text(document.page_content),
            }
        self._save()

    def delete_ids(self, ids: list[str]) -> None:
        changed = False
        for doc_id in ids:
            changed = self.records.pop(str(doc_id), None) is not None or changed
        if changed:
            self._save()

    def delete_source(self, source_id: str) -> None:
        ids = [doc_id for doc_id, item in self.records.items() if item["metadata"].get("source_id") == source_id]
        self.delete_ids(ids)

    def search(
        self,
        query: str,
        *,
        k: int,
        filter_query: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []
        eligible = {
            doc_id: item for doc_id, item in self.records.items()
            if _matches_filter(item["metadata"], filter_query)
        }
        if not eligible:
            return []
        total_docs = len(eligible)
        df = Counter()
        for item in eligible.values():
            df.update(set(item["tokens"]))
        avg_len = sum(len(item["tokens"]) for item in eligible.values()) / total_docs
        scored: list[tuple[Document, float]] = []
        for item in eligible.values():
            tokens = item["tokens"]
            counts = Counter(tokens)
            score = 0.0
            for term in query_tokens:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                term_df = df.get(term, 0)
                idf = math.log(1 + (total_docs - term_df + 0.5) / (term_df + 0.5))
                denominator = tf + 1.5 * (1 - 0.75 + 0.75 * len(tokens) / (avg_len or 1))
                score += idf * (tf * 2.5) / denominator
            if score > 0:
                scored.append((Document(page_content=item["text"], metadata=dict(item["metadata"])), score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


