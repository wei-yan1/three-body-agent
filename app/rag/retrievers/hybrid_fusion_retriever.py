"""Hybrid retrieval with Min-Max weighted fusion and DashScope rerank."""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


from langchain_core.documents import Document


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass
class ScoredDocument:
    """Document with retriever scores."""

    document: Document
    doc_id: str
    route_scores: dict[str, float] = field(default_factory=dict)
    fused_score: float = 0.0
    rerank_score: float | None = None


def tokenize_text(text: str) -> list[str]:
    """Lightweight tokenizer for mixed Chinese and English retrieval."""
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Normalize score values into [0, 1]."""
    if not scores:
        return {}
    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {key: 0.5 for key in scores}
    return {
        key: (score - min_score) / (max_score - min_score)
        for key, score in scores.items()
    }


def weighted_sum_fusion(
    route_score_maps: list[dict[str, float]],
    weights: list[float],
) -> dict[str, float]:
    """Fuse multiple route scores with Min-Max normalization and weights."""
    if len(route_score_maps) != len(weights):
        raise ValueError("route_score_maps and weights must have the same length.")
    all_doc_ids: set[str] = set()
    for route_scores in route_score_maps:
        all_doc_ids.update(route_scores)

    normalized_routes = [min_max_normalize(scores) for scores in route_score_maps]
    fused_scores: dict[str, float] = {}
    for doc_id in all_doc_ids:
        fused_scores[doc_id] = sum(
            weight * normalized.get(doc_id, 0.0)
            for weight, normalized in zip(weights, normalized_routes, strict=True)
        )
    return fused_scores


class BM25Scorer:
    """Small in-memory BM25 scorer over a candidate document set."""

    def __init__(self, documents: Iterable[Document]) -> None:
        self.documents = list(documents)
        self.doc_tokens: list[list[str]] = [
            tokenize_text(document.page_content) for document in self.documents
        ]
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))
        self.avg_doc_len = (
            sum(len(tokens) for tokens in self.doc_tokens) / len(self.doc_tokens)
            if self.doc_tokens
            else 0.0
        )

    def score(self, query: str, k1: float = 1.5, b: float = 0.75) -> dict[str, float]:
        query_terms = tokenize_text(query)
        if not query_terms or not self.documents:
            return {}
        total_docs = len(self.documents)
        scores: dict[str, float] = {}
        for document, tokens in zip(self.documents, self.doc_tokens, strict=True):
            token_counts = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for term in query_terms:
                tf = token_counts.get(term, 0)
                if tf == 0:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denominator = tf + k1 * (1 - b + b * doc_len / (self.avg_doc_len or 1))
                score += idf * (tf * (k1 + 1)) / denominator
            if score > 0:
                doc_id = str(document.metadata.get("chunk_id"))
                scores[doc_id] = score
        return scores


class DashScopeReranker:
    """Cross-Encoder reranker backed by qwen3-rerank."""

    def __init__(
        self,
        model: str | None = None,
        dashscope_api_key: str | None = None,
    ) -> None:
        self.model = model or os.getenv("DASHSCOPE_RERANK_MODEL", "qwen3-rerank")
        self.dashscope_api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for DashScope rerank.")

    def rerank(
        self,
        query: str,
        scored_documents: list[ScoredDocument],
        top_k: int,
    ) -> list[ScoredDocument]:
        if not scored_documents:
            return []

        import dashscope

        documents = [item.document.page_content for item in scored_documents]
        response = dashscope.TextReRank.call(
            model=self.model,
            query=query,
            documents=documents,
            top_n=min(top_k, len(documents)),
            return_documents=False,
            api_key=self.dashscope_api_key,
        )
        if response.status_code != 200:
            message = (
                f"DashScope rerank failed: code={response.code}, "
                f"message={response.message}"
            )
            raise RuntimeError(message)

        reranked: list[ScoredDocument] = []
        for result in _extract_rerank_results(response.output):
            index = int(result["index"])
            item = scored_documents[index]
            item.rerank_score = float(result["relevance_score"])
            reranked.append(item)
        return reranked


def _extract_rerank_results(output: Any) -> list[dict[str, Any]]:
    if isinstance(output, dict) and isinstance(output.get("results"), list):
        return output["results"]
    results = getattr(output, "results", None)
    if isinstance(results, list):
        normalized: list[dict[str, Any]] = []
        for result in results:
            if isinstance(result, dict):
                normalized.append(result)
            else:
                normalized.append(
                    {
                        "index": getattr(result, "index"),
                        "relevance_score": getattr(result, "relevance_score"),
                    }
                )
        return normalized
    raise RuntimeError(f"Cannot extract rerank results from output: {output!r}")


class HybridFusionRetriever:
    """Dense + sparse retrieval, Min-Max weighted fusion, then rerank."""

    def __init__(
        self,
        vectorstore: Any,
        reranker: DashScopeReranker | None = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        candidate_multiplier: int = 6,
        rerank_enabled: bool = True,
    ) -> None:
        self.vectorstore = vectorstore
        self.reranker = reranker or DashScopeReranker()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.candidate_multiplier = candidate_multiplier
        self.rerank_enabled = rerank_enabled

    def retrieve(
        self,
        query: str,
        *,
        k: int,
        filter: dict[str, Any] | None = None,
        sparse_candidates: list[Document] | None = None,
    ) -> list[Document]:
        candidate_k = max(k * self.candidate_multiplier, k)
        dense_results = self.vectorstore.similarity_search_with_score(
            query,
            k=candidate_k,
            filter=filter,
        )

        raw_docs: dict[str, Document] = {}
        dense_scores: dict[str, float] = {}
        for document, distance in dense_results:
            doc_id = str(document.metadata.get("chunk_id"))
            raw_docs[doc_id] = document
            dense_scores[doc_id] = 1.0 / (1.0 + max(float(distance), 0.0))

        sparse_pool = sparse_candidates or list(raw_docs.values())
        bm25_scores = BM25Scorer(sparse_pool).score(query)
        for document in sparse_pool:
            doc_id = str(document.metadata.get("chunk_id"))
            raw_docs.setdefault(doc_id, document)

        fused_scores = weighted_sum_fusion(
            [dense_scores, bm25_scores],
            [self.dense_weight, self.sparse_weight],
        )
        scored_documents = [
            ScoredDocument(
                document=raw_docs[doc_id],
                doc_id=doc_id,
                route_scores={
                    "dense": dense_scores.get(doc_id, 0.0),
                    "bm25": bm25_scores.get(doc_id, 0.0),
                },
                fused_score=fused_score,
            )
            for doc_id, fused_score in fused_scores.items()
            if doc_id in raw_docs
        ]
        scored_documents.sort(key=lambda item: item.fused_score, reverse=True)
        rerank_candidates = scored_documents[:candidate_k]

        final_documents = (
            self.reranker.rerank(query, rerank_candidates, top_k=k)
            if self.rerank_enabled
            else rerank_candidates[:k]
        )

        for rank, item in enumerate(final_documents, start=1):
            item.document.metadata["fusion_score"] = item.fused_score
            item.document.metadata["dense_score"] = item.route_scores.get("dense", 0.0)
            item.document.metadata["bm25_score"] = item.route_scores.get("bm25", 0.0)
            item.document.metadata["rerank_score"] = item.rerank_score or 0.0
            item.document.metadata["rerank_rank"] = rank
        return [item.document for item in final_documents]
