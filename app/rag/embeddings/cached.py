"""Persistent, provider-agnostic embedding cache."""
from __future__ import annotations

import json
import time
import unicodedata
from datetime import UTC, datetime
from typing import Any

from app.observability import model_cost, record_model_invocation, sha256_text
from app.storage.postgres.client import postgres_connection


class CachedEmbeddings:
    """LangChain embedding adapter which reuses vectors by model and text hash."""

    def __init__(self, backend: Any, *, provider: str = "dashscope", model: str | None = None, model_version: str = "", trace_id: str | None = None) -> None:
        self.backend = backend
        self.provider = provider
        self.model = model or getattr(backend, "model", "unknown")
        self.model_version = model_version
        self.trace_id = trace_id

    def _key(self, text: str) -> str:
        return sha256_text(unicodedata.normalize("NFKC", text).strip())

    def _lookup(self, text_hash: str) -> list[float] | None:
        try:
            from app.storage.postgres.schema import init_postgres_schema
            init_postgres_schema()
            with postgres_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT vector FROM embedding_cache WHERE provider=%s AND model=%s "
                    "AND model_version=%s AND text_hash=%s ORDER BY dimension LIMIT 1",
                    (self.provider, self.model, self.model_version, text_hash),
                )
                row = cur.fetchone()
            if not row:
                return None
            vector = row["vector"] if isinstance(row, dict) else row[0]
            return [float(item) for item in (json.loads(vector) if isinstance(vector, str) else vector)]
        except Exception:
            return None

    def _store(self, text_hash: str, vector: list[float]) -> None:
        try:
            from app.storage.postgres.schema import init_postgres_schema
            init_postgres_schema()
            with postgres_connection() as conn, conn.cursor() as cur:
                cur.execute("""INSERT INTO embedding_cache(provider,model,model_version,text_hash,dimension,vector)
                    VALUES (%s,%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT (provider,model,model_version,text_hash,dimension) DO NOTHING""", (self.provider, self.model, self.model_version, text_hash, len(vector), json.dumps(vector)))
        except Exception:
            return

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        started = time.perf_counter()
        started_at = datetime.now(UTC)
        keys = [self._key(text) for text in texts]
        result: list[list[float] | None] = [self._lookup(key) for key in keys]
        missing_indices = [index for index, vector in enumerate(result) if vector is None]
        if missing_indices:
            vectors = self.backend.embed_documents([texts[index] for index in missing_indices])
            for index, vector in zip(missing_indices, vectors, strict=True):
                normalized = [float(value) for value in vector]
                result[index] = normalized
                self._store(keys[index], normalized)
        cache_hits = len(texts) - len(missing_indices)
        estimated_tokens = sum(max(1, len(text) // 4) for text in texts) if texts else 0
        usage = {"input_tokens": estimated_tokens, "output_tokens": 0, "cached_tokens": 0, "total_tokens": estimated_tokens}
        record_model_invocation(
            trace_id=self.trace_id, operation="embedding", provider=self.provider, model=self.model,
            started_at=started_at, duration_ms=(time.perf_counter() - started) * 1000,
            usage=usage, cost=model_cost(self.model, usage),
            cache_hit=bool(texts) and not missing_indices,
            call_count=len(texts), remote_call_count=len(missing_indices), cache_hit_count=cache_hits,
            metadata={"count": len(texts), "cache_hits": cache_hits, "remote_count": len(missing_indices)},
        )
        return [vector or [] for vector in result]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)
