"""Episodic memory retrieval backed by MySQL metadata and Chroma vectors."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.rag.embeddings.embedding_model import create_dashscope_embeddings
from app.storage.repositories.episodic_memory_repository import (
    add_episodic_memory,
    boost_episodic_memories,
    get_episodic_memories_by_ids,
    list_episodic_memories,
    touch_episodic_memories,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data/indexes/chroma"
EPISODIC_MEMORY_COLLECTION = "three_body_episodic_memories"


@dataclass(frozen=True)
class MemoryItem:
    """Portable memory item used by MemoryTool and middleware."""

    id: str
    content: str
    memory_type: str = "episodic"
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class EpisodicMemory:
    """Current-thread episodic memory with structured filters + vector search."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_CHROMA_DIR,
        collection_name: str = EPISODIC_MEMORY_COLLECTION,
        embedding_function: Any | None = None,
    ) -> None:
        from langchain_chroma import Chroma

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function or create_dashscope_embeddings(),
            persist_directory=str(persist_directory),
        )

    def add(
        self,
        *,
        user: Any,
        character: str,
        timeline_stage: str,
        mode: str,
        thread_name: str,
        content: str,
        summary: str | None = None,
        importance: float = 0.5,
        thread_id: int | None = None,
        source_turn_range: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add one episodic memory to MySQL and Chroma."""
        memory_id = _new_memory_id(
            username=str(user.username),
            character=character,
            timeline_stage=timeline_stage,
            thread_name=thread_name,
        )
        memory_metadata = {
            "memory_id": memory_id,
            "memory_type": "episodic",
            "user_id": int(user.id),
            "username": str(user.username),
            "character": character,
            "character_name": character,
            "timeline_stage": timeline_stage,
            "mode": mode,
            "thread_name": thread_name,
            "thread_id": int(thread_id) if thread_id else 0,
            "importance": max(0.0, min(float(importance), 1.0)),
            "source_turn_range": source_turn_range or "",
            **(metadata or {}),
        }
        row = add_episodic_memory(
            memory_id=memory_id,
            user=user,
            character=character,
            timeline_stage=timeline_stage,
            mode=mode,
            thread_name=thread_name,
            thread_id=thread_id,
            content=content,
            summary=summary,
            importance=memory_metadata["importance"],
            source_turn_range=source_turn_range,
            metadata=memory_metadata,
        )
        created_at = row.get("created_at")
        memory_metadata["created_at"] = (
            created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
        )
        vector_content = _memory_vector_text(
            content=content,
            summary=summary,
            metadata=memory_metadata,
        )
        self.vectorstore.add_documents(
            [
                Document(
                    page_content=vector_content,
                    metadata={
                        **memory_metadata,
                        "chunk_id": memory_id,
                        "source": "episodic_memory",
                    },
                )
            ],
            ids=[memory_id],
        )
        return memory_id

    def retrieve(
        self,
        *,
        query: str,
        user: Any,
        character: str,
        timeline_stage: str,
        mode: str,
        thread_name: str,
        limit: int = 3,
        min_importance: float = 0.1,
    ) -> list[MemoryItem]:
        """Retrieve memories for exactly one user, character, stage, mode and thread."""
        limit = max(1, int(limit))
        filter_query = {
            "$and": [
                {"user_id": {"$eq": int(user.id)}},
                {"character": {"$eq": character}},
                {"timeline_stage": {"$eq": timeline_stage}},
                {"mode": {"$eq": mode}},
                {"thread_name": {"$eq": thread_name}},
                {"importance": {"$gte": max(0.0, min(float(min_importance), 1.0))}},
            ]
        }
        try:
            hits = self.vectorstore.similarity_search_with_score(
                query,
                k=max(limit * 5, 10),
                filter=filter_query,
            )
        except Exception:
            hits = []

        items = self._rank_vector_hits(hits, limit=limit)
        if not items:
            rows = list_episodic_memories(
                user=user,
                character=character,
                timeline_stage=timeline_stage,
                mode=mode,
                thread_name=thread_name,
                min_importance=min_importance,
                limit=limit,
            )
            items = [_memory_item_from_row(row) for row in rows]

        items = items[:limit]
        if items:
            boosted_rows = boost_episodic_memories([item.id for item in items])
            if boosted_rows:
                self._sync_vector_metadata(boosted_rows)
                row_map = {str(row["memory_id"]): row for row in boosted_rows}
                items = [
                    _merge_memory_item_with_row(item, row_map.get(item.id))
                    for item in items
                ]
        return items

    def delete(self, memory_ids: list[str]) -> int:
        """Delete vector entries by memory ids."""
        ids = [memory_id for memory_id in memory_ids if memory_id]
        if not ids:
            return 0
        self.vectorstore.delete(ids=ids)
        return len(ids)

    def _sync_vector_metadata(self, rows: list[dict[str, Any]]) -> None:
        documents = []
        ids = []
        for row in rows:
            memory_id = str(row["memory_id"])
            metadata = _metadata_from_row(row)
            metadata.update(
                {
                    "chunk_id": memory_id,
                    "source": "episodic_memory",
                    "user_id": int(row["user_id"]),
                    "username": str(row["username"]),
                    "character": str(row["character_name"]),
                    "character_name": str(row["character_name"]),
                    "timeline_stage": str(row["timeline_stage"]),
                    "mode": str(row["mode"]),
                    "thread_name": str(row["thread_name"]),
                    "thread_id": int(row["thread_id"] or 0),
                }
            )
            documents.append(
                Document(
                    page_content=_memory_vector_text(
                        content=str(row["content"]),
                        summary=str(row.get("summary") or ""),
                        metadata=metadata,
                    ),
                    metadata=metadata,
                )
            )
            ids.append(memory_id)
        if not ids:
            return
        try:
            self.vectorstore.update_documents(ids=ids, documents=documents)
        except Exception:
            return

    def _rank_vector_hits(
        self,
        hits: list[tuple[Document, float]],
        *,
        limit: int,
    ) -> list[MemoryItem]:
        scored: list[MemoryItem] = []
        row_map = get_episodic_memories_by_ids(
            [
                str(document.metadata.get("memory_id") or document.metadata.get("chunk_id"))
                for document, _ in hits
            ]
        )
        for document, distance in hits:
            metadata = dict(document.metadata)
            memory_id = str(metadata.get("memory_id") or metadata.get("chunk_id"))
            row = row_map.get(memory_id)
            importance = _coerce_float(
                row.get("importance") if row else metadata.get("importance"),
                0.5,
            )
            vec_score = 1.0 / (1.0 + max(float(distance), 0.0))
            recency_score = _calculate_recency(
                row.get("last_used_at") or row.get("created_at") if row else metadata.get("created_at")
            )
            score = (vec_score * 0.8 + recency_score * 0.2) * (0.8 + importance * 0.4)
            if row:
                metadata.update(_metadata_from_row(row))
            scored.append(
                MemoryItem(
                    id=memory_id,
                    content=str(row.get("summary") or row.get("content")) if row else _extract_original_content(document.page_content),
                    memory_type=str(metadata.get("memory_type") or "episodic"),
                    importance=importance,
                    metadata=metadata,
                    score=score,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]


def _memory_item_from_row(row: dict[str, Any]) -> MemoryItem:
    metadata = row.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    return MemoryItem(
        id=str(row["memory_id"]),
        content=str(row["summary"] or row["content"]),
        memory_type=str(row.get("memory_type") or "episodic"),
        importance=_coerce_float(row.get("importance"), 0.5),
        metadata=metadata,
        score=_coerce_float(row.get("importance"), 0.5),
    )


def _merge_memory_item_with_row(item: MemoryItem, row: dict[str, Any] | None) -> MemoryItem:
    if not row:
        return item
    metadata = {**item.metadata, **_metadata_from_row(row)}
    return MemoryItem(
        id=item.id,
        content=str(row.get("summary") or row.get("content") or item.content),
        memory_type=str(row.get("memory_type") or item.memory_type),
        importance=_coerce_float(row.get("importance"), item.importance),
        metadata=metadata,
        score=item.score,
    )


def _metadata_from_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    return {
        **metadata,
        "memory_id": str(row.get("memory_id") or metadata.get("memory_id") or ""),
        "importance": _coerce_float(row.get("importance"), metadata.get("importance", 0.5)),
        "created_at": row["created_at"].isoformat()
        if hasattr(row.get("created_at"), "isoformat")
        else str(row.get("created_at") or ""),
        "last_used_at": row["last_used_at"].isoformat()
        if hasattr(row.get("last_used_at"), "isoformat")
        else str(row.get("last_used_at") or ""),
    }


def _memory_vector_text(
    *,
    content: str,
    summary: str | None,
    metadata: dict[str, Any],
) -> str:
    return "\n".join(
        [
            f"记忆类型：episodic",
            f"角色：{metadata.get('character')}",
            f"时间线阶段：{metadata.get('timeline_stage')}",
            f"线程：{metadata.get('thread_name')}",
            f"摘要：{summary or content}",
            f"原始内容：{content}",
        ]
    )


def _extract_original_content(page_content: str) -> str:
    marker = "原始内容："
    if marker in page_content:
        return page_content.split(marker, 1)[1].strip()
    return page_content


def _new_memory_id(
    *,
    username: str,
    character: str,
    timeline_stage: str,
    thread_name: str,
) -> str:
    safe = "_".join(
        part.replace(" ", "_").replace(":", "_")
        for part in (username, character, timeline_stage, thread_name)
    )
    return f"mem_{safe}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def _calculate_recency(created_at: Any) -> float:
    if not created_at:
        return 0.5
    if hasattr(created_at, "timestamp"):
        age_seconds = max(0.0, time.time() - float(created_at.timestamp()))
    else:
        try:
            from datetime import datetime

            normalized = str(created_at).replace("Z", "+00:00")
            age_seconds = max(0.0, time.time() - datetime.fromisoformat(normalized).timestamp())
        except ValueError:
            return 0.5
    one_week = 7 * 24 * 60 * 60
    return max(0.0, 1.0 - min(age_seconds / one_week, 1.0))


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
