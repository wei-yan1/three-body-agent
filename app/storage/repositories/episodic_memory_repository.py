"""Persistence helpers for user episodic memories."""

from __future__ import annotations

import json
from typing import Any

from app.core.config.settings import (
    EPISODIC_MAX_IMPORTANCE,
    EPISODIC_RECALL_BOOST,
)
from app.schemas.auth import UserOut
from app.storage.mysql.client import mysql_connection
from app.storage.mysql.schema import init_mysql_schema


def add_episodic_memory(
    *,
    memory_id: str,
    user: UserOut,
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
) -> dict[str, Any]:
    """Insert one episodic memory and return the persisted row."""
    init_mysql_schema()
    bounded_importance = max(0.0, min(float(importance), 1.0))
    metadata_payload = {
        "original_importance": bounded_importance,
        **(metadata or {}),
    }
    metadata_json = json.dumps(metadata_payload, ensure_ascii=False)
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO episodic_memories (
                    memory_id,
                    user_id,
                    username,
                    character_name,
                    timeline_stage,
                    mode,
                    thread_name,
                    thread_id,
                    memory_type,
                    content,
                    summary,
                    importance,
                    source_turn_range,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'episodic', %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    content=VALUES(content),
                    summary=VALUES(summary),
                    importance=VALUES(importance),
                    metadata=VALUES(metadata),
                    updated_at=NOW()
                """,
                (
                    memory_id,
                    user.id,
                    user.username,
                    character,
                    timeline_stage,
                    mode,
                    thread_name,
                    thread_id,
                    content,
                    summary,
                    bounded_importance,
                    source_turn_range,
                    metadata_json,
                ),
            )
            cursor.execute(
                "SELECT * FROM episodic_memories WHERE memory_id=%s",
                (memory_id,),
            )
            return cursor.fetchone()


def list_episodic_memories(
    *,
    user: UserOut,
    character: str,
    timeline_stage: str,
    mode: str,
    thread_name: str | None = None,
    min_importance: float = 0.1,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List recent episodic memories from MySQL for a strict scope."""
    init_mysql_schema()
    conditions = [
        "user_id=%s",
        "character_name=%s",
        "timeline_stage=%s",
        "mode=%s",
        "importance >= %s",
    ]
    params: list[Any] = [
        user.id,
        character,
        timeline_stage,
        mode,
        max(0.0, min(float(min_importance), 1.0)),
    ]
    if thread_name:
        conditions.append("thread_name=%s")
        params.append(thread_name)

    params.append(max(1, int(limit)))
    where_clause = " AND ".join(conditions)
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM episodic_memories
                WHERE {where_clause}
                ORDER BY importance DESC, updated_at DESC, id DESC
                LIMIT %s
                """,
                tuple(params),
            )
            return list(cursor.fetchall())


def get_episodic_memories_by_ids(memory_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Return episodic memory rows keyed by memory_id."""
    ids = [memory_id for memory_id in memory_ids if memory_id]
    if not ids:
        return {}
    placeholders = ", ".join(["%s"] * len(ids))
    init_mysql_schema()
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM episodic_memories
                WHERE memory_id IN ({placeholders})
                """,
                tuple(ids),
            )
            return {
                str(row["memory_id"]): row
                for row in cursor.fetchall()
            }


def touch_episodic_memories(memory_ids: list[str]) -> None:
    """Mark retrieved memories as used."""
    ids = [memory_id for memory_id in memory_ids if memory_id]
    if not ids:
        return
    placeholders = ", ".join(["%s"] * len(ids))
    init_mysql_schema()
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE episodic_memories
                SET last_used_at=NOW()
                WHERE memory_id IN ({placeholders})
                """,
                tuple(ids),
            )


def boost_episodic_memories(
    memory_ids: list[str],
    *,
    recall_boost: float = EPISODIC_RECALL_BOOST,
    max_importance: float = EPISODIC_MAX_IMPORTANCE,
) -> list[dict[str, Any]]:
    """Increase importance for memories that were successfully recalled."""
    ids = [memory_id for memory_id in memory_ids if memory_id]
    if not ids:
        return []
    placeholders = ", ".join(["%s"] * len(ids))
    safe_boost = max(0.0, float(recall_boost))
    safe_max = max(0.0, min(float(max_importance), 1.0))
    init_mysql_schema()
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                UPDATE episodic_memories
                SET
                    importance=LEAST(%s, importance + %s),
                    last_used_at=NOW()
                WHERE memory_id IN ({placeholders})
                """,
                (safe_max, safe_boost, *ids),
            )
            cursor.execute(
                f"""
                SELECT *
                FROM episodic_memories
                WHERE memory_id IN ({placeholders})
                """,
                tuple(ids),
            )
            return list(cursor.fetchall())


def forget_capacity_based_memories(
    *,
    user: UserOut,
    character: str,
    timeline_stage: str,
    mode: str,
    thread_name: str,
    capacity: int,
    threshold: float = 0.4,
) -> list[str]:
    """Forget memories when one scoped thread exceeds capacity.

    Retention score is intentionally simple and explainable:
    dynamic importance 60% + recency 25% + original importance 15%.
    The lowest retention-score memories are forgotten first.
    """
    safe_capacity = max(1, int(capacity))
    safe_threshold = max(0.0, min(float(threshold), 1.0))
    init_mysql_schema()
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*) AS memory_count
                FROM episodic_memories
                WHERE user_id=%s
                  AND character_name=%s
                  AND timeline_stage=%s
                  AND mode=%s
                  AND thread_name=%s
                """,
                (user.id, character, timeline_stage, mode, thread_name),
            )
            count_row = cursor.fetchone() or {}
            memory_count = int(count_row.get("memory_count") or 0)
            overflow = memory_count - safe_capacity
            if overflow <= 0:
                return []

            cursor.execute(
                """
                SELECT
                    memory_id,
                    importance,
                    metadata,
                    created_at,
                    last_used_at,
                    (
                        importance * 0.60
                        + (
                            GREATEST(
                                0,
                                1 - (
                                    TIMESTAMPDIFF(
                                        SECOND,
                                        COALESCE(last_used_at, created_at),
                                        NOW()
                                    ) / 604800
                                )
                            ) * 0.25
                        )
                        + (
                            CAST(
                                COALESCE(
                                    JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.original_importance')),
                                    importance
                                ) AS DECIMAL(6,4)
                            ) * 0.15
                        )
                    ) AS retention_score
                FROM episodic_memories
                WHERE user_id=%s
                  AND character_name=%s
                  AND timeline_stage=%s
                  AND mode=%s
                  AND thread_name=%s
                  AND importance <= %s
                ORDER BY retention_score ASC, importance ASC, COALESCE(last_used_at, created_at) ASC, id ASC
                LIMIT %s
                """,
                (
                    user.id,
                    character,
                    timeline_stage,
                    mode,
                    thread_name,
                    safe_threshold,
                    overflow,
                ),
            )
            memory_ids = [str(row["memory_id"]) for row in cursor.fetchall()]
            if not memory_ids:
                return []

            placeholders = ", ".join(["%s"] * len(memory_ids))
            cursor.execute(
                f"""
                DELETE FROM episodic_memories
                WHERE user_id=%s AND memory_id IN ({placeholders})
                """,
                (user.id, *memory_ids),
            )
            return memory_ids


def delete_episodic_memories_for_threads(
    *,
    user: UserOut,
    thread_ids: list[int],
) -> list[str]:
    """Delete episodic memories for threads and return their memory ids."""
    ids = sorted({int(thread_id) for thread_id in thread_ids if int(thread_id) > 0})
    if not ids:
        return []
    placeholders = ", ".join(["%s"] * len(ids))
    init_mysql_schema()
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT memory_id
                FROM episodic_memories
                WHERE user_id=%s AND thread_id IN ({placeholders})
                """,
                (user.id, *ids),
            )
            memory_ids = [str(row["memory_id"]) for row in cursor.fetchall()]
            cursor.execute(
                f"""
                DELETE FROM episodic_memories
                WHERE user_id=%s AND thread_id IN ({placeholders})
                """,
                (user.id, *ids),
            )
            return memory_ids
