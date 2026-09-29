"""Conversation thread persistence for PostgreSQL and Redis."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from app.schemas.auth import UserOut
from app.storage.postgres.client import postgres_connection
from app.storage.postgres.schema import init_postgres_schema
from app.storage.redis.client import redis_client


DEFAULT_THREAD_NAME = "线程1"


def normalize_thread_name(thread_name: str | None) -> str:
    text = (thread_name or "").strip()
    return text or DEFAULT_THREAD_NAME


def redis_thread_key(
    *,
    username: str,
    character: str,
    timeline_stage: str,
    mode: str,
    thread_name: str,
    novel_id: str | None = None,
    character_id: int | None = None,
    period_id: int | None = None,
) -> str:
    safe_thread_name = normalize_thread_name(thread_name)
    scope = f":{novel_id}" if novel_id else ""
    character_scope = f":c{character_id}" if character_id is not None else ""
    period_scope = f":p{period_id}" if period_id is not None else ":p0"
    return f"chat:{username}{scope}{character_scope}{period_scope}:{character}:{timeline_stage}:{mode}:{safe_thread_name}"


def get_or_create_thread(
    *,
    user: UserOut,
    character: str,
    timeline_stage: str,
    mode: str,
    thread_name: str | None = None,
    novel_id: str | None = None,
    character_id: int | None = None,
    period_id: int | None = None,
) -> dict[str, Any]:
    init_postgres_schema()
    normalized_thread_name = normalize_thread_name(thread_name)
    with postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM chat_threads
                WHERE user_id=%s
                  AND character_name=%s
                  AND timeline_stage=%s
                  AND mode=%s
                  AND thread_name=%s
                  AND COALESCE(novel_id, '') = COALESCE(%s, '')
                  AND COALESCE(character_id, 0) = COALESCE(%s, 0)
                  AND COALESCE(period_id, 0) = COALESCE(%s, 0)
                """,
                (user.id, character, timeline_stage, mode, normalized_thread_name, novel_id, character_id, period_id),
            )
            thread = cursor.fetchone()
            if thread:
                return thread

            cursor.execute(
                """
                INSERT INTO chat_threads (
                    user_id,
                    username,
                    character_name,
                    timeline_stage,
                    mode,
                    thread_name,
                    novel_id,
                    character_id,
                    period_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING *
                """,
                (
                    user.id,
                    user.username,
                    character,
                    timeline_stage,
                    mode,
                    normalized_thread_name,
                    novel_id,
                    character_id,
                    period_id,
                ),
            )
            thread = cursor.fetchone()
            if thread:
                return thread

            # A previous schema version allowed only one thread per novel and
            # character. Reuse that row when its newer scope columns are NULL.
            cursor.execute(
                """
                SELECT *
                FROM chat_threads
                WHERE user_id=%s
                  AND character_name=%s
                  AND timeline_stage=%s
                  AND mode=%s
                  AND thread_name=%s
                  AND COALESCE(novel_id, '') = COALESCE(%s, '')
                ORDER BY id
                LIMIT 1
                """,
                (user.id, character, timeline_stage, mode, normalized_thread_name, novel_id),
            )
            return cursor.fetchone()


def list_threads(
    *,
    user: UserOut,
    character: str,
    timeline_stage: str,
    mode: str,
) -> list[dict[str, Any]]:
    init_postgres_schema()
    with postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    thread_name,
                    timeline_stage,
                    mode,
                    updated_at,
                    created_at
                FROM chat_threads
                WHERE user_id=%s
                  AND character_name=%s
                  AND timeline_stage=%s
                  AND mode=%s
                ORDER BY updated_at DESC, id DESC
                """,
                (user.id, character, timeline_stage, mode),
            )
            return list(cursor.fetchall())


def append_message(
    *,
    thread: dict[str, Any],
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    init_postgres_schema()
    created_at = datetime.now(timezone.utc).isoformat()
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
    with postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chat_messages (thread_id, role, content, metadata)
                VALUES (%s, %s, %s, %s::jsonb)
                """,
                (thread["id"], role, content, metadata_json),
            )
            cursor.execute(
                "UPDATE chat_threads SET updated_at=NOW() WHERE id=%s",
                (thread["id"],),
            )

    _append_recent_message(
        thread=thread,
        role=role,
        content=content,
        metadata=metadata or {},
        created_at=created_at,
    )


def get_recent_messages(thread: dict[str, Any]) -> list[dict[str, Any]]:
    client = redis_client()
    key = _redis_key_from_thread(thread)
    raw_messages = client.lrange(key, 0, -1)
    messages = [json.loads(item) for item in raw_messages]
    return [
        message for message in messages
        if (message.get("metadata") or {}).get("status", "completed") == "completed"
        and str((message.get("metadata") or {}).get("count_in_context", "true")).lower() != "false"
    ]


def get_context_checkpoint(thread_id: int) -> dict[str, Any] | None:
    """Return the durable history compaction checkpoint for one thread."""
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT thread_id, covered_message_id, summary, degraded, updated_at "
            "FROM chat_context_checkpoints WHERE thread_id=%s",
            (thread_id,),
        )
        return cursor.fetchone()


def get_messages_after_checkpoint(
    thread_id: int, covered_message_id: int | None = None,
) -> list[dict[str, Any]]:
    """Load canonical history from PostgreSQL, oldest first."""
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        if covered_message_id is None:
            cursor.execute(
                """SELECT id, role, content, metadata, created_at
                   FROM chat_messages
                   WHERE thread_id=%s
                     AND COALESCE(metadata->>'status', 'completed') = 'completed'
                     AND COALESCE(metadata->>'count_in_context', 'true') <> 'false'
                   ORDER BY id ASC""",
                (thread_id,),
            )
        else:
            cursor.execute(
                """SELECT id, role, content, metadata, created_at
                   FROM chat_messages
                   WHERE thread_id=%s AND id>%s
                     AND COALESCE(metadata->>'status', 'completed') = 'completed'
                     AND COALESCE(metadata->>'count_in_context', 'true') <> 'false'
                   ORDER BY id ASC""",
                (thread_id, covered_message_id),
            )
        rows = list(cursor.fetchall())
    return [
        {
            "id": int(row["id"]),
            "role": row["role"],
            "content": row["content"],
            "metadata": row.get("metadata") or {},
            "created_at": row["created_at"].isoformat()
            if hasattr(row["created_at"], "isoformat") else str(row["created_at"]),
        }
        for row in rows
    ]


def save_context_checkpoint(
    *, thread_id: int, covered_message_id: int, summary: str, degraded: bool = False,
) -> None:
    """Upsert the newest summary boundary without deleting raw messages."""
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO chat_context_checkpoints
               (thread_id, covered_message_id, summary, degraded)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (thread_id) DO UPDATE SET
                 covered_message_id=EXCLUDED.covered_message_id,
                 summary=EXCLUDED.summary,
                 degraded=EXCLUDED.degraded,
                 updated_at=NOW()""",
            (thread_id, covered_message_id, summary, degraded),
        )


def get_thread_for_scope(
    *, user: UserOut, character: str, timeline_stage: str, mode: str,
    thread_name: str | None, novel_id: str, character_id: int, period_id: int | None,
) -> dict[str, Any] | None:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT * FROM chat_threads
            WHERE user_id=%s AND character_name=%s AND timeline_stage=%s AND mode=%s
              AND thread_name=%s AND novel_id=%s
              AND character_id=%s AND COALESCE(period_id, 0)=COALESCE(%s, 0)""",
            (user.id, character, timeline_stage, mode, normalize_thread_name(thread_name), novel_id, character_id, period_id),
        )
        return cursor.fetchone()


def get_thread_messages_for_scope(
    *, user: UserOut, character: str, timeline_stage: str, mode: str,
    thread_name: str | None, novel_id: str, character_id: int, period_id: int | None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    thread = get_thread_for_scope(
        user=user, character=character, timeline_stage=timeline_stage, mode=mode,
        thread_name=thread_name, novel_id=novel_id, character_id=character_id, period_id=period_id,
    )
    return get_thread_messages(user=user, thread_id=int(thread["id"]), limit=limit) if thread else []


def get_thread_messages(
    *,
    user: UserOut,
    thread_id: int,
    limit: int = 20,
) -> list[dict[str, Any]]:
    init_postgres_schema()
    with postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id
                FROM chat_threads
                WHERE id=%s AND user_id=%s
                """,
                (thread_id, user.id),
            )
            if not cursor.fetchone():
                return []
            cursor.execute(
                """
                SELECT role, content, metadata, created_at
                FROM chat_messages
                WHERE thread_id=%s
                ORDER BY created_at DESC, id DESC
                LIMIT %s
                """,
                (thread_id, limit),
            )
            rows = list(cursor.fetchall())

    rows.reverse()
    messages = []
    for row in rows:
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        messages.append(
            {
                "role": row["role"],
                "content": row["content"],
                "metadata": metadata,
                "created_at": row["created_at"].isoformat()
                if hasattr(row["created_at"], "isoformat")
                else str(row["created_at"]),
            }
        )
    return messages


def delete_threads(
    *,
    user: UserOut,
    thread_ids: list[int],
) -> int:
    ids = sorted({int(thread_id) for thread_id in thread_ids if int(thread_id) > 0})
    if not ids:
        return 0

    placeholders = ", ".join(["%s"] * len(ids))
    init_postgres_schema()
    with postgres_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM chat_threads
                WHERE user_id=%s AND id IN ({placeholders})
                """,
                (user.id, *ids),
            )
            threads = list(cursor.fetchall())
            if not threads:
                return 0

            redis_keys = [_redis_key_from_thread(thread) for thread in threads]
            thread_ids_to_delete = [int(thread["id"]) for thread in threads]
            _clear_thread_memories(user=user, thread_ids=thread_ids_to_delete)
            delete_placeholders = ", ".join(["%s"] * len(thread_ids_to_delete))
            cursor.execute(
                f"""
                DELETE FROM chat_threads
                WHERE user_id=%s AND id IN ({delete_placeholders})
                """,
                (user.id, *thread_ids_to_delete),
            )

    if redis_keys:
        redis_client().delete(*redis_keys)
    return len(threads)


def _clear_thread_memories(*, user: UserOut, thread_ids: list[int]) -> None:
    try:
        from app.memory.tools import MemoryTool

        MemoryTool().execute("clear_all", user=user, thread_ids=thread_ids)
    except Exception:
        return


def _append_recent_message(
    *,
    thread: dict[str, Any],
    role: str,
    content: str,
    metadata: dict[str, Any],
    created_at: str,
) -> None:
    client = redis_client()
    key = _redis_key_from_thread(thread)
    max_messages = int(os.getenv("CHAT_MEMORY_MAX_MESSAGES", "20"))
    ttl_seconds = int(os.getenv("CHAT_MEMORY_TTL_SECONDS", "43200"))
    payload = {
        "role": role,
        "content": content,
        "metadata": metadata,
        "created_at": created_at,
    }
    client.rpush(key, json.dumps(payload, ensure_ascii=False))
    client.ltrim(key, -max_messages, -1)
    client.expire(key, ttl_seconds)


def _redis_key_from_thread(thread: dict[str, Any]) -> str:
    return redis_thread_key(
        username=str(thread["username"]),
        character=str(thread["character_name"]),
        timeline_stage=str(thread["timeline_stage"]),
        mode=str(thread["mode"]),
        thread_name=str(thread["thread_name"]),
        novel_id=str(thread.get("novel_id") or "") or None,
        character_id=int(thread["character_id"]) if thread.get("character_id") is not None else None,
        period_id=int(thread["period_id"]) if thread.get("period_id") is not None else None,
    )
