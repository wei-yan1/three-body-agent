"""PostgreSQL persistence for user-created StoryRole workspaces and profiles."""

from __future__ import annotations

import json
from typing import Any

from app.storage.postgres.client import postgres_connection
from app.storage.postgres.schema import init_postgres_schema


def upsert_novel(
    *, novel_id: str, owner_id: int, name: str, status: str = "ready",
    source_kind: str = "imported", chunks_path: str | None = None,
) -> dict[str, Any]:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO storyrole_novels (novel_id, owner_id, name, status, source_kind, chunks_path)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (novel_id, owner_id) DO UPDATE SET
                name=EXCLUDED.name, status=EXCLUDED.status,
                source_kind=EXCLUDED.source_kind, chunks_path=EXCLUDED.chunks_path,
                updated_at=NOW()
            RETURNING *
            """,
            (novel_id, owner_id, name, status, source_kind, chunks_path),
        )
        return cursor.fetchone()


def get_novel(*, novel_id: str, owner_id: int) -> dict[str, Any] | None:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT * FROM storyrole_novels WHERE novel_id=%s AND owner_id=%s",
            (novel_id, owner_id),
        )
        return cursor.fetchone()


def list_novels(*, owner_id: int) -> list[dict[str, Any]]:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT * FROM storyrole_novels WHERE owner_id=%s ORDER BY updated_at DESC, id DESC",
            (owner_id,),
        )
        return list(cursor.fetchall())


def rename_novel(*, novel_id: str, owner_id: int, name: str) -> dict[str, Any] | None:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """UPDATE storyrole_novels SET name=%s, updated_at=NOW()
            WHERE novel_id=%s AND owner_id=%s RETURNING *""",
            (name, novel_id, owner_id),
        )
        return cursor.fetchone()


def delete_novel(*, novel_id: str, owner_id: int) -> bool:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "DELETE FROM storyrole_novels WHERE novel_id=%s AND owner_id=%s",
            (novel_id, owner_id),
        )
        return cursor.rowcount > 0


def upsert_character(
    *, novel_id: str, owner_id: int, canonical_name: str, aliases: list[str],
    mention_count: int, first_chapter: str | None, last_chapter: str | None,
    evidence_summary: str,
) -> dict[str, Any]:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO storyrole_characters (
                novel_id, owner_id, canonical_name, aliases, mention_count,
                first_chapter, last_chapter, evidence_summary
            )
            VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s)
            ON CONFLICT (novel_id, owner_id, canonical_name) DO UPDATE SET
                aliases=EXCLUDED.aliases,
                mention_count=EXCLUDED.mention_count,
                first_chapter=EXCLUDED.first_chapter,
                last_chapter=EXCLUDED.last_chapter,
                evidence_summary=EXCLUDED.evidence_summary,
                updated_at=NOW()
            RETURNING *
            """,
            (novel_id, owner_id, canonical_name, json.dumps(aliases, ensure_ascii=False), mention_count,
             first_chapter, last_chapter, evidence_summary),
        )
        return cursor.fetchone()


def get_character(*, character_id: int, owner_id: int) -> dict[str, Any] | None:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT * FROM storyrole_characters WHERE id=%s AND owner_id=%s",
            (character_id, owner_id),
        )
        return cursor.fetchone()


def list_characters(*, novel_id: str, owner_id: int) -> list[dict[str, Any]]:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT * FROM storyrole_characters
            WHERE novel_id=%s AND owner_id=%s
            ORDER BY mention_count DESC, canonical_name
            """,
            (novel_id, owner_id),
        )
        return list(cursor.fetchall())


def save_persona_profile(
    *, character_id: int, novel_id: str, owner_id: int, profile: dict[str, Any],
    model: str, evidence: list[dict[str, Any]], period_id: int | None = None,
) -> dict[str, Any]:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO storyrole_persona_profiles (
                character_id, period_id, novel_id, owner_id, version, profile, model, status
            )
            VALUES (
                %s, %s, %s, %s,
                COALESCE((SELECT MAX(version) + 1 FROM storyrole_persona_profiles WHERE character_id=%s), 1),
                %s::jsonb, %s, 'ready'
            )
            RETURNING *
            """,
            (character_id, period_id, novel_id, owner_id, character_id, json.dumps(profile, ensure_ascii=False), model),
        )
        row = cursor.fetchone()
        cursor.execute("DELETE FROM storyrole_persona_evidence WHERE profile_id=%s", (row["id"],))
        for item in evidence:
            cursor.execute(
                """
                INSERT INTO storyrole_persona_evidence (profile_id, chapter, chunk_id, claim, evidence_text)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (row["id"], item.get("chapter"), item.get("chunk_id"), item.get("claim", ""), item.get("text", "")),
            )
        return row



def save_period_analysis(
    *, character_id: int, novel_id: str, owner_id: int,
    decision: str, confidence: float, result: dict[str, Any],
) -> dict[str, Any]:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO storyrole_period_analyses
            (character_id, novel_id, owner_id, decision, confidence, result, status)
            VALUES (%s,%s,%s,%s,%s,%s::jsonb,'ready') RETURNING *""",
            (character_id, novel_id, owner_id, decision,
             max(0.0, min(float(confidence), 1.0)), json.dumps(result, ensure_ascii=False)),
        )
        return cursor.fetchone()


def get_latest_period_analysis(*, character_id: int, owner_id: int) -> dict[str, Any] | None:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT * FROM storyrole_period_analyses
            WHERE character_id=%s AND owner_id=%s ORDER BY created_at DESC, id DESC LIMIT 1""",
            (character_id, owner_id),
        )
        return cursor.fetchone()


def upsert_character_period(
    *, character_id: int, novel_id: str, owner_id: int, name: str,
    chapter_start: str | None, chapter_end: str | None, keywords: list[str],
    evidence_ids: list[str], rationale: str = '', confidence: float = 0.0,
    status: str = 'confirmed',
) -> dict[str, Any]:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO storyrole_character_periods
            (character_id, novel_id, owner_id, name, chapter_start, chapter_end,
             keywords, evidence_ids, rationale, confidence, status)
            VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s)
            ON CONFLICT (character_id, name) DO UPDATE SET
              chapter_start=EXCLUDED.chapter_start, chapter_end=EXCLUDED.chapter_end,
              keywords=EXCLUDED.keywords, evidence_ids=EXCLUDED.evidence_ids,
              rationale=EXCLUDED.rationale, confidence=EXCLUDED.confidence,
              status=EXCLUDED.status, updated_at=NOW()
            RETURNING *""",
            (character_id, novel_id, owner_id, name, chapter_start, chapter_end,
             json.dumps(keywords, ensure_ascii=False), json.dumps(evidence_ids, ensure_ascii=False),
             rationale, max(0.0, min(float(confidence), 1.0)), status),
        )
        return cursor.fetchone()


def get_character_period(*, period_id: int, owner_id: int) -> dict[str, Any] | None:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT * FROM storyrole_character_periods WHERE id=%s AND owner_id=%s",
            (period_id, owner_id),
        )
        return cursor.fetchone()


def list_character_periods(*, character_id: int, owner_id: int) -> list[dict[str, Any]]:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT * FROM storyrole_character_periods
            WHERE character_id=%s AND owner_id=%s ORDER BY id""",
            (character_id, owner_id),
        )
        return list(cursor.fetchall())


def get_latest_persona_profile(*, character_id: int, owner_id: int) -> dict[str, Any] | None:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT p.*, c.canonical_name, c.novel_id
            FROM storyrole_persona_profiles p
            JOIN storyrole_characters c ON c.id=p.character_id
            WHERE p.character_id=%s AND p.owner_id=%s
            ORDER BY (p.period_id IS NULL) DESC, p.version DESC LIMIT 1
            """,
            (character_id, owner_id),
        )
        row = cursor.fetchone()
        if not row:
            return None
        cursor.execute(
            """
            SELECT chapter, chunk_id, claim, evidence_text AS text
            FROM storyrole_persona_evidence
            WHERE profile_id=%s ORDER BY id
            """,
            (row["id"],),
        )
        row["evidence"] = list(cursor.fetchall())
        return row




def get_persona_profile_for_period(
    *, character_id: int, owner_id: int, period_id: int | None,
) -> dict[str, Any] | None:
    if period_id is None:
        return get_latest_persona_profile(character_id=character_id, owner_id=owner_id)
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT p.*, c.canonical_name, c.novel_id
            FROM storyrole_persona_profiles p
            JOIN storyrole_characters c ON c.id=p.character_id
            WHERE p.character_id=%s AND p.owner_id=%s AND p.period_id=%s
            ORDER BY p.version DESC LIMIT 1""",
            (character_id, owner_id, period_id),
        )
        row = cursor.fetchone()
        if not row:
            return None
        cursor.execute(
            """SELECT chapter, chunk_id, claim, evidence_text AS text
            FROM storyrole_persona_evidence WHERE profile_id=%s ORDER BY id""",
            (row['id'],),
        )
        row['evidence'] = list(cursor.fetchall())
        return row


def get_relationship_state(*, character_id: int, owner_id: int, target_type: str, target_key: str) -> dict[str, Any] | None:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT * FROM storyrole_relationship_states
            WHERE character_id=%s AND owner_id=%s AND target_type=%s AND target_key=%s""",
            (character_id, owner_id, target_type, target_key),
        )
        return cursor.fetchone()


def upsert_relationship_state(
    *, character_id: int, novel_id: str, owner_id: int, target_type: str,
    target_key: str, target_name: str, trust: float, intimacy: float,
    tension: float, dependency: float, state: dict[str, Any], evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    init_postgres_schema()
    values = [max(0.0, min(float(value), 1.0)) for value in (trust, intimacy, tension, dependency)]
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO storyrole_relationship_states
            (novel_id, character_id, owner_id, target_type, target_key, target_name,
             trust, intimacy, tension, dependency, state, evidence)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
            ON CONFLICT (character_id, owner_id, target_type, target_key) DO UPDATE SET
              trust=EXCLUDED.trust, intimacy=EXCLUDED.intimacy, tension=EXCLUDED.tension,
              dependency=EXCLUDED.dependency, state=EXCLUDED.state, evidence=EXCLUDED.evidence,
              target_name=EXCLUDED.target_name, updated_at=NOW()
            RETURNING *""",
            (novel_id, character_id, owner_id, target_type, target_key, target_name,
             *values, json.dumps(state, ensure_ascii=False), json.dumps(evidence, ensure_ascii=False)),
        )
        return cursor.fetchone()


def create_memory_candidate(
    *, novel_id: str, character_id: int, owner_id: int, thread_id: int | None,
    thread_name: str = "线程1", memory_type: str, content: str, summary: str,
    importance: float, reason: str, status: str = "pending",
) -> dict[str, Any]:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT * FROM storyrole_memory_candidates
            WHERE owner_id=%s AND character_id=%s AND status='pending' AND summary=%s
            ORDER BY created_at DESC LIMIT 1""",
            (owner_id, character_id, summary),
        )
        existing = cursor.fetchone()
        if existing:
            return existing
        cursor.execute(
            """INSERT INTO storyrole_memory_candidates
            (novel_id, character_id, owner_id, thread_id, thread_name, memory_type, content, summary, importance, status, reason)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *""",
            (novel_id, character_id, owner_id, thread_id, thread_name, memory_type, content, summary,
             max(0.0, min(float(importance), 1.0)), status, reason),
        )
        return cursor.fetchone()


def list_memory_candidates(*, owner_id: int, status: str = "pending", limit: int = 50) -> list[dict[str, Any]]:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """SELECT * FROM storyrole_memory_candidates
            WHERE owner_id=%s AND status=%s ORDER BY created_at DESC LIMIT %s""",
            (owner_id, status, max(1, min(int(limit), 200))),
        )
        return list(cursor.fetchall())


def get_memory_candidate(*, candidate_id: int, owner_id: int) -> dict[str, Any] | None:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT * FROM storyrole_memory_candidates WHERE id=%s AND owner_id=%s",
            (candidate_id, owner_id),
        )
        return cursor.fetchone()


def review_memory_candidate(*, candidate_id: int, owner_id: int, status: str) -> dict[str, Any] | None:
    if status not in {"approved", "rejected"}:
        raise ValueError("status 必须是 approved 或 rejected")
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """UPDATE storyrole_memory_candidates SET status=%s, reviewed_at=NOW()
            WHERE id=%s AND owner_id=%s AND status='pending' RETURNING *""",
            (status, candidate_id, owner_id),
        )
        return cursor.fetchone()



def mark_memory_candidate_persisted(*, candidate_id: int, owner_id: int) -> dict[str, Any] | None:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """UPDATE storyrole_memory_candidates SET status='persisted', reviewed_at=NOW()
            WHERE id=%s AND owner_id=%s AND status='approved' RETURNING *""",
            (candidate_id, owner_id),
        )
        return cursor.fetchone()

