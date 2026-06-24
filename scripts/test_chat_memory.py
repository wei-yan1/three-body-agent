"""Smoke test for Redis + MySQL chat memory persistence."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas.auth import UserOut
from app.storage.mysql.client import mysql_connection
from app.storage.mysql.schema import init_mysql_schema
from app.storage.redis.client import redis_client
from app.storage.repositories.session_repository import (
    append_message,
    get_or_create_thread,
    get_recent_messages,
    redis_thread_key,
)


def main() -> None:
    init_mysql_schema()
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (username, password_hash, display_name)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE display_name=VALUES(display_name)
                """,
                ("redis_test_user", "x", "redis_test_user"),
            )
            cursor.execute("SELECT * FROM users WHERE username=%s", ("redis_test_user",))
            row = cursor.fetchone()

    user = UserOut(
        id=int(row["id"]),
        username=str(row["username"]),
        display_name=row.get("display_name"),
        avatar_url=row.get("avatar_url"),
    )
    thread = get_or_create_thread(
        user=user,
        character="罗辑",
        timeline_stage="T1",
        mode="temporal",
        thread_name="测试线程",
    )
    key = redis_thread_key(
        username=user.username,
        character="罗辑",
        timeline_stage="T1",
        mode="temporal",
        thread_name="测试线程",
    )
    redis_client().delete(key)

    for index in range(21):
        append_message(
            thread=thread,
            role="user" if index % 2 == 0 else "assistant",
            content=f"msg-{index}",
            metadata={"test": True},
        )

    recent = get_recent_messages(thread)
    with mysql_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) AS total FROM chat_messages WHERE thread_id=%s",
                (thread["id"],),
            )
            total = int(cursor.fetchone()["total"])

    print(
        {
            "redis_key": key,
            "redis_recent_count": len(recent),
            "redis_first": recent[0]["content"],
            "redis_last": recent[-1]["content"],
            "redis_ttl_seconds": redis_client().ttl(key),
            "mysql_message_total": total,
        }
    )


if __name__ == "__main__":
    main()
