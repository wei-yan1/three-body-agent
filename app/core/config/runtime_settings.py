"""Resolve per-user settings with environment fallback."""
from __future__ import annotations

from typing import Any

from app.storage.postgres.client import postgres_connection
from app.storage.postgres.schema import init_postgres_schema


def get_user_setting(user_id: int | None, key: str, default: Any = None) -> Any:
    if not user_id:
        return default
    try:
        init_postgres_schema()
        with postgres_connection() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT setting_value FROM app_settings WHERE user_id=%s AND setting_key=%s", (user_id, key))
            row = cursor.fetchone()
        if not row:
            return default
        return row["setting_value"] if isinstance(row, dict) else row[0]
    except Exception:
        return default
