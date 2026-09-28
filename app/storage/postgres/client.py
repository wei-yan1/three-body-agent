"""PostgreSQL connection helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row


load_dotenv()


def postgres_config(database: str | None = None) -> dict[str, Any]:
    """Build psycopg connection settings from environment variables.

    POSTGRES_DSN (or DATABASE_URL) takes precedence when supplied. The
    individual POSTGRES_* variables remain useful for local development.
    """
    dsn = os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL")
    if dsn:
        return {"conninfo": dsn}

    config: dict[str, Any] = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
        "dbname": database or os.getenv("POSTGRES_DATABASE", "three_body_agent"),
        "connect_timeout": int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "2")),
        "row_factory": dict_row,
    }
    return config


@contextmanager
def postgres_connection(database: str | None = None) -> Iterator[psycopg.Connection]:
    """Yield a transactional PostgreSQL connection with dictionary rows."""
    config = postgres_config(database=database)
    config.setdefault("row_factory", dict_row)
    connection = psycopg.connect(**config)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
