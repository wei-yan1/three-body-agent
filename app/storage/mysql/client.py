"""Small MySQL client helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import pymysql
from dotenv import load_dotenv


load_dotenv()


def mysql_config(database: str | None = None) -> dict:
    config = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", "123456"),
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.DictCursor,
        "autocommit": False,
    }
    db_name = database if database is not None else os.getenv("MYSQL_DATABASE", "three_body_agent")
    if db_name:
        config["database"] = db_name
    return config


@contextmanager
def mysql_connection(database: str | None = None) -> Iterator[pymysql.connections.Connection]:
    connection = pymysql.connect(**mysql_config(database=database))
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
