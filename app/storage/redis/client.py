"""Redis client helpers."""

from __future__ import annotations

import os
from functools import lru_cache

import redis
from dotenv import load_dotenv


load_dotenv()


def redis_config() -> dict:
    password = os.getenv("REDIS_PASSWORD") or None
    return {
        "host": os.getenv("REDIS_HOST", "localhost"),
        "port": int(os.getenv("REDIS_PORT", "6379")),
        "db": int(os.getenv("REDIS_DB", "0")),
        "password": password,
        "decode_responses": True,
    }


@lru_cache(maxsize=1)
def redis_client() -> redis.Redis:
    return redis.Redis(**redis_config())
