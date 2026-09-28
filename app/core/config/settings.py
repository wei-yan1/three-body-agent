"""Project-level runtime settings."""

from __future__ import annotations

import os
from functools import lru_cache


MIN_JWT_SECRET_BYTES = 32


@lru_cache
def get_runtime_settings() -> dict[str, object]:
    secret = os.environ.get("JWT_SECRET_KEY", "")
    if len(secret.encode("utf-8")) < MIN_JWT_SECRET_BYTES:
        raise RuntimeError("JWT_SECRET_KEY must be provided through the environment and contain at least 32 UTF-8 bytes")
    return {
        "jwt_secret_key": secret,
        "jwt_expire_minutes": int(os.environ.get("JWT_EXPIRE_MINUTES", "10080")),
    }


def get_jwt_secret() -> str:
    return str(get_runtime_settings()["jwt_secret_key"])


TEST_CHAT_MODEL = os.getenv("TEST_CHAT_MODEL", "qwen3.7-plus")
PROD_CHAT_MODEL = os.getenv("PROD_CHAT_MODEL", "qwen3.7-max")
DEFAULT_CHAT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", TEST_CHAT_MODEL)
DEFAULT_REWRITE_MODEL = os.getenv("REWRITE_CHAT_MODEL", TEST_CHAT_MODEL)
DEFAULT_ROUTER_MODEL = os.getenv("ROUTER_MODEL", "glm-5")
DEFAULT_ROUTER_BASE_URL = os.getenv(
    "ROUTER_BASE_URL",
    os.getenv("DASHSCOPE_BASE_URL", ""),
)

DOCUMENT_EMBEDDING_MODEL = os.getenv(
    "DASHSCOPE_DOCUMENT_EMBEDDING_MODEL",
    "text-embedding-v2",
)
QUERY_EMBEDDING_MODEL = os.getenv(
    "DASHSCOPE_QUERY_EMBEDDING_MODEL",
    "text-embedding-v2",
)
EMBEDDING_MODEL = QUERY_EMBEDDING_MODEL
RERANK_MODEL = os.getenv("DASHSCOPE_RERANK_MODEL", "qwen3-rerank")

EPISODIC_MEMORY_CAPACITY = int(os.getenv("EPISODIC_MEMORY_CAPACITY", "40"))
EPISODIC_FORGET_THRESHOLD = float(os.getenv("EPISODIC_FORGET_THRESHOLD", "0.4"))
EPISODIC_RECALL_BOOST = float(os.getenv("EPISODIC_RECALL_BOOST", "0.05"))
EPISODIC_MAX_IMPORTANCE = float(os.getenv("EPISODIC_MAX_IMPORTANCE", "0.95"))
EPISODIC_AUTO_SAVE = os.getenv("EPISODIC_AUTO_SAVE", "0").lower() in {"1", "true", "yes"}
EPISODIC_AUTO_SAVE_MIN_MESSAGE_CHARS = int(os.getenv("EPISODIC_AUTO_SAVE_MIN_MESSAGE_CHARS", "24"))
