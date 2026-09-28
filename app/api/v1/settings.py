"""User-editable runtime settings."""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.v1.auth import current_user
from app.schemas.auth import UserOut
from app.storage.postgres.client import postgres_connection
from app.storage.postgres.schema import init_postgres_schema

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])

SETTING_DEFINITIONS: dict[str, dict[str, Any]] = {
    "embedding.provider": {"label": "向量模型来源", "type": "select", "options": ["dashscope", "ollama"], "default": "dashscope", "group": "embedding"},
    "embedding.model": {"label": "向量模型", "type": "text", "default": "text-embedding-v2", "group": "embedding"},
    "embedding.ollama_base_url": {"label": "Ollama 地址", "type": "url", "default": "http://127.0.0.1:11434", "group": "embedding"},
    "embedding.batch_size": {"label": "Embedding 批大小", "type": "number", "default": 40, "min": 1, "max": 256, "group": "indexing"},
    "embedding.workers": {"label": "Embedding 并发数", "type": "number", "default": 5, "min": 1, "max": 32, "group": "indexing"},
    "models.chat": {"label": "对话模型", "type": "text", "default": "qwen3.7-plus", "group": "models"},
    "models.router": {"label": "路由模型", "type": "text", "default": "glm-5", "group": "models"},
    "models.rerank": {"label": "重排模型", "type": "text", "default": "qwen3-rerank", "group": "models"},
    "runtime.guard_enabled": {"label": "开启一致性检查", "type": "boolean", "default": True, "group": "runtime"},
    "runtime.decision_backend": {"label": "决策后端", "type": "select", "options": ["rule", "laya_shadow", "laya"], "default": "laya", "group": "runtime"},
    "keys.dashscope": {"label": "DashScope API Key", "type": "secret", "default": "", "group": "keys", "secret": True},
    "keys.tavily": {"label": "Tavily API Key", "type": "secret", "default": "", "group": "keys", "secret": True},
}


class SettingsUpdate(BaseModel):
    values: dict[str, Any] = Field(default_factory=dict)


def _env_default(key: str, definition: dict[str, Any]) -> Any:
    env_map = {
        "embedding.provider": "EMBEDDING_PROVIDER", "embedding.model": "OLLAMA_EMBEDDING_MODEL",
        "embedding.ollama_base_url": "OLLAMA_BASE_URL", "embedding.batch_size": "STORYROLE_EMBED_BATCH_SIZE",
        "embedding.workers": "STORYROLE_EMBED_WORKERS", "models.chat": "STORYROLE_CHAT_MODEL",
        "models.router": "ROUTER_MODEL", "models.rerank": "DASHSCOPE_RERANK_MODEL",
        "runtime.guard_enabled": "STORYROLE_GUARD_ENABLED", "runtime.decision_backend": "STORYROLE_DECISION_BACKEND",
    }
    raw = os.getenv(env_map.get(key, "")) if env_map.get(key) else None
    if raw is None or raw == "":
        return definition.get("default")
    if definition["type"] == "number":
        return int(raw)
    if definition["type"] == "boolean":
        return raw.lower() in {"1", "true", "yes", "on"}
    return raw


def _validate(key: str, value: Any) -> Any:
    definition = SETTING_DEFINITIONS[key]
    kind = definition["type"]
    if kind == "number":
        value = int(value)
        if not definition["min"] <= value <= definition["max"]:
            raise HTTPException(422, f"{definition['label']}超出范围")
    elif kind == "boolean":
        if not isinstance(value, bool):
            raise HTTPException(422, f"{definition['label']}必须是布尔值")
    elif kind == "select" and value not in definition["options"]:
        raise HTTPException(422, f"{definition['label']}不是有效选项")
    elif kind in {"text", "url", "secret"} and not isinstance(value, str):
        raise HTTPException(422, f"{definition['label']}格式不正确")
    return value


@router.get("")
def get_settings(user: UserOut = Depends(current_user)) -> dict[str, Any]:
    init_postgres_schema()
    with postgres_connection() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT setting_key, setting_value, is_secret FROM app_settings WHERE user_id=%s", (user.id,))
        saved = {row["setting_key"]: row for row in cursor.fetchall()}
    values = {}
    for key, definition in SETTING_DEFINITIONS.items():
        if key in saved:
            value = saved[key]["setting_value"]
            if definition.get("secret"):
                value = "" if not value else "••••••••"
        else:
            value = "" if definition.get("secret") else _env_default(key, definition)
        values[key] = value
    return {"groups": sorted({d["group"] for d in SETTING_DEFINITIONS.values()}), "definitions": SETTING_DEFINITIONS, "values": values}


@router.put("")
def update_settings(payload: SettingsUpdate, user: UserOut = Depends(current_user)) -> dict[str, Any]:
    init_postgres_schema()
    unknown = set(payload.values) - set(SETTING_DEFINITIONS)
    if unknown:
        raise HTTPException(422, f"不支持的设置: {', '.join(sorted(unknown))}")
    with postgres_connection() as connection, connection.cursor() as cursor:
        for key, raw_value in payload.values.items():
            definition = SETTING_DEFINITIONS[key]
            if definition.get("secret") and raw_value in {"", "••••••••", None}:
                continue
            value = _validate(key, raw_value)
            cursor.execute(
                """INSERT INTO app_settings(user_id, setting_key, setting_value, is_secret)
                   VALUES (%s,%s,%s::jsonb,%s)
                   ON CONFLICT(user_id, setting_key) DO UPDATE SET setting_value=EXCLUDED.setting_value, is_secret=EXCLUDED.is_secret, updated_at=NOW()""",
                (user.id, key, __import__("json").dumps(value, ensure_ascii=False), bool(definition.get("secret"))),
            )
    return get_settings(user)
