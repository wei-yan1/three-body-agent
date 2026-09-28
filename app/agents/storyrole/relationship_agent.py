"""Dynamic relationship context backed by PostgreSQL."""

from __future__ import annotations

from typing import Any

from app.storage.repositories.storyrole_repository import (
    get_relationship_state,
    upsert_relationship_state,
)


class RelationshipStateAgent:
    agent_id = "relationship-state"

    def resolve(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile = payload.get("profile") or {}
        character_id = payload.get("character_id")
        owner_id = payload.get("owner_id")
        if character_id and owner_id:
            user_state = get_relationship_state(
                character_id=int(character_id), owner_id=int(owner_id),
                target_type="user", target_key=str(owner_id),
            )
            if user_state:
                return {"relationships": profile.get("relationships") or {}, "user_relationship": user_state}
        relationships = profile.get("relationships") or {}
        if isinstance(relationships, list):
            relationships = {str(item.get("target", "")): item for item in relationships if isinstance(item, dict)}
        return {"relationships": relationships, "user_relationship": {}}

    def update_user_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        row = upsert_relationship_state(
            character_id=int(payload["character_id"]), novel_id=str(payload["novel_id"]),
            owner_id=int(payload["owner_id"]), target_type="user", target_key=str(payload["owner_id"]),
            target_name=str(payload.get("target_name") or "用户"),
            trust=float(payload.get("trust", 0.5)), intimacy=float(payload.get("intimacy", 0.0)),
            tension=float(payload.get("tension", 0.0)), dependency=float(payload.get("dependency", 0.0)),
            state=dict(payload.get("state") or {}), evidence=list(payload.get("evidence") or []),
        )
        return {"relationship": row}
