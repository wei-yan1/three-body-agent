"""A2A workflow orchestration for the StoryRole setup lifecycle."""

from __future__ import annotations

from typing import Any

from app.agents.a2a.registry import client
from app.services.novel_import_service import novel_import_service
from app.services.bundled_storyrole_service import THREE_BODY_NOVEL_ID, ensure_three_body_workspace
from app.storage.repositories.storyrole_repository import get_novel, get_persona_profile_for_period, list_character_periods


class StoryRoleSetupWorkflow:
    """Resolve characters while leaving persona profiling to an explicit action."""

    async def setup_character(self, *, novel_id: str, owner_id: int, character_name: str, trace_id: str | None = None) -> dict[str, Any]:
        if novel_id == THREE_BODY_NOVEL_ID and not get_novel(novel_id=novel_id, owner_id=owner_id):
            import asyncio
            await asyncio.to_thread(ensure_three_body_workspace, owner_id)
        resolved = await client.send("character-resolver", {
            "trace_id": trace_id,
            "novel_id": novel_id,
            "owner_id": owner_id,
            "character_name": character_name,
        })
        if not resolved.get("found"):
            return resolved
        workspace = get_novel(novel_id=novel_id, owner_id=owner_id)
        if workspace and workspace.get("source_kind") == "bundled":
            import asyncio
            await asyncio.to_thread(ensure_three_body_workspace, owner_id)
            character_id = int(resolved["character_id"])
            periods = list_character_periods(character_id=character_id, owner_id=owner_id)
            profiles = [get_persona_profile_for_period(character_id=character_id, owner_id=owner_id, period_id=int(period["id"])) for period in periods]
            profiles = [profile for profile in profiles if profile]
            if profiles:
                return {**resolved, "ready": True, "reused": True, "model": "bundled_existing_nuwa", "periods": periods, "profiles": profiles}
        periods = list_character_periods(character_id=int(resolved["character_id"]), owner_id=owner_id)
        return {**resolved, "ready": True, "profile_pending": True, "periods": periods}


storyrole_setup_workflow = StoryRoleSetupWorkflow()
