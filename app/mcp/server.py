"""StoryRole MCP server.

The server exposes read-oriented capabilities used by external agents. The
conversation path may use the same Python functions locally; MCP is the stable
boundary for future remote tool consumers.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from app.agents.storyrole.character_agent import CharacterConversationAgent
from app.services.novel_import_service import novel_import_service
from app.storage.repositories.storyrole_repository import (
    get_latest_persona_profile,
)

mcp = FastMCP("storyrole-knowledge")
_searcher = CharacterConversationAgent()


@mcp.tool()
def get_character_profile(character_id: int, owner_id: int) -> dict[str, Any]:
    """Read the latest persisted Nuwa profile for a character."""
    profile = get_latest_persona_profile(character_id=character_id, owner_id=owner_id)
    if not profile:
        return {"found": False, "error": "profile_not_found"}
    return {"found": True, "profile": profile["profile"], "version": profile["version"], "evidence": profile.get("evidence", [])}


@mcp.tool()
def get_relationship_state(character_id: int, owner_id: int) -> dict[str, Any]:
    """Read relationships from the persisted character profile."""
    profile = get_latest_persona_profile(character_id=character_id, owner_id=owner_id)
    if not profile:
        return {"found": False, "relationships": {}}
    return {"found": True, "relationships": profile["profile"].get("relationships", [])}


@mcp.tool()
def check_timeline_boundary(character_id: int, owner_id: int, query: str) -> dict[str, Any]:
    """Return known/unknown knowledge boundary hints for the character."""
    profile = get_latest_persona_profile(character_id=character_id, owner_id=owner_id)
    if not profile:
        return {"allowed": False, "reason": "profile_not_found"}
    boundary = profile["profile"].get("knowledge_boundary") or {}
    return {"allowed": True, "query": query, "known": boundary.get("known", []), "unknown": boundary.get("unknown", []), "rules": boundary.get("rules", [])}


@mcp.tool()
def search_novel(novel_id: str, owner_id: int, query: str, character_name: str = "") -> list[dict[str, Any]]:
    """Search indexed novel evidence for a conversation turn."""
    return _searcher.search_evidence(novel_id=novel_id, owner_id=owner_id, query=query, character_name=character_name)


@mcp.tool()
def get_import_status(novel_id: str, owner_id: int) -> dict[str, Any]:
    """Read the status of an imported novel workspace."""
    manifest = novel_import_service.get_job(novel_id, owner_id=owner_id)
    return {key: manifest.get(key) for key in ("import_id", "novel_name", "status", "stage", "chunk_count", "index", "warnings")}


if __name__ == "__main__":
    mcp.run(transport="stdio")
