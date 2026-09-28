"""MCP HTTP route adapter for local development."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/mcp", tags=["mcp"])


@router.get("/tools")
def list_tools() -> dict:
    """Small discovery endpoint; stdio remains the canonical MCP transport."""
    return {
        "server": "storyrole-knowledge",
        "tools": [
            "get_character_profile",
            "get_relationship_state",
            "check_timeline_boundary",
            "search_novel",
            "get_import_status",
        ],
        "transport": "stdio",
    }
