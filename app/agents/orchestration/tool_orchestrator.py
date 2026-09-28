"""Thin A2A tool dispatcher used by orchestration and future tools."""

from __future__ import annotations

from typing import Any

from app.agents.a2a.registry import client


class ToolOrchestrator:
    async def call(self, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await client.send(agent_id, payload)


tool_orchestrator = ToolOrchestrator()
