"""Structured query planning with configurable rule/Laya decision backend."""

from __future__ import annotations

from typing import Any

from app.agents.decision.factory import get_decision_backend


class QueryPlannerAgent:
    agent_id = "query-planner"

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        message = str(payload.get("message") or "").strip()
        return get_decision_backend().plan(message)
