"""Timeline and spoiler boundary checks for dynamic novels."""
from __future__ import annotations

from typing import Any


class TimelineGuardAgent:
    agent_id = "timeline-guard"
    def check(self, payload: dict[str, Any]) -> dict[str, Any]:
        plan = payload.get("plan") or {}
        profile = payload.get("profile") or {}
        boundary = profile.get("knowledge_boundary") or {}
        future_probe = bool(plan.get("future_probe"))
        return {
            "allowed": not future_probe or not boundary.get("unknown"),
            "future_probe": future_probe,
            "known": boundary.get("known", []),
            "unknown": boundary.get("unknown", []),
            "notes": "涉及角色未知范围时，只能以推测或不确定口吻回答。" if future_probe else "未检测到明显未来剧情询问。",
        }
