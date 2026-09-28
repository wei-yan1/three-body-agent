"""Agent that assembles only the context required for one turn."""

from __future__ import annotations

from typing import Any

from app.agents.storyrole.character_agent import CharacterConversationAgent


class ContextCuratorAgent:
    agent_id = "context-curator"

    def __init__(self) -> None:
        self.character_agent = CharacterConversationAgent()

    def curate(self, payload: dict[str, Any]) -> dict[str, Any]:
        plan = payload.get("plan") or {}
        message = str(payload.get("message") or "")
        context: dict[str, Any] = {
            "profile": payload.get("profile") or {},
            "recent_messages": list(payload.get("recent_messages") or [])[-12:],
            "memories": list(payload.get("memories") or [])[:5],
            "relationship": payload.get("relationship") or {},
            "timeline": payload.get("timeline") or {"allowed": True, "notes": "未启用阶段限制"},
            "evidence": list(payload.get("evidence") or [])[:8],
            "web_evidence": list(payload.get("web_evidence") or [])[:6],
        }
        supplied_evidence = payload.get("retrieved_evidence")
        if isinstance(supplied_evidence, list):
            context["novel_evidence"] = supplied_evidence[:20]
        elif plan.get("needs_novel_search"):
            context["novel_evidence"] = self.character_agent.search_evidence(
                novel_id=str(payload["novel_id"]),
                owner_id=int(payload["owner_id"]),
                query=message,
                character_name=str(payload["character_name"]),
                period_id=payload.get("period_id"),
                k=4,
            )
        else:
            context["novel_evidence"] = []
        return {"context": context, "context_notes": ["只选择与本轮问题相关的上下文"]}
