"""Conversation memory decision agent; persistence stays behind MemoryTool."""
from __future__ import annotations

from typing import Any


class MemoryDecisionAgent:
    agent_id = "memory-decision"
    def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        message = str(payload.get("message") or "")
        important = any(token in message for token in ("记住", "以后", "我喜欢", "我不喜欢", "我的名字", "请别"))
        return {
            "requires_confirmation": important,
            "should_save": False,
            "candidate_status": "pending" if important else "discarded",
            "memory_type": "user_preference" if important else "turn_context",
            "importance": 0.75 if important else 0.25,
            "content": message if important else "",
            "reason": "用户明确表达了可能影响后续互动的长期信息" if important else "普通单轮内容，不写入长期记忆",
        }
