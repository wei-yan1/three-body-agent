"""Deterministic decision backend used as the safe default."""
from __future__ import annotations
from typing import Any

class RuleDecisionBackend:
    name = "rule"
    def plan(self, message: str) -> dict[str, Any]:
        relationship_words = ("关系", "喜欢", "讨厌", "爱", "恨", "怎么看", "对我", "信任")
        fact_words = ("为什么", "发生", "哪一章", "当时", "原文", "经历", "是谁", "怎么")
        deep_words = ("分析", "原因", "动机", "代价", "后果", "选择", "矛盾", "比较", "如果", "意义", "为什么会")
        emotion_words = ("难过", "痛苦", "害怕", "孤独", "焦虑", "委屈", "怎么办", "安慰")
        future_words = ("后来", "最后", "结局", "未来", "之后", "最终")
        needs_relationship = any(x in message for x in relationship_words)
        needs_novel = any(x in message for x in fact_words) or len(message) > 80
        needs_memory = needs_relationship or any(x in message for x in ("还记得", "上次", "我们", "我说过"))
        future_probe = any(x in message for x in future_words)
        auto_deep = any(x in message for x in deep_words) or len(message) > 120
        return {
            "intent": "relationship_reflection" if needs_relationship else "emotional_support" if any(x in message for x in emotion_words) else "plot_or_fact" if needs_novel else "daily_chat",
            "confidence": 0.55,
            "needs_novel_search": needs_novel, "needs_memory": needs_memory,
            "needs_relationship": needs_relationship,
            "needs_timeline_check": future_probe,
            "auto_deep": auto_deep,
            "answer_mode": "in_character_chat",
            "search_queries": [message], "future_probe": future_probe,
            "planner_notes": "规则决策后端。", "decision_backend": self.name,
        }
