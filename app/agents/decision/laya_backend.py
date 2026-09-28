"""Optional Laya decision backend with safe rule fallback."""
from __future__ import annotations
import os
from typing import Any
from .rule_backend import RuleDecisionBackend

class LayaDecisionBackend:
    name = "laya"
    def __init__(self) -> None:
        self.model_name = os.getenv("STORYROLE_LAYA_MODEL", "convaiinnovations/laya-multilingual")
        self.device = os.getenv("STORYROLE_LAYA_DEVICE", "auto")
        self._agent: Any | None = None
    def _load(self) -> Any:
        if self._agent is not None:
            return self._agent
        import laya
        loader = getattr(laya, "load", None)
        if loader is None:
            raise RuntimeError("laya.load is not available")
        self._agent = loader(self.model_name, **({} if self.device == "auto" else {"device": self.device}))
        return self._agent
    def plan(self, message: str) -> dict[str, Any]:
        questions = {
            "intent": {"type": "choice", "instructions": "判断用户本轮主要意图", "criteria": {
                "daily_chat": "日常闲聊", "emotional_support": "情绪陪伴", "relationship_reflection": "关系和态度",
                "plot_or_fact": "小说事实", "value_reflection": "价值和选择", "future_probe": "未来或结局",
            }},
            "needs_novel_search": {"type": "noul", "instructions": "是否需要检索小说原文"},
            "needs_memory": {"type": "noul", "instructions": "是否需要用户历史记忆"},
            "needs_relationship": {"type": "noul", "instructions": "是否需要关系状态"},
            "needs_timeline_check": {"type": "noul", "instructions": "是否需要知识边界检查"},
        }
        result = self._load().predict({"message": message}, questions)
        if not isinstance(result, dict):
            raise ValueError("unsupported Laya result")
        intent = str(result.get("intent") or "daily_chat")
        allowed = {"daily_chat", "emotional_support", "relationship_reflection", "plot_or_fact", "value_reflection", "future_probe"}
        if intent not in allowed:
            intent = "daily_chat"
        return {"intent": intent, "confidence": float(result.get("confidence", 0.5) or 0.5),
            "needs_novel_search": bool(result.get("needs_novel_search", False)), "needs_memory": bool(result.get("needs_memory", False)),
            "needs_relationship": bool(result.get("needs_relationship", False)),
            "needs_timeline_check": bool(result.get("needs_timeline_check", False)),
            "answer_mode": "in_character_chat", "search_queries": [message],
            "future_probe": intent == "future_probe", "planner_notes": "Laya 决策后端。", "decision_backend": "laya"}

class SafeLayaDecisionBackend:
    def __init__(self) -> None:
        self.laya = LayaDecisionBackend()
        self.rule = RuleDecisionBackend()
    def plan(self, message: str) -> dict[str, Any]:
        rule = self.rule.plan(message)
        try:
            laya = self.laya.plan(message)
        except Exception as error:
            return {**rule, "decision_backend": "rule_fallback", "laya_error": f"{type(error).__name__}: {error}"}
        if os.getenv("STORYROLE_DECISION_BACKEND", "laya").lower() == "laya_shadow":
            return {**rule, "decision_backend": "rule_shadow_laya", "laya_shadow": laya}
        return laya
