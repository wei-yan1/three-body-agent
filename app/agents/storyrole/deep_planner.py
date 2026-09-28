"""Deep-mode planning: decompose a question into evidence-oriented subqueries."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain.chat_models import init_chat_model
from app.observability import observe_async_call


class DeepQuestionPlannerAgent:
    agent_id = "deep-question-planner"

    async def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        message = str(payload.get("message") or "").strip()
        classification = payload.get("classification") or {}
        base = self._fallback(message, classification=classification)
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return base
        try:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            if os.getenv("DASHSCOPE_BASE_URL"):
                os.environ.setdefault("OPENAI_BASE_URL", os.environ["DASHSCOPE_BASE_URL"])
            model = init_chat_model(
                os.getenv("STORYROLE_DEEP_PLANNER_MODEL", os.getenv("STORYROLE_REASONING_MODEL", "qwen3.7-plus")),
                model_provider="openai", temperature=0,
            )
            prompt = f"""
你是 Deep 模式内部的计划制定器。问题分类和能力判断已由上游 Laya 完成；不要重新分类，不要回答用户，不要输出思维链，只输出 JSON。
按上游 intent 和能力标记，为本轮制定有限计划。核心命题最多 3 个，初始检索问题最多 3 个。
情感问题优先理解情绪需求、当前对话和关系边界；事实问题优先原文事实及时间线；关系问题优先关系状态和关系变化证据。
只有证据不足或相互矛盾时才建议一轮补充检索，最多两个补充查询。
字段：analysis_type, claims_to_resolve(list[str]), evidence_requirements(list[str]), search_queries(list[str]),
followup_queries(list[str]), needs_relationship(boolean), needs_memory(boolean), needs_timeline_check(boolean),
answer_focus(list[str]), answer_shape(string), verification_focus(list[str])。
上游分类与能力判断：{json.dumps(payload.get('classification') or {}, ensure_ascii=False)}
用户问题：{message}
""".strip()
            raw = str((await observe_async_call(lambda: model.ainvoke(prompt), operation="deep_question_planning", trace_id=payload.get("trace_id"), model=model, prompt=prompt)).content)
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict):
                return self._normalize(parsed, base)
        except Exception:
            pass
        return base

    @staticmethod
    def _fallback(message: str, *, classification: dict[str, Any] | None = None) -> dict[str, Any]:
        deep_terms = ("为什么", "原因", "动机", "代价", "后果", "选择", "矛盾", "比较", "如果", "意义", "分析")
        relationship = any(x in message for x in ("关系", "喜欢", "爱", "恨", "信任", "对我"))
        classification = classification or {}
        intent = str(classification.get("intent") or "")
        analysis_type = "relationship" if intent == "relationship_reflection" else "emotion" if intent == "emotional_support" else "fact" if intent == "plot_or_fact" else "mixed"
        return {
            "intent": "deep_analysis",
            "analysis_type": analysis_type,
            "claims_to_resolve": ["回答用户的核心问题，并区分明确事实与角色推断"],
            "evidence_requirements": ["使用当前时期允许的信息", "证据不足时明确保留不确定性"],
            "search_queries": [message],
            "followup_queries": [],
            "evidence_types": ["plot_fact", "character_behavior", "character_motivation"],
            "needs_relationship": relationship,
            "needs_memory": relationship or any(x in message for x in ("记得", "上次", "我们")),
            "needs_timeline_check": any(x in message for x in ("后来", "最后", "结局", "未来", "之后")),
            "answer_focus": ["先区分原文事实与角色推断", "以当前时期的价值观回应"],
            "answer_shape": "先回应核心问题，再给依据和必要的不确定性",
            "verification_focus": ["时期边界", "证据支持", "角色一致性"],
            "uncertainty_questions": [] if any(x in message for x in deep_terms) else ["问题是否需要小说原文支持"],
        }

    @staticmethod
    def _normalize(value: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
        queries = [str(x).strip() for x in value.get("search_queries") or [] if str(x).strip()]
        if not queries:
            queries = base["search_queries"]
        followup_queries = [str(x).strip() for x in value.get("followup_queries") or [] if str(x).strip()]
        analysis_type = str(value.get("analysis_type") or base["analysis_type"])
        if analysis_type not in {"emotion", "fact", "relationship", "mixed"}:
            analysis_type = base["analysis_type"]
        return {
            **base,
            **value,
            "analysis_type": analysis_type,
            "claims_to_resolve": [str(x) for x in value.get("claims_to_resolve") or base["claims_to_resolve"]][:3],
            "evidence_requirements": [str(x) for x in value.get("evidence_requirements") or base["evidence_requirements"]][:5],
            "search_queries": queries[:3],
            "followup_queries": followup_queries[:2],
            "evidence_types": [str(x) for x in value.get("evidence_types") or base["evidence_types"]][:5],
            "answer_focus": [str(x) for x in value.get("answer_focus") or base["answer_focus"]][:5],
            "answer_shape": str(value.get("answer_shape") or base["answer_shape"])[:300],
            "verification_focus": [str(x) for x in value.get("verification_focus") or base["verification_focus"]][:5],
            "uncertainty_questions": [str(x) for x in value.get("uncertainty_questions") or []][:5],
            "needs_relationship": bool(value.get("needs_relationship", base["needs_relationship"])),
            "needs_memory": bool(value.get("needs_memory", base["needs_memory"])),
            "needs_timeline_check": bool(value.get("needs_timeline_check", base["needs_timeline_check"])),
        }

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        text = raw.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                try:
                    value = json.loads(text[start:end + 1])
                    return value if isinstance(value, dict) else None
                except json.JSONDecodeError:
                    return None
        return None
