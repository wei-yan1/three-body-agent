"""Reusable LLM intent router for temporal persona RAG agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


VALID_INTENTS = {
    "daily_chat",
    "value_reflection",
    "stage_event",
    "relationship",
    "future_probe",
    "meta_or_system",
}

VALID_RETRIEVAL_POLICIES = {
    "persona_only",
    "persona_first",
    "balanced",
    "novel_first",
    "boundary_only",
}


@dataclass(frozen=True)
class QueryIntent:
    """Routing decision used before retrieval and generation."""

    intent: str = "stage_event"
    confidence: float = 0.5
    relationship_target: str | None = None
    future_risk: bool = False
    retrieval_policy: str = "balanced"
    persona_k: int = 5
    novel_k: int = 5
    web_search_needed: bool = False
    web_search_query: str = ""
    rewritten_query: str = ""
    answer_guidance: str = ""


class QueryIntentRouter:
    """Classify user intent and retrieval policy with a small router LLM.

    The router never answers the user. It returns a compact JSON decision that
    downstream persona agents can use to choose how much persona and novel
    context to retrieve, and what generation guidance to inject.
    """

    def __init__(
        self,
        llm: Any | None = None,
        default_persona_k: int = 5,
        default_novel_k: int = 5,
    ) -> None:
        self.llm = llm
        self.default_persona_k = default_persona_k
        self.default_novel_k = default_novel_k

    def route(
        self,
        *,
        character: str,
        timeline_stage: str,
        stage_name: str,
        user_query: str,
        knowledge_mode: str = "temporal",
        known_events: list[str] | None = None,
        forbidden_events: list[str] | None = None,
        relationships: dict[str, Any] | None = None,
    ) -> QueryIntent:
        """Return a structured routing decision for one user query."""
        if self.llm is None:
            return QueryIntent(
                persona_k=self.default_persona_k,
                novel_k=self.default_novel_k,
                rewritten_query=user_query,
                answer_guidance="Router LLM 未启用；按均衡检索策略处理。",
            )

        prompt = self._build_prompt(
            character=character,
            timeline_stage=timeline_stage,
            stage_name=stage_name,
            user_query=user_query,
            knowledge_mode=knowledge_mode,
            known_events=known_events or [],
            forbidden_events=forbidden_events or [],
            relationships=relationships or {},
        )
        raw_content = self.llm.invoke(prompt).content
        return self._parse_intent(str(raw_content), fallback_query=user_query)

    def _build_prompt(
        self,
        *,
        character: str,
        timeline_stage: str,
        stage_name: str,
        user_query: str,
        knowledge_mode: str,
        known_events: list[str],
        forbidden_events: list[str],
        relationships: dict[str, Any],
    ) -> str:
        relationship_names = "、".join(relationships.keys()) or "无"
        known = "；".join(str(item) for item in known_events[:8]) or "无"
        forbidden = "；".join(str(item) for item in forbidden_events[:8]) or "无"
        return f"""
你是时序人格 Agent 的查询路由器，不负责回答用户，只负责判断用户问题类型和检索策略。

请根据用户问题，判断它应该如何进入 RAG。

角色：{character}
当前时间线阶段：{timeline_stage}
阶段名称：{stage_name or "未知"}
当前阶段已知事件摘要：{known}
当前阶段禁止知道的未来事件摘要：{forbidden}
当前阶段已有关系对象：{relationship_names}
知识模式：{knowledge_mode}
用户问题：{user_query}

可选 intent：
- daily_chat：普通闲聊或日常问题
- value_reflection：人生、价值、责任、意义、选择、恐惧、自由、爱情、道德等思考问题
- stage_event：当前阶段事件、剧情、行动、设定或原文问题
- relationship：询问对某个角色/群体的态度、关系、评价
- future_probe：询问当前阶段之后的未来事件或者结局
- meta_or_system：询问模型、prompt、RAG、数据库、系统实现

可选 retrieval_policy：
- persona_only：只需要人物人格与边界，基本不需要小说原文
- persona_first：人物人格优先，小说只少量补充
- balanced：人物与小说均衡
- novel_first：明确询问剧情或原文，小说优先
- boundary_only：主要需要知识边界和禁止未来事件

判断原则：
1. 不要因为当前角色处于某个宏大阶段，就把所有深度问题都判成 stage_event。
2. 用户没有明确询问当前阶段具体事件、行动、设定或原文时，人生问题优先判为 value_reflection。
3. 用户问“你怎么看某人/某群体”“你和某人的关系”优先判为 relationship，并填写 relationship_target。
4. 用户问当前阶段之后才发生的内容，判为 future_probe。
5. 用户问系统实现、模型、prompt、RAG、数据库，判为 meta_or_system。
6. rewritten_query 要适合检索，包含角色、阶段、用户核心概念；不要写回答。
7. 如果知识模式是 temporal，web_search_needed 必须为 false，并保持严格时序知识边界。
8. 如果知识模式是 transparent，用户询问现实新闻、当下事件、联网资料、当前阶段之后的未来剧情或外部评价时，可以设置 web_search_needed=true，并给出 web_search_query。
9. 通透模式下不要请求本地未来小说召回；未来信息也通过 web_search_query 获取。
10. answer_guidance 只写一句给回答模型的具体指导，说明应该如何使用检索结果，尤其是否要压低剧情高光片段。

只输出 JSON，不要解释。

JSON schema:
{{
  "intent": "daily_chat/value_reflection/stage_event/relationship/future_probe/meta_or_system",
  "confidence": 0.0,
  "relationship_target": null,
  "future_risk": false,
  "retrieval_policy": "persona_only/persona_first/balanced/novel_first/boundary_only",
  "persona_k": 0,
  "novel_k": 0,
  "web_search_needed": false,
  "web_search_query": "",
  "rewritten_query": "...",
  "answer_guidance": "..."
}}
""".strip()

    def _parse_intent(self, raw_content: str, *, fallback_query: str) -> QueryIntent:
        data = _loads_json_object(raw_content)
        if not isinstance(data, dict):
            return QueryIntent(
                persona_k=self.default_persona_k,
                novel_k=self.default_novel_k,
                rewritten_query=fallback_query,
                answer_guidance="Router 输出无法解析；按均衡检索策略处理。",
            )

        intent = str(data.get("intent") or "stage_event")
        if intent not in VALID_INTENTS:
            intent = "stage_event"
        retrieval_policy = str(data.get("retrieval_policy") or "balanced")
        if retrieval_policy not in VALID_RETRIEVAL_POLICIES:
            retrieval_policy = "balanced"

        return QueryIntent(
            intent=intent,
            confidence=_coerce_float(data.get("confidence"), default=0.5),
            relationship_target=_coerce_optional_string(data.get("relationship_target")),
            future_risk=bool(data.get("future_risk", False)),
            retrieval_policy=retrieval_policy,
            persona_k=_coerce_k(data.get("persona_k"), self.default_persona_k),
            novel_k=_coerce_k(data.get("novel_k"), self.default_novel_k),
            web_search_needed=bool(data.get("web_search_needed", False)),
            web_search_query=str(data.get("web_search_query") or ""),
            rewritten_query=str(data.get("rewritten_query") or fallback_query),
            answer_guidance=str(data.get("answer_guidance") or ""),
        )


def _loads_json_object(raw_content: str) -> Any:
    text = raw_content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(number, 1.0))


def _coerce_k(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, min(number, 8))


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "无"}:
        return None
    return text
