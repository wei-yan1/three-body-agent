"""Role-specific decision layer for deep conversations."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain.chat_models import init_chat_model
from app.observability import observe_async_call


class RoleCognitionAgent:
    agent_id = "role-cognition"

    async def decide(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile = payload.get("profile") or {}
        evidence_analysis = payload.get("evidence_analysis") or {}
        fallback = {
            "character_emotion": (profile.get("emotional_patterns") or ["保持角色原有情绪底色"])[0],
            "value_judgment": (profile.get("core_beliefs") or ["依据自己的价值观而非讨好用户"])[0],
            "response_action": "先回应用户真正的困惑，再保留角色立场",
            "relationship_posture": "结合当前关系自然回应",
            "answer_shape": "观点 + 原因 + 必要的不确定性",
            "must_not_say": profile.get("anti_patterns") or ["全知全能", "百科式解释"],
            "analysis_type": str(payload.get("analysis_type") or "mixed"),
        }
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return fallback
        try:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            if os.getenv("DASHSCOPE_BASE_URL"):
                os.environ.setdefault("OPENAI_BASE_URL", os.environ["DASHSCOPE_BASE_URL"])
            model = init_chat_model(
                os.getenv("STORYROLE_COGNITION_MODEL", os.getenv("STORYROLE_REASONING_MODEL", "qwen3.7-plus")),
                model_provider="openai", temperature=0,
            )
            prompt = f"""
你是角色的内部认知决策 Agent，不负责写最终答案，不输出思维链。
根据角色画像、当前时期、用户问题和证据分析，决定角色应该如何回应。
如果 analysis_type 是 emotion，先承接用户情绪，再保持角色边界；如果是 relationship，结合关系状态但不要过度承诺；如果是 fact，优先依据原文事实。
优先保留角色的价值观、矛盾、情绪和关系姿态，不要把角色变成中立讲解员。
只输出 JSON：character_emotion, value_judgment, response_action,
relationship_posture, answer_shape, must_not_say。
分析类型：{payload.get('analysis_type', 'mixed')}
回答计划：{json.dumps(payload.get('deep_plan') or {}, ensure_ascii=False)[:5000]}
角色画像：{json.dumps(profile, ensure_ascii=False)[:12000]}
证据分析：{json.dumps(evidence_analysis, ensure_ascii=False)[:12000]}
用户问题：{payload.get('message', '')}
""".strip()
            raw = str((await observe_async_call(lambda: model.ainvoke(prompt), operation="role_cognition", trace_id=payload.get("trace_id"), model=model, prompt=prompt)).content)
            parsed = self._parse_json(raw)
            if isinstance(parsed, dict):
                return {**fallback, **parsed}
        except Exception:
            pass
        return fallback

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
