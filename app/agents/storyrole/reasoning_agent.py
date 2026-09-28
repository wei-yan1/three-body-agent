"""Bounded ReAct-style internal reasoning for a character turn."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain.chat_models import init_chat_model
from app.observability import observe_async_call


class CharacterReasoningAgent:
    agent_id = "character-reasoning"

    async def reason(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile = payload.get("context", {}).get("profile") or {}
        plan = payload.get("plan") or {}
        fallback = {
            "user_need": plan.get("intent", "daily_chat"),
            "character_emotion": (profile.get("emotional_patterns") or ["保持角色原有的情绪底色"])[0],
            "response_action": "回应并保留角色立场",
            "answer_style": ", ".join(str(x) for x in (profile.get("speech_style") or [])[:3]) or "自然、克制",
            "facts_used": [item.get("chunk_id") for item in payload.get("context", {}).get("novel_evidence", [])],
            "uncertainty": "证据不足时承认不确定",
            "must_avoid": profile.get("anti_patterns") or ["全知全能", "编造原文没有的事实"],
        }
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key or not bool(payload.get("deep_reasoning", False)):
            return fallback
        try:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            if os.getenv("DASHSCOPE_BASE_URL"):
                os.environ.setdefault("OPENAI_BASE_URL", os.environ["DASHSCOPE_BASE_URL"])
            model = init_chat_model(
                os.getenv("STORYROLE_REASONING_MODEL", os.getenv("STORYROLE_CHAT_MODEL", "qwen3.7-plus")),
                model_provider="openai",
                temperature=0,
            )
            from langchain.agents import create_agent
            from langchain_core.tools import tool

            @tool
            def read_character_profile() -> str:
                """Read the character profile relevant to this turn."""
                return json.dumps(profile, ensure_ascii=False)

            @tool
            def read_novel_evidence() -> str:
                """Read the bounded novel evidence selected by the context curator."""
                return json.dumps(payload.get("context", {}).get("novel_evidence", []), ensure_ascii=False)

            @tool
            def read_timeline_boundary() -> str:
                """Read the knowledge and spoiler boundary for this turn."""
                return json.dumps(payload.get("context", {}).get("timeline", {}), ensure_ascii=False)

            prompt = f"""
你是角色对话的内部 ReAct 决策 Agent，不负责输出最终回答。
先使用工具读取角色画像、小说证据和时间线边界，再做有限决策。
不要输出详细思维链，只输出 JSON 决策摘要。
字段：user_need, character_emotion, response_action, answer_style, facts_used, uncertainty, must_avoid。
问题规划：{json.dumps(plan, ensure_ascii=False)}
用户问题：{payload.get('message', '')}
""".strip()
            agent = create_agent(
                model=model,
                tools=[read_character_profile, read_novel_evidence, read_timeline_boundary],
                system_prompt=(
                    "你只负责内部决策。必须尊重时间线，不得让角色知道边界之外的事实；"
                    "最终输出必须是合法 JSON，不要输出 Markdown。"
                ),
            )
            result = await observe_async_call(lambda: agent.ainvoke({"messages": [{"role": "user", "content": prompt}]}), operation="character_reasoning", trace_id=payload.get("trace_id"), model=model, prompt=prompt)
            raw = str(result["messages"][-1].content)
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end + 1])
                if isinstance(parsed, dict):
                    return {**fallback, **parsed}
        except Exception:
            pass
        return fallback
