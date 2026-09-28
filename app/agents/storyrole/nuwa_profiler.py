"""StoryRole Agent B: Nuwa character profiling."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain.chat_models import init_chat_model
from app.observability import observe_async_call

from app.core.config.settings import TEST_CHAT_MODEL
from app.storage.repositories.storyrole_repository import save_persona_profile


class NuwaProfilerAgent:
    agent_id = "nuwa-profiler"

    async def profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        character_name = str(payload["character_name"])
        evidence = list(payload.get("evidence") or [])
        novel_name = str(payload.get("novel_name") or "未知小说")
        if not evidence:
            raise ValueError("没有足够的角色原文证据")
        profile, model = await self._llm_profile(
            character_name, novel_name, evidence, period_name=str(payload.get("period_name") or ""), trace_id=payload.get("trace_id")
        )
        profile.setdefault("character_name", character_name)
        profile.setdefault("novel_name", novel_name)
        profile.setdefault("evidence_count", len(evidence))
        if payload.get("period_id") is not None:
            profile["period_id"] = int(payload["period_id"])
            profile["period_name"] = str(payload.get("period_name") or "")
        profile_row = save_persona_profile(
            character_id=int(payload["character_id"]),
            novel_id=str(payload["novel_id"]),
            owner_id=int(payload["owner_id"]),
            profile=profile,
            model=model,
            evidence=evidence,
            period_id=int(payload["period_id"]) if payload.get("period_id") is not None else None,
        )
        return {"profile": profile, "model": model, "evidence": evidence,
                "profile_id": int(profile_row["id"]), "profile_version": int(profile_row["version"])}

    async def _llm_profile(
        self, character_name: str, novel_name: str, evidence: list[dict[str, Any]],
        *, period_name: str = "", trace_id: str | None = None,
    ) -> tuple[dict[str, Any], str]:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return self._heuristic_profile(character_name, novel_name, evidence), "heuristic"
        os.environ.setdefault("OPENAI_API_KEY", api_key)
        if os.getenv("DASHSCOPE_BASE_URL"):
            os.environ.setdefault("OPENAI_BASE_URL", os.environ["DASHSCOPE_BASE_URL"])
        model_name = os.getenv("STORYROLE_NUWA_MODEL", TEST_CHAT_MODEL)
        try:
            llm = init_chat_model(model_name, model_provider="openai", temperature=0)
            source = "\n\n".join(
                f"[{item.get('chapter', '未知章节')}] {item.get('text', '')[:1200]}" for item in evidence[:20]
            )
            prompt = f"""
你是 StoryRole 的女娲角色分析 Agent。请只依据下面的小说证据，为角色“{character_name}”生成可供对话 Agent 使用的结构化画像。
小说：{novel_name}
当前时期：{period_name or "全书时期"}

必须输出严格 JSON，不要 Markdown，不要补写证据中没有的事实。字段：
character_name, identity, personality(list), core_beliefs(list), emotional_patterns(list),
speech_style(list), decision_patterns(list), goals(list), fears(list), relationships(list of objects),
knowledge_boundary(object with known(list), unknown(list), rules(list)),
conversation_rules(list), anti_patterns(list), evidence_claims(list of objects with claim and evidence_index).

小说证据：
{source}
""".strip()
            raw = str((await observe_async_call(lambda: llm.ainvoke(prompt), operation="nuwa_profile", trace_id=trace_id, model=llm, prompt=prompt)).content)
            data = self._parse_json(raw)
            if isinstance(data, dict):
                return data, model_name
        except Exception:
            pass
        return self._heuristic_profile(character_name, novel_name, evidence), "heuristic"

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

    @staticmethod
    def _heuristic_profile(character_name: str, novel_name: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [str(item.get("text") or "") for item in evidence]
        dialogue = [text for text in texts if "“" in text or '"' in text or "：" in text]
        return {
            "character_name": character_name,
            "novel_name": novel_name,
            "identity": "待根据更多原文证据补充",
            "personality": ["从原文行为和对话中归纳，不确定处保持克制"],
            "core_beliefs": [], "emotional_patterns": [],
            "speech_style": ["优先使用原文中可观察到的表达方式"],
            "decision_patterns": [], "goals": [], "fears": [], "relationships": [],
            "knowledge_boundary": {"known": [], "unknown": [], "rules": ["不得把未出现的事实当作角色记忆"]},
            "conversation_rules": ["不自称 AI", "证据不足时承认不确定"],
            "anti_patterns": ["全知全能", "替作者解释一切"],
            "evidence_claims": [{"claim": "角色在小说中有可检索文本证据", "evidence_index": index} for index, _ in enumerate(texts[:5])],
            "observed_dialogue_count": len(dialogue),
        }
