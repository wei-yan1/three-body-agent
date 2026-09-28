"""Deep-mode evidence synthesis without exposing hidden chain-of-thought."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain.chat_models import init_chat_model
from app.observability import observe_async_call


class EvidenceAnalysisAgent:
    agent_id = "evidence-analysis"

    async def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        context = payload.get("context") or {}
        evidence = list(context.get("novel_evidence") or [])
        profile = context.get("profile") or {}
        fallback = {
            "supported_facts": [item.get("chunk_id") for item in evidence if item.get("chunk_id")],
            "behavioral_observations": list(profile.get("personality") or [])[:3],
            "motivations": list(profile.get("goals") or [])[:3],
            "relationship_implications": [],
            "inferences": [],
            "uncertainties": ["回答只依据当前检索到的证据"],
            "claim_review": [],
        }
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key or not evidence:
            return fallback
        try:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            if os.getenv("DASHSCOPE_BASE_URL"):
                os.environ.setdefault("OPENAI_BASE_URL", os.environ["DASHSCOPE_BASE_URL"])
            model = init_chat_model(
                os.getenv("STORYROLE_EVIDENCE_MODEL", os.getenv("STORYROLE_REASONING_MODEL", "qwen3.7-plus")),
                model_provider="openai", temperature=0,
            )
            prompt = f"""
你是小说证据分析 Agent。只做证据归纳，不写最终回答，不输出思维链。
严格区分：原文支持的事实、基于画像的合理观察、无法确认的推断。
输出严格 JSON：supported_facts(list of objects with fact and evidence_ids),
behavioral_observations(list), motivations(list), relationship_implications(list),
inferences(list), uncertainties(list), claim_review(list of objects with claim, status, evidence_ids, reason)。
逐条核对问题规划中的 claims_to_resolve，标记 supported、partial 或 unsupported。
不得新增证据之外的剧情。
用户问题：{payload.get('message', '')}
问题规划：{json.dumps(payload.get('deep_plan') or {}, ensure_ascii=False)}
角色画像：{json.dumps(profile, ensure_ascii=False)[:10000]}
小说证据：{json.dumps(evidence, ensure_ascii=False)[:18000]}
""".strip()
            raw = str((await observe_async_call(lambda: model.ainvoke(prompt), operation="evidence_analysis", trace_id=payload.get("trace_id"), model=model, prompt=prompt)).content)
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
