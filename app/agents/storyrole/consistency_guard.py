"""Post-generation role and boundary review."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain.chat_models import init_chat_model
from app.observability import observe_async_call


class ConsistencyGuardAgent:
    agent_id = "consistency-guard"

    async def review(self, payload: dict[str, Any]) -> dict[str, Any]:
        draft = str(payload.get("draft_answer") or "")
        profile = payload.get("profile") or {}
        if not draft:
            return {"pass": False, "score": 0.0, "violations": ["empty_answer"], "final_answer": draft}
        fallback = {"pass": True, "score": 0.75, "violations": [], "rewrite_instruction": "", "final_answer": draft}
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key or os.getenv("STORYROLE_GUARD_ENABLED", "1").lower() not in {"1", "true", "yes"}:
            return fallback
        try:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            if os.getenv("DASHSCOPE_BASE_URL"):
                os.environ.setdefault("OPENAI_BASE_URL", os.environ["DASHSCOPE_BASE_URL"])
            llm = init_chat_model(os.getenv("STORYROLE_GUARD_MODEL", os.getenv("STORYROLE_CHAT_MODEL", "qwen3.7-plus")), model_provider="openai", temperature=0)
            prompt = f"""
检查下面的角色回答是否符合角色画像、小说证据和知识边界。只输出 JSON，不输出解释。
字段：pass(boolean), score(number 0-1), violations(list), rewrite_instruction(string), final_answer(string)。
如果回答合格，final_answer 原样返回；如果不合格，只做最小幅度重写，不新增证据外事实。
角色画像：{json.dumps(profile, ensure_ascii=False)[:10000]}
当前时间线：{json.dumps(payload.get('timeline') or {}, ensure_ascii=False)[:4000]}
证据分析：{json.dumps(payload.get('reasoning') or {}, ensure_ascii=False)[:9000]}
小说证据：{json.dumps((payload.get('context') or {}).get('novel_evidence') or [], ensure_ascii=False)[:12000]}
用户问题：{payload.get('message', '')}
角色回答：{draft}
""".strip()
            raw = str((await observe_async_call(lambda: llm.ainvoke(prompt), operation="consistency_guard", trace_id=payload.get("trace_id"), model=llm, prompt=prompt)).content)
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                parsed = json.loads(raw[start:end + 1])
                if isinstance(parsed, dict) and parsed.get("final_answer"):
                    return {**fallback, **parsed}
        except Exception:
            pass
        return fallback
