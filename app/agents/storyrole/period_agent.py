"""StoryRole Agent for deciding whether a character needs temporal persona snapshots."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from app.observability import observe_async_call

from app.core.config.settings import TEST_CHAT_MODEL
from app.services.novel_import_service import novel_import_service
from app.storage.repositories.storyrole_repository import (
    get_character,
    get_novel,
    save_period_analysis,
)


class CharacterPeriodAnalysisAgent:
    """Investigate a character through indexed evidence instead of receiving a novel dump."""

    agent_id = "character-period-analysis"

    async def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        novel_id = str(payload["novel_id"])
        character_id = int(payload["character_id"])
        owner_id = int(payload["owner_id"])
        character = get_character(character_id=character_id, owner_id=owner_id)
        if not character or str(character["novel_id"]) != novel_id:
            raise ValueError("角色不存在或不属于当前小说")
        workspace = get_novel(novel_id=novel_id, owner_id=owner_id)
        manifest = workspace if workspace and workspace.get("source_kind") == "bundled" else novel_import_service.get_job(novel_id, owner_id=owner_id)
        if manifest.get("status") not in {"completed", "ready"}:
            raise ValueError("小说仍在导入中，请等待导入完成")
        records = self.collect_character_records(
            Path(str(manifest["chunks_path"])), str(character["canonical_name"])
        )
        if not records:
            raise ValueError("没有找到足够的角色原文证据")
        period_hints = [str(item).strip() for item in payload.get("period_hints") or [] if str(item).strip()]
        result, model = await self._decide(
            character_name=str(character["canonical_name"]),
            novel_name=str(manifest.get("novel_name") or "未知小说"),
            records=records,
            period_hints=period_hints,
            trace_id=payload.get("trace_id"),
        )
        result.setdefault("character_id", character_id)
        result.setdefault("character_name", character["canonical_name"])
        result.setdefault("evidence_count", len(records))
        row = save_period_analysis(
            character_id=character_id, novel_id=novel_id, owner_id=owner_id,
            decision=str(result.get("decision") or "stable"),
            confidence=float(result.get("confidence") or 0.0), result=result,
        )
        return {"analysis_id": int(row["id"]), "model": model, **result}

    async def _decide(
        self, *, character_name: str, novel_name: str, records: list[dict[str, Any]],
        period_hints: list[str] | None = None, trace_id: str | None = None
    ) -> tuple[dict[str, Any], str]:
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            if os.getenv("DASHSCOPE_BASE_URL"):
                os.environ.setdefault("OPENAI_BASE_URL", os.environ["DASHSCOPE_BASE_URL"])
            try:
                llm = init_chat_model(
                    os.getenv("STORYROLE_PERIOD_MODEL", TEST_CHAT_MODEL),
                    model_provider="openai", temperature=0,
                )
                evidence = "\n\n".join(
                    f"[{i}] 章节={item['chapter']} chunk_id={item['chunk_id']}\n{item['text'][:1000]}"
                    for i, item in enumerate(records[:48])
                )
                prompt = f"""
你是 StoryRole 的“角色时期判断 Agent”。你只通过角色原文证据判断是否需要多个时期人格画像。
小说：{novel_name}；角色：{character_name}
用户希望的时期划分：{", ".join(period_hints or []) or "由你判断"}

严格规则：
1. 用户给出“年轻、中年、老年”“恋爱前、恋爱后”等描述时，将其作为候选标签，结合原文证据判断对应范围。
2. 不要因为年龄、地点、普通经历、单次情绪、一次战斗或一次失败就切分。
3. 只有核心信念、核心目标、长期行为模式、关键关系或稳定表达方式发生明显且持续的变化，才建议切分。
4. 变化必须能被至少两条不同章节/场景证据支持；证据不足时 decision 必须为 stable。
5. stable 时只输出一个覆盖角色全部出现范围的时期。
6. chapter_start/chapter_end 从证据中推断，用户不需要输入章节。
7. 只能引用给出的证据，不能凭模型记忆补写剧情，不能联网。
8. 输出严格 JSON，不要 Markdown。

JSON 字段：
{{
  "decision": "stable|suggest_split|split",
  "confidence": 0.0,
  "reason": "简短原因",
  "periods": [{{"name":"", "chapter_start":"", "chapter_end":"", "keywords":[], "evidence_ids":[], "rationale":""}}],
  "change_points": [{{"chapter":"", "reason":"", "evidence_ids":[]}}]
}}

角色证据：
{evidence}
""".strip()
                raw = str((await observe_async_call(lambda: llm.ainvoke(prompt), operation="period_analysis", trace_id=trace_id, model=llm, prompt=prompt)).content)
                parsed = self._parse_json(raw)
                if parsed and isinstance(parsed.get("periods"), list):
                    return self._normalize(parsed, records), os.getenv(
                        "STORYROLE_PERIOD_MODEL", TEST_CHAT_MODEL
                    )
            except Exception:
                pass
        return self._heuristic(records), "heuristic"

    @staticmethod
    def collect_character_records(path: Path, character_name: str) -> list[dict[str, Any]]:
        return CharacterPeriodAnalysisAgent._collect_character_records(path, character_name)

    @staticmethod
    def _collect_character_records(path: Path, character_name: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file:
            for record in file:
                if not record.strip():
                    continue
                item = json.loads(record)
                text = str(item.get("text") or "")
                if character_name not in text:
                    continue
                metadata = item.get("metadata") or {}
                chapter = CharacterPeriodAnalysisAgent._chapter_name(metadata)
                records.append({
                    "chapter": chapter,
                    "chunk_id": str(item.get("chunk_id") or metadata.get("chunk_id") or ""),
                    "text": text,
                })
        return records

    @staticmethod
    def _chapter_name(metadata: dict[str, Any]) -> str:
        for key in ("section_title", "chapter_title", "chapter", "header_path"):
            value = metadata.get(key)
            if isinstance(value, list):
                return " / ".join(str(item) for item in value)
            if value:
                return str(value)
        return "未知章节"

    @classmethod
    def _normalize(cls, result: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
        periods = []
        valid_ids = {item["chunk_id"] for item in records}
        for item in result.get("periods") or []:
            if not isinstance(item, dict):
                continue
            ids = [str(value) for value in item.get("evidence_ids") or [] if str(value) in valid_ids]
            periods.append({
                "name": str(item.get("name") or "全时期"),
                "chapter_start": str(item.get("chapter_start") or records[0]["chapter"]),
                "chapter_end": str(item.get("chapter_end") or records[-1]["chapter"]),
                "keywords": [str(value) for value in item.get("keywords") or []][:12],
                "evidence_ids": ids[:20],
                "rationale": str(item.get("rationale") or ""),
            })
        if not periods:
            return cls._heuristic(records)
        decision = str(result.get("decision") or "stable")
        if len(periods) == 1:
            decision = "stable"
        elif decision not in {"suggest_split", "split"}:
            decision = "suggest_split"
        result["decision"] = decision
        result["confidence"] = max(0.0, min(float(result.get("confidence") or 0.0), 1.0))
        result["periods"] = periods
        result["change_points"] = list(result.get("change_points") or [])[:12]
        return result

    @staticmethod
    def _heuristic(records: list[dict[str, Any]]) -> dict[str, Any]:
        # Safe fallback: never invent a split without an LLM decision.
        return {
            "decision": "stable",
            "confidence": 0.42,
            "reason": "未启用时期分析模型，按保守策略使用单一画像",
            "periods": [{
                "name": "全时期",
                "chapter_start": records[0]["chapter"],
                "chapter_end": records[-1]["chapter"],
                "keywords": [],
                "evidence_ids": [item["chunk_id"] for item in records[:20]],
                "rationale": "证据不足以安全地自动切分时期",
            }],
            "change_points": [],
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


def evidence_for_period(records: list[dict[str, Any]], period: dict[str, Any]) -> list[dict[str, Any]]:
    """Select bounded character evidence for a confirmed period."""
    wanted_ids = {str(item) for item in period.get("evidence_ids") or []}
    keywords = [str(item).strip() for item in period.get("keywords") or [] if str(item).strip()]
    selected = [item for item in records if item.get("chunk_id") in wanted_ids]
    if not selected and keywords:
        selected = [
            item for item in records
            if any(keyword in str(item.get("text") or "") for keyword in keywords)
        ]
    if not selected:
        start = str(period.get("chapter_start") or "")
        end = str(period.get("chapter_end") or "")
        start_index = next((i for i, item in enumerate(records) if item.get("chapter") == start), 0)
        end_index = next(
            (i for i, item in reversed(list(enumerate(records))) if item.get("chapter") == end),
            len(records) - 1,
        )
        if end_index >= start_index and (start or end):
            selected = records[start_index:end_index + 1]
    if not selected:
        # A user-confirmed period without evidence ids is still usable. Keep a
        # bounded subset rather than accidentally passing the whole novel.
        selected = records
    return selected[:40]


def collect_character_records(path: Path, character_name: str) -> list[dict[str, Any]]:
    return CharacterPeriodAnalysisAgent.collect_character_records(path, character_name)
