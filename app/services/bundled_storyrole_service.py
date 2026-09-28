"""Provision the existing Three-Body assets into a user's StoryRole workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.rag.loaders.persona_skill_loader import iter_temporal_persona_skill_paths, load_jsonl_records
from app.storage.repositories.storyrole_repository import (
    get_novel, list_character_periods, save_persona_profile,
    upsert_character, upsert_character_period, upsert_novel,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
THREE_BODY_NOVEL_ID = "builtin_three_body_v1"
THREE_BODY_CHUNKS = PROJECT_ROOT / "data/processed/rag_chunks/novel_chunks.jsonl"
THREE_BODY_PROFILES = PROJECT_ROOT / "data/processed/persona_profiles"


def _load_chunks() -> list[dict[str, Any]]:
    with THREE_BODY_CHUNKS.open("r", encoding="utf-8") as handle:
        import json
        return [json.loads(line) for line in handle if line.strip()]


def _character_name(record: dict[str, Any]) -> str:
    return str(record.get("character") or record.get("RAG知识卡片", {}).get("metadata", {}).get("character") or "").strip()


def ensure_three_body_workspace(owner_id: int) -> dict[str, Any]:
    if not THREE_BODY_CHUNKS.exists():
        raise FileNotFoundError(str(THREE_BODY_CHUNKS))
    novel = upsert_novel(
        novel_id=THREE_BODY_NOVEL_ID, owner_id=owner_id, name="《三体》三部曲",
        status="ready", source_kind="bundled", chunks_path=str(THREE_BODY_CHUNKS),
    )
    stages_by_character: dict[str, list[dict[str, Any]]] = {}
    for path in iter_temporal_persona_skill_paths(THREE_BODY_PROFILES):
        for record in load_jsonl_records(path):
            if record.get("record_type") != "stage":
                continue
            name = _character_name(record)
            if name:
                stages_by_character.setdefault(name, []).append(record)

    chunks = _load_chunks()
    output: list[dict[str, Any]] = []
    for name, stages in sorted(stages_by_character.items()):
        hits = [item for item in chunks if name in str(item.get("text") or "")]
        character = upsert_character(
            novel_id=THREE_BODY_NOVEL_ID, owner_id=owner_id, canonical_name=name,
            aliases=[], mention_count=sum(str(item.get("text") or "").count(name) for item in hits),
            first_chapter=str((hits[0].get("metadata") or {}).get("section_title") or "未知章节") if hits else None,
            last_chapter=str((hits[-1].get("metadata") or {}).get("section_title") or "未知章节") if hits else None,
            evidence_summary="；".join(str(item.get("text") or "")[:180].replace("\n", " ") for item in hits[:3]),
        )
        character_id = int(character["id"])
        existing = {str(item["name"]): item for item in list_character_periods(character_id=character_id, owner_id=owner_id)}
        for stage in sorted(stages, key=lambda item: int(item.get("stage_order") or 0)):
            stage_id = str(stage.get("stage_id") or "全时期")
            period = existing.get(stage_id) or upsert_character_period(
                character_id=character_id, novel_id=THREE_BODY_NOVEL_ID, owner_id=owner_id,
                name=stage_id, chapter_start=None, chapter_end=None,
                keywords=[str(stage.get("阶段定位") or ""), *[str(item) for item in stage.get("核心信念", [])[:4]]],
                evidence_ids=[], rationale="复用已有三体时期画像", confidence=1.0, status="confirmed",
            )
            stage_id = str(stage.get("stage_id") or "")
            stage_chunks = [item for item in hits if str((item.get("metadata") or {}).get("timeline_stage") or "") == stage_id]
            evidence = [{
                "chapter": str((item.get("metadata") or {}).get("section_title") or "未知章节"),
                "chunk_id": str(item.get("chunk_id") or (item.get("metadata") or {}).get("chunk_id") or ""),
                "claim": "三体原文阶段证据", "text": str(item.get("text") or "")[:1200],
            } for item in stage_chunks[:20]]
            save_persona_profile(
                character_id=character_id, novel_id=THREE_BODY_NOVEL_ID, owner_id=owner_id,
                period_id=int(period["id"]), profile={**stage, "character_name": name,
                    "novel_name": "《三体》三部曲", "period_id": int(period["id"]), "period_name": stage_id},
                model="bundled_existing_nuwa", evidence=evidence,
            )
        output.append(character)
    return {"novel": novel, "characters": output, "source_kind": "bundled", "reused": True}
