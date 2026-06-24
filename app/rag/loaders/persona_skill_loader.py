"""Loader for temporal persona skill JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


DEFAULT_LUOJI_PERSONA_SKILL_PATH = Path(
    "data/processed/persona_profiles/luoji_temporal_persona_skill.jsonl"
)
DEFAULT_PERSONA_PROFILE_DIR = Path("data/processed/persona_profiles")


def load_jsonl_records(file_path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL rows as dictionaries."""
    path = Path(file_path)
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                message = f"Invalid JSONL row at {path}:{line_number}"
                raise ValueError(message) from error

    return records


def build_persona_stage_content(record: dict[str, Any]) -> str:
    """Build vector-searchable text from one persona stage record."""
    rag_card = record.get("RAG知识卡片", {})
    novel_enrichment = record.get("小说融合增强", {})
    relationships = record.get("与其他角色的关系状态", {})
    relationship_text = "；".join(
        f"{name}：{description}" for name, description in relationships.items()
    )
    sections = [
        f"角色：{record.get('character', rag_card.get('metadata', {}).get('character', ''))}",
        f"阶段：{record.get('stage_id')} {record.get('stage_name')}",
        f"阶段顺序：{record.get('stage_order', '')}",
        f"阶段定位：{record.get('阶段定位', '')}",
        f"当前人格状态：{record.get('当前人格状态', '')}",
        f"核心信念：{'；'.join(record.get('核心信念', []))}",
        f"情绪底色：{record.get('情绪底色', '')}",
        f"语言风格：{record.get('语言风格', '')}",
        f"已知事件：{'；'.join(record.get('已知事件', []))}",
        f"禁止知道的未来事件：{'；'.join(record.get('禁止知道的未来事件', []))}",
        f"与其他角色的关系状态：{relationship_text}",
        f"典型回答倾向：{'；'.join(record.get('典型回答倾向', []))}",
        f"容易发生的人设漂移：{'；'.join(record.get('容易发生的人设漂移', []))}",
        f"人设一致性校验规则：{'；'.join(record.get('人设一致性校验规则', []))}",
        f"小说阶段锚点：{novel_enrichment.get('stage_anchor', '')}",
        f"小说事件蒸馏：{'；'.join(novel_enrichment.get('event_distillation', []))}",
        f"人格纹理：{'；'.join(novel_enrichment.get('persona_texture', []))}",
        f"语气模拟：{'；'.join(novel_enrichment.get('dialogue_mimic', []))}",
        f"小说融合补强：{novel_enrichment.get('content_boost', '')}",
        f"RAG知识卡片：{rag_card.get('content', '')}",
    ]
    return "\n".join(section for section in sections if section.strip())


def build_persona_stage_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Build metadata for vector DB filtering and MySQL traceability."""
    rag_card = record.get("RAG知识卡片", {})
    rag_metadata = rag_card.get("metadata", {})

    return {
        "skill_id": record.get("skill_id"),
        "record_type": record.get("record_type"),
        "character": rag_metadata.get("character", record.get("character", "罗辑")),
        "timeline_stage": rag_metadata.get("timeline_stage", record.get("stage_id")),
        "stage_id": record.get("stage_id"),
        "stage_name": record.get("stage_name"),
        "stage_order": record.get("stage_order"),
        "card_id": rag_card.get("card_id"),
        "knowledge_type": rag_metadata.get("knowledge_type"),
        "anti_future_sight_level": rag_metadata.get("anti_future_sight_level"),
        "global_timeline_anchor": rag_metadata.get("global_timeline_anchor"),
        "source": rag_metadata.get("source"),
        "source_format": "jsonl",
    }


def persona_stage_record_to_document(record: dict[str, Any]) -> Document:
    """Convert one persona stage record into a LangChain Document."""
    return Document(
        page_content=build_persona_stage_content(record),
        metadata=build_persona_stage_metadata(record),
    )


def load_temporal_persona_skill_documents(
    file_path: str | Path = DEFAULT_LUOJI_PERSONA_SKILL_PATH,
    include_meta: bool = False,
) -> list[Document]:
    """Load temporal persona skill JSONL as LangChain Documents."""
    records = load_jsonl_records(file_path)
    documents: list[Document] = []

    for record in records:
        record_type = record.get("record_type")
        if record_type == "meta" and not include_meta:
            continue
        if record_type != "stage":
            continue
        documents.append(persona_stage_record_to_document(record))

    return documents


def load_luoji_temporal_persona_skill_documents(
    include_meta: bool = False,
) -> list[Document]:
    """Load the default Luo Ji temporal persona skill documents."""
    return load_temporal_persona_skill_documents(
        DEFAULT_LUOJI_PERSONA_SKILL_PATH,
        include_meta=include_meta,
    )


def iter_temporal_persona_skill_paths(
    profile_dir: str | Path = DEFAULT_PERSONA_PROFILE_DIR,
) -> list[Path]:
    """Return all temporal persona JSONL files in deterministic order."""
    return sorted(Path(profile_dir).glob("*_temporal_persona_skill.jsonl"))


def load_all_temporal_persona_skill_documents(
    profile_dir: str | Path = DEFAULT_PERSONA_PROFILE_DIR,
    include_meta: bool = False,
) -> list[Document]:
    """Load every temporal persona skill JSONL file as LangChain documents."""
    documents: list[Document] = []
    for path in iter_temporal_persona_skill_paths(profile_dir):
        documents.extend(
            load_temporal_persona_skill_documents(path, include_meta=include_meta)
        )
    return documents
