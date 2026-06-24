"""Structure-aware chunk builders for persona and novel RAG sources."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.loaders.persona_skill_loader import (
    iter_temporal_persona_skill_paths,
    load_jsonl_records,
)


DEFAULT_PERSONA_PROFILE_DIR = Path("data/processed/persona_profiles")
DEFAULT_THREE_BODY_TEXT_PATH = Path("data/raw/three_body_characters/three_body.txt")
DEFAULT_RAG_CHUNK_DIR = Path("data/processed/rag_chunks")

NOVEL_CHUNK_SIZE = 1800
NOVEL_CHUNK_OVERLAP = 180

TIMELINE_STAGE_ORDER = {
    "T0": 0,
    "T1": 1,
    "T2": 2,
    "T3": 3,
    "T4": 4,
    "T5": 5,
    "T6": 6,
}

TIMELINE_STAGE_ANCHORS = {
    "T0": "叶文洁出生 / 早期人生起点 / 《三体 I》主线人物活跃期",
    "T1": "青年罗辑时期 / 老年叶文洁时期 / 危机初期人物活跃期",
    "T2": "罗辑面壁者早期",
    "T3": "罗辑雪地悟道阶段",
    "T4": "罗辑威慑建立阶段 / 末日战役、自然选择号逃亡、黑暗战役前后",
    "T5": "罗辑执剑人时期",
    "T6": "罗辑死亡 / 罗辑晚年终点",
}

KNOWN_CHARACTER_NAMES = [
    "叶文洁",
    "汪淼",
    "罗辑",
    "章北海",
    "史强",
    "大史",
    "杨冬",
    "丁仪",
    "申玉菲",
    "魏成",
    "伊文斯",
    "常伟思",
    "吴岳",
    "庄颜",
    "东方延绪",
    "程心",
    "云天明",
    "维德",
    "艾AA",
    "关一帆",
    "智子",
]

SKIP_NOVEL_CHUNK_MARKERS = [
    "刘慈欣给电子书读者的寄语",
    "刘慈欣2018克拉克奖获奖感言",
    "版权信息",
    "激发个人成长",
    "认准读客熊猫",
    "您可能还会喜欢",
    "Table of Contents",
    "感谢您阅读《三体全集》",
]

BOOK_HEADING_PATTERN = re.compile(r"^三体(?P<book>I{1,3}|[123]|II·黑暗森林|III·死神永生)")
PART_HEADING_PATTERN = re.compile(r"^(上部|中部|下部|第[一二三四五六七八九十百零〇0-9]+部)\s*")
PROLOGUE_HEADING_PATTERN = re.compile(r"^序\s*章$")
CHAPTER_HEADING_PATTERN = re.compile(r"^第[一二三四五六七八九十百零〇0-9]+章\s+")


def _join_list(values: Iterable[Any]) -> str:
    return "；".join(str(value) for value in values if value)


def _relationships_to_text(relationships: dict[str, Any]) -> str:
    return "\n".join(
        f"- {name}：{description}" for name, description in relationships.items()
    )


def _make_document(text: str, metadata: dict[str, Any]) -> Document:
    return Document(page_content=text.strip(), metadata=metadata)


def _base_persona_metadata(
    record: dict[str, Any],
    source_file: Path,
    chunk_kind: str,
    chunk_id: str,
) -> dict[str, Any]:
    rag_card = record.get("RAG知识卡片", {})
    rag_metadata = rag_card.get("metadata", {})
    return {
        "chunk_id": chunk_id,
        "source_type": "persona_profile",
        "source": str(source_file),
        "source_format": "jsonl",
        "skill_id": record.get("skill_id"),
        "record_type": record.get("record_type"),
        "character": rag_metadata.get("character", record.get("character")),
        "timeline_stage": rag_metadata.get("timeline_stage", record.get("stage_id")),
        "global_timeline_anchor": rag_metadata.get("global_timeline_anchor"),
        "stage_id": record.get("stage_id"),
        "stage_name": record.get("stage_name"),
        "stage_order": record.get("stage_order"),
        "card_id": rag_card.get("card_id"),
        "knowledge_type": rag_metadata.get("knowledge_type"),
        "anti_future_sight_level": rag_metadata.get("anti_future_sight_level"),
        "chunk_kind": chunk_kind,
    }


def persona_stage_record_to_structure_documents(
    record: dict[str, Any],
    source_file: str | Path,
) -> list[Document]:
    """Split one stage record into semantic persona chunks."""
    source_path = Path(source_file)
    character = record.get("RAG知识卡片", {}).get("metadata", {}).get(
        "character", record.get("character", "unknown")
    )
    stage_id = record.get("stage_id", "unknown")
    safe_character = str(character).replace(" ", "_")

    novel = record.get("小说融合增强", {})
    rag_card = record.get("RAG知识卡片", {})
    relationships = record.get("与其他角色的关系状态", {})

    sections = [
        (
            "persona_core",
            "\n".join(
                [
                    f"角色：{character}",
                    f"阶段：{stage_id} {record.get('stage_name', '')}",
                    f"阶段定位：{record.get('阶段定位', '')}",
                    f"当前人格状态：{record.get('当前人格状态', '')}",
                    f"核心信念：{_join_list(record.get('核心信念', []))}",
                    f"情绪底色：{record.get('情绪底色', '')}",
                    f"语言风格：{record.get('语言风格', '')}",
                ]
            ),
        ),
        (
            "knowledge_boundary",
            "\n".join(
                [
                    f"角色：{character}",
                    f"阶段：{stage_id} {record.get('stage_name', '')}",
                    f"已知事件：{_join_list(record.get('已知事件', []))}",
                    f"禁止知道的未来事件：{_join_list(record.get('禁止知道的未来事件', []))}",
                    f"典型回答倾向：{_join_list(record.get('典型回答倾向', []))}",
                    f"容易发生的人设漂移：{_join_list(record.get('容易发生的人设漂移', []))}",
                    f"人设一致性校验规则：{_join_list(record.get('人设一致性校验规则', []))}",
                ]
            ),
        ),
        (
            "relationships",
            "\n".join(
                [
                    f"角色：{character}",
                    f"阶段：{stage_id} {record.get('stage_name', '')}",
                    "与其他角色的关系状态：",
                    _relationships_to_text(relationships),
                ]
            ),
        ),
        (
            "novel_enrichment",
            "\n".join(
                [
                    f"角色：{character}",
                    f"阶段：{stage_id} {record.get('stage_name', '')}",
                    f"小说阶段锚点：{novel.get('stage_anchor', '')}",
                    f"小说事件蒸馏：{_join_list(novel.get('event_distillation', []))}",
                    f"人格纹理：{_join_list(novel.get('persona_texture', []))}",
                    f"语气模拟：{_join_list(novel.get('dialogue_mimic', []))}",
                    f"小说融合补强：{novel.get('content_boost', '')}",
                ]
            ),
        ),
        (
            "rag_card",
            "\n".join(
                [
                    f"角色：{character}",
                    f"阶段：{stage_id} {record.get('stage_name', '')}",
                    f"RAG知识卡片：{rag_card.get('content', '')}",
                ]
            ),
        ),
    ]

    documents: list[Document] = []
    for chunk_kind, text in sections:
        if not text.strip():
            continue
        chunk_id = f"persona_{safe_character}_{stage_id}_{chunk_kind}"
        documents.append(
            _make_document(
                text,
                _base_persona_metadata(record, source_path, chunk_kind, chunk_id),
            )
        )
    return documents


def build_persona_structure_documents(
    profile_dir: str | Path = DEFAULT_PERSONA_PROFILE_DIR,
) -> list[Document]:
    """Build semantic chunks from every persona JSONL file."""
    documents: list[Document] = []
    for path in iter_temporal_persona_skill_paths(profile_dir):
        for record in load_jsonl_records(path):
            if record.get("record_type") != "stage":
                continue
            documents.extend(persona_stage_record_to_structure_documents(record, path))
    return documents


def _normalize_novel_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines()]


def _is_structural_heading(line: str) -> bool:
    if not line:
        return False
    return bool(
        BOOK_HEADING_PATTERN.match(line)
        or PART_HEADING_PATTERN.match(line)
        or PROLOGUE_HEADING_PATTERN.match(line)
        or CHAPTER_HEADING_PATTERN.match(line)
    )


def _infer_timeline_stage(metadata: dict[str, Any], text: str) -> str:
    """Infer the global T0-T6 stage for a novel chunk.

    The rule is intentionally deterministic and conservative. It can be refined
    later with an override table for specific chapters, but it should never ask
    an LLM to decide retrieval visibility at index time.
    """
    book = str(metadata.get("book") or "")
    part = str(metadata.get("part") or "")
    title = str(metadata.get("section_title") or "")
    haystack = f"{book}\n{part}\n{title}\n{text[:1200]}"

    if "三体III" in book or "死神永生" in book:
        if "第一部" in part:
            return "T2"
        if "第二部" in part:
            return "T5"
        return "T6"

    if "三体II" in book or "黑暗森林" in book:
        if PROLOGUE_HEADING_PATTERN.match(title) or "杨冬墓" in haystack:
            return "T1"
        if "中部" in part:
            return "T3"
        if "下部" in part:
            return "T4"
        if "上部" in part:
            return "T2"
        return "T2"

    if "三体I" in book or "三体1" in book:
        return "T0"

    return "T0"


def _character_mentions(text: str) -> str:
    mentions = [name for name in KNOWN_CHARACTER_NAMES if name in text]
    if "大史" in mentions and "史强" not in mentions:
        mentions.append("史强")
    if not mentions:
        return ""
    return "|" + "|".join(dict.fromkeys(mentions)) + "|"


def _is_indexable_novel_chunk(text: str) -> bool:
    return not any(marker in text for marker in SKIP_NOVEL_CHUNK_MARKERS)


def _section_documents_from_text(text: str, source_path: Path) -> list[Document]:
    """Split the novel into broad chapter/part sections before chunking."""
    sections: list[tuple[str, str | None, str | None, list[str]]] = []
    current_book: str | None = None
    current_part: str | None = None
    current_title = "front_matter"
    current_lines: list[str] = []

    for line in _normalize_novel_lines(text):
        if not line:
            if current_lines:
                current_lines.append("")
            continue

        if BOOK_HEADING_PATTERN.match(line):
            if current_lines:
                sections.append(
                    (current_title, current_book, current_part, current_lines)
                )
                current_lines = []
            current_book = line
            current_part = None
            current_title = line
            continue

        if (
            PART_HEADING_PATTERN.match(line)
            or PROLOGUE_HEADING_PATTERN.match(line)
            or CHAPTER_HEADING_PATTERN.match(line)
        ):
            if current_lines:
                sections.append(
                    (current_title, current_book, current_part, current_lines)
                )
                current_lines = []
            if PART_HEADING_PATTERN.match(line):
                current_part = line
            elif PROLOGUE_HEADING_PATTERN.match(line):
                current_part = None
            current_title = line
            continue

        current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_book, current_part, current_lines))

    docs: list[Document] = []
    section_index = 0
    for title, book, part, lines in sections:
        content = "\n".join(lines).strip()
        # Short repeated table-of-contents sections are not useful for retrieval.
        if len(content) < 500 and _is_structural_heading(title):
            continue
        section_index += 1
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source_id": "three_body_txt_local",
                    "source_type": "novel_text",
                    "source": str(source_path),
                    "source_format": "txt",
                    "work": "《三体》三部曲",
                    "book": book,
                    "part": part,
                    "section_title": title,
                    "section_index": section_index,
                },
            )
        )
    return docs


def build_novel_structure_documents(
    file_path: str | Path = DEFAULT_THREE_BODY_TEXT_PATH,
    chunk_size: int = NOVEL_CHUNK_SIZE,
    chunk_overlap: int = NOVEL_CHUNK_OVERLAP,
) -> list[Document]:
    """Build chapter-aware chunks from the imported novel text."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    section_docs = _section_documents_from_text(text, path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", ""],
    )
    chunks = splitter.split_documents(section_docs)

    chunks = [chunk for chunk in chunks if _is_indexable_novel_chunk(chunk.page_content)]

    for index, chunk in enumerate(chunks):
        timeline_stage = _infer_timeline_stage(chunk.metadata, chunk.page_content)
        chunk.metadata.update(
            {
                "chunk_id": f"novel_three_body_{timeline_stage}_{index:06d}",
                "chunk_index": index,
                "chunk_kind": "novel_structural_chunk",
                "timeline_stage": timeline_stage,
                "stage_order": TIMELINE_STAGE_ORDER[timeline_stage],
                "global_timeline_anchor": TIMELINE_STAGE_ANCHORS[timeline_stage],
                "character_mentions": _character_mentions(chunk.page_content),
                "anti_future_sight_level": "high",
            }
        )
    return chunks


def document_to_json_record(document: Document) -> dict[str, Any]:
    """Serialize a LangChain document into an index-friendly JSON record."""
    return {
        "chunk_id": document.metadata.get("chunk_id"),
        "text": document.page_content,
        "metadata": document.metadata,
    }


def write_documents_jsonl(documents: list[Document], output_path: str | Path) -> None:
    """Write documents as JSONL for audit and offline indexing."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(
                json.dumps(document_to_json_record(document), ensure_ascii=False)
                + "\n"
            )


def build_and_write_default_chunks(
    output_dir: str | Path = DEFAULT_RAG_CHUNK_DIR,
) -> dict[str, int]:
    """Build default persona and novel chunks and write them to JSONL files."""
    out_dir = Path(output_dir)
    persona_docs = build_persona_structure_documents()
    novel_docs = build_novel_structure_documents()

    write_documents_jsonl(persona_docs, out_dir / "persona_chunks.jsonl")
    write_documents_jsonl(novel_docs, out_dir / "novel_chunks.jsonl")
    return {"persona_chunks": len(persona_docs), "novel_chunks": len(novel_docs)}
