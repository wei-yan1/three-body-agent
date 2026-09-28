"""StoryRole Agent A: deterministic character existence resolver."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.services.novel_import_service import novel_import_service
from app.storage.repositories.storyrole_repository import (
    get_novel, upsert_character, upsert_novel,
)


class CharacterResolverAgent:
    agent_id = "character-resolver"

    def resolve(self, payload: dict[str, Any]) -> dict[str, Any]:
        novel_id = str(payload["novel_id"])
        owner_id = int(payload["owner_id"])
        requested_name = str(payload["character_name"]).strip()
        if not requested_name:
            raise ValueError("character_name 不能为空")
        bundled_novel = get_novel(novel_id=novel_id, owner_id=owner_id)
        is_bundled = bool(bundled_novel and bundled_novel.get("source_kind") == "bundled")
        if is_bundled:
            chunks_path = Path(str(bundled_novel.get("chunks_path") or ""))
            manifest = {"status": "completed", "novel_name": bundled_novel.get("name")}
        else:
            manifest = novel_import_service.get_job(novel_id, owner_id=owner_id)
            if manifest.get("status") != "completed":
                raise ValueError("小说仍在导入中，请等待导入完成")
            chunks_path = Path(manifest["chunks_path"])
        if not chunks_path.exists():
            raise FileNotFoundError(f"小说索引文本不存在：{chunks_path}")

        hits: list[dict[str, Any]] = []
        mention_count = 0
        for record in self._records(chunks_path):
            text = str(record.get("text") or "")
            count = text.count(requested_name)
            if count <= 0:
                continue
            mention_count += count
            metadata = record.get("metadata") or {}
            chapter = self._chapter_name(metadata)
            hits.append({
                "chapter": chapter,
                "chunk_id": str(record.get("chunk_id") or metadata.get("chunk_id") or ""),
                "text": text[:1200],
                "count": count,
            })

        # Exact matching is intentionally conservative. A low-confidence fuzzy
        # suggestion is returned without pretending the character was found.
        candidates = self._fuzzy_candidates(chunks_path, requested_name) if not hits else []
        if not hits:
            return {
                "novel_id": novel_id,
                "found": False,
                "character_name": requested_name,
                "confidence": 0.0,
                "mentions": 0,
                "candidates": candidates,
                "evidence": [],
            }

        first_chapter = hits[0]["chapter"] if hits else None
        last_chapter = hits[-1]["chapter"] if hits else None
        upsert_novel(
            novel_id=novel_id,
            owner_id=owner_id,
            name=str(manifest.get("novel_name") or "未命名小说"),
            status=str(manifest.get("stage") or "ready"),
            source_kind="bundled" if is_bundled else "imported",
            chunks_path=str(chunks_path),
        )
        row = upsert_character(
            novel_id=novel_id,
            owner_id=owner_id,
            canonical_name=requested_name,
            aliases=[],
            mention_count=mention_count,
            first_chapter=first_chapter,
            last_chapter=last_chapter,
            evidence_summary="；".join(item["text"][:180].replace("\n", " ") for item in hits[:3]),
        )
        return {
            "novel_id": novel_id,
            "found": True,
            "character_id": int(row["id"]),
            "character_name": requested_name,
            "confidence": min(1.0, 0.65 + min(mention_count, 20) / 100),
            "mentions": mention_count,
            "first_chapter": first_chapter,
            "last_chapter": last_chapter,
            "evidence": hits[:12],
            "candidates": [],
        }

    @staticmethod
    def _records(path: Path):
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    yield json.loads(line)

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
    def _fuzzy_candidates(cls, path: Path, requested_name: str) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for record in cls._records(path):
            text = str(record.get("text") or "")
            for candidate in re.findall(r"[\u4e00-\u9fff]{2,4}", text):
                if requested_name[:1] in candidate or candidate[:1] == requested_name[:1]:
                    counts[candidate] = counts.get(candidate, 0) + 1
        return [
            {"character_name": name, "mentions": count}
            for name, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:5]
            if count >= 2
        ]
