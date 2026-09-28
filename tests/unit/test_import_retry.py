from __future__ import annotations

import json
from pathlib import Path

from app.services.novel_import_service import NovelImportService


def test_retry_job_retries_only_failed_sources_and_preserves_existing_chunks(tmp_path: Path) -> None:
    service = NovelImportService(root=tmp_path, auto_index=False)
    manifest = service.create_job(novel_name="book", owner_id=7, files=[("a.txt", b"one"), ("b.txt", b"two")])
    manifest["sources"][0].update({"status": "completed", "chunk_count": 3})
    manifest["sources"][1].update({"status": "failed", "error": "parse failure"})
    chunks = Path(manifest["chunks_path"])
    chunks.write_text(json.dumps({"metadata": {"source_id": f"{manifest['import_id']}_source_0000"}}, ensure_ascii=False) + "\n", encoding="utf-8")
    service._write_manifest(manifest)

    retried = service.retry_job(manifest["import_id"], owner_id=7)

    assert retried["retry_source_ids"] == ["source_0001"]
    assert retried["sources"][0]["status"] == "completed"
    assert retried["sources"][1]["status"] == "queued"
    assert chunks.exists()


def test_retry_job_retries_all_sources_for_cancelled_without_failures(tmp_path: Path) -> None:
    service = NovelImportService(root=tmp_path, auto_index=False)
    manifest = service.create_job(novel_name="book", owner_id=7, files=[("a.txt", b"one")])
    manifest["status"] = "cancelled"
    service._write_manifest(manifest)

    retried = service.retry_job(manifest["import_id"], owner_id=7)

    assert retried["retry_source_ids"] == ["source_0000"]
