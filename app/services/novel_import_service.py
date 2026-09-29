"""Reliable, auditable novel import jobs inspired by WeKnora's task pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.rag.docreader.novel_parser import parse_novel_file
from app.rag.docreader.pipeline import build_text_documents_from_parsed
from app.observability import TraceContext


ALLOWED_SUFFIXES = {".txt", ".md", ".markdown"}
DEFAULT_IMPORT_ROOT = Path("data/imports")
DEFAULT_MAX_FILES = 100
DEFAULT_MAX_FILE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 200 * 1024 * 1024
_SAFE_NAME = re.compile(r"[^\w\-.\u4e00-\u9fff]+", re.UNICODE)
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip() or "source.txt"
    name = _SAFE_NAME.sub("_", name)
    suffix = Path(name).suffix.lower()
    return name if suffix in ALLOWED_SUFFIXES else f"{Path(name).stem}.txt"


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NovelImportService:
    def __init__(self, root: str | Path | None = None, *, max_files: int | None = None, max_file_bytes: int | None = None, max_total_bytes: int | None = None, auto_index: bool | None = None) -> None:
        self.root = Path(root or os.getenv("STORYROLE_IMPORT_ROOT", DEFAULT_IMPORT_ROOT))
        self.max_files = max_files or int(os.getenv("STORYROLE_IMPORT_MAX_FILES", DEFAULT_MAX_FILES))
        self.max_file_bytes = max_file_bytes or int(os.getenv("STORYROLE_IMPORT_MAX_FILE_BYTES", DEFAULT_MAX_FILE_BYTES))
        self.max_total_bytes = max_total_bytes or int(os.getenv("STORYROLE_IMPORT_MAX_TOTAL_BYTES", DEFAULT_MAX_TOTAL_BYTES))
        self.auto_index = auto_index if auto_index is not None else os.getenv("STORYROLE_AUTO_INDEX", "1").lower() in {"1", "true", "yes"}

    def _job_dir(self, import_id: str) -> Path:
        return self.root / import_id

    def _manifest_path(self, import_id: str) -> Path:
        return self._job_dir(import_id) / "manifest.json"

    def _read_manifest(self, import_id: str) -> dict[str, Any]:
        path = self._manifest_path(import_id)
        if not path.exists():
            raise FileNotFoundError(import_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = _now()
        _atomic_json_write(self._manifest_path(manifest["import_id"]), manifest)

    def _validate_sources(self, files: list[dict[str, Any]]) -> None:
        if not files:
            raise ValueError("至少上传一个文本文件")
        if len(files) > self.max_files:
            raise ValueError(f"单次最多上传 {self.max_files} 个文件")
        total_bytes = sum(int(item["size_bytes"]) for item in files)
        if total_bytes > self.max_total_bytes:
            raise ValueError(f"单次上传总大小不能超过 {self.max_total_bytes} 字节")
        hashes: set[str] = set()
        for item in files:
            path = Path(item["path"])
            suffix = path.suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise ValueError(f"不支持的文本格式：{path.name}")
            if int(item["size_bytes"]) > self.max_file_bytes:
                raise ValueError(f"文件 {path.name} 超过单文件大小限制")
            if item["sha256"] in hashes:
                raise ValueError(f"检测到重复文件：{path.name}")
            hashes.add(item["sha256"])

    def _new_manifest(self, *, import_id: str, owner_id: int, novel_name: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
        timestamp = _now()
        return {
            "schema_version": 2,
            "import_id": import_id,
            "owner_id": owner_id,
            "novel_name": novel_name.strip() or "未命名小说",
            "status": "queued",
            "stage": "uploaded",
            "index": None,
            "total_files": len(sources),
            "processed_files": 0,
            "total_bytes": sum(int(item["size_bytes"]) for item in sources),
            "chunk_count": 0,
            "warnings": [],
            "error": None,
            "sources": sources,
            "chunks_path": str(self._job_dir(import_id) / "chunks.jsonl"),
            "report_path": str(self._job_dir(import_id) / "report.json"),
            "created_at": timestamp,
            "updated_at": timestamp,
            "started_at": None,
            "completed_at": None,
        }

    def create_job_from_staged_files(self, *, novel_name: str, owner_id: int, files: list[dict[str, Any]]) -> dict[str, Any]:
        """Create a job by moving already streamed uploads into its owned directory."""
        self._validate_sources(files)
        import_id = uuid.uuid4().hex
        job_dir = self._job_dir(import_id)
        source_dir = job_dir / "sources"
        source_dir.mkdir(parents=True, exist_ok=False)
        sources: list[dict[str, Any]] = []
        try:
            for order, item in enumerate(files):
                source_path = Path(item["path"])
                file_name = _safe_filename(str(item.get("original_name", source_path.name)))
                target = source_dir / f"{order:04d}_{file_name}"
                shutil.move(str(source_path), str(target))
                sources.append({
                    "source_id": f"source_{order:04d}",
                    "file_name": file_name,
                    "original_name": item.get("original_name", file_name),
                    "path": str(target),
                    "size_bytes": int(item["size_bytes"]),
                    "sha256": item["sha256"],
                    "order": order,
                    "status": "queued",
                    "warnings": [],
                    "error": None,
                    "chapter_count": 0,
                    "paragraph_count": 0,
                    "dialogue_paragraph_count": 0,
                    "chunk_count": 0,
                    "quality_score": None,
                })
            manifest = self._new_manifest(import_id=import_id, owner_id=owner_id, novel_name=novel_name, sources=sources)
            _atomic_json_write(job_dir / "manifest.json", manifest)
            return manifest
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise

    def create_job(self, *, novel_name: str, owner_id: int, files: list[tuple[str, bytes]]) -> dict[str, Any]:
        """Compatibility helper for tests and local callers."""
        staging = self.root / ".staging" / uuid.uuid4().hex
        staging.mkdir(parents=True, exist_ok=False)
        staged: list[dict[str, Any]] = []
        try:
            for original_name, content in files:
                path = staging / _safe_filename(original_name)
                path.write_bytes(content)
                staged.append({"path": str(path), "original_name": original_name, "size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
            return self.create_job_from_staged_files(novel_name=novel_name, owner_id=owner_id, files=staged)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def get_job(self, import_id: str, *, owner_id: int) -> dict[str, Any]:
        manifest = self._read_manifest(import_id)
        if int(manifest["owner_id"]) != owner_id:
            raise PermissionError(import_id)
        return manifest

    def list_jobs(self, *, owner_id: int) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        jobs: list[dict[str, Any]] = []
        for manifest_path in self.root.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if int(manifest.get("owner_id", -1)) == owner_id:
                    jobs.append(manifest)
            except (OSError, ValueError, TypeError):
                continue
        return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)

    def rename_job(self, import_id: str, *, owner_id: int, novel_name: str) -> dict[str, Any]:
        manifest = self.get_job(import_id, owner_id=owner_id)
        manifest["novel_name"] = novel_name
        self._write_manifest(manifest)
        return manifest

    def delete_job(self, import_id: str, *, owner_id: int) -> None:
        manifest = self.get_job(import_id, owner_id=owner_id)
        if manifest["status"] not in TERMINAL_STATUSES:
            raise ValueError("导入任务正在处理，暂时不能删除")
        shutil.rmtree(self._job_dir(import_id), ignore_errors=True)

    def cancel_job(self, import_id: str, *, owner_id: int) -> dict[str, Any]:
        manifest = self.get_job(import_id, owner_id=owner_id)
        if manifest["status"] not in TERMINAL_STATUSES:
            manifest["status"] = "cancelled"
            manifest["stage"] = "cancelled"
            self._write_manifest(manifest)
        return manifest

    def retry_job(self, import_id: str, *, owner_id: int) -> dict[str, Any]:
        manifest = self.get_job(import_id, owner_id=owner_id)
        has_failed_source = any(
            source.get("status") == "failed"
            for source in manifest.get("sources", [])
        )
        if manifest["status"] not in {"failed", "cancelled"} and not has_failed_source:
            raise ValueError("只有失败或取消的任务可以重试")
        retryable = [source for source in manifest["sources"] if source.get("status") == "failed"]
        if not retryable:
            retryable = list(manifest["sources"])
        retryable_ids = {source["source_id"] for source in retryable}
        for source in retryable:
            source.update({
                "status": "queued", "warnings": [], "error": None,
                "chapter_count": 0, "paragraph_count": 0,
                "dialogue_paragraph_count": 0, "chunk_count": 0, "quality_score": None,
            })
        manifest.update({"status": "queued", "stage": "uploaded", "processed_files": 0, "chunk_count": 0, "warnings": [], "error": None, "completed_at": None})
        chunks_path = Path(manifest["chunks_path"])
        # A retry reuses the published chunks for sources that already completed;
        # the processing step will rebuild the temporary file from that snapshot.
        partial_path = Path(str(chunks_path) + ".partial")
        partial_path.unlink(missing_ok=True)
        manifest["retry_source_ids"] = sorted(retryable_ids)
        self._write_manifest(manifest)
        return manifest

    def _index_chunks(self, manifest: dict[str, Any], *, trace_id: str | None = None) -> dict[str, Any]:
        from app.rag.indexing.novel_indexer import NovelIndexer

        indexer = NovelIndexer(
            persist_directory=os.getenv("STORYROLE_CHROMA_DIR", "data/indexes/chroma"),
            collection_name=os.getenv("STORYROLE_INDEX_COLLECTION_PREFIX", "storyrole_novel_") + str(manifest["import_id"]),
            batch_size=int(os.getenv("STORYROLE_EMBED_BATCH_SIZE", "40")),
            max_workers=int(os.getenv("STORYROLE_EMBED_WORKERS", "5")),
            max_retries=int(os.getenv("STORYROLE_EMBED_MAX_RETRIES", "5")),
            trace_id=trace_id,
        )
        result = indexer.index_chunks(manifest["chunks_path"])
        return {
            "collection_name": result.collection_name,
            "vector_index_path": result.vector_index_path,
            "bm25_index_path": result.bm25_index_path,
            "input_documents": result.input_documents,
            "embedded_documents": result.embedded_documents,
            "skipped_documents": result.skipped_documents,
            "deleted_documents": result.deleted_documents,
            "batch_count": result.batch_count,
        }

    @staticmethod
    def _sync_novel_status(manifest: dict[str, Any]) -> None:
        try:
            from app.storage.repositories.storyrole_repository import upsert_novel
            upsert_novel(
                novel_id=str(manifest["import_id"]),
                owner_id=int(manifest["owner_id"]),
                name=str(manifest["novel_name"]),
                status=str(manifest.get("stage") or manifest.get("status") or "processing"),
            )
        except Exception:
            # Import status remains authoritative in the manifest if the DB is
            # temporarily unavailable; the next API operation can reconcile it.
            return

    def process_job(self, import_id: str) -> None:
        """Process sources independently and publish an atomic result."""
        manifest = self._read_manifest(import_id)
        if manifest["status"] in {"completed", "cancelled"}:
            return
        job_dir = self._job_dir(import_id)
        lock = job_dir / ".processing.lock"
        try:
            lock.mkdir()
        except FileExistsError:
            return

        partial_path = Path(str(manifest["chunks_path"]) + ".partial")
        retry_source_ids = set(manifest.get("retry_source_ids", []))
        trace = TraceContext(trace_id=import_id, run_type="document_import", metadata={"import_id": import_id, "novel_name": manifest.get("novel_name"), "owner_id": manifest.get("owner_id")})
        trace.start()
        try:
            manifest.update({
                "status": "processing",
                "stage": "parsing",
                "started_at": datetime.now(UTC).isoformat(),
                "completed_at": None,
                "error": None,
                "processed_files": 0,
                "chunk_count": 0,
                "warnings": [],
            })
            for source in manifest["sources"]:
                if source.get("status") == "completed" and source["source_id"] not in retry_source_ids:
                    manifest["processed_files"] += 1
                    manifest["chunk_count"] += int(source.get("chunk_count") or 0)
                    manifest["warnings"].extend(f"{source['file_name']}: {warning}" for warning in source.get("warnings", []))
                    continue
                source.update({
                    "status": "queued",
                    "warnings": [],
                    "error": None,
                    "chapter_count": 0,
                    "paragraph_count": 0,
                    "dialogue_paragraph_count": 0,
                    "chunk_count": 0,
                    "quality_score": None,
                })
            self._write_manifest(manifest)
            chapter_numbers: list[int] = []
            previous_chunks = Path(manifest["chunks_path"])
            partial_path.unlink(missing_ok=True)
            if previous_chunks.exists() and any(
                source.get("status") == "completed" and source["source_id"] not in retry_source_ids
                for source in manifest["sources"]
            ):
                preserved_ids = {
                    f"{import_id}_{source['source_id']}"
                    for source in manifest["sources"]
                    if source.get("status") == "completed" and source["source_id"] not in retry_source_ids
                }
                seen_ids: set[str] = set()
                preserved_chunk_count = 0
                with previous_chunks.open("r", encoding="utf-8") as previous, partial_path.open("w", encoding="utf-8") as output:
                    for line in previous:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        chunk_id = str(record.get("chunk_id") or (record.get("metadata") or {}).get("chunk_id") or "")
                        if (
                            str((record.get("metadata") or {}).get("source_id", "")) in preserved_ids
                            and chunk_id
                            and chunk_id not in seen_ids
                        ):
                            seen_ids.add(chunk_id)
                            output.write(line)
                            preserved_chunk_count += 1
                manifest["chunk_count"] = preserved_chunk_count
            elif previous_chunks.exists():
                previous_chunks.unlink()
            else:
                partial_path.unlink(missing_ok=True)

            seen_chunk_ids: set[str] = set()
            if partial_path.exists():
                # The preserved part was already filtered, but keep this
                # guard here so old malformed manifests cannot reintroduce
                # duplicate IDs during a retry.
                with partial_path.open("r", encoding="utf-8") as existing:
                    for line in existing:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        chunk_id = str(record.get("chunk_id") or (record.get("metadata") or {}).get("chunk_id") or "")
                        if chunk_id:
                            seen_chunk_ids.add(chunk_id)

            with partial_path.open("a", encoding="utf-8") as output:
                for source in sorted(manifest["sources"], key=lambda item: item["order"]):
                    if source.get("status") == "completed" and source["source_id"] not in retry_source_ids:
                        continue
                    current = self._read_manifest(import_id)
                    if current["status"] == "cancelled":
                        manifest["status"] = "cancelled"
                        manifest["stage"] = "cancelled"
                        self._write_manifest(manifest)
                        trace.finish(status="cancelled")
                        return
                    source["status"] = "processing"
                    self._write_manifest(manifest)
                    try:
                        with trace.step("parse", metadata={"source_id": source["source_id"], "file_name": source["file_name"]}):
                            parsed = parse_novel_file(source["path"])
                        chapter_numbers.extend(
                            chapter.chapter_number
                            for chapter in parsed.chapters
                            if chapter.chapter_number is not None
                        )
                        with trace.step("chunk", metadata={"source_id": source["source_id"]}):
                            documents = build_text_documents_from_parsed(
                                parsed,
                                file_path=source["path"],
                                source_id=f"{import_id}_{source['source_id']}",
                                source_order=int(source["order"]),
                                work=manifest["novel_name"],
                        )
                        written_documents = 0
                        for document in documents:
                            document.metadata["novel_id"] = import_id
                            document.metadata["novel_name"] = manifest["novel_name"]
                            chunk_id = str(document.metadata["chunk_id"])
                            if chunk_id in seen_chunk_ids:
                                continue
                            seen_chunk_ids.add(chunk_id)
                            output.write(json.dumps({
                                "chunk_id": chunk_id,
                                "text": document.page_content,
                                "metadata": document.metadata,
                            }, ensure_ascii=False) + "\n")
                            written_documents += 1
                        source.update({
                            "status": "completed",
                            "warnings": list(parsed.warnings),
                            "chapter_count": parsed.stats.get("chapter_count", 0),
                            "paragraph_count": parsed.stats.get("paragraph_count", 0),
                            "dialogue_paragraph_count": parsed.stats.get("dialogue_paragraph_count", 0),
                            "chunk_count": written_documents,
                            "quality_score": parsed.stats.get("quality_score"),
                        })
                        manifest["processed_files"] += 1
                        manifest["chunk_count"] += written_documents
                        manifest["warnings"].extend(
                            f"{source['file_name']}: {warning}" for warning in parsed.warnings
                        )
                    except Exception as error:  # noqa: BLE001
                        source.update({
                            "status": "failed",
                            "error": f"{type(error).__name__}: {error}",
                        })
                        manifest["warnings"].append(
                            f"{source['file_name']}: processing_failed:{type(error).__name__}"
                        )
                    self._write_manifest(manifest)

            failed_sources = [
                source for source in manifest["sources"]
                if source.get("status") == "failed"
            ]
            if manifest["processed_files"] == 0:
                manifest.update({
                    "status": "failed",
                    "stage": "failed",
                    "error": "所有文本文件处理失败",
                    "completed_at": _now(),
                })
                self._write_manifest(manifest)
                trace.finish(status="failed", error=manifest["error"])
                return
            if failed_sources:
                failed_names = [source["file_name"] for source in failed_sources]
                manifest["warnings"].append(f"partial_failure_files:{failed_names}")

            duplicates = sorted({
                number for number in chapter_numbers
                if chapter_numbers.count(number) > 1
            })
            if duplicates:
                manifest["warnings"].append(f"duplicate_chapter_numbers:{duplicates}")
            if chapter_numbers:
                expected = set(range(min(chapter_numbers), max(chapter_numbers) + 1))
                gaps = sorted(expected - set(chapter_numbers))
                if gaps:
                    manifest["warnings"].append(f"chapter_number_gaps:{gaps}")

            partial_path.replace(Path(manifest["chunks_path"]))
            if self.auto_index:
                manifest["stage"] = "embedding"
                self._write_manifest(manifest)
                manifest["stage"] = "vector_indexing"
                self._write_manifest(manifest)
                with trace.step("embedding_and_index", metadata={"chunk_count": manifest["chunk_count"]}):
                    manifest["index"] = self._index_chunks(manifest, trace_id=trace.trace_id)
                manifest["stage"] = "keyword_indexing"
                self._write_manifest(manifest)
                final_stage = "ready"
            else:
                final_stage = "chunks_ready"
            final_status = "failed" if failed_sources else "completed"
            manifest.update({
                "status": final_status,
                "stage": "failed" if failed_sources else final_stage,
                "error": f"部分文件处理失败：{', '.join(source['file_name'] for source in failed_sources)}" if failed_sources else None,
                "completed_at": _now(),
            })
            self._sync_novel_status(manifest)
            _atomic_json_write(Path(manifest["report_path"]), {
                "import_id": import_id,
                "status": manifest["status"],
                "warnings": manifest["warnings"],
                "sources": manifest["sources"],
                "chunk_count": manifest["chunk_count"],
            })
            manifest.pop("retry_source_ids", None)
            self._write_manifest(manifest)
            trace.finish(status=manifest["status"], error=manifest.get("error"))
            if manifest["status"] == "failed":
                raise RuntimeError(manifest["error"])
        except Exception as error:  # noqa: BLE001
            manifest["status"] = "failed"
            manifest["stage"] = "failed"
            manifest["error"] = f"{type(error).__name__}: {error}"
            self._sync_novel_status(manifest)
            self._write_manifest(manifest)
            trace.finish(status="failed", error=str(error))
            raise
        finally:
            try:
                lock.rmdir()
            except OSError:
                pass

novel_import_service = NovelImportService()













