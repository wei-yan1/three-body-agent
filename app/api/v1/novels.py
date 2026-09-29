"""Novel import and ingestion APIs."""

from __future__ import annotations

import hashlib
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status

from app.api.v1.auth import current_user
from app.schemas.auth import UserOut
from app.schemas.novel import NovelImportAccepted, NovelImportOut, NovelListItem, NovelRenameRequest, NovelSourceOut
from app.services.novel_import_service import ALLOWED_SUFFIXES, novel_import_service
from app.storage.repositories.storyrole_repository import delete_novel, get_novel, list_novels, rename_novel, upsert_novel

router = APIRouter(prefix="/api/v1/novels", tags=["novels"])


@router.get("/library")
def list_novel_library(user: UserOut = Depends(current_user)) -> list[dict]:
    jobs = {str(item["import_id"]): item for item in novel_import_service.list_jobs(owner_id=user.id)}
    result: list[dict] = []
    for workspace in list_novels(owner_id=user.id):
        novel_id = str(workspace["novel_id"])
        job = jobs.pop(novel_id, None)
        result.append({
            "novel_id": novel_id, "novel_name": workspace["name"],
            "source_kind": workspace.get("source_kind", "imported"),
            "status": workspace["status"],
            "stage": "ready" if workspace["status"] == "ready" else (job or {}).get("stage", workspace["status"]),
            "total_files": (job or {}).get("total_files", 0),
            "processed_files": (job or {}).get("processed_files", 0),
            "chunk_count": (job or {}).get("chunk_count", 0),
            "created_at": workspace["created_at"], "updated_at": workspace["updated_at"],
        })
    for job in jobs.values():
        result.append({
            "novel_id": job["import_id"], "novel_name": job["novel_name"],
            "source_kind": "imported", "status": job["status"], "stage": job.get("stage", "uploaded"),
            "total_files": job.get("total_files", 0), "processed_files": job.get("processed_files", 0),
            "chunk_count": job.get("chunk_count", 0), "created_at": job["created_at"], "updated_at": job["updated_at"],
        })
    return sorted(result, key=lambda item: item.get("updated_at", ""), reverse=True)


@router.patch("/{novel_id}")
def rename_novel_item(novel_id: str, payload: NovelRenameRequest, user: UserOut = Depends(current_user)) -> dict:
    name = payload.novel_name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="小说名称不能为空")
    workspace = get_novel(novel_id=novel_id, owner_id=user.id)
    try:
        job = novel_import_service.get_job(novel_id, owner_id=user.id)
    except FileNotFoundError:
        job = None
    except PermissionError as error:
        raise HTTPException(status_code=403, detail="无权访问该小说") from error
    if not workspace and not job:
        raise HTTPException(status_code=404, detail="小说档案不存在")
    if job:
        novel_import_service.rename_job(novel_id, owner_id=user.id, novel_name=name)
    updated = rename_novel(novel_id=novel_id, owner_id=user.id, name=name)
    if not updated:
        updated = upsert_novel(novel_id=novel_id, owner_id=user.id, name=name, status=(job or {}).get("stage", "ready"))
    return {"novel_id": novel_id, "novel_name": updated["name"]}


@router.delete("/{novel_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_novel_item(novel_id: str, user: UserOut = Depends(current_user)) -> None:
    workspace = get_novel(novel_id=novel_id, owner_id=user.id)
    try:
        job = novel_import_service.get_job(novel_id, owner_id=user.id)
    except FileNotFoundError:
        job = None
    except PermissionError as error:
        raise HTTPException(status_code=403, detail="无权访问该小说") from error
    if not workspace and not job:
        raise HTTPException(status_code=404, detail="小说档案不存在")
    if job:
        try:
            novel_import_service.delete_job(novel_id, owner_id=user.id)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
    delete_novel(novel_id=novel_id, owner_id=user.id)


def _to_import_out(manifest: dict) -> NovelImportOut:
    return NovelImportOut(
        import_id=manifest["import_id"], novel_name=manifest["novel_name"], status=manifest["status"], stage=manifest.get("stage", "uploaded"),
        total_files=manifest["total_files"], processed_files=manifest.get("processed_files", 0), total_bytes=manifest["total_bytes"], chunk_count=manifest.get("chunk_count", 0),
        index=manifest.get("index"),
        warnings=manifest.get("warnings", []), error=manifest.get("error"), sources=[NovelSourceOut(**source) for source in manifest.get("sources", [])],
        chunks_path=manifest.get("chunks_path"), report_path=manifest.get("report_path"), started_at=manifest.get("started_at"), completed_at=manifest.get("completed_at"),
        created_at=manifest["created_at"], updated_at=manifest["updated_at"],
    )


async def _stage_uploads(files: list[UploadFile]) -> tuple[Path, list[dict]]:
    staging = novel_import_service.root / ".staging" / uuid.uuid4().hex
    staging.mkdir(parents=True, exist_ok=False)
    staged: list[dict] = []
    total_bytes = 0
    try:
        for index, upload in enumerate(files):
            original_name = upload.filename or f"source_{index}.txt"
            suffix = Path(original_name).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise HTTPException(status_code=415, detail=f"不支持的文本格式：{original_name}")
            target = staging / f"{index:04d}_{Path(original_name).name}"
            digest = hashlib.sha256()
            size = 0
            with target.open("wb") as output:
                while True:
                    block = await upload.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    total_bytes += len(block)
                    if size > novel_import_service.max_file_bytes:
                        raise HTTPException(status_code=413, detail=f"文件 {original_name} 超过单文件大小限制")
                    if total_bytes > novel_import_service.max_total_bytes:
                        raise HTTPException(status_code=413, detail="单次上传总大小超过限制")
                    digest.update(block)
                    output.write(block)
            staged.append({"path": str(target), "original_name": original_name, "size_bytes": size, "sha256": digest.hexdigest()})
        return staging, staged
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


@router.post("/imports", response_model=NovelImportAccepted, status_code=status.HTTP_202_ACCEPTED)
async def import_novel(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...), novel_name: str = Form("未命名小说"), user: UserOut = Depends(current_user)) -> NovelImportAccepted:
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一个文本文件")
    if len(files) > novel_import_service.max_files:
        raise HTTPException(status_code=400, detail=f"单次最多上传 {novel_import_service.max_files} 个文件")
    staging, staged = await _stage_uploads(files)
    try:
        manifest = novel_import_service.create_job_from_staged_files(novel_name=novel_name, owner_id=user.id, files=staged)
    except ValueError as error:
        shutil.rmtree(staging, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(error)) from error
    shutil.rmtree(staging, ignore_errors=True)
    try:
        upsert_novel(
            novel_id=manifest["import_id"],
            owner_id=user.id,
            name=manifest["novel_name"],
            status="processing", source_kind="imported",
            chunks_path=manifest.get("chunks_path"),
        )
    except Exception as error:
        # Do not leave a file job running without a durable workspace row.
        shutil.rmtree(novel_import_service.root / manifest["import_id"], ignore_errors=True)
        raise HTTPException(status_code=503, detail="数据库暂时不可用，无法创建小说工作区") from error
    background_tasks.add_task(novel_import_service.process_job, manifest["import_id"])
    return NovelImportAccepted(import_id=manifest["import_id"], novel_name=manifest["novel_name"], status=manifest["status"], total_files=manifest["total_files"], total_bytes=manifest["total_bytes"], status_url=f"/api/v1/novels/imports/{manifest['import_id']}")


@router.get("/imports", response_model=list[NovelListItem])
def list_imports(user: UserOut = Depends(current_user)) -> list[NovelListItem]:
    return [NovelListItem(import_id=item["import_id"], novel_name=item["novel_name"], status=item["status"], stage=item.get("stage", "uploaded"), total_files=item["total_files"], processed_files=item.get("processed_files", 0), chunk_count=item.get("chunk_count", 0), created_at=item["created_at"], updated_at=item["updated_at"]) for item in novel_import_service.list_jobs(owner_id=user.id)]


@router.get("/imports/{import_id}", response_model=NovelImportOut)
def get_import(import_id: str, user: UserOut = Depends(current_user)) -> NovelImportOut:
    try:
        return _to_import_out(novel_import_service.get_job(import_id, owner_id=user.id))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="导入任务不存在") from error
    except PermissionError as error:
        raise HTTPException(status_code=403, detail="无权访问该导入任务") from error


@router.post("/imports/{import_id}/retry", response_model=NovelImportAccepted, status_code=status.HTTP_202_ACCEPTED)
def retry_import(import_id: str, background_tasks: BackgroundTasks, user: UserOut = Depends(current_user)) -> NovelImportAccepted:
    try:
        manifest = novel_import_service.retry_job(import_id, owner_id=user.id)
    except (FileNotFoundError, PermissionError) as error:
        raise HTTPException(status_code=404 if isinstance(error, FileNotFoundError) else 403, detail="导入任务不可用") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    background_tasks.add_task(novel_import_service.process_job, import_id)
    return NovelImportAccepted(import_id=import_id, novel_name=manifest["novel_name"], status=manifest["status"], total_files=manifest["total_files"], total_bytes=manifest["total_bytes"], status_url=f"/api/v1/novels/imports/{import_id}")


@router.post("/imports/{import_id}/cancel", response_model=NovelImportOut)
def cancel_import(import_id: str, user: UserOut = Depends(current_user)) -> NovelImportOut:
    try:
        return _to_import_out(novel_import_service.cancel_job(import_id, owner_id=user.id))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="导入任务不存在") from error
    except PermissionError as error:
        raise HTTPException(status_code=403, detail="无权访问该导入任务") from error


