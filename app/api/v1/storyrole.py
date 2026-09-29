"""StoryRole dynamic character API."""

from __future__ import annotations

from pathlib import Path
import logging
import asyncio
import json
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.agents.a2a.registry import client
from app.agents.orchestration.workflow import storyrole_setup_workflow
from app.api.v1.auth import current_user
from app.schemas.auth import UserOut
from app.schemas.character import (
    CharacterChatRequest, CharacterPeriodConfirmRequest, CharacterResolveRequest,
)
from app.storage.repositories.session_repository import (
    append_message,
    get_or_create_thread,
    get_recent_messages,
    get_thread_messages_for_scope,
)
from app.services.novel_import_service import novel_import_service
from app.services.bundled_storyrole_service import ensure_three_body_workspace
from app.storage.repositories.storyrole_repository import get_novel
from app.observability import TraceContext, trace_report
from app.agents.storyrole.period_agent import collect_character_records, evidence_for_period
from app.storage.repositories.storyrole_repository import (
    get_character,
    get_character_period,
    get_latest_period_analysis,
    get_persona_profile_for_period,
    get_latest_persona_profile,
    list_character_periods,
    upsert_character_period,
    get_relationship_state,
    list_characters,
    list_memory_candidates,
    mark_memory_candidate_persisted,
    review_memory_candidate,
)
from app.storage.postgres.client import postgres_connection

router = APIRouter(prefix="/api/v1/storyrole", tags=["storyrole"])
logger = logging.getLogger("storyrole")


def _latest_chat_trace(owner_id: int) -> dict | None:
    try:
        with postgres_connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT trace_id,status,metadata,started_at,ended_at,duration_ms
                FROM pipeline_runs WHERE run_type='storyrole_chat'
                AND (metadata->>'owner_id')::bigint=%s
                ORDER BY started_at DESC LIMIT 1""",
                (owner_id,),
            )
            return cursor.fetchone()
    except Exception:
        return None


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


def _trace_progress(trace_id: str | None, owner_id: int) -> dict:
    if not trace_id:
        return {}
    try:
        with postgres_connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """SELECT step_name,status,duration_ms,started_at,ended_at
                FROM pipeline_steps WHERE trace_id=%s ORDER BY started_at,id""",
                (trace_id,),
            )
            steps = list(cursor.fetchall())
            cursor.execute(
                """SELECT status,duration_ms FROM pipeline_runs
                WHERE trace_id=%s AND (metadata->>'owner_id')::bigint=%s""",
                (trace_id, owner_id),
            )
            run = cursor.fetchone() or {}
            current = next((step for step in reversed(steps) if step["status"] == "running"), None)
            return {
                "trace_id": trace_id,
                "status": run.get("status", "running"),
                "current_step": current["step_name"] if current else (steps[-1]["step_name"] if steps else None),
                "steps": steps,
            }
    except Exception:
        return {"trace_id": trace_id}


def _finish_api_trace(result: dict, trace: TraceContext, user_id: int) -> dict:
    """Attach only this API request's observability data to its response."""
    trace.finish()
    report = trace_report(trace.trace_id, owner_id=user_id) or {
        "trace_id": trace.trace_id, "summary": {}, "agents": [], "steps": [],
        "model_invocations": [],
    }
    return {
        **result,
        "trace_id": trace.trace_id,
        "trace_summary": report["summary"],
        "trace_agents": report["agents"],
        "trace_steps": report["steps"],
        "trace_model_invocations": report["model_invocations"],
    }


@router.post("/bundled/three-body/activate")
def activate_three_body(user: UserOut = Depends(current_user)) -> dict:
    try:
        return ensure_three_body_workspace(user.id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="内置三体资源不存在") from error


@router.post("/novels/{novel_id}/characters/resolve")
async def resolve_character(novel_id: str, payload: CharacterResolveRequest, user: UserOut = Depends(current_user)) -> dict:
    trace = TraceContext(run_type="character_resolve", metadata={
        "owner_id": user.id, "novel_id": novel_id, "endpoint": "characters/resolve",
        "character_name": payload.character_name,
    })
    trace.start()
    try:
        result = await client.send("character-resolver", {
            "trace_id": trace.trace_id, "novel_id": novel_id,
            "owner_id": user.id, "character_name": payload.character_name,
        })
        return _finish_api_trace(result, trace, user.id)
    except FileNotFoundError as error:
        trace.finish(status="failed", error=str(error))
        raise HTTPException(status_code=404, detail="小说不存在") from error
    except PermissionError as error:
        trace.finish(status="failed", error=str(error))
        raise HTTPException(status_code=403, detail="无权访问该小说") from error
    except ValueError as error:
        trace.finish(status="failed", error=str(error))
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/novels/{novel_id}/characters")
def characters(novel_id: str, user: UserOut = Depends(current_user)) -> list[dict]:
    return list_characters(novel_id=novel_id, owner_id=user.id)


@router.post("/novels/{novel_id}/characters/setup")
async def setup_character(
    novel_id: str, payload: CharacterResolveRequest, user: UserOut = Depends(current_user)
) -> dict:
    """Resolve a character, hand evidence to Nuwa, and persist its profile."""
    trace = TraceContext(run_type="character_setup", metadata={
        "owner_id": user.id, "novel_id": novel_id, "endpoint": "characters/setup",
        "character_name": payload.character_name,
    })
    trace.start()
    try:
        result = await storyrole_setup_workflow.setup_character(
            novel_id=novel_id, owner_id=user.id, character_name=payload.character_name,
            trace_id=trace.trace_id,
        )
        return _finish_api_trace(result, trace, user.id)
    except FileNotFoundError as error:
        trace.finish(status="failed", error=str(error))
        raise HTTPException(status_code=404, detail="小说不存在") from error
    except PermissionError as error:
        trace.finish(status="failed", error=str(error))
        raise HTTPException(status_code=403, detail="无权访问该小说") from error
    except ValueError as error:
        trace.finish(status="failed", error=str(error))
        raise HTTPException(status_code=409, detail=str(error)) from error



@router.post("/novels/{novel_id}/characters/{character_id}/period-analysis")
async def analyze_character_periods(
    novel_id: str, character_id: int, payload: dict | None = None, user: UserOut = Depends(current_user),
) -> dict:
    character = get_character(character_id=character_id, owner_id=user.id)
    if not character or character["novel_id"] != novel_id:
        raise HTTPException(status_code=404, detail="角色不存在")
    trace = TraceContext(run_type="period_analysis", metadata={
        "owner_id": user.id, "novel_id": novel_id, "character_id": character_id,
        "endpoint": "characters/period-analysis",
    })
    trace.start()
    try:
        result = await client.send("character-period-analysis", {
            "trace_id": trace.trace_id, "novel_id": novel_id,
            "character_id": character_id, "owner_id": user.id,
            "period_hints": (payload or {}).get("period_hints", []),
        })
        return _finish_api_trace(result, trace, user.id)
    except ValueError as error:
        trace.finish(status="failed", error=str(error))
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/novels/{novel_id}/characters/{character_id}/period-analysis")
def get_character_period_analysis(
    novel_id: str, character_id: int, user: UserOut = Depends(current_user),
) -> dict:
    character = get_character(character_id=character_id, owner_id=user.id)
    if not character or character["novel_id"] != novel_id:
        raise HTTPException(status_code=404, detail="角色不存在")
    row = get_latest_period_analysis(character_id=character_id, owner_id=user.id)
    if not row:
        raise HTTPException(status_code=404, detail="尚未进行时期分析")
    return {**row, **(row.get("result") or {})}


@router.post("/novels/{novel_id}/characters/{character_id}/periods/confirm")
def confirm_character_periods(
    novel_id: str, character_id: int, payload: CharacterPeriodConfirmRequest,
    user: UserOut = Depends(current_user),
) -> dict:
    character = get_character(character_id=character_id, owner_id=user.id)
    if not character or character["novel_id"] != novel_id:
        raise HTTPException(status_code=404, detail="角色不存在")
    periods = []
    for item in payload.periods:
        row = upsert_character_period(
            character_id=character_id, novel_id=novel_id, owner_id=user.id,
            name=item.name, chapter_start=item.chapter_start, chapter_end=item.chapter_end,
            keywords=item.keywords, evidence_ids=item.evidence_ids,
            rationale=item.rationale, confidence=item.confidence, status="confirmed",
        )
        periods.append(row)
    return {"character_id": character_id, "periods": periods}


@router.get("/novels/{novel_id}/characters/{character_id}/periods")
def get_character_periods(
    novel_id: str, character_id: int, user: UserOut = Depends(current_user),
) -> list[dict]:
    character = get_character(character_id=character_id, owner_id=user.id)
    if not character or character["novel_id"] != novel_id:
        raise HTTPException(status_code=404, detail="角色不存在")
    periods = list_character_periods(character_id=character_id, owner_id=user.id)
    if not periods:
        periods = [upsert_character_period(
            character_id=character_id, novel_id=novel_id, owner_id=user.id,
            name="全时期", chapter_start=None, chapter_end=None, keywords=[],
            evidence_ids=[], rationale="用户可自行修改时期名称", confidence=0.0,
            status="draft",
        )]
    return periods


@router.post("/novels/{novel_id}/characters/{character_id}/periods/{period_id}/profile")
async def profile_character_period(
    novel_id: str, character_id: int, period_id: int, user: UserOut = Depends(current_user),
) -> dict:
    logger.info(
        "period profile started novel_id=%s character_id=%s period_id=%s user_id=%s",
        novel_id, character_id, period_id, user.id,
    )
    character = get_character(character_id=character_id, owner_id=user.id)
    period = get_character_period(period_id=period_id, owner_id=user.id)
    if not character or character["novel_id"] != novel_id or not period or int(period["character_id"]) != character_id:
        raise HTTPException(status_code=404, detail="角色时期不存在")
    workspace = get_novel(novel_id=novel_id, owner_id=user.id)
    manifest = workspace if workspace and workspace.get("source_kind") == "bundled" else novel_import_service.get_job(novel_id, owner_id=user.id)
    trace = TraceContext(run_type="persona_profile", metadata={
        "owner_id": user.id, "novel_id": novel_id, "character_id": character_id,
        "period_id": period_id, "endpoint": "periods/profile",
    })
    trace.start()
    try:
        with trace.step("evidence_collection", metadata={"period_id": period_id}):
            records = collect_character_records(
                Path(str(manifest["chunks_path"])), str(character["canonical_name"])
            )
            evidence = evidence_for_period(records, period)
        result = await client.send("nuwa-profiler", {
            "trace_id": trace.trace_id,
            "novel_id": novel_id, "owner_id": user.id, "character_id": character_id,
            "character_name": character["canonical_name"],
            "novel_name": manifest.get("novel_name", "未知小说"),
            "period_id": period_id, "period_name": period["name"],
            "evidence": evidence,
        })
        logger.info(
            "period profile completed character_id=%s period_id=%s profile_id=%s model=%s",
            character_id, period_id, result.get("profile_id"), result.get("model"),
        )
        return _finish_api_trace(result, trace, user.id)
    except FileNotFoundError as error:
        trace.finish(status="failed", error=str(error))
        raise HTTPException(status_code=404, detail="小说证据不存在") from error
    except ValueError as error:
        trace.finish(status="failed", error=str(error))
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        trace.finish(status="failed", error=str(error))
        raise


@router.post("/novels/{novel_id}/characters/{character_id}/profile")
async def profile_character(novel_id: str, character_id: int, user: UserOut = Depends(current_user)) -> dict:
    character = get_character(character_id=character_id, owner_id=user.id)
    if not character or character["novel_id"] != novel_id:
        raise HTTPException(status_code=404, detail="角色不存在")
    trace = TraceContext(run_type="persona_profile", metadata={
        "owner_id": user.id, "novel_id": novel_id, "character_id": character_id,
        "endpoint": "profile",
    })
    trace.start()
    try:
        # The period profile endpoint performs the explicit Nuwa call. This endpoint
        # refreshes character resolution and returns the exact resolver trace.
        result = await storyrole_setup_workflow.setup_character(
            novel_id=novel_id, owner_id=user.id,
            character_name=character["canonical_name"], trace_id=trace.trace_id,
        )
        return _finish_api_trace(result, trace, user.id)
    except Exception as error:
        trace.finish(status="failed", error=str(error))
        raise


@router.get("/novels/{novel_id}/characters/{character_id}/profile")
def get_profile(novel_id: str, character_id: int, user: UserOut = Depends(current_user)) -> dict:
    character = get_character(character_id=character_id, owner_id=user.id)
    if not character or character["novel_id"] != novel_id:
        raise HTTPException(status_code=404, detail="角色不存在")
    profile = get_latest_persona_profile(character_id=character_id, owner_id=user.id)
    if not profile:
        raise HTTPException(status_code=404, detail="角色尚未完成女娲分析")
    return profile




@router.get("/novels/{novel_id}/characters/{character_id}/relationship")
def get_relationship(novel_id: str, character_id: int, user: UserOut = Depends(current_user)) -> dict:
    character = get_character(character_id=character_id, owner_id=user.id)
    if not character or character["novel_id"] != novel_id:
        raise HTTPException(status_code=404, detail="角色不存在")
    state = get_relationship_state(
        character_id=character_id, owner_id=user.id,
        target_type="user", target_key=str(user.id),
    )
    return state or {
        "character_id": character_id,
        "target_type": "user",
        "target_key": str(user.id),
        "trust": 0.5,
        "intimacy": 0.0,
        "tension": 0.0,
        "dependency": 0.0,
        "state": {},
        "evidence": [],
    }


@router.post("/novels/{novel_id}/characters/{character_id}/chat")
async def chat_with_character(novel_id: str, character_id: int, payload: CharacterChatRequest, user: UserOut = Depends(current_user)) -> dict:
    character = get_character(character_id=character_id, owner_id=user.id)
    if not character or character["novel_id"] != novel_id:
        raise HTTPException(status_code=404, detail="角色不存在")
    if not get_persona_profile_for_period(
        character_id=character_id, owner_id=user.id, period_id=payload.period_id
    ):
        raise HTTPException(status_code=409, detail="请先完成对应时期的女娲角色分析")
    thread = get_or_create_thread(
        user=user, character=str(character["canonical_name"]), timeline_stage="dynamic",
        mode="storyrole", thread_name=payload.thread_name, novel_id=novel_id,
        character_id=character_id, period_id=payload.period_id,
    )
    trace = TraceContext(run_type="storyrole_chat", metadata={"owner_id": user.id, "novel_id": novel_id, "character_id": character_id, "mode": payload.mode})
    trace.start()
    try:
        answer = await client.send("character-conversation", {
            "trace_id": trace.trace_id,
            "novel_id": novel_id,
            "owner_id": user.id,
            "character_id": character_id,
            "period_id": payload.period_id,
            "message": payload.message,
            "recent_messages": get_recent_messages(thread),
            "mode": payload.mode,
            "deep_reasoning": payload.deep_reasoning,
            "user": user.model_dump(),
            "thread": thread,
        })
    except Exception as error:
        trace.finish(status="failed", error=str(error))
        raise
    trace.finish()
    trace_report_value = trace_report(trace.trace_id, owner_id=user.id) or {}
    trace_run = trace_report_value.get("summary") or {}
    # A failed request never reaches here, so only completed turns enter the
    # durable transcript and the next model context.
    append_message(thread=thread, role="user", content=payload.message, metadata={"novel_id": novel_id, "character_id": character_id, "period_id": payload.period_id, "retry": payload.retry, "status": "completed"})
    append_message(thread=thread, role="assistant", content=answer["answer"], metadata={"novel_id": novel_id, "character_id": character_id, "period_id": payload.period_id})
    return {
        "novel_id": novel_id,
        "character_id": character_id,
        "character": character["canonical_name"],
        "thread_id": int(thread["id"]),
        "thread_name": thread["thread_name"],
        "answer": answer["answer"],
        "profile_version": answer["profile_version"],
        "period_id": payload.period_id,
        "mode": answer.get("mode", payload.mode),
        "evidence_count": answer.get("evidence_count", 0),
        "latency_ms": answer.get("latency_ms"),
        "trace_id": trace.trace_id if payload.debug else None,
        "degraded": bool(answer.get("degraded")),
        "degradations": answer.get("degradations", []),
        "execution_plan": answer.get("execution_plan"),
        "evidence_graph_trace_id": answer.get("evidence_graph_trace_id"),
        "web_mode": answer.get("web_mode", payload.web_mode),
        "web_evidence_count": answer.get("web_evidence_count", 0),
        "trace_summary": {
            "duration_ms": trace_run.get("duration_ms"),
            "input_tokens": trace_run.get("input_tokens"),
            "output_tokens": trace_run.get("output_tokens"),
            "total_tokens": trace_run.get("total_tokens"),
            "model_call_count": trace_run.get("model_call_count", 0),
            "usage_complete": bool(trace_run.get("usage_complete")),
        },
        "trace_agents": trace_report_value.get("agents", []),
        "trace_steps": trace_report_value.get("steps", []),
        "trace_model_invocations": trace_report_value.get("model_invocations", []),
    }


@router.post("/novels/{novel_id}/characters/{character_id}/chat/stream")
async def chat_with_character_stream(
    novel_id: str, character_id: int, payload: CharacterChatRequest,
    user: UserOut = Depends(current_user),
) -> StreamingResponse:
    character = get_character(character_id=character_id, owner_id=user.id)
    if not character or character["novel_id"] != novel_id:
        raise HTTPException(status_code=404, detail="角色不存在")
    profile = get_persona_profile_for_period(
        character_id=character_id, owner_id=user.id, period_id=payload.period_id,
    )
    if not profile:
        raise HTTPException(status_code=409, detail="请先完成对应时期的女娲角色分析")

    from app.agents.storyrole.streaming import close_answer_stream, open_answer_stream

    thread = get_or_create_thread(
        user=user, character=str(character["canonical_name"]), timeline_stage="dynamic",
        mode="storyrole", thread_name=payload.thread_name, novel_id=novel_id,
        character_id=character_id, period_id=payload.period_id,
    )
    trace = TraceContext(run_type="storyrole_chat", metadata={
        "owner_id": user.id, "novel_id": novel_id, "character_id": character_id,
        "period_id": payload.period_id, "mode": payload.mode,
    })
    trace.start()
    answer_stream = open_answer_stream(trace.trace_id)

    async def run_chat() -> dict:
        try:
            answer = await client.send("character-conversation", {
                "trace_id": trace.trace_id, "novel_id": novel_id, "owner_id": user.id,
                "character_id": character_id, "period_id": payload.period_id,
                "message": payload.message, "recent_messages": get_recent_messages(thread),
                "mode": payload.mode, "deep_reasoning": payload.deep_reasoning,
                "web_mode": payload.web_mode, "retry": payload.retry,
                "user": user.model_dump(), "thread": thread,
            })
            trace.finish()
            append_message(
                thread=thread, role="user", content=payload.message,
                metadata={"novel_id": novel_id, "character_id": character_id,
                          "period_id": payload.period_id, "retry": payload.retry,
                          "status": "completed"},
            )
            append_message(
                thread=thread, role="assistant", content=answer["answer"],
                metadata={"novel_id": novel_id, "character_id": character_id, "period_id": payload.period_id},
            )
            report = trace_report(trace.trace_id, owner_id=user.id) or {}
            return {
                **answer, "novel_id": novel_id, "character_id": character_id,
                "character": character["canonical_name"], "thread_id": int(thread["id"]),
                "thread_name": thread["thread_name"], "period_id": payload.period_id,
                "trace_id": trace.trace_id,
                "trace_summary": report.get("summary", {}),
                "trace_agents": report.get("agents", []),
                "trace_steps": report.get("steps", []),
                "trace_model_invocations": report.get("model_invocations", []),
            }
        except asyncio.CancelledError:
            trace.finish(status="cancelled")
            raise
        except Exception as error:
            trace.finish(status="failed", error=str(error))
            raise
        finally:
            close_answer_stream(trace.trace_id)

    task = asyncio.create_task(run_chat())

    async def events():
        try:
            yield _sse("start", {"trace_id": trace.trace_id, "mode": payload.mode})
            last_signature = None
            done_seen = False
            last_heartbeat = asyncio.get_running_loop().time()
            while not (task.done() and done_seen):
                event_task = asyncio.create_task(answer_stream.get())
                done, _ = await asyncio.wait({task, event_task}, timeout=0.35, return_when=asyncio.FIRST_COMPLETED)
                if event_task in done:
                    event = event_task.result()
                    event_type = event.get("type")
                    if event_type == "done":
                        done_seen = True
                    elif event_type in {"token", "reset"}:
                        yield _sse(event_type, {"text": event.get("text", "")})
                else:
                    event_task.cancel()
                    try:
                        await event_task
                    except asyncio.CancelledError:
                        pass

                progress = await asyncio.to_thread(_trace_progress, trace.trace_id, user.id)
                steps = progress.get("steps") or []
                signature = tuple((step["step_name"], step["status"]) for step in steps)
                if signature != last_signature:
                    last_signature = signature
                    current = progress.get("current_step")
                    if current:
                        is_deep = payload.mode == "deep" or current in {
                            "deep_question_planning", "novel_retrieval", "evidence_rerank",
                            "evidence_analysis", "role_cognition", "consistency_review",
                        }
                        yield _sse("progress", {"trace_id": trace.trace_id, "step": current, "deep": is_deep, "steps": steps})
                now = asyncio.get_running_loop().time()
                if now - last_heartbeat >= 10:
                    last_heartbeat = now
                    yield _sse("heartbeat", {"trace_id": trace.trace_id})

            try:
                result = task.result()
            except Exception as error:
                yield _sse("error", {"message": str(error), "trace_id": trace.trace_id})
                return
            yield _sse("complete", result)
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    return StreamingResponse(
        events(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/novels/{novel_id}/characters/{character_id}/chat/history")
def chat_history(
    novel_id: str, character_id: int, period_id: int | None = None,
    thread_name: str = "线程1", user: UserOut = Depends(current_user),
) -> dict:
    character = get_character(character_id=character_id, owner_id=user.id)
    if not character or character["novel_id"] != novel_id:
        raise HTTPException(status_code=404, detail="角色不存在")
    messages = get_thread_messages_for_scope(
        user=user, character=str(character["canonical_name"]), timeline_stage="dynamic",
        mode="storyrole", thread_name=thread_name, novel_id=novel_id,
        character_id=character_id, period_id=period_id,
    )
    return {"novel_id": novel_id, "character_id": character_id, "period_id": period_id, "messages": messages}


@router.get("/memory-candidates")
def get_memory_candidates(status: str = "pending", limit: int = 50, user: UserOut = Depends(current_user)) -> list[dict]:
    return list_memory_candidates(owner_id=user.id, status=status, limit=limit)


@router.post("/memory-candidates/{candidate_id}/review")
def review_candidate(candidate_id: int, status: str, user: UserOut = Depends(current_user)) -> dict:
    try:
        row = review_memory_candidate(candidate_id=candidate_id, owner_id=user.id, status=status)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if not row:
        raise HTTPException(status_code=404, detail="记忆候选不存在或已经处理")
    if status == "approved":
        try:
            character = get_character(character_id=int(row["character_id"]), owner_id=user.id)
            if not character:
                raise HTTPException(status_code=404, detail="角色不存在")
            from app.memory.tools import MemoryTool
            MemoryTool().execute(
                "add", user=user, character=str(character["canonical_name"]),
                timeline_stage="dynamic", mode="storyrole",
                thread_name=str(row.get("thread_name") or "线程1"), thread_id=row.get("thread_id"),
                content=str(row["content"]), summary=str(row["summary"]),
                importance=float(row["importance"]), source_turn_range="reviewed_candidate",
                metadata={"source": "reviewed_memory_candidate", "candidate_id": int(row["id"])},
            )
            mark_memory_candidate_persisted(candidate_id=int(row["id"]), owner_id=user.id)
            row["status"] = "persisted"
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=503, detail="记忆候选已批准，但持久化失败") from error
    return row

