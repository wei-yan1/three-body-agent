"""Chat API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.v1.auth import current_user
from app.schemas.auth import UserOut
from app.schemas.chat import (
    ChatMessageOut,
    ChatThreadOut,
    DeleteThreadsRequest,
    DeleteThreadsResponse,
    KnowledgeMode,
    LuoJiChatRequest,
    LuoJiChatResponse,
)
from app.services.chat_service import chat_with_luoji
from app.services.chat_service import chat_with_zhangbeihai
from app.storage.repositories.session_repository import (
    append_message,
    delete_threads,
    get_or_create_thread,
    get_recent_messages,
    get_thread_messages,
    list_threads,
)


router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.get("/luoji/threads", response_model=list[ChatThreadOut])
def luoji_threads(
    timeline_stage: str,
    knowledge_mode: KnowledgeMode = "temporal",
    user: UserOut = Depends(current_user),
) -> list[ChatThreadOut]:
    threads = list_threads(
        user=user,
        character="罗辑",
        timeline_stage=timeline_stage,
        mode=knowledge_mode,
    )
    return [
        ChatThreadOut(
            id=int(thread["id"]),
            thread_name=str(thread["thread_name"]),
            timeline_stage=str(thread["timeline_stage"]),
            knowledge_mode=thread["mode"],
        )
        for thread in threads
    ]


@router.post("/luoji/threads/delete", response_model=DeleteThreadsResponse)
def delete_luoji_threads(
    payload: DeleteThreadsRequest,
    user: UserOut = Depends(current_user),
) -> DeleteThreadsResponse:
    deleted_count = delete_threads(user=user, thread_ids=payload.thread_ids)
    return DeleteThreadsResponse(deleted_count=deleted_count)


@router.get("/luoji/threads/{thread_id}/messages", response_model=list[ChatMessageOut])
def luoji_thread_messages(
    thread_id: int,
    user: UserOut = Depends(current_user),
) -> list[ChatMessageOut]:
    messages = get_thread_messages(user=user, thread_id=thread_id, limit=20)
    return [
        ChatMessageOut(
            role=str(message["role"]),
            content=str(message["content"]),
            created_at=str(message["created_at"]),
        )
        for message in messages
    ]


@router.post("/luoji", response_model=LuoJiChatResponse)
def chat_luoji(
    payload: LuoJiChatRequest,
    user: UserOut = Depends(current_user),
) -> LuoJiChatResponse:
    thread = get_or_create_thread(
        user=user,
        character="罗辑",
        timeline_stage=payload.timeline_stage,
        mode=payload.knowledge_mode,
        thread_name=payload.thread_name,
    )
    user_message = payload.message.strip()
    metadata = {
        "character": "罗辑",
        "timeline_stage": payload.timeline_stage,
        "knowledge_mode": payload.knowledge_mode,
        "thread_name": payload.thread_name,
    }
    append_message(
        thread=thread,
        role="user",
        content=user_message,
        metadata=metadata,
    )
    recent_messages = get_recent_messages(thread)

    try:
        answer = chat_with_luoji(
            user=user,
            thread=thread,
            timeline_stage=payload.timeline_stage,
            knowledge_mode=payload.knowledge_mode,
            message=user_message,
            recent_messages=recent_messages,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"罗辑 Agent 调用失败：{error}") from error

    append_message(
        thread=thread,
        role="assistant",
        content=answer,
        metadata=metadata,
    )

    return LuoJiChatResponse(
        timeline_stage=payload.timeline_stage,
        knowledge_mode=payload.knowledge_mode,
        thread_id=int(thread["id"]),
        thread_name=str(thread["thread_name"]),
        answer=answer,
    )


@router.get("/zhangbeihai/threads", response_model=list[ChatThreadOut])
def zhangbeihai_threads(
    timeline_stage: str,
    knowledge_mode: KnowledgeMode = "temporal",
    user: UserOut = Depends(current_user),
) -> list[ChatThreadOut]:
    threads = list_threads(
        user=user,
        character="章北海",
        timeline_stage=timeline_stage,
        mode=knowledge_mode,
    )
    return [
        ChatThreadOut(
            id=int(thread["id"]),
            thread_name=str(thread["thread_name"]),
            timeline_stage=str(thread["timeline_stage"]),
            knowledge_mode=thread["mode"],
        )
        for thread in threads
    ]


@router.post("/zhangbeihai/threads/delete", response_model=DeleteThreadsResponse)
def delete_zhangbeihai_threads(
    payload: DeleteThreadsRequest,
    user: UserOut = Depends(current_user),
) -> DeleteThreadsResponse:
    deleted_count = delete_threads(user=user, thread_ids=payload.thread_ids)
    return DeleteThreadsResponse(deleted_count=deleted_count)


@router.get("/zhangbeihai/threads/{thread_id}/messages", response_model=list[ChatMessageOut])
def zhangbeihai_thread_messages(
    thread_id: int,
    user: UserOut = Depends(current_user),
) -> list[ChatMessageOut]:
    messages = get_thread_messages(user=user, thread_id=thread_id, limit=20)
    return [
        ChatMessageOut(
            role=str(message["role"]),
            content=str(message["content"]),
            created_at=str(message["created_at"]),
        )
        for message in messages
    ]


@router.post("/zhangbeihai", response_model=LuoJiChatResponse)
def chat_zhangbeihai(
    payload: LuoJiChatRequest,
    user: UserOut = Depends(current_user),
) -> LuoJiChatResponse:
    thread = get_or_create_thread(
        user=user,
        character="章北海",
        timeline_stage=payload.timeline_stage,
        mode=payload.knowledge_mode,
        thread_name=payload.thread_name,
    )
    user_message = payload.message.strip()
    metadata = {
        "character": "章北海",
        "timeline_stage": payload.timeline_stage,
        "knowledge_mode": payload.knowledge_mode,
        "thread_name": payload.thread_name,
    }
    append_message(
        thread=thread,
        role="user",
        content=user_message,
        metadata=metadata,
    )
    recent_messages = get_recent_messages(thread)

    try:
        answer = chat_with_zhangbeihai(
            user=user,
            thread=thread,
            timeline_stage=payload.timeline_stage,
            knowledge_mode=payload.knowledge_mode,
            message=user_message,
            recent_messages=recent_messages,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"章北海 Agent 调用失败：{error}") from error

    append_message(
        thread=thread,
        role="assistant",
        content=answer,
        metadata=metadata,
    )

    return LuoJiChatResponse(
        timeline_stage=payload.timeline_stage,
        knowledge_mode=payload.knowledge_mode,
        thread_id=int(thread["id"]),
        thread_name=str(thread["thread_name"]),
        answer=answer,
    )
