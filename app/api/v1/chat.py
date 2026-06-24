"""Chat API routes."""

from __future__ import annotations

from collections.abc import Callable

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
from app.services.chat_service import chat_with_wangmiao
from app.services.chat_service import chat_with_yewenjie
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


AgentRunner = Callable[..., str]


AGENTS: dict[str, tuple[str, AgentRunner]] = {
    "luoji": ("罗辑", chat_with_luoji),
    "zhangbeihai": ("章北海", chat_with_zhangbeihai),
    "wangmiao": ("汪淼", chat_with_wangmiao),
    "yewenjie": ("叶文洁", chat_with_yewenjie),
}


def _agent_config(agent_slug: str) -> tuple[str, AgentRunner]:
    try:
        return AGENTS[agent_slug]
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Agent not found") from error


def _threads_response(
    *,
    character: str,
    timeline_stage: str,
    knowledge_mode: KnowledgeMode,
    user: UserOut,
) -> list[ChatThreadOut]:
    threads = list_threads(
        user=user,
        character=character,
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


def _messages_response(*, thread_id: int, user: UserOut) -> list[ChatMessageOut]:
    messages = get_thread_messages(user=user, thread_id=thread_id, limit=20)
    return [
        ChatMessageOut(
            role=str(message["role"]),
            content=str(message["content"]),
            created_at=str(message["created_at"]),
        )
        for message in messages
    ]


def _chat_response(
    *,
    character: str,
    runner: AgentRunner,
    payload: LuoJiChatRequest,
    user: UserOut,
) -> LuoJiChatResponse:
    thread = get_or_create_thread(
        user=user,
        character=character,
        timeline_stage=payload.timeline_stage,
        mode=payload.knowledge_mode,
        thread_name=payload.thread_name,
    )
    user_message = payload.message.strip()
    metadata = {
        "character": character,
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
        answer = runner(
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
        raise HTTPException(status_code=500, detail=f"{character} Agent 调用失败：{error}") from error

    append_message(
        thread=thread,
        role="assistant",
        content=answer,
        metadata=metadata,
    )

    return LuoJiChatResponse(
        character=character,
        timeline_stage=payload.timeline_stage,
        knowledge_mode=payload.knowledge_mode,
        thread_id=int(thread["id"]),
        thread_name=str(thread["thread_name"]),
        answer=answer,
    )


@router.get("/{agent_slug}/threads", response_model=list[ChatThreadOut])
def agent_threads(
    agent_slug: str,
    timeline_stage: str,
    knowledge_mode: KnowledgeMode = "temporal",
    user: UserOut = Depends(current_user),
) -> list[ChatThreadOut]:
    character, _runner = _agent_config(agent_slug)
    return _threads_response(
        character=character,
        timeline_stage=timeline_stage,
        knowledge_mode=knowledge_mode,
        user=user,
    )


@router.post("/{agent_slug}/threads/delete", response_model=DeleteThreadsResponse)
def delete_agent_threads(
    agent_slug: str,
    payload: DeleteThreadsRequest,
    user: UserOut = Depends(current_user),
) -> DeleteThreadsResponse:
    _agent_config(agent_slug)
    deleted_count = delete_threads(user=user, thread_ids=payload.thread_ids)
    return DeleteThreadsResponse(deleted_count=deleted_count)


@router.get("/{agent_slug}/threads/{thread_id}/messages", response_model=list[ChatMessageOut])
def agent_thread_messages(
    agent_slug: str,
    thread_id: int,
    user: UserOut = Depends(current_user),
) -> list[ChatMessageOut]:
    _agent_config(agent_slug)
    return _messages_response(thread_id=thread_id, user=user)


@router.post("/{agent_slug}", response_model=LuoJiChatResponse)
def chat_agent(
    agent_slug: str,
    payload: LuoJiChatRequest,
    user: UserOut = Depends(current_user),
) -> LuoJiChatResponse:
    character, runner = _agent_config(agent_slug)
    return _chat_response(
        character=character,
        runner=runner,
        payload=payload,
        user=user,
    )
