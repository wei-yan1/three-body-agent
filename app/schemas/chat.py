"""Chat request and response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


KnowledgeMode = Literal["temporal", "transparent"]


class LuoJiChatRequest(BaseModel):
    timeline_stage: Literal["T0", "T1", "T2", "T3", "T4", "T5", "T6"]
    knowledge_mode: KnowledgeMode = "temporal"
    thread_name: str = Field(default="线程1", max_length=128)
    message: str = Field(min_length=1, max_length=4000)


class LuoJiChatResponse(BaseModel):
    character: str
    timeline_stage: str
    knowledge_mode: KnowledgeMode
    thread_id: int
    thread_name: str
    answer: str


class ChatThreadOut(BaseModel):
    id: int | None = None
    thread_name: str
    timeline_stage: str
    knowledge_mode: KnowledgeMode


class ChatMessageOut(BaseModel):
    role: str
    content: str
    created_at: str


class DeleteThreadsRequest(BaseModel):
    thread_ids: list[int] = Field(default_factory=list)


class DeleteThreadsResponse(BaseModel):
    deleted_count: int
