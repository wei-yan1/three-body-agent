"""Pydantic models for dynamic StoryRole characters."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CharacterResolveRequest(BaseModel):
    character_name: str = Field(min_length=1, max_length=255)


class CharacterCandidate(BaseModel):
    character_name: str
    mentions: int = 0


class CharacterResolveResponse(BaseModel):
    novel_id: str
    found: bool
    character_name: str
    character_id: int | None = None
    confidence: float = 0.0
    mentions: int = 0
    first_chapter: str | None = None
    last_chapter: str | None = None
    candidates: list[CharacterCandidate] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)


class CharacterChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_name: str = Field(default="线程1", max_length=128)
    period_id: int | None = Field(default=None, ge=1)
    mode: Literal["quick", "deep", "auto"] = Field(default="auto", description="回答模式")
    web_mode: Literal["off", "on"] = Field(default="off", description="联网模式")
    # Backward-compatible alias for existing clients. When true, mode becomes deep.
    deep_reasoning: bool = Field(default=False, description="兼容旧客户端的深度推理开关")
    debug: bool = Field(default=False, description="返回请求 trace id，便于故障排查")
    retry: bool = Field(default=False, description="是否为失败消息重试")


class CharacterPeriodSpec(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    chapter_start: str | None = Field(default=None, max_length=255)
    chapter_end: str | None = Field(default=None, max_length=255)
    keywords: list[str] = Field(default_factory=list, max_length=20)
    evidence_ids: list[str] = Field(default_factory=list, max_length=40)
    rationale: str = Field(default="", max_length=2000)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class CharacterPeriodConfirmRequest(BaseModel):
    periods: list[CharacterPeriodSpec] = Field(min_length=1, max_length=20)
