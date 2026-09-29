"""Durable conversation compaction for StoryRole chat threads.

The raw chat transcript remains intact. A checkpoint only records the newest
message already represented by a summary, so later requests can replay the
summary plus the messages after that boundary.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from app.observability import observe_async_call
from app.storage.repositories.session_repository import (
    get_context_checkpoint,
    get_messages_after_checkpoint,
    save_context_checkpoint,
)


@dataclass
class PreparedHistory:
    messages: list[dict[str, Any]]
    summary: str
    compacted: bool = False


def _estimate_tokens(text: str) -> int:
    chinese = sum("\u3400" <= char <= "\u9fff" for char in text)
    other = len(text) - chinese
    return max(1, chinese + (other + 3) // 4)


def _message_tokens(message: dict[str, Any]) -> int:
    return _estimate_tokens(str(message.get("content") or "")) + 4


def _transcript(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{message.get('role', 'user')}: {str(message.get('content') or '')}"
        for message in messages
        if message.get("content")
    )


def _tail(messages: list[dict[str, Any]], *, max_messages: int, budget: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    used = 0
    for message in reversed(messages):
        cost = _message_tokens(message)
        if kept and (len(kept) >= max_messages or used + cost > budget):
            break
        kept.insert(0, message)
        used += cost
    # A checkpoint must end after an assistant answer, never between the user
    # message and its reply. This may retain nine messages instead of ten.
    prefix_length = len(messages) - len(kept)
    if prefix_length % 2 == 1 and kept:
        prefix_length += 1
        kept.pop(0)
    return messages[:prefix_length], messages[prefix_length:]


async def prepare_history(
    *, thread_id: int, fallback_messages: list[dict[str, Any]], trace_id: str | None,
) -> PreparedHistory:
    """Load a checkpointed history and compact it once it reaches its budget."""
    try:
        checkpoint = get_context_checkpoint(thread_id)
        checkpoint_id = int(checkpoint["covered_message_id"]) if checkpoint and checkpoint.get("covered_message_id") else None
        summary = str(checkpoint.get("summary") or "") if checkpoint else ""
        messages = get_messages_after_checkpoint(thread_id, checkpoint_id)
    except Exception:
        return PreparedHistory(messages=fallback_messages, summary="")

    if not messages:
        return PreparedHistory(messages=fallback_messages, summary=summary)

    window = max(1, int(os.getenv("STORYROLE_CONTEXT_WINDOW_TOKENS", "128000")))
    reserve = max(1024, int(os.getenv("STORYROLE_CONTEXT_RESERVE_TOKENS", "16384")))
    threshold = max(1, window - reserve)
    total = _estimate_tokens(summary) + sum(_message_tokens(item) for item in messages)
    if total <= threshold:
        return PreparedHistory(messages=messages, summary=summary)

    prefix, kept = _tail(
        messages,
        max_messages=max(1, int(os.getenv("STORYROLE_HISTORY_RECENT_MESSAGES", "10"))),
        budget=max(1024, int(os.getenv("STORYROLE_CONTEXT_KEEP_RECENT_TOKENS", "16000"))),
    )
    if not prefix or not kept:
        return PreparedHistory(messages=messages, summary=summary)

    previous = summary or "无此前摘要。"
    prompt = f"""请压缩以下角色对话历史，供后续继续扮演角色使用。
保留：用户的重要事实、偏好、承诺、关系变化、未解决问题、角色已经表达的立场。
删除：寒暄、重复内容、无关措辞。不要编造，不要泄露这是摘要。
输出结构化但简洁的中文摘要，控制在 4000 字以内。

此前已保存摘要：
{previous}

需要并入的较早对话：
{_transcript(prefix)}"""
    try:
        from langchain.chat_models import init_chat_model

        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return PreparedHistory(messages=kept, summary=summary)
        os.environ.setdefault("OPENAI_API_KEY", api_key)
        if os.getenv("DASHSCOPE_BASE_URL"):
            os.environ.setdefault("OPENAI_BASE_URL", os.environ["DASHSCOPE_BASE_URL"])
        model = init_chat_model(
            os.getenv("STORYROLE_CHAT_MODEL", os.getenv("TEST_CHAT_MODEL", "qwen3.7-plus")),
            model_provider="openai",
        )
        response = await observe_async_call(
            lambda: model.ainvoke(prompt),
            operation="storyrole_context_compaction",
            trace_id=trace_id,
            model=model,
            prompt=prompt,
        )
        compacted = str(getattr(response, "content", response) or "").strip()
        if not compacted:
            return PreparedHistory(messages=kept, summary=summary)
        save_context_checkpoint(
            thread_id=thread_id,
            covered_message_id=int(prefix[-1]["id"]),
            summary=compacted,
        )
        return PreparedHistory(messages=kept, summary=compacted, compacted=True)
    except Exception:
        # A failed compaction must not break chat or destroy raw history.
        return PreparedHistory(messages=kept, summary=summary)
