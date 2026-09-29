"""Cross-process answer event channels used by the StoryRole SSE endpoint."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from app.storage.redis.client import redis_client

_local_channels: dict[str, asyncio.Queue[dict[str, str]]] = {}


def _use_redis() -> bool:
    configured = os.getenv("STORYROLE_STREAM_TRANSPORT", "auto").lower()
    if configured in {"redis", "local"}:
        return configured == "redis"
    return os.getenv("STORYROLE_A2A_TRANSPORT", "local").lower() == "http"


def _redis_key(trace_id: str) -> str:
    return f"storyrole:answer:{trace_id}"


@dataclass
class AnswerStream:
    trace_id: str
    redis: bool
    queue: asyncio.Queue[dict[str, str]] | None = None
    cursor: str = "0-0"

    async def get(self) -> dict[str, str]:
        if not self.redis:
            if self.queue is None:
                raise RuntimeError("local answer stream is not initialized")
            return await self.queue.get()
        while True:
            rows = await asyncio.to_thread(
                redis_client().xread,
                {_redis_key(self.trace_id): self.cursor},
                count=50,
                block=1000,
            )
            if not rows:
                continue
            _, entries = rows[0]
            entry_id, fields = entries[0]
            self.cursor = entry_id
            return {str(key): str(value) for key, value in fields.items()}


def open_answer_stream(trace_id: str) -> AnswerStream:
    trace_id = str(trace_id)
    use_redis = _use_redis()
    if not use_redis:
        queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()
        _local_channels[trace_id] = queue
    else:
        try:
            redis_client().delete(_redis_key(trace_id))
        except Exception:
            pass
    return AnswerStream(trace_id=trace_id, redis=use_redis, queue=queue if not use_redis else None)


def register_local_stream(trace_id: str, stream: AnswerStream) -> None:
    if not stream.redis:
        _local_channels.setdefault(str(trace_id), asyncio.Queue())


def publish_answer_event(trace_id: str | None, event_type: str, text: str = "") -> None:
    trace_id = str(trace_id or "")
    if not trace_id:
        return
    fields = {"type": event_type, "text": text}
    if _use_redis():
        redis_client().xadd(_redis_key(trace_id), fields, maxlen=2000, approximate=True)
        redis_client().expire(_redis_key(trace_id), 3600)
        return
    queue = _local_channels.get(trace_id)
    if queue is not None:
        queue.put_nowait(fields)


def publish_answer_token(trace_id: str | None, token: str) -> None:
    if token:
        publish_answer_event(trace_id, "token", token)


def reset_answer_stream(trace_id: str | None) -> None:
    publish_answer_event(trace_id, "reset")


def close_answer_stream(trace_id: str) -> None:
    trace_id = str(trace_id)
    publish_answer_event(trace_id, "done")
    if not _use_redis():
        _local_channels.pop(trace_id, None)
