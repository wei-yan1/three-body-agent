"""A2A agent registry and in-process client.

The in-process client serializes every hop as an A2A message/task. This keeps
local development reliable while the same handlers remain callable through the
HTTP JSON-RPC endpoint exposed by the API.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .protocol import A2AAgentCard, A2AMessage, A2ATask
from app.observability import TraceContext, observe_async_call

Handler = Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]


@dataclass
class RegisteredAgent:
    card: A2AAgentCard
    handler: Handler


class A2ARegistry:
    def __init__(self) -> None:
        self._agents: dict[str, RegisteredAgent] = {}
        self._tasks: OrderedDict[str, A2ATask] = OrderedDict()
        self._max_tasks = 1000

    def register(self, agent_id: str, card: A2AAgentCard, handler: Handler) -> None:
        self._agents[agent_id] = RegisteredAgent(card=card, handler=handler)

    def get(self, agent_id: str) -> RegisteredAgent:
        try:
            return self._agents[agent_id]
        except KeyError as error:
            raise KeyError(f"Unknown A2A agent: {agent_id}") from error

    def cards(self) -> dict[str, dict[str, Any]]:
        return {agent_id: agent.card.as_dict() for agent_id, agent in self._agents.items()}

    def save_task(self, task: A2ATask) -> None:
        self._tasks[task.task_id] = task
        self._tasks.move_to_end(task.task_id)
        while len(self._tasks) > self._max_tasks:
            self._tasks.popitem(last=False)

    def task(self, task_id: str) -> A2ATask:
        try:
            return self._tasks[task_id]
        except KeyError as error:
            raise KeyError(f"Unknown A2A task: {task_id}") from error

    async def dispatch(self, agent_id: str, data: dict[str, Any]) -> dict[str, Any]:
        registered = self.get(agent_id)
        if inspect.iscoroutinefunction(registered.handler):
            result = await registered.handler(data)
        else:
            # Keep synchronous DB/file/vector operations off the event loop.
            result = await asyncio.to_thread(registered.handler, data)
        if inspect.isawaitable(result):
            return await result
        return result


class A2AClient:
    def __init__(self, registry: A2ARegistry) -> None:
        self.registry = registry

    async def send(self, agent_id: str, data: dict[str, Any], *, context_id: str | None = None) -> dict[str, Any]:
        trace_id = str(data.get("trace_id") or "") or None
        owned_trace: TraceContext | None = None
        if trace_id is None:
            trace_metadata = {"agent_id": agent_id}
            if data.get("owner_id") is not None:
                trace_metadata["owner_id"] = int(data["owner_id"])
            owned_trace = TraceContext(run_type="a2a_call", metadata=trace_metadata)
            owned_trace.start()
            trace_id = owned_trace.trace_id
            data = {**data, "trace_id": trace_id}
        # Build the envelope after adding trace_id so remote and local handlers
        # receive the same context.
        message = A2AMessage.from_text("StoryRole internal task", data=data)
        task = A2ATask(context_id=context_id or A2ATask().context_id, history=[message.as_dict()])
        transport = os.getenv("STORYROLE_A2A_TRANSPORT", "local").lower()
        try:
            if transport == "http":
                return await observe_async_call(
                    lambda: self._send_http(agent_id, message, task),
                    operation="a2a:" + agent_id, trace_id=trace_id, model=None, prompt=str(data),
                )

            self.registry.save_task(task)
            with TraceContext(trace_id=trace_id).step(
                "a2a:" + agent_id, metadata={"agent_id": agent_id}
            ):
                result = await self.registry.dispatch(agent_id, message.data())
            task.complete(result)
            self.registry.save_task(task)
            artifact = task.artifacts[-1]
            return dict(artifact["parts"][0]["data"])
        except Exception as error:
            if transport != "http":
                task.fail(f"{type(error).__name__}: {error}")
                self.registry.save_task(task)
            raise
        finally:
            if owned_trace is not None:
                owned_trace.finish()

    async def _send_http(self, agent_id: str, message: A2AMessage, task: A2ATask) -> dict[str, Any]:
        """Send the same envelope to a separately deployed A2A agent."""
        import httpx

        env_key = "STORYROLE_A2A_URL_" + agent_id.upper().replace("-", "_")
        base_url = os.getenv(env_key) or os.getenv("STORYROLE_A2A_BASE_URL", "http://127.0.0.1:1314")
        url = f"{base_url.rstrip('/')}/api/v1/a2a/{agent_id}"
        payload = {
            "jsonrpc": "2.0",
            "id": task.task_id,
            "method": "message/send",
            "params": {
                "contextId": task.context_id,
                "message": message.as_dict(),
            },
        }
        headers = {}
        shared_secret = os.getenv("STORYROLE_A2A_SHARED_SECRET")
        if shared_secret:
            headers["X-StoryRole-A2A-Key"] = shared_secret
        async with httpx.AsyncClient(timeout=float(os.getenv("STORYROLE_A2A_TIMEOUT", "60"))) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
        if body.get("error"):
            raise RuntimeError(str(body["error"]))
        remote_task = body.get("result") or {}
        artifacts = remote_task.get("artifacts") or []
        if not artifacts:
            raise RuntimeError("A2A response has no artifacts")
        parts = artifacts[-1].get("parts") or []
        for part in parts:
            if part.get("kind") == "data" and isinstance(part.get("data"), dict):
                return dict(part["data"])
        raise RuntimeError("A2A response has no data artifact")


registry = A2ARegistry()
client = A2AClient(registry)
