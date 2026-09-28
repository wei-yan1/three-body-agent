"""Async DAG executor with bounded parallelism and replay-friendly results."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.agents.runtime.graph import ExecutionPlan, PlanNode

NodeHandler = Callable[[PlanNode, dict[str, Any]], Awaitable[Any]]


@dataclass
class ExecutionResult:
    outputs: dict[str, Any] = field(default_factory=dict)
    node_status: dict[str, dict[str, Any]] = field(default_factory=dict)


class GraphRuntime:
    def __init__(self, handlers: dict[str, NodeHandler], *, max_parallel: int = 8) -> None:
        self.handlers = handlers
        self.max_parallel = max(1, max_parallel)

    async def execute(self, plan: ExecutionPlan, payload: dict[str, Any]) -> ExecutionResult:
        result = ExecutionResult()
        pending = {node.node_id: node for node in plan.nodes}
        semaphore = asyncio.Semaphore(self.max_parallel)
        while pending:
            ready = [
                node for node in pending.values()
                if all(parent in result.outputs for parent in node.depends_on)
            ]
            if not ready:
                raise RuntimeError("执行计划无法继续，可能存在循环或未满足依赖")

            async def run(node: PlanNode) -> tuple[str, Any, dict[str, Any]]:
                started = time.perf_counter()
                handler = self.handlers.get(node.agent)
                if handler is None:
                    if node.required:
                        raise RuntimeError(f"没有注册执行器: {node.agent}")
                    return node.node_id, None, {"status": "skipped", "duration_ms": 0}
                node_payload = {**payload, "plan_outputs": {key: result.outputs[key] for key in node.depends_on}}
                async with semaphore:
                    try:
                        value = await asyncio.wait_for(handler(node, node_payload), node.timeout_ms / 1000)
                        return node.node_id, value, {"status": "completed", "duration_ms": round((time.perf_counter() - started) * 1000, 2)}
                    except Exception as error:
                        if node.required:
                            raise
                        return node.node_id, None, {"status": "degraded", "error": type(error).__name__, "duration_ms": round((time.perf_counter() - started) * 1000, 2)}

            completed = await asyncio.gather(*(run(node) for node in ready))
            for node_id, value, status in completed:
                pending.pop(node_id, None)
                result.outputs[node_id] = value
                result.node_status[node_id] = status
        return result
