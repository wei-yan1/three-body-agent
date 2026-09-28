"""Serializable plans used to compile and replay agent workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PlanNode:
    node_id: str
    agent: str
    depends_on: tuple[str, ...] = ()
    timeout_ms: int = 10_000
    required: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionPlan:
    plan_version: str
    nodes: tuple[PlanNode, ...]
    constraints: dict[str, Any] = field(default_factory=dict)
    compiler: str = "rule"

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "compiler": self.compiler,
            "nodes": [node.as_dict() for node in self.nodes],
            "constraints": dict(self.constraints),
        }
