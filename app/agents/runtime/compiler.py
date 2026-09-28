"""Plan compilation and deterministic validation."""

from __future__ import annotations

from typing import Any

from app.agents.runtime.graph import ExecutionPlan, PlanNode


class PlanValidator:
    """Reject unsafe plans before they reach the execution runtime."""

    allowed_agents = {
        "memory",
        "relationship-state",
        "timeline-guard",
        "deep-question-planner",
        "novel-retriever",
        "evidence-analysis",
        "role-cognition",
        "context-curator",
        "character-reasoning",
        "answer-generation",
        "consistency-guard",
        "web-search",
    }

    def validate(self, plan: ExecutionPlan) -> ExecutionPlan:
        node_ids = {node.node_id for node in plan.nodes}
        if len(node_ids) != len(plan.nodes):
            raise ValueError("执行计划包含重复节点")
        for node in plan.nodes:
            if node.agent not in self.allowed_agents:
                raise ValueError(f"执行计划包含未授权 Agent: {node.agent}")
            if node.timeout_ms <= 0 or node.timeout_ms > 120_000:
                raise ValueError(f"节点超时不合法: {node.node_id}")
            if any(parent not in node_ids for parent in node.depends_on):
                raise ValueError(f"节点依赖不存在: {node.node_id}")
        self._assert_acyclic(plan)
        constraints = dict(plan.constraints)
        constraints.setdefault("max_model_calls", 8)
        constraints.setdefault("max_total_time_ms", 30_000)
        constraints.setdefault("max_cost", 1.0)
        if constraints["max_model_calls"] > 32:
            raise ValueError("模型调用预算过高")
        return ExecutionPlan(plan.plan_version, plan.nodes, constraints, plan.compiler)

    @staticmethod
    def _assert_acyclic(plan: ExecutionPlan) -> None:
        graph = {node.node_id: set(node.depends_on) for node in plan.nodes}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("执行计划存在循环依赖")
            if node_id in visited:
                return
            visiting.add(node_id)
            for parent in graph[node_id]:
                visit(parent)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in graph:
            visit(node_id)


class PlanCompiler:
    """Compile the current planner output into a stable execution plan."""

    def compile(self, decision: dict[str, Any], *, mode: str) -> ExecutionPlan:
        nodes: list[PlanNode] = [
            PlanNode("timeline", "timeline-guard", timeout_ms=2_000, required=False),
        ]
        if decision.get("needs_memory") or mode == "deep":
            nodes.append(PlanNode("memory", "memory", timeout_ms=1_000, required=False))
        if decision.get("needs_relationship") or mode == "deep":
            nodes.append(PlanNode("relationship", "relationship-state", timeout_ms=2_000, required=False))
        if mode == "deep":
            nodes.extend([
                PlanNode("deep-plan", "deep-question-planner", timeout_ms=20_000),
                PlanNode("novel", "novel-retriever", ("deep-plan", "timeline"), timeout_ms=35_000),
                PlanNode("evidence", "evidence-analysis", ("novel",), timeout_ms=25_000),
                PlanNode("cognition", "role-cognition", ("evidence",), timeout_ms=25_000),
                PlanNode("context", "context-curator", ("novel", "memory"), timeout_ms=15_000),
                PlanNode("answer", "answer-generation", ("context", "cognition"), timeout_ms=55_000),
                PlanNode("guard", "consistency-guard", ("answer",), timeout_ms=20_000),
            ])
        else:
            context_dependencies = ("memory",) if decision.get("needs_memory") else ()
            nodes.extend([
                PlanNode("context", "context-curator", context_dependencies, timeout_ms=15_000),
                PlanNode("reasoning", "character-reasoning", ("context",), timeout_ms=8_000),
                PlanNode("answer", "answer-generation", ("reasoning",), timeout_ms=25_000),
            ])
        return PlanValidator().validate(ExecutionPlan(
            plan_version="storyrole-plan-v1",
            nodes=tuple(nodes),
            compiler="laya" if decision.get("decision_backend") == "laya" else "rule",
            constraints={"no_future_facts": True, "max_model_calls": 8},
        ))
