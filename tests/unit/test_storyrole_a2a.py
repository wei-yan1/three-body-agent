from __future__ import annotations

import pytest

from app.agents.a2a.protocol import A2AMessage, A2ATask
from app.agents.a2a.registry import A2AClient, A2ARegistry
from app.agents.storyrole.bootstrap import register_storyrole_agents


def test_a2a_message_and_task_round_trip() -> None:
    message = A2AMessage.from_text("resolve", data={"novel_id": "n1"})
    assert message.text() == "resolve"
    assert message.data() == {"novel_id": "n1"}
    task = A2ATask(history=[message.as_dict()]).complete({"found": True})
    assert task.state == "completed"
    assert task.as_dict()["artifacts"][0]["parts"][0]["data"] == {"found": True}


@pytest.mark.asyncio
async def test_in_process_a2a_client_dispatches_through_task_envelope() -> None:
    registry = A2ARegistry()
    registry.register(
        "echo",
        card=type("Card", (), {"as_dict": lambda self: {"name": "echo"}})(),
        handler=lambda data: {"echo": data["value"]},
    )
    result = await A2AClient(registry).send("echo", {"value": "ok"})
    assert result == {"echo": "ok"}


def test_storyrole_registers_three_a2a_agents() -> None:
    register_storyrole_agents()
    from app.agents.a2a.registry import registry

    assert {
        "character-resolver", "nuwa-profiler", "character-period-analysis",
        "deep-question-planner", "evidence-analysis", "role-cognition", "query-planner",
        "context-curator", "character-reasoning", "consistency-guard",
        "relationship-state", "timeline-guard", "memory-decision",
        "character-conversation",
    }.issubset(registry.cards())


@pytest.mark.asyncio
async def test_query_planner_and_guards_return_structured_outputs() -> None:
    from app.agents.storyrole.consistency_guard import ConsistencyGuardAgent
    from app.agents.storyrole.query_planner import QueryPlannerAgent
    from app.agents.storyrole.timeline_guard import TimelineGuardAgent

    plan = QueryPlannerAgent().plan({"message": "你还记得我上次说过的选择吗？"})
    assert plan["needs_memory"] is True
    assert "needs_deep_reasoning" not in plan
    review = await ConsistencyGuardAgent().review({"draft_answer": "我会听着。", "profile": {}})
    assert review["final_answer"] == "我会听着。"
    boundary = TimelineGuardAgent().check({"plan": {"future_probe": True}, "profile": {"knowledge_boundary": {"unknown": ["结局"]}}})
    assert boundary["allowed"] is False


@pytest.mark.asyncio
async def test_registry_bounds_task_history() -> None:
    from app.agents.a2a.protocol import A2AAgentCard
    registry = A2ARegistry()
    registry._max_tasks = 2
    registry.register("echo", A2AAgentCard("echo", "echo", "echo", "echo", "echo", "/echo"), lambda data: data)
    client = A2AClient(registry)
    await client.send("echo", {"value": 1})
    await client.send("echo", {"value": 2})
    await client.send("echo", {"value": 3})
    assert len(registry._tasks) == 2


def test_memory_decision_requires_confirmation_for_durable_facts() -> None:
    from app.agents.storyrole.memory_agent import MemoryDecisionAgent

    decision = MemoryDecisionAgent().decide({"message": "请记住我以后不喜欢被说教"})
    assert decision["requires_confirmation"] is True
    assert decision["should_save"] is False
    assert decision["candidate_status"] == "pending"

    ordinary = MemoryDecisionAgent().decide({"message": "今天天气不错"})
    assert ordinary["requires_confirmation"] is False


def test_timeline_guard_remains_lightweight() -> None:
    from app.agents.storyrole.timeline_guard import TimelineGuardAgent

    result = TimelineGuardAgent().check({"plan": {"future_probe": False}, "profile": {}})
    assert result["allowed"] is True
    assert result["future_probe"] is False


def test_query_planner_defaults_to_laya_and_falls_back_to_rules(monkeypatch) -> None:
    from app.agents.storyrole.query_planner import QueryPlannerAgent
    from app.agents.decision.laya_backend import LayaDecisionBackend
    from app.agents.decision.factory import get_decision_backend

    monkeypatch.delenv("STORYROLE_DECISION_BACKEND", raising=False)
    get_decision_backend.cache_clear()
    assert get_decision_backend().__class__.__name__ == "SafeLayaDecisionBackend"

    def unavailable(self, message):
        raise RuntimeError("Laya unavailable")

    monkeypatch.setattr(LayaDecisionBackend, "plan", unavailable)
    plan = QueryPlannerAgent().plan({"message": "你还记得我吗？"})
    assert plan["decision_backend"] == "rule_fallback"
    get_decision_backend.cache_clear()


def test_conversation_mode_selection_is_deterministic():
    from app.agents.decision.mode import resolve_mode

    assert resolve_mode("quick", {"auto_deep": True}) == "quick"
    assert resolve_mode("deep", {}) == "deep"
    assert resolve_mode("auto", {"auto_deep": True}) == "deep"
    assert resolve_mode("auto", {"auto_deep": False}) == "quick"


def test_rule_planner_marks_analysis_questions_for_auto_deep():
    from app.agents.storyrole.query_planner import QueryPlannerAgent

    plan = QueryPlannerAgent().plan({"message": "请分析他做出这个选择的动机和代价"})
    assert plan["auto_deep"] is True


def test_deep_agents_have_safe_non_llm_fallbacks():
    from app.agents.storyrole.deep_planner import DeepQuestionPlannerAgent
    from app.agents.storyrole.evidence_analyst import EvidenceAnalysisAgent
    from app.agents.storyrole.role_cognition_agent import RoleCognitionAgent

    deep_plan = DeepQuestionPlannerAgent()._fallback("为什么？")
    assert deep_plan["search_queries"]
    evidence = __import__("asyncio").run(EvidenceAnalysisAgent().analyze({"context": {}}))
    assert "uncertainties" in evidence
    cognition = __import__("asyncio").run(RoleCognitionAgent().decide({"profile": {}}))
    assert cognition["response_action"]
