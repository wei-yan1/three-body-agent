from __future__ import annotations

from app.agents.storyrole.character_agent import CharacterConversationAgent


def test_unwrap_fallback_marks_degradation() -> None:
    value, degraded = CharacterConversationAgent._unwrap_fallback({"_fallback_used": True, "_fallback_value": {"answer": "safe"}})
    assert value == {"answer": "safe"}
    assert degraded is True


def test_unwrap_success_does_not_mark_degradation() -> None:
    value, degraded = CharacterConversationAgent._unwrap_fallback({"answer": "ok"})
    assert value == {"answer": "ok"}
    assert degraded is False
