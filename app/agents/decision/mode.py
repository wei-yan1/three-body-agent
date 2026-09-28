"""Conversation mode selection without an extra routing-model call."""

from __future__ import annotations

from typing import Any


VALID_MODES = {"quick", "deep", "auto"}


def normalize_mode(value: Any, *, legacy_deep_reasoning: bool = False) -> str:
    if legacy_deep_reasoning:
        return "deep"
    mode = str(value or "auto").strip().lower()
    return mode if mode in VALID_MODES else "auto"


def resolve_mode(requested: Any, plan: dict[str, Any]) -> str:
    mode = normalize_mode(requested)
    if mode != "auto":
        return mode
    if plan.get("auto_deep") or plan.get("needs_deep_reasoning"):
        return "deep"
    return "quick"
