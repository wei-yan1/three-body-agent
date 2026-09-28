"""Decision backend abstractions for StoryRole routing."""
from __future__ import annotations
import os
from typing import Any, Protocol

class DecisionBackend(Protocol):
    name: str
    def plan(self, message: str) -> dict[str, Any]: ...

def backend_mode() -> str:
    return os.getenv("STORYROLE_DECISION_BACKEND", "laya").strip().lower()
