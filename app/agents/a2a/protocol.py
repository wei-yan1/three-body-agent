"""Minimal A2A-compatible message/task models used by StoryRole."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class A2AMessage:
    role: str
    parts: list[dict[str, Any]]
    message_id: str = field(default_factory=lambda: str(uuid4()))

    @classmethod
    def from_text(cls, text: str, *, role: str = "user", data: dict[str, Any] | None = None) -> A2AMessage:
        parts: list[dict[str, Any]] = [{"kind": "text", "text": text}]
        if data is not None:
            parts.append({"kind": "data", "data": data})
        return cls(role=role, parts=parts)

    def text(self) -> str:
        return "\n".join(str(part.get("text", "")) for part in self.parts if part.get("kind") == "text")

    def data(self) -> dict[str, Any]:
        for part in self.parts:
            if part.get("kind") == "data" and isinstance(part.get("data"), dict):
                return dict(part["data"])
        return {}

    def as_dict(self) -> dict[str, Any]:
        return {"messageId": self.message_id, "role": self.role, "parts": self.parts}


@dataclass
class A2ATask:
    task_id: str = field(default_factory=lambda: str(uuid4()))
    context_id: str = field(default_factory=lambda: str(uuid4()))
    state: str = "working"
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def complete(self, data: dict[str, Any]) -> A2ATask:
        self.state = "completed"
        self.artifacts.append({
            "artifactId": str(uuid4()),
            "parts": [{"kind": "data", "data": data}],
        })
        return self

    def fail(self, message: str) -> A2ATask:
        self.state = "failed"
        self.error = message
        return self

    def as_dict(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "state": self.state,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if self.error:
            status["message"] = {"role": "agent", "parts": [{"kind": "text", "text": self.error}]}
        return {
            "id": self.task_id,
            "contextId": self.context_id,
            "status": status,
            "artifacts": self.artifacts,
            "history": self.history,
        }


@dataclass(frozen=True)
class A2AAgentCard:
    name: str
    description: str
    skill_id: str
    skill_name: str
    skill_description: str
    url: str
    version: str = "0.1.0"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "supportedInterfaces": [
                {"protocolBinding": "JSONRPC", "url": self.url},
            ],
            "defaultInputModes": ["application/json", "text/plain"],
            "defaultOutputModes": ["application/json", "text/plain"],
            "capabilities": {"streaming": False, "extendedAgentCard": False},
            "skills": [{
                "id": self.skill_id,
                "name": self.skill_name,
                "description": self.skill_description,
                "tags": ["storyrole", "小说", "角色"],
                "inputModes": ["application/json", "text/plain"],
                "outputModes": ["application/json", "text/plain"],
            }],
        }
