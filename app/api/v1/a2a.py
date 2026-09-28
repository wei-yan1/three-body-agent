"""A2A HTTP endpoints for StoryRole agents."""

from __future__ import annotations

import os

from fastapi import APIRouter, Header, HTTPException

from app.agents.a2a.protocol import A2AMessage, A2ATask
from app.agents.a2a.registry import registry
from app.agents.storyrole.bootstrap import register_storyrole_agents

register_storyrole_agents()
router = APIRouter(prefix="/api/v1/a2a", tags=["a2a"])


@router.get("/cards")
def agent_cards() -> dict[str, dict]:
    return registry.cards()


@router.get("/{agent_id}/agent-card.json")
def agent_card(agent_id: str) -> dict:
    try:
        return registry.get(agent_id).card.as_dict()
    except KeyError as error:
        raise HTTPException(status_code=404, detail="A2A agent not found") from error


@router.get("/{agent_id}/.well-known/agent-card.json")
def agent_card_well_known(agent_id: str) -> dict:
    return agent_card(agent_id)


@router.post("/{agent_id}")
async def a2a_message_send(
    agent_id: str,
    body: dict,
    a2a_key: str | None = Header(default=None, alias="X-StoryRole-A2A-Key"),
) -> dict:
    """Handle the A2A JSON-RPC message/send operation.

    The payload supports both `params.message.parts[data]` and the convenient
    `params.data` form used by StoryRole's internal client.
    """
    request_id = body.get("id")
    shared_secret = os.getenv("STORYROLE_A2A_SHARED_SECRET")
    if shared_secret and a2a_key != shared_secret:
        raise HTTPException(status_code=401, detail="Invalid A2A service key")
    method = body.get("method")
    if method == "tasks/get":
        try:
            return {"jsonrpc": "2.0", "id": request_id, "result": registry.task(str((body.get("params") or {}).get("id"))).as_dict()}
        except KeyError as error:
            return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32004, "message": str(error)}}
    if method not in {"message/send", "message/stream"}:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "Unsupported A2A method"}}
    params = body.get("params") or {}
    message_data = params.get("data")
    if not isinstance(message_data, dict):
        message = params.get("message") or {}
        message_data = A2AMessage(
            role=str(message.get("role", "user")),
            parts=list(message.get("parts") or []),
            message_id=str(message.get("messageId") or ""),
        ).data()
    task = A2ATask(context_id=str(params.get("contextId") or A2ATask().context_id))
    registry.save_task(task)
    try:
        result = await registry.dispatch(agent_id, message_data)
        task.complete(result)
    except Exception as error:  # noqa: BLE001
        task.fail(f"{type(error).__name__}: {error}")
    registry.save_task(task)
    if task.state == "failed":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": task.error or "A2A task failed"},
        }
    return {"jsonrpc": "2.0", "id": request_id, "result": task.as_dict()}
