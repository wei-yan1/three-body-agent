from __future__ import annotations

import pytest

from app.agents.storyrole.web_search_agent import StoryRoleWebSearchAgent


@pytest.mark.asyncio
async def test_web_search_without_key_degrades_without_call(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = await StoryRoleWebSearchAgent().search({"query": "test"})
    assert result["results"] == []
    assert result["degraded"] is True
    assert result["reason"] == "missing_tavily_api_key"


@pytest.mark.asyncio
async def test_empty_web_query_is_skipped() -> None:
    result = await StoryRoleWebSearchAgent().search({"query": ""})
    assert result["skipped"] is True
