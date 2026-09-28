"""Bounded Tavily search for StoryRole external evidence."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any


class StoryRoleWebSearchAgent:
    agent_id = "web-search"

    async def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload.get("query") or payload.get("message") or "").strip()
        if not query:
            return {"results": [], "query": "", "skipped": True}
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return {"results": [], "query": query, "degraded": True, "reason": "missing_tavily_api_key"}
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=api_key)
            started = time.perf_counter()
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.search,
                    query=query,
                    search_depth="basic",
                    max_results=min(max(int(payload.get("max_results", 4)), 1), 6),
                    include_answer=False,
                    include_raw_content=False,
                ),
                timeout=float(os.getenv("STORYROLE_WEB_SEARCH_TIMEOUT", "8")),
            )
            raw_results = response.get("results", []) if isinstance(response, dict) else []
            results = []
            for item in raw_results:
                if not isinstance(item, dict) or not item.get("url"):
                    continue
                results.append({
                    "source_type": "web",
                    "url": str(item.get("url")),
                    "title": str(item.get("title") or ""),
                    "content": str(item.get("content") or "")[:3000],
                    "published_at": item.get("published_date"),
                    "score": float(item.get("score") or 0.0),
                    "query": query,
                })
            return {"results": results, "query": query, "duration_ms": round((time.perf_counter() - started) * 1000, 1)}
        except Exception as error:
            return {"results": [], "query": query, "degraded": True, "reason": type(error).__name__}
