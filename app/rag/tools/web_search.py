"""Small web search provider used by transparent knowledge mode."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class WebSearchResult:
    """Normalized external search result."""

    title: str
    url: str
    content: str

    @property
    def chunk_id(self) -> str:
        return f"web_search:{self.url or self.title}"


class TavilyWebSearchProvider:
    """Tavily-backed web search provider.

    This class is intentionally optional: if TAVILY_API_KEY is not configured,
    search returns an empty list instead of breaking normal persona dialogue.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

    def search(self, query: str, max_results: int = 5) -> list[WebSearchResult]:
        if not self.api_key or not query.strip():
            return []

        from tavily import TavilyClient

        client = TavilyClient(api_key=self.api_key)
        try:
            response = client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_answer=False,
                include_raw_content=False,
            )
        except Exception as error:
            warnings.warn(
                f"Tavily search failed; transparent mode will continue without web results. {error}",
                stacklevel=2,
            )
            return []
        results = response.get("results", []) if isinstance(response, dict) else []
        normalized: list[WebSearchResult] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            normalized.append(
                WebSearchResult(
                    title=str(item.get("title") or ""),
                    url=str(item.get("url") or ""),
                    content=str(item.get("content") or ""),
                )
            )
        return normalized
