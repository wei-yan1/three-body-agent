"""Tool-style interface for episodic memory operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.memory.long_term.episodic_memory import EpisodicMemory, MemoryItem
from app.storage.repositories.episodic_memory_repository import (
    delete_episodic_memories_for_threads,
    forget_capacity_based_memories,
)


@dataclass
class MemorySearchResult:
    """Structured search result for middleware use."""

    items: list[MemoryItem]
    text: str


class MemoryTool:
    """Memory operations facade.

    The current project only enables episodic memory in production use, while
    keeping the action-based interface extensible for future memory types.
    """

    def __init__(self, episodic_memory: EpisodicMemory | None = None) -> None:
        self.episodic_memory = episodic_memory or EpisodicMemory()

    def execute(self, action: str, **kwargs) -> str:
        """执行记忆操作。

        已实现：
        - add: 添加情景记忆
        - search: 搜索情景记忆
        - summary: 获取当前线程记忆摘要
        - stats: 获取基础统计信息
        - remove: 删除指定情景记忆向量
        - clear_all: 清空指定线程范围内的所有情景记忆
        - forget: 基于容量的遗忘
        """
        if action == "add":
            return self._add_memory(**kwargs)
        if action == "search":
            return self._search_memory(**kwargs)
        if action == "summary":
            return self._get_summary(**kwargs)
        if action == "stats":
            return self._get_stats(**kwargs)
        if action == "remove":
            return self._remove_memory(**kwargs)
        if action == "clear_all":
            return self._clear_all(**kwargs)
        if action == "forget":
            return self._forget(**kwargs)
        return f"❌ 不支持的记忆操作: {action}"

    def search_items(
        self,
        *,
        query: str,
        user: Any,
        character: str,
        timeline_stage: str,
        mode: str,
        thread_name: str,
        limit: int = 3,
        min_importance: float = 0.1,
    ) -> MemorySearchResult:
        """Return structured memory items for middleware retrieval."""
        results = self.episodic_memory.retrieve(
            query=query,
            user=user,
            character=character,
            timeline_stage=timeline_stage,
            mode=mode,
            thread_name=thread_name,
            limit=limit,
            min_importance=min_importance,
        )
        return MemorySearchResult(
            items=results,
            text=self._format_search_results(results, query=query),
        )

    def _add_memory(self, **kwargs) -> str:
        memory_id = self.episodic_memory.add(**kwargs)
        return f"✅ 已写入情景记忆: {memory_id}"

    def _search_memory(
        self,
        query: str,
        limit: int = 5,
        memory_types: list[str] | None = None,
        memory_type: str | None = None,
        min_importance: float = 0.1,
        **kwargs,
    ) -> str:
        if memory_type and not memory_types:
            memory_types = [memory_type]
        if memory_types and "episodic" not in memory_types:
            return f"🔍 未找到与 '{query}' 相关的记忆"
        results = self.episodic_memory.retrieve(
            query=query,
            limit=limit,
            min_importance=min_importance,
            **kwargs,
        )
        return self._format_search_results(results, query=query)

    def _get_summary(self, **kwargs) -> str:
        query = str(kwargs.pop("query", "") or "")
        results = self.episodic_memory.retrieve(limit=5, query=query, **kwargs)
        if not results:
            return "🧠 当前线程还没有可用的情景记忆。"
        joined = "；".join(item.content for item in results[:3])
        return f"🧠 当前线程记忆摘要：{joined}"

    def _get_stats(self, **kwargs) -> str:
        query = str(kwargs.pop("query", "") or "")
        results = self.episodic_memory.retrieve(limit=20, query=query, **kwargs)
        return f"📊 当前线程可召回情景记忆数：{len(results)}"

    def _remove_memory(self, memory_ids: list[str]) -> str:
        deleted = self.episodic_memory.delete(memory_ids)
        return f"🗑️ 已删除 {deleted} 条情景记忆向量"

    def _clear_all(self, *, user: Any, thread_ids: list[int]) -> str:
        memory_ids = delete_episodic_memories_for_threads(
            user=user,
            thread_ids=thread_ids,
        )
        deleted_vectors = self.episodic_memory.delete(memory_ids)
        return f"🧹 已清空 {len(memory_ids)} 条线程情景记忆，删除 {deleted_vectors} 条记忆向量"

    def _forget(
        self,
        *,
        strategy: str,
        user: Any,
        character: str,
        timeline_stage: str,
        mode: str,
        thread_name: str,
        capacity: int = 40,
        threshold: float = 0.4,
    ) -> str:
        if strategy != "capacity_based":
            return f"❌ 当前只支持 capacity_based 遗忘，收到: {strategy}"
        memory_ids = forget_capacity_based_memories(
            user=user,
            character=character,
            timeline_stage=timeline_stage,
            mode=mode,
            thread_name=thread_name,
            capacity=capacity,
            threshold=threshold,
        )
        deleted_vectors = self.episodic_memory.delete(memory_ids)
        return (
            f"🧠 基于容量遗忘完成：删除 {len(memory_ids)} 条低保留分情景记忆，"
            f"删除 {deleted_vectors} 条记忆向量，阈值={threshold:.1f}"
        )

    @staticmethod
    def _format_search_results(results: list[MemoryItem], *, query: str) -> str:
        if not results:
            return f"🔍 未找到与 '{query}' 相关的情景记忆"
        formatted = [f"🔍 找到 {len(results)} 条相关情景记忆:"]
        for index, memory in enumerate(results, start=1):
            preview = memory.content[:80] + "..." if len(memory.content) > 80 else memory.content
            formatted.append(
                f"{index}. [情景记忆] {preview} (重要性: {memory.importance:.2f}, 相关度: {memory.score:.2f})"
            )
        return "\n".join(formatted)
