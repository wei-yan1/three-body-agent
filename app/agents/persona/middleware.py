"""Agent middleware primitives for persona RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, Runtime

from app.agents.persona.intent_router import QueryIntentRouter
from app.memory.tools import MemoryTool


RELATIONSHIP_ALIASES = {
    "叶文洁": ("叶文洁", "叶老师", "叶教授"),
    "史强": ("史强", "大史"),
    "庄颜": ("庄颜",),
    "程心": ("程心",),
    "三体世界": ("三体世界", "三体", "三体文明"),
    "人类社会": ("人类社会", "人类"),
}


@dataclass(frozen=True)
class QueryOptimizationInput:
    """Inputs known before retrieval."""

    character: str
    timeline_stage: str
    user_query: str
    stage_name: str | None = None
    known_events: list[str] = field(default_factory=list)
    forbidden_events: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OptimizedQuery:
    """Query and metadata filters produced by query optimization middleware."""

    query: str
    filters: dict[str, str]
    original_query: str


class QueryOptimizationMiddleware:
    """Rewrite user questions into persona-aware retrieval queries.

    The frontend already provides the target T stage. This middleware therefore
    does not infer timeline stage; it strengthens the query with the selected
    character and stage while creating strict metadata filters for Chroma.
    """

    def optimize(self, request: QueryOptimizationInput) -> OptimizedQuery:
        query_parts = [
            f"角色：{request.character}",
            f"时间线阶段：{request.timeline_stage}",
        ]
        if request.stage_name:
            query_parts.append(f"阶段名称：{request.stage_name}")
        if request.known_events:
            query_parts.append(f"当前阶段已知事件：{'；'.join(request.known_events[:6])}")
        query_parts.append(f"用户问题：{request.user_query}")
        query_parts.append("检索目标：召回当前阶段的人格、知识边界、关系、小说融合增强和RAG卡片。")

        return OptimizedQuery(
            query="\n".join(query_parts),
            filters={
                "$and": [
                    {"character": {"$eq": request.character}},
                    {"timeline_stage": {"$eq": request.timeline_stage}},
                ]
            },
            original_query=request.user_query,
        )


class TemporalPersonaRAGMiddleware(AgentMiddleware):
    """LangChain middleware for temporal persona RAG orchestration.

    This middleware is character-agnostic: callers provide the target
    character, timeline stage, stage profile, retrievers and optional memory
    tool. Luo Ji is just the first role using it.
    """

    def __init__(
        self,
        retriever: Any,
        stage_profile: dict[str, Any],
        novel_retriever: Any | None = None,
        memory_tool: MemoryTool | None = None,
        user: Any | None = None,
        thread_name: str = "线程1",
        llm: Any | None = None,
        intent_router: QueryIntentRouter | None = None,
        web_search_provider: Any | None = None,
        knowledge_mode: str = "temporal",
        character: str = "罗辑",
        timeline_stage: str = "T2",
        top_k: int = 3,
        novel_top_k: int = 4,
    ) -> None:
        super().__init__()
        self.retriever = retriever
        self.novel_retriever = novel_retriever
        self.memory_tool = memory_tool
        self.user = user
        self.thread_name = thread_name
        self.stage_profile = stage_profile
        self.llm = llm
        self.intent_router = intent_router
        self.web_search_provider = web_search_provider
        self.knowledge_mode = knowledge_mode
        self.character = character
        self.timeline_stage = timeline_stage
        self.top_k = top_k
        self.novel_top_k = novel_top_k
        self.optimizer = QueryOptimizationMiddleware()

    def before_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        last_message = messages[-1]
        query = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )
        route = (
            self.intent_router.route(
                character=self.character,
                timeline_stage=self.timeline_stage,
                stage_name=str(self.stage_profile.get("stage_name", "")),
                user_query=query,
                knowledge_mode=self.knowledge_mode,
                known_events=self.stage_profile.get("已知事件", []),
                forbidden_events=self.stage_profile.get("禁止知道的未来事件", []),
                relationships=self.stage_profile.get("与其他角色的关系状态", {}),
            )
            if self.intent_router is not None
            else None
        )
        rewritten_query = (
            route.rewritten_query
            if route is not None and route.rewritten_query
            else self._rewrite_query(query)
        )
        optimized = self.optimizer.optimize(
            QueryOptimizationInput(
                character=self.character,
                timeline_stage=self.timeline_stage,
                user_query=rewritten_query,
                stage_name=self.stage_profile.get("stage_name"),
                known_events=self.stage_profile.get("已知事件", []),
                forbidden_events=self.stage_profile.get("禁止知道的未来事件", []),
            )
        )
        persona_k = route.persona_k if route is not None else self.top_k
        novel_k = route.novel_k if route is not None else self.novel_top_k
        memory_search = self._search_memory(rewritten_query)
        retrieved_docs = (
            self.retriever.retrieve(
                optimized.query,
                character=self.character,
                timeline_stage=self.timeline_stage,
                k=persona_k,
            )
            if persona_k > 0
            else []
        )
        novel_docs = (
            self.novel_retriever.retrieve(
                optimized.query,
                character=self.character,
                timeline_stage=self.timeline_stage,
                k=novel_k,
            )
            if self.novel_retriever is not None
            and novel_k > 0
            else []
        )
        web_results = self._search_web(route)
        if (
            not memory_search
            and not retrieved_docs
            and not novel_docs
            and not web_results
        ):
            return None

        memory_content = self._format_memory_context(memory_search or [])
        persona_content = "\n\n".join(
            f"[{doc.metadata.get('chunk_id', '?')}] {doc.page_content}"
            for doc in retrieved_docs
        )
        novel_content = "\n\n".join(
            f"[{doc.metadata.get('chunk_id', '?')}] "
            f"阶段={doc.metadata.get('timeline_stage', '?')} "
            f"章节={doc.metadata.get('section_title', '?')}\n{doc.page_content}"
            for doc in novel_docs
        )
        web_content = "\n\n".join(
            f"[{result.chunk_id}] 标题={result.title}\nURL={result.url}\n{result.content}"
            for result in web_results
        )
        relationship_content, _relationship_chunk_ids = self._relationship_context(
            query,
            route.relationship_target if route is not None else None,
        )
        docs_content = (
            "【最高优先级：当前线程情景记忆，只能用于维持你和用户的对话连续性，不得突破当前阶段知识边界】\n"
            f"{memory_content or '无'}\n\n"
            "【最高优先级：当前阶段角色关系约束，若用户询问某角色必须优先使用】\n"
            f"{relationship_content or '无'}\n\n"
            "【最高优先级：人物时序人格 Skill】\n"
            f"{persona_content or '无'}\n\n"
            "【补充证据：小说原文，可回看当前阶段及以前，不可覆盖人物 Skill】\n"
            f"{novel_content or '无'}\n\n"
            "【通透模式外部资料：仅在 transparent 模式下用于新闻、现实事件或未来资料参考，不得改写当前阶段人格】\n"
            f"{web_content or '无'}"
        )
        forbidden = "；".join(self.stage_profile.get("禁止知道的未来事件", []))
        consistency_rules = "；".join(
            self.stage_profile.get("人设一致性校验规则", [])
        )
        route_context = (
            "\n".join(
                [
                    f"Router意图：{route.intent}",
                    f"Router检索策略：{route.retrieval_policy}",
                    f"Router置信度：{route.confidence:.2f}",
                    f"Router关系对象：{route.relationship_target or '无'}",
                    f"Router未来风险：{route.future_risk}",
                    f"Router联网搜索：{route.web_search_needed}",
                    f"Router联网查询：{route.web_search_query or '无'}",
                    f"Router回答指导：{route.answer_guidance or '无'}",
                ]
            )
            if route is not None
            else "Router意图：未启用"
        )
        mode_instruction = (
            "当前知识模式：transparent / 通透模式。你仍然停留在当前时间线阶段，"
            "人格、心理状态和说话方式不能跳到未来阶段；但你可以阅读外部资料、新闻资料和联网查询结果。"
            "如果外部资料包含当前阶段之后的未来信息，你可以知道并评价，但要像当前阶段的你读到资料后的反应，"
            "不要变成全知旁白或晚年人格。\n"
            if self.knowledge_mode == "transparent"
            else "当前知识模式：temporal / 严格时序。不得使用当前阶段之后的知识。\n"
        )
        augmented = (
            f"{query}\n\n"
            f"你现在必须扮演：{self.character}，时间线阶段：{self.timeline_stage}。\n"
            f"{mode_instruction}"
            f"阶段名称：{self.stage_profile.get('stage_name', '')}\n"
            f"当前人格状态：{self.stage_profile.get('当前人格状态', '')}\n"
            f"语言风格：{self.stage_profile.get('语言风格', '')}\n"
            f"禁止知道的未来事件：{forbidden}\n"
            f"人设一致性校验规则：{consistency_rules}\n\n"
            f"本轮路由结果：\n{route_context}\n\n"
            "上下文优先级规则：人物时序人格 Skill 高于小说原文；"
            "当前线程情景记忆只用于保持和用户的连续性，不得引入当前阶段不知道的未来知识；"
            "如果用户询问某个角色，当前阶段角色关系约束高于普通小说片段；"
            "通透模式外部资料只用于补充事实和新闻，不得改变当前阶段人格；"
            "小说原文只用于补充当前阶段及以前的事件、语气和场景；"
            "如果小说原文与人物 Skill 冲突，以人物 Skill 为准。\n\n"
            f"请根据以下上下文回答：\n{docs_content}\n\n"
            "正式输出时不要列出 chunk_id、引用清单或检索调试信息。"
        )

        return {"messages": [last_message.model_copy(update={"content": augmented})]}

    def _search_memory(self, query: str) -> list[Any]:
        if self.memory_tool is None or self.user is None or not self.thread_name:
            return []
        try:
            return self.memory_tool.search_items(
                query=query,
                user=self.user,
                character=self.character,
                timeline_stage=self.timeline_stage,
                mode=self.knowledge_mode,
                thread_name=self.thread_name,
                limit=3,
                min_importance=0.15,
            ).items
        except Exception:
            return []

    @staticmethod
    def _format_memory_context(items: list[Any]) -> str:
        if not items:
            return ""
        lines = []
        for item in items:
            lines.append(
                f"[{item.id}] 重要性={item.importance:.2f} 相关度={item.score:.2f} {item.content}"
            )
        return "\n".join(lines)

    def _search_web(self, route: Any | None) -> list[Any]:
        if (
            self.knowledge_mode != "transparent"
            or self.web_search_provider is None
            or route is None
            or not route.web_search_needed
            or not route.web_search_query.strip()
        ):
            return []
        return self.web_search_provider.search(route.web_search_query, max_results=5)

    def _relationship_context(
        self,
        query: str,
        relationship_target: str | None = None,
    ) -> tuple[str, list[str]]:
        relationships = self.stage_profile.get("与其他角色的关系状态", {})
        if not isinstance(relationships, dict) or not relationships:
            return "", []

        matched: list[tuple[str, str]] = []
        for character_name, description in relationships.items():
            aliases = RELATIONSHIP_ALIASES.get(character_name, (character_name,))
            target_matches = (
                bool(relationship_target)
                and (
                    relationship_target == character_name
                    or relationship_target in aliases
                    or character_name in relationship_target
                )
            )
            if target_matches or any(alias and alias in query for alias in aliases):
                matched.append((character_name, str(description)))

        if not matched:
            return "", []

        stage = self.timeline_stage
        lines = []
        chunk_ids = []
        for character_name, description in matched:
            chunk_id = f"stage_profile_{self.character}_{stage}_relationship_{character_name}"
            chunk_ids.append(chunk_id)
            lines.append(
                f"[{chunk_id}] {self.character}在{stage}阶段对{character_name}的关系态度：{description}"
            )
        return "\n".join(lines), chunk_ids

    def _rewrite_query(self, query: str) -> str:
        if self.llm is None:
            return query

        rewrite_prompt = f"""
将以下问题改写为适合检索当前角色、当前阶段人设知识库的关键词形式。
提取核心概念，用空格分隔。只输出关键词，不要解释。

角色: {self.character}
时间线阶段: {self.timeline_stage}
问题: {query}
关键词:
"""
        rewritten = self.llm.invoke(rewrite_prompt).content.strip()
        return rewritten or query


LuoJiRAGMiddleware = TemporalPersonaRAGMiddleware
