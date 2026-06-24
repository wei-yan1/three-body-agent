"""Chat service for persona agents."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from app.agents.persona.intent_router import QueryIntentRouter
from app.agents.persona.luoji_agent import LuoJiAgent
from app.agents.persona.middleware import TemporalPersonaRAGMiddleware
from app.agents.persona.zhangbeihai_agent import ZhangBeihaiAgent
from app.core.config.settings import (
    DEFAULT_ROUTER_BASE_URL,
    DEFAULT_ROUTER_MODEL,
    EPISODIC_FORGET_THRESHOLD,
    EPISODIC_MEMORY_CAPACITY,
    TEST_CHAT_MODEL,
)
from app.memory.tools import MemoryTool
from app.rag.tools.web_search import TavilyWebSearchProvider


KnowledgeMode = Literal["temporal", "transparent"]


load_dotenv(override=True)


@lru_cache(maxsize=1)
def _luoji_agent_shell() -> LuoJiAgent:
    return LuoJiAgent()


@lru_cache(maxsize=1)
def _zhangbeihai_agent_shell() -> ZhangBeihaiAgent:
    return ZhangBeihaiAgent()


def _prepare_openai_compatible_env() -> None:
    os.environ.setdefault("OPENAI_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
    if os.getenv("DASHSCOPE_BASE_URL"):
        os.environ.setdefault("OPENAI_BASE_URL", os.environ["DASHSCOPE_BASE_URL"])


def _build_router_llm():
    router_api_key = (
        os.getenv("ROUTER_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("ZHIPUAI_API_KEY")
        or os.getenv("GLM_API_KEY")
    )
    if not router_api_key:
        return None
    return init_chat_model(
        DEFAULT_ROUTER_MODEL,
        model_provider="openai",
        api_key=router_api_key,
        base_url=DEFAULT_ROUTER_BASE_URL or os.getenv("OPENAI_BASE_URL"),
        temperature=0,
    )


def chat_with_luoji(
    *,
    user,
    thread: dict,
    timeline_stage: str,
    knowledge_mode: KnowledgeMode,
    message: str,
    recent_messages: list[dict] | None = None,
) -> str:
    """Run one Luo Ji Agent response."""
    return _chat_with_temporal_persona(
        agent_shell=_luoji_agent_shell(),
        character="罗辑",
        user=user,
        thread=thread,
        timeline_stage=timeline_stage,
        knowledge_mode=knowledge_mode,
        message=message,
        recent_messages=recent_messages,
    )


def chat_with_zhangbeihai(
    *,
    user,
    thread: dict,
    timeline_stage: str,
    knowledge_mode: KnowledgeMode,
    message: str,
    recent_messages: list[dict] | None = None,
) -> str:
    """Run one Zhang Beihai Agent response."""
    return _chat_with_temporal_persona(
        agent_shell=_zhangbeihai_agent_shell(),
        character="章北海",
        user=user,
        thread=thread,
        timeline_stage=timeline_stage,
        knowledge_mode=knowledge_mode,
        message=message,
        recent_messages=recent_messages,
    )


def _chat_with_temporal_persona(
    *,
    agent_shell,
    character: str,
    user,
    thread: dict,
    timeline_stage: str,
    knowledge_mode: KnowledgeMode,
    message: str,
    recent_messages: list[dict] | None = None,
) -> str:
    """Run one temporal persona Agent response with shared RAG middleware."""
    _prepare_openai_compatible_env()

    stage_profile = agent_shell.get_stage_profile(timeline_stage)
    router_llm = _build_router_llm()
    intent_router = (
        QueryIntentRouter(llm=router_llm, default_persona_k=5, default_novel_k=5)
        if router_llm is not None
        else None
    )
    web_search_provider = (
        TavilyWebSearchProvider() if knowledge_mode == "transparent" else None
    )
    rag_middleware = TemporalPersonaRAGMiddleware(
        retriever=agent_shell.retriever,
        novel_retriever=agent_shell.novel_retriever,
        memory_tool=MemoryTool(),
        user=user,
        thread_name=str(thread.get("thread_name") or "线程1"),
        stage_profile=stage_profile,
        character=character,
        intent_router=intent_router,
        web_search_provider=web_search_provider,
        knowledge_mode=knowledge_mode,
        timeline_stage=timeline_stage,
        top_k=5,
        novel_top_k=5,
    )
    chat_model = init_chat_model(TEST_CHAT_MODEL, model_provider="openai")
    agent = create_agent(
        model=chat_model,
        middleware=[rag_middleware],
        system_prompt=_build_temporal_persona_system_prompt(
            character=character,
            timeline_stage=timeline_stage,
            knowledge_mode=knowledge_mode,
        ),
    )
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in (recent_messages or [])
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]
    if history and history[-1]["role"] == "user" and history[-1]["content"] == message:
        history = history[:-1]
    response = agent.invoke({"messages": [*history, {"role": "user", "content": message}]})
    answer = response["messages"][-1].content
    _store_episodic_memory(
        character=character,
        user=user,
        thread=thread,
        timeline_stage=timeline_stage,
        knowledge_mode=knowledge_mode,
        message=message,
        answer=answer,
    )
    return answer


def _build_temporal_persona_system_prompt(
    *,
    character: str,
    timeline_stage: str,
    knowledge_mode: KnowledgeMode,
) -> str:
    mode_rule = (
        f"当前是通透模式。你仍然以所选时间段的{character}心理状态、人格状态和表达方式说话，"
        "但可以参考联网资料评价现实新闻、外部资料或未来情节。不要把人格改写成未来阶段。"
        if knowledge_mode == "transparent"
        else "当前是拟真模式。你只能使用当前时间段及此前可知的信息，不得提前知道未来事件。"
    )
    return (
        f"你是《三体》世界中 {timeline_stage} 阶段的{character}。"
        f"{mode_rule}"
        "你不是剧情解说员，也不是知识库摘要器。"
        "先自然回应用户，再在必要时体现当前角色在该阶段的思考方式、情绪底色和责任边界。"
        "日常问题要像真人一样正常回答，深度问题要认真思考。"
        "不要机械复读标志性词句；专有名词只有上下文真的需要时才提。"
        "不要输出 chunk_id、引用清单或检索调试信息。"
    )


def _store_episodic_memory(
    *,
    character: str,
    user,
    thread: dict,
    timeline_stage: str,
    knowledge_mode: KnowledgeMode,
    message: str,
    answer: str,
) -> None:
    try:
        memory_tool = MemoryTool()
        thread_name = str(thread.get("thread_name") or "线程1")
        memory_tool.execute(
            "add",
            user=user,
            character=character,
            timeline_stage=timeline_stage,
            mode=knowledge_mode,
            thread_name=thread_name,
            thread_id=int(thread.get("id")) if thread.get("id") else None,
            importance=0.4,
            summary=f"用户提问：{message[:160]} | {character}回答：{answer[:220]}",
            content=f"用户提问：{message}\n{character}回答：{answer}",
            source_turn_range="single_turn",
            metadata={
                "source": "chat_turn",
                "character": character,
                "timeline_stage": timeline_stage,
                "mode": knowledge_mode,
            },
        )
        memory_tool.execute(
            "forget",
            strategy="capacity_based",
            user=user,
            character=character,
            timeline_stage=timeline_stage,
            mode=knowledge_mode,
            thread_name=thread_name,
            capacity=EPISODIC_MEMORY_CAPACITY,
            threshold=EPISODIC_FORGET_THRESHOLD,
        )
    except Exception:
        return
