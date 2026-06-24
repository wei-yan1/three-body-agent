"""Luo Ji Agentic RAG entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.agents.persona.middleware import (
    OptimizedQuery,
    QueryOptimizationInput,
    QueryOptimizationMiddleware,
)
from app.rag.loaders.persona_skill_loader import load_jsonl_records
from app.rag.retrievers.novel_vector_retriever import NovelVectorRetriever
from app.rag.retrievers.persona_vector_retriever import PersonaVectorRetriever


if TYPE_CHECKING:
    from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LUOJI_PERSONA_PATH = Path(
    PROJECT_ROOT / "data/processed/persona_profiles/luoji_temporal_persona_skill.jsonl"
)


@dataclass(frozen=True)
class LuoJiAgentRequest:
    """Request from the frontend after character and T stage are selected."""

    timeline_stage: str
    user_query: str
    top_k: int = 5


@dataclass(frozen=True)
class LuoJiAgentContext:
    """Retrieval result for the next agent-generation step."""

    character: str
    timeline_stage: str
    stage_profile: dict
    optimized_query: OptimizedQuery
    retrieved_documents: list[Document]
    novel_documents: list[Document]


class LuoJiAgent:
    """First role-specific Agentic RAG shell for Luo Ji.

    This class stops at query optimization and retrieval. Response generation,
    consistency checking, session memory, and API integration can be layered on
    later without changing the retrieval contract.
    """

    character = "罗辑"

    def __init__(
        self,
        persona_path: str | Path = LUOJI_PERSONA_PATH,
        retriever: PersonaVectorRetriever | None = None,
        novel_retriever: NovelVectorRetriever | None = None,
        query_middleware: QueryOptimizationMiddleware | None = None,
    ) -> None:
        self.persona_path = Path(persona_path)
        self.retriever = retriever or PersonaVectorRetriever()
        self.novel_retriever = novel_retriever or NovelVectorRetriever()
        self.query_middleware = query_middleware or QueryOptimizationMiddleware()
        self._stage_profiles = self._load_stage_profiles()

    def _load_stage_profiles(self) -> dict[str, dict]:
        records = load_jsonl_records(self.persona_path)
        return {
            record["stage_id"]: record
            for record in records
            if record.get("record_type") == "stage"
        }

    def get_stage_profile(self, timeline_stage: str) -> dict:
        """Return Luo Ji's profile for the requested T stage."""
        try:
            return self._stage_profiles[timeline_stage]
        except KeyError as error:
            valid = ", ".join(sorted(self._stage_profiles))
            message = f"Unsupported Luo Ji timeline_stage={timeline_stage!r}; valid stages: {valid}"
            raise ValueError(message) from error

    def prepare_context(self, request: LuoJiAgentRequest) -> LuoJiAgentContext:
        """Optimize the query and retrieve current-stage persona context."""
        stage_profile = self.get_stage_profile(request.timeline_stage)
        optimized_query = self.query_middleware.optimize(
            QueryOptimizationInput(
                character=self.character,
                timeline_stage=request.timeline_stage,
                user_query=request.user_query,
                stage_name=stage_profile.get("stage_name"),
                known_events=stage_profile.get("已知事件", []),
                forbidden_events=stage_profile.get("禁止知道的未来事件", []),
            )
        )
        documents = self.retriever.retrieve(
            optimized_query.query,
            character=self.character,
            timeline_stage=request.timeline_stage,
            k=request.top_k,
        )
        novel_documents = self.novel_retriever.retrieve(
            optimized_query.query,
            character=self.character,
            timeline_stage=request.timeline_stage,
            k=request.top_k,
        )
        return LuoJiAgentContext(
            character=self.character,
            timeline_stage=request.timeline_stage,
            stage_profile=stage_profile,
            optimized_query=optimized_query,
            retrieved_documents=documents,
            novel_documents=novel_documents,
        )

    def build_test_reply(self, context: LuoJiAgentContext) -> str:
        """Build a deterministic test reply without calling an LLM."""
        profile = context.stage_profile
        forbidden = "；".join(profile.get("禁止知道的未来事件", [])[:3])
        style = profile.get("语言风格", "")
        tendency = "；".join(profile.get("典型回答倾向", [])[:3])
        retrieved = "\n".join(
            f"- {doc.metadata.get('chunk_kind')}: {doc.page_content[:180].replace(chr(10), ' ')}"
            for doc in context.retrieved_documents
        )

        return (
            f"【测试回复：{self.character} {context.timeline_stage}】\n"
            f"我会停在这个阶段说话。此时的我：{profile.get('当前人格状态', '')}\n\n"
            f"说话风格约束：{style}\n"
            f"回答倾向：{tendency}\n"
            f"不能知道：{forbidden}\n\n"
            f"基于当前检索到的资料，我会这样回应你的问题：\n"
            f"{self._stage_specific_test_answer(context)}\n\n"
            f"【检索片段】\n{retrieved}"
        )

    def _stage_specific_test_answer(self, context: LuoJiAgentContext) -> str:
        query = context.optimized_query.original_query
        stage = context.timeline_stage
        if stage == "T1":
            return f"你问“{query}”。这事听起来太大了，大到不太像该交给我这种人。我可以推演，但别急着把推演当使命。"
        if stage == "T2":
            return f"你问“{query}”。如果你们一定要把沉默也解释成计划，那它当然可以是计划的一部分。至于里面有没有东西，我现在不能说。"
        if stage == "T3":
            return f"你问“{query}”。有些答案不是让人兴奋的，是让人冷下来的。一旦位置被说出，善意和恶意就都不重要了。"
        if stage == "T4":
            return f"你问“{query}”。威慑必须可信。问题不在于我想不想按下去，而在于对方是否相信我会按下去。"
        if stage == "T5":
            return f"你问“{query}”。剑不是用来解释的。它存在的意义，是让最坏的事不要发生。"
        if stage == "T6":
            return f"你问“{query}”。老了以后，有些事不再需要争辩。人会忘记森林，但森林不会因为人忘记它就消失。"
        return f"你问“{query}”。我只能按当前阶段能知道的东西回答。"
