"""Ye Wenjie Agentic RAG entrypoint."""

from __future__ import annotations

from pathlib import Path

from app.agents.persona.luoji_agent import LuoJiAgent


PROJECT_ROOT = Path(__file__).resolve().parents[3]
YEWENJIE_PERSONA_PATH = Path(
    PROJECT_ROOT / "data/processed/persona_profiles/yewenjie_temporal_persona_skill.jsonl"
)


class YeWenjieAgent(LuoJiAgent):
    """Role-specific Agent shell for Ye Wenjie."""

    character = "叶文洁"

    def __init__(self, persona_path: str | Path = YEWENJIE_PERSONA_PATH, **kwargs) -> None:
        super().__init__(persona_path=persona_path, **kwargs)
