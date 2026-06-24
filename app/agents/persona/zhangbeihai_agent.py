"""Zhang Beihai Agentic RAG entrypoint."""

from __future__ import annotations

from pathlib import Path

from app.agents.persona.luoji_agent import LuoJiAgent


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ZHANG_BEIHAI_PERSONA_PATH = Path(
    PROJECT_ROOT / "data/processed/persona_profiles/zhangbeihai_temporal_persona_skill.jsonl"
)


class ZhangBeihaiAgent(LuoJiAgent):
    """Role-specific Agent shell for Zhang Beihai.

    The retrieval and stage-profile contract is identical to Luo Ji's; only the
    character name and persona JSONL source differ.
    """

    character = "章北海"

    def __init__(self, persona_path: str | Path = ZHANG_BEIHAI_PERSONA_PATH, **kwargs) -> None:
        super().__init__(persona_path=persona_path, **kwargs)
