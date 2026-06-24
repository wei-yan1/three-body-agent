"""Try the Luo Ji persona agent from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.persona import LuoJiAgent, LuoJiAgentRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        default="T2",
        choices=["T1", "T2", "T3", "T4", "T5", "T6"],
        help="Luo Ji timeline stage selected by the frontend.",
    )
    parser.add_argument(
        "--message",
        default="你为什么一直说这是计划的一部分？",
        help="User message to test.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Retriever top-k.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    agent = LuoJiAgent()
    context = agent.prepare_context(
        LuoJiAgentRequest(
            timeline_stage=args.stage,
            user_query=args.message,
            top_k=args.top_k,
        )
    )
    print(agent.build_test_reply(context))


if __name__ == "__main__":
    main()
