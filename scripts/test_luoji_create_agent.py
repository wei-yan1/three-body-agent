"""One-shot interactive test for the Luo Ji temporal persona Agent."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.persona.luoji_agent import LuoJiAgent
from app.agents.persona.intent_router import QueryIntentRouter
from app.agents.persona.middleware import TemporalPersonaRAGMiddleware
from app.core.config.settings import (
    DEFAULT_REWRITE_MODEL,
    DEFAULT_ROUTER_BASE_URL,
    DEFAULT_ROUTER_MODEL,
    TEST_CHAT_MODEL,
)
from app.rag.tools.web_search import TavilyWebSearchProvider


VALID_STAGES = ("T1", "T2", "T3", "T4", "T5", "T6")
VALID_KNOWLEDGE_MODES = ("temporal", "transparent")

STAGE_HELP = {
    "T1": "普通学者时期：青年罗辑，尚未承担文明责任。",
    "T2": "面壁者早期：被迫成为面壁者，用模糊和逃避保护自己。",
    "T3": "雪地悟道阶段：开始理解黑暗森林基本逻辑。",
    "T4": "威慑建立阶段：把黑暗森林理论转化为现实威慑。",
    "T5": "执剑人时期：作为威慑体系核心执行者。",
    "T6": "晚年时期：历史见证者和反思者。",
}

KNOWLEDGE_MODE_HELP = {
    "temporal": "拟真模式：严格锁定当前 T 阶段知识边界，不联网，不知道未来。",
    "transparent": "通透模式：人格仍锁定当前 T 阶段，但允许联网读取外部资料和未来资料。",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "阶段选择说明：\n"
            + "\n".join(f"  {stage}: {description}" for stage, description in STAGE_HELP.items())
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        default=None,
        choices=VALID_STAGES,
        help="罗辑的时间线阶段。未传入时，脚本启动后会先让你输入 T1/T2/T3/T4/T5/T6。",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="这次要问罗辑的问题。未传入时，脚本会在选择阶段后提示你输入一次问题。",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--novel-top-k",
        type=int,
        default=5,
        help="小说原文补充检索数量。小说检索会自动限制在当前 T 阶段及以前。",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help=f"Use {DEFAULT_REWRITE_MODEL} to rewrite the retrieval query before search.",
    )
    parser.add_argument(
        "--model",
        default=TEST_CHAT_MODEL,
        help="Chat model used by create_agent. Defaults to the project test model.",
    )
    parser.add_argument(
        "--router-model",
        default=DEFAULT_ROUTER_MODEL,
        help="Router LLM model used for intent classification and retrieval policy.",
    )
    parser.add_argument(
        "--disable-router",
        action="store_true",
        help="Disable Router LLM and use the legacy balanced retrieval behavior.",
    )
    parser.add_argument(
        "--knowledge-mode",
        choices=["temporal", "transparent", "拟真", "通透", "0", "1"],
        default=None,
        help="知识模式。未传入时，脚本会在选择 T 阶段后提示你选择 0=拟真 或 1=通透。",
    )
    return parser.parse_args()


def load_env_file(path: Path, *, override: bool = False) -> None:
    """Load simple KEY=VALUE pairs without requiring python-dotenv."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)


def prompt_for_stage() -> str:
    print("请选择罗辑时间段：")
    for stage, description in STAGE_HELP.items():
        print(f"  {stage}: {description}")

    while True:
        stage = input("时间段（T1/T2/T3/T4/T5/T6）：").strip().upper()
        if stage in VALID_STAGES:
            return stage
        print("输入无效，请输入 T1、T2、T3、T4、T5 或 T6。")


def prompt_for_message() -> str:
    while True:
        message = input("你：").strip()
        if message:
            return message
        print("问题不能为空，请重新输入。")


def normalize_knowledge_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized in {"0", "temporal", "拟真", "realistic", "simulation"}:
        return "temporal"
    if normalized in {"1", "transparent", "通透"}:
        return "transparent"
    raise ValueError("知识模式必须是 0/1，或 temporal/transparent，或 拟真/通透。")


def prompt_for_knowledge_mode() -> str:
    print("请选择知识模式：")
    print(f"  0: {KNOWLEDGE_MODE_HELP['temporal']}")
    print(f"  1: {KNOWLEDGE_MODE_HELP['transparent']}")

    while True:
        mode = input("知识模式（0=拟真 / 1=通透）：").strip()
        try:
            return normalize_knowledge_mode(mode)
        except ValueError:
            print("输入无效，请输入 0 或 1。")


def build_agent(stage: str, args: argparse.Namespace):
    try:
        from langchain.agents import create_agent
        from langchain.chat_models import init_chat_model
    except ImportError as error:
        message = (
            "缺少测试脚本依赖。请确认 IDE 使用的是当前项目解释器："
            f"{PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'}，"
            "或者在项目根目录运行：uv run python scripts\\test_luoji_create_agent.py --help"
        )
        raise RuntimeError(message) from error

    load_env_file(PROJECT_ROOT / ".env", override=True)
    os.environ.setdefault("OPENAI_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
    if os.getenv("DASHSCOPE_BASE_URL"):
        os.environ.setdefault("OPENAI_BASE_URL", os.environ["DASHSCOPE_BASE_URL"])
    router_api_key = (
        os.getenv("ROUTER_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("ZHIPUAI_API_KEY")
        or os.getenv("GLM_API_KEY")
    )
    router_base_url = DEFAULT_ROUTER_BASE_URL or os.getenv("OPENAI_BASE_URL")

    luoji = LuoJiAgent()
    stage_profile = luoji.get_stage_profile(stage)
    rewrite_llm = (
        init_chat_model(DEFAULT_REWRITE_MODEL, model_provider="openai")
        if args.rewrite
        else None
    )
    chat_model = init_chat_model(args.model, model_provider="openai")
    router_llm = None
    if not args.disable_router:
        if not router_api_key:
            raise RuntimeError(
                "Router LLM 需要 DASHSCOPE_API_KEY，或用 ROUTER_API_KEY 单独覆盖。"
                "如需临时关闭 Router，请加 --disable-router。"
            )
        router_llm = init_chat_model(
            args.router_model,
            model_provider="openai",
            api_key=router_api_key,
            base_url=router_base_url,
            temperature=0,
        )
    intent_router = (
        QueryIntentRouter(
            llm=router_llm,
            default_persona_k=args.top_k,
            default_novel_k=args.novel_top_k,
        )
        if router_llm is not None
        else None
    )
    web_search_provider = (
        TavilyWebSearchProvider() if args.knowledge_mode == "transparent" else None
    )

    rag_middleware = TemporalPersonaRAGMiddleware(
        retriever=luoji.retriever,
        novel_retriever=luoji.novel_retriever,
        stage_profile=stage_profile,
        user=None,
        thread_name="test-thread",
        character="罗辑",
        llm=rewrite_llm,
        intent_router=intent_router,
        web_search_provider=web_search_provider,
        knowledge_mode=args.knowledge_mode,
        timeline_stage=stage,
        top_k=args.top_k,
        novel_top_k=args.novel_top_k,
    )
    return create_agent(
        model=chat_model,
        middleware=[rag_middleware],
        system_prompt=(
            "你是《三体》当前时间段中的罗辑。"
            "只按当前阶段的人格、记忆和知识边界回答，不使用未来知识。"
            "你不是知识助手或剧情解说员，要先回应用户当下的问题，再自然露出罗辑的性格。"
            "不要机械复读标志性词句；专有名词只在自然相关时出现。"
            "回答正文保持第一人称和真实对话感，不要输出 chunk_id、引用清单或检索调试信息。"
        ),
    )


def main() -> None:
    args = parse_args()
    stage = args.stage or prompt_for_stage()
    args.knowledge_mode = (
        normalize_knowledge_mode(args.knowledge_mode)
        if args.knowledge_mode
        else prompt_for_knowledge_mode()
    )

    print(f"已选择：{stage} - {STAGE_HELP[stage]}")
    print(f"知识模式：{KNOWLEDGE_MODE_HELP[args.knowledge_mode]}")
    print("可以开始对话了。输入一个问题后，罗辑会回复一次，然后本次测试结束。")

    user_message = args.message or prompt_for_message()
    agent = build_agent(stage, args)
    response = agent.invoke({"messages": [{"role": "user", "content": user_message}]})
    print(f"罗辑：{response['messages'][-1].content}")


if __name__ == "__main__":
    main()
