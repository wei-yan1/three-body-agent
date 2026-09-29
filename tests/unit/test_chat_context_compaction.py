from app.services.chat_context_compaction import _estimate_tokens, _tail


def _turns(count: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for index in range(count):
        messages.extend([
            {"role": "user", "content": f"问题 {index}"},
            {"role": "assistant", "content": f"回答 {index}"},
        ])
    return messages


def test_tail_keeps_at_most_ten_messages_and_complete_turn_boundary() -> None:
    prefix, kept = _tail(_turns(8), max_messages=10, budget=100_000)

    assert len(kept) <= 10
    assert len(prefix) % 2 == 0
    assert kept[0]["role"] == "user"


def test_tail_respects_token_budget() -> None:
    messages = _turns(8)
    prefix, kept = _tail(messages, max_messages=10, budget=20)

    assert prefix
    assert len(kept) <= 10
    assert sum(_estimate_tokens(item["content"]) + 4 for item in kept) <= 20
