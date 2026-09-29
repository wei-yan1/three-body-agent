import asyncio

from app.agents.storyrole.streaming import (
    close_answer_stream,
    open_answer_stream,
    publish_answer_token,
    reset_answer_stream,
)


def test_local_answer_stream_preserves_control_events(monkeypatch):
    monkeypatch.setenv("STORYROLE_STREAM_TRANSPORT", "local")

    async def read_events():
        stream = open_answer_stream("stream-test")
        publish_answer_token("stream-test", "partial")
        reset_answer_stream("stream-test")
        close_answer_stream("stream-test")
        return [await stream.get(), await stream.get(), await stream.get()]

    events = asyncio.run(read_events())
    assert [event["type"] for event in events] == ["token", "reset", "done"]
    assert events[0]["text"] == "partial"
