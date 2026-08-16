import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.api.v1 import chat


def parse_sse_frame(frame: str) -> dict:
    assert frame.startswith("data: ")
    return json.loads(frame.removeprefix("data: ").strip())


@pytest.mark.asyncio
async def test_stream_sends_heartbeats_and_only_the_top_level_final_message(monkeypatch):
    class FakeAgent:
        async def ainvoke(self, input_data, config):
            await asyncio.sleep(0.03)
            return {
                "messages": [
                    *input_data["messages"],
                    AIMessage(content="顶层最终回复"),
                ]
            }

    saved_messages = []

    async def fake_create_travel_agent():
        return FakeAgent()

    async def fake_save_message(db, conversation_id, role, content, metadata=None):
        saved_messages.append((role, content))

    monkeypatch.setattr(chat, "create_travel_agent", fake_create_travel_agent)
    monkeypatch.setattr(chat, "save_message", fake_save_message)
    monkeypatch.setattr(chat, "HEARTBEAT_INTERVAL_SECONDS", 0.005)
    monkeypatch.setattr(
        chat,
        "build_agent_trace_config",
        lambda conversation_id: {"configurable": {"thread_id": conversation_id}},
    )

    frames = [
        parse_sse_frame(frame)
        async for frame in chat.generate_sse_stream(
            "conversation-1",
            "用户问题",
            db=object(),
            user=SimpleNamespace(id="user-1"),
        )
    ]

    assert any(frame["type"] == "heartbeat" for frame in frames)
    assert [frame for frame in frames if frame["type"] == "token"] == [
        {"type": "token", "content": "顶层最终回复"}
    ]
    assert frames[-1] == {"type": "done"}
    assert saved_messages == [
        ("user", "用户问题"),
        ("assistant", "顶层最终回复"),
    ]


@pytest.mark.asyncio
async def test_stream_reports_missing_final_ai_message(monkeypatch):
    class FakeAgent:
        async def ainvoke(self, input_data, config):
            return {"messages": [HumanMessage(content="只有用户消息")]}

    async def fake_create_travel_agent():
        return FakeAgent()

    async def fake_save_message(db, conversation_id, role, content, metadata=None):
        return None

    monkeypatch.setattr(chat, "create_travel_agent", fake_create_travel_agent)
    monkeypatch.setattr(chat, "save_message", fake_save_message)

    frames = [
        parse_sse_frame(frame)
        async for frame in chat.generate_sse_stream(
            "conversation-1",
            "用户问题",
            db=object(),
            user=SimpleNamespace(id="user-1"),
        )
    ]

    assert frames == [
        {"type": "error", "message": "Agent 未返回有效的最终回复"}
    ]
