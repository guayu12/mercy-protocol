import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from core import session_store
from core.agent import graph as agent_graph


class ScriptedChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        message = next(self.messages)
        if isinstance(message, str):
            message = AIMessage(content=message)
        content = message.content if isinstance(message.content, str) else ""
        for char in content:
            yield ChatGenerationChunk(message=AIMessageChunk(content=char))
        for index, call in enumerate(message.tool_calls):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    id=message.id,
                    tool_call_chunks=[
                        {
                            "name": call["name"],
                            "args": json.dumps(call["args"]),
                            "id": call["id"],
                            "index": index,
                        }
                    ],
                )
            )


async def _install_graph(monkeypatch, tmp_path, responses):
    await session_store.init_store(str(tmp_path / "session.db"))
    connection = await agent_graph.aiosqlite.connect(str(tmp_path / "checkpoints.db"))
    checkpointer = AsyncSqliteSaver(connection)
    await checkpointer.setup()
    agent_graph.app_graph = agent_graph.workflow.compile(checkpointer=checkpointer)

    async def initialized_graph():
        return agent_graph.app_graph

    monkeypatch.setattr(agent_graph, "init_graph", initialized_graph)
    monkeypatch.setattr(
        agent_graph.memory_manager, "format_context", lambda query: ""
    )
    fake = ScriptedChatModel(messages=iter(responses))
    monkeypatch.setattr(agent_graph, "llm_with_tools", fake)
    monkeypatch.setattr(agent_graph, "vision_llm_with_tools", fake)
    return connection


async def _close_graph(connection):
    await session_store.close_store()
    await connection.close()
    agent_graph.app_graph = None


@pytest.mark.asyncio
async def test_process_query_splits_reports_from_speech(monkeypatch, tmp_path):
    connection = await _install_graph(
        monkeypatch,
        tmp_path,
        [
            AIMessage(
                content="我整理一下。",
                tool_calls=[
                    {
                        "name": "show_report",
                        "args": {
                            "title": "语言对比",
                            "markdown": "|语言|性能|\n|---|---|\n|Python|易学|",
                        },
                        "id": "show_1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="三种语言各有取舍。我建议按项目需求选择。你更看重哪一项？"),
        ],
    )
    try:
        events = [
            event
            async for event in agent_graph.process_query("对比语言", "protocol-report")
        ]
        assert events[0]["type"] == "turn"
        assert events[1] == {"type": "status", "state": "thinking", "label": "正在思考"}
        assert events[-1]["type"] == "done"
        assert events[-1]["spoken_text"] == "我整理一下。三种语言各有取舍。我建议按项目需求选择。你更看重哪一项？"
        report_events = [event for event in events if event["type"].startswith("report.")]
        assert [event["type"] for event in report_events] == [
            "report.open",
            "report.delta",
            "report.done",
        ]
        assert "|语言|性能|" in report_events[1]["markdown"]
        speech_events = [event for event in events if event["type"] == "speech.sentence"]
        assert all("|" not in event["text"] for event in speech_events)
        assert all(not any(mark in event["text"] for mark in "#*`") for event in speech_events)
        assert all("http" not in event["text"] for event in speech_events)
        assert next(event for event in events if event["type"] == "tool")["name"] == "show_report"
    finally:
        await _close_graph(connection)


@pytest.mark.asyncio
async def test_process_query_moves_overflow_to_a_report(monkeypatch, tmp_path):
    markdown = "|语言|说明|\n|---|---|\n|Rust|高性能|"
    connection = await _install_graph(
        monkeypatch, tmp_path, [AIMessage(content=markdown)]
    )
    try:
        events = [
            event
            async for event in agent_graph.process_query("做个表格", "protocol-overflow")
        ]
        report_events = [event for event in events if event["type"].startswith("report.")]
        assert [event["type"] for event in report_events] == [
            "report.open",
            "report.delta",
            "report.done",
        ]
        assert report_events[1]["markdown"] == markdown
        assert [
            event["text"]
            for event in events
            if event["type"] == "speech.sentence"
        ] == ["详细内容我放在屏幕上了。"]
        assert events[-1]["spoken_text"] == "详细内容我放在屏幕上了。"
    finally:
        await _close_graph(connection)


@pytest.mark.asyncio
async def test_next_turn_rewrites_checkpoint_to_spoken_interruption(
    monkeypatch, tmp_path
):
    connection = await _install_graph(
        monkeypatch,
        tmp_path,
        [
            AIMessage(content="第一句。第二句。"),
            AIMessage(content="我接着说。"),
        ],
    )
    try:
        first_events = [
            event
            async for event in agent_graph.process_query("说一说", "protocol-interrupt")
        ]
        turn_id = first_events[0]["turn_id"]
        first_sentence = next(
            event for event in first_events if event["type"] == "speech.sentence"
        )
        generator = agent_graph.process_query(
            "刚才说到哪？",
            "protocol-interrupt",
            interrupted={
                "turn_id": turn_id,
                "spoken_sentence_ids": [first_sentence["id"]],
            },
        )
        await anext(generator)
        await generator.aclose()
        state = await agent_graph.app_graph.aget_state(
            {"configurable": {"thread_id": "protocol-interrupt"}}
        )
        interrupted_message = next(
            message
            for message in state.values["messages"]
            if isinstance(message, AIMessage)
            and "说到这里被用户打断" in str(message.content)
        )
        assert interrupted_message.content == "第一句。……（说到这里被用户打断）"
        utterances = await session_store.recent_utterances("protocol-interrupt")
        assert utterances[1]["text"] == "第一句。……（说到这里被用户打断）"
        assert utterances[1]["interrupted"] == 1
    finally:
        await _close_graph(connection)
