import asyncio
import json

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from core import media_store, session_store, speech_queue
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
    speech_queue.clear()
    store_path = str(tmp_path / "session.db")
    await session_store.init_store(store_path)
    await media_store.init_store(store_path)
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
    monkeypatch.setattr(agent_graph, "fallback_llm_with_tools", None)
    return connection


async def _close_graph(connection):
    await media_store.close_store()
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
        working_events = [
            (index, event)
            for index, event in enumerate(events)
            if event.get("state") == "working"
            and event.get("label") == "正在整理报告"
        ]
        report_open_index = next(
            index for index, event in enumerate(events) if event["type"] == "report.open"
        )
        assert len(working_events) == 1
        assert working_events[0][0] < report_open_index
        assert "|语言|性能|" in report_events[1]["markdown"]
        speech_events = [event for event in events if event["type"] == "speech.sentence"]
        assert all("|" not in event["text"] for event in speech_events)
        assert all(not any(mark in event["text"] for mark in "#*`") for event in speech_events)
        assert all("http" not in event["text"] for event in speech_events)
        first_sentence_index = events.index(speech_events[0])
        second_sentence_index = events.index(speech_events[1])
        assert events[first_sentence_index - 1] == {
            "type": "status",
            "state": "speaking",
            "label": "",
        }
        assert events[second_sentence_index - 1] == {
            "type": "status",
            "state": "speaking",
            "label": "",
        }
        assert next(event for event in events if event["type"] == "tool")["name"] == "show_report"
        state = await agent_graph.app_graph.aget_state(
            {"configurable": {"thread_id": "protocol-report"}}
        )
        ai_messages = [
            message
            for message in state.values["messages"]
            if isinstance(message, AIMessage)
        ]
        assert ai_messages[0].content == "我整理一下。"
        assert ai_messages[-1].content == "三种语言各有取舍。我建议按项目需求选择。你更看重哪一项？"
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
        state = await agent_graph.app_graph.aget_state(
            {"configurable": {"thread_id": "protocol-overflow"}}
        )
        last_ai_message = next(
            message
            for message in reversed(state.values["messages"])
            if isinstance(message, AIMessage)
        )
        report_id = report_events[0]["report_id"]
        assert last_ai_message.content == (
            f"详细内容我放在屏幕上了。\n[完整内容已作为报告 {report_id} 上屏]"
        )
    finally:
        await _close_graph(connection)


@pytest.mark.asyncio
async def test_wrong_interruption_turn_is_ignored(monkeypatch, tmp_path):
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
            async for event in agent_graph.process_query(
                "说一说", "protocol-wrong-interrupt"
            )
        ]
        turn_id = first_events[0]["turn_id"]
        generator = agent_graph.process_query(
            "刚才说到哪？",
            "protocol-wrong-interrupt",
            interrupted={
                "turn_id": "t_wrong",
                "spoken_sentence_ids": ["t_wrong.1"],
            },
        )
        assert (await anext(generator))["type"] == "turn"
        await generator.aclose()

        state = await agent_graph.app_graph.aget_state(
            {"configurable": {"thread_id": "protocol-wrong-interrupt"}}
        )
        prior_message = next(
            message
            for message in state.values["messages"]
            if isinstance(message, AIMessage) and not message.tool_calls
        )
        assert prior_message.content == "第一句。第二句。"
        utterances = await session_store.recent_utterances("protocol-wrong-interrupt")
        prior_assistant = next(
            item
            for item in utterances
            if item["role"] == "assistant" and item["turn_id"] == turn_id
        )
        assert prior_assistant["text"] == "第一句。第二句。"
        assert prior_assistant["interrupted"] == 0
    finally:
        await _close_graph(connection)


@pytest.mark.asyncio
async def test_missing_interruption_sentence_preserves_checkpoint_and_utterance(
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
            async for event in agent_graph.process_query(
                "说一说", "protocol-missing-interrupt"
            )
        ]
        turn_id = first_events[0]["turn_id"]
        first_sentence = next(
            event for event in first_events if event["type"] == "speech.sentence"
        )
        speech_queue.clear()
        generator = agent_graph.process_query(
            "刚才说到哪？",
            "protocol-missing-interrupt",
            interrupted={
                "turn_id": turn_id,
                "spoken_sentence_ids": [first_sentence["id"]],
            },
        )
        assert (await anext(generator))["type"] == "turn"
        await generator.aclose()

        state = await agent_graph.app_graph.aget_state(
            {"configurable": {"thread_id": "protocol-missing-interrupt"}}
        )
        prior_message = next(
            message
            for message in state.values["messages"]
            if isinstance(message, AIMessage) and not message.tool_calls
        )
        assert prior_message.content == "第一句。第二句。"
        utterances = await session_store.recent_utterances("protocol-missing-interrupt")
        prior_assistant = next(
            item
            for item in utterances
            if item["role"] == "assistant" and item["turn_id"] == turn_id
        )
        assert prior_assistant["text"] == "第一句。第二句。"
        assert prior_assistant["interrupted"] == 1
    finally:
        await _close_graph(connection)


@pytest.mark.asyncio
async def test_invalid_highlight_anchor_does_not_dispatch_event(monkeypatch, tmp_path):
    connection = await _install_graph(monkeypatch, tmp_path, [])
    try:
        await session_store.create_report(
            "protocol-highlight", "报告", "问题", "原文中包含有效标题。"
        )
        report = (await session_store.list_reports("protocol-highlight"))[0]
        scripted_model = ScriptedChatModel(
            messages=iter(
                [
                AIMessage(
                    content="我帮你看一下。",
                    tool_calls=[
                        {
                            "name": "highlight_report",
                            "args": {
                                "report_id": report["report_id"],
                                "anchor": "missing phrase",
                            },
                            "id": "highlight_2",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="报告里没有这段文字。"),
                ]
            )
        )
        monkeypatch.setattr(agent_graph, "llm_with_tools", scripted_model)
        events = [
            event
            async for event in agent_graph.process_query(
                "高亮一段", "protocol-highlight"
            )
        ]
        assert not any(event["type"] == "report.highlight" for event in events)
        assert events[-1]["spoken_text"] == "我帮你看一下。报告里没有这段文字。"
    finally:
        await _close_graph(connection)


class FailingChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for char in "第一句。第二句":
            yield ChatGenerationChunk(message=AIMessageChunk(content=char))
        raise RuntimeError("scripted stream failure")


@pytest.mark.asyncio
async def test_stream_failure_persists_sentences_already_emitted(monkeypatch, tmp_path):
    connection = await _install_graph(monkeypatch, tmp_path, [])
    failing_model = FailingChatModel(messages=iter([]))
    monkeypatch.setattr(agent_graph, "llm_with_tools", failing_model)
    monkeypatch.setattr(agent_graph, "fallback_llm_with_tools", None)
    try:
        with pytest.raises(RuntimeError, match="scripted stream failure"):
            async for _ in agent_graph.process_query(
                "说一说", "protocol-stream-error"
            ):
                pass
        utterances = await session_store.recent_utterances("protocol-stream-error")
        assistant = next(item for item in utterances if item["role"] == "assistant")
        assert assistant["text"] == "第一句。"
    finally:
        await _close_graph(connection)


@pytest.mark.asyncio
async def test_disconnect_persists_emitted_sentence_as_interrupted(monkeypatch, tmp_path):
    connection = await _install_graph(
        monkeypatch,
        tmp_path,
        [AIMessage(content="第一句。第二句")],
    )
    try:
        generator = agent_graph.process_query(
            "说两句", "protocol-disconnect"
        )
        first_sentence = None
        async for event in generator:
            if event["type"] == "speech.sentence":
                first_sentence = event
                break
        assert first_sentence is not None

        await asyncio.sleep(0.05)
        await generator.aclose()

        assistants = [
            item
            for item in await session_store.recent_utterances("protocol-disconnect")
            if item["role"] == "assistant"
        ]
        assert len(assistants) == 1
        assert assistants[0]["text"] == first_sentence["text"]
        assert assistants[0]["interrupted"] == 1
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


@pytest.mark.asyncio
async def test_interruption_rewrites_only_the_last_spoken_tool_segment(
    monkeypatch, tmp_path
):
    report_call = {
        "name": "show_report",
        "args": {"title": "摘要", "markdown": "第一条\n第二条"},
        "id": "segment_report",
        "type": "tool_call",
    }
    connection = await _install_graph(
        monkeypatch,
        tmp_path,
        [
            AIMessage(content="我先整理一下。", tool_calls=[report_call]),
            AIMessage(content="报告放屏幕上了。重点有两条。你看看。"),
            AIMessage(content="新一轮回复。"),
            AIMessage(content="我先整理一下。", tool_calls=[report_call]),
            AIMessage(content="报告放屏幕上了。重点有两条。你看看。"),
            AIMessage(content="新一轮回复。"),
        ],
    )
    try:
        async def interrupt_after(thread_id: str, sentence_count: int):
            first_events = [
                event
                async for event in agent_graph.process_query(
                    "做个摘要", thread_id
                )
            ]
            turn_id = first_events[0]["turn_id"]
            sentence_ids = [
                event["id"]
                for event in first_events
                if event["type"] == "speech.sentence"
            ][:sentence_count]
            second_events = [
                event
                async for event in agent_graph.process_query(
                    "刚才说到哪？",
                    thread_id,
                    interrupted={
                        "turn_id": turn_id,
                        "spoken_sentence_ids": sentence_ids,
                    },
                )
            ]
            state = await agent_graph.app_graph.aget_state(
                {"configurable": {"thread_id": thread_id}}
            )
            return first_events, second_events, state

        first_events, _, state = await interrupt_after(
            "protocol-segment-interrupt", 2
        )
        first_sentence, second_sentence = [
            event
            for event in first_events
            if event["type"] == "speech.sentence"
        ][:2]
        checkpoint = next(
            message
            for message in state.values["messages"]
            if isinstance(message, AIMessage)
            and message.content.startswith("报告放屏幕上了。")
        )
        assert checkpoint.content == (
            second_sentence["text"] + "……（说到这里被用户打断）"
        )
        utterances = await session_store.recent_utterances(
            "protocol-segment-interrupt"
        )
        assistant = next(
            item
            for item in utterances
            if item["role"] == "assistant"
            and item["turn_id"] == first_events[0]["turn_id"]
        )
        assert assistant["text"] == (
            first_sentence["text"]
            + second_sentence["text"]
            + "……（说到这里被用户打断）"
        )
        assert assistant["interrupted"] == 1

        first_events, _, state = await interrupt_after(
            "protocol-earlier-segment-only", 1
        )
        checkpoint = next(
            message
            for message in state.values["messages"]
            if isinstance(message, AIMessage)
            and message.content == "（还没开口就被用户打断了）"
        )
        assert checkpoint.content == "（还没开口就被用户打断了）"
        utterances = await session_store.recent_utterances(
            "protocol-earlier-segment-only"
        )
        assistant = next(
            item
            for item in utterances
            if item["role"] == "assistant"
            and item["turn_id"] == first_events[0]["turn_id"]
        )
        assert assistant["text"] == (
            "我先整理一下。……（说到这里被用户打断）"
        )
        assert assistant["interrupted"] == 1
    finally:
        await _close_graph(connection)


@pytest.mark.asyncio
async def test_tool_start_advances_interruption_segment_before_tool_finishes(
    monkeypatch, tmp_path
):
    report_call = {
        "name": "show_report",
        "args": {"title": "摘要", "markdown": "第一条\n第二条"},
        "id": "slow_report",
        "type": "tool_call",
    }
    connection = await _install_graph(
        monkeypatch,
        tmp_path,
        [
            AIMessage(content="我先整理一下。", tool_calls=[report_call]),
            AIMessage(content="报告已经整理好了。"),
        ],
    )
    original_create_report = session_store.create_report
    report_started = asyncio.Event()

    async def slow_create_report(*args, **kwargs):
        report_started.set()
        await asyncio.sleep(5)
        return await original_create_report(*args, **kwargs)

    monkeypatch.setattr(session_store, "create_report", slow_create_report)
    thread_id = "protocol-tool-start-disconnect"
    generator = agent_graph.process_query("做个摘要", thread_id)
    first_sentence = None
    turn_id = None
    try:
        async for event in generator:
            if event["type"] == "turn":
                turn_id = event["turn_id"]
            elif event["type"] == "speech.sentence":
                first_sentence = event
            elif (
                event["type"] == "tool"
                and event["name"] == "show_report"
                and event["status"] == "start"
            ):
                await asyncio.wait_for(report_started.wait(), timeout=1)
                break

        assert first_sentence is not None
        await generator.aclose()

        interruption = agent_graph.process_query(
            "刚才说到哪？",
            thread_id,
            interrupted={
                "turn_id": turn_id,
                "spoken_sentence_ids": [first_sentence["id"]],
            },
        )
        assert (await anext(interruption))["type"] == "turn"
        await interruption.aclose()

        state = await agent_graph.app_graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        last_ai_message = next(
            message
            for message in reversed(state.values["messages"])
            if isinstance(message, AIMessage)
        )
        assert last_ai_message.content == "（还没开口就被用户打断了）"
    finally:
        await generator.aclose()
        await _close_graph(connection)


@pytest.mark.asyncio
async def test_disconnect_before_speech_does_not_persist_empty_assistant(
    monkeypatch, tmp_path
):
    connection = await _install_graph(
        monkeypatch, tmp_path, [AIMessage(content="稍后再说。")]
    )
    generator = agent_graph.process_query(
        "稍后", "protocol-empty-disconnect"
    )
    try:
        assert (await anext(generator))["type"] == "turn"
        await generator.aclose()
        utterances = await session_store.recent_utterances(
            "protocol-empty-disconnect"
        )
        assert not any(item["role"] == "assistant" for item in utterances)
    finally:
        await generator.aclose()
        await _close_graph(connection)
