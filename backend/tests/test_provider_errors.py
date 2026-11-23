import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph._internal._constants import CONFIG_KEY_RUNTIME
from langgraph.prebuilt import ToolNode
from langgraph.runtime import DEFAULT_RUNTIME

from core import provider_errors
from core.agent import graph, search
from conftest import make_test_runtime


class LeakyError(Exception):
    status_code = 401
    body = {"error": "sk-FAKEKEY1234567890abcdef boom"}

    def __str__(self):
        return "sk-FAKEKEY1234567890abcdef boom"


def test_provider_error_classification_never_uses_exception_text():
    error = LeakyError()
    classified = provider_errors.classify(error)
    assert classified.message == "key 无效"
    assert "FAKEKEY" not in classified.message
    assert "boom" not in classified.message


@pytest.mark.asyncio
async def test_chat_sse_and_log_redact_provider_error(monkeypatch):
    import main

    async def no_links(*_args):
        return None

    async def failed_query(*_args):
        raise LeakyError()
        yield

    monkeypatch.setattr(main, "_register_message_links", no_links)
    monkeypatch.setattr(main, "process_query", failed_query)
    queue = main.log_bus.subscribe()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/chat", json={"message": "hello", "thread_id": "redaction"}
            )
        log = await queue.get()
        assert "模型调用失败：key 无效" in response.text
        assert "sk-FAKEKEY1234567890abcdef" not in response.text
        assert "boom" not in response.text
        assert "sk-FAKEKEY1234567890abcdef" not in log
        assert "boom" not in log
        assert "LeakyError" in log
    finally:
        main.log_bus.unsubscribe(queue)


@pytest.mark.asyncio
async def test_toolnode_errors_and_stale_tool_calls_are_safe(monkeypatch):
    execution = []

    @tool
    async def failing_tool(value: str) -> str:
        """Raise a provider error for redaction testing."""
        execution.append(value)
        raise LeakyError()

    @tool
    async def unavailable_tool(value: str) -> str:
        """Track whether a stale tool call is executed."""
        execution.append(value)
        return "should not execute"

    runtime = make_test_runtime(set())
    monkeypatch.setattr(graph, "runtime_from", lambda _config: runtime)
    config = {
        "configurable": {
            CONFIG_KEY_RUNTIME: DEFAULT_RUNTIME,
            "runtime": runtime,
        }
    }

    error_node = ToolNode([failing_tool], handle_tool_errors=graph._safe_tool_error)
    error_result = await error_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "failing_tool", "args": {"value": "x"}, "id": "call-1"}
                    ],
                )
            ]
        },
        config,
    )
    error_text = error_result["messages"][0].content
    assert error_text == "工具执行失败：key 无效。请口头告诉用户。"
    assert "FAKEKEY" not in error_text and "boom" not in error_text

    stale_node = ToolNode(
        [unavailable_tool],
        awrap_tool_call=graph._runtime_tool_gate,
    )
    stale_result = await stale_node.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "unavailable_tool",
                            "args": {"value": "x"},
                            "id": "call-2",
                        }
                    ],
                )
            ]
        },
        config,
    )
    assert stale_result["messages"][0].content == "没有这个工具。"
    assert "unavailable_tool" not in stale_result["messages"][0].content
    assert execution == ["x"]


@pytest.mark.asyncio
async def test_tavily_error_redaction(monkeypatch):
    class FailingSearch:
        async def ainvoke(self, *_args, **_kwargs):
            raise LeakyError()

    monkeypatch.setattr(search, "_web_search", lambda: FailingSearch())
    result = await search.tavily_raw(
        "query", {"configurable": {"runtime": make_test_runtime(set())}}
    )
    assert result == {"error": "联网搜索失败：key 无效"}
    assert "FAKEKEY" not in str(result)
    assert "boom" not in str(result)
