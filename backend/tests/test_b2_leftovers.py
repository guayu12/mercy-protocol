from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from core import models
from core.agent import graph
from core.provider_errors import classify
import main


@pytest.mark.asyncio
async def test_unknown_tool_gate_returns_fixed_message():
    request = SimpleNamespace(
        tool_call={"name": "secret_unknown_tool", "id": "call-1"}
    )

    async def handler(_request):
        pytest.fail("unknown tools must not reach ToolNode")

    result = await graph._runtime_tool_gate(request, handler)

    assert result.content == "没有这个工具。"
    assert "secret_unknown_tool" not in result.content
    assert result.tool_call_id == "call-1"


def test_responses_api_marker_overrides_model_spec_fallback():
    spec = models.ModelSpec("openai", "fake-model", None)

    assert graph._actual_api_kind(
        AIMessage(content="ok", response_metadata={"id": "resp_fake"}), spec
    ) == "responses"
    assert graph._actual_api_kind(
        AIMessage(content="ok", response_metadata={"id": "chatcmpl-fake"}),
        spec,
    ) == "chat"
    assert graph._actual_api_kind(AIMessage(content="ok"), spec) == "chat"


def test_loopback_addresses_include_ipv4_mapped_ipv6():
    assert main._loopback_host("127.0.0.1")
    assert main._loopback_host("127.255.255.255")
    assert main._loopback_host("::1")
    assert main._loopback_host("::ffff:127.0.0.1")
    assert not main._loopback_host("192.0.2.1")


def test_bad_request_error_has_fixed_provider_classification():
    error = type("BadRequest", (Exception,), {"status_code": 400})()

    assert classify(error).category == "bad_request"
    assert classify(error).message == "模型不接受这个设置"
