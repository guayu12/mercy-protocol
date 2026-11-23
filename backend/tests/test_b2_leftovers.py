from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage

from core import gemini_files, models
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


def test_gemini_api_key_invalid_400_is_distinct_from_bad_request():
    body = {
        "error": {
            "code": 400,
            "message": "API key not valid. Please pass a valid API key.",
            "status": "INVALID_ARGUMENT",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "API_KEY_INVALID",
                    "domain": "googleapis.com",
                }
            ],
        }
    }
    request = httpx.Request("GET", "https://example.invalid")
    response = httpx.Response(400, json=body, request=request)
    error = httpx.HTTPStatusError(
        "fake Gemini error", request=request, response=response
    )
    plain_response = httpx.Response(
        400,
        json={"error": {"status": "INVALID_ARGUMENT"}},
        request=request,
    )
    plain_error = httpx.HTTPStatusError(
        "fake bad request", request=request, response=plain_response
    )

    assert classify(error).category == "invalid_key"
    assert classify(error).message == "key 无效"
    assert classify(plain_error).category == "bad_request"
    with pytest.raises(gemini_files.GeminiError) as wrapped:
        gemini_files._check_response(response)
    assert classify(wrapped.value).category == "invalid_key"
    with pytest.raises(gemini_files.GeminiRejected) as rejected:
        gemini_files._check_response(plain_response, reject_400=True)
    assert classify(rejected.value).category == "bad_request"

    sdk_error = type(
        "GeminiSdkError",
        (Exception,),
        {"status_code": 400, "body": body},
    )()
    assert classify(sdk_error).category == "invalid_key"
