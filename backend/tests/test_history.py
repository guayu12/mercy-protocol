import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from core.agent.history import prepare_history
from core.models import ModelSpec


def _response(provider="openai", model="gpt-a", api="responses"):
    return AIMessage(
        content=[
            {"type": "reasoning", "id": "rs_provider_id", "summary": []},
            {"type": "text", "id": "msg_provider_id", "text": "已完成"},
            {
                "type": "function_call",
                "call_id": "fc_provider_id",
                "name": "lookup",
                "arguments": "{}",
            },
        ],
        tool_calls=[{"name": "lookup", "args": {}, "id": "call-preserved"}],
        additional_kwargs={"reasoning_content": "provider-private"},
        response_metadata={
            "mercy_source": {"provider": provider, "model": model, "api": api}
        },
    )


def _payload(spec, messages):
    options = {
        "model": spec.model,
        "api_key": "test-provider-key",
        "base_url": (
            "https://api.deepseek.com"
            if spec.provider == "deepseek"
            else "https://api.openai.com/v1"
        ),
    }
    if spec.provider == "deepseek" and spec.effort in {"low", "high", "max"}:
        options.update(
            reasoning={"effort": spec.effort},
            use_responses_api=True,
        )
    elif spec.provider == "deepseek":
        options["extra_body"] = {"thinking": {"type": "disabled"}}
    elif spec.effort is not None:
        options["reasoning"] = {"effort": spec.effort}
    return ChatOpenAI(**options)._get_request_payload(messages)


def test_openai_responses_history_rebuilds_for_deepseek_chat_and_responses():
    original = _response()
    tool = ToolMessage(content="lookup result", tool_call_id="call-preserved", name="lookup")
    human = HumanMessage(
        content=[
            {"type": "text", "text": "看图"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ],
        additional_kwargs={"mercy_media_ids": ["m_image"]},
    )
    before = original.model_dump()

    for spec in (
        ModelSpec("deepseek", "deepseek-chat", "none"),
        ModelSpec("deepseek", "deepseek-reasoner", "high"),
    ):
        prepared = prepare_history(
            [original, tool, human],
            spec,
            image_input=True,
            inspect_image_available=True,
        )
        payload = _payload(spec, prepared)
        serialized = json.dumps(payload)
        assert all(
            identifier not in serialized
            for identifier in ("rs_provider_id", "fc_provider_id", "msg_provider_id")
        )
        assert '"type": "reasoning"' not in serialized
        assert "call-preserved" in serialized
        assert "data:image/png;base64,AAAA" in serialized
        assert prepared[1] is tool
    assert original.model_dump() == before


def test_deepseek_responses_history_rebuilds_for_openai_and_model_changes():
    original = _response("deepseek", "deepseek-reasoner", "responses")
    for target in (
        ModelSpec("openai", "gpt-b", "high"),
        ModelSpec("openai", "gpt-c", "high"),
    ):
        prepared = prepare_history(
            [original], target, image_input=True, inspect_image_available=False
        )
        payload = _payload(target, prepared)
        serialized = json.dumps(payload)
        assert "rs_provider_id" not in serialized
        assert "fc_provider_id" not in serialized
        assert "msg_provider_id" not in serialized
        assert prepared[0] is not original


def test_openai_same_provider_different_model_history_is_rebuilt():
    original = _response("openai", "gpt-a", "responses")
    target = ModelSpec("openai", "gpt-b", "high")
    prepared = prepare_history(
        [original], target, image_input=True, inspect_image_available=False
    )
    serialized = json.dumps(_payload(target, prepared))
    assert "rs_provider_id" not in serialized
    assert "fc_provider_id" not in serialized
    assert prepared[0] is not original


def test_same_spec_and_untagged_openai_messages_are_preserved():
    same = _response("openai", "gpt-a", "responses")
    untagged = AIMessage(content="legacy answer")
    target = ModelSpec("openai", "gpt-a", "high")
    prepared = prepare_history(
        [same, untagged], target, image_input=True, inspect_image_available=False
    )
    assert prepared[0] is same
    assert prepared[1] is untagged


def test_untagged_deepseek_history_is_rebuilt_without_provider_metadata():
    original = AIMessage(
        content=[
            {"type": "reasoning", "id": "rs_old", "summary": []},
            {"type": "text", "text": "旧答复"},
        ],
        additional_kwargs={"reasoning_content": "not for another provider"},
    )
    prepared = prepare_history(
        [original],
        ModelSpec("deepseek", "deepseek-chat", None),
        image_input=True,
        inspect_image_available=False,
    )
    assert prepared[0].content == "旧答复"
    assert prepared[0].additional_kwargs == {}
    assert prepared[0].response_metadata == {}
    assert prepared[0] is not original
    assert original.content[0]["id"] == "rs_old"


def test_unsupported_images_become_safe_id_aligned_placeholders():
    original = HumanMessage(
        content=[
            {"type": "text", "text": "请看"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
        ],
        additional_kwargs={"mercy_media_ids": ["m_one", "m_two"]},
    )
    prepared = prepare_history(
        [original],
        ModelSpec("deepseek", "deepseek-chat", None),
        image_input=False,
        inspect_image_available=True,
    )
    assert prepared[0].content[1]["text"] == (
        "［用户发了一张图片 m_one，当前对话模型看不到图片内容；需要了解时用 inspect_image］"
    )
    assert prepared[0].content[2]["text"] == (
        "［用户发了一张图片 m_two，当前对话模型看不到图片内容；需要了解时用 inspect_image］"
    )
    assert original.content[1]["type"] == "image_url"

    no_inspect = prepare_history(
        [original],
        ModelSpec("deepseek", "deepseek-chat", None),
        image_input=False,
        inspect_image_available=False,
    )
    assert "请如实告诉用户" in no_inspect[0].content[1]["text"]
