from langchain_core.messages import AIMessage, HumanMessage

from core import models


def _text_content(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _matching_source(message: AIMessage, target_spec: models.ModelSpec) -> bool:
    source = (message.response_metadata or {}).get("mercy_source")
    return (
        isinstance(source, dict)
        and source.get("provider") == target_spec.provider
        and source.get("model") == target_spec.model
        and source.get("api") == models.api_kind(target_spec)
    )


def _rebuild_ai_message(message: AIMessage) -> AIMessage:
    tool_calls = [
        {
            "name": call["name"],
            "args": call["args"],
            "id": call["id"],
        }
        for call in (getattr(message, "tool_calls", None) or [])
        if isinstance(call, dict)
        and call.get("name")
        and isinstance(call.get("args"), dict)
        and call.get("id")
    ]
    return AIMessage(content=_text_content(message.content), tool_calls=tool_calls)


def _image_placeholder(media_id: str | None, inspect_image_available: bool) -> str:
    identifier = f" {media_id}" if media_id else ""
    if inspect_image_available:
        return (
            f"［用户发了一张图片{identifier}，当前对话模型看不到图片内容；"
            "需要了解时用 inspect_image］"
        )
    return (
        f"［用户发了一张图片{identifier}，当前对话模型看不到图片内容，"
        "请如实告诉用户］"
    )


def _replace_unsupported_images(
    message: HumanMessage, inspect_image_available: bool
) -> HumanMessage:
    if not isinstance(message.content, list):
        return message
    media_ids = (message.additional_kwargs or {}).get("mercy_media_ids", [])
    if not isinstance(media_ids, list):
        media_ids = []
    image_index = 0
    content = []
    replaced = False
    for block in message.content:
        if isinstance(block, dict) and block.get("type") == "image_url":
            media_id = media_ids[image_index] if image_index < len(media_ids) else None
            content.append(
                {
                    "type": "text",
                    "text": _image_placeholder(media_id, inspect_image_available),
                }
            )
            image_index += 1
            replaced = True
        else:
            content.append(block)
    return (
        message.model_copy(deep=True, update={"content": content})
        if replaced
        else message
    )


def prepare_history(
    messages,
    target_spec: models.ModelSpec,
    *,
    image_input: bool,
    inspect_image_available: bool,
) -> list:
    prepared = []
    for message in messages:
        if isinstance(message, AIMessage):
            if _matching_source(message, target_spec):
                prepared.append(message)
            elif (
                (message.response_metadata or {}).get("mercy_source") is None
                and target_spec.provider == "openai"
            ):
                prepared.append(message)
            else:
                prepared.append(_rebuild_ai_message(message))
        elif isinstance(message, HumanMessage) and not image_input:
            prepared.append(
                _replace_unsupported_images(message, inspect_image_available)
            )
        else:
            prepared.append(message)
    return prepared
