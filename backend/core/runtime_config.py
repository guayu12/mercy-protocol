import os

from core.models import PROVIDER_KEYS, get_model_spec, role_available


def _key_status(names: tuple[str, ...]) -> dict:
    value = next(
        (os.environ.get(name, "").strip() for name in names if os.environ.get(name, "").strip()),
        "",
    )
    return {
        "configured": bool(value),
        "hint": f"••••{value[-4:]}" if len(value) >= 20 else None,
    }


def _role_config(
    role: str,
    label: str,
    model_role: str,
    tools: set[str],
    tier: str = "normal",
    required_tool: str | None = None,
) -> dict:
    spec = get_model_spec(model_role, tier)
    model_variable = (
        "MODEL_RESEARCH_MAX"
        if role == "research_max"
        else f"MODEL_{model_role.upper()}"
    )
    reason = None
    if role == "research_max" and not os.environ.get(model_variable, "").strip():
        reason = f"未配置 {model_variable}"
    elif spec is None:
        reason = f"未配置 {model_variable}"
    elif not role_available(model_role, tier):
        keys = PROVIDER_KEYS.get(spec.provider, ())
        key_name = (
            "GEMINI_API_KEY"
            if spec.provider == "gemini"
            else keys[0]
            if keys
            else "API_KEY"
        )
        reason = f"缺少 {key_name}"
    elif role in {"research", "research_max"} and not os.environ.get(
        "TAVILY_API_KEY", ""
    ).strip():
        reason = "未配置 Tavily key"
    elif required_tool and required_tool not in tools:
        reason = "对应工具未绑定"
    return {
        "role": role,
        "label": label,
        "provider": spec.provider if spec else None,
        "model": spec.model if spec else None,
        "effort": spec.effort if spec else None,
        "available": reason is None,
        "reason": reason,
    }


def current_config(bound_tools: set[str]) -> dict:
    roles = [
        _role_config("conversation", "对话", "conversation", bound_tools),
        _role_config("report", "写报告", "report", bound_tools, required_tool="write_report"),
        _role_config("research", "调查", "research", bound_tools, required_tool="investigate"),
        _role_config(
            "research_max",
            "深度调查",
            "research",
            bound_tools,
            "max",
            required_tool="investigate",
        ),
        _role_config("vision", "看图", "vision", bound_tools, required_tool="inspect_image"),
        _role_config("video", "视频与 PDF", "video", bound_tools, required_tool="analyze_video"),
    ]
    services = [
        {
            "name": "search",
            "label": "联网搜索",
            "available": bool(os.environ.get("TAVILY_API_KEY", "").strip()),
            "reason": (
                None
                if os.environ.get("TAVILY_API_KEY", "").strip()
                else "未配置 Tavily key"
            ),
        },
        {
            "name": "memory",
            "label": "长期记忆",
            "available": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
            "reason": (
                None
                if os.environ.get("OPENAI_API_KEY", "").strip()
                else "缺少 OPENAI_API_KEY"
            ),
        },
        {
            "name": "speech",
            "label": "语音合成",
            "available": True,
            "detail": os.environ.get("TTS_VOICE", "zh-CN-XiaoyiNeural"),
        },
    ]
    keys = {
        "openai": _key_status(PROVIDER_KEYS["openai"]),
        "gemini": _key_status(PROVIDER_KEYS["gemini"]),
        "deepseek": _key_status(PROVIDER_KEYS["deepseek"]),
        "tavily": _key_status(("TAVILY_API_KEY",)),
    }
    capabilities = {
        "max_tier": (
            bool(os.environ.get("MODEL_RESEARCH_MAX", "").strip())
            and role_available("research", "max")
            and "investigate" in bound_tools
        ),
        "video": "analyze_video" in bound_tools,
        "document": "analyze_document" in bound_tools,
    }
    return {
        "profile": "完整模式",
        "roles": roles,
        "services": services,
        "keys": keys,
        "capabilities": capabilities,
    }
