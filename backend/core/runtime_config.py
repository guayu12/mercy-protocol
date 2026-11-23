import os

from core.env import provider_key_source
from core.models import provider_key, provider_key_variable
from core.runtime import RuntimeSnapshot, current_snapshot


ROLE_LABELS = (
    ("conversation", "对话"),
    ("report", "写报告"),
    ("research", "调查"),
    ("research_max", "深度调查"),
    ("vision", "看图"),
    ("video", "视频与 PDF"),
    ("memory", "长期记忆"),
)


def _key_status(provider: str) -> dict:
    resolved = provider_key(provider)
    variable = resolved[1] if resolved else provider_key_variable(provider)
    value = resolved[0] if resolved else ""
    return {
        "configured": bool(value),
        "hint": f"••••{value[-4:]}" if len(value) >= 20 else None,
        "variable": variable,
        "source": provider_key_source(provider),
    }


def _role_config(
    runtime: RuntimeSnapshot,
    role: str,
    label: str,
    tool_names: frozenset[str],
) -> dict:
    spec = runtime.spec(role)
    available = runtime.available(role)
    if role in {"research", "research_max"}:
        available = available and "investigate" in tool_names
    reason = None
    if spec is None:
        if role == "memory":
            reason = "长期记忆已关闭"
        else:
            if role == "research_max":
                reason = "未配置 MODEL_RESEARCH_MAX"
            else:
                reason = f"未配置 MODEL_{role.upper()}"
    elif not available:
        reason = runtime.unavailable_reason(
            {
                "report": "write_report",
                "research": "investigate",
                "research_max": "investigate",
                "vision": "inspect_image",
                "video": "analyze_video",
                "memory": "save_memory",
            }.get(role, "")
        )
        if reason is None:
            key = provider_key(spec.provider)
            variable = key[1] if key else provider_key_variable(spec.provider)
            reason = f"缺少 {variable}" if variable else "当前模型不可用"
    return {
        "role": role,
        "label": label,
        "provider": spec.provider if spec else None,
        "model": spec.model if spec else None,
        "effort": spec.effort if spec else None,
        "available": available,
        "reason": reason,
    }


def current_config(runtime: RuntimeSnapshot | None = None) -> dict:
    runtime = runtime or current_snapshot()
    tool_names = runtime.tool_names()
    roles = [
        _role_config(runtime, role, label, tool_names)
        for role, label in ROLE_LABELS
    ]
    services = [
        {
            "name": "search",
            "label": "联网搜索",
            "available": runtime.tavily,
            "reason": None if runtime.tavily else "未配置 Tavily key",
        },
        {
            "name": "memory",
            "label": "长期记忆",
            "available": runtime.available("memory"),
            "reason": (
                None
                if runtime.available("memory")
                else (
                    "长期记忆已关闭"
                    if runtime.spec("memory") is None
                    else "缺少 OPENAI_API_KEY"
                )
            ),
        },
        {
            "name": "speech",
            "label": "语音合成",
            "available": True,
            "detail": os.environ.get("TTS_VOICE", "zh-CN-XiaoyiNeural"),
        },
    ]
    capabilities = {
        "max_tier": runtime.available("research_max") and "investigate" in tool_names,
        "video": "analyze_video" in tool_names,
        "document": "analyze_document" in tool_names,
        "image_input": runtime.can_see_images(),
    }
    return {
        "profile": "便捷模式" if runtime.active == "lite" else "完整模式",
        "source": runtime.source,
        "active": runtime.active,
        "profiles": runtime.profiles_dict(),
        "profile_error": runtime.error,
        "roles": roles,
        "services": services,
        "keys": {
            "openai": _key_status("openai"),
            "gemini": _key_status("gemini"),
            "deepseek": _key_status("deepseek"),
            "tavily": _key_status("tavily"),
        },
        "capabilities": capabilities,
    }
