import asyncio
import os
from dataclasses import dataclass
from typing import Literal

import openai
from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    effort: str | None


CHAT_ROLES = ("conversation", "report", "research", "vision")
ROLES = CHAT_ROLES + ("video",)
PROVIDERS = {"openai", "gemini", "deepseek"}
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
PROVIDER_KEYS = {
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
}
ROLE_TIMEOUTS = {
    "conversation": 45,
    "report": 120,
    "research": 120,
    "vision": 60,
}
DEFAULT_VIDEO_MAX_MINUTES = 30
DEFAULT_GEMINI_FILE_RETENTION_SECONDS = 900


class ModelConfigError(ValueError):
    pass


def _validate_effort(var_name: str, effort: str | None) -> None:
    if effort is not None and effort not in REASONING_EFFORTS:
        raise ModelConfigError(
            f"{var_name}: unsupported reasoning effort {effort!r}"
        )


def is_transient(error: Exception) -> bool:
    return (
        isinstance(
            error,
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                asyncio.TimeoutError,
            ),
        )
        or (
            isinstance(error, openai.APIStatusError)
            and error.status_code >= 500
        )
    )


def parse_spec(var_name: str, value: str | None) -> ModelSpec | None:
    if value is None or not value.strip():
        return None

    raw_value = value.strip()
    provider, separator, rest = raw_value.partition(":")
    if not separator:
        raise ModelConfigError(
            f"{var_name}: expected provider:model[@effort], got {raw_value!r}"
        )

    provider = provider.strip()
    effort = None
    if "@" in rest:
        model, effort = rest.rsplit("@", 1)
        model = model.strip()
        effort = effort.strip()
        if not effort:
            raise ModelConfigError(
                f"{var_name}: expected provider:model[@effort], got {raw_value!r}"
            )
        _validate_effort(var_name, effort)
    else:
        model = rest.strip()

    if not provider or not model:
        raise ModelConfigError(
            f"{var_name}: expected provider:model[@effort], got {raw_value!r}"
        )
    if provider not in PROVIDERS:
        raise ModelConfigError(f"{var_name}: unknown provider {provider!r}")
    return ModelSpec(provider=provider, model=model, effort=effort)


def _role_name(role: str) -> str:
    normalized = role.lower()
    if normalized not in ROLES:
        raise ModelConfigError(f"MODEL_{role.upper()}: unknown model role")
    return normalized


def _configured_spec(var_name: str) -> ModelSpec | None:
    return parse_spec(var_name, os.environ.get(var_name))


def _legacy_spec(role: str) -> ModelSpec | None:
    if role == "conversation" and not os.environ.get("MODEL_CONVERSATION", "").strip():
        model = os.environ.get("DEFAULT_MODEL", "").strip() or "gpt-6-luna"
        effort = os.environ.get("DEFAULT_REASONING_EFFORT", "").strip() or None
        _validate_effort("DEFAULT_REASONING_EFFORT", effort)
        return ModelSpec("openai", model, effort)
    if role == "vision" and not os.environ.get("MODEL_VISION", "").strip():
        model = os.environ.get("OPENAI_VISION_MODEL", "").strip()
        if model:
            return ModelSpec("openai", model, None)
    return None


def get_model_spec(
    role: str,
    tier: Literal["normal", "max"] = "normal",
    *,
    fallback: bool = False,
) -> ModelSpec | None:
    role = _role_name(role)
    if tier not in ("normal", "max"):
        raise ModelConfigError(f"MODEL_{role.upper()}: unknown tier {tier!r}")

    base_var = f"MODEL_{role.upper()}"
    if fallback:
        if role == "video":
            return None
        fallback_var = f"{base_var}_FALLBACK"
        if tier == "max":
            max_fallback_var = f"{base_var}_MAX_FALLBACK"
            if os.environ.get(max_fallback_var, "").strip():
                return _configured_spec(max_fallback_var)
        return _configured_spec(fallback_var)

    if tier == "max":
        max_var = f"{base_var}_MAX"
        if os.environ.get(max_var, "").strip():
            return _configured_spec(max_var)

    spec = _configured_spec(base_var)
    if spec is not None:
        return spec
    return _legacy_spec(role)


def role_available(role: str, tier: str = "normal") -> bool:
    spec = get_model_spec(role, tier)  # type: ignore[arg-type]
    return spec is not None and any(
        os.environ.get(key, "").strip() for key in PROVIDER_KEYS[spec.provider]
    )


def _integer_setting(name: str, default: int, *, minimum: int = 0) -> int:
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ModelConfigError(f"{name}: expected an integer") from error
    if value < minimum:
        raise ModelConfigError(f"{name}: must be at least {minimum}")
    return value


def video_max_minutes() -> int:
    return _integer_setting("VIDEO_MAX_MINUTES", DEFAULT_VIDEO_MAX_MINUTES, minimum=1)


def gemini_file_retention_seconds() -> int:
    return _integer_setting(
        "GEMINI_FILE_RETENTION_SECONDS",
        DEFAULT_GEMINI_FILE_RETENTION_SECONDS,
    )


def validate_config() -> None:
    for suffix in ("_FALLBACK", "_MAX_FALLBACK"):
        var_name = f"MODEL_VIDEO{suffix}"
        if os.environ.get(var_name, "").strip():
            raise ModelConfigError(f"{var_name}: video models do not support fallbacks")

    for role in ROLES:
        for suffix in ("", "_MAX", "_FALLBACK", "_MAX_FALLBACK"):
            var_name = f"MODEL_{role.upper()}{suffix}"
            spec = _configured_spec(var_name)
            if role in CHAT_ROLES and spec and spec.provider != "openai":
                raise ModelConfigError(
                    f"{var_name}: provider '{spec.provider}' 1c 未启用（聊天类角色只支持 openai）"
                )
            if role == "video" and spec:
                if spec.provider != "gemini":
                    raise ModelConfigError(
                        f"{var_name}: video models must use the gemini provider"
                    )
                if spec.effort is not None:
                    raise ModelConfigError(
                        f"{var_name}: reasoning effort is not supported for video"
                    )
    if not os.environ.get("MODEL_CONVERSATION", "").strip():
        _legacy_spec("conversation")
    video_max_minutes()
    gemini_file_retention_seconds()


def get_llm(
    role: str,
    tier: str = "normal",
    *,
    fallback: bool = False,
    max_tokens: int | None = None,
) -> ChatOpenAI | None:
    role = _role_name(role)
    if role == "video":
        raise ModelConfigError("MODEL_VIDEO: video models do not use get_llm")
    spec = get_model_spec(role, tier, fallback=fallback)  # type: ignore[arg-type]
    if spec is None or not any(
        os.environ.get(key, "").strip() for key in PROVIDER_KEYS[spec.provider]
    ):
        return None
    if spec.provider != "openai":
        suffix = "_MAX_FALLBACK" if fallback and tier == "max" else (
            "_FALLBACK" if fallback else ("_MAX" if tier == "max" else "")
        )
        raise ModelConfigError(
            f"MODEL_{role.upper()}{suffix}: provider '{spec.provider}' is not supported by get_llm"
        )
    if role not in ROLE_TIMEOUTS:
        raise ModelConfigError(f"MODEL_{role.upper()}: no timeout is configured")

    options = {
        "model": spec.model,
        "api_key": os.environ["OPENAI_API_KEY"],
        "base_url": os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1",
        "streaming": True,
        "stream_usage": True,
        "max_retries": 1,
        "timeout": ROLE_TIMEOUTS[role],
    }
    if spec.effort:
        options["reasoning"] = {"effort": spec.effort}
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    return ChatOpenAI(**options)
