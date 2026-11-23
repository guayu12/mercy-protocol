import asyncio
import hashlib
import os
from dataclasses import dataclass
from typing import Literal

import openai
from langchain_openai import ChatOpenAI

from core.usage_stats import UsageCallback


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    effort: str | None


CHAT_ROLES = ("conversation", "report", "research", "vision")
ROLES = CHAT_ROLES + ("video",)
PROVIDERS = {"openai", "gemini", "deepseek"}
REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
DEEPSEEK_EFFORTS = {"none", "low", "high", "max"}
PROVIDER_KEY_VARIABLES = {
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "tavily": ("TAVILY_API_KEY",),
}
PROVIDER_KEYS = PROVIDER_KEY_VARIABLES
ROLE_TIMEOUTS = {
    "conversation": 45,
    "report": 120,
    "research": 120,
    "vision": 60,
}
DEFAULT_VIDEO_MAX_MINUTES = 30
DEFAULT_GEMINI_FILE_RETENTION_SECONDS = 900
DEFAULT_PDF_MAX_PAGES = 300


class ModelConfigError(ValueError):
    pass


def _validate_effort(var_name: str, effort: str | None) -> None:
    if effort is not None and effort not in REASONING_EFFORTS:
        raise ModelConfigError(
            f"{var_name}: unsupported reasoning effort"
        )


def provider_key(provider: str) -> tuple[str, str] | None:
    for variable in PROVIDER_KEY_VARIABLES.get(provider, ()):
        value = os.environ.get(variable, "").strip()
        if value:
            return value, variable
    return None


def provider_key_variable(provider: str) -> str | None:
    variables = PROVIDER_KEY_VARIABLES.get(provider, ())
    return variables[0] if variables else None


def api_kind(spec: ModelSpec) -> Literal["chat", "responses"]:
    if spec.provider == "openai":
        return "responses" if spec.effort is not None else "chat"
    if spec.provider == "deepseek":
        return (
            "responses"
            if spec.effort in {"low", "high", "max"}
            else "chat"
        )
    return "chat"


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
        raise ModelConfigError(f"{var_name}: expected provider:model[@effort]")

    provider = provider.strip()
    effort = None
    if "@" in rest:
        model, effort = rest.rsplit("@", 1)
        model = model.strip()
        effort = effort.strip()
        if not effort:
            raise ModelConfigError(
                f"{var_name}: expected provider:model[@effort]"
            )
        supported_efforts = (
            DEEPSEEK_EFFORTS if provider == "deepseek" else REASONING_EFFORTS
        )
        if effort not in supported_efforts:
            raise ModelConfigError(f"{var_name}: unsupported reasoning effort")
    else:
        model = rest.strip()

    if not provider or not model:
        raise ModelConfigError(
            f"{var_name}: expected provider:model[@effort]"
        )
    if provider not in PROVIDERS:
        raise ModelConfigError(f"{var_name}: unknown provider")
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


def _env_model_spec(
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


def get_model_spec(
    role: str,
    tier: Literal["normal", "max"] = "normal",
    *,
    fallback: bool = False,
    runtime=None,
) -> ModelSpec | None:
    if runtime is None:
        from core.runtime import current_snapshot

        runtime = current_snapshot()
    return runtime.spec(role, tier, fallback=fallback)


def role_available(role: str, tier: str = "normal", runtime=None) -> bool:
    from core.runtime import current_snapshot

    return (runtime or current_snapshot()).available(role, tier)


def model_cache_key(
    role: str,
    tier: str,
    fallback: bool,
    max_tokens: int | None,
    spec: ModelSpec,
    *,
    tool_names: tuple[str, ...] | None = None,
) -> tuple | None:
    key = provider_key(spec.provider)
    if key is None:
        return None
    base_url = _base_url(spec.provider)
    return (
        role,
        tier,
        fallback,
        max_tokens,
        spec,
        tuple(sorted(tool_names)) if tool_names is not None else None,
        spec.provider,
        hashlib.sha256(key[0].encode()).hexdigest()[:12],
        base_url,
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


def pdf_max_pages() -> int:
    return _integer_setting("PDF_MAX_PAGES", DEFAULT_PDF_MAX_PAGES, minimum=1)


def validate_environment_models() -> None:
    for suffix in ("_FALLBACK", "_MAX_FALLBACK"):
        var_name = f"MODEL_VIDEO{suffix}"
        if os.environ.get(var_name, "").strip():
            raise ModelConfigError(f"{var_name}: video models do not support fallbacks")

    for role in CHAT_ROLES:
        for suffix in ("", "_MAX", "_FALLBACK", "_MAX_FALLBACK"):
            var_name = f"MODEL_{role.upper()}{suffix}"
            spec = _configured_spec(var_name)
            if spec and spec.provider not in {"openai", "deepseek"}:
                raise ModelConfigError(
                    f"{var_name}: chat roles require openai or deepseek"
                )
            if spec and spec.provider == "deepseek":
                allowed_efforts = (
                    DEEPSEEK_EFFORTS
                    if role == "report" and suffix in {"", "_FALLBACK"}
                    else {"none"}
                )
                if spec.effort is not None and spec.effort not in allowed_efforts:
                    raise ModelConfigError(
                        f"{var_name}: unsupported DeepSeek reasoning effort"
                    )

    for suffix in ("", "_MAX"):
        var_name = f"MODEL_VIDEO{suffix}"
        spec = _configured_spec(var_name)
        if spec and spec.provider != "gemini":
            raise ModelConfigError(f"{var_name}: video models must use gemini")
        if spec and spec.effort is not None:
            raise ModelConfigError(
                f"{var_name}: reasoning effort is not supported for video"
            )
    if not os.environ.get("MODEL_CONVERSATION", "").strip():
        _legacy_spec("conversation")


def validate_config() -> None:
    from core.runtime import validate_startup_config

    validate_startup_config()
    video_max_minutes()
    gemini_file_retention_seconds()
    pdf_max_pages()


def get_llm(
    role: str,
    tier: str = "normal",
    *,
    fallback: bool = False,
    max_tokens: int | None = None,
    runtime=None,
) -> ChatOpenAI | None:
    if role in {"video", "memory"}:
        raise ModelConfigError(f"MODEL_{role.upper()}: role does not use get_llm")
    if runtime is None:
        from core.runtime import current_snapshot

        runtime = current_snapshot()
    spec = runtime.spec(role, tier, fallback=fallback)
    if spec is None:
        return None
    if role == "vision" and not runtime.image_input.get(spec, False):
        return None
    key = provider_key(spec.provider)
    if key is None:
        return None
    if spec.provider not in {"openai", "deepseek"}:
        raise ModelConfigError(
            f"MODEL_{role.upper()}: provider is not supported by get_llm"
        )
    timeout_role = "research" if role == "research_max" else role
    if timeout_role not in ROLE_TIMEOUTS:
        raise ModelConfigError(f"MODEL_{role.upper()}: no timeout is configured")

    base_url = _base_url(spec.provider)
    options = {
        "model": spec.model,
        "api_key": key[0],
        "base_url": base_url,
        "streaming": True,
        "stream_usage": True,
        "max_retries": 1,
        "timeout": ROLE_TIMEOUTS[timeout_role],
        "callbacks": [
            UsageCallback(
                "research_max" if role == "research" and tier == "max" else role,
                spec.provider,
                spec.model,
            )
        ],
    }
    if spec.provider == "deepseek" and api_kind(spec) == "chat":
        options["extra_body"] = {"thinking": {"type": "disabled"}}
        options["use_responses_api"] = False
    elif spec.provider == "deepseek":
        options["reasoning"] = {"effort": spec.effort}
        options["use_responses_api"] = True
    elif spec.effort is not None:
        options["reasoning"] = {"effort": spec.effort}
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    return ChatOpenAI(**options)


def _base_url(provider: str) -> str:
    if provider == "deepseek":
        return "https://api.deepseek.com"
    return os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
