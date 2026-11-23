from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import replace
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError

from core import env, models, runtime, usage_stats
from core.env_file import ConfigWriteError, config_lock, set_env_value
from core.provider_errors import classify
from core.runtime_config import current_config


PROVIDER_VARIABLES = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "tavily": "TAVILY_API_KEY",
}
_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,512}$")
_NON_CHAT_MODEL_PARTS = (
    "embedding", "tts", "whisper", "audio", "realtime", "transcribe",
    "image", "moderation", "dall-e",
)
_MODEL_CACHE_TTL = 600
_model_cache_lock = threading.Lock()
_model_cache: dict[tuple[str, str, str], tuple[float, list[dict]]] = {}
_provider_key_locks = {
    provider: asyncio.Lock() for provider in PROVIDER_VARIABLES
}
_profile_operation_lock = asyncio.Lock()


class ConfigApiError(Exception):
    def __init__(
        self,
        status_code: int,
        category: str,
        message: str | None = None,
        **extra,
    ):
        self.status_code = status_code
        self.payload = {"category": category}
        if message is not None:
            self.payload["message"] = message
        self.payload.update(extra)
        super().__init__(message or category)


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelsRequest(_RequestModel):
    provider: Literal["openai", "deepseek", "gemini", "tavily"]
    api_key: SecretStr | None = None


class KeyRequest(_RequestModel):
    provider: Literal["openai", "deepseek", "gemini", "tavily"]
    api_key: SecretStr


class ProfilesRequest(_RequestModel):
    profiles: dict


class ActiveRequest(_RequestModel):
    active: Literal["full", "lite"]


def parse_payload(raw: bytes, schema: type[_RequestModel]):
    try:
        document = json.loads(raw)
        return schema.model_validate(document)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, TypeError):
        raise ConfigApiError(400, "invalid_request", "请求格式不正确") from None


def _valid_key(key: str) -> bool:
    return bool(_KEY_PATTERN.fullmatch(key))


def _base_url(provider: str) -> str:
    if provider == "openai":
        return (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    if provider == "deepseek":
        return "https://api.deepseek.com"
    if provider == "gemini":
        return "https://generativelanguage.googleapis.com/v1beta"
    return "https://api.tavily.com"


def _cache_key(provider: str, key: str) -> tuple[str, str, str]:
    return provider, hashlib.sha256(key.encode()).hexdigest(), _base_url(provider)


def _cached_models(cache_key):
    now = time.monotonic()
    with _model_cache_lock:
        entry = _model_cache.get(cache_key)
        if entry and entry[0] > now:
            return [dict(model) for model in entry[1]]
        _model_cache.pop(cache_key, None)
    return None


def _cache_models(cache_key, models_list):
    with _model_cache_lock:
        _model_cache[cache_key] = (
            time.monotonic() + _MODEL_CACHE_TTL,
            [dict(model) for model in models_list],
        )


def clear_model_cache(provider: str) -> None:
    with _model_cache_lock:
        for key in list(_model_cache):
            if key[0] == provider:
                _model_cache.pop(key, None)


async def _fetch_models(provider: str, key: str) -> list[dict]:
    base_url = _base_url(provider)
    headers = {}
    if provider == "gemini":
        url = f"{base_url}/models?pageSize=1000"
        headers["x-goog-api-key"] = key
    elif provider == "deepseek":
        url = f"{base_url}/models"
        headers["Authorization"] = f"Bearer {key}"
    else:
        url = f"{base_url}/models"
        headers["Authorization"] = f"Bearer {key}"

    async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        payload = response.json()

    entries = payload.get("data", []) if provider != "gemini" else payload.get("models", [])
    if not isinstance(entries, list):
        return []
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if provider == "gemini":
            name = entry.get("name")
            methods = entry.get("supportedGenerationMethods", [])
            if (
                not isinstance(name, str)
                or not isinstance(methods, list)
                or "generateContent" not in methods
            ):
                continue
            model_id = name.removeprefix("models/")
            lowered = model_id.lower()
            if not lowered.startswith("gemini-") or any(
                part in lowered for part in ("image", "tts", "embedding", "live", "audio")
            ):
                continue
            result.append({"id": model_id, "image_input": True, "efforts": []})
            continue
        model_id = entry.get("id") or entry.get("model")
        if not isinstance(model_id, str):
            continue
        if provider == "openai":
            if any(part in model_id.lower() for part in _NON_CHAT_MODEL_PARTS):
                continue
            image_input = True
            efforts = [
                effort
                for effort in (
                    "none",
                    "minimal",
                    "low",
                    "medium",
                    "high",
                    "xhigh",
                )
                if effort in models.REASONING_EFFORTS
            ]
        else:
            modalities = entry.get("input_modalities", [])
            image_input = isinstance(modalities, list) and "image" in modalities
            effort = entry.get("effort", {})
            supported = (
                effort.get("supported_levels", [])
                if isinstance(effort, dict)
                else []
            )
            if not isinstance(supported, list):
                supported = []
            efforts = ["none"]
            for value in supported:
                if (
                    isinstance(value, str)
                    and value in models.DEEPSEEK_EFFORTS
                    and value not in efforts
                ):
                    efforts.append(value)
        result.append(
            {"id": model_id, "image_input": image_input, "efforts": efforts}
        )
    return result


async def list_models(provider: str, key: str) -> list[dict]:
    cache_key = _cache_key(provider, key)
    cached = _cached_models(cache_key)
    if cached is not None:
        return cached
    result = await _fetch_models(provider, key)
    _cache_models(cache_key, result)
    return result


async def _validate_provider_key(provider: str, key: str) -> None:
    if provider == "tavily":
        async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
            response = await client.get(
                "https://api.tavily.com/usage",
                headers={"Authorization": f"Bearer {key}"},
            )
            response.raise_for_status()
    else:
        await list_models(provider, key)


def _api_error(error: Exception) -> ConfigApiError:
    result = classify(error)
    return ConfigApiError(400, result.category, result.message)


async def discover_models(request: ModelsRequest) -> dict:
    if request.provider == "tavily":
        raise ConfigApiError(400, "unsupported", "Tavily 没有模型列表")
    key = (
        request.api_key.get_secret_value()
        if request.api_key is not None
        else (models.provider_key(request.provider) or (None, None))[0]
    )
    if key is None:
        raise ConfigApiError(400, "missing_key", "缺少 key")
    if not _valid_key(key):
        raise ConfigApiError(400, "invalid_key", "key 格式不正确")
    try:
        found = await list_models(request.provider, key)
    except Exception as error:
        raise _api_error(error) from None
    return {"models": found}


def _key_status(provider: str) -> dict:
    resolved = models.provider_key(provider)
    variable = PROVIDER_VARIABLES[provider]
    value = resolved[0] if resolved else ""
    return {
        "configured": bool(value),
        "hint": f"••••{value[-4:]}" if len(value) >= 20 else None,
        "variable": resolved[1] if resolved else variable,
        "source": env.provider_key_source(provider),
    }


async def save_key(request: KeyRequest) -> dict:
    async with _provider_key_locks[request.provider]:
        return await _save_key_locked(request)


async def _save_key_locked(request: KeyRequest) -> dict:
    key = request.api_key.get_secret_value()
    if not _valid_key(key):
        raise ConfigApiError(400, "invalid_key", "key 格式不正确")
    env_error = env.env_error()
    if env_error:
        if env_error == env.ENV_READ_ERROR:
            raise ConfigApiError(
                400, "config_write", "无法写入配置文件（可能被占用或只读）"
            )
        raise ConfigApiError(400, "env_error", env_error)
    provider = request.provider
    if env.provider_key_source(provider) == "system":
        raise ConfigApiError(
            409, "system_key", "这个 key 来自系统环境变量，请在系统里修改"
        )
    variable = PROVIDER_VARIABLES[provider]
    try:
        await _validate_provider_key(provider, key)
    except Exception as error:
        raise _api_error(error) from None
    try:
        await asyncio.to_thread(set_env_value, env.ENV_PATH, variable, key)
    except ConfigWriteError as error:
        raise ConfigApiError(400, "config_write", error.message) from None
    os.environ[variable] = key
    clear_model_cache(provider)
    if provider == "deepseek":
        runtime.clear_capability_cache()
    if provider == "tavily":
        from core.agent import search

        search.web_search = None
    status = _key_status(provider)
    return {"key": status}


def _profile_file_document():
    path = runtime.profiles_path()
    with config_lock(path):
        try:
            if not path.exists():
                return None
            data = path.read_bytes()
        except OSError:
            raise ConfigApiError(
                400, "config_write", "无法读取配置文件（可能被占用）"
            ) from None
        try:
            return json.loads(data.decode("utf-8-sig"))
        except (UnicodeError, json.JSONDecodeError):
            raise ConfigApiError(
                400, "invalid_profiles", "配置文件格式不正确"
            ) from None


def _role_parts(role: str) -> tuple[str, str, bool]:
    fallback = role.endswith("_fallback")
    if fallback:
        role = role.removesuffix("_fallback")
    if role == "research_max":
        return "research", "max", fallback
    if role == "research":
        return "research", "normal", fallback
    return role, "normal", fallback


async def _ping_role(snapshot, role: str) -> None:
    role_name, tier, fallback = _role_parts(role)
    llm = models.get_llm(
        role_name,
        tier,
        fallback=fallback,
        max_tokens=16,
        runtime=snapshot,
    )
    if llm is None:
        raise models.ModelConfigError("model unavailable")
    with usage_stats.suppress_recording():
        await llm.ainvoke("ping")


async def _test_role(snapshot, profile: str, role: str) -> None:
    role_name, tier, fallback = _role_parts(role)
    spec = snapshot.spec(role_name, tier, fallback=fallback)
    if spec is None:
        return
    key = models.provider_key(spec.provider)
    if key is None:
        raise ConfigApiError(400, "missing_key", "缺少 key")
    if role_name == "vision":
        if spec.provider == "deepseek":
            found = await list_models("deepseek", key[0])
            runtime.cache_deepseek_capabilities(
                key[0],
                {item["id"]: item["image_input"] for item in found},
            )
            capable = next(
                (item["image_input"] for item in found if item["id"] == spec.model),
                False,
            )
            if not capable:
                raise models.ModelConfigError("image capability missing")
            snapshot = replace(
                snapshot,
                image_input={**snapshot.image_input, spec: True},
            )
        elif spec.provider != "openai":
            raise models.ModelConfigError("image capability missing")
    if role_name == "video":
        found = await list_models("gemini", key[0])
        if not any(item["id"] == spec.model for item in found):
            raise models.ModelConfigError("video model unavailable")
        return
    await _ping_role(snapshot, role)


async def _test_memory(snapshot) -> None:
    spec = snapshot.spec("memory")
    if spec is None:
        return
    key = models.provider_key("openai")
    if key is None:
        raise ConfigApiError(400, "missing_key", "缺少 key")
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    embedder = OpenAIEmbeddingFunction(
        api_key=key[0],
        model_name=spec.model,
        api_base=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    await asyncio.to_thread(embedder, ["ping"])


async def _run_role_test(snapshot, profile: str, role: str):
    try:
        if role == "memory":
            await asyncio.wait_for(_test_memory(snapshot), timeout=20)
        else:
            await asyncio.wait_for(
                _test_role(snapshot, profile, role), timeout=20
            )
    except ConfigApiError as error:
        return {
            "profile": profile,
            "role": role,
            "category": error.payload.get("category", "provider"),
            "message": error.payload.get("message", "模型服务出错"),
        }
    except Exception as error:
        provider_error = classify(error)
        return {
            "profile": profile,
            "role": role,
            "category": provider_error.category,
            "message": (
                "当前配置的看图模型不支持图片输入"
                if _role_parts(role)[0] == "vision"
                and isinstance(error, models.ModelConfigError)
                else provider_error.message
            ),
        }
    return None


async def save_profile_configuration(
    request: ProfilesRequest, *, editable: bool = False
) -> dict:
    async with _profile_operation_lock:
        return await _save_profile_configuration_locked(
            request, editable=editable
        )


async def _save_profile_configuration_locked(
    request: ProfilesRequest, *, editable: bool = False
) -> dict:
    profiles = request.profiles
    try:
        previous = _profile_file_document()
    except ConfigApiError as error:
        if error.payload.get("category") == "config_write":
            raise
        previous = None
    active = (
        previous.get("active")
        if isinstance(previous, dict)
        and previous.get("active") in {"full", "lite"}
        else "full"
    )
    valid_previous = previous
    if previous is not None:
        try:
            runtime._parse_profiles_document(previous)
        except models.ModelConfigError:
            valid_previous = None
    document = {
        "version": 1,
        "active": active,
        "profiles": profiles,
    }
    try:
        runtime._parse_profiles_document(document)
    except models.ModelConfigError:
        raise ConfigApiError(
            400, "invalid_profiles", "配置格式不正确"
        ) from None
    old_profiles = valid_previous.get("profiles", {}) if valid_previous else {}
    tests = []
    untested = []
    for profile, profile_data in profiles.items():
        old_roles = (
            old_profiles.get(profile, {}).get("roles", {}) if previous else {}
        )
        new_roles = profile_data.get("roles", {})
        changed_roles = [
            role
            for role in sorted(set(old_roles) | set(new_roles))
            if old_roles.get(role) != new_roles.get(role) and role in new_roles
        ]
        if not changed_roles:
            continue
        try:
            snapshot = runtime.candidate_snapshot(document, profile)
        except models.ModelConfigError:
            untested.extend(
                {
                    "profile": profile,
                    "role": role,
                    "reason": "未设置对话模型",
                }
                for role in changed_roles
            )
            continue
        for role in changed_roles:
            spec = snapshot.spec(*_role_parts(role)[:2], fallback=_role_parts(role)[2])
            if spec is None:
                continue
            if models.provider_key(spec.provider) is None:
                untested.append(
                    {"profile": profile, "role": role, "reason": "缺少 key"}
                )
                continue
            tests.append(_run_role_test(snapshot, profile, role))
    results = await asyncio.gather(*tests)
    errors = [error for error in results if error is not None]
    if errors:
        raise ConfigApiError(400, "role_test_failed", errors=errors)
    try:
        await asyncio.to_thread(runtime.save_profiles, profiles)
    except models.ModelConfigError:
        raise ConfigApiError(
            400, "invalid_profiles", "配置格式不正确"
        ) from None
    except ConfigWriteError as error:
        raise ConfigApiError(400, "invalid_profiles", error.message) from None
    return {
        "ok": True,
        "untested": untested,
        "config": config_view(runtime.current_snapshot(), editable=editable),
    }


async def switch_active_profile(
    request: ActiveRequest, *, editable: bool = False
) -> dict:
    async with _profile_operation_lock:
        return await _switch_active_profile_locked(
            request, editable=editable
        )


async def _switch_active_profile_locked(
    request: ActiveRequest, *, editable: bool = False
) -> dict:
    try:
        document = _profile_file_document()
    except ConfigApiError as error:
        if error.payload.get("category") == "config_write":
            raise
        raise ConfigApiError(
            400,
            "invalid_profiles",
            "配置文件格式不正确，请先在界面保存一次配置",
        ) from None
    if document is None:
        raise ConfigApiError(400, "profiles_missing", "请先保存配置")
    try:
        snapshot = runtime.candidate_snapshot(document, request.active)
    except models.ModelConfigError:
        raise ConfigApiError(
            400,
            "invalid_profiles",
            "配置文件格式不正确，请先在界面保存一次配置",
        ) from None
    spec = snapshot.spec("conversation")
    if spec is None:
        raise ConfigApiError(400, "configuration", "目标配置未设置对话模型")
    if models.provider_key(spec.provider) is None:
        raise ConfigApiError(400, "missing_key", "目标配置缺少 key")
    result = await _run_role_test(snapshot, request.active, "conversation")
    if result:
        raise ConfigApiError(
            400, "role_test_failed", errors=[result]
        )
    try:
        await asyncio.to_thread(runtime.set_active, request.active)
    except models.ModelConfigError:
        raise ConfigApiError(
            400, "invalid_profiles", "配置格式不正确"
        ) from None
    except ConfigWriteError as error:
        raise ConfigApiError(400, "invalid_profiles", error.message) from None
    return {
        "ok": True,
        "config": config_view(runtime.current_snapshot(), editable=editable),
    }


def config_view(snapshot=None, *, editable: bool = False) -> dict:
    snapshot = snapshot or runtime.current_snapshot()
    config = current_config(snapshot)
    if runtime.profiles_path().exists():
        config["profile_source"] = "file"
        config["profiles"] = snapshot.profiles_dict()
    else:
        config["profile_source"] = "env"
        config["seed"] = runtime.seed_document()["profiles"]
        config["seed_dropped"] = runtime.seed_dropped()
    config["editable"] = bool(editable and not env.env_error())
    config["env_error"] = env.env_error()
    return config
