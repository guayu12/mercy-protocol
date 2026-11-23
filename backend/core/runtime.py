import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import httpx

from core import models
from core.env_file import atomic_write_bytes, config_lock


PROFILE_CHAT_ROLES = (
    "conversation",
    "report",
    "research",
    "research_max",
    "vision",
)
PROFILE_FALLBACK_ROLES = tuple(f"{role}_fallback" for role in PROFILE_CHAT_ROLES)
PROFILE_ROLE_NAMES = (
    *PROFILE_CHAT_ROLES,
    "video",
    "memory",
    *PROFILE_FALLBACK_ROLES,
)
_VALID_SPEC_ROLES = frozenset(PROFILE_ROLE_NAMES)
BASE_TOOL_NAMES = frozenset(
    {
        "check_weather",
        "get_current_time",
        "create_local_directory",
        "show_report",
        "show_media",
        "clear_stage",
        "read_report",
        "highlight_report",
    }
)
DEEPSEEK_MODELS_URL = "https://api.deepseek.com/models"
DEEPSEEK_SUCCESS_TTL_SECONDS = 600
DEEPSEEK_FAILURE_TTL_SECONDS = 30


@dataclass(frozen=True)
class RuntimeSnapshot:
    source: str
    active: str | None
    version: int
    roles: Mapping[str, models.ModelSpec | None]
    tavily: bool
    image_input: Mapping[models.ModelSpec, bool]
    profiles_view: Mapping[str, Mapping[str, str]]
    error: str | None = None

    def spec(
        self,
        role: str,
        tier: str = "normal",
        *,
        fallback: bool = False,
    ) -> models.ModelSpec | None:
        role = role.lower()
        if tier not in {"normal", "max"}:
            raise models.ModelConfigError(f"MODEL_{role.upper()}: unknown tier")
        if role not in _VALID_SPEC_ROLES:
            raise models.ModelConfigError(f"MODEL_{role.upper()}: unknown model role")
        if role in {"video", "memory"} and fallback:
            return None
        if fallback:
            if tier == "max":
                max_spec = self.roles.get(f"{role}_max_fallback")
                if max_spec is not None:
                    return max_spec
            return self.roles.get(f"{role}_fallback")
        if tier == "max":
            max_spec = self.roles.get(f"{role}_max")
            if max_spec is not None:
                return max_spec
        return self.roles.get(role)

    def available(self, role: str, tier: str = "normal") -> bool:
        spec = self.spec(role, tier)
        if spec is None or models.provider_key(spec.provider) is None:
            return False
        return role.lower() != "vision" or self.image_input.get(spec, False)

    def can_see_images(self, role: str = "conversation") -> bool:
        spec = self.spec(role)
        return spec is not None and self.image_input.get(spec, False)

    def tool_names(self) -> frozenset[str]:
        names = set(BASE_TOOL_NAMES)
        if self.tavily:
            names.add("tavily_search")
        if self.available("memory"):
            names.add("save_memory")
        if self.available("report"):
            names.add("write_report")
        if self.available("research") and self.tavily:
            names.add("investigate")
        if self.available("vision"):
            names.add("inspect_image")
        if self.available("video"):
            names.update({"analyze_video", "analyze_document"})
        return frozenset(names)

    def unavailable_reason(self, tool_name: str) -> str | None:
        if tool_name in BASE_TOOL_NAMES:
            return None
        if tool_name == "tavily_search":
            return None if self.tavily else "未配置 Tavily key"
        if tool_name == "save_memory":
            return self._role_reason("memory", "长期记忆已关闭")
        if tool_name == "write_report":
            return self._role_reason("report", "当前配置没有写报告模型")
        if tool_name == "investigate":
            reason = self._role_reason("research", "当前配置没有调查模型")
            return reason or (None if self.tavily else "未配置 Tavily key")
        if tool_name == "inspect_image":
            return self._role_reason("vision", "当前配置没有看图模型")
        if tool_name in {"analyze_video", "analyze_document"}:
            return self._role_reason("video", "当前配置没有视频与 PDF 模型")
        return None

    def _role_reason(self, role: str, missing_spec: str) -> str | None:
        spec = self.spec(role)
        if spec is None:
            return missing_spec
        key = models.provider_key(spec.provider)
        if key is None:
            variable = models.provider_key_variable(spec.provider)
            return f"缺少 {variable}" if variable else missing_spec
        if role == "vision" and not self.image_input.get(spec, False):
            return "当前配置的看图模型不支持图片输入"
        return None

    def profiles_dict(self) -> dict[str, dict[str, str]]:
        return {profile: dict(roles) for profile, roles in self.profiles_view.items()}

    def __hash__(self) -> int:
        profiles = tuple(
            (profile, tuple(roles.items()))
            for profile, roles in self.profiles_view.items()
        )
        return hash(
            (
                self.source,
                self.active,
                self.version,
                tuple(self.roles.items()),
                self.tavily,
                tuple(self.image_input.items()),
                profiles,
                self.error,
            )
        )


@dataclass(frozen=True)
class _CapabilityEntry:
    expires_at: float
    models: Mapping[str, bool]
    success: bool


_logger = logging.getLogger(__name__)
_lock = threading.RLock()
_snapshot: RuntimeSnapshot | None = None
_last_valid_snapshot: RuntimeSnapshot | None = None
_snapshot_signature = None
_snapshot_capability_version = -1
_snapshot_version = 0
_capability_version = 0
_capability_cache: dict[str, _CapabilityEntry] = {}
_startup_profile_logged = False


def profiles_path() -> Path:
    override = os.environ.get("MERCY_PROFILES_PATH")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent.parent / "data" / "profiles.json"


def _profile_signature(path: Path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _environment_signature():
    model_names = []
    for role in models.CHAT_ROLES:
        for suffix in ("", "_MAX", "_FALLBACK", "_MAX_FALLBACK"):
            model_names.append(f"MODEL_{role.upper()}{suffix}")
    model_names.extend(
        (
            "MODEL_VIDEO",
            "MODEL_VIDEO_MAX",
            "MODEL_VIDEO_FALLBACK",
            "MODEL_VIDEO_MAX_FALLBACK",
            "DEFAULT_MODEL",
            "DEFAULT_REASONING_EFFORT",
            "OPENAI_VISION_MODEL",
            "OPENAI_EMBEDDING_MODEL",
        )
    )
    deepseek_key = models.provider_key("deepseek")
    key_fingerprint = (
        _capability_fingerprint(deepseek_key[0]) if deepseek_key else None
    )
    return (
        tuple((name, os.environ.get(name)) for name in model_names),
        bool(models.provider_key("tavily")),
        key_fingerprint,
    )


def _config_signature():
    path = profiles_path()
    stat_signature = _profile_signature(path)
    if stat_signature is not None:
        env_signature = (
            os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            bool(models.provider_key("tavily")),
            (
                _capability_fingerprint(key[0])
                if (key := models.provider_key("deepseek"))
                else None
            ),
        )
    else:
        env_signature = _environment_signature()
    return str(path), stat_signature, env_signature


def _profile_error(field: str, message: str) -> models.ModelConfigError:
    return models.ModelConfigError(f"profiles.json: {field}: {message}")


def _validate_profile_spec(role_name: str, field: str, value) -> models.ModelSpec:
    if not isinstance(value, str):
        raise _profile_error(field, "expected provider:model[@effort]")
    spec = models.parse_spec(field, value)
    if spec is None:
        raise _profile_error(field, "expected provider:model[@effort]")

    base_role = role_name.removesuffix("_fallback")
    if base_role in PROFILE_CHAT_ROLES:
        if spec.provider not in {"openai", "deepseek"}:
            raise _profile_error(field, "chat roles require openai or deepseek")
        if spec.provider == "deepseek":
            allowed = (
                models.DEEPSEEK_EFFORTS
                if role_name in {"report", "report_fallback"}
                else {"none"}
            )
            if spec.effort is not None and spec.effort not in allowed:
                raise _profile_error(field, "unsupported DeepSeek reasoning effort")
    elif role_name == "video":
        if spec.provider != "gemini":
            raise _profile_error(field, "video models must use gemini")
        if spec.effort is not None:
            raise _profile_error(field, "video models do not support effort")
    elif role_name == "memory":
        expected_model = os.environ.get(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        if spec.provider != "openai":
            raise _profile_error(field, "memory must use openai")
        if spec.effort is not None:
            raise _profile_error(field, "memory does not support effort")
        if spec.model != expected_model:
            raise _profile_error(field, "memory model must match OPENAI_EMBEDDING_MODEL")
    return spec


def _read_profiles_unlocked(path: Path):
    try:
        document = json.loads(path.read_bytes().decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _profile_error("profiles.json", "invalid configuration file") from error
    return _parse_profiles_document(document)


def _read_profiles(path: Path):
    with config_lock(path):
        return _read_profiles_unlocked(path)


def _parse_profiles_document(document):
    if not isinstance(document, dict):
        raise _profile_error("profiles.json", "expected an object")
    unknown_top_level = set(document) - {"version", "active", "profiles"}
    if unknown_top_level:
        field = sorted(unknown_top_level)[0]
        raise _profile_error(f"profiles.json.{field}", "unknown top-level field")
    version = document.get("version")
    if type(version) is not int or version != 1:
        raise _profile_error("version", "must be 1")

    active = document.get("active")
    if not isinstance(active, str) or active not in {"full", "lite"}:
        raise _profile_error("active", "must be full or lite")
    profiles = document.get("profiles")
    if not isinstance(profiles, dict):
        raise _profile_error("profiles", "expected an object")
    unknown_profiles = set(profiles) - {"full", "lite"}
    if unknown_profiles:
        profile = sorted(unknown_profiles)[0]
        raise _profile_error(f"profiles.{profile}", "unknown profile id")
    if not profiles:
        raise _profile_error("profiles", "profile ids must be full or lite")
    if active not in profiles:
        raise _profile_error("active", "active profile is not defined")

    parsed_profiles: dict[str, dict[str, models.ModelSpec | None]] = {}
    profiles_view: dict[str, dict[str, str]] = {}
    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise _profile_error(f"profiles.{profile_name}", "expected an object")
        unknown_profile_fields = set(profile) - {"roles"}
        if unknown_profile_fields:
            unknown = sorted(unknown_profile_fields)[0]
            raise _profile_error(
                f"profiles.{profile_name}.{unknown}", "unknown profile field"
            )
        role_values = profile.get("roles")
        if not isinstance(role_values, dict):
            raise _profile_error(f"profiles.{profile_name}.roles", "expected an object")
        unknown_roles = set(role_values) - set(PROFILE_ROLE_NAMES)
        if unknown_roles:
            unknown = sorted(unknown_roles)[0]
            raise _profile_error(
                f"profiles.{profile_name}.roles.{unknown}", "unknown role field"
            )

        parsed_roles: dict[str, models.ModelSpec | None] = {
            role: None for role in PROFILE_ROLE_NAMES
        }
        view_roles: dict[str, str] = {}
        for role_name, value in role_values.items():
            field = f"profiles.{profile_name}.roles.{role_name}"
            spec = _validate_profile_spec(role_name, field, value)
            parsed_roles[role_name] = spec
            view_roles[role_name] = _format_spec(spec)
        if profile_name == active and parsed_roles["conversation"] is None:
            raise _profile_error(
                f"profiles.{profile_name}.roles.conversation",
                "active profile must define conversation",
            )
        parsed_profiles[profile_name] = parsed_roles
        profiles_view[profile_name] = view_roles

    return active, parsed_profiles[active], profiles_view


def format_spec(spec: models.ModelSpec) -> str:
    suffix = f"@{spec.effort}" if spec.effort is not None else ""
    return f"{spec.provider}:{spec.model}{suffix}"


_format_spec = format_spec


def seed_document() -> dict:
    roles = {
        name: format_spec(spec)
        for name, spec in _environment_roles().items()
        if name in PROFILE_ROLE_NAMES and spec is not None
    }
    return {
        "version": 1,
        "active": "full",
        "profiles": {
            "full": {"roles": roles},
            "lite": {
                "roles": {
                    "conversation": "deepseek:deepseek-flash@none",
                    "research": "deepseek:deepseek-flash@none",
                    "vision": "deepseek:deepseek-flash@none",
                    "report": "deepseek:deepseek-flash@high",
                }
            },
        },
    }


def seed_dropped() -> list[str]:
    names = [
        *(f"MODEL_{role}_MAX" for role in ("CONVERSATION", "REPORT", "VISION")),
        *(
            f"MODEL_{role}_MAX_FALLBACK"
            for role in ("CONVERSATION", "REPORT", "VISION")
        ),
        "MODEL_VIDEO_MAX",
        "MODEL_VIDEO_FALLBACK",
        "MODEL_VIDEO_MAX_FALLBACK",
    ]
    return sorted(name for name in names if os.environ.get(name, "").strip())


def candidate_snapshot(document: dict, profile: str) -> RuntimeSnapshot:
    candidate = dict(document)
    candidate["active"] = profile
    active, roles, profiles_view = _parse_profiles_document(candidate)
    return _make_snapshot(
        source="profiles",
        active=active,
        version=_snapshot_version + 1,
        roles=roles,
        profiles_view=profiles_view,
    )


def _invalidate_snapshot() -> None:
    global _snapshot, _snapshot_signature, _snapshot_capability_version
    with _lock:
        _snapshot = None
        _snapshot_signature = None
        _snapshot_capability_version = -1


def save_profiles(profiles: dict) -> None:
    path = profiles_path()
    with config_lock(path):
        existing_text = None
        has_bom = False
        active = "full"
        if path.exists():
            try:
                existing_bytes = path.read_bytes()
                has_bom = existing_bytes.startswith(b"\xef\xbb\xbf")
                existing_text = existing_bytes.decode("utf-8-sig")
            except OSError:
                existing_text = None
                has_bom = False
            except UnicodeError:
                existing_text = None
                has_bom = False
            if existing_text is not None:
                try:
                    existing = json.loads(existing_text)
                except json.JSONDecodeError:
                    existing = None
                if isinstance(existing, dict) and existing.get("active") in {
                    "full",
                    "lite",
                }:
                    active = existing["active"]
        if active not in {"full", "lite"}:
            active = "full"
        document = {"version": 1, "active": active, "profiles": profiles}
        _parse_profiles_document(document)
        newline = "\r\n" if existing_text and "\r\n" in existing_text else "\n"
        text = json.dumps(document, ensure_ascii=False, indent=2)
        if newline == "\r\n":
            text = text.replace("\n", "\r\n")
        if has_bom:
            text = "\ufeff" + text
        atomic_write_bytes(path, text.encode("utf-8"))
    _invalidate_snapshot()


def set_active(active: str) -> None:
    path = profiles_path()
    with config_lock(path):
        if not path.exists():
            raise _profile_error("profiles.json", "请先保存配置")
        try:
            existing_bytes = path.read_bytes()
            has_bom = existing_bytes.startswith(b"\xef\xbb\xbf")
            existing_text = existing_bytes.decode("utf-8-sig")
            document = json.loads(existing_text)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise _profile_error("profiles.json", "invalid configuration file") from error
        if not isinstance(document, dict):
            raise _profile_error("profiles.json", "expected an object")
        document["active"] = active
        _parse_profiles_document(document)
        newline = "\r\n" if "\r\n" in existing_text else "\n"
        text = json.dumps(document, ensure_ascii=False, indent=2)
        if newline == "\r\n":
            text = text.replace("\n", "\r\n")
        if has_bom:
            text = "\ufeff" + text
        atomic_write_bytes(path, text.encode("utf-8"))
    _invalidate_snapshot()


def validate_profiles_file() -> bool:
    path = profiles_path()
    if not path.exists():
        return False
    _read_profiles(path)
    return True


def validate_startup_config() -> None:
    global _startup_profile_logged
    if validate_profiles_file():
        if not _startup_profile_logged:
            message = "profiles.json 生效：.env 中的 MODEL_* 不再使用"
            _logger.info(message)
            from core import log_bus

            log_bus.emit("config", "CONFIG", message)
            _startup_profile_logged = True
        current_snapshot()
        return
    models.validate_environment_models()
    current_snapshot()


def _environment_roles() -> dict[str, models.ModelSpec | None]:
    roles: dict[str, models.ModelSpec | None] = {
        role: None for role in PROFILE_ROLE_NAMES
    }
    for role in models.CHAT_ROLES:
        roles[role] = models._env_model_spec(role)
        roles[f"{role}_fallback"] = models._env_model_spec(
            role, fallback=True
        )
        roles[f"{role}_max"] = models._env_model_spec(role, "max")
        roles[f"{role}_max_fallback"] = models._env_model_spec(
            role, "max", fallback=True
        )
    roles["research_max"] = (
        models._env_model_spec("research", "max")
        if os.environ.get("MODEL_RESEARCH_MAX", "").strip()
        else None
    )
    roles["research_max_fallback"] = (
        models._env_model_spec("research", "max", fallback=True)
        if os.environ.get("MODEL_RESEARCH_MAX_FALLBACK", "").strip()
        else None
    )
    roles["video"] = models._env_model_spec("video")
    roles["video_max"] = models._env_model_spec("video", "max")
    embedding_model = os.environ.get(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    roles["memory"] = models.ModelSpec("openai", embedding_model, None)
    return roles


def _capability_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _image_input_for(roles: Mapping[str, models.ModelSpec | None]):
    image_input: dict[models.ModelSpec, bool] = {}
    for spec in roles.values():
        if spec is None or spec in image_input:
            continue
        if spec.provider == "openai":
            image_input[spec] = True
        elif spec.provider == "deepseek":
            key = models.provider_key("deepseek")
            fingerprint = _capability_fingerprint(key[0]) if key else None
            with _lock:
                entry = _capability_cache.get(fingerprint or "")
            image_input[spec] = bool(
                entry
                and entry.expires_at > time.monotonic()
                and entry.success
                and entry.models.get(spec.model, False)
            )
        else:
            image_input[spec] = False
    return MappingProxyType(image_input)


def _freeze_profiles_view(profiles_view):
    return MappingProxyType(
        {
            profile: MappingProxyType(dict(roles))
            for profile, roles in profiles_view.items()
        }
    )


def _make_snapshot(
    *,
    source: str,
    active: str | None,
    version: int,
    roles: Mapping[str, models.ModelSpec | None],
    profiles_view: Mapping[str, Mapping[str, str]],
    error: str | None = None,
) -> RuntimeSnapshot:
    frozen_roles = MappingProxyType(dict(roles))
    return RuntimeSnapshot(
        source=source,
        active=active,
        version=version,
        roles=frozen_roles,
        tavily=models.provider_key("tavily") is not None,
        image_input=_image_input_for(frozen_roles),
        profiles_view=_freeze_profiles_view(profiles_view),
        error=error,
    )


def current_snapshot() -> RuntimeSnapshot:
    global _snapshot, _last_valid_snapshot, _snapshot_signature
    global _snapshot_capability_version, _snapshot_version

    signature = _config_signature()
    with _lock:
        if _snapshot is not None and signature == _snapshot_signature:
            if _snapshot_capability_version == _capability_version:
                return _snapshot
            base = _last_valid_snapshot or _snapshot
            _snapshot_version += 1
            _snapshot = _make_snapshot(
                source=base.source,
                active=base.active,
                version=_snapshot_version,
                roles=base.roles,
                profiles_view=base.profiles_view,
                error=_snapshot.error,
            )
            if _snapshot.error is None:
                _last_valid_snapshot = _snapshot
            _snapshot_capability_version = _capability_version
            return _snapshot

        try:
            path = profiles_path()
            if path.exists():
                active, roles, profiles_view = _read_profiles(path)
                source = "profiles"
            else:
                models.validate_environment_models()
                roles = _environment_roles()
                active = None
                profiles_view = {}
                source = "env"
        except models.ModelConfigError as error:
            if _last_valid_snapshot is None:
                raise
            safe_error = str(error)
            _logger.warning("Runtime profile reload failed: %s", safe_error)
            from core import log_bus

            log_bus.emit("error", "CONFIG", safe_error)
            _snapshot = replace(_last_valid_snapshot, error=safe_error)
            _snapshot_signature = signature
            _snapshot_capability_version = _capability_version
            return _snapshot

        _snapshot_version += 1
        _snapshot = _make_snapshot(
            source=source,
            active=active,
            version=_snapshot_version,
            roles=roles,
            profiles_view=profiles_view,
        )
        _last_valid_snapshot = _snapshot
        _snapshot_signature = signature
        _snapshot_capability_version = _capability_version
        return _snapshot


async def _fetch_deepseek_capabilities(key: str) -> dict[str, bool]:
    async def request():
        async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
            response = await client.get(
                DEEPSEEK_MODELS_URL,
                headers={"Authorization": f"Bearer {key}"},
            )
            response.raise_for_status()
            payload = response.json()
        entries = payload.get("data", []) if isinstance(payload, dict) else []
        result = {}
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("id") or entry.get("model")
                modalities = entry.get("input_modalities")
                if isinstance(name, str) and isinstance(modalities, list):
                    result[name] = "image" in modalities
        return result

    return await asyncio.wait_for(request(), timeout=2.0)


def _error_http_status(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


async def _load_deepseek_capabilities(key: str) -> None:
    global _capability_version
    fingerprint = _capability_fingerprint(key)
    now = time.monotonic()
    with _lock:
        cached = _capability_cache.get(fingerprint)
        if cached and cached.expires_at > now:
            return
    try:
        capabilities = await _fetch_deepseek_capabilities(key)
    except Exception as error:
        status = _error_http_status(error)
        entry = _CapabilityEntry(
            expires_at=time.monotonic() + DEEPSEEK_FAILURE_TTL_SECONDS,
            models=MappingProxyType({}),
            success=False,
        )
        _logger.warning(
            "DeepSeek capability lookup failed (%s, status=%s)",
            type(error).__name__,
            status,
        )
        from core import log_bus

        log_bus.emit(
            "warn",
            "CONFIG",
            f"DeepSeek capability lookup failed ({type(error).__name__}, status={status})",
        )
    else:
        entry = _CapabilityEntry(
            expires_at=time.monotonic() + DEEPSEEK_SUCCESS_TTL_SECONDS,
            models=MappingProxyType(dict(capabilities)),
            success=True,
        )
    with _lock:
        _capability_cache[fingerprint] = entry
        _capability_version += 1


async def load_snapshot() -> RuntimeSnapshot:
    snapshot = current_snapshot()
    specs = {spec for spec in snapshot.roles.values() if spec is not None}
    if any(spec.provider == "deepseek" for spec in specs):
        key = models.provider_key("deepseek")
        if key is not None:
            await _load_deepseek_capabilities(key[0])
    return current_snapshot()


def runtime_from(config) -> RuntimeSnapshot:
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    runtime = configurable.get("runtime")
    return runtime if isinstance(runtime, RuntimeSnapshot) else current_snapshot()


def reset_runtime_cache() -> None:
    global _snapshot, _last_valid_snapshot, _snapshot_signature
    global _snapshot_capability_version, _snapshot_version, _capability_version
    global _startup_profile_logged
    with _lock:
        _snapshot = None
        _last_valid_snapshot = None
        _snapshot_signature = None
        _snapshot_capability_version = -1
        _snapshot_version = 0
        _capability_version = 0
        _capability_cache.clear()
        _startup_profile_logged = False


def clear_capability_cache() -> None:
    global _capability_version
    with _lock:
        _capability_cache.clear()
        _capability_version += 1


def cache_deepseek_capabilities(key: str, capabilities: Mapping[str, bool]) -> None:
    global _capability_version
    fingerprint = _capability_fingerprint(key)
    entry = _CapabilityEntry(
        expires_at=time.monotonic() + DEEPSEEK_SUCCESS_TTL_SECONDS,
        models=MappingProxyType(dict(capabilities)),
        success=True,
    )
    with _lock:
        _capability_cache[fingerprint] = entry
        _capability_version += 1
