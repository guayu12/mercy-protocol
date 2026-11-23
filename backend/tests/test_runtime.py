import json
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from core import models, runtime, runtime_config


def _document(roles, *, active="full", profiles=None):
    return {
        "version": 1,
        "active": active,
        "profiles": profiles
        or {
            active: {"roles": roles},
        },
    }


def _write(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")


def _clear_model_environment(monkeypatch):
    for role in models.ROLES:
        for suffix in ("", "_MAX", "_FALLBACK", "_MAX_FALLBACK"):
            monkeypatch.delenv(f"MODEL_{role.upper()}{suffix}", raising=False)
    for name in (
        "DEFAULT_MODEL",
        "DEFAULT_REASONING_EFFORT",
        "OPENAI_VISION_MODEL",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
        "TAVILY_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def profile_path(tmp_path, monkeypatch):
    path = tmp_path / "profiles.json"
    monkeypatch.setenv("MERCY_PROFILES_PATH", str(path))
    runtime.reset_runtime_cache()
    yield path
    runtime.reset_runtime_cache()


def test_profile_snapshot_is_immutable_and_takes_precedence(profile_path, monkeypatch):
    monkeypatch.setenv("MODEL_CONVERSATION", "openai:env-model")
    _write(
        profile_path,
        _document(
            {"conversation": "openai:profile-model@none"},
            profiles={
                "full": {"roles": {"conversation": "openai:profile-model@none"}},
                "lite": {"roles": {"conversation": "deepseek:deepseek-chat@none"}},
            },
        ),
    )

    snapshot = runtime.current_snapshot()

    assert snapshot.source == "profiles"
    assert snapshot.active == "full"
    assert snapshot.spec("conversation") == models.ModelSpec(
        "openai", "profile-model", "none"
    )
    assert snapshot.profiles_dict()["lite"]["conversation"] == "deepseek:deepseek-chat@none"
    assert hash(snapshot)
    with pytest.raises(TypeError):
        snapshot.roles["conversation"] = None
    with pytest.raises(TypeError):
        snapshot.profiles_view["full"]["conversation"] = "openai:changed"


def test_saving_seed_profiles_preserves_environment_roles(profile_path, monkeypatch):
    _clear_model_environment(monkeypatch)
    for variable, value in {
        "MODEL_CONVERSATION": "openai:conversation@none",
        "MODEL_CONVERSATION_FALLBACK": "deepseek:deepseek-chat@none",
        "MODEL_RESEARCH": "deepseek:deepseek-chat@none",
        "MODEL_RESEARCH_FALLBACK": "openai:research-fallback",
        "MODEL_RESEARCH_MAX": "deepseek:deepseek-reasoner@none",
        "MODEL_RESEARCH_MAX_FALLBACK": "deepseek:deepseek-chat@none",
        "MODEL_REPORT": "openai:report",
    }.items():
        monkeypatch.setenv(variable, value)
    for variable in (
        "MODEL_CONVERSATION_MAX",
        "MODEL_REPORT_MAX",
        "MODEL_VISION_MAX",
        "MODEL_CONVERSATION_MAX_FALLBACK",
        "MODEL_REPORT_MAX_FALLBACK",
        "MODEL_VISION_MAX_FALLBACK",
        "MODEL_VIDEO_MAX",
        "MODEL_VIDEO_FALLBACK",
        "MODEL_VIDEO_MAX_FALLBACK",
    ):
        monkeypatch.delenv(variable, raising=False)
    expected = {
        name: spec
        for name, spec in runtime._environment_roles().items()
        if name in runtime.PROFILE_ROLE_NAMES
    }

    runtime.save_profiles(runtime.seed_document()["profiles"])

    assert runtime.current_snapshot().roles == expected


def test_set_active_invalidates_snapshot_with_same_signature(profile_path, monkeypatch):
    roles = {"conversation": "openai:conversation"}
    runtime.save_profiles(
        {
            "full": {"roles": roles},
            "lite": {"roles": {"conversation": "openai:lite"}},
        }
    )
    monkeypatch.setattr(runtime, "_profile_signature", lambda _path: (1, 1))

    assert runtime.current_snapshot().active == "full"
    runtime.set_active("lite")
    assert runtime.current_snapshot().active == "lite"
    runtime.set_active("full")
    assert runtime.current_snapshot().active == "full"


def test_profile_writes_preserve_crlf_and_existing_active(profile_path):
    profiles = {
        "full": {"roles": {"conversation": "openai:full"}},
        "lite": {"roles": {"conversation": "openai:lite"}},
    }
    initial = {
        "version": 1,
        "active": "lite",
        "profiles": profiles,
    }
    profile_path.write_bytes(
        json.dumps(initial, indent=2).replace("\n", "\r\n").encode("utf-8")
    )

    runtime.save_profiles(profiles)

    saved = profile_path.read_bytes()
    assert b"\r\n" in saved
    assert b"\n" not in saved.replace(b"\r\n", b"")
    assert json.loads(saved)["active"] == "lite"
    runtime.set_active("full")
    assert b"\r\n" in profile_path.read_bytes()
    assert json.loads(profile_path.read_bytes())["active"] == "full"


def test_seed_dropped_lists_unrepresentable_environment_models(monkeypatch):
    values = (
        "MODEL_CONVERSATION_MAX",
        "MODEL_REPORT_MAX_FALLBACK",
        "MODEL_VIDEO_MAX",
    )
    for name in values:
        monkeypatch.setenv(name, "openai:fake-model")

    assert runtime.seed_dropped() == sorted(values)


@pytest.mark.parametrize(
    ("has_max", "has_tavily"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_research_status_tracks_model_and_tavily(
    profile_path, monkeypatch, has_max, has_tavily
):
    _clear_model_environment(monkeypatch)
    monkeypatch.setenv("MODEL_RESEARCH", "openai:research")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    if has_max:
        monkeypatch.setenv("MODEL_RESEARCH_MAX", "openai:research-max")
    if has_tavily:
        monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")
    runtime.reset_runtime_cache()

    config = runtime_config.current_config()
    rows = {row["role"]: row for row in config["roles"]}
    assert config["capabilities"]["max_tier"] is (has_max and has_tavily)
    assert rows["research"]["available"] is has_tavily
    assert rows["research"]["reason"] == (
        None if has_tavily else "未配置 Tavily key"
    )
    assert rows["research_max"]["available"] is (has_max and has_tavily)
    assert rows["research_max"]["reason"] == (
        None
        if has_max and has_tavily
        else "未配置 MODEL_RESEARCH_MAX"
        if not has_max
        else "未配置 Tavily key"
    )


def test_profile_research_status_requires_tavily(profile_path, monkeypatch):
    _clear_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    _write(
        profile_path,
        _document(
            {
                "conversation": "openai:conversation",
                "research": "openai:research",
                "research_max": "openai:research-max",
            }
        ),
    )

    config = runtime_config.current_config()
    rows = {row["role"]: row for row in config["roles"]}
    assert config["capabilities"]["max_tier"] is False
    for role in ("research", "research_max"):
        assert rows[role]["available"] is False
        assert rows[role]["reason"] == "未配置 Tavily key"


@pytest.mark.parametrize(
    ("document", "field"),
    [
        ({"version": 1, "active": "full", "profiles": {"full": {"roles": {"conversation": "openai:x"}}}, "extra": 1}, "profiles.json.extra"),
        (_document({"conversation": "openai:x"}, profiles={"full": {"roles": {"conversation": "openai:x"}, "extra": 1}}), "profiles.full.extra"),
        (_document({"conversation": "openai:x", "other": "openai:y"}), "profiles.full.roles.other"),
        (_document({"conversation": "openai:x"}, active="other"), "active"),
        (_document({"report": "openai:x"}), "profiles.full.roles.conversation"),
        (_document({"conversation": "gemini:x"}), "profiles.full.roles.conversation"),
        (_document({"conversation": "deepseek:x@high"}), "profiles.full.roles.conversation"),
        (_document({"conversation": "openai:x", "video": "openai:y"}), "profiles.full.roles.video"),
        (_document({"conversation": "openai:x", "video": "gemini:y@low"}), "profiles.full.roles.video"),
        (_document({"conversation": "openai:x", "memory": "openai:other"}), "profiles.full.roles.memory"),
        (_document({"conversation": "openai:x", "memory": "deepseek:text-embedding-3-small"}), "profiles.full.roles.memory"),
    ],
)
def test_profile_schema_rejects_invalid_fields(profile_path, document, field):
    _write(profile_path, document)
    with pytest.raises(models.ModelConfigError, match=re.escape(field)):
        runtime.validate_profiles_file()


def test_profiles_allow_inactive_profile_without_conversation(profile_path):
    _write(
        profile_path,
        _document(
            {},
            profiles={
                "full": {"roles": {"conversation": "openai:main"}},
                "lite": {"roles": {"report": "openai:report"}},
            },
        ),
    )
    assert runtime.current_snapshot().spec("conversation").model == "main"


def test_environment_mode_includes_memory_and_deepseek(profile_path, monkeypatch):
    profile_path.unlink(missing_ok=True)
    monkeypatch.setenv("MODEL_CONVERSATION", "deepseek:deepseek-chat@none")
    monkeypatch.setenv("MODEL_RESEARCH", "openai:research-model")
    monkeypatch.delenv("MODEL_RESEARCH_MAX", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "test-embedding")

    snapshot = runtime.current_snapshot()

    assert snapshot.source == "env"
    assert snapshot.spec("conversation") == models.ModelSpec(
        "deepseek", "deepseek-chat", "none"
    )
    assert snapshot.spec("memory") == models.ModelSpec(
        "openai", "test-embedding", None
    )
    assert snapshot.spec("research", "max") == snapshot.spec("research")
    assert not snapshot.available("research_max")


def test_startup_validation_seeds_snapshot_before_invalid_profile_reload(profile_path):
    _write(profile_path, _document({"conversation": "openai:startup-model"}))
    runtime.validate_startup_config()

    profile_path.write_text("{ invalid", encoding="utf-8")
    stat = profile_path.stat()
    os.utime(profile_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    snapshot = runtime.current_snapshot()

    assert snapshot.spec("conversation") == models.ModelSpec(
        "openai", "startup-model", None
    )
    assert snapshot.error is not None


def test_profile_reload_uses_new_mtime_and_retains_last_valid(profile_path):
    _write(profile_path, _document({"conversation": "openai:first"}))
    first = runtime.current_snapshot()
    _write(profile_path, _document({"conversation": "openai:second-model"}))
    stat = profile_path.stat()
    os.utime(profile_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    second = runtime.current_snapshot()
    assert second.spec("conversation").model == "second-model"
    assert second.version == first.version + 1

    _write(
        profile_path,
        {
            "version": 1,
            "active": "full",
            "profiles": {"full": {"roles": {"conversation": "openai:broken"}}},
            "unexpected": True,
        },
    )
    stat = profile_path.stat()
    os.utime(profile_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    invalid = runtime.current_snapshot()
    assert invalid.spec("conversation").model == "second-model"
    assert invalid.version == second.version
    assert "profiles.json.unexpected" in invalid.error


@pytest.mark.parametrize(
    (
        "max_configured",
        "tavily_configured",
        "expected_max_tier",
        "expected_research_reason",
        "expected_research_max_reason",
    ),
    [
        (False, False, False, "未配置 Tavily key", "未配置 MODEL_RESEARCH_MAX"),
        (False, True, False, None, "未配置 MODEL_RESEARCH_MAX"),
        (True, False, False, "未配置 Tavily key", "未配置 Tavily key"),
        (True, True, True, None, None),
    ],
)
def test_environment_config_research_rows_follow_investigate_availability(
    profile_path,
    monkeypatch,
    max_configured,
    tavily_configured,
    expected_max_tier,
    expected_research_reason,
    expected_research_max_reason,
):
    profile_path.unlink(missing_ok=True)
    _clear_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("MODEL_RESEARCH", "openai:research-model")
    if max_configured:
        monkeypatch.setenv("MODEL_RESEARCH_MAX", "openai:research-max-model")
    if tavily_configured:
        monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")

    config = runtime_config.current_config()

    research = next(row for row in config["roles"] if row["role"] == "research")
    research_max = next(
        row for row in config["roles"] if row["role"] == "research_max"
    )
    assert config["capabilities"]["max_tier"] is expected_max_tier
    assert research["available"] is tavily_configured
    assert research["reason"] == expected_research_reason
    assert research_max["available"] is (max_configured and tavily_configured)
    assert research_max["reason"] == expected_research_max_reason


def test_profile_config_research_rows_require_tavily(profile_path, monkeypatch):
    _clear_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    _write(
        profile_path,
        _document(
            {
                "conversation": "openai:conversation-model",
                "research": "openai:research-model",
                "research_max": "openai:research-max-model",
            }
        ),
    )

    config = runtime_config.current_config()

    research = next(row for row in config["roles"] if row["role"] == "research")
    research_max = next(
        row for row in config["roles"] if row["role"] == "research_max"
    )
    assert config["capabilities"]["max_tier"] is False
    for row in (research, research_max):
        assert row["available"] is False
        assert row["reason"] == "未配置 Tavily key"


@pytest.mark.asyncio
async def test_deepseek_capability_cache_success_ttl_and_model_match(
    profile_path, monkeypatch
):
    _write(profile_path, _document({"conversation": "deepseek:deepseek-flash@none"}))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    now = [100.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    calls = []

    async def fetch(_key):
        calls.append(1)
        return {"deepseek-flash": True, "another-model": True}

    monkeypatch.setattr(runtime, "_fetch_deepseek_capabilities", fetch)
    snapshot = await runtime.load_snapshot()
    assert snapshot.can_see_images()
    await runtime.load_snapshot()
    assert len(calls) == 1
    now[0] += runtime.DEEPSEEK_SUCCESS_TTL_SECONDS + 1
    assert (await runtime.load_snapshot()).can_see_images()
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_deepseek_capability_failure_is_cached_and_fails_closed(
    profile_path, monkeypatch
):
    _write(profile_path, _document({"conversation": "deepseek:deepseek-flash"}))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    now = [100.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    calls = []

    async def fail(_key):
        calls.append(1)
        raise httpx.ConnectError("test network failure")

    monkeypatch.setattr(runtime, "_fetch_deepseek_capabilities", fail)
    snapshot = await runtime.load_snapshot()
    assert not snapshot.can_see_images()
    await runtime.load_snapshot()
    assert len(calls) == 1
    now[0] += runtime.DEEPSEEK_FAILURE_TTL_SECONDS + 1
    await runtime.load_snapshot()
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_deepseek_key_change_invalidates_sync_capability_snapshot(
    profile_path, monkeypatch
):
    _write(profile_path, _document({"conversation": "deepseek:deepseek-flash"}))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-first-key")
    calls = []

    async def fetch(key):
        calls.append(key)
        return {"deepseek-flash": True}

    monkeypatch.setattr(runtime, "_fetch_deepseek_capabilities", fetch)
    assert (await runtime.load_snapshot()).can_see_images()
    assert len(calls) == 1

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-second-key")
    assert not runtime.current_snapshot().can_see_images()
    assert len(calls) == 1
    assert (await runtime.load_snapshot()).can_see_images()
    assert len(calls) == 2


def test_runtime_from_uses_snapshot_attached_to_turn_config(profile_path):
    _write(profile_path, _document({"conversation": "openai:turn-one"}))
    turn_snapshot = runtime.current_snapshot()
    _write(profile_path, _document({"conversation": "openai:turn-two"}))
    stat = profile_path.stat()
    os.utime(profile_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert runtime.current_snapshot().spec("conversation").model == "turn-two"
    attached = runtime.runtime_from({"configurable": {"runtime": turn_snapshot}})
    assert attached is turn_snapshot
    assert attached.spec("conversation").model == "turn-one"


def test_provider_key_resolution_uses_documented_precedence(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    assert models.provider_key("gemini") == ("test-gemini-key", "GEMINI_API_KEY")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert models.provider_key("gemini") == ("test-google-key", "GOOGLE_API_KEY")


def test_lite_profile_imports_main_without_openai_client_or_network(
    tmp_path, monkeypatch
):
    profile_file = tmp_path / "profiles.json"
    _write(
        profile_file,
        _document(
            {"conversation": "deepseek:deepseek-chat@none"},
            active="lite",
        ),
    )
    environment = os.environ.copy()
    environment.update(
        {
            "MERCY_PROFILES_PATH": str(profile_file),
            "DEEPSEEK_API_KEY": "test-deepseek-key",
            "OPENAI_API_KEY": "",
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "TAVILY_API_KEY": "",
            "MERCY_SYSTEM_KEYS": '["DEEPSEEK_API_KEY"]',
        }
    )
    code = """
import main
from core import runtime
from fastapi.testclient import TestClient

async def no_network(_key):
    return {}

runtime._fetch_deepseek_capabilities = no_network
client = TestClient(main.app)
config = client.get("/api/config")
capabilities = client.get("/api/capabilities")
assert config.status_code == 200
assert capabilities.status_code == 200
payload = config.json()
assert payload["profile"] == "便捷模式"
assert next(role for role in payload["roles"] if role["role"] == "conversation")["available"]
assert not next(role for role in payload["roles"] if role["role"] == "memory")["available"]
assert not next(role for role in payload["roles"] if role["role"] == "video")["available"]
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
