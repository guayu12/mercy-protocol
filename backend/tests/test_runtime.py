import json
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from core import models, runtime


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
