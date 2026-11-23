import json
import os

import httpx
from fastapi.testclient import TestClient
import pytest
from langchain_core.messages import AIMessage

from core import config_service, env, models, runtime, usage_stats
from core.agent import graph
import main


@pytest.fixture(autouse=True)
def _local_backend(monkeypatch):
    monkeypatch.setenv("BACKEND_HOST", "127.0.0.1")
    monkeypatch.setattr(main, "CORS_ORIGINS", ["http://127.0.0.1:1420"])


def _client(address=("127.0.0.1", 43123)):
    return TestClient(main.app, client=address)


def _headers():
    return {
        "Host": "127.0.0.1:8000",
        "Origin": "http://127.0.0.1:1420",
        "Content-Type": "application/json",
    }


def _clear_keys(monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "TAVILY_API_KEY",
        "MERCY_SYSTEM_KEYS",
    ):
        monkeypatch.delenv(name, raising=False)
    for role in (*models.CHAT_ROLES, "video"):
        for suffix in ("", "_MAX", "_FALLBACK", "_MAX_FALLBACK"):
            monkeypatch.delenv(f"MODEL_{role.upper()}{suffix}", raising=False)
    for name in (
        "DEFAULT_MODEL",
        "DEFAULT_REASONING_EFFORT",
        "OPENAI_VISION_MODEL",
        "OPENAI_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_local_config_request_rejects_remote_client_and_bad_payload():
    client = _client(("192.0.2.5", 43123))
    response = client.post(
        "/api/config/active",
        headers=_headers(),
        json={"active": "full"},
    )
    assert response.status_code == 403
    assert response.json() == {
        "category": "forbidden",
        "message": "只能在本机修改配置",
    }

    response = _client().post(
        "/api/config/key",
        headers=_headers(),
        json={"provider": "invalid", "api_key": "fake-secret-sentinel"},
    )
    assert response.status_code == 400
    assert response.json() == {
        "category": "invalid_request",
        "message": "请求格式不正确",
    }
    assert "fake-secret-sentinel" not in response.text


def test_local_config_request_checks_content_type_size_and_host():
    client = _client()
    response = client.post(
        "/api/config/active",
        headers={**_headers(), "Content-Type": "text/plain"},
        content=b'{"active":"full"}',
    )
    assert response.status_code == 415

    response = client.post(
        "/api/config/active",
        headers=_headers(),
        content=b" " * 65537,
    )
    assert response.status_code == 413

    response = client.post(
        "/api/config/active",
        headers={**_headers(), "Host": "192.0.2.5:8000"},
        json={"active": "full"},
    )
    assert response.status_code == 403

    response = _client().post(
        "/api/config/key",
        headers=_headers(),
        json={"provider": "openai", "api_key": "bad key"},
    )
    assert response.status_code == 400
    assert response.json() == {
        "category": "invalid_key",
        "message": "key 格式不正确",
    }


def test_config_models_response_and_tavily_rejection(monkeypatch):
    async def fake_fetch(provider, _key):
        assert provider == "openai"
        return [{"id": "gpt-fake", "image_input": True, "efforts": ["high"]}]

    monkeypatch.setattr(config_service, "_fetch_models", fake_fetch)
    response = _client().post(
        "/api/config/models",
        headers=_headers(),
        json={"provider": "openai", "api_key": "fake-model-key"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "models": [
            {"id": "gpt-fake", "image_input": True, "efforts": ["high"]}
        ]
    }

    response = _client().post(
        "/api/config/models",
        headers=_headers(),
        json={"provider": "tavily"},
    )
    assert response.status_code == 400
    assert response.json()["message"] == "Tavily 没有模型列表"


@pytest.mark.asyncio
async def test_provider_model_listing_filters_and_auth_headers(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, *, timeout):
            assert timeout.connect == 8.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, *, headers):
            calls.append((url, headers))
            if "generativelanguage" in url:
                payload = {
                    "models": [
                        {
                            "name": "models/gemini-2.5-flash",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                        {
                            "name": "models/gemini-tts",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                        {
                            "name": "models/other",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                    ]
                }
            elif "deepseek" in url:
                payload = {
                    "data": [
                        {
                            "id": "deepseek-chat",
                            "input_modalities": ["text", "image"],
                            "effort": {
                                "supported_levels": [
                                    "low",
                                    "high",
                                    "max",
                                    "unsupported",
                                    "high",
                                    "none",
                                ],
                                "default_level": "high",
                            },
                        }
                    ]
                }
            else:
                payload = {
                    "data": [
                        {"id": "gpt-chat"},
                        {"id": "text-embedding-3-small"},
                        {"id": "whisper-1"},
                    ]
                }
            request = httpx.Request("GET", url, headers=headers)
            return httpx.Response(200, json=payload, request=request)

    monkeypatch.setattr(config_service.httpx, "AsyncClient", FakeClient)
    openai_models = await config_service._fetch_models("openai", "fake-openai-key")
    deepseek_models = await config_service._fetch_models(
        "deepseek", "fake-deepseek-key"
    )
    gemini_models = await config_service._fetch_models("gemini", "fake-gemini-key")

    assert openai_models == [
        {
            "id": "gpt-chat",
            "image_input": True,
            "efforts": [
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
            ],
        }
    ]
    assert deepseek_models == [
        {
            "id": "deepseek-chat",
            "image_input": True,
            "efforts": ["none", "low", "high", "max"],
        }
    ]
    assert gemini_models == [
        {"id": "gemini-2.5-flash", "image_input": True, "efforts": []}
    ]
    assert calls[0][1]["Authorization"] == "Bearer fake-openai-key"
    assert calls[1][1]["Authorization"] == "Bearer fake-deepseek-key"
    assert calls[2][1]["x-goog-api-key"] == "fake-gemini-key"
    assert "fake-gemini-key" not in calls[2][0]
    assert calls[2][0].endswith("/models?pageSize=1000")


@pytest.mark.asyncio
async def test_model_cache_only_keeps_successes(monkeypatch):
    calls = []

    async def succeeds(provider, _key):
        calls.append(provider)
        return [{"id": provider, "image_input": False, "efforts": []}]

    monkeypatch.setattr(config_service, "_fetch_models", succeeds)
    config_service.clear_model_cache("openai")
    assert await config_service.list_models("openai", "fake-cache-key")
    assert await config_service.list_models("openai", "fake-cache-key")
    assert calls == ["openai"]

    async def fails(provider, _key):
        calls.append(provider)
        raise httpx.ConnectError("failed")

    monkeypatch.setattr(config_service, "_fetch_models", fails)
    config_service.clear_model_cache("deepseek")
    for _ in range(2):
        with pytest.raises(httpx.ConnectError):
            await config_service.list_models("deepseek", "fake-failure-key")
    assert calls[-2:] == ["deepseek", "deepseek"]


def test_invalid_provider_key_leaves_env_file_unchanged(tmp_path, monkeypatch):
    _clear_keys(monkeypatch)
    path = tmp_path / ".env"
    path.write_text("OTHER=value\n", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", path)
    env.load_env()
    before = path.read_bytes()

    async def invalid(_provider, _key):
        response = httpx.Response(401, request=httpx.Request("GET", "https://example.invalid"))
        response.raise_for_status()

    monkeypatch.setattr(config_service, "_validate_provider_key", invalid)
    response = _client().post(
        "/api/config/key",
        headers=_headers(),
        json={"provider": "openai", "api_key": "fake-invalid-key"},
    )

    assert response.status_code == 400
    assert response.json() == {"category": "invalid_key", "message": "key 无效"}
    assert path.read_bytes() == before


def test_sourced_env_key_can_be_saved_from_config(tmp_path, monkeypatch):
    _clear_keys(monkeypatch)
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=fake-original-openai-key\n", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", path)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-original-openai-key")
    env.load_env()
    assert env.provider_key_source("openai") == "env_file"

    async def valid(_provider, _key):
        return []

    monkeypatch.setattr(config_service, "_validate_provider_key", valid)
    response = _client().post(
        "/api/config/key",
        headers=_headers(),
        json={"provider": "openai", "api_key": "fake-replacement-openai-key"},
    )

    assert response.status_code == 200
    assert response.json()["key"]["source"] == "env_file"
    assert "OPENAI_API_KEY=fake-replacement-openai-key" in path.read_text(
        encoding="utf-8"
    )


def test_gemini_invalid_key_400_does_not_write_env_file(tmp_path, monkeypatch):
    _clear_keys(monkeypatch)
    path = tmp_path / ".env"
    path.write_text("OTHER=fake-value\n", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", path)
    env.load_env()
    before = path.read_bytes()
    body = {
        "error": {
            "code": 400,
            "message": "API key not valid. Please pass a valid API key.",
            "status": "INVALID_ARGUMENT",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "API_KEY_INVALID",
                    "domain": "googleapis.com",
                }
            ],
        }
    }

    async def invalid(_provider, _key):
        request = httpx.Request(
            "GET", "https://generativelanguage.googleapis.com/v1beta/models"
        )
        response = httpx.Response(400, json=body, request=request)
        raise httpx.HTTPStatusError(
            "fake Gemini error", request=request, response=response
        )

    monkeypatch.setattr(config_service, "_validate_provider_key", invalid)
    response = _client().post(
        "/api/config/key",
        headers=_headers(),
        json={"provider": "gemini", "api_key": "fake-invalid-gemini-key"},
    )

    assert response.status_code == 400
    assert response.json() == {"category": "invalid_key", "message": "key 无效"}
    assert path.read_bytes() == before


def test_key_save_rejects_system_source_and_bad_env_encoding(tmp_path, monkeypatch):
    _clear_keys(monkeypatch)
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=file-fake-key\n", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", path)
    monkeypatch.setenv("OPENAI_API_KEY", "system-fake-key")
    env.load_env()
    response = _client().post(
        "/api/config/key",
        headers=_headers(),
        json={"provider": "openai", "api_key": "another-fake-key"},
    )
    assert response.status_code == 409
    assert response.json()["message"] == "这个 key 来自系统环境变量，请在系统里修改"

    path.write_bytes("OPENAI_API_KEY=fake".encode("utf-16"))
    env.load_env()
    response = _client().post(
        "/api/config/key",
        headers=_headers(),
        json={"provider": "openai", "api_key": "another-fake-key"},
    )
    assert response.status_code == 400
    assert response.json()["message"] == (
        "backend/.env 不是 UTF-8 编码，请另存为 UTF-8 后重启"
    )


def test_active_without_profiles_returns_fixed_error(tmp_path, monkeypatch):
    monkeypatch.setenv("MERCY_PROFILES_PATH", str(tmp_path / "missing.json"))
    runtime.reset_runtime_cache()
    response = _client().post(
        "/api/config/active",
        headers=_headers(),
        json={"active": "full"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "category": "profiles_missing",
        "message": "请先保存配置",
    }
    runtime.reset_runtime_cache()


def test_save_profiles_allows_inactive_profile_without_conversation(
    tmp_path, monkeypatch
):
    _clear_keys(monkeypatch)
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", env_path)
    env.load_env()
    monkeypatch.setenv("MERCY_PROFILES_PATH", str(tmp_path / "profiles.json"))
    runtime.reset_runtime_cache()
    profiles = {
        "full": {"roles": {"conversation": "openai:fake-full"}},
        "lite": {"roles": {"report": "openai:fake-lite-report"}},
    }

    response = _client().post(
        "/api/config/profiles",
        headers=_headers(),
        json={"profiles": profiles},
    )

    assert response.status_code == 200
    assert response.json()["config"]["active"] == "full"
    assert {
        "profile": "lite",
        "role": "report",
        "reason": "未设置对话模型",
    } in response.json()["untested"]
    runtime.reset_runtime_cache()


@pytest.mark.asyncio
async def test_deepseek_vision_fallback_checks_image_capability(monkeypatch):
    _clear_keys(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek-capability-key")
    document = {
        "version": 1,
        "active": "full",
        "profiles": {
            "full": {
                "roles": {
                    "conversation": "deepseek:deepseek-chat@none",
                    "vision_fallback": "deepseek:deepseek-vision@none",
                }
            }
        },
    }
    snapshot = runtime.candidate_snapshot(document, "full")
    model_requests = []
    llm_requests = []

    async def listed(provider, key):
        model_requests.append((provider, key))
        return [{"id": "deepseek-vision", "image_input": True}]

    class FakeLLM:
        async def ainvoke(self, _prompt):
            return None

    def get_llm(role, tier, *, fallback, runtime, **_kwargs):
        spec = runtime.spec(role, tier, fallback=fallback)
        llm_requests.append((role, tier, fallback, runtime.image_input.get(spec)))
        return FakeLLM() if runtime.image_input.get(spec) else None

    monkeypatch.setattr(config_service, "list_models", listed)
    monkeypatch.setattr(models, "get_llm", get_llm)

    await config_service._test_role(snapshot, "full", "vision_fallback")

    assert model_requests == [
        ("deepseek", "fake-deepseek-capability-key")
    ]
    assert llm_requests == [("vision", "normal", True, True)]


def test_profiles_without_keys_are_saved_as_untested(tmp_path, monkeypatch):
    _clear_keys(monkeypatch)
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", env_path)
    env.load_env()
    monkeypatch.setenv("MERCY_PROFILES_PATH", str(tmp_path / "profiles.json"))
    runtime.reset_runtime_cache()
    profiles = {
        "full": {"roles": {"conversation": "openai:fake-chat"}},
        "lite": {"roles": {"conversation": "deepseek:deepseek-chat@none"}},
    }

    response = _client().post(
        "/api/config/profiles",
        headers=_headers(),
        json={"profiles": profiles},
    )

    assert response.status_code == 200
    assert response.json()["untested"] == [
        {"profile": "full", "role": "conversation", "reason": "缺少 key"},
        {"profile": "lite", "role": "conversation", "reason": "缺少 key"},
    ]
    runtime.reset_runtime_cache()


def test_failed_profile_test_does_not_write_file(tmp_path, monkeypatch):
    _clear_keys(monkeypatch)
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=fake-profile-test-key\n", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", env_path)
    env.load_env()
    profiles_path = tmp_path / "profiles.json"
    monkeypatch.setenv("MERCY_PROFILES_PATH", str(profiles_path))
    original_profiles = {
        "full": {"roles": {"conversation": "openai:old-chat"}},
        "lite": {"roles": {"conversation": "openai:lite-chat"}},
    }
    runtime.save_profiles(original_profiles)
    before = profiles_path.read_bytes()

    class BrokenLLM:
        async def ainvoke(self, _prompt):
            response = httpx.Response(
                400, request=httpx.Request("POST", "https://example.invalid")
            )
            response.raise_for_status()

    monkeypatch.setattr(
        config_service.models,
        "get_llm",
        lambda *_args, **_kwargs: BrokenLLM(),
    )
    changed = {
        "full": {"roles": {"conversation": "openai:new-chat"}},
        "lite": {"roles": {"conversation": "openai:lite-chat"}},
    }

    response = _client().post(
        "/api/config/profiles",
        headers=_headers(),
        json={"profiles": changed},
    )

    assert response.status_code == 400
    assert response.json()["category"] == "role_test_failed"
    assert response.json()["errors"] == [
        {
            "profile": "full",
            "role": "conversation",
            "category": "bad_request",
            "message": "模型不接受这个设置",
        }
    ]
    assert profiles_path.read_bytes() == before
    runtime.reset_runtime_cache()


def test_key_save_response_and_profiles_active_shapes(tmp_path, monkeypatch):
    _clear_keys(monkeypatch)
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", env_path)
    env.load_env()

    async def valid(_provider, _key):
        return None

    monkeypatch.setattr(config_service, "_validate_provider_key", valid)
    new_key = "fake-provider-key-123456"
    response = _client().post(
        "/api/config/key",
        headers=_headers(),
        json={"provider": "openai", "api_key": new_key},
    )
    assert response.status_code == 200
    assert response.json() == {
        "key": {
            "configured": True,
            "hint": f"••••{new_key[-4:]}",
            "variable": "OPENAI_API_KEY",
            "source": "env_file",
        }
    }
    assert new_key not in response.text

    profiles_path = tmp_path / "profiles.json"
    monkeypatch.setenv("MERCY_PROFILES_PATH", str(profiles_path))
    runtime.reset_runtime_cache()
    profiles = {
        "full": {"roles": {"conversation": "openai:fake-chat"}},
        "lite": {"roles": {"conversation": "openai:fake-lite"}},
    }
    monkeypatch.setattr(
        config_service.models,
        "get_llm",
        lambda *_args, **_kwargs: type(
            "FakeLLM", (), {"ainvoke": lambda self, _prompt: _async_ok()}
        )(),
    )
    usage_stats._model_stats.clear()
    response = _client().post(
        "/api/config/profiles",
        headers=_headers(),
        json={"profiles": profiles},
    )
    assert response.status_code == 200, response.text
    saved = response.json()
    assert saved["ok"] is True
    assert saved["untested"] == []
    assert saved["config"]["profile_source"] == "file"
    assert saved["config"]["profiles"] == {
        "full": {"conversation": "openai:fake-chat"},
        "lite": {"conversation": "openai:fake-lite"},
    }
    assert usage_stats.snapshot()["models"] == []

    response = _client().post(
        "/api/config/active",
        headers=_headers(),
        json={"active": "lite"},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["config"]["active"] == "lite"
    runtime.reset_runtime_cache()


async def _async_ok():
    return AIMessage(content="ok")


def test_get_config_has_seed_and_key_source_without_raw_key(tmp_path, monkeypatch):
    _clear_keys(monkeypatch)
    path = tmp_path / ".env"
    path.write_text("", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", path)
    monkeypatch.setenv("MERCY_PROFILES_PATH", str(tmp_path / "absent.json"))
    monkeypatch.setenv("MODEL_CONVERSATION", "openai:fake-chat")
    env.load_env()
    runtime.reset_runtime_cache()

    response = _client().get("/api/config", headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["profile_source"] == "env"
    assert "seed" in body and "seed_dropped" in body
    assert body["keys"]["openai"]["source"] is None
    assert body["env_error"] is None
    assert body["editable"] is True
    runtime.reset_runtime_cache()


@pytest.mark.parametrize(
    "corrupt",
    [
        b"{not json",
        b'{"version":1,"active":"invalid","profiles":{}}',
    ],
)
def test_get_config_keeps_last_valid_profiles_after_corruption(
    tmp_path, monkeypatch, corrupt
):
    _clear_keys(monkeypatch)
    path = tmp_path / "profiles.json"
    monkeypatch.setenv("MERCY_PROFILES_PATH", str(path))
    profiles = {
        "full": {"roles": {"conversation": "openai:last-valid"}},
        "lite": {"roles": {"conversation": "openai:lite"}},
    }
    runtime.save_profiles(profiles)
    runtime.current_snapshot()
    path.write_bytes(corrupt)

    response = _client().get("/api/config", headers=_headers())

    assert response.status_code == 200
    body = response.json()
    assert body["profile_source"] == "file"
    assert body["profiles"] == {
        "full": {"conversation": "openai:last-valid"},
        "lite": {"conversation": "openai:lite"},
    }
    assert body["profile_error"]
    runtime.reset_runtime_cache()


@pytest.mark.parametrize(
    ("corrupt", "active"),
    [
        (b"{not json", "full"),
        (b'{"version":1,"active":"invalid","profiles":{}}', "full"),
        (b'{"version":2,"active":"lite","profiles":{"unknown":{}}}', "lite"),
    ],
)
def test_save_profiles_overwrites_corrupt_file(tmp_path, monkeypatch, corrupt, active):
    _clear_keys(monkeypatch)
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", env_path)
    env.load_env()
    path = tmp_path / "profiles.json"
    path.write_bytes(corrupt)
    monkeypatch.setenv("MERCY_PROFILES_PATH", str(path))
    runtime.reset_runtime_cache()
    profiles = {
        "full": {"roles": {"conversation": "openai:fake-full"}},
        "lite": {"roles": {"conversation": "openai:fake-lite"}},
    }

    response = _client().post(
        "/api/config/profiles",
        headers=_headers(),
        json={"profiles": profiles},
    )

    assert response.status_code == 200
    assert json.loads(path.read_bytes()) == {
        "version": 1,
        "active": active,
        "profiles": profiles,
    }
    assert len(response.json()["untested"]) == 2
    runtime.reset_runtime_cache()


def test_active_profile_rejects_corrupt_file_with_save_guidance(
    tmp_path, monkeypatch
):
    path = tmp_path / "profiles.json"
    path.write_bytes(b"{not json")
    monkeypatch.setenv("MERCY_PROFILES_PATH", str(path))
    runtime.reset_runtime_cache()

    response = _client().post(
        "/api/config/active",
        headers=_headers(),
        json={"active": "full"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "category": "invalid_profiles",
        "message": "配置文件格式不正确，请先在界面保存一次配置",
    }
    runtime.reset_runtime_cache()
