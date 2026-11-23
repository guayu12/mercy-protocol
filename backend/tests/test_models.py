from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import HumanMessage

from core import models


def clear_model_environment(monkeypatch):
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
        "OPENAI_BASE_URL",
        "VIDEO_MAX_MINUTES",
        "GEMINI_FILE_RETENTION_SECONDS",
        "PDF_MAX_PAGES",
        "MERCY_PROFILES_PATH",
    ):
        monkeypatch.delenv(name, raising=False)


def test_parse_spec_splits_provider_model_and_last_effort():
    assert models.parse_spec(
        "MODEL_RESEARCH", " openai:ft:gpt-x:org@reason@high "
    ) == models.ModelSpec("openai", "ft:gpt-x:org@reason", "high")
    assert models.parse_spec("MODEL_X", "openai:gpt-x") == models.ModelSpec(
        "openai", "gpt-x", None
    )
    assert models.parse_spec("MODEL_X", "  ") is None


@pytest.mark.parametrize(
    "value",
    [
        "gpt-x",
        ":gpt-x",
        "openai:",
        "openai:gpt-x@",
    ],
)
def test_parse_spec_errors_name_the_variable(value):
    with pytest.raises(models.ModelConfigError, match="MODEL_TEST"):
        models.parse_spec("MODEL_TEST", value)


def test_parse_spec_rejects_unknown_provider():
    with pytest.raises(models.ModelConfigError, match="MODEL_TEST"):
        models.parse_spec("MODEL_TEST", "other:gpt-x")


@pytest.mark.parametrize(
    "effort",
    ["none", "minimal", "low", "medium", "high", "xhigh"],
)
def test_parse_spec_accepts_supported_reasoning_efforts(effort):
    assert models.parse_spec(
        "MODEL_REPORT", f"openai:report-model@{effort}"
    ) == models.ModelSpec("openai", "report-model", effort)


def test_validate_config_rejects_unsupported_reasoning_effort(monkeypatch):
    clear_model_environment(monkeypatch)
    monkeypatch.setenv("MODEL_CONVERSATION", "openai:chat-model@unsupported")

    with pytest.raises(
        models.ModelConfigError,
        match="MODEL_CONVERSATION: unsupported reasoning effort",
    ):
        models.validate_config()


def test_validate_config_checks_legacy_reasoning_effort(monkeypatch):
    clear_model_environment(monkeypatch)
    monkeypatch.setenv("DEFAULT_REASONING_EFFORT", "unsupported")

    with pytest.raises(
        models.ModelConfigError,
        match="DEFAULT_REASONING_EFFORT: unsupported reasoning effort",
    ):
        models.validate_config()


def test_validate_config_rejects_gemini_chat_provider(monkeypatch):
    clear_model_environment(monkeypatch)
    monkeypatch.setenv("MODEL_CONVERSATION_MAX_FALLBACK", "gemini:model")
    with pytest.raises(models.ModelConfigError, match="chat roles require"):
        models.validate_config()


def test_deepseek_is_allowed_for_chat_roles_and_efforts_are_role_specific(
    monkeypatch,
):
    clear_model_environment(monkeypatch)
    monkeypatch.setenv("MODEL_CONVERSATION", "deepseek:deepseek-chat@none")
    monkeypatch.setenv("MODEL_REPORT", "deepseek:deepseek-chat@high")
    monkeypatch.setenv("MODEL_REPORT_FALLBACK", "deepseek:backup@max")
    models.validate_config()
    assert models.get_model_spec("conversation") == models.ModelSpec(
        "deepseek", "deepseek-chat", "none"
    )
    assert models.get_model_spec("report") == models.ModelSpec(
        "deepseek", "deepseek-chat", "high"
    )

    monkeypatch.setenv("MODEL_RESEARCH", "deepseek:deepseek-chat@low")
    with pytest.raises(
        models.ModelConfigError,
        match="MODEL_RESEARCH: unsupported DeepSeek reasoning effort",
    ):
        models.validate_config()


@pytest.mark.parametrize(
    ("role", "value", "expected_api"),
    [
        ("conversation", "deepseek:deepseek-chat", "chat"),
        ("conversation", "deepseek:deepseek-chat@none", "chat"),
        ("report", "deepseek:deepseek-reasoner@low", "responses"),
        ("report", "deepseek:deepseek-reasoner@high", "responses"),
        ("report", "deepseek:deepseek-reasoner@max", "responses"),
        ("conversation", "openai:gpt-6-luna", "chat"),
        ("conversation", "openai:gpt-6-luna@none", "responses"),
    ],
)
def test_api_kind_matches_chat_openai_payload_detection(
    monkeypatch, role, value, expected_api
):
    clear_model_environment(monkeypatch)
    provider = value.split(":", 1)[0]
    key_name = "DEEPSEEK_API_KEY" if provider == "deepseek" else "OPENAI_API_KEY"
    monkeypatch.setenv(key_name, "test-provider-key")
    monkeypatch.setenv(f"MODEL_{role.upper()}", value)
    spec = models.get_model_spec(role)
    llm = models.get_llm(role)
    payload = llm._get_request_payload([HumanMessage(content="payload check")])
    assert models.api_kind(spec) == expected_api
    assert llm._use_responses_api(payload) == (expected_api == "responses")
    if provider == "deepseek" and expected_api == "chat":
        assert llm.extra_body == {"thinking": {"type": "disabled"}}
        assert llm.openai_api_base == "https://api.deepseek.com"
    if provider == "deepseek" and expected_api == "responses":
        assert llm.reasoning == {"effort": spec.effort}
        assert llm.use_responses_api is True


def test_video_role_accepts_gemini(monkeypatch):
    clear_model_environment(monkeypatch)
    monkeypatch.setenv("MODEL_VIDEO", "gemini:gemini-video")
    models.validate_config()
    assert models.get_model_spec("video") == models.ModelSpec(
        "gemini", "gemini-video", None
    )
    assert models.get_model_spec("video", fallback=True) is None


def test_video_limit_and_gemini_retention_defaults(monkeypatch):
    clear_model_environment(monkeypatch)
    assert models.video_max_minutes() == 30
    assert models.gemini_file_retention_seconds() == 900
    monkeypatch.setenv("VIDEO_MAX_MINUTES", "not-an-integer")
    with pytest.raises(models.ModelConfigError, match="VIDEO_MAX_MINUTES"):
        models.validate_config()


def test_pdf_max_pages_default_and_validation(monkeypatch):
    clear_model_environment(monkeypatch)
    assert models.pdf_max_pages() == 300
    monkeypatch.setenv("PDF_MAX_PAGES", "0")
    with pytest.raises(models.ModelConfigError, match="PDF_MAX_PAGES"):
        models.validate_config()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("MODEL_VIDEO", "openai:video-model", "MODEL_VIDEO: video models"),
        ("MODEL_VIDEO", "gemini:video-model@high", "MODEL_VIDEO: reasoning effort"),
        (
            "MODEL_VIDEO_FALLBACK",
            "gemini:backup",
            "MODEL_VIDEO_FALLBACK: video models do not support fallbacks",
        ),
        (
            "MODEL_VIDEO_MAX_FALLBACK",
            "gemini:backup",
            "MODEL_VIDEO_MAX_FALLBACK: video models do not support fallbacks",
        ),
    ],
)
def test_validate_config_rejects_unsupported_video_config(
    name, value, message, monkeypatch
):
    clear_model_environment(monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(models.ModelConfigError, match=message):
        models.validate_config()


def test_role_resolution_legacy_max_fallback_and_keys(monkeypatch):
    clear_model_environment(monkeypatch)
    monkeypatch.setenv("DEFAULT_MODEL", "legacy-model")
    monkeypatch.setenv("DEFAULT_REASONING_EFFORT", "high")
    assert models.get_model_spec("conversation") == models.ModelSpec(
        "openai", "legacy-model", "high"
    )
    assert models.get_model_spec("report") is None
    assert models.get_model_spec("vision") is None

    monkeypatch.setenv("OPENAI_VISION_MODEL", "legacy-vision")
    assert models.get_model_spec("vision") == models.ModelSpec(
        "openai", "legacy-vision", None
    )
    monkeypatch.setenv("MODEL_CONVERSATION", "openai:configured@low")
    assert models.get_model_spec("conversation") == models.ModelSpec(
        "openai", "configured", "low"
    )

    monkeypatch.setenv("MODEL_RESEARCH", "openai:normal@low")
    monkeypatch.setenv("MODEL_RESEARCH_MAX", "openai:max@high")
    monkeypatch.setenv("MODEL_RESEARCH_FALLBACK", "openai:fallback@high")
    assert models.get_model_spec("research", "max") == models.ModelSpec(
        "openai", "max", "high"
    )
    assert models.get_model_spec("research", "normal", fallback=True) == models.ModelSpec(
        "openai", "fallback", "high"
    )
    assert models.get_model_spec("research", "max", fallback=True) == models.ModelSpec(
        "openai", "fallback", "high"
    )
    monkeypatch.setenv(
        "MODEL_RESEARCH_MAX_FALLBACK", "openai:max-fallback@medium"
    )
    assert models.get_model_spec("research", "max", fallback=True) == models.ModelSpec(
        "openai", "max-fallback", "medium"
    )
    monkeypatch.delenv("MODEL_RESEARCH_MAX")
    assert models.get_model_spec("research", "max") == models.ModelSpec(
        "openai", "normal", "low"
    )
    assert not models.role_available("research")
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    assert models.role_available("research")


def test_provider_availability_uses_provider_specific_keys(monkeypatch):
    clear_model_environment(monkeypatch)
    monkeypatch.setenv("MODEL_VIDEO", "gemini:video-model")
    assert not models.role_available("video")
    monkeypatch.setenv("GOOGLE_API_KEY", "not-a-real-key")
    assert models.role_available("video")
    monkeypatch.setenv("MODEL_CONVERSATION", "deepseek:chat-model")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
    assert models.role_available("conversation")


def test_get_llm_uses_langchain_token_limit_for_each_api_mode(monkeypatch):
    clear_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("MODEL_RESEARCH", "openai:gpt-6.2-sol@high")
    research = models.get_llm("research", max_tokens=8000)
    payload = research._get_request_payload(
        [HumanMessage(content="payload check")]
    )
    assert payload["max_output_tokens"] == 8000
    assert payload["reasoning"] == {"effort": "high"}
    assert research.openai_api_base == "https://api.openai.com/v1"

    monkeypatch.setenv("MODEL_REPORT", "openai:gpt-6.1-terra")
    report = models.get_llm("report", max_tokens=12000)
    payload = report._get_request_payload([HumanMessage(content="payload check")])
    assert payload["max_completion_tokens"] == 12000


@pytest.mark.asyncio
async def test_chat_api_model_tier_validation_routing_and_capabilities(monkeypatch):
    import main

    clear_model_environment(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    seen = []

    async def fake_process_query(*args):
        seen.append(args)
        yield {"type": "done", "turn_id": "test", "spoken_text": ""}

    monkeypatch.setattr(main, "process_query", fake_process_query)
    monkeypatch.setenv("MODEL_RESEARCH", "openai:configured-research")
    monkeypatch.setenv("MODEL_RESEARCH_MAX", "openai:configured-max")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        invalid = await client.post(
            "/api/chat", json={"message": "hello", "model_tier": "ultra"}
        )
        assert invalid.status_code == 422

        response = await client.post(
            "/api/chat", json={"message": "hello", "model_tier": "max"}
        )
        assert response.status_code == 200
        assert seen[0][-1] == "max"

        capabilities = await client.get("/api/capabilities")
        assert capabilities.json() == {
            "max_tier": True,
            "video": False,
            "document": False,
            "image_input": True,
        }
        monkeypatch.delenv("TAVILY_API_KEY")
        capabilities = await client.get("/api/capabilities")
        assert capabilities.json() == {
            "max_tier": False,
            "video": False,
            "document": False,
            "image_input": True,
        }
        monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
        monkeypatch.delenv("MODEL_RESEARCH")
        capabilities = await client.get("/api/capabilities")
        assert capabilities.json() == {
            "max_tier": False,
            "video": False,
            "document": False,
            "image_input": True,
        }
        monkeypatch.setenv("MODEL_RESEARCH", "openai:configured-research")
        monkeypatch.delenv("MODEL_RESEARCH_MAX")
        capabilities = await client.get("/api/capabilities")
        assert capabilities.json() == {
            "max_tier": True,
            "video": False,
            "document": False,
            "image_input": True,
        }
        monkeypatch.setenv("MODEL_VIDEO", "gemini:configured-video")
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
        capabilities = await client.get("/api/capabilities")
        assert capabilities.json() == {
            "max_tier": True,
            "video": True,
            "document": True,
            "image_input": True,
        }
