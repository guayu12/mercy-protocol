import asyncio
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from core import usage_stats
from core.agent import experts


@pytest.fixture(autouse=True)
def reset_usage_stats(monkeypatch):
    monkeypatch.setattr(usage_stats, "_since", time.time())
    monkeypatch.setattr(usage_stats, "_model_stats", {})


def _record(model, *, output=0, reasoning=0, generation=None, ttft=None, outcome="success"):
    usage_stats.record(
        "conversation",
        "openai",
        model,
        input_tokens=None,
        output_tokens=output,
        reasoning_tokens=reasoning,
        latency_ms=100,
        ttft_ms=ttft,
        generation_ms=generation,
        outcome=outcome,
    )


def test_usage_stats_aggregate_speed_excluding_reasoning_and_short_calls():
    _record("chat", output=50, reasoning=20, generation=1000, ttft=10)
    _record("chat", output=20, reasoning=0, generation=500, ttft=30)
    _record("chat", output=19, generation=900)
    _record("chat", output=30, generation=400)
    row = usage_stats.snapshot()["models"][0]

    assert row["calls"] == 4
    assert row["input_tokens"] == 0
    assert row["output_tokens"] == 119
    assert row["reasoning_tokens"] == 20
    assert row["avg_ttft_ms"] == 20
    assert row["avg_tokens_per_sec"] == pytest.approx(50 / 1.5)
    assert row["last_tokens_per_sec"] is None


def test_usage_stats_keys_provider_and_model_and_tracks_outcomes():
    _record("primary")
    usage_stats.record(
        "conversation",
        "openai",
        "primary",
        input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        latency_ms=10,
        outcome="failed",
    )
    usage_stats.record(
        "conversation",
        "openai",
        "primary",
        input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        latency_ms=10,
        outcome="cancelled",
    )
    usage_stats.record(
        "conversation",
        "deepseek",
        "primary",
        input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        latency_ms=10,
    )
    _record("fallback")
    rows = usage_stats.snapshot()["models"]

    assert {
        (row["role"], row["provider"], row["model"])
        for row in rows
    } == {
        ("conversation", "openai", "primary"),
        ("conversation", "openai", "fallback"),
        ("conversation", "deepseek", "primary"),
    }
    primary = next(row for row in rows if row["model"] == "primary" and row["provider"] == "openai")
    assert primary["calls"] == 1
    assert primary["failed_calls"] == 1
    assert primary["cancelled_calls"] == 1


def test_usage_stats_speed_and_ttft_are_null_without_qualifying_calls():
    _record("slow", output=19, generation=500)
    row = usage_stats.snapshot()["models"][0]
    assert row["avg_tokens_per_sec"] is None
    assert row["last_tokens_per_sec"] is None
    assert row["avg_ttft_ms"] is None
    assert row["last_ttft_ms"] is None


@pytest.mark.asyncio
async def test_usage_callback_records_tool_call_ttft_and_fallback_model():
    primary = usage_stats.UsageCallback("conversation", "openai", "primary")
    primary_id = uuid4()
    await primary.on_chat_model_start({}, [[]], run_id=primary_id)
    await primary.on_llm_error(RuntimeError("failed"), run_id=primary_id)

    fallback = usage_stats.UsageCallback("conversation", "openai", "fallback")
    fallback_id = uuid4()
    await fallback.on_chat_model_start({}, [[]], run_id=fallback_id)
    await fallback.on_llm_new_token(
        "",
        run_id=fallback_id,
        chunk=SimpleNamespace(
            message=SimpleNamespace(content="", tool_call_chunks=[{"name": "tool"}])
        ),
    )
    response = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        usage_metadata={
                            "input_tokens": 12,
                            "output_tokens": 8,
                            "total_tokens": 20,
                            "output_token_details": {"reasoning": 3},
                        },
                    )
                )
            ]
        ]
    )
    await fallback.on_llm_end(response, run_id=fallback_id)

    rows = {row["model"]: row for row in usage_stats.snapshot()["models"]}
    assert rows["primary"]["calls"] == 0
    assert rows["primary"]["failed_calls"] == 1
    assert rows["fallback"]["calls"] == 1
    assert rows["fallback"]["provider"] == "openai"
    assert rows["fallback"]["input_tokens"] == 12
    assert rows["fallback"]["output_tokens"] == 8
    assert rows["fallback"]["reasoning_tokens"] == 3
    assert rows["fallback"]["last_ttft_ms"] is not None


@pytest.mark.asyncio
async def test_usage_callback_marks_cancelled_runs():
    callback = usage_stats.UsageCallback("report", "openai", "report-model")
    run_id = uuid4()
    await callback.on_chat_model_start({}, [[]], run_id=run_id)
    await callback.on_llm_error(asyncio.CancelledError(), run_id=run_id)
    row = usage_stats.snapshot()["models"][0]
    assert row["calls"] == 0
    assert row["failed_calls"] == 0
    assert row["cancelled_calls"] == 1


def test_get_llm_attaches_role_and_model_usage_callback(monkeypatch):
    from core import models

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("MODEL_RESEARCH_MAX", "openai:deep-research")
    monkeypatch.setenv("MODEL_RESEARCH_MAX_FALLBACK", "openai:deep-research-fallback")
    llm = models.get_llm("research", tier="max", fallback=True)
    callback = llm.callbacks[0]
    assert isinstance(callback, usage_stats.UsageCallback)
    assert (callback.role, callback.provider, callback.model) == (
        "research_max",
        "openai",
        "deep-research-fallback",
    )


def test_gemini_analysis_records_thought_tokens():
    experts._record_gemini_analysis(
        "document",
        "gemini-model",
        time.time() - 0.1,
        {
            "total_input_tokens": 40,
            "total_output_tokens": 164,
            "total_thought_tokens": 459,
        },
        pages=(1, 4),
    )

    row = usage_stats.snapshot()["models"][0]
    assert row["role"] == "document"
    assert row["provider"] == "gemini"
    assert row["model"] == "gemini-model"
    assert row["input_tokens"] == 40
    assert row["output_tokens"] == 623
    assert row["reasoning_tokens"] == 459
    assert row["avg_ttft_ms"] is None
    assert row["avg_tokens_per_sec"] is None


@pytest.mark.parametrize(
    ("usage", "expected_output_tokens"),
    [
        ({"total_output_tokens": 164}, 164),
        ({"total_thought_tokens": 459}, 459),
        ({}, None),
    ],
)
def test_gemini_analysis_handles_missing_token_counts(
    monkeypatch, usage, expected_output_tokens
):
    recorded = {}
    monkeypatch.setattr(
        usage_stats,
        "record",
        lambda *args, **kwargs: recorded.update(kwargs),
    )

    experts._record_gemini_analysis(
        "document",
        "gemini-model",
        time.time(),
        usage,
    )

    assert recorded["output_tokens"] == expected_output_tokens


def test_graph_imports_without_tavily_key_and_search_is_unbound():
    environment = os.environ.copy()
    environment["TAVILY_API_KEY"] = ""
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from core.agent import graph, search; "
                "from core.runtime import current_snapshot; "
                "assert 'tavily_search' not in {tool.name for tool in graph.tools_for(current_snapshot())}; "
                "assert search.web_search is None"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.asyncio
async def test_stats_endpoint_returns_usage_snapshot(monkeypatch):
    import main

    expected = {"since": 123.0, "models": []}
    monkeypatch.setattr(main.usage_stats, "snapshot", lambda: expected)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.get("/api/stats")
    assert response.status_code == 200
    assert response.json() == expected


@pytest.mark.asyncio
async def test_config_key_hints_roles_and_secret_exclusion(monkeypatch):
    import main

    env_values = {
        "OPENAI_API_KEY": "openai-secret-value-1234",
        "GEMINI_API_KEY": "gemini-secret-value-5678",
        "DEEPSEEK_API_KEY": "short",
        "TAVILY_API_KEY": "tavily-secret-value-9012",
        "MODEL_CONVERSATION": "openai:chat-model@none",
        "MODEL_REPORT": "openai:report-model@medium",
        "MODEL_RESEARCH": "openai:research-model@high",
        "MODEL_RESEARCH_MAX": "openai:deep-research-model@high",
        "MODEL_VISION": "openai:vision-model",
        "MODEL_VIDEO": "gemini:gemini-model",
        "TTS_VOICE": "zh-CN-TestVoice",
    }
    for name, value in env_values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MERCY_SYSTEM_KEYS", "[]")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.get("/api/config")

    config = response.json()
    assert response.status_code == 200
    assert config["profile"] == "完整模式"
    assert [row["role"] for row in config["roles"]] == [
        "conversation",
        "report",
        "research",
        "research_max",
        "vision",
        "video",
        "memory",
    ]
    assert all(row["available"] and row["reason"] is None for row in config["roles"])
    assert config["keys"]["openai"] == {
        "configured": True,
        "hint": "••••1234",
        "variable": "OPENAI_API_KEY",
        "source": "env_file",
    }
    assert config["keys"]["gemini"] == {
        "configured": True,
        "hint": "••••5678",
        "variable": "GEMINI_API_KEY",
        "source": "env_file",
    }
    assert config["keys"]["deepseek"] == {
        "configured": True,
        "hint": None,
        "variable": "DEEPSEEK_API_KEY",
        "source": "env_file",
    }
    assert config["keys"]["tavily"] == {
        "configured": True,
        "hint": "••••9012",
        "variable": "TAVILY_API_KEY",
        "source": "env_file",
    }
    assert config["services"][0]["reason"] is None
    assert config["services"][2]["detail"] == "zh-CN-TestVoice"
    for secret in (
        env_values["OPENAI_API_KEY"],
        env_values["GEMINI_API_KEY"],
        env_values["DEEPSEEK_API_KEY"],
        env_values["TAVILY_API_KEY"],
    ):
        assert secret not in response.text


@pytest.mark.asyncio
async def test_config_reports_missing_keys_and_capabilities_share_config(monkeypatch):
    import main

    for name in (
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "TAVILY_API_KEY",
        "MODEL_CONVERSATION",
        "MODEL_REPORT",
        "MODEL_RESEARCH",
        "MODEL_RESEARCH_MAX",
        "MODEL_VISION",
        "MODEL_VIDEO",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODEL_VIDEO", "gemini:configured-video")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        config_response = await client.get("/api/config")
        capabilities_response = await client.get("/api/capabilities")

    config = config_response.json()
    video = next(row for row in config["roles"] if row["role"] == "video")
    assert video["available"] is False
    assert video["reason"] == "缺少 GEMINI_API_KEY"
    research_max = next(row for row in config["roles"] if row["role"] == "research_max")
    assert research_max["reason"] == "未配置 MODEL_RESEARCH_MAX"
    assert config["services"][0]["reason"] == "未配置 Tavily key"
    assert config["services"][1]["reason"] == "缺少 OPENAI_API_KEY"
    assert capabilities_response.json() == {
        "max_tier": False,
        "video": False,
        "document": False,
        "image_input": True,
    }
