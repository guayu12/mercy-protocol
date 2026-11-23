"""In-memory usage aggregates for model calls."""

import asyncio
import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler


_since = time.time()
_lock = threading.Lock()
_model_stats: dict[tuple[str, str, str], dict] = {}


def _token_count(value: int | None) -> int:
    return value if type(value) is int and value >= 0 else 0


def record(
    role: str,
    provider: str,
    model: str,
    *,
    input_tokens: int | None,
    output_tokens: int | None,
    reasoning_tokens: int | None,
    latency_ms: float,
    ttft_ms: float | None = None,
    generation_ms: float | None = None,
    outcome: str = "success",
) -> None:
    input_count = _token_count(input_tokens)
    output_count = _token_count(output_tokens)
    reasoning_count = _token_count(reasoning_tokens)
    visible_tokens = max(output_count - reasoning_count, 0)
    generation_seconds = (
        generation_ms / 1000
        if generation_ms is not None and generation_ms >= 500
        else None
    )
    speed = (
        visible_tokens / generation_seconds
        if outcome == "success"
        and generation_seconds is not None
        and visible_tokens >= 20
        else None
    )

    with _lock:
        key = (role, provider, model)
        stats = _model_stats.get(key)
        if stats is None:
            stats = {
                "role": role,
                "provider": provider,
                "model": model,
                "calls": 0,
                "failed_calls": 0,
                "cancelled_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "_latency_total": 0.0,
                "_latency_calls": 0,
                "_ttft_total": 0.0,
                "_ttft_calls": 0,
                "_speed_tokens": 0,
                "_speed_seconds": 0.0,
            }
            _model_stats[key] = stats

        if outcome == "cancelled":
            stats["cancelled_calls"] += 1
        elif outcome == "failed":
            stats["failed_calls"] += 1
        else:
            stats["calls"] += 1
        stats["input_tokens"] += input_count
        stats["output_tokens"] += output_count
        stats["reasoning_tokens"] += reasoning_count
        stats["_latency_total"] += latency_ms
        stats["_latency_calls"] += 1
        stats["last_latency_ms"] = latency_ms
        if ttft_ms is not None:
            stats["_ttft_total"] += ttft_ms
            stats["_ttft_calls"] += 1
        stats["last_ttft_ms"] = ttft_ms
        stats["last_tokens_per_sec"] = speed
        if speed is not None and outcome == "success":
            stats["_speed_tokens"] += visible_tokens
            stats["_speed_seconds"] += generation_seconds


def _usage_from_response(response: object) -> dict:
    direct_usage = getattr(response, "usage_metadata", None)
    if isinstance(direct_usage, dict):
        return direct_usage
    for generation_group in getattr(response, "generations", []) or []:
        if not isinstance(generation_group, list):
            generation_group = [generation_group]
        for generation in generation_group:
            usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if isinstance(usage, dict):
                return usage
    return {}


def _visible_chunk(chunk: object, token: object) -> bool:
    message = getattr(chunk, "message", None)
    if message is None:
        return bool(token)
    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        return True
    if isinstance(content, list) and any(
        isinstance(block, str) and block
        or isinstance(block, dict)
        and block.get("type") in {"text", "text_delta"}
        and bool(block.get("text"))
        for block in content
    ):
        return True
    return bool(getattr(message, "tool_call_chunks", None))


class UsageCallback(AsyncCallbackHandler):
    def __init__(self, role: str, provider: str, model: str):
        super().__init__()
        self.role = role
        self.provider = provider
        self.model = model
        self._runs: dict[UUID, dict[str, float | None]] = {}
        self._runs_lock = threading.Lock()

    def _start(self, run_id: UUID) -> None:
        with self._runs_lock:
            self._runs.setdefault(
                run_id,
                {"started_at": time.perf_counter(), "first_output_at": None},
            )

    def _finish(
        self,
        run_id: UUID,
        outcome: str,
        response: object = None,
    ) -> None:
        finished_at = time.perf_counter()
        with self._runs_lock:
            state = self._runs.pop(
                run_id,
                {"started_at": finished_at, "first_output_at": None},
            )
        started_at = state["started_at"] or finished_at
        first_output_at = state["first_output_at"]
        usage = _usage_from_response(response)
        output_details = usage.get("output_token_details") or {}
        record(
            self.role,
            self.provider,
            self.model,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            reasoning_tokens=output_details.get("reasoning"),
            latency_ms=(finished_at - started_at) * 1000,
            ttft_ms=(
                (first_output_at - started_at) * 1000
                if first_output_at is not None
                else None
            ),
            generation_ms=(
                (finished_at - first_output_at) * 1000
                if first_output_at is not None
                else None
            ),
            outcome=outcome,
        )

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            self._start(run_id)
        except BaseException:
            pass

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            self._start(run_id)
        except BaseException:
            pass

    async def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        run_id: UUID,
        chunk: object = None,
        **kwargs: Any,
    ) -> None:
        try:
            if not _visible_chunk(chunk, token):
                return
            with self._runs_lock:
                state = self._runs.setdefault(
                    run_id,
                    {"started_at": time.perf_counter(), "first_output_at": None},
                )
                if state["first_output_at"] is None:
                    state["first_output_at"] = time.perf_counter()
        except BaseException:
            pass

    async def on_llm_end(
        self,
        response: object,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            self._finish(run_id, "success", response)
        except BaseException:
            pass

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            outcome = (
                "cancelled"
                if isinstance(error, asyncio.CancelledError)
                else "failed"
            )
            self._finish(run_id, outcome, kwargs.get("response"))
        except BaseException:
            pass


def snapshot() -> dict:
    with _lock:
        models = []
        for stats in _model_stats.values():
            calls = stats["calls"]
            speed_seconds = stats["_speed_seconds"]
            models.append(
                {
                    "role": stats["role"],
                    "provider": stats["provider"],
                    "model": stats["model"],
                    "calls": calls,
                    "failed_calls": stats["failed_calls"],
                    "cancelled_calls": stats["cancelled_calls"],
                    "input_tokens": stats["input_tokens"],
                    "output_tokens": stats["output_tokens"],
                    "reasoning_tokens": stats["reasoning_tokens"],
                    "avg_latency_ms": (
                        stats["_latency_total"] / stats["_latency_calls"]
                        if stats["_latency_calls"]
                        else None
                    ),
                    "last_latency_ms": stats.get("last_latency_ms"),
                    "avg_ttft_ms": (
                        stats["_ttft_total"] / stats["_ttft_calls"]
                        if stats["_ttft_calls"]
                        else None
                    ),
                    "last_ttft_ms": stats.get("last_ttft_ms"),
                    "avg_tokens_per_sec": (
                        stats["_speed_tokens"] / speed_seconds
                        if speed_seconds
                        else None
                    ),
                    "last_tokens_per_sec": stats.get("last_tokens_per_sec"),
                }
            )
    models.sort(key=lambda item: (item["role"], item["provider"], item["model"]))
    return {"since": _since, "models": models}
