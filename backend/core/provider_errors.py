import asyncio
from dataclasses import dataclass

import httpx
import openai

from core.models import ModelConfigError


@dataclass(frozen=True)
class ProviderError:
    category: str
    message: str


def _status(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _code(error: Exception) -> str | None:
    value = getattr(error, "code", None)
    if isinstance(value, str):
        return value
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        nested = body.get("error")
        if isinstance(nested, dict):
            value = nested.get("code")
        if value is None:
            value = body.get("code")
    return value if isinstance(value, str) else None


def classify(error: Exception) -> ProviderError:
    status = _status(error)
    code = _code(error)
    if status == 400:
        return ProviderError("bad_request", "模型不接受这个设置")
    if status == 401:
        return ProviderError("invalid_key", "key 无效")
    if status in {403, 404} or code == "model_not_found":
        return ProviderError("model_access", "无权使用该模型")
    if status == 402 or (status == 429 and code == "insufficient_quota"):
        return ProviderError("quota", "额度不足")
    if status == 429:
        return ProviderError("rate_limit", "请求太频繁")
    if isinstance(error, ModelConfigError):
        return ProviderError("configuration", "模型未配置")
    if isinstance(
        error,
        (
            openai.APIConnectionError,
            openai.APITimeoutError,
            httpx.ConnectError,
            httpx.TimeoutException,
            asyncio.TimeoutError,
        ),
    ):
        return ProviderError("network", "网络失败")
    return ProviderError("provider", "模型服务出错")
