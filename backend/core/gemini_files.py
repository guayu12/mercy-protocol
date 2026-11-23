import asyncio
import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx

from core.models import provider_key
from core.provider_errors import is_api_key_invalid_body


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

API_BASE = "https://generativelanguage.googleapis.com"
UPLOAD_START_URL = f"{API_BASE}/upload/v1beta/files"
INTERACTIONS_URL = f"{API_BASE}/v1beta/interactions"
UPLOAD_CHUNK_SIZE = 1024 * 1024
PROCESSING_POLL_INTERVAL = 2
PROCESSING_TIMEOUT = 180
HTTP_TIMEOUT = httpx.Timeout(connect=20, read=120, write=120, pool=20)

_analyze_locks: dict[str, asyncio.Lock] = {}
_analyze_lock_refs: dict[str, int] = {}


class GeminiError(Exception):
    def __init__(self, *, status_code: int | None = None, body=None):
        self.status_code = status_code
        self.body = body
        super().__init__("Gemini request failed.")


class GeminiQuotaExceeded(GeminiError):
    def __init__(self, kind: str = "unknown"):
        self.kind = kind if kind in {"minute", "day", "unknown"} else "unknown"
        Exception.__init__(self, "Gemini quota exceeded.")


class GeminiFileUnavailable(GeminiError):
    def __init__(self):
        super().__init__()


class GeminiRejected(GeminiError):
    def __init__(self):
        self.status_code = 400
        self.body = None
        Exception.__init__(
            self,
            "Gemini rejected the supplied document.",
        )


@dataclass(frozen=True)
class GeminiFile:
    name: str
    uri: str
    mime: str
    duration_seconds: float | None


@dataclass(frozen=True)
class GeminiAnswer:
    text: str
    status: str
    usage: dict


@asynccontextmanager
async def analyze_lock(media_id: str):
    lock = _analyze_locks.get(media_id)
    if lock is None:
        lock = asyncio.Lock()
        _analyze_locks[media_id] = lock
        _analyze_lock_refs[media_id] = 0
    _analyze_lock_refs[media_id] += 1
    try:
        async with lock:
            yield
    finally:
        remaining = _analyze_lock_refs[media_id] - 1
        if remaining == 0:
            _analyze_lock_refs.pop(media_id)
            _analyze_locks.pop(media_id)
        else:
            _analyze_lock_refs[media_id] = remaining


@asynccontextmanager
async def _client_context(client: httpx.AsyncClient | None):
    if client is not None:
        yield client
        return
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as owned_client:
        yield owned_client


def _api_key() -> str:
    key = provider_key("gemini")
    if key is None:
        raise GeminiError() from None
    return key[0]


def _quota_kind(response: httpx.Response) -> str:
    try:
        error = response.json().get("error", {})
        details = error.get("details", [])
    except (ValueError, AttributeError, TypeError):
        return "unknown"

    quota_values = []
    for detail in details if isinstance(details, list) else []:
        if not isinstance(detail, dict) or not str(
            detail.get("@type", "")
        ).endswith("google.rpc.QuotaFailure"):
            continue
        violations = detail.get("violations", [])
        for violation in violations if isinstance(violations, list) else []:
            if isinstance(violation, dict):
                quota_values.extend(
                    str(violation.get(field, ""))
                    for field in ("quotaId", "quotaMetric")
                )
    combined = " ".join(quota_values).casefold()
    if "perday" in combined:
        return "day"
    if "perminute" in combined:
        return "minute"
    return "unknown"


async def _request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs,
) -> httpx.Response:
    try:
        return await client.request(method, url, **kwargs)
    except asyncio.CancelledError:
        raise
    except httpx.HTTPStatusError as error:
        response = error.response
        try:
            body = response.json()
        except Exception:
            body = None
        raise GeminiError(
            status_code=response.status_code,
            body=body,
        ) from None
    except Exception:
        raise GeminiError() from None


def _check_response(
    response: httpx.Response,
    *,
    file_request: bool = False,
    reject_400: bool = False,
) -> None:
    if response.status_code == 429:
        raise GeminiQuotaExceeded(_quota_kind(response)) from None
    if response.status_code == 400:
        try:
            body = response.json()
        except Exception:
            body = None
        if is_api_key_invalid_body(body):
            raise GeminiError(status_code=400, body=body) from None
    if reject_400 and response.status_code == 400:
        raise GeminiRejected() from None
    if file_request and response.status_code in {403, 404}:
        raise GeminiFileUnavailable() from None
    if response.status_code == 400:
        raise GeminiError(status_code=400, body=body) from None
    if not 200 <= response.status_code < 300:
        raise GeminiError() from None


def _json(response: httpx.Response) -> dict:
    try:
        value = response.json()
    except (ValueError, json.JSONDecodeError):
        raise GeminiError() from None
    if not isinstance(value, dict):
        raise GeminiError() from None
    return value


async def _file_chunks(path: str | Path):
    file_obj = await asyncio.to_thread(Path(path).open, "rb")
    try:
        while True:
            chunk = await asyncio.to_thread(file_obj.read, UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            yield chunk
    finally:
        await asyncio.to_thread(file_obj.close)


def _video_duration(file_data: dict) -> float | None:
    metadata = file_data.get("videoMetadata")
    value = metadata.get("videoDuration") if isinstance(metadata, dict) else None
    if not isinstance(value, str) or not value.endswith("s"):
        return None
    try:
        duration = float(value[:-1])
    except ValueError:
        return None
    return duration if math.isfinite(duration) and duration >= 0 else None


def _file_url(name: str) -> str:
    return f"{API_BASE}/v1beta/{quote(name, safe='/')}"


async def upload_file(
    path: str | Path,
    mime: str,
    display_name: str,
    *,
    client: httpx.AsyncClient | None = None,
    poll_interval: float = PROCESSING_POLL_INTERVAL,
    poll_timeout: float = PROCESSING_TIMEOUT,
) -> GeminiFile:
    file_name = None
    async with _client_context(client) as request_client:
        try:
            api_key = _api_key()
            size = await asyncio.to_thread(os.path.getsize, path)
            start_response = await _request(
                request_client,
                "POST",
                UPLOAD_START_URL,
                headers={
                    "x-goog-api-key": api_key,
                    "X-Goog-Upload-Protocol": "resumable",
                    "X-Goog-Upload-Command": "start",
                    "X-Goog-Upload-Header-Content-Length": str(size),
                    "X-Goog-Upload-Header-Content-Type": mime,
                    "Content-Type": "application/json",
                },
                json={"file": {"display_name": display_name}},
            )
            _check_response(start_response)
            upload_url = start_response.headers.get("x-goog-upload-url")
            parsed_upload_url = urlsplit(upload_url or "")
            if (
                parsed_upload_url.scheme != "https"
                or parsed_upload_url.hostname != "generativelanguage.googleapis.com"
            ):
                raise GeminiError() from None

            upload_response = await _request(
                request_client,
                "POST",
                upload_url,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Length": str(size),
                    "X-Goog-Upload-Offset": "0",
                    "X-Goog-Upload-Command": "upload, finalize",
                },
                content=_file_chunks(path),
            )
            _check_response(upload_response)
            uploaded = _json(upload_response).get("file")
            if not isinstance(uploaded, dict):
                raise GeminiError() from None
            file_name = uploaded.get("name")
            file_uri = uploaded.get("uri")
            if not isinstance(file_name, str) or not isinstance(file_uri, str):
                raise GeminiError() from None

            deadline = time.monotonic() + poll_timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GeminiError() from None
                try:
                    poll_response = await asyncio.wait_for(
                        _request(
                            request_client,
                            "GET",
                            _file_url(file_name),
                            headers={"x-goog-api-key": api_key},
                        ),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    raise GeminiError() from None
                _check_response(poll_response, file_request=True)
                polled = _json(poll_response)
                state = polled.get("state")
                if state == "ACTIVE":
                    return GeminiFile(
                        name=file_name,
                        uri=file_uri,
                        mime=polled.get("mimeType") or uploaded.get("mimeType") or mime,
                        duration_seconds=_video_duration(polled),
                    )
                if state != "PROCESSING":
                    raise GeminiError() from None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GeminiError() from None
                await asyncio.sleep(min(poll_interval, remaining))
        except BaseException as error:
            if file_name:
                await asyncio.shield(delete_file(file_name, client=request_client))
            if not isinstance(error, Exception) or isinstance(error, GeminiError):
                raise
            raise GeminiError() from None


async def ask(
    file: GeminiFile,
    prompt: str,
    model: str,
    *,
    client: httpx.AsyncClient | None = None,
    processing: dict | None = None,
) -> GeminiAnswer:
    async with _client_context(client) as request_client:
        api_key = _api_key()
        input_type = "video" if file.mime.lower().startswith("video/") else "document"
        media_input = {
            "type": input_type,
            "uri": file.uri,
            "mime_type": file.mime,
        }
        if processing is not None:
            media_input["processing"] = processing
        response = await _request(
            request_client,
            "POST",
            INTERACTIONS_URL,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json={
                "model": model,
                "store": False,
                "input": [
                    media_input,
                    {"type": "text", "text": prompt},
                ],
            },
        )
        _check_response(
            response,
            file_request=True,
            reject_400=processing is not None,
        )
        data = _json(response)
        steps = data.get("steps", [])
        output_text = []
        for step in steps if isinstance(steps, list) else []:
            if not isinstance(step, dict) or step.get("type") != "model_output":
                continue
            content = step.get("content", [])
            if isinstance(content, list):
                output_text.extend(
                    item["text"]
                    for item in content
                    if isinstance(item, dict) and isinstance(item.get("text"), str)
                )
            elif isinstance(content, dict) and isinstance(content.get("text"), str):
                output_text.append(content["text"])
        usage_data = data.get("usage")
        usage_data = usage_data if isinstance(usage_data, dict) else {}
        usage = {
            "total_tokens": usage_data.get("total_tokens"),
            "total_input_tokens": usage_data.get("total_input_tokens"),
            "total_output_tokens": usage_data.get("total_output_tokens"),
            "total_thought_tokens": usage_data.get("total_thought_tokens"),
            "input_tokens_by_modality": usage_data.get(
                "input_tokens_by_modality", []
            ),
        }
        status = data.get("status")
        return GeminiAnswer(
            text="\n".join(output_text),
            status=status if isinstance(status, str) else "unknown",
            usage=usage,
        )


async def delete_file(
    name: str, *, client: httpx.AsyncClient | None = None
) -> bool:
    try:
        async with _client_context(client) as request_client:
            response = await _request(
                request_client,
                "DELETE",
                _file_url(name),
                headers={"x-goog-api-key": _api_key()},
            )
            if 200 <= response.status_code < 300 or response.status_code == 404:
                return True
            if response.status_code == 429:
                kind = _quota_kind(response)
                logging.getLogger(__name__).warning(
                    "Gemini file cleanup failed (quota: %s)", kind
                )
            else:
                logging.getLogger(__name__).warning("Gemini file cleanup failed")
    except asyncio.CancelledError:
        raise
    except Exception:
        logging.getLogger(__name__).warning("Gemini file cleanup failed")
    return False
