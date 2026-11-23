import os
import re
import sys
import asyncio
import ipaddress
import json
import logging
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from typing import Literal, Optional
from core.env import load_env

load_env()

from core import log_bus  # real-time log broadcaster
from core import config_service
from core.config_service import (
    ActiveRequest,
    ConfigApiError,
    KeyRequest,
    ModelsRequest,
    ProfilesRequest,
)
from core.agent.graph import (
    close_graph,
    init_graph,
    process_query,
)
from core import media_store, session_store, speech_queue, usage_stats
from core.gemini_file_cache import cleanup_gemini_files, gemini_file_cleanup_loop
from core.provider_errors import classify as classify_provider_error
from core.runtime import load_snapshot
from core.runtime_config import current_config
from core.safe_fetch import DownloadTooLarge, SafeFetchError, safe_media_fetcher
from core.speech import synthesize
from core.speakable import to_speakable

@asynccontextmanager
async def lifespan(app: FastAPI):
    store_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "session.db"
    )
    await session_store.init_store(store_path)
    await media_store.init_store(store_path)
    cleanup_task = asyncio.create_task(gemini_file_cleanup_loop())
    try:
        await init_graph()
        yield
    finally:
        cleanup_task.cancel()
        await asyncio.gather(cleanup_task, return_exceptions=True)
        await close_graph()
        try:
            await asyncio.wait_for(
                cleanup_gemini_files(delete_all=True),
                timeout=GEMINI_FILE_SHUTDOWN_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logging.getLogger(__name__).warning(
                "Gemini file shutdown cleanup timed out"
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "Gemini file shutdown cleanup failed"
            )
        await media_store.close_store()
        await session_store.close_store()

app = FastAPI(title="Mercy Protocol — AI Judicial Interface", version="2.0.0", lifespan=lifespan)

UPLOAD_MULTIPART_OVERHEAD = 1024 * 1024
GEMINI_FILE_SHUTDOWN_TIMEOUT_SECONDS = 10


@app.middleware("http")
async def reject_oversized_upload_request(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/api/upload":
        try:
            content_length = int(request.headers.get("content-length", "0"))
        except ValueError:
            content_length = 0
        if content_length > media_store.VIDEO_LIMIT + UPLOAD_MULTIPART_OVERHEAD:
            return JSONResponse(
                status_code=413,
                content={"detail": "Upload exceeds size limit"},
            )
    return await call_next(request)


DEFAULT_CORS_ORIGINS = [
    "http://localhost:1420",   # Vite dev server (tauri dev)
    "http://127.0.0.1:1420",
    "http://localhost:5173",   # Vite default port
    "http://127.0.0.1:5173",
    "tauri://localhost",       # Tauri on macOS / Linux
    "http://tauri.localhost",  # Tauri v2 on Windows (Surface)
    "https://tauri.localhost",
]


def _effective_cors_origins() -> list[str]:
    configured = os.environ.get("CORS_ORIGINS")
    if not configured:
        return DEFAULT_CORS_ORIGINS
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


CORS_ORIGINS = _effective_cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Range"],
    expose_headers=["Accept-Ranges", "Content-Range", "Content-Length", "X-Content-Type-Options"],
)


def _loopback_host(value: str | None) -> bool:
    if not value:
        return False
    value = value.strip().strip("[]")
    if value.lower() == "localhost":
        return True
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_loopback


def _allowed_config_host(value: str | None) -> bool:
    if not value:
        return False
    value = value.strip().lower()
    if value.startswith("[::1]"):
        hostname, suffix = "[::1]", value[5:]
        if suffix and (
            not suffix.startswith(":")
            or not suffix[1:].isdigit()
            or not 0 <= int(suffix[1:]) <= 65535
        ):
            return False
    else:
        hostname, separator, port = value.partition(":")
        suffix = f":{port}" if separator else ""
        if separator and (not port.isdigit() or not 0 <= int(port) <= 65535):
            return False
    return hostname in {"127.0.0.1", "localhost", "[::1]"} and (
        not suffix or suffix.startswith(":")
    )


def _config_client_is_local(request: Request) -> bool:
    backend_host = os.environ.get("BACKEND_HOST", "127.0.0.1")
    client_host = request.client.host if request.client else None
    return _loopback_host(backend_host) and _loopback_host(client_host)


async def _require_local_config_request(request: Request, schema):
    if (
        not _config_client_is_local(request)
        or not _allowed_config_host(request.headers.get("host"))
        or request.headers.get("origin") not in CORS_ORIGINS
        or "*" in CORS_ORIGINS
    ):
        raise ConfigApiError(403, "forbidden", "只能在本机修改配置")
    content_type = request.headers.get("content-type", "")
    if content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise ConfigApiError(415, "unsupported_media_type", "仅支持 JSON 请求")
    try:
        content_length = int(request.headers.get("content-length", "0"))
    except ValueError:
        raise ConfigApiError(400, "invalid_request", "请求格式不正确") from None
    if content_length < 0:
        raise ConfigApiError(400, "invalid_request", "请求格式不正确")
    if content_length > 65536:
        raise ConfigApiError(413, "payload_too_large", "请求内容过大")
    body = await request.body()
    if len(body) > 65536:
        raise ConfigApiError(413, "payload_too_large", "请求内容过大")
    return config_service.parse_payload(body, schema)


@app.exception_handler(ConfigApiError)
async def _config_api_error_handler(_request: Request, error: ConfigApiError):
    return JSONResponse(status_code=error.status_code, content=error.payload)

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

class ScreenState(BaseModel):
    viewing_report_id: Optional[str] = None
    stage_media_ids: list[str] = Field(default_factory=list)
    stage_current_media_id: Optional[str] = None


class Interrupted(BaseModel):
    turn_id: str
    spoken_sentence_ids: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    model_tier: Literal["normal", "max"] = "normal"
    image_base64: Optional[str] = None  # raw base64 or full data: URL
    thread_id: str = "default_user"
    attachments: list[str] = Field(default_factory=list)
    screen_state: Optional[ScreenState] = None
    interrupted: Optional[Interrupted] = None


def _require_allowed_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin is None:
        return
    if origin not in CORS_ORIGINS:
        raise HTTPException(status_code=403, detail="Origin not allowed")


@app.get("/")
async def root():
    return {"status": "ok", "message": "Mercy Protocol Backend Running"}


@app.get("/api/capabilities")
async def capabilities():
    runtime = await load_snapshot()
    return current_config(runtime)["capabilities"]


@app.get("/api/stats")
async def stats_endpoint():
    return usage_stats.snapshot()


@app.get("/api/config")
async def config_endpoint(request: Request):
    runtime = await load_snapshot()
    config = config_service.config_view(
        runtime, editable=_config_client_is_local(request)
    )
    config.pop("capabilities")
    return config


@app.post("/api/config/models")
async def config_models_endpoint(request: Request):
    payload = await _require_local_config_request(request, ModelsRequest)
    return await config_service.discover_models(payload)


@app.post("/api/config/key")
async def config_key_endpoint(request: Request):
    payload = await _require_local_config_request(request, KeyRequest)
    return await config_service.save_key(payload)


@app.post("/api/config/profiles")
async def config_profiles_endpoint(request: Request):
    payload = await _require_local_config_request(request, ProfilesRequest)
    return await config_service.save_profile_configuration(
        payload, editable=_config_client_is_local(request)
    )


@app.post("/api/config/active")
async def config_active_endpoint(request: Request):
    payload = await _require_local_config_request(request, ActiveRequest)
    return await config_service.switch_active_profile(
        payload, editable=_config_client_is_local(request)
    )


# ─── SSE Log Stream ──────────────────────────────────────────────────
@app.get("/api/logs")
async def logs_sse(request: Request):
    """
    Server-Sent Events endpoint that streams real-time backend log events
    to the frontend sidebar. Each event is a JSON-serialised log_bus payload.
    """
    queue = log_bus.subscribe()
    log_bus.emit("flow", "FLOW", "SSE client connected — log stream open")

    async def event_stream():
        try:
            # Wait for events (no polling); send a heartbeat every 15 s to keep the connection alive
            while not await request.is_disconnected():
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            log_bus.unsubscribe(queue)
            log_bus.emit("flow", "FLOW", "SSE client disconnected")

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)

# ─── Chat (SSE, speech and report events) ────────────────────────────
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Streams spoken sentences, reports, status and tool events as Server-Sent Events.
    """
    screen_state = {}
    if request.screen_state is not None:
        screen_state = request.screen_state.model_dump()
        await media_store.reconcile_stage(
            request.thread_id, screen_state.get("stage_media_ids", [])
        )
    await _register_message_links(request.thread_id, request.message)

    async def event_stream():
        try:
            async for event in process_query(
                request.message,
                request.thread_id,
                request.image_base64,
                screen_state,
                request.interrupted.model_dump() if request.interrupted else None,
                request.attachments,
                request.model_tier,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as error:
            classified = classify_provider_error(error)
            status = getattr(error, "status_code", None)
            if status is None:
                status = getattr(
                    getattr(error, "response", None), "status_code", None
                )
            if not isinstance(status, int):
                status = None
            log_bus.emit(
                "error",
                "ERROR",
                f"chat_endpoint exception: {type(error).__name__} "
                f"HTTP {status} category={classified.category}",
            )
            error_event = {
                "type": "error",
                "message": f"模型调用失败：{classified.message}",
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)


_MEDIA_LINK = re.compile(r"https?://[^\s<]+", re.IGNORECASE)
_LINK_MIMES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}


def _trim_link_url(url: str) -> str:
    pairs = {")": "(", "]": "[", ">": "<"}
    punctuation = ".,!?;:，。！？）】"
    while url:
        trailing = url[-1]
        if trailing in punctuation:
            url = url[:-1]
        elif trailing in pairs and url.count(trailing) > url.count(pairs[trailing]):
            url = url[:-1]
        elif trailing in {"'", '"'} and url.count(trailing) % 2:
            url = url[:-1]
        else:
            break
    return url


async def _register_message_links(thread_id: str, message: str) -> None:
    for match in _MEDIA_LINK.finditer(message):
        url = _trim_link_url(match.group(0))
        try:
            parsed = urlsplit(url)
            extension = Path(parsed.path).suffix.lower()
            mime = _LINK_MIMES.get(extension)
            if parsed.scheme.lower() not in {"http", "https"} or not mime:
                continue
        except ValueError:
            continue
        await media_store.register_media(
            thread_id,
            "video" if mime.startswith("video/") else "image",
            "link",
            mime,
            Path(parsed.path).name or url,
            url=url,
        )


@app.post("/api/upload")
async def upload_media(
    request: Request, thread_id: str = Form(...), file: UploadFile = File(...)
):
    _require_allowed_origin(request)
    try:
        record = await media_store.register_upload(thread_id, file)
    except media_store.UnsupportedUpload as error:
        raise HTTPException(status_code=415, detail=str(error)) from error
    except media_store.UploadTooLarge as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    finally:
        await file.close()
    return {
        "media_id": record["media_id"],
        "kind": record["kind"],
        "mime": record["mime"],
    }


def _cache_limit_bytes() -> int:
    try:
        megabytes = max(0, int(os.environ.get("MEDIA_CACHE_MAX_MB", "500")))
    except ValueError:
        megabytes = 500
    return megabytes * 1024 * 1024


def _media_file_response(
    path: Path,
    mime: str,
    *,
    filename: str | None = None,
    disposition: Literal["inline", "attachment"] = "inline",
) -> FileResponse:
    return FileResponse(
        path,
        media_type=mime,
        headers={"X-Content-Type-Options": "nosniff"},
        filename=filename,
        content_disposition_type=disposition,
    )


@app.get("/api/media/{media_id}")
async def get_media(media_id: str, request: Request):
    record = await media_store.get_media(media_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Media not found")
    await media_store.update_last_used(media_id)

    if record["origin"] == "upload":
        path = Path(record["path"] or "").resolve()
        if path.parent != media_store.media_directory().resolve() or not path.is_file():
            raise HTTPException(status_code=404, detail="Uploaded media not found")
        if record["kind"] == "document":
            return _media_file_response(
                path,
                "application/pdf",
                filename=record["caption"],
                disposition="attachment",
            )
        return _media_file_response(path, record["mime"])

    if not record["url"]:
        raise HTTPException(status_code=404, detail="Remote media URL not found")
    if record["kind"] == "image":
        try:
            record = await media_store.cache_remote_image(
                media_id, safe_media_fetcher, _cache_limit_bytes()
            )
        except DownloadTooLarge as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except SafeFetchError as error:
            raise HTTPException(
                status_code=502, detail="Unable to fetch remote media"
            ) from error
        except ValueError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        path = Path(record["path"] or "")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Cached media not found")
        return _media_file_response(path, record["mime"])

    try:
        upstream = await safe_media_fetcher.open_stream(
            record["url"], request.headers.get("range"), media_store.VIDEO_LIMIT
        )
    except SafeFetchError as error:
        raise HTTPException(
            status_code=502, detail="Unable to fetch remote media"
        ) from error
    status_code = upstream.response.status_code
    mime = upstream.response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if status_code not in {200, 206, 416} or mime not in {"video/mp4", "video/webm"}:
        await upstream.aclose()
        raise HTTPException(status_code=502, detail="Remote media response is invalid")
    if mime != record["mime"]:
        await media_store.update_media_mime(media_id, mime)
    headers = {"X-Content-Type-Options": "nosniff"}
    for name in ("content-range", "content-length"):
        if name in upstream.response.headers:
            headers[name] = upstream.response.headers[name]
    headers["Accept-Ranges"] = upstream.response.headers.get("accept-ranges", "bytes")

    async def stream_body():
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(
        stream_body(),
        status_code=status_code,
        media_type=mime,
        headers=headers,
    )


class SpeechRequest(BaseModel):
    text: str


@app.post("/api/speech")
async def speech_endpoint(req: SpeechRequest):
    """Synthesize speech and return base64 audio plus a timed viseme track for avatar lip-sync."""
    text = to_speakable(req.text)
    if not text:
        raise HTTPException(status_code=400, detail="Speech text is empty after cleaning")
    return await synthesize(text)


@app.get("/api/speech/{sentence_id}")
async def sentence_speech_endpoint(sentence_id: str):
    """Synthesize a sentence previously emitted by the chat stream."""
    text = speech_queue.get_sentence(sentence_id)
    if text is None:
        raise HTTPException(status_code=404, detail="Speech sentence not found")
    return await synthesize(text)


@app.get("/api/session/{thread_id}")
async def session_endpoint(thread_id: str):
    """Return saved reports and recent spoken conversation for a thread."""
    stage = await media_store.get_stage(thread_id)
    return {
        "reports": await session_store.list_reports(thread_id),
        "utterances": await session_store.recent_utterances(thread_id),
        "stage": (
            {
                "caption": stage["caption"],
                "focus": stage["focus"],
                "items": stage["items"],
            }
            if stage
            else None
        ),
    }


@app.delete("/api/session/{thread_id}/stage", status_code=204)
async def delete_session_stage(thread_id: str, request: Request):
    _require_allowed_origin(request)
    await media_store.clear_stage(thread_id)
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.environ.get("BACKEND_HOST", "127.0.0.1"),
        port=8000,
        reload=True,
    )
