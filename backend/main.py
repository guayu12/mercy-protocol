import os
import re
import sys
import asyncio
import json
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from typing import Optional
from core import log_bus  # real-time log broadcaster
from core.agent.graph import process_query, init_graph, close_graph
from core import media_store, session_store, speech_queue
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
    await init_graph()
    yield
    await close_graph()
    await media_store.close_store()
    await session_store.close_store()

app = FastAPI(title="Mercy Protocol — AI Judicial Interface", version="2.0.0", lifespan=lifespan)

UPLOAD_MULTIPART_OVERHEAD = 1024 * 1024


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
DEFAULT_MEDIA_MUTATION_ORIGINS = {
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
}
_cors_env = os.environ.get("CORS_ORIGINS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_env.split(",") if o.strip()] if _cors_env else DEFAULT_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Range"],
    expose_headers=["Accept-Ranges", "Content-Range", "Content-Length", "X-Content-Type-Options"],
)

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
    image_base64: Optional[str] = None  # raw base64 or full data: URL
    thread_id: str = "default_user"
    attachments: list[str] = Field(default_factory=list)
    screen_state: Optional[ScreenState] = None
    interrupted: Optional[Interrupted] = None


def _require_allowed_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin is None:
        return
    configured = os.environ.get("ALLOWED_ORIGINS")
    allowed = (
        {value.strip() for value in configured.split(",") if value.strip()}
        if configured is not None
        else DEFAULT_MEDIA_MUTATION_ORIGINS
    )
    if origin not in allowed:
        raise HTTPException(status_code=403, detail="Origin not allowed")


@app.get("/")
async def root():
    return {"status": "ok", "message": "Mercy Protocol Backend Running"}

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
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            log_bus.emit("error", "ERROR", f"chat_endpoint exception: {str(e)[:200]}")
            error_event = {"type": "error", "message": f"Agent error: {e}"}
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


def _media_file_response(path: Path, mime: str) -> FileResponse:
    return FileResponse(
        path,
        media_type=mime,
        headers={"X-Content-Type-Options": "nosniff"},
        content_disposition_type="inline",
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
