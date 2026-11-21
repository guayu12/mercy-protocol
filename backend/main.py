from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import os
import sys
import asyncio
import json

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from typing import Optional
from core import log_bus  # real-time log broadcaster
from core.agent.graph import process_query, init_graph, close_graph
from core import session_store, speech_queue
from core.speech import synthesize
from core.speakable import to_speakable

@asynccontextmanager
async def lifespan(app: FastAPI):
    await session_store.init_store(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "session.db")
    )
    await init_graph()
    yield
    await close_graph()
    await session_store.close_store()

app = FastAPI(title="Mercy Protocol — AI Judicial Interface", version="2.0.0", lifespan=lifespan)

DEFAULT_CORS_ORIGINS = [
    "http://localhost:1420",   # Vite dev server (tauri dev)
    "http://127.0.0.1:1420",
    "http://localhost:5173",   # Vite default port
    "http://127.0.0.1:5173",
    "tauri://localhost",       # Tauri on macOS / Linux
    "http://tauri.localhost",  # Tauri v2 on Windows (Surface)
    "https://tauri.localhost",
]
_cors_env = os.environ.get("CORS_ORIGINS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_env.split(",") if o.strip()] if _cors_env else DEFAULT_CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

class ScreenState(BaseModel):
    viewing_report_id: Optional[str] = None


class Interrupted(BaseModel):
    turn_id: str
    spoken_sentence_ids: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    image_base64: Optional[str] = None  # raw base64 or full data: URL
    thread_id: str = "default_user"
    screen_state: Optional[ScreenState] = None
    interrupted: Optional[Interrupted] = None


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
    async def event_stream():
        try:
            async for event in process_query(
                request.message,
                request.thread_id,
                request.image_base64,
                request.screen_state.model_dump() if request.screen_state else None,
                request.interrupted.model_dump() if request.interrupted else None,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            log_bus.emit("error", "ERROR", f"chat_endpoint exception: {str(e)[:200]}")
            error_event = {"type": "error", "message": f"Agent error: {e}"}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)

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
    return {
        "reports": await session_store.list_reports(thread_id),
        "utterances": await session_store.recent_utterances(thread_id),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
