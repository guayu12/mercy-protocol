from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import edge_tts
from pydantic import BaseModel
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
from core.speech import synthesize

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_graph()
    yield
    await close_graph()

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

class ChatRequest(BaseModel):
    message: str
    image_base64: Optional[str] = None  # raw base64 or full data: URL
    thread_id: str = "default_user"

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

# ─── Chat (SSE, token-level streaming) ───────────────────────────────
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Streams the agent reply as Server-Sent Events. Each event is JSON:
    {"type": "token", "content": ...} | {"type": "tool", "name": ..., "status": ...}
    | {"type": "done", "reply": ...} | {"type": "error", "message": ...}
    """
    async def event_stream():
        try:
            async for event in process_query(request.message, request.thread_id, request.image_base64):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            log_bus.emit("error", "ERROR", f"chat_endpoint exception: {str(e)[:200]}")
            error_event = {"type": "error", "message": f"Agent error: {e}"}
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)

# ─── TTS ──────────────────────────────────────────────────────────────
@app.get("/api/tts")
async def tts_endpoint(text: str):
    """Convert text to speech stream using edge-tts."""
    async def generate_audio():
        communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    return StreamingResponse(generate_audio(), media_type="audio/mpeg")


class SpeechRequest(BaseModel):
    text: str


@app.post("/api/speech")
async def speech_endpoint(req: SpeechRequest):
    """Synthesize speech and return base64 audio plus a timed viseme track for avatar lip-sync."""
    return await synthesize(req.text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
