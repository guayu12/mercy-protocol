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

app = FastAPI(title="Mercy Protocol — AI Judicial Interface", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from typing import Optional
from core import log_bus  # real-time log broadcaster

class ChatRequest(BaseModel):
    message: str
    image_base64: Optional[str] = None

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
            # Send a heartbeat every 15 s to keep the connection alive
            while not await request.is_disconnected():
                try:
                    msg = queue.get_nowait()
                    yield f"data: {msg}\n\n"
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.25)
        finally:
            log_bus.unsubscribe(queue)
            log_bus.emit("flow", "FLOW", "SSE client disconnected")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )

# ─── Chat ─────────────────────────────────────────────────────────────
from core.agent.graph import process_query

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        augmented_message = request.message
        if request.image_base64:
            augmented_message += (
                f"\n\n[SYSTEM]: The user has uploaded an image. "
                f"You MUST use the `analyze_image` tool to understand it. "
                f"Pass the user's question as `prompt` and the following base64 string as `base64_image`: "
                f"{request.image_base64}"
            )
        reply = process_query(augmented_message)
        return {"reply": reply, "status": "success"}
    except Exception as e:
        log_bus.emit("error", "ERROR", f"chat_endpoint exception: {str(e)[:200]}")
        return {"reply": f"Agent error: {str(e)}", "status": "error"}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
