"""
log_bus.py — Global real-time log broadcaster for the Mercy AI backend.
Captures structured events (LLM calls, tool use, flow steps) and
broadcasts them to all connected SSE clients.
"""
import asyncio
import json
import time
from typing import List, Tuple

# ── Global subscriber list: (queue, owning event loop) ──────────────
_subscribers: List[Tuple[asyncio.Queue, asyncio.AbstractEventLoop]] = []

def subscribe() -> asyncio.Queue:
    """Create a new subscriber queue and register it. Must be called from the event loop."""
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _subscribers.append((q, asyncio.get_running_loop()))
    return q

def unsubscribe(q: asyncio.Queue):
    """Remove a subscriber when client disconnects."""
    for entry in list(_subscribers):
        if entry[0] is q:
            _subscribers.remove(entry)

def _put(q: asyncio.Queue, serialized: str):
    try:
        q.put_nowait(serialized)
    except asyncio.QueueFull:
        pass  # client too slow — drop message

def emit(level: str, category: str, message: str, data: dict = None):
    """
    Emit a structured log event to all SSE subscribers.
    Safe to call from synchronous code and from worker threads
    (e.g. sync tools that LangGraph runs in an executor).
    
    Parameters
    ----------
    level    : "info" | "warn" | "tool" | "llm" | "flow" | "error"
    category : Short label, e.g. "LLM", "TOOL", "FLOW"
    message  : Human-readable log line
    data     : Optional dict of extra structured fields
    """
    payload = {
        "ts": round(time.time() * 1000),       # ms epoch
        "level": level,
        "category": category,
        "message": message,
        "data": data or {}
    }
    serialized = json.dumps(payload, ensure_ascii=False)
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
    # Put into every subscriber queue (non-blocking; drop if full)
    for q, loop in list(_subscribers):
        if loop is current_loop:
            _put(q, serialized)
        elif not loop.is_closed():
            loop.call_soon_threadsafe(_put, q, serialized)
