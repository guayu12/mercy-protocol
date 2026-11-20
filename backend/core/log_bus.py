"""
log_bus.py — Global real-time log broadcaster for the Mercy AI backend.
Captures structured events (LLM calls, tool use, flow steps) and
broadcasts them to all connected SSE clients.
"""
import asyncio
import json
import time
from typing import List

# ── Global subscriber list ──────────────────────────────────────────
_subscribers: List[asyncio.Queue] = []

def subscribe() -> asyncio.Queue:
    """Create a new subscriber queue and register it."""
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _subscribers.append(q)
    return q

def unsubscribe(q: asyncio.Queue):
    """Remove a subscriber when client disconnects."""
    try:
        _subscribers.remove(q)
    except ValueError:
        pass

def emit(level: str, category: str, message: str, data: dict = None):
    """
    Emit a structured log event to all SSE subscribers.
    This is safe to call from synchronous (non-async) code.
    
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
    # Put into every subscriber queue (non-blocking; drop if full)
    for q in list(_subscribers):
        try:
            q.put_nowait(serialized)
        except asyncio.QueueFull:
            pass  # client too slow — drop message
