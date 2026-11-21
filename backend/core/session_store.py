"""Async SQLite persistence for reports and spoken conversation history."""

import os
import secrets
import time
from typing import Optional

import aiosqlite


_connection: Optional[aiosqlite.Connection] = None
_store_lock = None


def _get_lock():
    global _store_lock
    if _store_lock is None:
        import asyncio

        _store_lock = asyncio.Lock()
    return _store_lock


async def init_store(path: str):
    """Create the session database and its tables."""
    global _connection
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    async with _get_lock():
        if _connection is not None:
            await _connection.close()
        _connection = await aiosqlite.connect(path)
        _connection.row_factory = aiosqlite.Row
        await _connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS reports (
                thread_id TEXT NOT NULL,
                report_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                query TEXT NOT NULL,
                markdown TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS utterances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                interrupted INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            );
            """
        )
        await _connection.commit()


async def close_store():
    """Close the session database connection."""
    global _connection
    async with _get_lock():
        if _connection is not None:
            await _connection.close()
        _connection = None


def _db() -> aiosqlite.Connection:
    if _connection is None:
        raise RuntimeError("session store is not initialized")
    return _connection


def _row_dict(row) -> dict:
    return dict(row)


async def create_report(thread_id: str, title: str, query: str, markdown: str) -> dict:
    report = {
        "thread_id": thread_id,
        "report_id": f"r_{secrets.token_hex(4)}",
        "title": title,
        "query": query,
        "markdown": markdown,
        "created_at": time.time(),
    }
    async with _get_lock():
        db = _db()
        await db.execute(
            "INSERT INTO reports (thread_id, report_id, title, query, markdown, created_at) "
            "VALUES (:thread_id, :report_id, :title, :query, :markdown, :created_at)",
            report,
        )
        await db.commit()
    return report


async def get_report(thread_id: str, report_id: str) -> dict | None:
    async with _get_lock():
        cursor = await _db().execute(
            "SELECT * FROM reports WHERE thread_id = ? AND report_id = ?",
            (thread_id, report_id),
        )
        row = await cursor.fetchone()
        await cursor.close()
    return _row_dict(row) if row else None


async def list_reports(thread_id: str) -> list[dict]:
    async with _get_lock():
        cursor = await _db().execute(
            "SELECT * FROM reports WHERE thread_id = ? ORDER BY created_at ASC, rowid ASC",
            (thread_id,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
    return [_row_dict(row) for row in rows]


async def add_utterance(
    thread_id: str,
    turn_id: str,
    role: str,
    text: str,
    interrupted: bool = False,
):
    async with _get_lock():
        db = _db()
        await db.execute(
            "INSERT INTO utterances (thread_id, turn_id, role, text, interrupted, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (thread_id, turn_id, role, text, int(interrupted), time.time()),
        )
        await db.commit()


async def mark_interrupted(thread_id: str, turn_id: str, spoken_text: str):
    async with _get_lock():
        db = _db()
        cursor = await db.execute(
            "UPDATE utterances SET text = ?, interrupted = 1 "
            "WHERE id = (SELECT id FROM utterances WHERE thread_id = ? AND turn_id = ? "
            "AND role = 'assistant' ORDER BY id DESC LIMIT 1)",
            (spoken_text, thread_id, turn_id),
        )
        if cursor.rowcount == 0:
            await db.execute(
                "INSERT INTO utterances (thread_id, turn_id, role, text, interrupted, created_at) "
                "VALUES (?, ?, 'assistant', ?, 1, ?)",
                (thread_id, turn_id, spoken_text, time.time()),
            )
        await db.commit()


async def recent_utterances(thread_id: str, limit: int = 50) -> list[dict]:
    async with _get_lock():
        cursor = await _db().execute(
            "SELECT * FROM (SELECT * FROM utterances WHERE thread_id = ? "
            "ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
            (thread_id, limit),
        )
        rows = await cursor.fetchall()
        await cursor.close()
    return [_row_dict(row) for row in rows]
