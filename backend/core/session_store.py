"""Async SQLite persistence for reports and spoken conversation history."""

import json
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
                created_at REAL NOT NULL,
                link_policy TEXT NOT NULL DEFAULT 'open',
                sources TEXT
            );
            CREATE TABLE IF NOT EXISTS utterances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                interrupted INTEGER DEFAULT 0,
                attachments TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS app_flags (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_reports_thread_id
                ON reports(thread_id);
            CREATE INDEX IF NOT EXISTS idx_utterances_thread_id_id
                ON utterances(thread_id, id);
            """
        )
        cursor = await _connection.execute("PRAGMA table_info(reports)")
        report_columns = {row["name"] for row in await cursor.fetchall()}
        await cursor.close()
        if "link_policy" not in report_columns:
            await _connection.execute(
                "ALTER TABLE reports ADD COLUMN "
                "link_policy TEXT NOT NULL DEFAULT 'open'"
            )
        if "sources" not in report_columns:
            await _connection.execute("ALTER TABLE reports ADD COLUMN sources TEXT")
        cursor = await _connection.execute("PRAGMA table_info(utterances)")
        utterance_columns = {row["name"] for row in await cursor.fetchall()}
        await cursor.close()
        if "attachments" not in utterance_columns:
            await _connection.execute(
                "ALTER TABLE utterances ADD COLUMN "
                "attachments TEXT NOT NULL DEFAULT '[]'"
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
    report = dict(row)
    report.setdefault("link_policy", "open")
    sources = report.get("sources")
    report["sources"] = json.loads(sources) if sources is not None else None
    return report


def _utterance_dict(row) -> dict:
    utterance = dict(row)
    attachments = utterance.get("attachments")
    utterance["attachments"] = json.loads(attachments) if attachments else []
    utterance["interrupted"] = bool(utterance.get("interrupted", False))
    return utterance


async def create_report(
    thread_id: str,
    title: str,
    query: str,
    markdown: str,
    *,
    link_policy: str = "open",
    sources: list[dict] | None = None,
) -> dict:
    report = {
        "thread_id": thread_id,
        "report_id": f"r_{secrets.token_hex(4)}",
        "title": title,
        "query": query,
        "markdown": markdown,
        "created_at": time.time(),
        "link_policy": link_policy,
        "sources": sources,
    }
    values = {
        **report,
        "sources_json": json.dumps(sources, ensure_ascii=False)
        if sources is not None
        else None,
    }
    async with _get_lock():
        db = _db()
        await db.execute(
            "INSERT INTO reports "
            "(thread_id, report_id, title, query, markdown, created_at, link_policy, sources) "
            "VALUES (:thread_id, :report_id, :title, :query, :markdown, :created_at, "
            ":link_policy, :sources_json)",
            values,
        )
        await db.commit()
    return report


async def update_report_markdown(report_id: str, markdown: str) -> None:
    async with _get_lock():
        db = _db()
        await db.execute(
            "UPDATE reports SET markdown = ? WHERE report_id = ?",
            (markdown, report_id),
        )
        await db.commit()


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


async def get_flag(key: str) -> str | None:
    async with _get_lock():
        cursor = await _db().execute(
            "SELECT value FROM app_flags WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        await cursor.close()
    return row["value"] if row else None


async def set_flag(key: str, value: str) -> None:
    async with _get_lock():
        db = _db()
        await db.execute(
            "INSERT INTO app_flags (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        await db.commit()


async def add_utterance(
    thread_id: str,
    turn_id: str,
    role: str,
    text: str,
    interrupted: bool = False,
    attachments: list[dict] | None = None,
):
    if role == "assistant" and not text.strip():
        return
    async with _get_lock():
        db = _db()
        await db.execute(
            "INSERT INTO utterances "
            "(thread_id, turn_id, role, text, interrupted, attachments, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                thread_id,
                turn_id,
                role,
                text,
                int(interrupted),
                json.dumps(attachments or [], ensure_ascii=False),
                time.time(),
            ),
        )
        await db.commit()


async def mark_interrupted(
    thread_id: str, turn_id: str, spoken_text: str | None
):
    async with _get_lock():
        db = _db()
        if spoken_text is None:
            cursor = await db.execute(
                "UPDATE utterances SET interrupted = 1 "
                "WHERE id = (SELECT id FROM utterances WHERE thread_id = ? AND turn_id = ? "
                "AND role = 'assistant' ORDER BY id DESC LIMIT 1)",
                (thread_id, turn_id),
            )
        else:
            cursor = await db.execute(
                "UPDATE utterances SET text = ?, interrupted = 1 "
                "WHERE id = (SELECT id FROM utterances WHERE thread_id = ? AND turn_id = ? "
                "AND role = 'assistant' ORDER BY id DESC LIMIT 1)",
                (spoken_text, thread_id, turn_id),
            )
        if cursor.rowcount == 0 and spoken_text:
            await db.execute(
                "INSERT INTO utterances (thread_id, turn_id, role, text, interrupted, created_at) "
                "VALUES (?, ?, 'assistant', ?, 1, ?)",
                (thread_id, turn_id, spoken_text or "", time.time()),
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
    return [_utterance_dict(row) for row in rows]


async def latest_user_or_assistant_utterance(thread_id: str) -> dict | None:
    async with _get_lock():
        cursor = await _db().execute(
            "SELECT * FROM utterances WHERE thread_id = ? "
            "AND role IN ('user', 'assistant') ORDER BY id DESC LIMIT 1",
            (thread_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
    return _utterance_dict(row) if row else None
