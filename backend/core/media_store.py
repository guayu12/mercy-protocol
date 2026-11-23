"""Persistent media and stage metadata stored alongside session history."""

import asyncio
import json
import os
import secrets
import time
from pathlib import Path
import aiosqlite

from core.file_cleanup import discard_file


_connection: aiosqlite.Connection | None = None
_store_lock: asyncio.Lock | None = None
_media_root = Path(__file__).resolve().parents[1] / "data" / "media"
_cache_locks: dict[str, asyncio.Lock] = {}
_cache_eviction_lock = asyncio.Lock()

IMAGE_LIMIT = 20 * 1024 * 1024
VIDEO_LIMIT = 200 * 1024 * 1024
DOCUMENT_LIMIT = 50 * 1024 * 1024
MP4_BRANDS = {
    b"isom",
    b"iso2",
    b"iso4",
    b"iso5",
    b"iso6",
    b"mp41",
    b"mp42",
    b"avc1",
    b"dash",
    b"M4V ",
}


class UnsupportedUpload(Exception):
    pass


class UploadTooLarge(Exception):
    pass


def _lock() -> asyncio.Lock:
    global _store_lock
    if _store_lock is None:
        _store_lock = asyncio.Lock()
    return _store_lock


def _row_dict(row) -> dict | None:
    return dict(row) if row is not None else None


def media_directory() -> Path:
    return _media_root


def cache_directory() -> Path:
    return _media_root / "cache"


def sniff_media(header: bytes) -> tuple[str, str] | None:
    if header.startswith(b"%PDF-"):
        return "document", "application/pdf"
    if header.startswith(b"\xff\xd8\xff"):
        return "image", "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", "image/png"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image", "image/gif"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image", "image/webp"
    if len(header) >= 16 and header[4:8] == b"ftyp":
        brands = {header[8:12]}
        brands.update(
            header[index:index + 4]
            for index in range(16, len(header) - 3, 4)
        )
        if brands & MP4_BRANDS:
            return "video", "video/mp4"
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return "video", "video/webm"
    return None


def _safe_filename(filename: str | None) -> str:
    name = Path(filename or "上传文件").name
    return "".join(char for char in name if char.isprintable() and char not in "/\\")[:160]


async def _migrate_media_table(connection: aiosqlite.Connection) -> None:
    cursor = await connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'media'"
    )
    row = await cursor.fetchone()
    await cursor.close()
    schema = str(row["sql"] or "").lower() if row is not None else ""
    if "'document'" in schema:
        return

    await connection.execute("BEGIN IMMEDIATE")
    try:
        await connection.execute(
            """
            CREATE TABLE media_new (
                media_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                kind TEXT NOT NULL CHECK (kind IN ('image', 'video', 'document')),
                origin TEXT NOT NULL CHECK (origin IN ('upload', 'search', 'link')),
                url TEXT,
                path TEXT,
                mime TEXT NOT NULL,
                bytes INTEGER,
                caption TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_used_at REAL NOT NULL
            )
            """
        )
        await connection.execute(
            """
            INSERT INTO media_new (
                media_id, thread_id, kind, origin, url, path, mime, bytes,
                caption, created_at, last_used_at
            )
            SELECT
                media_id, thread_id, kind, origin, url, path, mime, bytes,
                caption, created_at, last_used_at
            FROM media
            """
        )
        await connection.execute("DROP TABLE media")
        await connection.execute("ALTER TABLE media_new RENAME TO media")
        await connection.execute(
            "CREATE INDEX idx_media_thread_created ON media(thread_id, created_at)"
        )
        await connection.execute(
            "CREATE UNIQUE INDEX idx_media_thread_url "
            "ON media(thread_id, url) WHERE url IS NOT NULL"
        )
        await connection.commit()
    except Exception:
        await connection.rollback()
        raise


async def init_store(path: str) -> None:
    global _connection, _media_root
    database_path = Path(path).resolve()
    _media_root = database_path.parent / "media"
    _media_root.mkdir(parents=True, exist_ok=True)
    cache_directory().mkdir(parents=True, exist_ok=True)
    async with _lock():
        if _connection is not None:
            await _connection.close()
        _connection = await aiosqlite.connect(str(database_path))
        _connection.row_factory = aiosqlite.Row
        await _connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS media (
                media_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                kind TEXT NOT NULL CHECK (kind IN ('image', 'video', 'document')),
                origin TEXT NOT NULL CHECK (origin IN ('upload', 'search', 'link')),
                url TEXT,
                path TEXT,
                mime TEXT NOT NULL,
                bytes INTEGER,
                caption TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_used_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS gemini_files (
                media_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                uri TEXT NOT NULL,
                mime TEXT NOT NULL,
                duration_seconds REAL,
                uploaded_at REAL NOT NULL,
                delete_after REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS stage_state (
                thread_id TEXT PRIMARY KEY,
                items_json TEXT NOT NULL,
                caption TEXT NOT NULL,
                focus TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            """
        )
        await _migrate_media_table(_connection)
        await _connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_media_thread_created "
            "ON media(thread_id, created_at)"
        )
        await _connection.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_media_thread_url "
            "ON media(thread_id, url) WHERE url IS NOT NULL"
        )
        await _connection.commit()


async def close_store() -> None:
    global _connection
    async with _lock():
        if _connection is not None:
            await _connection.close()
        _connection = None


def _db() -> aiosqlite.Connection:
    if _connection is None:
        raise RuntimeError("media store is not initialized")
    return _connection


async def register_media(
    thread_id: str,
    kind: str,
    origin: str,
    mime: str,
    caption: str,
    *,
    url: str | None = None,
    path: str | None = None,
    byte_count: int | None = None,
) -> dict:
    if kind == "document" and (origin != "upload" or url is not None):
        raise ValueError("Documents can only be uploaded")
    now = time.time()
    async with _lock():
        db = _db()
        if url is not None:
            cursor = await db.execute(
                "SELECT * FROM media WHERE thread_id = ? AND url = ?",
                (thread_id, url),
            )
            existing = await cursor.fetchone()
            await cursor.close()
            if existing:
                return dict(existing)
        record = {
            "media_id": f"m_{secrets.token_urlsafe(16)}",
            "thread_id": thread_id,
            "kind": kind,
            "origin": origin,
            "url": url,
            "path": path,
            "mime": mime,
            "bytes": byte_count,
            "caption": caption or "",
            "created_at": now,
            "last_used_at": now,
        }
        await db.execute(
            "INSERT OR IGNORE INTO media "
            "(media_id, thread_id, kind, origin, url, path, mime, bytes, caption, created_at, last_used_at) "
            "VALUES (:media_id, :thread_id, :kind, :origin, :url, :path, :mime, :bytes, :caption, :created_at, :last_used_at)",
            record,
        )
        await db.commit()
        if url is not None:
            cursor = await db.execute(
                "SELECT * FROM media WHERE thread_id = ? AND url = ?",
                (thread_id, url),
            )
            existing = await cursor.fetchone()
            await cursor.close()
            if existing:
                return dict(existing)
        return record


async def register_upload(thread_id: str, upload) -> dict:
    header = await upload.read(16 + 128)
    detected = sniff_media(header)
    if detected is None:
        raise UnsupportedUpload("Unsupported media format")
    kind, mime = detected
    limit = {
        "image": IMAGE_LIMIT,
        "video": VIDEO_LIMIT,
        "document": DOCUMENT_LIMIT,
    }[kind]
    media_id = f"m_{secrets.token_urlsafe(16)}"
    path = _media_root / media_id
    total = 0
    try:
        with path.open("wb") as output:
            chunk = header
            while chunk:
                total += len(chunk)
                if total > limit:
                    raise UploadTooLarge("Upload exceeds size limit")
                output.write(chunk)
                chunk = await upload.read(1024 * 1024)
        return await register_media(
            thread_id,
            kind,
            "upload",
            mime,
            _safe_filename(getattr(upload, "filename", None)),
            path=str(path),
            byte_count=total,
        )
    except Exception:
        discard_file(path)
        raise


async def register_legacy_image(thread_id: str, data: bytes) -> dict:
    detected = sniff_media(data[:16])
    if detected is None or detected[0] != "image":
        raise UnsupportedUpload("Unsupported image format")
    if len(data) > IMAGE_LIMIT:
        raise UploadTooLarge("Upload exceeds size limit")
    kind, mime = detected
    media_id = f"m_{secrets.token_urlsafe(16)}"
    path = _media_root / media_id
    try:
        path.write_bytes(data)
        return await register_media(
            thread_id, kind, "upload", mime, "用户上传图片", path=str(path), byte_count=len(data)
        )
    except Exception:
        discard_file(path)
        raise


async def get_media(media_id: str, thread_id: str | None = None) -> dict | None:
    async with _lock():
        if thread_id is None:
            cursor = await _db().execute(
                "SELECT * FROM media WHERE media_id = ?", (media_id,)
            )
        else:
            cursor = await _db().execute(
                "SELECT * FROM media WHERE media_id = ? AND thread_id = ?",
                (media_id, thread_id),
            )
        row = await cursor.fetchone()
        await cursor.close()
    return _row_dict(row)


async def list_media(
    thread_id: str, limit: int = 20, extra_ids: list[str] | None = None
) -> list[dict]:
    async with _lock():
        db = _db()
        cursor = await db.execute(
            "SELECT * FROM media WHERE thread_id = ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (thread_id, limit),
        )
        rows = [dict(row) for row in await cursor.fetchall()]
        await cursor.close()
        known = {row["media_id"] for row in rows}
        missing = [item for item in (extra_ids or []) if item not in known]
        if missing:
            placeholders = ",".join("?" for _ in missing)
            cursor = await db.execute(
                f"SELECT * FROM media WHERE thread_id = ? AND media_id IN ({placeholders})",
                (thread_id, *missing),
            )
            rows.extend(dict(row) for row in await cursor.fetchall())
            await cursor.close()
    return rows


async def set_stage(
    thread_id: str, items: list[dict], caption: str, focus: str
) -> None:
    async with _lock():
        await _db().execute(
            "INSERT INTO stage_state (thread_id, items_json, caption, focus, updated_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT(thread_id) DO UPDATE SET "
            "items_json=excluded.items_json, caption=excluded.caption, "
            "focus=excluded.focus, updated_at=excluded.updated_at",
            (thread_id, json.dumps(items, ensure_ascii=False), caption, focus, time.time()),
        )
        await _db().commit()


async def get_stage(thread_id: str) -> dict | None:
    async with _lock():
        cursor = await _db().execute(
            "SELECT * FROM stage_state WHERE thread_id = ?", (thread_id,)
        )
        row = await cursor.fetchone()
        await cursor.close()
    if row is None:
        return None
    result = dict(row)
    original_items = json.loads(result.pop("items_json"))
    items = [
        item
        for item in original_items
        if isinstance(item, dict) and item.get("kind") != "document"
    ]
    if len(items) != len(original_items):
        async with _lock():
            await _db().execute(
                "UPDATE stage_state SET items_json = ?, updated_at = ? "
                "WHERE thread_id = ?",
                (json.dumps(items, ensure_ascii=False), time.time(), thread_id),
            )
            await _db().commit()
    result["items"] = items
    return result


async def clear_stage(thread_id: str) -> None:
    async with _lock():
        await _db().execute("DELETE FROM stage_state WHERE thread_id = ?", (thread_id,))
        await _db().commit()


async def reconcile_stage(thread_id: str, stage_media_ids: list[str]) -> None:
    if stage_media_ids:
        return
    await clear_stage(thread_id)


async def update_last_used(media_id: str) -> None:
    async with _lock():
        await _db().execute(
            "UPDATE media SET last_used_at = ? WHERE media_id = ?",
            (time.time(), media_id),
        )
        await _db().commit()


def cache_lock(media_id: str) -> asyncio.Lock:
    lock = _cache_locks.get(media_id)
    if lock is None:
        lock = asyncio.Lock()
        _cache_locks[media_id] = lock
    return lock


async def cached_media() -> list[dict]:
    async with _lock():
        cursor = await _db().execute(
            "SELECT * FROM media WHERE path LIKE ? ORDER BY last_used_at ASC",
            (str(cache_directory()) + os.sep + "%",),
        )
        rows = [dict(row) for row in await cursor.fetchall()]
        await cursor.close()
    return rows


async def set_cached_path(media_id: str, path: str, byte_count: int) -> None:
    async with _lock():
        await _db().execute(
            "UPDATE media SET path = ?, bytes = ?, last_used_at = ? WHERE media_id = ?",
            (path, byte_count, time.time(), media_id),
        )
        await _db().commit()


async def update_media_mime(media_id: str, mime: str) -> None:
    async with _lock():
        await _db().execute(
            "UPDATE media SET mime = ?, last_used_at = ? WHERE media_id = ?",
            (mime, time.time(), media_id),
        )
        await _db().commit()


async def update_media_probe(
    media_id: str, mime: str, byte_count: int | None
) -> None:
    async with _lock():
        await _db().execute(
            "UPDATE media SET mime = ?, bytes = COALESCE(?, bytes), last_used_at = ? "
            "WHERE media_id = ?",
            (mime, byte_count, time.time(), media_id),
        )
        await _db().commit()


async def clear_cached_path(media_id: str) -> None:
    async with _lock():
        await _db().execute(
            "UPDATE media SET path = NULL WHERE media_id = ?", (media_id,)
        )
        await _db().commit()


async def get_gemini_file(media_id: str) -> dict | None:
    async with _lock():
        cursor = await _db().execute(
            "SELECT * FROM gemini_files WHERE media_id = ?", (media_id,)
        )
        row = await cursor.fetchone()
        await cursor.close()
    return dict(row) if row is not None else None


async def list_gemini_files() -> list[dict]:
    async with _lock():
        cursor = await _db().execute(
            "SELECT * FROM gemini_files ORDER BY uploaded_at ASC"
        )
        rows = await cursor.fetchall()
        await cursor.close()
    return [dict(row) for row in rows]


async def put_gemini_file(
    media_id: str,
    name: str,
    uri: str,
    mime: str,
    duration_seconds: float | None,
    uploaded_at: float,
    delete_after: float,
) -> None:
    async with _lock():
        await _db().execute(
            "INSERT INTO gemini_files "
            "(media_id, name, uri, mime, duration_seconds, uploaded_at, delete_after) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(media_id) DO UPDATE SET "
            "name = excluded.name, uri = excluded.uri, mime = excluded.mime, "
            "duration_seconds = excluded.duration_seconds, "
            "uploaded_at = excluded.uploaded_at, delete_after = excluded.delete_after",
            (
                media_id,
                name,
                uri,
                mime,
                duration_seconds,
                uploaded_at,
                delete_after,
            ),
        )
        await _db().commit()


async def set_gemini_file_delete_after(media_id: str, delete_after: float) -> None:
    async with _lock():
        await _db().execute(
            "UPDATE gemini_files SET delete_after = ? WHERE media_id = ?",
            (delete_after, media_id),
        )
        await _db().commit()


async def delete_gemini_file(media_id: str) -> None:
    async with _lock():
        await _db().execute(
            "DELETE FROM gemini_files WHERE media_id = ?", (media_id,)
        )
        await _db().commit()


async def cache_remote_image(media_id: str, fetcher, max_cache_bytes: int) -> dict:
    async with cache_lock(media_id):
        record = await get_media(media_id)
        if record is None or record["kind"] != "image" or not record["url"]:
            raise ValueError("Remote image is unavailable")
        cache_root = cache_directory().resolve()
        cached_path = Path(record["path"]).resolve() if record["path"] else None
        if (
            cached_path
            and cached_path.parent == cache_root
            and cached_path.is_file()
        ):
            await update_last_used(media_id)
            return record

        destination = cache_root / media_id
        try:
            result = await fetcher.download_to_file(
                record["url"], destination, IMAGE_LIMIT
            )
            mime = result["content_type"]
            if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
                raise ValueError("Remote media is not a supported image")
            size = destination.stat().st_size
            await update_media_mime(media_id, mime)
            async with _cache_eviction_lock:
                await set_cached_path(media_id, str(destination), size)
                total = 0
                entries = await cached_media()
                for entry in entries:
                    path = Path(entry["path"])
                    if path.is_file():
                        total += path.stat().st_size
                    else:
                        await clear_cached_path(entry["media_id"])
                for entry in entries:
                    if total <= max_cache_bytes:
                        break
                    if entry["media_id"] == media_id:
                        continue
                    path = Path(entry["path"])
                    if path.is_file():
                        size = path.stat().st_size
                        if not discard_file(path):
                            continue
                        total -= size
                    await clear_cached_path(entry["media_id"])
                if total > max_cache_bytes:
                    discard_file(destination)
                    await clear_cached_path(media_id)
                    raise ValueError("Remote media cache is full")
                await update_last_used(media_id)
                return await get_media(media_id)
        except Exception:
            discard_file(destination)
            raise
