import asyncio
import logging
import time

from core import media_store
from core import gemini_files


GEMINI_FILE_MAX_AGE_SECONDS = 48 * 60 * 60
CLEANUP_INTERVAL_SECONDS = 60


async def cleanup_gemini_files(*, delete_all: bool = False) -> None:
    for record in await media_store.list_gemini_files():
        media_id = record["media_id"]
        async with gemini_files.analyze_lock(media_id):
            now = time.time()
            current = await media_store.get_gemini_file(media_id)
            if current is None:
                continue
            if now - current["uploaded_at"] >= GEMINI_FILE_MAX_AGE_SECONDS:
                await media_store.delete_gemini_file(media_id)
                continue
            if not delete_all and current["delete_after"] > now:
                continue
            if await gemini_files.delete_file(current["name"]):
                await media_store.delete_gemini_file(media_id)


async def gemini_file_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            await cleanup_gemini_files()
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.getLogger(__name__).warning("Gemini file cleanup sweep failed")
