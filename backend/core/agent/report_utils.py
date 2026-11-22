import re

from core import media_store


_MEDIA_REFERENCE = re.compile(r"!\[([^\]]*)\]\(media:(m_[A-Za-z0-9_-]+)\)")


async def rewrite_report_media(markdown: str, thread_id: str) -> tuple[str, list[str]]:
    parts = []
    removed = []
    cursor = 0
    for match in _MEDIA_REFERENCE.finditer(markdown):
        parts.append(markdown[cursor : match.start()])
        record = await media_store.get_media(match.group(2), thread_id)
        if record and record["kind"] != "document":
            parts.append(match.group(0))
        else:
            parts.append(match.group(1))
            removed.append(match.group(2))
        cursor = match.end()
    parts.append(markdown[cursor:])
    return "".join(parts), removed
