import re
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch

from core import media_store


web_search = TavilySearch(
    max_results=3,
    include_images=True,
    include_image_descriptions=True,
    description="Search the web for current events, facts, or any information you do not know.",
)


def _without_urls(value: str) -> str:
    return re.sub(r"https?://[^\s)\]>]+", "", value, flags=re.IGNORECASE).strip()


async def tavily_raw(query: str, config: RunnableConfig) -> dict:
    try:
        result = await web_search.ainvoke({"query": query}, config=config)
    except Exception as error:
        return {"error": str(error)}
    if isinstance(result, dict):
        return result
    return {"answer": str(result)}


async def format_search_result(
    result: dict,
    thread_id: str,
    numbered_from: int | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    if result.get("error"):
        return str(result["error"]), []

    sections = []
    sources = []
    if result.get("answer"):
        sections.append(_without_urls(str(result["answer"])))

    next_number = numbered_from
    for item in result.get("results", []):
        if not isinstance(item, dict):
            continue
        title = _without_urls(str(item.get("title", "")))
        content = _without_urls(str(item.get("content", "")))
        raw_url = item.get("url")
        url = (
            raw_url
            if isinstance(raw_url, str)
            and raw_url.startswith(("http://", "https://"))
            else ""
        )
        section = f"{title}\n{content}".strip()
        if next_number is not None:
            section = f"[{next_number}] {section}".strip()
            next_number += 1
        if section:
            sections.append(section)
        sources.append((title, url))

    image_lines = []
    image_mimes = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    for image in result.get("images", []):
        if not isinstance(image, dict):
            continue
        description = _without_urls(str(image.get("description") or "")).strip()
        url = image.get("url") or image.get("image_url")
        if (
            not description
            or not isinstance(url, str)
            or not url.startswith(("http://", "https://"))
        ):
            continue
        extension = Path(url.split("?", 1)[0]).suffix.lower()
        record = await media_store.register_media(
            thread_id,
            "image",
            "search",
            image_mimes.get(extension, "image/jpeg"),
            description,
            url=url,
        )
        image_lines.append(f"可展示图片：{record['media_id']}：{description}")
    if image_lines:
        sections.append("\n".join(image_lines))
    return "\n\n".join(section for section in sections if section), sources
