import base64
import io
import os
import re
import time
import asyncio
import secrets
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables FIRST — before any module reads them
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BACKEND_DIR, ".env"))
from typing import TypedDict, Annotated, AsyncIterator, Optional, Literal
import aiosqlite
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import adispatch_custom_event
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from core.memory.memory_manager import MemoryManager
from core import log_bus  # ← real-time log broadcaster
from core import media_store, session_store, speech_queue
from core.safe_fetch import safe_media_fetcher
from core.speakable import SpeechRouter

# 1. Initialize Memory
memory_manager = MemoryManager(os.path.join(BACKEND_DIR, "data", "chroma_db"))

# 2. Define the State
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    context: str

# 3. Mock & Real tools
@tool
def check_weather(location: str) -> str:
    """Mock tool to check the weather."""
    log_bus.emit("tool", "TOOL", f"check_weather({location!r})")
    return f"The weather in {location} is 75°F and sunny."

@tool
def get_current_time() -> str:
    """Mock tool to get current time."""
    from datetime import datetime
    result = f"Current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    log_bus.emit("tool", "TOOL", f"get_current_time → {result}")
    return result

@tool
async def save_memory(fact: str) -> str:
    """Save an important fact about the user or instructions to long-term memory. USE EXACTLY when user says 'remember' or explicitly tells you a preference."""
    log_bus.emit("tool", "TOOL", f"save_memory: {fact[:80]}")
    saved = await asyncio.to_thread(memory_manager.add_memory, fact)
    if not saved:
        log_bus.emit("tool", "TOOL", "save_memory skipped — near-duplicate already stored")
        return f"Already in permanent memory (duplicate skipped): {fact}"
    return f"Saved to permanent memory: {fact}"

# Web Search Tool (requires TAVILY_API_KEY in .env)
web_search = TavilySearch(
    max_results=3,
    include_images=True,
    include_image_descriptions=True,
    description="Search the web for current events, facts, or any information you do not know.",
)


def _without_urls(value: str) -> str:
    return re.sub(r"https?://[^\s)\]>]+", "", value, flags=re.IGNORECASE).strip()


@tool
async def tavily_search(query: str, config: RunnableConfig) -> str:
    """Search the web and return text results plus registered, described images."""
    result = await web_search.ainvoke({"query": query}, config=config)
    if not isinstance(result, dict):
        return str(result)
    if result.get("error"):
        return str(result["error"])

    sections = []
    if result.get("answer"):
        sections.append(_without_urls(str(result["answer"])))
    for item in result.get("results", []):
        if not isinstance(item, dict):
            continue
        title = _without_urls(str(item.get("title", "")))
        content = _without_urls(str(item.get("content", "")))
        if title or content:
            sections.append(f"{title}\n{content}".strip())

    thread_id = config.get("configurable", {}).get("thread_id", "default_user")
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
        if not description or not isinstance(url, str) or not url.startswith(("http://", "https://")):
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
    return "\n\n".join(sections) if sections else "没有找到可用的文字结果。"

@tool
def create_local_directory(dir_name: str) -> str:
    """Create a new folder/directory on the local computer. 
    Provide the name of the folder you want to create. It will be created in the user's designated workspace.
    """
    raw_base = os.environ.get("AURA_WORKSPACE_DIR", "~/Desktop/AuraWorkspace")
    base_dir = os.path.expanduser(raw_base)
    os.makedirs(base_dir, exist_ok=True)
    base_path = os.path.realpath(base_dir)
    target_path = os.path.realpath(os.path.join(base_dir, dir_name))
    try:
        if os.path.commonpath((base_path, target_path)) != base_path:
            return "Failed to create directory: target must remain inside the workspace."
    except ValueError:
        return "Failed to create directory: target must remain inside the workspace."
    log_bus.emit("tool", "TOOL", f"create_local_directory: {target_path}")
    try:
        os.makedirs(target_path, exist_ok=True)
        return f"Successfully created directory at: {target_path}"
    except Exception as e:
        return f"Failed to create directory: {str(e)}"


async def _rewrite_report_media(markdown: str, thread_id: str) -> tuple[str, list[str]]:
    pattern = re.compile(r"!\[([^\]]*)\]\(media:(m_[A-Za-z0-9_-]+)\)")
    parts = []
    removed = []
    cursor = 0
    for match in pattern.finditer(markdown):
        parts.append(markdown[cursor : match.start()])
        record = await media_store.get_media(match.group(2), thread_id)
        if record:
            parts.append(match.group(0))
        else:
            parts.append(match.group(1))
            removed.append(match.group(2))
        cursor = match.end()
    parts.append(markdown[cursor:])
    return "".join(parts), removed


@tool
async def show_report(title: str, markdown: str, config: RunnableConfig) -> str:
    """Put a document on screen. ANY table, code, list of links, step-by-step document, or content longer than about three sentences MUST use this tool and never be included in the spoken reply."""
    configurable = config.get("configurable", {})
    thread_id = configurable.get("thread_id", "default_user")
    query = configurable.get("turn_query", "")
    markdown, removed_media_ids = await _rewrite_report_media(markdown, thread_id)
    report = await session_store.create_report(thread_id, title, query, markdown)
    report_id = report["report_id"]
    await adispatch_custom_event(
        "report",
        {"type": "report.open", "report_id": report_id, "title": title, "query": query},
        config=config,
    )
    await adispatch_custom_event(
        "report",
        {"type": "report.delta", "report_id": report_id, "markdown": markdown},
        config=config,
    )
    await adispatch_custom_event(
        "report", {"type": "report.done", "report_id": report_id}, config=config
    )
    result = (
        f"已上屏，报告编号 {report_id}。用户现在能看到它，"
        "口头只需概括要点并询问意见。"
    )
    if removed_media_ids:
        result += " 已移除不属于当前会话的媒体编号：" + "、".join(removed_media_ids)
    return result


@tool
async def show_media(
    media_ids: list[str],
    caption: str,
    config: RunnableConfig,
    focus: Literal["normal", "primary"] = "normal",
) -> str:
    """Show up to six registered media items on the evidence stage."""
    if len(media_ids) > 6:
        return "一次最多展示六项媒体。"
    thread_id = config.get("configurable", {}).get("thread_id", "default_user")
    allowed_images = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    allowed_videos = {"video/mp4", "video/webm"}
    limits = {"image": media_store.IMAGE_LIMIT, "video": media_store.VIDEO_LIMIT}
    failures = []

    async def validate(media_id: str):
        record = await media_store.get_media(media_id, thread_id)
        if record is None:
            return None, "不存在或不属于当前会话"
        if record["origin"] == "upload":
            path = Path(record["path"] or "")
            if not path.is_file():
                return None, "上传文件不存在"
            size = path.stat().st_size
            if size > limits[record["kind"]]:
                return None, "媒体超过大小限制"
            mime = record["mime"]
        else:
            if not record["url"]:
                return None, "媒体地址不可用"
            try:
                result = await asyncio.wait_for(
                    safe_media_fetcher.probe(record["url"]), timeout=3
                )
            except asyncio.TimeoutError:
                return None, "验证超时"
            except Exception:
                return None, "无法安全访问媒体"
            if result["status"] not in {200, 206}:
                return None, f"远程服务器返回 HTTP {result['status']}"
            mime = result["content_type"]
            if result["length"] is not None and result["length"] > limits[record["kind"]]:
                return None, "媒体超过大小限制"
        allowed = allowed_images if record["kind"] == "image" else allowed_videos
        if mime not in allowed:
            return None, "媒体类型不受支持"
        await media_store.update_media_probe(media_id, mime, result["length"] if record["origin"] != "upload" else None)
        return {
            "media_id": media_id,
            "kind": record["kind"],
            "mime": mime,
            "src": f"/api/media/{media_id}",
            "caption": record["caption"],
        }, None

    checked = await asyncio.gather(*(validate(media_id) for media_id in media_ids))
    items = []
    for media_id, (item, reason) in zip(media_ids, checked):
        if item:
            items.append(item)
        else:
            failures.append(f"{media_id}：{reason}")
    if not items:
        details = "；".join(failures) or "没有提供媒体编号"
        return f"没有可展示的媒体。{details}"

    await media_store.set_stage(thread_id, items, caption, focus)
    await adispatch_custom_event(
        "stage",
        {"type": "stage.show", "caption": caption, "focus": focus, "items": items},
        config=config,
    )
    result = f"展示台已更新，显示 {len(items)} 项。"
    if failures:
        result += " 未展示：" + "；".join(failures)
    return result


@tool
async def clear_stage(config: RunnableConfig) -> str:
    """Close the evidence stage when the current topic moves on."""
    thread_id = config.get("configurable", {}).get("thread_id", "default_user")
    await media_store.clear_stage(thread_id)
    await adispatch_custom_event(
        "stage", {"type": "stage.clear"}, config=config
    )
    return "展示台已关闭。"


@tool
async def read_report(report_id: str, config: RunnableConfig) -> str:
    """Read the full text of an existing on-screen report when the user asks for more detail."""
    thread_id = config.get("configurable", {}).get("thread_id", "default_user")
    report = await session_store.get_report(thread_id, report_id)
    return report["markdown"] if report else "找不到这份报告"


@tool
async def highlight_report(
    report_id: str, anchor: str, config: RunnableConfig
) -> str:
    """Highlight a specific part of an on-screen report when the user asks to see it; anchor must be its exact heading or a short exact phrase."""
    thread_id = config.get("configurable", {}).get("thread_id", "default_user")
    report = await session_store.get_report(thread_id, report_id)
    if not report:
        return "找不到这份报告"
    if not anchor.strip() or anchor.casefold() not in report["markdown"].casefold():
        return "报告里没有这段文字，请用报告中的原文标题或短语"
    await adispatch_custom_event(
        "report",
        {"type": "report.highlight", "report_id": report_id, "anchor": anchor},
        config=config,
    )
    return "已在屏幕上高亮。"


tools = [
    check_weather,
    get_current_time,
    save_memory,
    tavily_search,
    create_local_directory,
    show_report,
    show_media,
    clear_stage,
    read_report,
    highlight_report,
]

# 4. Setup LLMs (OpenAI official API; vision model handles turns with images)
_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
_api_key = os.environ.get("OPENAI_API_KEY", "dummy")
_model_name = os.environ.get("DEFAULT_MODEL", "gpt-6-luna")
_vision_model_name = os.environ.get("OPENAI_VISION_MODEL") or _model_name
_reasoning_effort = os.environ.get("DEFAULT_REASONING_EFFORT", "")

def _make_llm(model: str):
    options = dict(
        model=model,
        api_key=_api_key,
        base_url=_base_url,
        streaming=True,
        stream_usage=True,
    )
    if _reasoning_effort:
        options["reasoning"] = {"effort": _reasoning_effort}
    return ChatOpenAI(**options).bind_tools(tools)

llm_with_tools = _make_llm(_model_name)
vision_llm_with_tools = llm_with_tools if _vision_model_name == _model_name else _make_llm(_vision_model_name)

def _text_of(content) -> str:
    """Extract plain text from str or multimodal content blocks."""
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")

def _has_image(content) -> bool:
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "image_url" for b in content
    )

async def _media_screen_context(thread_id: str, screen_state: dict) -> str:
    stage = await media_store.get_stage(thread_id)
    stage_items = stage["items"] if stage else []
    stage_ids = [item["media_id"] for item in stage_items]
    records = await media_store.list_media(thread_id, 20, stage_ids)
    kind_labels = {"image": "图片", "video": "视频"}
    origin_labels = {"upload": "上传", "search": "搜索", "link": "链接"}
    media_lines = [
        f"{record['media_id']} · {kind_labels[record['kind']]} · "
        f"{origin_labels[record['origin']]} · {_without_urls(record['caption'])}"
        for record in records
    ]
    if stage:
        stage_line = "展示台：" + "、".join(
            f"{item['media_id']} · {_without_urls(item.get('caption', ''))}"
            for item in stage_items
        )
        current_id = screen_state.get("stage_current_media_id")
        if current_id in stage_ids:
            stage_line += f"\n用户正在看 {current_id}"
    else:
        stage_line = "展示台：关闭"
    return (
        "已登记媒体：\n"
        + ("\n".join(media_lines) if media_lines else "暂无已登记媒体。")
        + "\n"
        + stage_line
    )


def _image_data_url(path: str) -> str | None:
    from PIL import Image

    image_path = Path(path)
    if not image_path.is_file():
        return None
    try:
        with Image.open(image_path) as source:
            source.seek(0)
            image = source.copy()
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            image_format = (source.format or "").upper()
            if image_format == "JPEG":
                image = image.convert("RGB")
                mime = "image/jpeg"
            elif image_format == "WEBP":
                mime = "image/webp"
            else:
                image_format = "PNG"
                mime = "image/png"
            output = io.BytesIO()
            image.save(output, format=image_format)
    except Exception as error:
        log_bus.emit("warn", "MEDIA", f"could not decode uploaded image: {error}")
        return None
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


async def _prepare_user_media(
    query: str,
    thread_id: str,
    attachments: list[str] | None,
    image_base64: str | None,
) -> tuple[str, list[str]]:
    image_urls = []
    video_ids = []
    if image_base64:
        try:
            encoded = (
                image_base64.split(",", 1)[1]
                if image_base64.startswith("data:")
                else image_base64
            )
            legacy_data = base64.b64decode(encoded)
            legacy = await media_store.register_legacy_image(thread_id, legacy_data)
            data_url = _image_data_url(legacy["path"])
            if data_url:
                image_urls.append(data_url)
        except Exception as error:
            log_bus.emit("warn", "MEDIA", f"ignored legacy image attachment: {error}")

    try:
        cache_limit = max(0, int(os.environ.get("MEDIA_CACHE_MAX_MB", "500"))) * 1024 * 1024
    except ValueError:
        cache_limit = 500 * 1024 * 1024
    for media_id in attachments or []:
        record = await media_store.get_media(media_id, thread_id)
        if record is None:
            continue
        if record["kind"] == "video":
            video_ids.append(media_id)
            continue
        try:
            if record["origin"] != "upload":
                record = await media_store.cache_remote_image(
                    media_id, safe_media_fetcher, cache_limit
                )
            data_url = _image_data_url(record.get("path") or "")
            if data_url:
                image_urls.append(data_url)
        except Exception as error:
            log_bus.emit("warn", "MEDIA", f"ignored image attachment {media_id}: {error}")

    if video_ids:
        notices = [
            f"［用户上传了视频 {media_id}，你看不到画面内容］"
            for media_id in video_ids
        ]
        query = "\n".join([query, *notices]).strip()
    return query, image_urls


# 5. Define Graph Nodes
async def agent_node(state: AgentState, config: RunnableConfig):
    """The thinking node of the agent."""
    messages = state["messages"]
    
    user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    latest = user_msgs[-1] if user_msgs else None
    latest_query = _text_of(latest.content) if latest else ""
    use_vision = latest is not None and _has_image(latest.content)
    model_name = _vision_model_name if use_vision else _model_name
    llm = vision_llm_with_tools if use_vision else llm_with_tools

    # Embedding lookup is a blocking HTTP call — keep it off the event loop
    past_contexts = await asyncio.to_thread(memory_manager.format_context, latest_query)
    configurable = config.get("configurable", {})
    thread_id = configurable.get("thread_id", "default_user")
    screen_state = configurable.get("screen_state") or {}
    reports = await session_store.list_reports(thread_id)
    screen_lines = []
    viewing_report_id = screen_state.get("viewing_report_id")
    focused_report = next(
        (
            report
            for report in reports
            if report["report_id"] == viewing_report_id
        ),
        None,
    )
    if focused_report is None and reports:
        focused_report = reports[-1]
    if focused_report:
        markdown = focused_report["markdown"]
        if len(markdown) > 4000:
            markdown = markdown[:4000] + "…（已截断，可用 read_report 读全文）"
        screen_lines.append(
            f"报告编号：{focused_report['report_id']}\n标题：{focused_report['title']}\n"
            f"来源问题：{focused_report['query']}\n正文：\n{markdown}"
        )
        other_reports = [
            report for report in reports if report is not focused_report
        ]
        if other_reports:
            screen_lines.append("其他报告：")
            screen_lines.extend(
                f"- {report['report_id']}：{report['title']}"
                for report in other_reports
            )
    else:
        screen_lines.append("当前还没有报告。")
    if "viewing_report_id" in screen_state:
        screen_lines.append(
            f"用户当前正在看：{screen_state['viewing_report_id'] or '无'}"
        )
    screen_block = "\n".join(screen_lines)
    media_block = await _media_screen_context(thread_id, screen_state)
    assistant_name = os.environ.get("ASSISTANT_NAME", "Aura")
    local_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    sys_prompt = SystemMessage(content=f"""
你叫{assistant_name}，是用户的工作伙伴。说话自然、亲切、简洁，像同事当面汇报。
页面左侧是系统状态栏；中间是“报告区”，显示你用 show_report 放上去的文档；右侧是你本人的形象，下方有字幕和对话记录。用户能看到屏幕上的一切。
你的文字回复会被直接朗读：只写口语短句，不写 Markdown、符号、列表、表格、链接、代码；一般不超过三四句。
表格、代码、链接、分步骤文档、长解释，必须先调用 show_report 上屏，然后口头用两三句概括要点，再问用户的意见；不要复述屏幕上的内容。
闲聊和简单问题直接口头回答，不要上屏。追问已有报告时，优先口头回答；需要指给用户看时用 highlight_report；需要更多细节时用 read_report；只有用户要新的文档时才再次 show_report。
调用搜索等耗时工具前，先说一句简短的过渡语，比如“我查一下”。
如果之前的回复被用户打断，历史里会标注实际说到哪里；用户问“刚才说到哪”时据此接上。

当前本地时间：{local_time}

{past_contexts}

屏幕状态：
{screen_block}

{media_block}

媒体与展示台规则：只能通过 show_media 展示上面列出的媒体编号；只有图片本身是当前主题时才使用 focus=primary；话题转移后调用 clear_stage。视频只能展示，你看不到视频画面，绝不能描述其内容。搜索图片只知道已提供的描述；本轮上传的图片可直接分析。
""")
    full_messages = [sys_prompt] + list(messages)
    input_tokens_est = sum(len(_text_of(m.content)) // 4 for m in full_messages)

    log_bus.emit("flow", "FLOW", f"agent_node → invoking {model_name} ({len(full_messages)} msgs, ~{input_tokens_est} tokens est.)")

    # Retry with exponential backoff for transient API errors
    last_err = None
    for attempt in range(3):
        t0 = time.time()
        try:
            # Pass config explicitly so token callbacks reach astream_events on Python < 3.11
            response = await llm.ainvoke(full_messages, config)
            latency = round((time.time() - t0) * 1000)
            usage = getattr(response, "usage_metadata", None)
            if usage:
                in_tok = usage.get("input_tokens", "?")
                out_tok = usage.get("output_tokens", "?")
            else:
                in_tok, out_tok = "?", "?"

            tool_calls = getattr(response, "tool_calls", [])
            log_bus.emit("llm", "LLM",
                         f"← response {latency}ms | in:{in_tok} out:{out_tok} tok | {'→ ' + ', '.join(tc['name'] for tc in tool_calls) if tool_calls else 'direct reply'}",
                         {"model": model_name, "latency_ms": latency,
                          "input_tokens": in_tok, "output_tokens": out_tok,
                          "tool_calls": [tc["name"] for tc in tool_calls]})
            return {"messages": [response]}
        except Exception as e:
            last_err = e
            err_str = str(e)
            log_bus.emit("warn", "RETRY", f"attempt {attempt+1}/3 — {err_str[:100]}")
            if "429" in err_str or "500" in err_str or "503" in err_str:
                await asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s
                continue
            raise  # Non-transient error, propagate immediately
    raise last_err  # All retries exhausted

# 6. Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
# Tool exceptions become ToolMessages so the model sees the error (and the thread stays valid)
workflow.add_node("tools", ToolNode(tools, handle_tool_errors=True))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

# Compiled lazily: AsyncSqliteSaver needs a running event loop (see init_graph)
app_graph = None
_conn: Optional[aiosqlite.Connection] = None

async def init_graph():
    """Open the async SQLite checkpointer and compile the graph. Call on app startup."""
    global app_graph, _conn
    if app_graph is not None:
        return app_graph
    _conn = await aiosqlite.connect(os.path.join(BACKEND_DIR, "aura_checkpoints.db"))
    checkpointer = AsyncSqliteSaver(_conn)
    await checkpointer.setup()
    app_graph = workflow.compile(checkpointer=checkpointer)
    return app_graph

async def close_graph():
    global app_graph, _conn
    if _conn is not None:
        await _conn.close()
    _conn = None
    app_graph = None

def build_user_message(
    query: str,
    image_base64: Optional[str] = None,
    image_data_urls: list[str] | None = None,
) -> HumanMessage:
    """Build a HumanMessage with native image_url content blocks."""
    images = list(image_data_urls or [])
    if image_base64:
        images.append(
            image_base64
            if image_base64.startswith("data:")
            else f"data:image/jpeg;base64,{image_base64}"
        )
    if not images:
        return HumanMessage(content=query)
    content = [{"type": "text", "text": query}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in images
    )
    return HumanMessage(content=content)

_thread_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

async def _close_pending_tool_calls(graph, config):
    """If a previous run stopped between the agent and tools nodes (e.g. client
    disconnect), answer the dangling tool calls so OpenAI accepts the history."""
    state = await graph.aget_state(config)
    messages = state.values.get("messages", []) if state.values else []
    if not messages or not isinstance(messages[-1], AIMessage):
        return
    pending = getattr(messages[-1], "tool_calls", None) or []
    if not pending:
        return
    log_bus.emit("warn", "FLOW", f"closing {len(pending)} interrupted tool call(s) from previous run")
    await graph.aupdate_state(
        config,
        {"messages": [ToolMessage(content="Tool call interrupted.", tool_call_id=tc["id"], name=tc["name"])
                      for tc in pending]},
        as_node="tools",
    )

async def process_query(
    query: str,
    thread_id: str,
    image_base64: Optional[str] = None,
    screen_state: Optional[dict] = None,
    interrupted: Optional[dict] = None,
    attachments: Optional[list[str]] = None,
) -> AsyncIterator[dict]:
    """Run one turn and stream status, spoken sentences, reports, tools and completion."""
    # Serialize turns per thread so concurrent requests don't overwrite each other's checkpoints
    async with _thread_locks[thread_id]:
        graph = await init_graph()
        query, image_data_urls = await _prepare_user_media(
            query, thread_id, attachments, image_base64
        )
        turn_id = f"t_{secrets.token_hex(4)}"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "turn_query": query,
                "screen_state": screen_state or {},
            }
        }
        await _close_pending_tool_calls(graph, config)
        if interrupted:
            previous_turn_id = interrupted.get("turn_id", "")
            sentence_ids = interrupted.get("spoken_sentence_ids", [])
            latest_utterance = await session_store.latest_user_or_assistant_utterance(
                thread_id
            )
            valid_interruption = (
                isinstance(previous_turn_id, str)
                and latest_utterance is not None
                and latest_utterance["turn_id"] == previous_turn_id
                and isinstance(sentence_ids, list)
                and all(
                    isinstance(sentence_id, str)
                    and sentence_id.startswith(f"{previous_turn_id}.")
                    for sentence_id in sentence_ids
                )
            )
            if valid_interruption:
                current_segment = speech_queue.get_turn_segment(previous_turn_id)
                spoken_entries = [
                    (
                        speech_queue.get_sentence(sentence_id),
                        speech_queue.get_sentence_segment(sentence_id),
                    )
                    for sentence_id in sentence_ids
                ]
                if any(
                    text is None or segment is None
                    for text, segment in spoken_entries
                ) or (sentence_ids and current_segment is None):
                    await session_store.mark_interrupted(
                        thread_id, previous_turn_id, None
                    )
                else:
                    spoken = "".join(text for text, _ in spoken_entries if text)
                    interruption_text = (
                        spoken + "……（说到这里被用户打断）"
                        if spoken
                        else "（还没开口就被用户打断了）"
                    )
                    last_segment = current_segment if current_segment is not None else 0
                    last_segment_text = "".join(
                        text
                        for text, segment in spoken_entries
                        if segment == last_segment and text
                    )
                    checkpoint_text = (
                        last_segment_text + "……（说到这里被用户打断）"
                        if last_segment_text
                        else "（还没开口就被用户打断了）"
                    )
                    state = await graph.aget_state(config)
                    messages = (
                        state.values.get("messages", []) if state.values else []
                    )
                    if messages and isinstance(messages[-1], AIMessage) and not (
                        getattr(messages[-1], "tool_calls", None) or []
                    ):
                        await graph.aupdate_state(
                            config,
                            {
                                "messages": [
                                    AIMessage(
                                        content=checkpoint_text,
                                        id=messages[-1].id,
                                    )
                                ]
                            },
                        )
                    elif messages and isinstance(
                        messages[-1], (HumanMessage, ToolMessage)
                    ):
                        await graph.aupdate_state(
                            config,
                            {"messages": [AIMessage(content=checkpoint_text)]},
                            as_node="agent",
                        )
                    await session_store.mark_interrupted(
                        thread_id, previous_turn_id, interruption_text
                    )
        await session_store.add_utterance(
            thread_id, turn_id, "user", query, interrupted=False
        )
        router = SpeechRouter(turn_id)
        speaking_started = False
        report_id = None
        emitted_sentences = []
        last_tool_end_sentence_count = 0
        reported_show_report_calls = set()
        report_status_pending = 0
        assistant_utterance_saved = False
        segment_index = 0

        def remember(sentence: dict):
            speech_queue.remember_sentence(
                turn_id, sentence["id"], sentence["text"], segment_index
            )
            emitted_sentences.append(sentence["text"])

        async def save_assistant_utterance(interrupted: bool):
            nonlocal assistant_utterance_saved
            if assistant_utterance_saved:
                return
            assistant_utterance_saved = True
            save_task = asyncio.create_task(
                session_store.add_utterance(
                    thread_id,
                    turn_id,
                    "assistant",
                    "".join(emitted_sentences),
                    interrupted=interrupted,
                )
            )
            await asyncio.shield(save_task)

        def speech_status():
            nonlocal speaking_started
            if not speaking_started:
                speaking_started = True
                return {"type": "status", "state": "speaking", "label": ""}
            return None

        log_bus.emit("flow", "FLOW",
                     f"◈ new query → thread={thread_id!r}, len={len(query)} chars{', +image' if image_data_urls else ''}")

        try:
            yield {"type": "turn", "turn_id": turn_id}
            yield {"type": "status", "state": "thinking", "label": "正在思考"}
            async for event in graph.astream_events(
                {"messages": [build_user_message(query, image_data_urls=image_data_urls)], "context": ""},
                config=config,
                version="v2",
            ):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    if event.get("metadata", {}).get("langgraph_node") != "agent":
                        continue
                    chunk = event["data"]["chunk"]
                    for index, tool_call in enumerate(
                        getattr(chunk, "tool_call_chunks", []) or []
                    ):
                        if not isinstance(tool_call, dict) or tool_call.get("name") != "show_report":
                            continue
                        call_key = (
                            f"{event.get('run_id')}:"
                            f"{tool_call.get('id') or tool_call.get('index', index)}"
                        )
                        if call_key not in reported_show_report_calls:
                            reported_show_report_calls.add(call_key)
                            report_status_pending += 1
                            yield {
                                "type": "status",
                                "state": "working",
                                "label": "正在整理报告",
                            }
                    text = _text_of(chunk.content)
                    if text:
                        for sentence in router.feed(text):
                            status = speech_status()
                            if status:
                                yield status
                            remember(sentence)
                            yield {
                                "type": "speech.sentence",
                                "turn_id": turn_id,
                                **sentence,
                            }
                elif kind == "on_tool_start":
                    for sentence in router.flush_pending():
                        status = speech_status()
                        if status:
                            yield status
                        remember(sentence)
                        yield {
                            "type": "speech.sentence",
                            "turn_id": turn_id,
                            **sentence,
                        }
                    segment_index += 1
                    speech_queue.advance_turn_segment(turn_id)
                    speaking_started = False
                    log_bus.emit("tool", "TOOL", f"► {event['name']}({str(event['data'].get('input', ''))[:80]})")
                    labels = {
                        "tavily_search": "正在搜索",
                        "show_report": "正在整理报告",
                        "read_report": "正在翻看报告",
                    }
                    if event["name"] == "show_report" and report_status_pending:
                        report_status_pending -= 1
                    else:
                        yield {
                            "type": "status",
                            "state": "working",
                            "label": labels.get(event["name"], "正在处理"),
                        }
                    yield {"type": "tool", "name": event["name"], "status": "start"}
                elif kind == "on_tool_end":
                    output = event["data"].get("output")
                    output = getattr(output, "content", output)
                    log_bus.emit("tool", "TOOL", f"✓ {event['name']} → {str(output)[:80]}")
                    last_tool_end_sentence_count = len(emitted_sentences)
                    yield {"type": "tool", "name": event["name"], "status": "end"}
                elif kind == "on_custom_event" and event.get("name") in {"report", "stage"}:
                    yield event["data"]
            state = await graph.aget_state(config)
            summary = router.finish()
            for sentence in summary["sentences"]:
                status = speech_status()
                if status:
                    yield status
                remember(sentence)
                yield {"type": "speech.sentence", "turn_id": turn_id, **sentence}

            if summary["overflow_markdown"]:
                report = await session_store.create_report(
                    thread_id,
                    query[:20],
                    query,
                    summary["overflow_markdown"],
                )
                report_id = report["report_id"]
                yield {
                    "type": "report.open",
                    "report_id": report_id,
                    "title": report["title"],
                    "query": query,
                }
                yield {
                    "type": "report.delta",
                    "report_id": report_id,
                    "markdown": report["markdown"],
                }
                yield {"type": "report.done", "report_id": report_id}
                extra = router.add_sentence(
                    "详细内容我放在屏幕上了。"
                    if not router.spoken_text
                    else "后面的内容我放在屏幕上了。"
                )
                if extra:
                    status = speech_status()
                    if status:
                        yield status
                    remember(extra)
                    yield {"type": "speech.sentence", "turn_id": turn_id, **extra}

            spoken_text = router.spoken_text
            messages = state.values.get("messages", []) if state.values else []
            if report_id and messages and isinstance(messages[-1], AIMessage):
                checkpoint_text = "".join(
                    emitted_sentences[last_tool_end_sentence_count:]
                )
                checkpoint_text += f"\n[完整内容已作为报告 {report_id} 上屏]"
                await graph.aupdate_state(
                    config,
                    {
                        "messages": [
                            AIMessage(content=checkpoint_text, id=messages[-1].id)
                        ]
                    },
                )
            await save_assistant_utterance(interrupted=False)
            log_bus.emit("flow", "FLOW", f"◈ reply ready → {len(spoken_text)} spoken chars")
            yield {"type": "status", "state": "idle", "label": ""}
            yield {"type": "done", "turn_id": turn_id, "spoken_text": spoken_text}
        except asyncio.CancelledError:
            await save_assistant_utterance(interrupted=True)
            raise
        except GeneratorExit:
            await save_assistant_utterance(interrupted=True)
            raise
        except Exception:
            await save_assistant_utterance(interrupted=False)
            raise
        finally:
            if not assistant_utterance_saved:
                await save_assistant_utterance(interrupted=True)
