import os
import time
import asyncio
import secrets
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables FIRST — before any module reads them
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BACKEND_DIR, ".env"))
from typing import TypedDict, Annotated, AsyncIterator, Optional
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
from core import session_store, speech_queue
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
web_search = TavilySearch(max_results=3, description="Search the web for current events, facts, or any information you do not know.")

@tool
def create_local_directory(dir_name: str) -> str:
    """Create a new folder/directory on the local computer. 
    Provide the name of the folder you want to create. It will be created in the user's designated workspace.
    """
    raw_base = os.environ.get("AURA_WORKSPACE_DIR", "~/Desktop/AuraWorkspace")
    base_dir = os.path.expanduser(raw_base)
    os.makedirs(base_dir, exist_ok=True)
    
    target_path = os.path.join(base_dir, dir_name)
    log_bus.emit("tool", "TOOL", f"create_local_directory: {target_path}")
    try:
        os.makedirs(target_path, exist_ok=True)
        return f"Successfully created directory at: {target_path}"
    except Exception as e:
        return f"Failed to create directory: {str(e)}"

@tool
async def show_report(title: str, markdown: str, config: RunnableConfig) -> str:
    """Put a document on screen. ANY table, code, list of links, step-by-step document, or content longer than about three sentences MUST use this tool and never be included in the spoken reply."""
    configurable = config.get("configurable", {})
    thread_id = configurable.get("thread_id", "default_user")
    query = configurable.get("turn_query", "")
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
    return (
        f"已上屏，报告编号 {report_id}。用户现在能看到它，"
        "口头只需概括要点并询问意见。"
    )


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
    web_search,
    create_local_directory,
    show_report,
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

def build_user_message(query: str, image_base64: Optional[str] = None) -> HumanMessage:
    """Build a HumanMessage; images become native image_url content blocks."""
    if not image_base64:
        return HumanMessage(content=query)
    url = image_base64 if image_base64.startswith("data:") else f"data:image/jpeg;base64,{image_base64}"
    return HumanMessage(content=[
        {"type": "text", "text": query},
        {"type": "image_url", "image_url": {"url": url}},
    ])

_thread_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

async def _close_pending_tool_calls(graph, config):
    """If a previous run stopped between the agent and tools nodes (e.g. client
    disconnect), answer the dangling tool calls so OpenAI accepts the history."""
    state = await graph.aget_state(config)
    messages = state.values.get("messages", []) if state.values else []
    if not state.next or not messages or not isinstance(messages[-1], AIMessage):
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
) -> AsyncIterator[dict]:
    """Run one turn and stream status, spoken sentences, reports, tools and completion."""
    # Serialize turns per thread so concurrent requests don't overwrite each other's checkpoints
    async with _thread_locks[thread_id]:
        graph = await init_graph()
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
                     f"◈ new query → thread={thread_id!r}, len={len(query)} chars{', +image' if image_base64 else ''}")

        try:
            yield {"type": "turn", "turn_id": turn_id}
            yield {"type": "status", "state": "thinking", "label": "正在思考"}
            async for event in graph.astream_events(
                {"messages": [build_user_message(query, image_base64)], "context": ""},
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
                    segment_index += 1
                    speech_queue.advance_turn_segment(turn_id)
                    yield {"type": "tool", "name": event["name"], "status": "end"}
                elif kind == "on_custom_event" and event.get("name") == "report":
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
