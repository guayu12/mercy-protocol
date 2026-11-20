import os
import time
import asyncio
from collections import defaultdict
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
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from core.memory.memory_manager import MemoryManager
from core import log_bus  # ← real-time log broadcaster

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

tools = [check_weather, get_current_time, save_memory, web_search, create_local_directory]

# 4. Setup LLMs (OpenAI official API; vision model handles turns with images)
_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
_api_key = os.environ.get("OPENAI_API_KEY", "dummy")
_model_name = os.environ.get("DEFAULT_MODEL", "gpt-4o-mini")
_vision_model_name = os.environ.get("OPENAI_VISION_MODEL") or _model_name

def _make_llm(model: str):
    return ChatOpenAI(
        model=model,
        api_key=_api_key,
        base_url=_base_url,
        streaming=True,
        stream_usage=True,
    ).bind_tools(tools)

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
    
    sys_prompt = SystemMessage(content=f"""
You are Aura, a highly capable personal AI assistant running locally.
You can execute tasks, remember user preferences, and help plan long-term goals.
You can see images the user attaches directly.
Always be concise, professional, and helpful.

{past_contexts}
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

async def process_query(query: str, thread_id: str, image_base64: Optional[str] = None) -> AsyncIterator[dict]:
    """Run the graph for one turn, yielding stream events:
    {"type": "token", "content": str} — incremental reply text
    {"type": "tool", "name": str, "status": "start"|"end"}
    {"type": "done", "reply": str} — final full reply
    """
    # Serialize turns per thread so concurrent requests don't overwrite each other's checkpoints
    async with _thread_locks[thread_id]:
        graph = await init_graph()
        config = {"configurable": {"thread_id": thread_id}}
        await _close_pending_tool_calls(graph, config)
        log_bus.emit("flow", "FLOW",
                     f"◈ new query → thread={thread_id!r}, len={len(query)} chars{', +image' if image_base64 else ''}")

        async for event in graph.astream_events(
            {"messages": [build_user_message(query, image_base64)], "context": ""},
            config=config,
            version="v2",
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                if event.get("metadata", {}).get("langgraph_node") != "agent":
                    continue
                text = _text_of(event["data"]["chunk"].content)
                if text:
                    yield {"type": "token", "content": text}
            elif kind == "on_tool_start":
                log_bus.emit("tool", "TOOL", f"► {event['name']}({str(event['data'].get('input', ''))[:80]})")
                yield {"type": "tool", "name": event["name"], "status": "start"}
            elif kind == "on_tool_end":
                output = event["data"].get("output")
                output = getattr(output, "content", output)
                log_bus.emit("tool", "TOOL", f"✓ {event['name']} → {str(output)[:80]}")
                yield {"type": "tool", "name": event["name"], "status": "end"}

        state = await graph.aget_state(config)
        reply = _text_of(state.values["messages"][-1].content)
        log_bus.emit("flow", "FLOW", f"◈ reply ready → {len(reply)} chars")
        yield {"type": "done", "reply": reply}
