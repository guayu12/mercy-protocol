import os
import time
from dotenv import load_dotenv

# Load environment variables FIRST — before any module reads them
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BACKEND_DIR, ".env"))
from typing import TypedDict, Annotated, Sequence, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver # using SQLite for production
import operator

from core.memory.memory_manager import MemoryManager
from core import log_bus  # ← real-time log broadcaster

# 1. Initialize Memory
memory_manager = MemoryManager(os.path.join(BACKEND_DIR, "data", "chroma_db"))

# 2. Define the State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: str

from langchain_community.tools.tavily_search import TavilySearchResults

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
def save_memory(fact: str) -> str:
    """Save an important fact about the user or instructions to long-term memory. USE EXACTLY when user says 'remember' or explicitly tells you a preference."""
    log_bus.emit("tool", "TOOL", f"save_memory: {fact[:80]}")
    memory_manager.add_memory(fact)
    return f"Saved to permanent memory: {fact}"

# Web Search Tool (requires TAVILY_API_KEY in .env)
web_search = TavilySearchResults(max_results=3, description="Search the web for current events, facts, or any information you do not know.")

@tool
def analyze_image(prompt: str, base64_image: str) -> str:
    """USE THIS TOOL ONLY WHEN THE USER UPLOADS OR ASKS ABOUT AN IMAGE (base64 encoded). 
    Pass the user's question as 'prompt' and the raw base64 string as 'base64_image'.
    This tool forwards the image to the dedicated Vision Model (Dual-Endpoint Architecture) for analysis.
    """
    vision_model_name = os.environ.get("OPENAI_VISION_MODEL")
    if not vision_model_name:
        return "Error: Vision model endpoint not configured in .env"
    
    log_bus.emit("tool", "TOOL", f"analyze_image — forwarding to vision model: {vision_model_name}")

    vision_llm = ChatOpenAI(
        model=vision_model_name,
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    
    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    )
    
    t0 = time.time()
    try:
        response = vision_llm.invoke([msg])
        latency = round((time.time() - t0) * 1000)
        usage = getattr(response, "usage_metadata", None)
        log_bus.emit("llm", "LLM", f"vision_model response — {latency}ms",
                     {"model": vision_model_name, "latency_ms": latency,
                      "input_tokens": usage.get("input_tokens") if usage else "?",
                      "output_tokens": usage.get("output_tokens") if usage else "?"})
        return response.content
    except Exception as e:
        log_bus.emit("error", "ERROR", f"Vision model error: {str(e)[:120]}")
        return f"Vision Model Error: {str(e)}"

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

tools = [check_weather, get_current_time, save_memory, web_search, analyze_image, create_local_directory]

# (load_dotenv already called at top of file)

# 4. Setup LLM
_model_name = os.environ.get("DEFAULT_MODEL", "gpt-4o")
llm = ChatOpenAI(
    model=_model_name,
    api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

llm_with_tools = llm.bind_tools(tools)

# 5. Define Graph Nodes
def agent_node(state: AgentState):
    """The thinking node of the agent."""
    messages = state["messages"]
    
    user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    latest_query = user_msgs[-1].content if user_msgs else ""
    
    past_contexts = memory_manager.format_context(latest_query)
    
    sys_prompt = SystemMessage(content=f"""
You are Aura, a highly capable personal AI assistant running locally.
You can execute tasks, remember user preferences, and help plan long-term goals.
Always be concise, professional, and helpful.

{past_contexts}
""")
    full_messages = [sys_prompt] + list(messages)
    input_tokens_est = sum(len(str(m.content)) // 4 for m in full_messages)

    log_bus.emit("flow", "FLOW", f"agent_node → invoking LLM ({len(full_messages)} msgs, ~{input_tokens_est} tokens est.)")

    # Retry with exponential backoff for transient API errors
    last_err = None
    for attempt in range(3):
        t0 = time.time()
        try:
            response = llm_with_tools.invoke(full_messages)
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
                         {"model": _model_name, "latency_ms": latency,
                          "input_tokens": in_tok, "output_tokens": out_tok,
                          "tool_calls": [tc["name"] for tc in tool_calls]})
            return {"messages": [response]}
        except Exception as e:
            last_err = e
            err_str = str(e)
            log_bus.emit("warn", "RETRY", f"attempt {attempt+1}/3 — {err_str[:100]}")
            if "NoAvailableModel" in err_str or "429" in err_str or "503" in err_str:
                wait = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait)
                continue
            raise  # Non-transient error, propagate immediately
    raise last_err  # All retries exhausted

def tool_node(state: AgentState):
    """The execution node that runs tools requested by the LLM."""
    messages = state["messages"]
    last_message = messages[-1]
    
    tool_responses = []
    tool_map = {t.name: t for t in tools}
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        log_bus.emit("tool", "TOOL", f"► {tool_name}({', '.join(f'{k}={str(v)[:40]}' for k, v in tool_args.items())})")

        if tool_name in tool_map:
            t0 = time.time()
            try:
                result = tool_map[tool_name].invoke(tool_args)
                latency = round((time.time() - t0) * 1000)
                tool_msg = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                    name=tool_name
                )
                tool_responses.append(tool_msg)
                log_bus.emit("tool", "TOOL", f"✓ {tool_name} done {latency}ms → {str(result)[:80]}")
            except Exception as e:
                log_bus.emit("error", "ERROR", f"✗ {tool_name} failed: {str(e)[:100]}")
                tool_responses.append(
                    ToolMessage(content=f"Error: {e}", tool_call_id=tool_call["id"], name=tool_name)
                )
                
    return {"messages": tool_responses}

# 6. Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if not getattr(last_message, "tool_calls", None):
        return "end"
    return "tools"

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")

import sqlite3

# Compile with a checkpointer for short term thread memory
conn = sqlite3.connect(os.path.join(BACKEND_DIR, "aura_checkpoints.db"), check_same_thread=False)
checkpointer = SqliteSaver(conn)
checkpointer.setup()
app_graph = workflow.compile(checkpointer=checkpointer)

def process_query(query: str, thread_id: str = "default_user") -> str:
    """Run the graph, maintaining conversation state by thread_id."""
    config = {"configurable": {"thread_id": thread_id}}
    log_bus.emit("flow", "FLOW", f"◈ new query → thread={thread_id!r}, len={len(query)} chars")
    final_state = app_graph.invoke(
        {"messages": [HumanMessage(content=query)], "context": ""},
        config=config
    )
    reply = final_state["messages"][-1].content
    log_bus.emit("flow", "FLOW", f"◈ reply ready → {len(reply)} chars")
    return reply
