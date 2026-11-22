import asyncio
import base64
import httpx
import io
import json

import openai
import pytest
import aiosqlite
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from core import media_store, session_store, speech_queue
from core.agent import experts, graph as agent_graph, media_utils


def turn_config(thread_id, turn_id="expert-turn", model_tier="normal"):
    return {
        "configurable": {
            "thread_id": thread_id,
            "turn_id": turn_id,
            "turn_query": "本轮原话",
            "model_tier": model_tier,
        }
    }


async def init_stores(tmp_path):
    path = str(tmp_path / "session.db")
    await session_store.init_store(path)
    await media_store.init_store(path)


async def close_stores():
    await media_store.close_store()
    await session_store.close_store()


class StreamModel:
    model_name = "fake-expert"

    def __init__(self, chunks=(), error=None):
        self.chunks = list(chunks)
        self.error = error
        self.inputs = []
        self.cancelled = False

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def astream(self, messages, config=None):
        self.inputs.append((messages, config))
        for chunk in self.chunks:
            yield AIMessageChunk(content=chunk)
        if self.error:
            raise self.error

    async def ainvoke(self, messages, config=None):
        self.inputs.append((messages, config))
        return AIMessage(content="看到了。")


class SearchModel:
    model_name = "fake-research"

    def __init__(self, answers):
        self.answers = list(answers)
        self.inputs = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages, config=None):
        self.inputs.append((messages, config))
        return self.answers.pop(0)


class ScriptedConversation:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.inputs = []

    def bind_tools(self, tools, **kwargs):
        return self

    async def astream(self, messages, config=None):
        self.inputs.append(messages)
        response = next(self.responses)
        if isinstance(response, str):
            response = AIMessage(content=response)
        if isinstance(response.content, str):
            for character in response.content:
                yield AIMessageChunk(content=character)
        for index, call in enumerate(response.tool_calls):
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": call["name"],
                        "args": json.dumps(call["args"]),
                        "id": call["id"],
                        "index": index,
                    }
                ],
            )


def _tool_call(name, args, call_id="call-1"):
    return {
        "name": name,
        "args": args,
        "id": call_id,
        "type": "tool_call",
    }


async def install_process_graph(tmp_path, monkeypatch, toolset, conversation):
    speech_queue.clear()
    monkeypatch.setattr(agent_graph, "tools", toolset)
    monkeypatch.setattr(agent_graph, "llm_with_tools", conversation)
    monkeypatch.setattr(agent_graph, "fallback_llm_with_tools", None)
    monkeypatch.setattr(agent_graph.memory_manager, "format_context", lambda _: "")
    connection = await aiosqlite.connect(str(tmp_path / "checkpoints.db"))
    checkpointer = AsyncSqliteSaver(connection)
    await checkpointer.setup()
    workflow = StateGraph(agent_graph.AgentState)
    workflow.add_node("agent", agent_graph.agent_node)
    workflow.add_node("tools", ToolNode(toolset, handle_tool_errors=True))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    agent_graph.app_graph = workflow.compile(checkpointer=checkpointer)

    async def initialized_graph():
        return agent_graph.app_graph

    monkeypatch.setattr(agent_graph, "init_graph", initialized_graph)
    return connection


async def close_process_graph(connection):
    await media_store.close_store()
    await session_store.close_store()
    await connection.close()
    agent_graph.app_graph = None


def test_build_tools_and_prompt_guidance_are_role_gated(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    for role in ("REPORT", "RESEARCH", "VISION"):
        monkeypatch.delenv(f"MODEL_{role}", raising=False)
    assert not {
        "write_report",
        "investigate",
        "inspect_image",
    } & {item.name for item in agent_graph.build_tools()}

    monkeypatch.setenv("MODEL_REPORT", "openai:report-model")
    monkeypatch.setenv("MODEL_RESEARCH", "openai:research-model")
    monkeypatch.setenv("MODEL_VISION", "openai:vision-model")
    assert {
        "write_report",
        "investigate",
        "inspect_image",
    } <= {item.name for item in agent_graph.build_tools()}


@pytest.mark.asyncio
async def test_prompt_contains_only_bound_expert_guidance(tmp_path, monkeypatch):
    await init_stores(tmp_path)
    model = StreamModel(["答复。"])
    monkeypatch.setattr(agent_graph.memory_manager, "format_context", lambda _: "")
    monkeypatch.setattr(agent_graph, "llm_with_tools", model)
    monkeypatch.setattr(
        agent_graph,
        "tools",
        agent_graph.base_tools + [experts.write_report],
    )
    try:
        await agent_graph.agent_node(
            {"messages": [HumanMessage(content="问题")], "context": ""},
            turn_config("prompt-test"),
        )
        prompt = model.inputs[-1][0][0].content
        assert "需要详细、严谨或篇幅较长的文档时，调用 write_report" in prompt
        assert "调用 write_report 或 investigate 前" in prompt
        assert "需要调查、核实或综合多个来源时" not in prompt
        assert "需要看清某张已登记图片的细节时" not in prompt
        assert "搜索图片只知道已提供的描述" in prompt
        assert "用户开了深度模式" not in prompt
    finally:
        await close_stores()


@pytest.mark.asyncio
async def test_write_report_streams_complete_lines_rewrites_media_and_updates_db(
    tmp_path, monkeypatch
):
    await init_stores(tmp_path)
    own = await media_store.register_media(
        "report-thread", "image", "upload", "image/png", "owned"
    )
    foreign = await media_store.register_media(
        "other-thread", "image", "upload", "image/png", "foreign"
    )
    model = StreamModel(
        [
            "结论。\n![本会话图片](media:",
            f"{own['media_id']})\n![外部图片](media:{foreign['media_id']})\n",
        ]
    )
    model_requests = []

    def report_model(*args):
        model_requests.append(args)
        return model

    monkeypatch.setattr(experts, "expert_llm", report_model)
    events = []

    async def dispatch(name, data, config=None):
        if name == "report":
            events.append(data)

    monkeypatch.setattr(experts, "adispatch_custom_event", dispatch)
    try:
        result = await experts.write_report.ainvoke(
            {"title": "标题", "brief": "写一份报告"},
            config=turn_config("report-thread"),
        )
        report = (await session_store.list_reports("report-thread"))[0]
        deltas = [
            event["markdown"] for event in events if event["type"] == "report.delta"
        ]
        assert [event["type"] for event in events] == [
            "report.open",
            "report.delta",
            "report.delta",
            "report.done",
        ]
        assert all(delta.endswith("\n") for delta in deltas)
        assert "".join(deltas) == report["markdown"]
        assert f"![本会话图片](media:{own['media_id']})" in report["markdown"]
        assert "外部图片" in report["markdown"]
        assert foreign["media_id"] not in report["markdown"]
        assert "报告已上屏" in result
        assert model_requests == [("report", "normal", 8000)]
        assert model.inputs[0][0][0].content.startswith("你是报告撰写专家")
        assert model.inputs[0][0][1].content.startswith("任务：写一份报告")
    finally:
        await close_stores()


@pytest.mark.asyncio
async def test_report_stream_rewrites_multiline_media_alt_by_ownership(
    tmp_path, monkeypatch
):
    await init_stores(tmp_path)
    own = await media_store.register_media(
        "multiline-report", "image", "upload", "image/png", "owned"
    )
    foreign = await media_store.register_media(
        "other-thread", "image", "upload", "image/png", "foreign"
    )
    model = StreamModel(
        [
            "前文\n![外部\n",
            f"图片](media:{foreign['media_id']})\n![本地\n",
            f"图片](media:{own['media_id']})\n",
        ]
    )
    monkeypatch.setattr(experts, "expert_llm", lambda *args: model)
    events = []

    async def dispatch(name, data, config=None):
        if name == "report":
            events.append(data)

    monkeypatch.setattr(experts, "adispatch_custom_event", dispatch)
    try:
        await experts.write_report.ainvoke(
            {"title": "图片报告", "brief": "检查媒体所有权"},
            config=turn_config("multiline-report"),
        )
        report = (await session_store.list_reports("multiline-report"))[0]
        deltas = [
            event["markdown"] for event in events if event["type"] == "report.delta"
        ]
        streamed = "".join(deltas)

        assert streamed == report["markdown"]
        assert f"(media:{foreign['media_id']})" not in streamed
        assert f"![外部\n图片](media:{foreign['media_id']})" not in streamed
        assert "外部\n图片" in streamed
        assert f"![本地\n图片](media:{own['media_id']})" in streamed
    finally:
        await close_stores()


@pytest.mark.asyncio
async def test_based_on_report_ownership_is_checked_before_budget(tmp_path, monkeypatch):
    await init_stores(tmp_path)
    foreign = await session_store.create_report("owner", "原报告", "问题", "全文内容")
    config = turn_config("not-owner", "ownership-turn")
    monkeypatch.setattr(experts, "expert_llm", lambda *args: StreamModel(["改好了。"]))
    try:
        result = await experts.write_report.ainvoke(
            {
                "title": "修订版",
                "brief": "修订",
                "based_on_report_id": foreign["report_id"],
            },
            config=config,
        )
        assert result == f"找不到报告 {foreign['report_id']}，请确认编号。"
        assert experts._turn_budgets.get("ownership-turn") is None

        original = await session_store.create_report(
            "not-owner", "本地原报告", "问题", "需要保留的全文"
        )
        model = StreamModel(["已修订。"])
        monkeypatch.setattr(experts, "expert_llm", lambda *args: model)
        await experts.write_report.ainvoke(
            {
                "title": "新报告",
                "brief": "请修订",
                "based_on_report_id": original["report_id"],
            },
            config=turn_config("not-owner", "revision-turn"),
        )
        human = model.inputs[0][0][1].content
        assert "需要修订的报告" in human
        assert "需要保留的全文" in human
    finally:
        await close_stores()


@pytest.mark.asyncio
async def test_report_error_emits_incomplete_marker_without_leaking_exception(
    tmp_path, monkeypatch
):
    await init_stores(tmp_path)
    model = StreamModel(["部分内容\n"], RuntimeError("https://secret.invalid/key"))
    monkeypatch.setattr(experts, "expert_llm", lambda *args: model)
    events = []

    async def dispatch(name, data, config=None):
        events.append(data)

    monkeypatch.setattr(experts, "adispatch_custom_event", dispatch)
    try:
        result = await experts.write_report.ainvoke(
            {"title": "中断报告", "brief": "写一份"},
            config=turn_config("error-thread", "error-turn"),
        )
        report = (await session_store.list_reports("error-thread"))[0]
        deltas = [event["markdown"] for event in events if event["type"] == "report.delta"]
        assert report["markdown"] == "".join(deltas)
        assert report["markdown"].endswith("（生成中断）\n")
        assert events[-1]["type"] == "report.done"
        assert "生成中断" in result
        assert "内容不完整" in result
        assert "secret.invalid" not in result
    finally:
        await close_stores()


@pytest.mark.asyncio
async def test_report_finish_reason_length_marks_stream_incomplete(
    tmp_path, monkeypatch
):
    await init_stores(tmp_path)

    class TruncatedModel(StreamModel):
        async def astream(self, messages, config=None):
            self.inputs.append((messages, config))
            yield AIMessageChunk(content="报告正文。\n")
            yield AIMessageChunk(
                content="", response_metadata={"finish_reason": "length"}
            )

    model = TruncatedModel()
    monkeypatch.setattr(experts, "expert_llm", lambda *args: model)
    events = []

    async def dispatch(name, data, config=None):
        if name == "report":
            events.append(data)

    monkeypatch.setattr(experts, "adispatch_custom_event", dispatch)
    try:
        result = await experts.write_report.ainvoke(
            {"title": "截断报告", "brief": "写一份"},
            config=turn_config("length-report"),
        )
        report = (await session_store.list_reports("length-report"))[0]
        deltas = [
            event["markdown"] for event in events if event["type"] == "report.delta"
        ]

        assert "报告正文。\n" in report["markdown"]
        assert report["markdown"] == "".join(deltas)
        assert any("（生成中断）" in delta for delta in deltas)
        assert events[-1]["type"] == "report.done"
        assert "内容不完整" in result
    finally:
        await close_stores()


@pytest.mark.asyncio
async def test_cancelled_report_persists_marker_without_emitting_it(tmp_path, monkeypatch):
    await init_stores(tmp_path)
    started = asyncio.Event()

    class WaitingModel(StreamModel):
        async def astream(self, messages, config=None):
            self.inputs.append((messages, config))
            yield AIMessageChunk(content="首行\n")
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    model = WaitingModel()
    monkeypatch.setattr(experts, "expert_llm", lambda *args: model)
    events = []

    async def dispatch(name, data, config=None):
        events.append(data)

    monkeypatch.setattr(experts, "adispatch_custom_event", dispatch)
    try:
        task = asyncio.create_task(
            experts.write_report.ainvoke(
                {"title": "取消", "brief": "写一份"},
                config=turn_config("cancel-thread", "cancel-turn"),
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        report = (await session_store.list_reports("cancel-thread"))[0]
        emitted = "".join(
            event["markdown"] for event in events if event["type"] == "report.delta"
        )
        assert report["markdown"] == emitted + "\n\n（生成中断）\n"
        assert not any("生成中断" in event.get("markdown", "") for event in events)
        assert not any(event["type"] == "report.done" for event in events)
        assert model.cancelled
    finally:
        await close_stores()


@pytest.mark.asyncio
async def test_parallel_delegations_are_serial_and_capped_at_two(tmp_path, monkeypatch):
    await init_stores(tmp_path)
    active = 0
    maximum = 0
    calls = 0

    class SerialModel(StreamModel):
        async def astream(self, messages, config=None):
            nonlocal active, maximum, calls
            self.inputs.append((messages, config))
            calls += 1
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.02)
            yield AIMessageChunk(content=f"报告 {calls}\n")
            active -= 1

    model = SerialModel()
    monkeypatch.setattr(experts, "expert_llm", lambda *args: model)
    async def dispatch(*args, **kwargs):
        return None
    monkeypatch.setattr(experts, "adispatch_custom_event", dispatch)
    try:
        config = turn_config("budget-thread", "budget-turn")
        results = await asyncio.gather(
            *(
                experts.write_report.ainvoke(
                    {"title": f"报告{i}", "brief": "写一份"},
                    config=config,
                )
                for i in range(3)
            )
        )
        assert calls == 2
        assert maximum == 1
        assert sum(result == experts.DELEGATION_LIMIT_TEXT for result in results) == 1
    finally:
        await close_stores()


@pytest.mark.asyncio
async def test_investigate_uses_three_numbered_searches_without_model_urls(
    tmp_path, monkeypatch
):
    await init_stores(tmp_path)
    searches = SearchModel(
        [
            AIMessage(content="", tool_calls=[_tool_call("tavily_search", {"query": "angle 1"}, "s1")]),
            AIMessage(content="", tool_calls=[_tool_call("tavily_search", {"query": "angle 2"}, "s2")]),
            AIMessage(content="", tool_calls=[_tool_call("tavily_search", {"query": "angle 3"}, "s3")]),
            AIMessage(content="资料已足够"),
        ]
    )
    compose = StreamModel(["结论见 [1]。\n"])
    model_requests = []

    def research_model(role, tier="normal", max_tokens=None):
        model_requests.append((role, tier, max_tokens))
        return compose if max_tokens == 12000 else searches

    monkeypatch.setattr(
        experts, "expert_llm", research_model
    )
    raw_queries = []

    async def search_raw(query, config):
        raw_queries.append((query, config))
        number = len(raw_queries)
        return {
            "results": [
                {
                    "title": f"来源{number}",
                    "url": f"https://source{number}.test/page",
                    "content": f"事实{number} https://source{number}.test/page",
                }
            ]
        }

    monkeypatch.setattr(experts, "search_raw", search_raw)
    events = []

    async def dispatch(name, data, config=None):
        events.append(data)

    monkeypatch.setattr(experts, "adispatch_custom_event", dispatch)
    try:
        result = await experts.investigate.ainvoke(
            {"question": "研究问题"},
            config=turn_config("research-thread", "research-turn"),
        )
        report = (await session_store.list_reports("research-thread"))[0]
        assert len(raw_queries) == 3
        assert all("mercy_expert" in config["tags"] for _, config in raw_queries)
        model_contents = [
            message.content
            for messages, _ in searches.inputs + compose.inputs
            for message in messages
            if isinstance(message.content, str)
        ]
        assert not any("https://source" in content for content in model_contents)
        assert report["markdown"].endswith(
            "\n\n## 来源\n"
            "1. [来源1](https://source1.test/page)\n"
            "2. [来源2](https://source2.test/page)\n"
            "3. [来源3](https://source3.test/page)\n"
        )
        human = compose.inputs[0][0][1].content
        assert "研究问题" in human
        assert "搜索资料：" in human
        assert "source1.test" not in human
        assert "报告已上屏" in result
        assert model_requests == [
            ("research", "normal", None),
            ("research", "normal", 12000),
        ]
    finally:
        await close_stores()


@pytest.mark.asyncio
async def test_investigate_search_cap_is_four(tmp_path, monkeypatch):
    await init_stores(tmp_path)
    call = 0

    class RepeatedSearch(SearchModel):
        async def ainvoke(self, messages, config=None):
            nonlocal call
            self.inputs.append((messages, config))
            call += 1
            return AIMessage(
                content="",
                tool_calls=[
                    _tool_call("tavily_search", {"query": f"q{call}"}, f"c{call}")
                ]
            )

    search_model = RepeatedSearch([])
    compose = StreamModel(["报告完成。\n"])
    model_requests = []

    def max_research_model(role, tier="normal", max_tokens=None):
        model_requests.append((role, tier, max_tokens))
        return compose if max_tokens == 12000 else search_model

    monkeypatch.setattr(experts, "expert_llm", max_research_model)
    raw_count = 0

    async def search_raw(query, config):
        nonlocal raw_count
        raw_count += 1
        return {"answer": f"结果{raw_count}"}

    monkeypatch.setattr(experts, "search_raw", search_raw)
    async def silent_dispatch(*args, **kwargs):
        return None

    monkeypatch.setattr(experts, "adispatch_custom_event", silent_dispatch)
    try:
        result = await experts.investigate.ainvoke(
            {"question": "四次搜索"},
            config=turn_config("cap-thread", "cap-turn", "max"),
        )
        assert raw_count == 4
        assert len(search_model.inputs) == 4
        assert "报告已上屏" in result
        assert model_requests == [
            ("research", "max", None),
            ("research", "max", 12000),
        ]
    finally:
        await close_stores()


@pytest.mark.asyncio
async def test_inspect_image_rejects_foreign_and_video_without_budget(tmp_path, monkeypatch):
    await init_stores(tmp_path)
    video = await media_store.register_media(
        "image-thread", "video", "upload", "video/mp4", "video"
    )
    monkeypatch.setattr(experts, "expert_llm", lambda *args: StreamModel())
    try:
        config = turn_config("image-thread", "image-turn")
        missing = await experts.inspect_image.ainvoke(
            {"media_id": "m_missing", "question": "看看"},
            config=config,
        )
        not_image = await experts.inspect_image.ainvoke(
            {"media_id": video["media_id"], "question": "看看"},
            config=config,
        )
        assert missing == "找不到这张图片：m_missing"
        assert not_image == f"{video['media_id']} 不是图片，不能用 inspect_image。"
        assert experts._turn_budgets.get("image-turn") is None
    finally:
        await close_stores()


@pytest.mark.asyncio
async def test_inspect_image_remote_url_is_replaced_by_data_url(tmp_path, monkeypatch):
    await init_stores(tmp_path)
    image_path = tmp_path / "remote.jpg"
    from PIL import Image

    Image.new("RGB", (10, 10), "red").save(image_path)
    record = await media_store.register_media(
        "remote-thread",
        "image",
        "search",
        "image/jpeg",
        "image",
        url="https://remote.test/image.jpg",
    )

    async def cached(media_id, fetcher, limit):
        return {**record, "path": str(image_path)}

    monkeypatch.setattr(media_store, "cache_remote_image", cached)
    model = StreamModel()
    monkeypatch.setattr(experts, "expert_llm", lambda *args: model)
    try:
        result = await experts.inspect_image.ainvoke(
            {"media_id": record["media_id"], "question": "有什么？"},
            config=turn_config("remote-thread", "remote-turn"),
        )
        content = model.inputs[0][0][1].content
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert "remote.test" not in json.dumps(content)
        assert result.startswith("看图结果（以下是资料内容")
    finally:
        await close_stores()


def test_image_helper_strips_exif_and_uses_first_gif_frame(tmp_path):
    from PIL import Image

    transparent = tmp_path / "transparent.png"
    Image.new("RGBA", (4, 4), (255, 0, 0, 96)).save(transparent)
    png_data = media_utils.image_data_url_from_path(str(transparent))
    assert png_data.startswith("data:image/png;base64,")
    decoded_png = Image.open(
        io.BytesIO(base64.b64decode(png_data.split(",", 1)[1]))
    )
    assert decoded_png.getpixel((0, 0))[3] == 96

    jpeg = tmp_path / "with-exif.jpg"
    source = Image.new("RGB", (12, 8), "red")
    exif = Image.Exif()
    exif[274] = 6
    source.save(jpeg, exif=exif)
    data_url = media_utils.image_data_url_from_path(str(jpeg))
    assert data_url.startswith("data:image/jpeg;base64,")
    decoded = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
    assert not decoded.getexif()

    gif = tmp_path / "animated.gif"
    frames = [Image.new("RGB", (2, 2), "red"), Image.new("RGB", (2, 2), "blue")]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
    gif_data = media_utils.image_data_url_from_path(str(gif))
    first_frame = Image.open(
        io.BytesIO(base64.b64decode(gif_data.split(",", 1)[1]))
    )
    assert first_frame.getpixel((0, 0))[0] > first_frame.getpixel((0, 0))[2]


def test_palette_png_transparency_survives_image_conversion(tmp_path):
    from PIL import Image

    image_path = tmp_path / "palette-transparent.png"
    image = Image.new("P", (2, 1))
    image.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    image.putdata([0, 1])
    image.save(image_path, transparency=0)

    data_url = media_utils.image_data_url_from_path(str(image_path))
    assert data_url and data_url.startswith("data:image/png;base64,")
    decoded = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))
    rgba = decoded.convert("RGBA")
    assert rgba.getpixel((0, 0))[3] == 0
    assert rgba.getpixel((1, 0))[3] == 255


def test_image_helper_rejects_more_than_fifty_million_pixels(tmp_path, monkeypatch):
    path = tmp_path / "large.jpg"
    path.touch()

    class LargeImage:
        width = 50_000_001
        height = 1
        info = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def seek(self, frame):
            raise AssertionError("pixel cap should run before decoding")

    monkeypatch.setattr(media_utils.Image, "open", lambda _: LargeImage())
    with pytest.raises(media_utils.ImageTooLarge):
        media_utils.image_data_url_from_path(str(path))


@pytest.mark.asyncio
async def test_process_query_hides_expert_stream_and_advances_one_segment(
    tmp_path, monkeypatch
):
    await init_stores(tmp_path)
    search_queries = []
    search_model = SearchModel(
        [
            AIMessage(content="", tool_calls=[_tool_call("tavily_search", {"query": "q1"}, "s1")]),
            AIMessage(content="", tool_calls=[_tool_call("tavily_search", {"query": "q2"}, "s2")]),
            AIMessage(content="", tool_calls=[_tool_call("tavily_search", {"query": "q3"}, "s3")]),
            AIMessage(content="资料已足够"),
        ]
    )
    compose = StreamModel(["专家报告内容不应朗读。\n"])

    async def raw(query, config):
        search_queries.append(query)
        number = len(search_queries)
        return {
            "results": [
                {
                    "title": f"来源{number}",
                    "url": f"https://source{number}.test",
                    "content": f"内容{number}",
                }
            ]
        }

    monkeypatch.setattr(experts, "search_raw", raw)
    monkeypatch.setattr(
        experts,
        "expert_llm",
        lambda role, tier="normal", max_tokens=None: (
            compose if max_tokens == 12000 else search_model
        ),
    )
    conversation = ScriptedConversation(
        [
            AIMessage(
                content="我来调查。",
                tool_calls=[_tool_call("investigate", {"question": "研究问题"})],
            ),
            AIMessage(content="报告已经完成。"),
        ]
    )
    toolset = agent_graph.base_tools + [experts.investigate]
    connection = await install_process_graph(
        tmp_path, monkeypatch, toolset, conversation
    )
    try:
        events = [
            event
            async for event in agent_graph.process_query("研究一下", "expert-process")
        ]
        turn_id = events[0]["turn_id"]
        assert len(search_queries) == 3
        assert speech_queue.get_turn_segment(turn_id) == 1
        assert not any(
            event.get("name") == "tavily_search" for event in events if event["type"] == "tool"
        )
        assert not any(
            "专家报告内容" in event["text"]
            for event in events
            if event["type"] == "speech.sentence"
        )
        report = (await session_store.list_reports("expert-process"))[0]
        assert "## 来源" in report["markdown"]
        assert "https://source1.test" in report["markdown"]
    finally:
        await close_process_graph(connection)


@pytest.mark.asyncio
async def test_three_report_tool_calls_are_serial_and_third_is_limited(
    tmp_path, monkeypatch
):
    await init_stores(tmp_path)
    active = 0
    maximum = 0
    calls = 0

    class SerialModel(StreamModel):
        async def astream(self, messages, config=None):
            nonlocal active, maximum, calls
            self.inputs.append((messages, config))
            calls += 1
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.02)
            yield AIMessageChunk(content=f"报告 {calls}\n")
            active -= 1

    model = SerialModel()
    monkeypatch.setattr(experts, "expert_llm", lambda *args: model)
    conversation = ScriptedConversation(
        [
            AIMessage(
                content="我来整理。",
                tool_calls=[
                    _tool_call("write_report", {"title": f"报告{i}", "brief": "写一份"}, f"write-{i}")
                    for i in range(3)
                ],
            ),
            AIMessage(content="两份报告已经完成。"),
        ]
    )
    toolset = agent_graph.base_tools + [experts.write_report]
    connection = await install_process_graph(
        tmp_path, monkeypatch, toolset, conversation
    )
    try:
        events = [
            event
            async for event in agent_graph.process_query(
                "写三份报告", "three-report-calls"
            )
        ]
        state = await agent_graph.app_graph.aget_state(
            {"configurable": {"thread_id": "three-report-calls"}}
        )
        tool_messages = [
            message
            for message in state.values["messages"]
            if isinstance(message, ToolMessage)
        ]
        assert calls == 2
        assert maximum == 1
        assert len(await session_store.list_reports("three-report-calls")) == 2
        assert sum(
            message.content == experts.DELEGATION_LIMIT_TEXT
            for message in tool_messages
        ) == 1
        assert events[-1]["type"] == "done"
    finally:
        await close_process_graph(connection)


@pytest.mark.asyncio
async def test_investigate_disconnect_persists_marker_and_closes_pending_tool_call(
    tmp_path, monkeypatch
):
    await init_stores(tmp_path)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class WaitingComposer(StreamModel):
        async def astream(self, messages, config=None):
            self.inputs.append((messages, config))
            yield AIMessageChunk(content="报告首行\n")
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                cancelled.set()
                raise

    search_model = SearchModel([AIMessage(content="资料已足够")])
    composer = WaitingComposer()
    monkeypatch.setattr(
        experts,
        "expert_llm",
        lambda role, tier="normal", max_tokens=None: (
            composer if max_tokens == 12000 else search_model
        ),
    )
    conversation = ScriptedConversation(
        [
            AIMessage(
                content="我来调查。",
                tool_calls=[
                    _tool_call(
                        "investigate",
                        {"question": "调查问题"},
                        "investigate-disconnect",
                    )
                ],
            ),
            AIMessage(content="我接着说。"),
        ]
    )
    toolset = agent_graph.base_tools + [experts.investigate]
    connection = await install_process_graph(
        tmp_path, monkeypatch, toolset, conversation
    )
    thread_id = "investigate-disconnect"
    generator = agent_graph.process_query("调查问题", thread_id)
    events = []
    try:
        async for event in generator:
            events.append(event)
            if event["type"] == "report.delta":
                break
        await asyncio.wait_for(started.wait(), timeout=1)
        await generator.aclose()
        await asyncio.wait_for(cancelled.wait(), timeout=1)
        report = (await session_store.list_reports(thread_id))[0]
        emitted = "".join(
            event["markdown"] for event in events if event["type"] == "report.delta"
        )
        assert composer.cancelled
        assert report["markdown"] == emitted + "\n\n（生成中断）\n"

        next_events = [
            event
            async for event in agent_graph.process_query("继续", thread_id)
        ]
        assert next_events[-1]["type"] == "done"
        state = await agent_graph.app_graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        messages = state.values["messages"]
        assert any(
            isinstance(message, ToolMessage)
            and message.tool_call_id == "investigate-disconnect"
            and message.content == "Tool call interrupted."
            for message in messages
        )
    finally:
        await generator.aclose()
        await close_process_graph(connection)


@pytest.mark.asyncio
async def test_image_turn_uses_the_conversation_model(tmp_path, monkeypatch):
    await init_stores(tmp_path)
    model = StreamModel(["收到。"])
    monkeypatch.setattr(agent_graph.memory_manager, "format_context", lambda _: "")
    monkeypatch.setattr(agent_graph, "llm_with_tools", model)
    monkeypatch.setattr(agent_graph, "tools", agent_graph.base_tools)
    image_message = HumanMessage(
        content=[
            {"type": "text", "text": "描述这张图"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AA=="},
            },
        ]
    )
    try:
        await agent_graph.agent_node(
            {"messages": [image_message], "context": ""},
            turn_config("image-routing"),
        )
        assert any(
            block.get("type") == "image_url"
            for message in model.inputs[0][0]
            if isinstance(message, HumanMessage)
            and isinstance(message.content, list)
            for block in message.content
        )
    finally:
        await close_stores()


def rate_limit_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return openai.RateLimitError("rate limited", response=response, body=None)


def test_expert_llm_caches_fallback_model(monkeypatch):
    monkeypatch.setattr(experts, "_expert_models", {})
    spec = experts.ModelSpec("openai", "fallback-model", None)
    calls = []
    model = object()

    monkeypatch.setattr(
        experts, "get_model_spec", lambda role, tier, fallback=False: spec
    )

    def get_llm(role, tier, fallback=False, max_tokens=None):
        calls.append((role, tier, fallback, max_tokens))
        return model

    monkeypatch.setattr(experts, "get_llm", get_llm)

    assert experts.expert_llm("report", "normal", 8000, fallback=True) is model
    assert experts.expert_llm("report", "normal", 8000, fallback=True) is model
    assert calls == [("report", "normal", True, 8000)]


@pytest.mark.asyncio
async def test_report_uses_transient_fallback_only_before_first_delta(
    tmp_path, monkeypatch
):
    await init_stores(tmp_path)
    primary = StreamModel([], rate_limit_error())
    fallback_model = StreamModel(["备用报告。\n"])
    fallback_calls = []

    def expert_model(role, tier, max_tokens=None, *, fallback=False):
        fallback_calls.append(fallback)
        return fallback_model if fallback else primary

    monkeypatch.setattr(experts, "expert_llm", expert_model)
    events = []

    async def dispatch(name, data, config=None):
        if name == "report":
            events.append(data)

    monkeypatch.setattr(experts, "adispatch_custom_event", dispatch)
    try:
        result = await experts.write_report.ainvoke(
            {"title": "备用报告", "brief": "写一份"},
            config=turn_config("report-fallback-before-output"),
        )
        report = (await session_store.list_reports(
            "report-fallback-before-output"
        ))[0]
        assert report["markdown"] == "备用报告。\n"
        assert "内容不完整" not in result
        assert [event["type"] for event in events] == [
            "report.open",
            "report.delta",
            "report.done",
        ]
        assert fallback_calls == [False, True]
    finally:
        await close_stores()


@pytest.mark.asyncio
async def test_report_does_not_fallback_after_first_delta(tmp_path, monkeypatch):
    await init_stores(tmp_path)
    primary = StreamModel(["部分内容。\n"], rate_limit_error())
    fallback_model = StreamModel(["不应使用。\n"])
    fallback_calls = []

    def expert_model(role, tier, max_tokens=None, *, fallback=False):
        fallback_calls.append(fallback)
        return fallback_model if fallback else primary

    monkeypatch.setattr(experts, "expert_llm", expert_model)
    events = []

    async def dispatch(name, data, config=None):
        if name == "report":
            events.append(data)

    monkeypatch.setattr(experts, "adispatch_custom_event", dispatch)
    try:
        result = await experts.write_report.ainvoke(
            {"title": "不完整报告", "brief": "写一份"},
            config=turn_config("report-fallback-after-output"),
        )
        report = (await session_store.list_reports(
            "report-fallback-after-output"
        ))[0]
        assert fallback_calls == [False]
        assert report["markdown"].startswith("部分内容。\n")
        assert report["markdown"].endswith("（生成中断）\n")
        assert "内容不完整" in result
        assert events[-1]["type"] == "report.done"
    finally:
        await close_stores()


@pytest.mark.asyncio
async def test_investigate_search_uses_transient_fallback(tmp_path, monkeypatch):
    await init_stores(tmp_path)

    class FailingSearchModel(SearchModel):
        async def ainvoke(self, messages, config=None):
            self.inputs.append((messages, config))
            raise rate_limit_error()

    primary = FailingSearchModel([])
    fallback_search = SearchModel([AIMessage(content="搜索结束")])
    report_model = StreamModel(["报告正文。\n"])

    def expert_model(role, tier, max_tokens=None, *, fallback=False):
        if max_tokens is not None:
            return report_model
        return fallback_search if fallback else primary

    monkeypatch.setattr(experts, "expert_llm", expert_model)
    try:
        result = await experts.investigate.ainvoke(
            {"question": "研究一个主题"},
            config=turn_config("investigate-fallback"),
        )
        assert "报告已上屏" in result
        assert len(primary.inputs) == 1
        assert len(fallback_search.inputs) == 1
    finally:
        experts.clear_turn_budget("expert-turn")
        await close_stores()


@pytest.mark.asyncio
async def test_inspect_image_uses_transient_fallback(tmp_path, monkeypatch):
    from PIL import Image

    image_path = tmp_path / "vision.png"
    Image.new("RGB", (2, 2), "red").save(image_path)

    async def get_media(media_id, thread_id):
        return {
            "media_id": media_id,
            "kind": "image",
            "origin": "upload",
            "path": str(image_path),
        }

    class FailingVisionModel:
        model_name = "primary-vision"

        def __init__(self):
            self.inputs = []

        async def ainvoke(self, messages, config=None):
            self.inputs.append((messages, config))
            raise rate_limit_error()

    primary = FailingVisionModel()
    fallback_model = StreamModel([])

    def expert_model(role, tier, max_tokens=None, *, fallback=False):
        return fallback_model if fallback else primary

    monkeypatch.setattr(experts.media_store, "get_media", get_media)
    monkeypatch.setattr(experts, "expert_llm", expert_model)
    try:
        result = await experts.inspect_image.ainvoke(
            {"media_id": "m_vision", "question": "图片里是什么？"},
            config=turn_config("vision-fallback"),
        )
        assert "看到了" in result
        assert len(primary.inputs) == 1
        assert len(fallback_model.inputs) == 1
    finally:
        experts.clear_turn_budget("expert-turn")


@pytest.mark.asyncio
async def test_conversation_fallback_runs_only_before_any_output(tmp_path, monkeypatch):
    await init_stores(tmp_path)
    primary = StreamModel([], rate_limit_error())
    fallback = StreamModel(["备用回答。"])
    monkeypatch.setattr(agent_graph.memory_manager, "format_context", lambda _: "")
    monkeypatch.setattr(agent_graph, "llm_with_tools", primary)
    monkeypatch.setattr(agent_graph, "fallback_llm_with_tools", fallback)
    try:
        result = await agent_graph.agent_node(
            {"messages": [HumanMessage(content="问题")], "context": ""},
            turn_config("fallback-thread"),
        )
        assert result["messages"][-1].content == "备用回答。"
        assert len(primary.inputs) == 1
        assert len(fallback.inputs) == 1
    finally:
        await close_stores()


@pytest.mark.asyncio
async def test_conversation_does_not_fallback_after_output_or_for_bad_request(
    tmp_path, monkeypatch
):
    await init_stores(tmp_path)
    primary = StreamModel(["已经说出。"], rate_limit_error())
    fallback = StreamModel(["不应使用。"])
    monkeypatch.setattr(agent_graph.memory_manager, "format_context", lambda _: "")
    monkeypatch.setattr(agent_graph, "llm_with_tools", primary)
    monkeypatch.setattr(agent_graph, "fallback_llm_with_tools", fallback)
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(400, request=request)
    bad_request_model = StreamModel(
        [], openai.BadRequestError("bad request", response=response, body=None)
    )

    class ToolThenFailure(StreamModel):
        async def astream(self, messages, config=None):
            self.inputs.append((messages, config))
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "show_report",
                        "args": '{"title":"t","markdown":"m"}',
                        "id": "partial-tool",
                        "index": 0,
                    }
                ],
            )
            raise rate_limit_error()

    try:
        with pytest.raises(openai.RateLimitError):
            await agent_graph.agent_node(
                {"messages": [HumanMessage(content="问题")], "context": ""},
                turn_config("no-fallback-after-output"),
            )
        assert not fallback.inputs
        monkeypatch.setattr(agent_graph, "llm_with_tools", bad_request_model)
        with pytest.raises(openai.BadRequestError):
            await agent_graph.agent_node(
                {"messages": [HumanMessage(content="问题")], "context": ""},
                turn_config("no-fallback-bad-request"),
            )
        assert not fallback.inputs
        monkeypatch.setattr(agent_graph, "llm_with_tools", ToolThenFailure())
        with pytest.raises(openai.RateLimitError):
            await agent_graph.agent_node(
                {"messages": [HumanMessage(content="问题")], "context": ""},
                turn_config("no-fallback-after-tool-call"),
            )
        assert not fallback.inputs
    finally:
        await close_stores()
