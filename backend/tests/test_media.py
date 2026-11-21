import asyncio
from pathlib import Path

import httpcore
import httpx
import pytest
from langchain_core.messages import HumanMessage

from core import media_store, session_store
from core.agent import graph
from core.safe_fetch import (
    SafeHTTPXTransport,
    SafeMediaFetcher,
    UnsafeAddress,
    ValidatingNetworkBackend,
)
import main


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "::1",
        "fc00::1",
        "::ffff:127.0.0.1",
        "64:ff9b::7f00:1",
        "64:ff9b:1:7f00:0:100::",
    ],
)
async def test_safe_fetch_rejects_private_resolved_addresses(address):
    async def resolver(host, port):
        return [address]

    backend = ValidatingNetworkBackend(resolver=resolver)
    with pytest.raises(UnsafeAddress):
        await backend.connect_tcp("media.test", 443)


@pytest.mark.asyncio
async def test_safe_fetch_rejects_mixed_public_and_private_dns():
    async def resolver(host, port):
        return ["8.8.8.8", "10.0.0.1"]

    with pytest.raises(UnsafeAddress):
        await ValidatingNetworkBackend(resolver=resolver).connect_tcp("media.test", 443)


@pytest.mark.asyncio
async def test_safe_fetch_maps_httpcore_errors_and_media_endpoint_returns_502(
    tmp_path, monkeypatch
):
    class FailingPool:
        async def handle_async_request(self, request):
            raise httpcore.ConnectError("upstream unavailable")

        async def aclose(self):
            return None

    async def resolver(host, port):
        return ["8.8.8.8"]

    def transport_factory():
        transport = SafeHTTPXTransport(resolver=resolver)
        transport._pool = FailingPool()
        return transport

    transport = transport_factory()
    with pytest.raises(httpx.ConnectError):
        await transport.handle_async_request(
            httpx.Request("GET", "https://media.test/image.jpg")
        )
    await transport.aclose()

    await media_store.init_store(str(tmp_path / "session.db"))
    await media_store.register_media(
        "upstream-failure",
        "image",
        "link",
        "image/jpeg",
        "image.jpg",
        url="https://media.test/image.jpg",
    )
    monkeypatch.setattr(
        main,
        "safe_media_fetcher",
        SafeMediaFetcher(resolver=resolver, transport_factory=transport_factory),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.get("/api/media/" + (await media_store.list_media("upstream-failure"))[0]["media_id"])
    assert response.status_code == 502
    await media_store.close_store()


@pytest.mark.asyncio
async def test_safe_fetch_validates_redirect_target_before_request():
    requested = []

    async def resolver(host, port):
        return ["8.8.8.8" if host == "public.test" else "10.0.0.1"]

    def handler(request):
        requested.append(request.url.host)
        return httpx.Response(
            302,
            headers={"location": "http://private.test/image.jpg"},
            request=request,
        )

    fetcher = SafeMediaFetcher(
        resolver=resolver,
        transport_factory=lambda: httpx.MockTransport(handler),
    )
    with pytest.raises(UnsafeAddress):
        await fetcher.probe("https://public.test/image.jpg")
    assert requested == ["public.test"]


@pytest.mark.asyncio
async def test_safe_fetch_sets_descriptive_user_agent():
    user_agents = []

    async def resolver(host, port):
        return ["8.8.8.8"]

    def handler(request):
        user_agents.append(request.headers.get("user-agent"))
        return httpx.Response(
            200,
            headers={"content-type": "image/jpeg", "content-length": "4"},
            request=request,
        )

    fetcher = SafeMediaFetcher(
        resolver=resolver,
        transport_factory=lambda: httpx.MockTransport(handler),
    )
    await fetcher.probe("https://media.test/image.jpg")
    assert user_agents == [
        "MercyProtocol/1.0 (+https://github.com/guayu12/mercy-protocol)"
    ]


@pytest.mark.asyncio
async def test_safe_fetch_connects_to_resolved_ip_and_keeps_tls_hostname():
    class RecordingStream:
        def __init__(self):
            self.response = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
            self.server_hostname = None

        async def read(self, max_bytes, timeout=None):
            data, self.response = self.response[:max_bytes], self.response[max_bytes:]
            return data

        async def write(self, buffer, timeout=None):
            return None

        async def aclose(self):
            return None

        async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            self.server_hostname = server_hostname
            return self

        def get_extra_info(self, info):
            return None

    class RecordingBackend:
        def __init__(self):
            self.host = None
            self.stream = RecordingStream()

        async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
            self.host = host
            return self.stream

        async def connect_unix_socket(self, path, timeout=None, socket_options=None):
            raise AssertionError("Unix sockets are not used")

        async def sleep(self, seconds):
            return None

    async def resolver(host, port):
        return ["8.8.8.8"]

    backend = RecordingBackend()
    pool = httpcore.AsyncConnectionPool(
        network_backend=ValidatingNetworkBackend(resolver=resolver, backend=backend),
        retries=0,
    )
    response = await pool.handle_async_request(
        httpcore.Request(
            "GET",
            "https://media.test/image.jpg",
            headers=[(b"host", b"media.test")],
        )
    )
    await response.stream.aclose()
    await pool.aclose()

    assert response.status == 200
    assert backend.host == "8.8.8.8"
    assert backend.stream.server_hostname == "media.test"


def test_mp4_sniffing_checks_major_and_compatible_brands():
    def ftyp(major, *compatible):
        return (
            b"\x00\x00\x00\x18ftyp"
            + major
            + b"\x00\x00\x00\x00"
            + b"".join(compatible)
        )

    assert media_store.sniff_media(ftyp(b"isom")) == ("video", "video/mp4")
    assert media_store.sniff_media(ftyp(b"free", b"mp42")) == (
        "video",
        "video/mp4",
    )
    for rejected in (
        ftyp(b"heic", b"mif1"),
        ftyp(b"heix", b"hevc"),
        ftyp(b"avif", b"avis"),
        ftyp(b"qt  "),
    ):
        assert media_store.sniff_media(rejected) is None


@pytest.mark.asyncio
async def test_upload_sniffing_and_overflow_cleanup(tmp_path, monkeypatch):
    await media_store.init_store(str(tmp_path / "session.db"))
    monkeypatch.setattr(media_store, "IMAGE_LIMIT", 10)
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        invalid = await client.post(
            "/api/upload",
            data={"thread_id": "upload-test"},
            files={"file": ("bad.svg", b"<svg>not allowed", "image/svg+xml")},
        )
        oversized = await client.post(
            "/api/upload",
            data={"thread_id": "upload-test"},
            files={"file": ("too-large.jpg", b"\xff\xd8\xff" + b"x" * 30, "image/jpeg")},
        )
    assert invalid.status_code == 415
    assert oversized.status_code == 413
    assert list(media_store.media_directory().iterdir()) == [media_store.cache_directory()]
    await media_store.close_store()


@pytest.mark.asyncio
async def test_upload_rejects_excessive_content_length_before_parsing():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        request = client.build_request(
            "POST",
            "/api/upload",
            headers={"Content-Type": "multipart/form-data; boundary=test"},
            content=b"not a multipart body",
        )
        request.headers["Content-Length"] = str(
            media_store.VIDEO_LIMIT + main.UPLOAD_MULTIPART_OVERHEAD + 1
        )
        response = await client.send(request)
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_upload_and_stage_delete_enforce_cors_origin_list(tmp_path):
    await media_store.init_store(str(tmp_path / "session.db"))
    transport = httpx.ASGITransport(app=main.app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            rejected_upload = await client.post(
                "/api/upload",
                headers={"Origin": "https://evil.example"},
                data={"thread_id": "origin-test"},
                files={"file": ("ok.jpg", b"\xff\xd8\xffdata", "image/jpeg")},
            )
            allowed_upload = await client.post(
                "/api/upload",
                headers={"Origin": "http://localhost:5173"},
                data={"thread_id": "origin-test"},
                files={"file": ("ok.jpg", b"\xff\xd8\xffdata", "image/jpeg")},
            )
            missing_origin_upload = await client.post(
                "/api/upload",
                data={"thread_id": "origin-test"},
                files={"file": ("ok.jpg", b"\xff\xd8\xffdata", "image/jpeg")},
            )
            assert rejected_upload.status_code == 403
            assert allowed_upload.status_code == 200
            assert (
                allowed_upload.headers["access-control-allow-origin"]
                == "http://localhost:5173"
            )
            assert missing_origin_upload.status_code == 200

            media_id = allowed_upload.json()["media_id"]
            item = {
                "media_id": media_id,
                "kind": "image",
                "mime": "image/jpeg",
                "src": f"/api/media/{media_id}",
                "caption": "ok.jpg",
            }
            await media_store.set_stage("origin-test", [item], "展示图片", "normal")
            rejected_delete = await client.delete(
                "/api/session/origin-test/stage",
                headers={"Origin": "https://evil.example"},
            )
            assert rejected_delete.status_code == 403
            assert await media_store.get_stage("origin-test") is not None

            allowed_delete = await client.delete(
                "/api/session/origin-test/stage",
                headers={"Origin": "http://127.0.0.1:5173"},
            )
            assert allowed_delete.status_code == 204
            assert (
                allowed_delete.headers["access-control-allow-origin"]
                == "http://127.0.0.1:5173"
            )

            await media_store.set_stage("origin-test", [item], "展示图片", "normal")
            missing_origin_delete = await client.delete("/api/session/origin-test/stage")
            assert missing_origin_delete.status_code == 204
            assert await media_store.get_stage("origin-test") is None
    finally:
        await media_store.close_store()


def test_default_cors_origins_include_tauri_clients():
    assert {
        "tauri://localhost",
        "http://tauri.localhost",
        "https://tauri.localhost",
    }.issubset(main.DEFAULT_CORS_ORIGINS)


@pytest.mark.asyncio
async def test_custom_cors_origin_controls_upload(tmp_path, monkeypatch):
    custom_origin = "http://example.test:3000"
    monkeypatch.setenv("CORS_ORIGINS", custom_origin)
    origins = main._effective_cors_origins()
    assert origins == [custom_origin]
    monkeypatch.setattr(main, "CORS_ORIGINS", origins)

    custom_app = main.FastAPI()
    custom_app.add_middleware(
        main.CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Range"],
    )
    custom_app.add_api_route("/api/upload", main.upload_media, methods=["POST"])
    await media_store.init_store(str(tmp_path / "session.db"))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=custom_app), base_url="http://test"
        ) as client:
            allowed = await client.post(
                "/api/upload",
                headers={"Origin": custom_origin},
                data={"thread_id": "custom-origin-test"},
                files={"file": ("ok.jpg", b"\xff\xd8\xffdata", "image/jpeg")},
            )
            rejected = await client.post(
                "/api/upload",
                headers={"Origin": "http://foreign.test:3000"},
                data={"thread_id": "custom-origin-test"},
                files={"file": ("ok.jpg", b"\xff\xd8\xffdata", "image/jpeg")},
            )
        assert allowed.status_code == 200
        assert allowed.headers["access-control-allow-origin"] == custom_origin
        assert rejected.status_code == 403
    finally:
        await media_store.close_store()


@pytest.mark.asyncio
async def test_uploaded_media_supports_range_and_nosniff(tmp_path):
    await media_store.init_store(str(tmp_path / "session.db"))
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        uploaded = await client.post(
            "/api/upload",
            data={"thread_id": "range-test"},
            files={"file": ("image.jpg", b"\xff\xd8\xff012345", "image/jpeg")},
        )
        media_id = uploaded.json()["media_id"]
        response = await client.get(
            f"/api/media/{media_id}", headers={"Range": "bytes=1-3"}
        )
    assert uploaded.status_code == 200
    assert response.status_code == 206
    assert response.content == b"\xd8\xff0"
    assert response.headers["content-range"].startswith("bytes 1-3/")
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-content-type-options"] == "nosniff"
    await media_store.close_store()


@pytest.mark.asyncio
async def test_image_attachments_decode_in_worker_and_encode_opaque_jpeg(
    tmp_path, monkeypatch
):
    from PIL import Image

    await media_store.init_store(str(tmp_path / "session.db"))
    image_path = media_store.media_directory() / "opaque"
    Image.new("RGB", (16, 16), (200, 30, 10)).save(image_path, format="PNG")
    record = await media_store.register_media(
        "image-worker",
        "image",
        "upload",
        "image/png",
        "opaque.png",
        path=str(image_path),
        byte_count=image_path.stat().st_size,
    )
    original_to_thread = asyncio.to_thread
    calls = []

    async def tracked_to_thread(function, *args, **kwargs):
        calls.append(function)
        return await original_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(graph.asyncio, "to_thread", tracked_to_thread)
    try:
        _, images = await graph._prepare_user_media(
            "请看看", "image-worker", [record["media_id"]], None
        )
        assert len(calls) == 1
        assert images[0].startswith("data:image/jpeg;base64,")
    finally:
        await media_store.close_store()


@pytest.mark.asyncio
async def test_stage_persistence_clear_delete_and_chat_reconcile(tmp_path, monkeypatch):
    store_path = str(tmp_path / "session.db")
    await session_store.init_store(store_path)
    await media_store.init_store(store_path)
    item = {
        "media_id": "m_stage",
        "kind": "image",
        "mime": "image/png",
        "src": "/api/media/m_stage",
        "caption": "示例",
    }
    await media_store.set_stage("stage-test", [item], "展示示例", "primary")
    response_stage = await main.session_endpoint("stage-test")
    assert response_stage["stage"]["items"] == [item]
    assert response_stage["stage"]["focus"] == "primary"

    transport = httpx.ASGITransport(app=main.app)

    async def empty_process_query(*args):
        yield {"type": "done", "turn_id": "t_test", "spoken_text": ""}

    monkeypatch.setattr(main, "process_query", empty_process_query)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        deleted = await client.delete("/api/session/stage-test/stage")
        assert deleted.status_code == 204
        await media_store.set_stage("stage-test", [item], "展示示例", "normal")
        await client.post(
            "/api/chat",
            json={"message": "继续", "thread_id": "stage-test"},
        )
        session = await client.get("/api/session/stage-test")
        assert session.json()["stage"] is not None
        await client.post(
            "/api/chat",
            json={
                "message": "继续",
                "thread_id": "stage-test",
                "screen_state": {"stage_media_ids": []},
            },
        )
        session = await client.get("/api/session/stage-test")
    assert session.json()["stage"] is None
    await media_store.close_store()
    await session_store.close_store()


@pytest.mark.asyncio
async def test_show_media_partial_and_all_failure(tmp_path, monkeypatch):
    await media_store.init_store(str(tmp_path / "session.db"))
    good = await media_store.register_media(
        "show-test", "image", "link", "image/jpeg", "good", url="https://good.test/a.jpg"
    )
    bad = await media_store.register_media(
        "show-test", "image", "link", "image/jpeg", "bad", url="https://bad.test/a.jpg"
    )

    async def probe(url):
        if "bad.test" in url:
            return {"status": 200, "content_type": "text/html", "length": 5}
        return {"status": 200, "content_type": "image/jpeg", "length": 5}

    events = []

    async def dispatch(name, data, config):
        events.append(data)

    monkeypatch.setattr(graph.safe_media_fetcher, "probe", probe)
    monkeypatch.setattr(graph, "adispatch_custom_event", dispatch)
    config = {"configurable": {"thread_id": "show-test"}}
    result = await graph.show_media.ainvoke(
        {"media_ids": [good["media_id"], bad["media_id"]], "caption": "对比图"},
        config=config,
    )
    stage = await media_store.get_stage("show-test")
    assert len(events) == 1
    assert events[0]["type"] == "stage.show"
    assert [entry["media_id"] for entry in stage["items"]] == [good["media_id"]]
    assert bad["media_id"] in result

    events.clear()
    result = await graph.show_media.ainvoke(
        {"media_ids": [bad["media_id"]], "caption": "不应显示"},
        config=config,
    )
    assert "没有可展示" in result
    assert events == []
    await graph.clear_stage.ainvoke({}, config=config)
    assert await media_store.get_stage("show-test") is None
    assert events[-1]["type"] == "stage.clear"
    await media_store.close_store()


@pytest.mark.asyncio
async def test_report_media_rewrite_and_prompt_omit_urls(tmp_path, monkeypatch):
    store_path = str(tmp_path / "session.db")
    await session_store.init_store(store_path)
    await media_store.init_store(store_path)
    record = await media_store.register_media(
        "prompt-test",
        "image",
        "search",
        "image/jpeg",
        "图片描述 https://secret.test/image.jpg",
        url="https://secret.test/image.jpg",
    )
    markdown, removed = await graph._rewrite_report_media(
        f"![图](media:{record['media_id']}) ![外部](media:m_foreign)",
        "prompt-test",
    )
    assert f"![图](media:{record['media_id']})" in markdown
    assert "外部" in markdown and "media:m_foreign" not in markdown
    assert removed == ["m_foreign"]
    item = {
        "media_id": record["media_id"],
        "kind": "image",
        "mime": "image/jpeg",
        "src": f"/api/media/{record['media_id']}",
        "caption": record["caption"],
    }
    await media_store.set_stage("prompt-test", [item], "正在看", "primary")
    captured = {}

    class CapturingModel:
        async def ainvoke(self, messages, config):
            captured["prompt"] = messages[0].content
            return HumanMessage(content="已收到")

    monkeypatch.setattr(graph.memory_manager, "format_context", lambda query: "")
    monkeypatch.setattr(graph, "llm_with_tools", CapturingModel())
    result = await graph.agent_node(
        {"messages": [HumanMessage(content="你好")], "context": ""},
        {
            "configurable": {
                "thread_id": "prompt-test",
                "screen_state": {"stage_current_media_id": record["media_id"]},
            }
        },
    )
    assert "展示台：" in captured["prompt"]
    assert f"用户正在看 {record['media_id']}" in captured["prompt"]
    assert "https://" not in captured["prompt"]
    await media_store.close_store()
    await session_store.close_store()


@pytest.mark.asyncio
async def test_concurrent_remote_image_caches_serialize_eviction(tmp_path, monkeypatch):
    monkeypatch.setattr(media_store, "_store_lock", asyncio.Lock())
    monkeypatch.setattr(media_store, "_cache_eviction_lock", asyncio.Lock())
    monkeypatch.setattr(media_store, "_cache_locks", {})
    await media_store.init_store(str(tmp_path / "session.db"))
    old = await media_store.register_media(
        "cache-test", "image", "search", "image/jpeg", "old",
        url="https://media.test/old.jpg",
    )
    first = await media_store.register_media(
        "cache-test", "image", "search", "image/jpeg", "first",
        url="https://media.test/first.jpg",
    )
    second = await media_store.register_media(
        "cache-test", "image", "search", "image/jpeg", "second",
        url="https://media.test/second.jpg",
    )
    old_path = media_store.cache_directory() / old["media_id"]
    old_path.write_bytes(b"o" * 9)
    await media_store.set_cached_path(old["media_id"], str(old_path), 9)

    download_count = 0
    both_downloading = asyncio.Event()

    class Fetcher:
        async def download_to_file(self, url, destination, max_bytes):
            nonlocal download_count
            download_count += 1
            if download_count == 2:
                both_downloading.set()
            await both_downloading.wait()
            Path(destination).write_bytes(b"n" * 4)
            return {"content_type": "image/jpeg"}

    try:
        records = await asyncio.gather(
            media_store.cache_remote_image(first["media_id"], Fetcher(), 10),
            media_store.cache_remote_image(second["media_id"], Fetcher(), 10),
        )
        assert all(Path(record["path"]).is_file() for record in records)
    finally:
        await media_store.close_store()


@pytest.mark.asyncio
async def test_register_message_links_trims_unbalanced_closers(monkeypatch):
    registered = []

    async def capture_link(*args, **kwargs):
        registered.append(kwargs["url"])
        return {}

    monkeypatch.setattr(media_store, "register_media", capture_link)
    await main._register_message_links(
        "link-test",
        "[video](https://example.com/a.mp4) "
        "https://example.com/b.webm "
        "（https://example.com/c.gif） "
        "[balanced](https://example.com/folder(a)/d.mp4) "
        "https://example.com/e.mp4?download=1 "
        "https://example.com/f.jpg] "
        "<https://example.com/g.png> "
        '"https://example.com/h.webp"',
    )
    assert registered == [
        "https://example.com/a.mp4",
        "https://example.com/b.webm",
        "https://example.com/c.gif",
        "https://example.com/folder(a)/d.mp4",
        "https://example.com/e.mp4?download=1",
        "https://example.com/f.jpg",
        "https://example.com/g.png",
        "https://example.com/h.webp",
    ]
