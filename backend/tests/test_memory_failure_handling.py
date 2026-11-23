import sys
import types

import pytest

from core import config_service, env, runtime
from core.agent import graph
from core.memory.memory_manager import MemoryInitError, MemoryManager
from core.models import ModelSpec
from core.runtime import RuntimeSnapshot

_memory_context = graph._memory_context


def _memory_snapshot(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    runtime.reset_runtime_cache()
    return RuntimeSnapshot(
        source="test",
        active="full",
        version=0,
        roles={"memory": ModelSpec("openai", "text-embedding-3-small", None)},
        tavily=False,
        image_input={},
        profiles_view={},
    )


@pytest.mark.asyncio
async def test_request_time_memory_failures_do_not_disable_memory(monkeypatch):
    snapshot = _memory_snapshot(monkeypatch)

    class FailingRequests:
        context_calls = 0
        add_calls = 0

        def format_context(self, _query):
            self.context_calls += 1
            raise TimeoutError("embedding timeout")

        def add_memory(self, _fact):
            self.add_calls += 1
            raise TimeoutError("embedding timeout")

    manager = FailingRequests()
    warnings = []
    monkeypatch.setattr(graph, "_memory_manager", manager)
    monkeypatch.setattr(
        graph.log_bus,
        "emit",
        lambda *args: warnings.append(args) if args[0] == "warn" else None,
    )

    assert snapshot.available("memory")
    assert await _memory_context(snapshot, "first query") == ""
    assert snapshot.available("memory")
    assert await _memory_context(snapshot, "second query") == ""
    assert manager.context_calls == 2

    unavailable = "长期记忆当前不可用。"
    assert await graph.save_memory.ainvoke({"fact": "test fact"}) == unavailable
    assert snapshot.available("memory")
    assert await graph.save_memory.ainvoke({"fact": "test fact"}) == unavailable
    assert manager.add_calls == 2
    assert snapshot.available("memory")
    assert [warning[2] for warning in warnings] == [
        "Long-term memory request failed (TimeoutError)"
    ] * 4


@pytest.mark.asyncio
async def test_initialization_failure_disables_memory_across_calls(
    tmp_path, monkeypatch
):
    snapshot = _memory_snapshot(monkeypatch)
    client_calls = 0

    def fail_client(*_args, **_kwargs):
        nonlocal client_calls
        client_calls += 1
        raise RuntimeError("client initialization failed")

    chromadb = types.ModuleType("chromadb")
    chromadb.PersistentClient = fail_client
    monkeypatch.setitem(sys.modules, "chromadb", chromadb)

    manager = MemoryManager(str(tmp_path / "chroma"))
    monkeypatch.setattr(graph, "_memory_manager", manager)

    with pytest.raises(MemoryInitError):
        manager.initialize()
    assert graph._initialize_memory() is False
    assert client_calls == 2
    assert not snapshot.available("memory")
    assert runtime.memory_unavailable_reason() == "长期记忆初始化失败"

    assert await _memory_context(snapshot, "query") == ""
    assert await _memory_context(snapshot, "query") == ""
    assert (
        await graph.save_memory.ainvoke({"fact": "test fact"})
        == "长期记忆当前不可用。"
    )
    assert graph.get_memory_manager() is None
    assert not snapshot.available("memory")


@pytest.mark.asyncio
async def test_successful_openai_key_save_clears_memory_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / ".env"
    path.write_bytes(b"")
    monkeypatch.setattr(env, "ENV_PATH", path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MERCY_SYSTEM_KEYS", raising=False)
    env.load_env()

    async def validate_key(_provider, _key):
        return None

    monkeypatch.setattr(config_service, "_validate_provider_key", validate_key)
    runtime.mark_memory_unavailable()
    assert runtime.memory_unavailable_reason() is not None

    await config_service._save_key_locked(
        config_service.KeyRequest(
            provider="openai",
            api_key="fake-valid-openai-key",
        )
    )

    assert runtime.memory_unavailable_reason() is None
    assert path.read_bytes() == b"OPENAI_API_KEY=fake-valid-openai-key\n"
