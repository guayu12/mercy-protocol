import os


os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")

from core.models import ModelSpec

import pytest


class TestRuntime:
    def __init__(self, tool_names):
        self._tool_names = frozenset(tool_names)
        self.roles = {
            "conversation": ModelSpec("openai", "test-conversation", None),
            "report": (
                ModelSpec("openai", "test-report", None)
                if "write_report" in self._tool_names
                else None
            ),
            "research": (
                ModelSpec("openai", "test-research", None)
                if "investigate" in self._tool_names
                else None
            ),
            "vision": (
                ModelSpec("openai", "test-vision", None)
                if "inspect_image" in self._tool_names
                else None
            ),
            "video": (
                ModelSpec("gemini", "test-video", None)
                if {"analyze_video", "analyze_document"} & self._tool_names
                else None
            ),
            "memory": (
                ModelSpec("openai", "text-embedding-3-small")
                if "save_memory" in self._tool_names
                else None
            ),
        }
        self.image_input = {
            spec: True for spec in self.roles.values() if spec is not None
        }
        self.tavily = "investigate" in self._tool_names or "tavily_search" in self._tool_names

    def spec(self, role, tier="normal", *, fallback=False):
        if fallback:
            primary = self.roles.get(role)
            if primary is None:
                return None
            return ModelSpec(primary.provider, f"{primary.model}-fallback", primary.effort)
        return self.roles.get(role)

    def available(self, role, tier="normal"):
        spec = self.spec(role, tier)
        if spec is None:
            return False
        if role == "vision":
            return self.image_input.get(spec, False)
        return True

    def can_see_images(self, role="conversation"):
        spec = self.spec(role)
        return spec is not None and self.image_input.get(spec, False)

    def tool_names(self):
        return self._tool_names

    def unavailable_reason(self, name):
        return None if name in self._tool_names else "未配置测试模型"


def make_test_runtime(tool_names):
    return TestRuntime(tool_names)


@pytest.fixture(autouse=True)
def disable_deepseek_capability_network(monkeypatch):
    from core import runtime

    async def no_network(_key):
        return {}

    monkeypatch.setattr(runtime, "_fetch_deepseek_capabilities", no_network)


@pytest.fixture(autouse=True)
def isolate_memory_database(monkeypatch):
    from core.agent import graph

    async def empty_context(_runtime, _query):
        return ""

    monkeypatch.setattr(graph, "_memory_context", empty_context)
