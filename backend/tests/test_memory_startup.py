import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_utf16_env_app_startup_keeps_config_available(tmp_path):
    (tmp_path / ".env").write_bytes("UNREADABLE=value".encode("utf-16"))
    backend_dir = Path(__file__).resolve().parents[1]
    code = textwrap.dedent(
        """
        import asyncio
        import sys
        from pathlib import Path
        from fastapi.testclient import TestClient
        from core import env, media_store, runtime, session_store

        root = Path(sys.argv[1])
        env.ENV_PATH = root / ".env"
        import main
        from core.agent import graph

        graph.BACKEND_DIR = str(root / "backend-data")
        Path(graph.BACKEND_DIR).mkdir()
        async def no_store(*_args, **_kwargs):
            return None
        async def wait_forever():
            await asyncio.Future()
        session_store.init_store = no_store
        session_store.close_store = no_store
        media_store.init_store = no_store
        media_store.close_store = no_store
        main.gemini_file_cleanup_loop = wait_forever
        main.cleanup_gemini_files = no_store
        main.CORS_ORIGINS = ["http://127.0.0.1:1420"]

        assert "chromadb" not in sys.modules
        with TestClient(main.app, base_url="http://127.0.0.1:8000") as client:
            response = client.get(
                "/api/config",
                headers={
                    "Host": "127.0.0.1:8000",
                    "Origin": "http://127.0.0.1:1420",
                },
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["env_error"] == (
                "backend/.env 不是 UTF-8 编码，请另存为 UTF-8 后重启"
            )
            memory = next(role for role in body["roles"] if role["role"] == "memory")
            assert memory["available"] is False
            assert "save_memory" not in runtime.current_snapshot().tool_names()
            assert runtime.memory_unavailable_reason() == "长期记忆初始化失败"
        print("startup survived; memory unavailable")
        """
    )
    child_env = os.environ.copy()
    for name in list(child_env):
        if name.startswith("MODEL_") or name in {
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "TAVILY_API_KEY",
            "MERCY_SYSTEM_KEYS",
            "DEFAULT_MODEL",
            "DEFAULT_REASONING_EFFORT",
            "OPENAI_EMBEDDING_MODEL",
        }:
            child_env.pop(name, None)
    child_env.update(
        {
            "OPENAI_API_KEY": "test-openai-key",
            "MODEL_CONVERSATION": "openai:fake-chat",
            "MERCY_PROFILES_PATH": str(tmp_path / "profiles.json"),
            "PYTHONPATH": str(backend_dir),
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=tmp_path,
        env=child_env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "startup survived; memory unavailable" in result.stdout
