from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from core import runtime


@pytest.fixture(autouse=True)
def _local_backend(monkeypatch):
    monkeypatch.setenv("BACKEND_HOST", "127.0.0.1")
    monkeypatch.setattr(main, "CORS_ORIGINS", ["http://127.0.0.1:1420"])


def _headers():
    return {
        "Host": "127.0.0.1:8000",
        "Origin": "http://127.0.0.1:1420",
        "Content-Type": "application/json",
    }


@pytest.mark.parametrize("route", ["profiles", "active"])
def test_locked_profiles_read_returns_config_write(route, tmp_path, monkeypatch):
    profiles = {
        "full": {"roles": {"conversation": "openai:full-model"}},
        "lite": {"roles": {"conversation": "openai:lite-model"}},
    }
    runtime.save_profiles(profiles)
    path = runtime.profiles_path()
    read_bytes = Path.read_bytes
    before = read_bytes(path)

    def locked_read(file_path):
        if file_path == path:
            raise PermissionError("locked")
        return read_bytes(file_path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", locked_read)
        client = TestClient(main.app, client=("127.0.0.1", 43123))
        if route == "profiles":
            response = client.post(
                "/api/config/profiles",
                headers=_headers(),
                json={"profiles": profiles},
            )
        else:
            response = client.post(
                "/api/config/active",
                headers=_headers(),
                json={"active": "lite"},
            )

        assert response.status_code == 400
        assert response.json() == {
            "category": "config_write",
            "message": "无法读取配置文件（可能被占用）",
        }
        assert read_bytes(path) == before
