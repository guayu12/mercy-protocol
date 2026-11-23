from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

import main


def test_post_speech_cleans_markdown_and_rejects_empty_text(monkeypatch):
    synthesized = []

    async def fake_synthesize(text):
        synthesized.append(text)
        return {"audio": "YXVkaW8=", "mime": "audio/mpeg", "visemes": []}

    @asynccontextmanager
    async def no_lifespan(app):
        yield

    monkeypatch.setattr(main, "synthesize", fake_synthesize)
    monkeypatch.setattr(main.app.router, "lifespan_context", no_lifespan)

    with TestClient(main.app) as client:
        response = client.post("/api/speech", json={"text": "**你好**。"})
        empty_response = client.post(
            "/api/speech", json={"text": "```python\nprint(1)\n```"}
        )

    assert response.status_code == 200
    assert synthesized == ["你好。"]
    assert empty_response.status_code == 400
