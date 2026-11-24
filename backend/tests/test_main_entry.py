import runpy


def test_main_entry_sets_graceful_shutdown_timeout(tmp_path, monkeypatch):
    from core import env

    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_CONVERSATION=openai:test-chat\n", encoding="utf-8")
    monkeypatch.setattr(env, "ENV_PATH", env_file)
    monkeypatch.setenv("MERCY_PROFILES_PATH", str(tmp_path / "profiles.json"))
    monkeypatch.setenv("MODEL_CONVERSATION", "openai:test-chat")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv(
        "MERCY_SYSTEM_KEYS",
        '["OPENAI_API_KEY","DEEPSEEK_API_KEY","GEMINI_API_KEY",'
        '"GOOGLE_API_KEY","TAVILY_API_KEY"]',
    )

    import main
    import uvicorn

    calls = {}
    monkeypatch.setattr(
        uvicorn, "run", lambda *args, **kwargs: calls.update(args=args, kwargs=kwargs)
    )
    runpy.run_path(str(main.__file__), run_name="__main__")

    assert calls["kwargs"]["reload"] is True
    assert (
        calls["kwargs"]["timeout_graceful_shutdown"]
        == main.GRACEFUL_SHUTDOWN_SECONDS
    )
