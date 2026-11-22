import pytest
import aiosqlite

from core import session_store


@pytest.mark.asyncio
async def test_reports_and_utterances_round_trip(tmp_path):
    await session_store.init_store(str(tmp_path / "data" / "session.db"))
    try:
        first = await session_store.create_report("thread", "First", "question", "# report")
        sources = [{"number": 1, "title": "Source", "url": "https://source.test"}]
        second = await session_store.create_report(
            "thread",
            "Second",
            "question",
            "|a|b|",
            link_policy="sources_only",
            sources=sources,
        )
        assert first["report_id"].startswith("r_")
        assert len(first["report_id"]) == 10
        first_report = await session_store.get_report("thread", first["report_id"])
        second_report = await session_store.get_report("thread", second["report_id"])
        assert first_report["markdown"] == "# report"
        assert first_report["link_policy"] == "open"
        assert first_report["sources"] is None
        assert second_report["link_policy"] == "sources_only"
        assert second_report["sources"] == sources
        assert await session_store.get_report("other", first["report_id"]) is None
        listed = await session_store.list_reports("thread")
        assert [r["report_id"] for r in listed] == [
            first["report_id"],
            second["report_id"],
        ]
        assert listed[1]["sources"] == sources

        await session_store.add_utterance("thread", "t1", "user", "你好")
        await session_store.add_utterance("thread", "t1", "assistant", "你好。")
        await session_store.mark_interrupted("thread", "t1", "你好")
        await session_store.mark_interrupted("thread", "t2", "（还没开口就被用户打断了）")
        await session_store.add_utterance("thread", "t3", "assistant", "完整的原始回复。")
        await session_store.mark_interrupted("thread", "t3", None)
        utterances = await session_store.recent_utterances("thread")
        assert [(item["role"], item["text"], item["interrupted"]) for item in utterances] == [
            ("user", "你好", 0),
            ("assistant", "你好", 1),
            ("assistant", "（还没开口就被用户打断了）", 1),
            ("assistant", "完整的原始回复。", 1),
        ]
        latest = await session_store.latest_user_or_assistant_utterance("thread")
        assert latest["turn_id"] == "t3"
        assert latest["text"] == "完整的原始回复。"
    finally:
        await session_store.close_store()


@pytest.mark.asyncio
async def test_old_reports_schema_migrates_without_changing_existing_report(tmp_path):
    path = tmp_path / "old-session.db"
    async with aiosqlite.connect(path) as db:
        await db.execute(
            "CREATE TABLE reports ("
            "thread_id TEXT NOT NULL, report_id TEXT PRIMARY KEY, "
            "title TEXT NOT NULL, query TEXT NOT NULL, markdown TEXT NOT NULL, "
            "created_at REAL NOT NULL)"
        )
        await db.execute(
            "INSERT INTO reports VALUES (?, ?, ?, ?, ?, ?)",
            ("old-thread", "r_old", "Old", "Question", "# unchanged", 42.0),
        )
        await db.commit()

    await session_store.init_store(str(path))
    try:
        report = await session_store.get_report("old-thread", "r_old")
        assert report == {
            "thread_id": "old-thread",
            "report_id": "r_old",
            "title": "Old",
            "query": "Question",
            "markdown": "# unchanged",
            "created_at": 42.0,
            "link_policy": "open",
            "sources": None,
        }
    finally:
        await session_store.close_store()
