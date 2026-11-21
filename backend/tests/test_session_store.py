import pytest

from core import session_store


@pytest.mark.asyncio
async def test_reports_and_utterances_round_trip(tmp_path):
    await session_store.init_store(str(tmp_path / "data" / "session.db"))
    try:
        first = await session_store.create_report("thread", "First", "question", "# report")
        second = await session_store.create_report("thread", "Second", "question", "|a|b|")
        assert first["report_id"].startswith("r_")
        assert len(first["report_id"]) == 10
        assert (await session_store.get_report("thread", first["report_id"]))["markdown"] == "# report"
        assert await session_store.get_report("other", first["report_id"]) is None
        assert [r["report_id"] for r in await session_store.list_reports("thread")] == [
            first["report_id"],
            second["report_id"],
        ]

        await session_store.add_utterance("thread", "t1", "user", "你好")
        await session_store.add_utterance("thread", "t1", "assistant", "你好。")
        await session_store.mark_interrupted("thread", "t1", "你好")
        await session_store.mark_interrupted("thread", "t2", "（还没开口就被用户打断了）")
        utterances = await session_store.recent_utterances("thread")
        assert [(item["role"], item["text"], item["interrupted"]) for item in utterances] == [
            ("user", "你好", 0),
            ("assistant", "你好", 1),
            ("assistant", "（还没开口就被用户打断了）", 1),
        ]
    finally:
        await session_store.close_store()
