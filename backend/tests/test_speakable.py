from core.speakable import SpeechRouter, to_speakable


def _collect(router, text):
    sentences = []
    for char in text:
        sentences.extend(router.feed(char))
    sentences.extend(router.finish()["sentences"])
    return sentences


def test_to_speakable_removes_markdown_and_keeps_text():
    assert to_speakable("# 标题\n这是 **重要** 的 `内容`。") == "标题 这是 重要 的 内容。"
    assert to_speakable("运行 `print(1)`") == "运行 print(1)"
    assert to_speakable("保留这句。\n```python\nprint('hidden')\n```\n继续。") == "保留这句。 继续。"
    assert to_speakable("保留这句。\n~~~python\nprint('hidden')\n~~~\n继续。") == "保留这句。 继续。"
    assert to_speakable("[a](http://x) 和 https://example.com") == "a 和"
    assert to_speakable("- 第一项\n- 第二项\n1. third item") == "第一项，第二项，third item。"
    assert to_speakable("* first\n* second\nuser_name") == "first，second。 user_name"
    assert to_speakable("好。。真棒！！等一下……") == "好。真棒！等一下…"
    assert to_speakable("中 English 123 😀 继续") == "中 English 123 继续"


def test_to_speakable_removes_hidden_zero_width_characters_and_emoji_joiners():
    assert to_speakable("M\u200bE\u200cR\u2060C\ufeffY") == "MERCY"
    assert to_speakable("好的👨‍👩‍👧一家人。") == "好的一家人。"
    assert to_speakable("明白🧑‍💻。") == "明白。"


def test_to_speakable_requires_matching_fence_character_and_length():
    assert to_speakable("```markdown\n~~~\nexample\n```\n后面这句要念") == "后面这句要念"
    assert (
        to_speakable("````\nexample\n```\nstill code\n`````\n后面这句要念")
        == "后面这句要念"
    )


def test_speech_router_splits_punctuation_across_token_boundaries():
    router = SpeechRouter("t_test")
    sentences = _collect(router, "你好。世界！你呢？")
    assert [sentence["text"] for sentence in sentences] == ["你好。", "世界！", "你呢？"]
    assert [sentence["id"] for sentence in sentences] == [
        "t_test.1",
        "t_test.2",
        "t_test.3",
    ]


def test_speech_router_keeps_ascii_url_queries_in_one_clean_sentence():
    examples = (
        ("[文档](https://x.com/a?b=1) 可以查看。", "文档 可以查看。"),
        ("访问 https://x.com/a?b=1 后继续。", "访问 后继续。"),
        ("访问 https://x.com/a?文档=1 后继续。", "访问 后继续。"),
    )
    for raw_text, expected in examples:
        sentences = _collect(SpeechRouter("t_url"), raw_text)
        assert len(sentences) == 1
        assert sentences[0]["text"] == expected
        assert not any(mark in sentences[0]["text"] for mark in "[]?http")


def test_speech_router_defers_ascii_boundary_until_next_delta():
    router = SpeechRouter("t_ascii")
    assert router.feed("Really?") == []
    assert router.feed("你") == [{"id": "t_ascii.1", "text": "Really?"}]
    assert router.finish()["sentences"] == [{"id": "t_ascii.2", "text": "你"}]

    final_punctuation = SpeechRouter("t_final")
    assert final_punctuation.feed("Really?") == []
    assert final_punctuation.finish()["sentences"] == [
        {"id": "t_final.1", "text": "Really?"}
    ]


def test_speech_router_splits_long_clauses_at_commas():
    router = SpeechRouter("t_long")
    sentences = _collect(router, "甲" * 61 + "，乙乙。")
    assert len(sentences) == 2
    assert sentences[0]["text"] == "甲" * 61 + "，"
    assert sentences[1]["text"] == "乙乙。"


def test_speech_router_overflows_code_tables_and_long_text():
    for text in (
        "```python\nx = 1\n```",
        "~~~python\nx = 1\n~~~",
        "| a | b |\n|---|---|\n| 1 | 2 |",
        "a | b\n--- | ---",
        "字" * 301,
    ):
        router = SpeechRouter("t_overflow")
        router.feed(text)
        result = router.finish()
        assert result["overflow_markdown"] == text
