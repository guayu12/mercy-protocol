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
    assert to_speakable("[a](http://x) 和 https://example.com") == "a 和"
    assert to_speakable("- 第一项\n- 第二项\n1. third item") == "第一项，第二项，third item。"
    assert to_speakable("* first\n* second\nuser_name") == "first，second。 user_name"
    assert to_speakable("好。。真棒！！等一下……") == "好。真棒！等一下…"
    assert to_speakable("中 English 123 😀 继续") == "中 English 123 继续"


def test_speech_router_splits_punctuation_across_token_boundaries():
    router = SpeechRouter("t_test")
    sentences = _collect(router, "你好。世界！你呢？")
    assert [sentence["text"] for sentence in sentences] == ["你好。", "世界！", "你呢？"]
    assert [sentence["id"] for sentence in sentences] == [
        "t_test.1",
        "t_test.2",
        "t_test.3",
    ]


def test_speech_router_splits_long_clauses_at_commas():
    router = SpeechRouter("t_long")
    sentences = _collect(router, "甲" * 61 + "，乙乙。")
    assert len(sentences) == 2
    assert sentences[0]["text"] == "甲" * 61 + "，"
    assert sentences[1]["text"] == "乙乙。"


def test_speech_router_overflows_code_tables_and_long_text():
    for text in ("```python\nx = 1\n```", "| a | b |\n|---|---|\n| 1 | 2 |", "字" * 301):
        router = SpeechRouter("t_overflow")
        router.feed(text)
        result = router.finish()
        assert result["overflow_markdown"] == text
