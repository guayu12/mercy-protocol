"""Bounded in-memory lookup for synthesized speech sentences."""

from collections import OrderedDict


MAX_SENTENCES = 1000
_sentences: OrderedDict[str, str] = OrderedDict()
_sentence_segments: dict[str, int] = {}
_turn_sentences: OrderedDict[str, list[str]] = OrderedDict()
_turn_segments: OrderedDict[str, int] = OrderedDict()


def remember_sentence(
    turn_id: str, sentence_id: str, text: str, segment_index: int = 0
):
    _sentences[sentence_id] = text
    _sentence_segments[sentence_id] = segment_index
    _sentences.move_to_end(sentence_id)
    _turn_segments.setdefault(turn_id, segment_index)
    _turn_segments.move_to_end(turn_id)
    sentence_ids = _turn_sentences.setdefault(turn_id, [])
    if sentence_id not in sentence_ids:
        sentence_ids.append(sentence_id)
    _turn_sentences.move_to_end(turn_id)
    while len(_sentences) > MAX_SENTENCES:
        expired_id, _ = _sentences.popitem(last=False)
        _sentence_segments.pop(expired_id, None)
    while len(_turn_sentences) > MAX_SENTENCES:
        expired_turn_id, _ = _turn_sentences.popitem(last=False)
        _turn_segments.pop(expired_turn_id, None)


def advance_turn_segment(turn_id: str) -> int:
    segment_index = _turn_segments.get(turn_id, 0) + 1
    _turn_segments[turn_id] = segment_index
    _turn_segments.move_to_end(turn_id)
    while len(_turn_segments) > MAX_SENTENCES:
        _turn_segments.popitem(last=False)
    return segment_index


def get_sentence(sentence_id: str) -> str | None:
    text = _sentences.get(sentence_id)
    if text is not None:
        _sentences.move_to_end(sentence_id)
    return text


def get_sentence_segment(sentence_id: str) -> int | None:
    if sentence_id not in _sentence_segments:
        return None
    _sentences.move_to_end(sentence_id)
    return _sentence_segments[sentence_id]


def get_turn_segment(turn_id: str) -> int | None:
    segment_index = _turn_segments.get(turn_id)
    if segment_index is not None:
        _turn_segments.move_to_end(turn_id)
    return segment_index


def get_turn_sentence_ids(turn_id: str) -> list[str]:
    return list(_turn_sentences.get(turn_id, []))


def clear():
    """Clear cached sentences (primarily useful in tests)."""
    _sentences.clear()
    _sentence_segments.clear()
    _turn_sentences.clear()
    _turn_segments.clear()
