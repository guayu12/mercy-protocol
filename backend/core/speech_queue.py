"""Bounded in-memory lookup for synthesized speech sentences."""

from collections import OrderedDict


MAX_SENTENCES = 1000
_sentences: OrderedDict[str, str] = OrderedDict()
_turn_sentences: OrderedDict[str, list[str]] = OrderedDict()


def remember_sentence(turn_id: str, sentence_id: str, text: str):
    _sentences[sentence_id] = text
    _sentences.move_to_end(sentence_id)
    sentence_ids = _turn_sentences.setdefault(turn_id, [])
    if sentence_id not in sentence_ids:
        sentence_ids.append(sentence_id)
    _turn_sentences.move_to_end(turn_id)
    while len(_sentences) > MAX_SENTENCES:
        _sentences.popitem(last=False)
    while len(_turn_sentences) > MAX_SENTENCES:
        _turn_sentences.popitem(last=False)


def get_sentence(sentence_id: str) -> str | None:
    text = _sentences.get(sentence_id)
    if text is not None:
        _sentences.move_to_end(sentence_id)
    return text


def get_turn_sentence_ids(turn_id: str) -> list[str]:
    return list(_turn_sentences.get(turn_id, []))


def clear():
    """Clear cached sentences (primarily useful in tests)."""
    _sentences.clear()
    _turn_sentences.clear()
