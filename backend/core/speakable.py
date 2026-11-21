"""Helpers for turning model output into short, spoken sentences."""

import re
import unicodedata


_FENCED_CODE = re.compile(r"```.*?(?:```|$)", re.DOTALL)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+?)\s*$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_REPEATED_PUNCTUATION = re.compile(r"([。！？!?…])\1+")
_BOUNDARIES = set("。！？!?\n")
_LONG_CLAUSE_MARKS = set("，,；;")


def _remove_emoji_like_symbols(text: str) -> str:
    return "".join(
        char
        for char in text
        if unicodedata.category(char) not in {"So", "Sk"}
        and char not in {"\u200d", "\ufe0e", "\ufe0f"}
        and not 0x1F1E6 <= ord(char) <= 0x1F1FF
    )


def to_speakable(text: str) -> str:
    """Remove Markdown and decorative symbols while preserving readable text."""
    text = _FENCED_CODE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    text = _remove_emoji_like_symbols(text)

    lines = text.splitlines()
    output: list[str] = []
    list_items: list[str] = []
    for line in lines:
        match = _LIST_ITEM.match(line)
        if match:
            list_items.append(match.group(1).strip())
            continue
        if list_items:
            output.append("，".join(list_items) + "。")
            list_items = []
        if line.strip():
            output.append(line.strip())
    if list_items:
        output.append("，".join(list_items) + "。")

    text = " ".join(output).replace("|", "")
    text = re.sub(r"~~(.*?)~~", r"\1", text)
    text = re.sub(r"(?<!\w)[*_]+|[*_]+(?!\w)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _REPEATED_PUNCTUATION.sub(r"\1", text)
    return text


class SpeechRouter:
    """Incrementally split agent text into cleaned, bounded speech sentences."""

    def __init__(self, turn_id: str, max_chars: int = 300):
        self.turn_id = turn_id
        self.max_chars = max_chars
        self.raw_full_text = ""
        self._scanned_text = ""
        self._buffer = ""
        self._sentences: list[dict] = []
        self._spoken_text = ""
        self._overflowed = False
        self._boundary_pending = False

    @property
    def overflowed(self) -> bool:
        return self._overflowed

    @property
    def spoken_text(self) -> str:
        return self._spoken_text

    def _make_sentence(self, text: str, enforce_limit: bool = True) -> dict | None:
        cleaned = to_speakable(text)
        if not cleaned:
            return None
        if enforce_limit and len(self._spoken_text) + len(cleaned) > self.max_chars:
            self._overflowed = True
            return None
        sentence = {
            "id": f"{self.turn_id}.{len(self._sentences) + 1}",
            "text": cleaned,
        }
        self._sentences.append(sentence)
        self._spoken_text += cleaned
        return sentence

    def _flush_buffer(self) -> dict | None:
        text, self._buffer = self._buffer, ""
        self._boundary_pending = False
        return self._make_sentence(text)

    def _is_unsafe_markdown(self) -> bool:
        return "```" in self._scanned_text or any(
            _TABLE_ROW.match(line) for line in self._scanned_text.splitlines()
        )

    def feed(self, delta: str) -> list[dict]:
        """Consume a token delta and return any complete spoken sentences."""
        self.raw_full_text += delta
        emitted: list[dict] = []
        if self._overflowed:
            return emitted

        for char in delta:
            self._scanned_text += char
            if self._is_unsafe_markdown():
                self._overflowed = True
                break

            if self._boundary_pending and char not in _BOUNDARIES:
                sentence = self._flush_buffer()
                if self._overflowed:
                    break
                if sentence:
                    emitted.append(sentence)

            self._buffer += char
            if self._is_unsafe_markdown():
                self._overflowed = True
                break

            cleaned_pending = to_speakable(self._buffer)
            if len(self._spoken_text) + len(cleaned_pending) > self.max_chars:
                self._overflowed = True
                break

            if char in _BOUNDARIES:
                self._boundary_pending = True
            elif len(self._buffer) > 60:
                split_at = max(
                    self._buffer.rfind(mark) for mark in _LONG_CLAUSE_MARKS
                )
                if split_at >= 0:
                    clause = self._buffer[: split_at + 1]
                    self._buffer = self._buffer[split_at + 1 :]
                    sentence = self._make_sentence(clause)
                    if self._overflowed:
                        break
                    if sentence:
                        emitted.append(sentence)

        return emitted

    def add_sentence(self, text: str) -> dict | None:
        """Add a final system-generated sentence to this turn."""
        return self._make_sentence(text, enforce_limit=False)

    def flush_pending(self) -> list[dict]:
        """Flush text accumulated before a tool boundary."""
        if self._overflowed or not self._buffer:
            return []
        sentence = self._flush_buffer()
        return [sentence] if sentence else []

    def finish(self) -> dict:
        """Flush any remaining text and return the complete turn summary."""
        remaining = self.flush_pending()
        return {
            "sentences": remaining,
            "overflow_markdown": self.raw_full_text if self._overflowed else None,
            "spoken_text": self._spoken_text,
        }
