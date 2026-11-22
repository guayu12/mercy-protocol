"""TTS synthesis with word-level timing converted to VRM viseme tracks."""
import base64
import os
import re

import edge_tts
from pypinyin import Style, lazy_pinyin

VOICE = os.environ.get("TTS_VOICE", "zh-CN-XiaoyiNeural")
TICKS_PER_SECOND = 10_000_000

_CJK = re.compile(r"[\u4e00-\u9fff]")
_DIGITS = "零一二三四五六七八九"
_LATIN_VOWEL = {"a": "aa", "e": "ee", "i": "ih", "y": "ih", "o": "oh", "u": "ou"}


def final_to_viseme(final: str) -> str:
    """Map a pinyin final (e.g. 'iang', 'ui', 'v') to a VRM viseme (aa/ih/ou/ee/oh)."""
    f = final.replace("ü", "v")
    if "a" in f:
        return "aa"
    if "o" in f:
        return "oh"
    if "e" in f:
        return "aa" if f == "er" else "ee"
    for ch in reversed(f):
        if ch == "i":
            return "ih"
        if ch in "uv":
            return "ou"
    return "ih"


def _units(word: str) -> list[str]:
    """Split a spoken word into syllable-like units, each mapped to a viseme."""
    visemes: list[str] = []
    for token in re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+|\d", word):
        if token.isdigit():
            token = _DIGITS[int(token)]
        if _CJK.match(token):
            visemes.append(final_to_viseme(lazy_pinyin(token, style=Style.FINALS, strict=False)[0]))
        else:
            vowels = [_LATIN_VOWEL[c] for c in token.lower() if c in _LATIN_VOWEL]
            visemes.extend(vowels[:: max(1, len(vowels) // 3)] or ["ih"])
    return visemes


def boundaries_to_visemes(boundaries: list[dict]) -> list[dict]:
    """Spread each word's duration evenly over its syllables."""
    track: list[dict] = []
    for b in boundaries:
        units = _units(b["text"])
        if not units:
            continue
        start = b["offset"] / TICKS_PER_SECOND
        step = (b["duration"] / TICKS_PER_SECOND) / len(units)
        for i, v in enumerate(units):
            track.append({"start": round(start + i * step, 3), "end": round(start + (i + 1) * step, 3), "v": v})
    return track


async def synthesize(text: str, voice: str = VOICE) -> dict:
    communicate = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    audio = bytearray()
    boundaries: list[dict] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            boundaries.append(chunk)
    return {
        "audio": base64.b64encode(bytes(audio)).decode("ascii"),
        "mime": "audio/mpeg",
        "visemes": boundaries_to_visemes(boundaries),
    }
