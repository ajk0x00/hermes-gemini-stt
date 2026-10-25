"""Automatic Gemini audio-tag rewriting for TTS.

Gemini 3.8 TTS treats the input text as a verbatim transcript and recognises only angle-bracket
vocal events (``<laugh>``, ``<sigh>``, ``<short pause>``). Hermes' built-in Gemini provider only
rewrites 3.1-style square-bracket tags, so the plugin owns the 3.8 rewrite.

The tagger is intentionally best-effort: a small, fast text model proposes a tagged transcript,
then :func:`sanitize_tagged_text` proves that only tags were added. Any failure or unexpected edit
falls back to the original text so tagging can never break speech.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("hermes.plugins.gemini_stt")

# Tags that Gemini 3.8 documents for point-in-time vocal events. Keeping this list tight also
# protects against a model inventing markup that the TTS engine would speak aloud.
_ANGLE_TAGS = (
    "argh", "breath", "cackle", "cheer", "chuckle", "chuckles", "cough", "cry", "gasp",
    "giggle", "groan", "growl", "grunt", "grr", "hiss", "laugh", "laughter", "moan", "pant",
    "pff", "phew", "scream", "shout", "shriek", "sigh", "sighs", "sneeze", "snicker", "snort",
    "sob", "throat-clearing", "tsk", "whimper", "whisper", "whispers", "whispering", "yawn",
    "short pause", "long pause", "heavy breath", "exhales",
)
_SQUARE_TAGS = (
    "amusement", "anger", "annoyance", "awe", "curiosity", "determination", "enthusiasm",
    "excitement", "frustration", "hope", "interest", "laughs", "nervousness", "positive",
    "sadness", "sarcastic", "sighs", "surprise", "tension", "whispers", "whispering",
    "short pause", "long pause", "breath",
)

_ALLOWED = {
    "angle": frozenset(tag.lower() for tag in _ANGLE_TAGS),
    "square": frozenset(tag.lower() for tag in _SQUARE_TAGS),
}

_TAG_RE = re.compile(
    r"<\s*([A-Za-z][A-Za-z0-9 _-]{0,30})\s*/?\s*>|\[\s*([A-Za-z][A-Za-z0-9 _-]{0,30})\s*\]"
)
_CLOSING_ANGLE_RE = re.compile(r"</\s*[A-Za-z][^>]{0,40}>")
_ANGLE_ALIASES = {"laughs": "laugh", "sighs": "sigh", "chuckles": "chuckle",
                  "whisper": "whispers", "whispering": "whispers"}
_SQUARE_ALIASES = {"laugh": "laughs", "sigh": "sighs", "chuckle": "laughs",
                   "whisper": "whispers", "whispering": "whispers"}
_FENCE_RE = re.compile(r"^```[A-Za-z0-9_-]*[ \t]*\r?\n(.*)\r?\n```$", re.DOTALL)
_LABEL_RE = re.compile(r"^(?:tagged|transcript|output)[ \t]*:[ \t]*\r?\n+", re.IGNORECASE)
_WORD_RE = re.compile(r"\w+(?:['\u2019]\w+)?", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TAG_ANY_RE = re.compile(r"<[A-Za-z][^>]{0,40}>|\[[A-Za-z][^\]]{0,40}\]")


def tag_style(model: str) -> str | None:
    """``"angle"`` for Gemini 3.8 TTS, ``"square"`` for 3.1, else ``None``."""
    normalized = (model or "").strip().lower().rsplit("/", 1)[-1]
    if "gemini-3.8" in normalized and "tts" in normalized:
        return "angle"
    if "gemini-3.1" in normalized and "tts" in normalized:
        return "square"
    return None


def has_tags(text: str, *, style: str | None = None) -> bool:
    """True when the text already contains an audio tag.

    With a *style*, only tags valid for that model count. That keeps natural angle-bracket
    placeholders such as ``<variable>`` from being mistaken for operator-supplied tags.
    """
    value = text or ""
    if style is None:
        return bool(_TAG_ANY_RE.search(value))
    allowed = _allowed_for(style)
    aliases = _ANGLE_ALIASES if style == "angle" else _SQUARE_ALIASES
    for match in _TAG_RE.finditer(value):
        name = " ".join((match.group(1) or match.group(2) or "").lower().split())
        if aliases.get(name, name) in allowed:
            return True
    return False


def canonicalize_tags(text: str, *, style: str) -> str:
    """Normalise operator-supplied tags without deleting unrelated bracketed text."""
    if style not in _ALLOWED:
        return text
    return _canonical_tags(text, style, drop_unknown=False)


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _words(text: str) -> list[str]:
    value = _CLOSING_ANGLE_RE.sub(" ", text or "")
    value = _TAG_RE.sub(" ", value)
    return [match.group(0).lower() for match in _WORD_RE.finditer(value)]


def _allowed_for(style: str) -> frozenset[str]:
    return _ALLOWED.get(style, frozenset())


def _tag(payload: str, style: str) -> str:
    value = " ".join(payload.lower().split())
    return f"<{value}>" if style == "angle" else f"[{value}]"


def _strip_wrapping(candidate: str) -> str:
    value = (candidate or "").strip()
    fence = _FENCE_RE.match(value)
    if fence:
        value = fence.group(1).strip()
    value = _LABEL_RE.sub("", value, count=1)
    return value.strip()


def _canonical_tags(text: str, style: str, *, drop_unknown: bool = True) -> str:
    """Convert either tag spelling to the target style.

    ``drop_unknown`` is true for tagger output, where anything unrecognised is untrusted markup.
    It is false for operator text, where a natural ``<variable>`` must survive verbatim.
    """
    allowed = _allowed_for(style)
    aliases = _ANGLE_ALIASES if style == "angle" else _SQUARE_ALIASES

    value = text
    if drop_unknown:
        # Closing tags are never valid for Gemini TTS. Remove them before looking for openings.
        value = _CLOSING_ANGLE_RE.sub(" ", value)
        value = re.sub(r"\[\s*/\s*[A-Za-z][^\]]*\]", " ", value)

    def replace(match: re.Match[str]) -> str:
        raw = match.group(1) or match.group(2) or ""
        name = " ".join(raw.lower().split())
        name = aliases.get(name, name)
        if name in allowed:
            return f" {_tag(name, style)} "
        return " " if drop_unknown else match.group(0)

    # One combined pass: every letter-led bracket expression is canonicalised, kept, or dropped.
    value = _TAG_RE.sub(replace, value)
    return " ".join(value.split())


def _one_tag_per_sentence(text: str) -> str:
    """Keep at most one tag per sentence; more than that tends to sound over-directed."""
    sentences: list[str] = []
    for sentence in _SENTENCE_RE.split(text):
        if not sentence.strip():
            continue
        tags = list(re.finditer(r"<[^>]+>|\[[^\]]+\]", sentence))
        if len(tags) > 1:
            first = tags[0]
            sentence = sentence[: first.end()] + re.sub(
                r"<[^>]+>|\[[^\]]+\]", "", sentence[first.end():]
            )
            sentence = " ".join(sentence.split())
        sentences.append(sentence)
    return " ".join(sentences).strip()


def _limit_tags(text: str, max_tags: int) -> str:
    if max_tags <= 0:
        return re.sub(r"<[^>]+>|\[[^\]]+\]", " ", text)
    count = 0
    out: list[str] = []
    position = 0
    for match in re.finditer(r"<[^>]+>|\[[^\]]+\]", text):
        out.append(text[position:match.start()])
        count += 1
        if count <= max_tags:
            out.append(match.group(0))
        position = match.end()
    out.append(text[position:])
    return " ".join("".join(out).split())


def sanitize_tagged_text(original: str, candidate: str, *, style: str) -> str | None:
    """Return a safe tagged transcript, or ``None`` when the rewrite is unusable.

    Safety means: every original word is still present, in order, with no additions; tags use
    the model's supported spelling; closing/unknown markup is gone; and the text is not turned
    into a wall of directions.
    """
    if style not in _ALLOWED:
        return None
    source = (original or "").strip()
    if not source:
        return None

    cleaned = _canonical_tags(_strip_wrapping(candidate), style)
    if not cleaned:
        return None
    if _words(source) != _words(cleaned):
        logger.debug("Gemini TTS tagger changed the transcript; falling back to untagged text")
        return None

    # One tag per sentence, and a hard density ceiling (~1 tag per 15 words).
    cleaned = _one_tag_per_sentence(cleaned)
    limit = max(1, word_count(source) // 15)
    cleaned = _limit_tags(cleaned, limit)

    if _words(source) != _words(cleaned):
        return None
    if not re.search(r"<[^>]+>|\[[^\]]+\]", cleaned):
        return None
    return cleaned.strip()


def build_tag_payload(text: str, *, style: str) -> dict[str, Any]:
    """GenerateContent request body for the tagging model."""
    if style == "angle":
        tag_examples = "<laugh>, <chuckle>, <sigh>, <breath>, <gasp>, <short pause>"
        rules = (
            "Use angle-bracket tags only. Never use closing tags or square brackets. "
            "Use at most one tag per sentence. Do not wrap a whole sentence in a tag. "
            "Do not tag a sentence shorter than 12 words."
        )
    else:
        tag_examples = "[laughs], [sighs], [whispers], [excited], [curious], [short pause]"
        rules = (
            "Use square-bracket tags only. Never use closing tags or angle brackets. "
            "Use at most one tag per sentence. Do not wrap a whole sentence in a tag. "
            "Do not tag a sentence shorter than 12 words."
        )

    system_prompt = (
        "You are a performance tagger for Gemini text-to-speech. Return only the transcript. "
        "Preserve every word, number, name and their order exactly; never add, remove, rephrase, "
        "translate or explain anything. Insert a tag only where the text clearly calls for a "
        "vocal event or pause. "
        f"Use these tags when appropriate: {tag_examples}. {rules} "
        "If no tag is clearly justified, return the transcript unchanged."
    )
    return {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "generationConfig": {
            "temperature": 0.25,
            "maxOutputTokens": max(256, min(4096, len(text) // 2 + 128)),
            "candidateCount": 1,
        },
    }


def extract_tagged_text(payload: dict[str, Any]) -> str:
    """First text part from a GenerateContent response; ``""`` when blocked/malformed."""
    if not isinstance(payload, dict):
        return ""
    feedback = payload.get("promptFeedback") or {}
    if feedback.get("blockReason"):
        return ""
    candidates = payload.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return ""
    candidate = candidates[0]
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    text = "".join(
        str(part.get("text") or "") for part in parts if isinstance(part, dict)
    )
    return text.strip()
