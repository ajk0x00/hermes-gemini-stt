"""Typed plugin settings, read once from the documented ``ctx.get_config`` surface.

Values live under ``plugins.entries.gemini-stt.settings.*`` in config.yaml and are rendered by
the Desktop settings form from the manifest's ``config_schema``. The model and language hints
do NOT live here: Hermes resolves ``stt.gemini.model`` / ``stt.gemini.language`` and passes them
to ``transcribe()`` per call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_TTS_MODEL = "gemini-3.8-flash-tts"
DEFAULT_TTS_VOICE = "Kore"
DEFAULT_TTS_TAG_MODEL = "gemini-3.1-flash-lite"

_THINKING_LEVELS = frozenset({"low", "medium", "high"})
_TRANSCRIPTION_MODES = frozenset({"verbatim", "smart"})
_LANGUAGE_MODES = frozenset({"hint", "auto"})


@dataclass(frozen=True)
class GeminiSettings:
    """Snapshot of the plugin's settings; changes require a plugin reload."""

    timeout: float = 180.0
    max_retries: int = 2
    thinking_level: str = "low"
    base_url: str = ""
    files_api_threshold_mb: float = 12.0
    transcription_mode: str = "verbatim"
    diarize: bool = False
    language_mode: str = "hint"
    keep_uploaded_files: bool = False
    api_key: str = ""
    user_agent: str = "hermes-gemini-stt/1.1.0"


def _read(ctx: Any, key: str, default: Any) -> Any:
    """One settings lookup; a broken/absent ctx degrades to the default rather than blocking load."""
    if ctx is None:
        return default
    try:
        value = ctx.get_config(key, default)
    except Exception:
        return default
    return default if value is None else value


def _number(ctx: Any, key: str, default: float, low: float, high: float) -> float:
    try:
        value = float(_read(ctx, key, default))
    except (TypeError, ValueError):
        return default
    return min(high, max(low, value))


def _flag(ctx: Any, key: str, default: bool) -> bool:
    raw = _read(ctx, key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off", ""}:
            return False
    return default


def _choice(ctx: Any, key: str, default: str, allowed: frozenset[str]) -> str:
    raw = _read(ctx, key, default)
    value = raw.strip().lower() if isinstance(raw, str) else ""
    return value if value in allowed else default


def _string(ctx: Any, key: str, default: str = "") -> str:
    raw = _read(ctx, key, default)
    return raw.strip() if isinstance(raw, str) else default


def load_settings(ctx: Any) -> GeminiSettings:
    """Build the immutable settings snapshot used for the life of this plugin load."""
    return GeminiSettings(
        timeout=_number(ctx, "timeout", 180.0, 10.0, 900.0),
        max_retries=int(_number(ctx, "max_retries", 2.0, 0.0, 5.0)),
        thinking_level=_choice(ctx, "thinking_level", "low", _THINKING_LEVELS),
        base_url=_string(ctx, "base_url", ""),
        files_api_threshold_mb=_number(ctx, "files_api_threshold_mb", 12.0, 1.0, 19.0),
        transcription_mode=_choice(ctx, "transcription_mode", "verbatim", _TRANSCRIPTION_MODES),
        diarize=_flag(ctx, "diarize", False),
        language_mode=_choice(ctx, "language_mode", "hint", _LANGUAGE_MODES),
        keep_uploaded_files=_flag(ctx, "keep_uploaded_files", False),
        api_key=_string(ctx, "api_key", ""),
    )


@dataclass(frozen=True)
class GeminiTTSSettings:
    """TTS-only knobs; auth and the endpoint are shared with the STT settings snapshot."""

    model: str = DEFAULT_TTS_MODEL
    voice: str = DEFAULT_TTS_VOICE
    style: str = ""
    audio_tags: bool = True
    audio_tags_streaming: bool = True
    tag_model: str = DEFAULT_TTS_TAG_MODEL
    tag_timeout: float = 5.0
    timeout: float = 120.0
    max_retries: int = 2
    ffmpeg_path: str = ""


def load_tts_settings(ctx: Any) -> GeminiTTSSettings:
    """Build the immutable TTS settings snapshot.

    Tagging is deliberately on by default: the plugin is meant to deliver Gemini 3.8 audio
    tags without any extra configuration. Operators can turn it off, or cap its latency.
    """
    return GeminiTTSSettings(
        model=_string(ctx, "tts_model", DEFAULT_TTS_MODEL) or DEFAULT_TTS_MODEL,
        voice=_string(ctx, "tts_voice", DEFAULT_TTS_VOICE) or DEFAULT_TTS_VOICE,
        style=_string(ctx, "tts_style", ""),
        audio_tags=_flag(ctx, "tts_audio_tags", True),
        audio_tags_streaming=_flag(ctx, "tts_audio_tags_streaming", True),
        tag_model=_string(ctx, "tts_tag_model", DEFAULT_TTS_TAG_MODEL) or DEFAULT_TTS_TAG_MODEL,
        tag_timeout=_number(ctx, "tts_tag_timeout", 5.0, 1.0, 30.0),
        timeout=_number(ctx, "tts_timeout", 120.0, 10.0, 900.0),
        max_retries=int(_number(ctx, "tts_max_retries", 2.0, 0.0, 5.0)),
        ffmpeg_path=_string(ctx, "tts_ffmpeg_path", ""),
    )
