"""Interactions API request builders for the two Gemini model families.

``gemini-3.5-transcribe`` takes an audio-only input plus ``transcription_config``; general
multimodal models (``gemini-3.8-flash``) take an instruction and a prompt hint block. Both
requests pin ``store: false`` so Google does not retain the interaction.
"""

from __future__ import annotations

import logging
from typing import Any

from ._settings import GeminiSettings
from ._text import SYSTEM_INSTRUCTION, build_hint_block, build_language_hint, vocabulary_terms

logger = logging.getLogger("hermes.plugins.gemini_stt")

_MB = 1024 * 1024
INLINE_REQUEST_LIMIT = 20 * _MB
_JSON_OVERHEAD = 4096


def is_transcribe_model(model: str) -> bool:
    return "-transcribe" in (model or "").lower()


def audio_block(*, mime: str, data: str | None = None, uri: str | None = None) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "audio", "mime_type": mime}
    if data is not None:
        block["data"] = data
    elif uri is not None:
        block["uri"] = uri
    else:  # pragma: no cover - callers always pass one
        raise ValueError("audio_block requires data or uri")
    return block


def inline_budget_bytes(settings: GeminiSettings, *, prompt: str | None, language: str | None) -> int:
    """Largest *raw* file we will base64 inline, before the Files API takes over.

    The 20 MB cap is on the request, and base64 inflates raw bytes by 4/3. This returns a raw-byte
    budget derived from both that request cap and the operator's ``files_api_threshold_mb``, so
    callers compare ``os.path.getsize()`` directly rather than doing their own base64 math.
    """
    overhead = (
        len(SYSTEM_INSTRUCTION)
        + len(build_hint_block(prompt))
        + len(build_language_hint(language))
        + _JSON_OVERHEAD
    )
    request_room = INLINE_REQUEST_LIMIT - overhead
    # 4 * ceil(n / 3) <= request_room, with slack for the ceil and the audio block's own JSON.
    raw_room = max(1, (request_room * 3) // 4 - 4)
    operator_room = int(settings.files_api_threshold_mb * _MB)
    return max(1, min(raw_room, operator_room))


def build_payload(
    *,
    model: str,
    mime: str,
    settings: GeminiSettings,
    prompt: str | None = None,
    language: str | None = None,
    data: str | None = None,
    uri: str | None = None,
) -> dict[str, Any]:
    """Full Interactions request body for one recording."""
    audio = audio_block(mime=mime, data=data, uri=uri)
    if is_transcribe_model(model):
        return _transcribe_payload(model=model, audio=audio, settings=settings, prompt=prompt, language=language)
    return _general_payload(
        model=model, audio=audio, settings=settings, prompt=prompt, language=language,
    )


def _transcribe_payload(
    *, model: str, audio: dict[str, Any], settings: GeminiSettings,
    prompt: str | None, language: str | None,
) -> dict[str, Any]:
    language_codes: list[str] = []
    if settings.language_mode == "hint" and (language or "").strip():
        language_codes = [(language or "").strip()]

    mode: Any = settings.transcription_mode
    if settings.diarize:
        if settings.transcription_mode != "verbatim":
            logger.warning(
                "Gemini STT: diarization requires verbatim mode; ignoring transcription_mode=%r",
                settings.transcription_mode,
            )
        # Object form is required to carry diarization, and only verbatim supports it.
        mode = {"type": "verbatim", "diarization_mode": "speaker"}

    transcription_config: dict[str, Any] = {"language_codes": language_codes, "mode": mode}
    vocabulary = vocabulary_terms(prompt)
    if vocabulary and not settings.diarize:
        # The API rejects custom_vocabulary combined with diarization/timestamps.
        transcription_config["custom_vocabulary"] = vocabulary

    return {
        "model": model,
        "input": [audio],
        "generation_config": {"transcription_config": transcription_config},
        "store": False,
    }


def _general_payload(
    *, model: str, audio: dict[str, Any], settings: GeminiSettings,
    prompt: str | None, language: str | None,
) -> dict[str, Any]:
    text_parts = [build_hint_block(prompt)]
    if settings.language_mode == "hint":
        text_parts.append(build_language_hint(language))
    hint_text = "\n\n".join(part for part in text_parts if part)

    blocks: list[dict[str, Any]] = []
    if hint_text:
        blocks.append({"type": "text", "text": hint_text})
    blocks.append(audio)

    return {
        "model": model,
        "system_instruction": SYSTEM_INSTRUCTION,
        "input": blocks,
        "generation_config": {"thinking_level": settings.thinking_level},
        "store": False,
    }
