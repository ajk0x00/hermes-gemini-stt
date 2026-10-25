"""Gemini voice providers for Hermes: speech-to-text and text-to-speech.

STT registers through ``ctx.register_transcription_provider`` and is selected with
``stt.provider: gemini``. TTS registers through ``ctx.register_tts_provider`` under
``tts.provider: gemini-tts`` (the built-in owns the ``gemini`` TTS name). Model and language
come from ``stt.gemini.*`` via the dispatcher; everything else comes from
``plugins.entries.gemini-stt.settings.*``.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any

from agent.transcription_provider import TranscriptionProvider

from ._auth import api_root, normalise_model, redact, resolve_api_key, resolve_base_url
from ._files import delete_file, upload_audio, wait_until_active
from ._mime import mime_for, needs_video_probe, probe_has_video
from ._parse import (
    decode_json, extract_transcript, is_terminal_failure, response_error, status_of,
)
from ._payload import build_payload, inline_budget_bytes
from ._settings import DEFAULT_MODEL, load_settings
from ._text import cleanup_transcript
from ._transport import GeminiAPIError, GeminiTransport
from ._tts import GeminiTTSProvider, TTS_PROVIDER_NAME

logger = logging.getLogger("hermes.plugins.gemini_stt")

PROVIDER_NAME = "gemini"


def _ok(transcript: str) -> dict[str, Any]:
    return {"success": True, "transcript": transcript, "provider": PROVIDER_NAME}


def _error(message: str) -> dict[str, Any]:
    return {"success": False, "transcript": "", "provider": PROVIDER_NAME, "error": message}


class GeminiTranscriptionProvider(TranscriptionProvider):
    """Google Gemini backend; ``gemini-3.8-flash`` by default, ASR-tuned models supported."""

    name = PROVIDER_NAME
    display_name = "Google Gemini"

    def __init__(self, ctx: Any = None) -> None:
        self._ctx = ctx
        self._settings = load_settings(ctx)
        self._dependency_error = ""
        try:
            import httpx  # noqa: F401
        except Exception as exc:  # pragma: no cover - Hermes always ships httpx
            self._dependency_error = str(exc)

    # ── catalog surface ────────────────────────────────────────────────────
    def is_available(self) -> bool:
        """Key present and HTTP client importable. No network, never raises."""
        try:
            if self._dependency_error:
                return False
            return bool(resolve_api_key(self._settings))
        except Exception:
            return False

    def default_model(self) -> str:
        return DEFAULT_MODEL

    def list_models(self) -> list[dict[str, Any]]:
        return [
            {"id": "gemini-3.8-flash", "display": "Gemini 3.8 Flash", "default": True,
             "note": "General multimodal model; prompt-driven verbatim transcription."},
            {"id": "gemini-3.5-transcribe", "display": "Gemini 3.5 Transcribe",
             "note": "ASR-tuned: language hints, vocabulary biasing, diarization."},
        ]

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "",
            "tag": "audio",
            "env_vars": [
                {
                    "key": "GEMINI_API_KEY",
                    "prompt": "Gemini API key",
                    "url": "https://aistudio.google.com/app/apikey",
                }
            ],
        }

    # ── transcription ──────────────────────────────────────────────────────
    def transcribe(
        self, file_path: str, *, model: str | None = None,
        language: str | None = None, **extra: Any,
    ) -> dict[str, Any]:
        """Transcribe one audio file; always returns the Hermes envelope and never raises."""
        api_key = ""
        try:
            api_key = resolve_api_key(self._settings)
            return self._transcribe(
                file_path, model=model, language=language,
                prompt=extra.get("prompt"), api_key=api_key,
            )
        except GeminiAPIError as exc:
            message = redact(str(exc), api_key)
            logger.warning("Gemini STT request failed: %s", message)
            return _error(message)
        except Exception as exc:  # contract: transcribe() must not raise
            message = redact(f"Gemini transcription failed: {exc}", api_key)
            logger.warning("Gemini STT crashed: %s", message, exc_info=True)
            return _error(message)

    def _transcribe(
        self, file_path: str, *, model: str | None, language: str | None,
        prompt: str | None, api_key: str,
    ) -> dict[str, Any]:
        if not api_key:
            return _error(
                "Gemini is not configured: set GEMINI_API_KEY (or GOOGLE_API_KEY) in "
                "~/.hermes/.env and restart Hermes."
            )
        path = Path(file_path)
        if not path.is_file():
            return _error(f"Audio file not found: {file_path}")
        mime = mime_for(file_path)
        if mime is None:
            return _error(
                f"Gemini STT does not support {path.suffix or 'this file type'}; "
                "convert it to mp3, m4a, wav, ogg/opus, webm, aac or flac."
            )
        if needs_video_probe(file_path) and probe_has_video(file_path) is True:
            return _error(
                "That file contains a video stream; Gemini STT accepts audio only. "
                "Extract the audio track first."
            )

        settings = self._settings
        model_id = normalise_model(model)
        base_url = resolve_base_url(settings)
        transport = GeminiTransport(
            api_key=api_key,
            base_url=base_url,
            api_root=api_root(base_url),
            timeout=settings.timeout,
            max_retries=settings.max_retries,
            user_agent=settings.user_agent,
        )
        started = time.monotonic()
        size = path.stat().st_size
        try:
            use_files = size > inline_budget_bytes(settings, prompt=prompt, language=language)
            logger.info(
                "Gemini STT: model=%s mime=%s bytes=%d transport=%s",
                model_id, mime, size, "files" if use_files else "inline",
            )
            if use_files:
                body = self._transcribe_via_files(
                    transport, path, mime, model_id, settings, prompt, language,
                )
            else:
                data = base64.b64encode(path.read_bytes()).decode("ascii")
                payload = build_payload(
                    model=model_id, mime=mime, settings=settings,
                    prompt=prompt, language=language, data=data,
                )
                body = decode_json(
                    transport.request("POST", f"{base_url}/interactions", json=payload)
                )
            return self._envelope(body, model_id, started)
        finally:
            transport.close()

    def _transcribe_via_files(
        self, transport: GeminiTransport, path: Path, mime: str, model_id: str,
        settings: Any, prompt: str | None, language: str | None,
    ) -> dict[str, Any]:
        file = upload_audio(transport, str(path), mime)
        try:
            file = wait_until_active(transport, file)
            uri = str(file.get("uri") or "").strip()
            if not uri:
                raise GeminiAPIError("Gemini Files API did not return a file URI.")
            payload = build_payload(
                model=model_id, mime=mime, settings=settings,
                prompt=prompt, language=language, uri=uri,
            )
            return decode_json(
                transport.request("POST", f"{transport.base_url}/interactions", json=payload)
            )
        finally:
            if not settings.keep_uploaded_files:
                delete_file(transport, file.get("name"))

    def _envelope(self, body: dict[str, Any], model_id: str, started: float) -> dict[str, Any]:
        status = status_of(body)
        transcript = cleanup_transcript(extract_transcript(body))
        error = response_error(body)
        latency = time.monotonic() - started
        if is_terminal_failure(status):
            message = error or f"Gemini interaction ended with status '{status}'."
            logger.warning("Gemini STT: model=%s status=%s error=%s", model_id, status, message)
            return _error(message)
        if error and not transcript:
            return _error(error)
        if error:
            logger.warning("Gemini STT: partial transcript (status=%s): %s", status, error)
        if not transcript:
            logger.warning("Gemini STT: model=%s returned no text (status=%s)", model_id, status)
        logger.info(
            "Gemini STT: ok model=%s status=%s chars=%d latency=%.2fs",
            model_id, status, len(transcript), latency,
        )
        return _ok(transcript)


def register(ctx: Any) -> None:
    """Hermes plugin entry point: STT plus the plugin-owned Gemini TTS provider."""
    ctx.register_transcription_provider(GeminiTranscriptionProvider(ctx))
    ctx.register_tts_provider(GeminiTTSProvider(ctx))


__all__ = [
    "GeminiTranscriptionProvider",
    "GeminiTTSProvider",
    "PROVIDER_NAME",
    "TTS_PROVIDER_NAME",
    "register",
]
