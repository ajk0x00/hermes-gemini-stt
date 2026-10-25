"""Google Gemini text-to-speech provider for Hermes.

Registered as ``gemini-tts`` because ``gemini`` is a reserved built-in TTS name. The provider
speaks through Gemini 3.8 TTS (or 3.1/2.5), handles the 3.8 WAV response without double-wrapping
it, and adds 3.8 angle-bracket audio tags by default through a small, fast tagging model.
"""

from __future__ import annotations

import base64
import glob
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

from agent.tts_provider import TTSProvider

from ._auth import normalise_model, redact, resolve_api_key, resolve_base_url
from ._settings import (
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_TAG_MODEL,
    DEFAULT_TTS_VOICE,
    GeminiSettings,
    GeminiTTSSettings,
    load_settings,
    load_tts_settings,
)
from ._transport import GeminiAPIError, GeminiTransport
from ._tts_tags import (
    build_tag_payload,
    canonicalize_tags,
    extract_tagged_text,
    has_tags,
    sanitize_tagged_text,
    tag_style,
    word_count,
)

logger = logging.getLogger("hermes.plugins.gemini_stt")

TTS_PROVIDER_NAME = "gemini-tts"
_TTS_SAMPLE_RATE = 24_000
_TTS_CHANNELS = 1
_TTS_SAMPLE_WIDTH = 2
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_MAX_STREAM_BYTES = 32 * 1024 * 1024
_MAX_TAG_CACHE = 256
_CONNECT_TIMEOUT = 10.0

_PREBUILT_VOICES: tuple[tuple[str, str], ...] = (
    ("Kore", "Firm"),
    ("Zephyr", "Bright"),
    ("Puck", "Upbeat"),
    ("Charon", "Informative"),
    ("Fenrir", "Excitable"),
    ("Leda", "Youthful"),
    ("Orus", "Firm"),
    ("Aoede", "Breezy"),
    ("Callirrhoe", "Easy-going"),
    ("Autonoe", "Bright"),
    ("Enceladus", "Breathy"),
    ("Iapetus", "Clear"),
    ("Umbriel", "Easy-going"),
    ("Algieba", "Smooth"),
    ("Despina", "Smooth"),
    ("Erinome", "Clear"),
    ("Algenib", "Gravelly"),
    ("Rasalgethi", "Informative"),
    ("Laomedeia", "Upbeat"),
    ("Achernar", "Soft"),
    ("Alnilam", "Firm"),
    ("Schedar", "Even"),
    ("Gacrux", "Mature"),
    ("Pulcherrima", "Forward"),
    ("Achird", "Friendly"),
    ("Zubenelgenubi", "Casual"),
    ("Vindemiatrix", "Gentle"),
    ("Sadachbia", "Lively"),
    ("Sadaltager", "Knowledgeable"),
    ("Sulafat", "Warm"),
)

_MODEL_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "gemini-3.8-flash-tts",
        "display": "Gemini 3.8 Flash TTS",
        "default": True,
        "note": "Flagship creative TTS; angle-bracket vocal tags and WAV unary output.",
    },
    {
        "id": "gemini-3.8-flash-lite-tts",
        "display": "Gemini 3.8 Flash-Lite TTS",
        "note": "Faster, cheaper 3.8 TTS for read-aloud and high-volume speech.",
    },
    {
        "id": "gemini-3.1-flash-tts-preview",
        "display": "Gemini 3.1 Flash TTS Preview",
        "note": "Square-bracket audio tags; L16 unary output.",
    },
    {
        "id": "gemini-2.5-flash-preview-tts",
        "display": "Gemini 2.5 Flash Preview TTS",
        "note": "Hermes' legacy default; L16 unary output.",
    },
    {
        "id": "gemini-2.5-pro-preview-tts",
        "display": "Gemini 2.5 Pro Preview TTS",
        "note": "Legacy pro voice; L16 unary output.",
    },
)

_PCM_RATE_RE = re.compile(r"rate\s*=\s*(\d+)", re.IGNORECASE)
_PCM_CHANNELS_RE = re.compile(r"channels\s*=\s*(\d+)", re.IGNORECASE)
_TAG_WORD_RE = re.compile(r"<[^>]+>|\[[^\]]+\]")
_VALID_FORMATS = frozenset({"mp3", "wav", "ogg", "opus", "flac"})


def _is_38_tts(model: str) -> bool:
    normalized = (model or "").strip().lower().rsplit("/", 1)[-1]
    return "gemini-3.8" in normalized and "tts" in normalized


def _wrap_pcm(
    pcm: bytes, sample_rate: int = _TTS_SAMPLE_RATE,
    channels: int = _TTS_CHANNELS, sample_width: int = _TTS_SAMPLE_WIDTH,
) -> bytes:
    """Wrap headerless little-endian PCM in a minimal RIFF/WAVE container."""
    block_align = channels * sample_width
    fmt_chunk = struct.pack(
        "<4sIHHIIHH", b"fmt ", 16, 1, channels, sample_rate,
        sample_rate * block_align, block_align, sample_width * 8,
    )
    data_header = struct.pack("<4sI", b"data", len(pcm))
    riff_size = 4 + len(fmt_chunk) + len(data_header) + len(pcm)
    return struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE") + fmt_chunk + data_header + pcm


def _pcm_params(mime: str) -> tuple[int, int]:
    rate_match = _PCM_RATE_RE.search(mime or "")
    channels_match = _PCM_CHANNELS_RE.search(mime or "")
    rate = int(rate_match.group(1)) if rate_match else _TTS_SAMPLE_RATE
    channels = int(channels_match.group(1)) if channels_match else _TTS_CHANNELS
    return (rate if rate > 0 else _TTS_SAMPLE_RATE, channels if channels > 0 else _TTS_CHANNELS)


def _as_wav(raw: bytes, mime: str) -> bytes:
    """Return WAV bytes, wrapping L16 only when the model did not already send a RIFF file."""
    if not raw:
        raise RuntimeError("Gemini TTS returned empty audio data")
    if raw[:4] == b"RIFF" or "wav" in (mime or "").lower():
        return raw
    if "l16" in (mime or "").lower() or "pcm" in (mime or "").lower() or not mime:
        rate, channels = _pcm_params(mime)
        return _wrap_pcm(raw, sample_rate=rate, channels=channels)
    raise RuntimeError(f"Gemini TTS returned an unsupported audio format: {mime or 'unknown'}")


def _extract_audio(payload: dict[str, Any]) -> tuple[bytes, str]:
    if not isinstance(payload, dict):
        raise RuntimeError("Gemini TTS returned an unreadable response")
    error = payload.get("error")
    if isinstance(error, dict) and error.get("message"):
        raise RuntimeError(str(error["message"]))
    candidates = payload.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        feedback = payload.get("promptFeedback") or {}
        reason = feedback.get("blockReason")
        raise RuntimeError(f"Gemini TTS returned no candidates{f' ({reason})' if reason else ''}")
    candidate = candidates[0]
    content = candidate.get("content") or {}
    for part in content.get("parts") or []:
        if not isinstance(part, dict):
            continue
        inline = part.get("inlineData") or part.get("inline_data") or {}
        data = inline.get("data")
        if not data:
            continue
        mime = str(inline.get("mimeType") or inline.get("mime_type") or "")
        try:
            raw = base64.b64decode(data)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Gemini TTS returned invalid base64 audio") from exc
        if raw:
            return raw, mime
    finish = str(candidate.get("finishReason") or "").strip()
    raise RuntimeError(f"Gemini TTS response contained no audio{f' (finish={finish})' if finish else ''}")


class GeminiTTSProvider(TTSProvider):
    """Gemini speech synthesis with native 3.8 audio tags and streaming PCM."""

    name = TTS_PROVIDER_NAME
    display_name = "Google Gemini TTS"
    streams_pcm = True
    stream_sample_rate = _TTS_SAMPLE_RATE

    def __init__(self, ctx: Any = None) -> None:
        self._ctx = ctx
        self._settings: GeminiSettings = load_settings(ctx)
        self._tts: GeminiTTSSettings = load_tts_settings(ctx)
        self._tag_cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._tag_lock = threading.Lock()
        self._tag_failure_logged = False
        self._ffmpeg_warning_logged = False
        self._dependency_error = ""
        try:
            import httpx  # noqa: F401
        except Exception as exc:  # pragma: no cover - Hermes ships httpx
            self._dependency_error = str(exc)

    # ── catalog surface ────────────────────────────────────────────────────
    def is_available(self) -> bool:
        try:
            if self._dependency_error:
                return False
            return bool(resolve_api_key(self._settings))
        except Exception:
            return False

    def default_model(self) -> str:
        return self._tts.model

    def default_voice(self) -> str:
        return self._tts.voice

    def list_models(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in _MODEL_CATALOG]

    def list_voices(self) -> list[dict[str, Any]]:
        return [
            {"id": voice, "display": f"{voice} ({description})", "language": "multilingual"}
            for voice, description in _PREBUILT_VOICES
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

    @property
    def voice_compatible(self) -> bool:
        """Opt into voice-bubble delivery when ffmpeg can produce Ogg/Opus."""
        return bool(self._find_ffmpeg())

    # ── synthesis ──────────────────────────────────────────────────────────
    def synthesize(
        self, text: str, output_path: str, *, voice: str | None = None,
        model: str | None = None, speed: float | None = None, format: str = "mp3",
        **extra: Any,
    ) -> str:
        """Synthesize one chunk and return the final file path. Raises on failure."""
        api_key = ""
        try:
            source_text = (text or "").strip()
            if not source_text:
                raise ValueError("Gemini TTS received empty text")
            api_key = resolve_api_key(self._settings)
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY not set. Add it to ~/.hermes/.env or the plugin settings."
                )
            model_id = self._model_id(model)
            voice_id = self._voice_id(voice)
            base_url = resolve_base_url(self._settings)
            prepared = self._prepare_text(
                source_text, model_id=model_id, api_key=api_key, base_url=base_url,
                streaming=False,
            )
            payload = self._payload(prepared, model_id=model_id, voice_id=voice_id)
            started = time.monotonic()
            transport = GeminiTransport(
                api_key=api_key,
                base_url=base_url,
                api_root="",
                timeout=self._tts.timeout,
                max_retries=self._tts.max_retries,
                user_agent=self._settings.user_agent,
            )
            try:
                response = transport.request(
                    "POST", f"{base_url}/models/{model_id}:generateContent", json=payload,
                )
                if len(response.content) > _MAX_RESPONSE_BYTES:
                    raise RuntimeError("Gemini TTS response exceeded the 32 MiB safety cap")
                try:
                    body = response.json()
                except Exception:
                    raise RuntimeError("Gemini TTS returned a non-JSON response") from None
                raw, mime = _extract_audio(body)
                written = self._write_audio(raw, mime, output_path, format)
                logger.info(
                    "Gemini TTS: model=%s voice=%s chars=%d tags=%d mime=%s bytes=%d "
                    "latency=%.2fs output=%s",
                    model_id, voice_id, len(prepared), self._tag_count(prepared),
                    mime or "unknown", len(raw), time.monotonic() - started, written,
                )
                return written
            finally:
                transport.close()
        except GeminiAPIError as exc:
            raise RuntimeError(redact(str(exc), api_key)) from None
        except Exception as exc:
            message = redact(str(exc), api_key)
            if isinstance(exc, (ValueError, RuntimeError)):
                raise type(exc)(message) from None
            raise RuntimeError(f"Gemini TTS failed: {message}") from exc

    def stream(
        self, text: str, *, voice: str | None = None, model: str | None = None,
        format: str = "opus", **extra: Any,
    ) -> Iterator[bytes]:
        """Yield 24 kHz int16 mono PCM chunks from ``streamGenerateContent``."""
        api_key = ""
        try:
            source_text = (text or "").strip()
            if not source_text:
                raise ValueError("Gemini TTS received empty text")
            api_key = resolve_api_key(self._settings)
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY not set. Add it to ~/.hermes/.env or the plugin settings."
                )
            model_id = self._model_id(model)
            voice_id = self._voice_id(voice)
            base_url = resolve_base_url(self._settings)
            prepared = self._prepare_text(
                source_text, model_id=model_id, api_key=api_key, base_url=base_url,
                streaming=True,
            )
            payload = self._payload(prepared, model_id=model_id, voice_id=voice_id)
            headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
            url = f"{base_url}/models/{model_id}:streamGenerateContent"
            import httpx

            timeout = httpx.Timeout(
                self._tts.timeout, connect=min(_CONNECT_TIMEOUT, self._tts.timeout),
            )
            total = 0
            saw_audio = False
            with httpx.stream(
                "POST", url, params={"alt": "sse"}, headers=headers, json=payload,
                timeout=timeout,
            ) as response:
                if response.status_code >= 400:
                    detail = " ".join((response.read().decode("utf-8", "replace") or "").split())[:300]
                    raise RuntimeError(
                        f"Gemini TTS streaming API error ({response.status_code})"
                        + (f": {detail}" if detail else "")
                    )
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except ValueError:
                        continue
                    candidates = event.get("candidates") or []
                    if not candidates or not isinstance(candidates[0], dict):
                        continue
                    content = candidates[0].get("content") or {}
                    for part in content.get("parts") or []:
                        if not isinstance(part, dict):
                            continue
                        inline = part.get("inlineData") or part.get("inline_data") or {}
                        chunk = inline.get("data")
                        if not chunk:
                            continue
                        mime = str(inline.get("mimeType") or inline.get("mime_type") or "")
                        try:
                            pcm = base64.b64decode(chunk)
                        except (ValueError, TypeError):
                            logger.warning("Gemini TTS stream: skipping invalid base64 chunk")
                            continue
                        if not pcm:
                            continue
                        if "wav" in mime.lower() or pcm[:4] == b"RIFF":
                            raise RuntimeError(
                                "Gemini TTS streaming returned WAV; expected raw L16 PCM"
                            )
                        total += len(pcm)
                        if total > _MAX_STREAM_BYTES:
                            raise RuntimeError("Gemini TTS stream exceeded the 32 MiB safety cap")
                        saw_audio = True
                        yield pcm
            if not saw_audio:
                raise RuntimeError("Gemini TTS stream returned no audio")
        except GeminiAPIError as exc:
            raise RuntimeError(redact(str(exc), api_key)) from None
        except Exception as exc:
            message = redact(str(exc), api_key)
            if isinstance(exc, (ValueError, RuntimeError)):
                raise type(exc)(message) from None
            raise RuntimeError(f"Gemini TTS streaming failed: {message}") from exc

    # ── text preparation / tagging ────────────────────────────────────────
    def _model_id(self, model: str | None) -> str:
        return normalise_model(
            model or self._tts.model or DEFAULT_TTS_MODEL, default=DEFAULT_TTS_MODEL,
        )

    def _voice_id(self, voice: str | None) -> str:
        value = (voice or self._tts.voice or DEFAULT_TTS_VOICE).strip()
        if not value or any(ch in value for ch in "\r\n"):
            raise ValueError("Gemini TTS voice must be a single non-empty line")
        return value[:200]

    def _prepare_text(
        self, text: str, *, model_id: str, api_key: str, base_url: str,
        streaming: bool,
    ) -> str:
        if not self._tts.audio_tags:
            return text
        if streaming and not self._tts.audio_tags_streaming:
            return text
        style = tag_style(model_id)
        if style is None:
            return text
        if has_tags(text, style=style):
            # Respect manual tags, but translate 3.1 square brackets to 3.8 angle brackets.
            return canonicalize_tags(text, style=style) or text
        if word_count(text) < 12:
            return text

        key = (text, style)
        with self._tag_lock:
            cached = self._tag_cache.get(key)
            if cached is not None:
                self._tag_cache.move_to_end(key)
                return cached

        tagged = self._run_tagger(text, style=style, api_key=api_key, base_url=base_url)
        if tagged is None:
            return text
        with self._tag_lock:
            self._tag_cache[key] = tagged
            self._tag_cache.move_to_end(key)
            while len(self._tag_cache) > _MAX_TAG_CACHE:
                self._tag_cache.popitem(last=False)
        return tagged

    def _run_tagger(
        self, text: str, *, style: str, api_key: str, base_url: str,
    ) -> str | None:
        model = normalise_model(self._tts.tag_model, default=DEFAULT_TTS_TAG_MODEL)
        started = time.monotonic()
        transport = GeminiTransport(
            api_key=api_key,
            base_url=base_url,
            api_root="",
            timeout=self._tts.tag_timeout,
            max_retries=0,
            user_agent=self._settings.user_agent,
        )
        try:
            response = transport.request(
                "POST", f"{base_url}/models/{model}:generateContent",
                json=build_tag_payload(text, style=style),
            )
            if len(response.content) > 1024 * 1024:
                raise RuntimeError("tagging response exceeded 1 MiB")
            body = response.json()
            candidate = extract_tagged_text(body)
            tagged = sanitize_tagged_text(text, candidate, style=style)
            if tagged:
                logger.info(
                    "Gemini TTS tags: model=%s source_chars=%d tags=%d latency=%.2fs",
                    model, len(text), self._tag_count(tagged), time.monotonic() - started,
                )
            return tagged
        except Exception as exc:
            if not self._tag_failure_logged:
                self._tag_failure_logged = True
                logger.warning(
                    "Gemini TTS tagger unavailable; continuing with untagged text: %s",
                    redact(str(exc), api_key),
                )
            else:
                logger.debug("Gemini TTS tagger failed: %s", redact(str(exc), api_key))
            return None
        finally:
            transport.close()

    @staticmethod
    def _tag_count(text: str) -> int:
        return len(_TAG_WORD_RE.findall(text or ""))

    # ── request / response handling ───────────────────────────────────────
    def _payload(self, text: str, *, model_id: str, voice_id: str) -> dict[str, Any]:
        part: dict[str, Any] = {"text": text}
        style = self._tts.style.strip()
        if style and _is_38_tts(model_id):
            # 3.8 separates sustained direction from the verbatim transcript.
            part["speech_metadata"] = {"style": style[:500]}
        elif style and not _is_38_tts(model_id):
            logger.debug("tts_style is only applied to Gemini 3.8 TTS models; ignoring for %s", model_id)
        return {
            "contents": [{"parts": [part]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice_id}}
                },
            },
        }

    def _write_audio(
        self, raw: bytes, mime: str, output_path: str, requested_format: str,
    ) -> str:
        wav_bytes = _as_wav(raw, mime)
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        desired = (requested_format or "").strip().lower()
        if desired not in _VALID_FORMATS:
            desired = path.suffix.lower().lstrip(".")
        if desired not in _VALID_FORMATS:
            desired = "wav"

        if desired == "wav":
            target = path if path.suffix.lower() == ".wav" else path.with_suffix(".wav")
            target.write_bytes(wav_bytes)
            return str(target)

        ffmpeg = self._find_ffmpeg()
        if not ffmpeg:
            target = path.with_suffix(".wav")
            target.write_bytes(wav_bytes)
            logger.warning(
                "ffmpeg not found; wrote WAV to %s instead of converting to %s", target, desired,
            )
            return str(target)

        extension = ".ogg" if desired in {"ogg", "opus"} else f".{desired}"
        target = path if path.suffix.lower() == extension else path.with_suffix(extension)
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                handle.write(wav_bytes)
                tmp_path = handle.name
            args = [ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", tmp_path]
            if desired in {"ogg", "opus"}:
                args += [
                    "-c:a", "libopus", "-ac", "1", "-b:a", "48k", "-vbr", "on",
                    "-application", "voip", "-compression_level", "10", str(target),
                ]
            elif desired == "mp3":
                args += ["-c:a", "libmp3lame", "-b:a", "128k", str(target)]
            elif desired == "flac":
                args += ["-c:a", "flac", str(target)]
            else:
                args += [str(target)]
            completed = subprocess.run(
                args, capture_output=True, timeout=120, check=False,
            )
            if completed.returncode != 0:
                stderr = (completed.stderr or b"").decode("utf-8", "replace")[:300]
                raise RuntimeError(f"ffmpeg conversion failed: {stderr}")
            if not target.is_file() or target.stat().st_size == 0:
                raise RuntimeError("ffmpeg produced no output")
            return str(target)
        except Exception as exc:
            target = path.with_suffix(".wav")
            target.write_bytes(wav_bytes)
            logger.warning(
                "ffmpeg conversion to %s failed (%s); wrote WAV to %s", desired, exc, target,
            )
            return str(target)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _find_ffmpeg(self) -> str | None:
        configured = (self._tts.ffmpeg_path or "").strip()
        if configured:
            candidate = Path(configured).expanduser()
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
            if not self._ffmpeg_warning_logged:
                self._ffmpeg_warning_logged = True
                logger.warning("Configured Gemini TTS ffmpeg path is not executable: %s", candidate)
        found = shutil.which("ffmpeg")
        if found:
            return found
        hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
        for name in ("ffmpeg", "ffmpeg.exe"):
            matches = sorted(
                glob.glob(os.path.join(hermes_home, "tools", "*", "bin", name)), reverse=True,
            )
            for match in matches:
                if os.path.isfile(match) and os.access(match, os.X_OK):
                    return match
        return None
