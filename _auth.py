"""Credential, base-URL and model-id resolution.

Precedence is deliberately small and explicit: an operator-set plugin value first, then the two
environment spellings Hermes already documents for Google AI Studio, then the public endpoint.
"""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlsplit, urlunsplit

from ._settings import DEFAULT_BASE_URL, DEFAULT_MODEL, GeminiSettings

logger = logging.getLogger("hermes.plugins.gemini_stt")

_ENV_KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
_CLEARTEXT_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})
_KEY_PATTERN = re.compile(r"(x-goog-api-key\s*[:=]\s*)\S+", re.IGNORECASE)


def resolve_api_key(settings: GeminiSettings) -> str:
    """Plugin setting > GEMINI_API_KEY > GOOGLE_API_KEY; ``""`` when nothing usable is set."""
    candidates = (settings.api_key, os.getenv("GEMINI_API_KEY"), os.getenv("GOOGLE_API_KEY"))
    for candidate in candidates:
        key = (candidate or "").strip()
        if key and not any(ch in key for ch in "\r\n"):
            return key
    return ""


def resolve_base_url(settings: GeminiSettings) -> str:
    """Normalized ``.../v1beta`` endpoint; rejects cleartext non-local hosts rather than leaking a key."""
    raw = (settings.base_url or os.getenv("GEMINI_BASE_URL") or DEFAULT_BASE_URL).strip()
    if not raw:
        return DEFAULT_BASE_URL
    parts = urlsplit(raw)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        logger.warning("Ignoring invalid Gemini base_url %r; using the public endpoint", raw)
        return DEFAULT_BASE_URL
    if parts.scheme != "https" and (parts.hostname or "") not in _CLEARTEXT_HOSTS:
        logger.warning("Refusing cleartext Gemini base_url %r; using the public endpoint", raw)
        return DEFAULT_BASE_URL
    path = parts.path.rstrip("/")
    if not path.endswith("/v1beta"):
        path = f"{path}/v1beta" if path else "/v1beta"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def api_root(base_url: str) -> str:
    """Strip the API version segment; the Files API mounts at ``<root>/upload/v1beta/files``."""
    trimmed = base_url.rstrip("/")
    return trimmed[: -len("/v1beta")] if trimmed.endswith("/v1beta") else trimmed


def normalise_model(model: str | None, default: str = DEFAULT_MODEL) -> str:
    """Accept ``google/gemini-3.8-flash`` / ``models/gemini-3.8-flash`` spellings, return the wire id."""
    value = (model or "").strip()
    if not value:
        return default
    for prefix in ("google/", "models/"):
        while value.startswith(prefix):
            value = value[len(prefix):]
    return value or default


def redact(message: str, api_key: str) -> str:
    """Remove the key (and any header echo) from text that may reach a log line or envelope."""
    text = str(message or "")
    if api_key and len(api_key) >= 4:
        text = text.replace(api_key, "***")
    return _KEY_PATTERN.sub(r"\1***", text)
