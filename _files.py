"""Gemini Files API: resumable upload, ACTIVE wait, delete.

Used when base64 would push the request past the 20 MB inline cap. Hermes caps its own upload
at 25 MB, so the only files that reach this path are the 12-25 MB tail; anything larger is split
by Hermes before the plugin is called.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from ._parse import decode_json
from ._transport import GeminiAPIError

logger = logging.getLogger("hermes.plugins.gemini_stt")

_ACTIVE_WAIT_SECONDS = 30.0
_POLL_INTERVAL_SECONDS = 1.0


def upload_audio(transport, path: str, mime: str, *, read_timeout: float = 300.0) -> dict[str, Any]:
    """Run the two-step resumable upload and return the file resource."""
    size = os.path.getsize(path)
    start = transport.request(
        "POST",
        f"{transport.api_root}/upload/v1beta/files",
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": mime,
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": Path(path).name}},
        read_timeout=30.0,
    )
    upload_url = start.headers.get("x-goog-upload-url") or start.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        raise GeminiAPIError("Gemini Files API did not return an upload URL.")

    with open(path, "rb") as handle:
        payload = handle.read()
    final = transport.request(
        "POST",
        upload_url,
        headers={
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
            "Content-Type": mime,
        },
        content=payload,
        read_timeout=read_timeout,
        authenticated=False,
    )
    body = decode_json(final)
    file = body.get("file")
    if not isinstance(file, dict) or not file.get("name"):
        raise GeminiAPIError("Gemini Files API returned no file resource.")
    return file


def wait_until_active(transport, file: dict[str, Any], *, timeout: float = _ACTIVE_WAIT_SECONDS) -> dict[str, Any]:
    """Return the file once ``state == ACTIVE``; audio is usually ready immediately."""
    if str(file.get("state") or "").upper() == "ACTIVE":
        return file
    name = str(file["name"]).lstrip("/")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = transport.request("GET", f"{transport.base_url}/{name}", read_timeout=15.0)
        current = decode_json(response).get("file") or {}
        state = str(current.get("state") or "").upper()
        if state == "ACTIVE":
            return current
        if state == "FAILED":
            raise GeminiAPIError(f"Gemini failed to process the uploaded file ({current.get('error') or 'no detail'}).")
        time.sleep(_POLL_INTERVAL_SECONDS)
    raise GeminiAPIError("Gemini did not finish processing the uploaded file within 30s.")


def delete_file(transport, name: str | None) -> None:
    """Best-effort cleanup: uploads persist for 48h otherwise, which is a privacy footgun."""
    if not name:
        return
    try:
        transport.request(
            "DELETE", f"{transport.base_url}/{str(name).lstrip('/')}",
            read_timeout=10.0, retry=False,
        )
    except Exception as exc:  # cleanup must never mask the real result
        logger.debug("Gemini file cleanup failed for %s: %s", name, exc)
