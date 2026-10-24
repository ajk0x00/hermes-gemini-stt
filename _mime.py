"""Map Hermes' accepted audio suffixes onto the Gemini ``AudioContent`` MIME enum.

Every suffix Hermes validates is covered. ``.mp4`` shares the ISO-BMFF container with ``.m4a``
(which Gemini does accept) but is not itself in the enum, so it is sent as ``audio/m4a`` after a
best-effort ffprobe check rejects files that actually carry a video stream.
"""

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger("hermes.plugins.gemini_stt")

MIME_BY_SUFFIX: dict[str, str] = {
    ".mp3": "audio/mp3",
    ".m4a": "audio/m4a",
    ".mp4": "audio/m4a",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/opus",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
}

_VIDEO_PROBE_SUFFIXES = frozenset({".mp4"})


def mime_for(path: str) -> str | None:
    """Gemini MIME type for *path*, or ``None`` when Hermes handed us something unexpected."""
    return MIME_BY_SUFFIX.get(Path(path).suffix.lower())


def needs_video_probe(path: str) -> bool:
    return Path(path).suffix.lower() in _VIDEO_PROBE_SUFFIXES


def _find_ffprobe() -> str | None:
    found = shutil.which("ffprobe")
    if found:
        return found
    hermes_home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    matches = sorted(glob.glob(os.path.join(hermes_home, "tools", "*", "bin", "ffprobe")), reverse=True)
    return matches[0] if matches else None


def probe_has_video(path: str) -> bool | None:
    """True/False from ffprobe, or ``None`` when ffprobe is unavailable or the file is unreadable.

    ``None`` means "unknown": the caller attempts the upload instead of refusing a valid file.
    """
    ffprobe = _find_ffprobe()
    if not ffprobe:
        return None
    try:
        completed = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v", "-show_entries", "stream=index",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("ffprobe failed for %s: %s", path, exc)
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())
