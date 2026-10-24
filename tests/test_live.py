"""Opt-in live checks against the real Gemini API.

Skipped unless ``HERMES_TEST_LIVE_GEMINI=1``. The slower general-model check additionally needs
``HERMES_TEST_LIVE_GEMINI_SLOW=1``. These tests cost a few cents at most and never print the key.

    set -a; source ~/.hermes/.env; set +a
    HERMES_TEST_LIVE_GEMINI=1 python -m unittest tests.test_live -v

Set ``HERMES_NEUTTS_SAMPLE`` to a speech WAV if the bundled Hermes sample is unavailable.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from ._load import FakeContext, load_plugin

plugin = load_plugin()

_LIVE = os.environ.get("HERMES_TEST_LIVE_GEMINI") == "1"
_SLOW = os.environ.get("HERMES_TEST_LIVE_GEMINI_SLOW") == "1"


def _env_key() -> str:
    key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if key:
        return key
    env_file = Path.home() / ".hermes" / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _sample() -> Path | None:
    configured = os.environ.get("HERMES_NEUTTS_SAMPLE")
    candidate = Path(configured) if configured else (
        Path(os.environ.get("HERMES_REPO") or (Path.home() / ".hermes" / "hermes-agent"))
        / "tools" / "neutts_samples" / "jo.wav"
    )
    return candidate if candidate.is_file() else None


@unittest.skipUnless(_LIVE, "set HERMES_TEST_LIVE_GEMINI=1 to call the real API")
class LiveTests(unittest.TestCase):
    def setUp(self):
        key = _env_key()
        if not key:
            self.skipTest("no GEMINI_API_KEY / GOOGLE_API_KEY configured")
        sample = _sample()
        if sample is None:
            self.skipTest("no speech sample; set HERMES_NEUTTS_SAMPLE")
        self.audio = sample
        self.env = patch.dict(os.environ, {"GEMINI_API_KEY": key}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)

    def _provider(self, **settings):
        values = {"timeout": 300, "max_retries": 1}
        values.update(settings)
        return plugin.GeminiTranscriptionProvider(FakeContext(values))

    def test_transcribe_model_returns_speech(self):
        result = self._provider().transcribe(
            str(self.audio), model="gemini-3.5-transcribe", language="en", prompt="Neuphonic",
        )
        self.assertTrue(result["success"], result.get("error"))
        self.assertGreater(len(result["transcript"]), 20)
        self.assertIn("impressed", result["transcript"].lower())

    @unittest.skipUnless(_SLOW, "set HERMES_TEST_LIVE_GEMINI_SLOW=1 for the slow general model")
    def test_general_model_returns_speech(self):
        result = self._provider().transcribe(
            str(self.audio), model="gemini-3.8-flash", language="en",
        )
        self.assertTrue(result["success"], result.get("error"))
        self.assertGreater(len(result["transcript"]), 20)


if __name__ == "__main__":
    unittest.main()
