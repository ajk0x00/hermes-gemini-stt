"""Opt-in live checks against the real Gemini API.

STT is skipped unless ``HERMES_TEST_LIVE_GEMINI=1``; the slower general-model check additionally
needs ``HERMES_TEST_LIVE_GEMINI_SLOW=1``. TTS is skipped unless
``HERMES_TEST_LIVE_GEMINI_TTS=1``. These tests cost a few cents at most and never print the key.

    set -a; source ~/.hermes/.env; set +a
    HERMES_TEST_LIVE_GEMINI=1 python -m unittest tests.test_live -v
    HERMES_TEST_LIVE_GEMINI_TTS=1 python -m unittest tests.test_live.LiveTTSTests -v

Set ``HERMES_NEUTTS_SAMPLE`` to a speech WAV if the bundled Hermes sample is unavailable.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from ._load import FakeContext, load_plugin

plugin = load_plugin()

_LIVE = os.environ.get("HERMES_TEST_LIVE_GEMINI") == "1"
_SLOW = os.environ.get("HERMES_TEST_LIVE_GEMINI_SLOW") == "1"
_LIVE_TTS = os.environ.get("HERMES_TEST_LIVE_GEMINI_TTS") == "1"


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


@unittest.skipUnless(_LIVE_TTS, "set HERMES_TEST_LIVE_GEMINI_TTS=1 to call the real TTS API")
class LiveTTSTests(unittest.TestCase):
    def setUp(self):
        key = _env_key()
        if not key:
            self.skipTest("no GEMINI_API_KEY / GOOGLE_API_KEY configured")
        self.env = patch.dict(os.environ, {"GEMINI_API_KEY": key}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.api_key = key

    def _provider(self, **settings):
        values = {
            "timeout": 300,
            "tts_timeout": 300,
            "tts_max_retries": 1,
            "tts_audio_tags": False,
            "tts_tag_timeout": 15,
        }
        values.update(settings)
        return plugin.GeminiTTSProvider(FakeContext(values))

    def test_flash_tts_returns_playable_wav(self):
        provider = self._provider()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "gemini.wav"
            result = provider.synthesize(
                "Hello from Gemini text to speech.", str(out),
                model="gemini-3.8-flash-tts", voice="Kore", format="wav",
            )
            self.assertTrue(Path(result).is_file())
            self.assertEqual(Path(result).read_bytes()[:4], b"RIFF")
            with wave.open(result, "rb") as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getframerate(), 24000)
                self.assertGreater(handle.getnframes(), 2400)

    def test_flash_tts_streams_pcm(self):
        provider = self._provider()
        chunks = list(provider.stream(
            "Hello from Gemini streaming text to speech.",
            model="gemini-3.8-flash-tts", voice="Kore", format="pcm",
        ))
        self.assertGreater(sum(len(chunk) for chunk in chunks), 0)
        self.assertEqual(len(chunks[0]) % 2, 0)

    def test_tagger_produces_angle_tags(self):
        provider = self._provider(tts_audio_tags=True)
        text = (
            "Honestly I was not expecting that. The build passed on the first try and every "
            "test is green. Let's ship it."
        )
        base_url = "https://generativelanguage.googleapis.com/v1beta"
        tagged = provider._run_tagger(
            text, style="angle", api_key=self.api_key, base_url=base_url,
        )
        if tagged is None:
            self.skipTest("tagger was unavailable within the live timeout")
        self.assertIn("<", tagged)
        self.assertNotIn("[", tagged)
        import re
        tagged_words = re.findall(r"\w+", re.sub(r"<[^>]+>|\[[^\]]+\]", " ", tagged).lower())
        source_words = re.findall(r"\w+", text.lower())
        self.assertEqual(tagged_words, source_words)


if __name__ == "__main__":
    unittest.main()
