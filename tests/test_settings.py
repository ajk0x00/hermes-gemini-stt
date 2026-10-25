from __future__ import annotations

import unittest

from ._load import FakeContext, load_plugin

load_plugin()
from gemini_stt_under_test import _settings  # noqa: E402


class SettingsTests(unittest.TestCase):
    def test_defaults_without_a_context(self):
        values = _settings.load_settings(None)
        self.assertEqual(values.timeout, 180.0)
        self.assertEqual(values.max_retries, 2)
        self.assertEqual(values.thinking_level, "low")
        self.assertEqual(values.transcription_mode, "verbatim")
        self.assertFalse(values.diarize)
        self.assertFalse(values.keep_uploaded_files)
        self.assertEqual(values.language_mode, "hint")

    def test_values_are_coerced(self):
        ctx = FakeContext({
            "timeout": "45",
            "max_retries": 1,
            "diarize": "yes",
            "keep_uploaded_files": "off",
            "api_key": "  key-123  ",
        })
        values = _settings.load_settings(ctx)
        self.assertEqual(values.timeout, 45.0)
        self.assertEqual(values.max_retries, 1)
        self.assertTrue(values.diarize)
        self.assertFalse(values.keep_uploaded_files)
        self.assertEqual(values.api_key, "key-123")

    def test_clamping(self):
        values = _settings.load_settings(FakeContext({
            "timeout": 5, "max_retries": 99, "files_api_threshold_mb": -2,
        }))
        self.assertEqual(values.timeout, 10.0)
        self.assertEqual(values.max_retries, 5)
        self.assertEqual(values.files_api_threshold_mb, 1.0)

    def test_invalid_choices_fall_back(self):
        values = _settings.load_settings(FakeContext({
            "thinking_level": "maximum", "transcription_mode": "poetic", "language_mode": "maybe",
        }))
        self.assertEqual(values.thinking_level, "low")
        self.assertEqual(values.transcription_mode, "verbatim")
        self.assertEqual(values.language_mode, "hint")

    def test_minimal_thinking_is_rejected(self):
        # Gemini 3.8 Flash returns HTTP 400 for THINKING_LEVEL_MINIMAL.
        values = _settings.load_settings(FakeContext({"thinking_level": "minimal"}))
        self.assertEqual(values.thinking_level, "low")

    def test_broken_context_degrades_to_defaults(self):
        class Broken:
            def get_config(self, key, default=None):
                raise RuntimeError("no config here")

        values = _settings.load_settings(Broken())
        self.assertEqual(values.timeout, 180.0)

    def test_tts_defaults_are_tagging_first(self):
        values = _settings.load_tts_settings(None)
        self.assertEqual(values.model, "gemini-3.8-flash-tts")
        self.assertEqual(values.voice, "Kore")
        self.assertTrue(values.audio_tags)
        self.assertTrue(values.audio_tags_streaming)
        self.assertEqual(values.tag_model, "gemini-3.1-flash-lite")
        self.assertEqual(values.tag_timeout, 5.0)

    def test_tts_values_are_coerced_and_clamped(self):
        values = _settings.load_tts_settings(FakeContext({
            "tts_audio_tags": "off",
            "tts_audio_tags_streaming": "no",
            "tts_tag_timeout": 99,
            "tts_timeout": 1,
            "tts_max_retries": 99,
            "tts_style": "  whisper  ",
        }))
        self.assertFalse(values.audio_tags)
        self.assertFalse(values.audio_tags_streaming)
        self.assertEqual(values.tag_timeout, 30.0)
        self.assertEqual(values.timeout, 10.0)
        self.assertEqual(values.max_retries, 5)
        self.assertEqual(values.style, "whisper")


if __name__ == "__main__":
    unittest.main()
