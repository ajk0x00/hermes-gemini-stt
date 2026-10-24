from __future__ import annotations

import unittest

from ._load import load_plugin

load_plugin()
from gemini_stt_under_test import _payload  # noqa: E402
from gemini_stt_under_test._settings import GeminiSettings  # noqa: E402

_MB = 1024 * 1024


class GeneralPayloadTests(unittest.TestCase):
    def test_shape_and_privacy_default(self):
        payload = _payload.build_payload(
            model="gemini-3.8-flash", mime="audio/m4a", settings=GeminiSettings(),
            prompt="ACME, Kubernetes", language="es", data="QUJD",
        )
        self.assertFalse(payload["store"])
        self.assertEqual(payload["model"], "gemini-3.8-flash")
        self.assertEqual(payload["generation_config"], {"thinking_level": "low"})
        self.assertIn("speech-to-text", payload["system_instruction"])
        self.assertEqual(payload["input"][0]["type"], "text")
        self.assertIn("ACME", payload["input"][0]["text"])
        self.assertIn("expected to be in es", payload["input"][0]["text"])
        self.assertEqual(
            payload["input"][1],
            {"type": "audio", "mime_type": "audio/m4a", "data": "QUJD"},
        )

    def test_auto_language_drops_the_hint(self):
        payload = _payload.build_payload(
            model="gemini-3.8-flash", mime="audio/mp3",
            settings=GeminiSettings(language_mode="auto"),
            prompt=None, language="en", data="QUJD",
        )
        self.assertEqual([block["type"] for block in payload["input"]], ["audio"])

    def test_no_hints_no_text_block(self):
        payload = _payload.build_payload(
            model="gemini-3.8-flash", mime="audio/mp3", settings=GeminiSettings(language_mode="auto"),
            data="QUJD",
        )
        self.assertEqual(payload["input"], [{"type": "audio", "mime_type": "audio/mp3", "data": "QUJD"}])

    def test_file_uri_block(self):
        payload = _payload.build_payload(
            model="gemini-3.8-flash", mime="audio/wav",
            settings=GeminiSettings(language_mode="auto"), uri="https://files/abc",
        )
        self.assertEqual(payload["input"][-1]["uri"], "https://files/abc")
        self.assertNotIn("data", payload["input"][-1])


class TranscribePayloadTests(unittest.TestCase):
    def test_audio_only_with_language_and_vocabulary(self):
        payload = _payload.build_payload(
            model="gemini-3.5-transcribe", mime="audio/m4a", settings=GeminiSettings(),
            prompt="Kubernetes, Gemini", language="en", data="QUJD",
        )
        self.assertFalse(payload["store"])
        self.assertNotIn("system_instruction", payload)
        self.assertEqual(payload["input"], [{"type": "audio", "mime_type": "audio/m4a", "data": "QUJD"}])
        config = payload["generation_config"]["transcription_config"]
        self.assertEqual(config["language_codes"], ["en"])
        self.assertEqual(config["mode"], "verbatim")
        self.assertEqual(config["custom_vocabulary"], ["Kubernetes", "Gemini"])

    def test_auto_language_sends_empty_codes(self):
        payload = _payload.build_payload(
            model="gemini-3.5-transcribe", mime="audio/m4a",
            settings=GeminiSettings(language_mode="auto"), language="en", data="x",
        )
        self.assertEqual(payload["generation_config"]["transcription_config"]["language_codes"], [])

    def test_smart_mode(self):
        payload = _payload.build_payload(
            model="gemini-3.5-transcribe", mime="audio/m4a",
            settings=GeminiSettings(transcription_mode="smart"), data="x",
        )
        self.assertEqual(payload["generation_config"]["transcription_config"]["mode"], "smart")

    def test_diarization_uses_object_mode_and_drops_vocabulary(self):
        payload = _payload.build_payload(
            model="gemini-3.5-transcribe", mime="audio/m4a",
            settings=GeminiSettings(diarize=True), prompt="Kubernetes", data="x",
        )
        config = payload["generation_config"]["transcription_config"]
        self.assertEqual(config["mode"], {"type": "verbatim", "diarization_mode": "speaker"})
        self.assertNotIn("custom_vocabulary", config)

    def test_diarization_forces_verbatim_over_smart(self):
        payload = _payload.build_payload(
            model="gemini-3.5-transcribe", mime="audio/m4a",
            settings=GeminiSettings(diarize=True, transcription_mode="smart"), data="x",
        )
        config = payload["generation_config"]["transcription_config"]
        self.assertEqual(config["mode"], {"type": "verbatim", "diarization_mode": "speaker"})


class InlineBudgetTests(unittest.TestCase):
    def test_default_threshold_wins(self):
        self.assertEqual(
            _payload.inline_budget_bytes(GeminiSettings(), prompt=None, language=None), 12 * _MB,
        )

    def test_request_ceiling_wins_for_large_thresholds(self):
        budget = _payload.inline_budget_bytes(
            GeminiSettings(files_api_threshold_mb=19.0), prompt="x" * 4000, language="en",
        )
        self.assertLess(budget, 19 * _MB)
        self.assertGreater(budget, 14 * _MB)
        # A full-budget file must still fit the 20 MB request after base64 inflation.
        self.assertLessEqual(4 * ((budget + 2) // 3), _payload.INLINE_REQUEST_LIMIT)

    def test_budget_never_zero(self):
        self.assertGreater(_payload.inline_budget_bytes(GeminiSettings(), prompt="x" * 100, language="en"), 0)


if __name__ == "__main__":
    unittest.main()
