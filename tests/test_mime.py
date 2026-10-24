from __future__ import annotations

import unittest
from unittest.mock import patch

from ._load import load_plugin

load_plugin()
from gemini_stt_under_test import _mime  # noqa: E402


class MimeTests(unittest.TestCase):
    def test_every_hermes_suffix_is_mapped(self):
        expected = {
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
        for suffix, mime in expected.items():
            with self.subTest(suffix=suffix):
                self.assertEqual(_mime.mime_for(f"voice{suffix}"), mime)

    def test_case_insensitive(self):
        self.assertEqual(_mime.mime_for("VOICE.MP3"), "audio/mp3")

    def test_unknown_suffix_is_none(self):
        self.assertIsNone(_mime.mime_for("voice.silk"))
        self.assertIsNone(_mime.mime_for("voice"))

    def test_only_mp4_needs_a_video_probe(self):
        self.assertTrue(_mime.needs_video_probe("clip.MP4"))
        self.assertFalse(_mime.needs_video_probe("clip.m4a"))

    def test_probe_returns_none_without_ffprobe(self):
        with patch.object(_mime, "_find_ffprobe", return_value=None):
            self.assertIsNone(_mime.probe_has_video("clip.mp4"))


if __name__ == "__main__":
    unittest.main()
