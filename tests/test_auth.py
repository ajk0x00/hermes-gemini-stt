from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from ._load import load_plugin

load_plugin()
from gemini_stt_under_test import _auth  # noqa: E402
from gemini_stt_under_test._settings import GeminiSettings  # noqa: E402

_NO_KEYS = {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "GEMINI_BASE_URL": ""}


class ApiKeyTests(unittest.TestCase):
    def test_plugin_setting_wins_over_environment(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key", "GOOGLE_API_KEY": "older"}, clear=False):
            key = _auth.resolve_api_key(GeminiSettings(api_key="setting-key"))
        self.assertEqual(key, "setting-key")

    def test_gemini_env_wins_over_google_env(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "first", "GOOGLE_API_KEY": "second"}, clear=False):
            self.assertEqual(_auth.resolve_api_key(GeminiSettings()), "first")

    def test_google_env_is_the_fallback(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "second"}, clear=False):
            self.assertEqual(_auth.resolve_api_key(GeminiSettings()), "second")

    def test_nothing_configured(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            self.assertEqual(_auth.resolve_api_key(GeminiSettings()), "")

    def test_key_with_newline_is_refused(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "bad\nkey", "GOOGLE_API_KEY": "good"}, clear=False):
            self.assertEqual(_auth.resolve_api_key(GeminiSettings()), "good")


class BaseUrlTests(unittest.TestCase):
    def test_default_is_the_public_endpoint(self):
        with patch.dict(os.environ, _NO_KEYS, clear=False):
            self.assertEqual(
                _auth.resolve_base_url(GeminiSettings()),
                "https://generativelanguage.googleapis.com/v1beta",
            )

    def test_version_segment_is_added_once(self):
        self.assertEqual(
            _auth.resolve_base_url(GeminiSettings(base_url="https://proxy.example.com")),
            "https://proxy.example.com/v1beta",
        )
        self.assertEqual(
            _auth.resolve_base_url(GeminiSettings(base_url="https://proxy.example.com/gemini")),
            "https://proxy.example.com/gemini/v1beta",
        )
        self.assertEqual(
            _auth.resolve_base_url(GeminiSettings(base_url="https://proxy.example.com/v1beta/")),
            "https://proxy.example.com/v1beta",
        )

    def test_cleartext_is_only_allowed_for_loopback(self):
        self.assertEqual(
            _auth.resolve_base_url(GeminiSettings(base_url="http://127.0.0.1:8080")),
            "http://127.0.0.1:8080/v1beta",
        )
        self.assertEqual(
            _auth.resolve_base_url(GeminiSettings(base_url="http://example.com")),
            "https://generativelanguage.googleapis.com/v1beta",
        )

    def test_invalid_scheme_falls_back(self):
        self.assertEqual(
            _auth.resolve_base_url(GeminiSettings(base_url="ftp://example.com")),
            "https://generativelanguage.googleapis.com/v1beta",
        )

    def test_api_root_strips_the_version(self):
        self.assertEqual(
            _auth.api_root("https://proxy.example.com/gemini/v1beta"),
            "https://proxy.example.com/gemini",
        )
        self.assertEqual(
            _auth.api_root("https://generativelanguage.googleapis.com/v1beta/"),
            "https://generativelanguage.googleapis.com",
        )


class ModelTests(unittest.TestCase):
    def test_empty_uses_the_default(self):
        self.assertEqual(_auth.normalise_model(None), "gemini-3.8-flash")
        self.assertEqual(_auth.normalise_model("   "), "gemini-3.8-flash")

    def test_provider_and_resource_prefixes_are_stripped(self):
        self.assertEqual(_auth.normalise_model("google/gemini-3.8-flash"), "gemini-3.8-flash")
        self.assertEqual(_auth.normalise_model("models/gemini-3.5-transcribe"), "gemini-3.5-transcribe")
        self.assertEqual(_auth.normalise_model("google/models/gemini-3.8-flash"), "gemini-3.8-flash")

    def test_unknown_ids_pass_through(self):
        self.assertEqual(_auth.normalise_model("my-tuned-model"), "my-tuned-model")


class RedactionTests(unittest.TestCase):
    def test_key_is_removed_from_text(self):
        self.assertNotIn("secret-key", _auth.redact("failed with secret-key", "secret-key"))

    def test_header_echo_is_masked(self):
        text = _auth.redact("x-goog-api-key: abc123 sent", "unrelated")
        self.assertNotIn("abc123", text)
        self.assertIn("x-goog-api-key: ***", text)

    def test_short_keys_and_empty_input_are_safe(self):
        self.assertEqual(_auth.redact("", "abc"), "")
        self.assertIsInstance(_auth.redact(None, ""), str)


if __name__ == "__main__":
    unittest.main()
