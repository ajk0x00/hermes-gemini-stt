from __future__ import annotations

import base64
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ._load import FakeContext, load_plugin
from ._server import StubServer

plugin = load_plugin()


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.server = StubServer().start()
        self.addCleanup(self.server.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.audio = self.dir / "voice.m4a"
        self.audio.write_bytes(b"fake-audio-bytes")
        self.env = patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)

    def _provider(self, **settings):
        values = {
            "base_url": f"{self.server.base_url}/v1beta",
            "timeout": 20,
            "max_retries": 0,
        }
        values.update(settings)
        return plugin.GeminiTranscriptionProvider(FakeContext(values))

    def _install_interaction(self, *, text="hello world", status="completed", payload=None, status_code=200):
        def handler(record):
            self.last_request = record
            if payload is not None:
                return status_code, {}, payload
            return 200, {}, {
                "id": "int_1",
                "status": status,
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}],
            }

        self.server.routes[("POST", "/v1beta/interactions")] = handler

    def test_inline_success_and_request_shape(self):
        self._install_interaction(text="  hello there  ")
        result = self._provider().transcribe(str(self.audio), model="gemini-3.8-flash", language="en")
        self.assertTrue(result["success"])
        self.assertEqual(result["transcript"], "hello there")
        self.assertEqual(result["provider"], "gemini")

        record = self.last_request
        self.assertEqual(record["headers"]["x-goog-api-key"], "test-key")
        body = self.server.json_body(record)
        self.assertEqual(body["model"], "gemini-3.8-flash")
        self.assertFalse(body["store"])
        self.assertEqual(body["generation_config"]["thinking_level"], "low")
        audio = body["input"][-1]
        self.assertEqual(audio["mime_type"], "audio/m4a")
        self.assertEqual(base64.b64decode(audio["data"]), self.audio.read_bytes())

    def test_transcribe_model_payload(self):
        self._install_interaction(text="asr text")
        result = self._provider(transcription_mode="smart").transcribe(
            str(self.audio), model="gemini-3.5-transcribe", language="es",
        )
        self.assertEqual(result["transcript"], "asr text")
        body = self.server.json_body(self.last_request)
        self.assertNotIn("system_instruction", body)
        self.assertEqual([block["type"] for block in body["input"]], ["audio"])
        config = body["generation_config"]["transcription_config"]
        self.assertEqual(config["language_codes"], ["es"])
        self.assertEqual(config["mode"], "smart")

    def test_missing_key_returns_actionable_error(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=False):
            result = self._provider().transcribe(str(self.audio))
        self.assertFalse(result["success"])
        self.assertIn("GEMINI_API_KEY", result["error"])

    def test_api_error_becomes_envelope(self):
        self._install_interaction(
            payload={"error": {"message": "API key not valid"}}, status_code=401,
        )
        result = self._provider().transcribe(str(self.audio))
        self.assertFalse(result["success"])
        self.assertIn("401", result["error"])
        self.assertIn("API key not valid", result["error"])

    def test_failed_status_is_error(self):
        self._install_interaction(payload={"status": "failed", "error": {"message": "blocked"}})
        result = self._provider().transcribe(str(self.audio))
        self.assertFalse(result["success"])
        self.assertIn("blocked", result["error"])

    def test_empty_transcript_is_success(self):
        self._install_interaction(payload={"status": "completed", "steps": []})
        result = self._provider().transcribe(str(self.audio))
        self.assertTrue(result["success"])
        self.assertEqual(result["transcript"], "")

    def test_unsupported_extension(self):
        weird = self.dir / "voice.silk"
        weird.write_bytes(b"x")
        result = self._provider().transcribe(str(weird))
        self.assertFalse(result["success"])
        self.assertIn("does not support", result["error"])

    def test_video_stream_is_refused(self):
        clip = self.dir / "clip.mp4"
        clip.write_bytes(b"x")
        with patch.object(plugin, "needs_video_probe", return_value=True), \
                patch.object(plugin, "probe_has_video", return_value=True):
            result = self._provider().transcribe(str(clip))
        self.assertFalse(result["success"])
        self.assertIn("video stream", result["error"])

    def test_files_path_uses_uri_and_deletes_upload(self):
        big = self.dir / "long.m4a"
        big.write_bytes(b"a" * (1_200_000))
        uploaded = {}

        def start(record):
            return 200, {"x-goog-upload-url": f"{self.server.base_url}/upload/session/9"}, {}

        def finalize(record):
            uploaded["body"] = record["body"]
            return 200, {}, {
                "file": {
                    "name": "files/big",
                    "uri": "https://generativelanguage.googleapis.com/v1beta/files/big",
                    "state": "ACTIVE",
                }
            }

        self.server.routes[("POST", "/upload/v1beta/files")] = start
        self.server.routes[("POST", "/upload/session/9")] = finalize
        self.server.routes[("GET", "/v1beta/files/big")] = lambda r: (200, {}, {"file": {"state": "ACTIVE"}})
        self.server.routes[("DELETE", "/v1beta/files/big")] = lambda r: (200, {}, {})
        self._install_interaction(text="long transcript")

        result = self._provider(files_api_threshold_mb=1).transcribe(str(big))
        self.assertTrue(result["success"])
        self.assertEqual(uploaded["body"], big.read_bytes())
        body = self.server.json_body(self.last_request)
        self.assertEqual(body["input"][-1]["uri"], "https://generativelanguage.googleapis.com/v1beta/files/big")
        self.assertNotIn("data", body["input"][-1])
        self.assertEqual(len(self.server.calls("DELETE", "/v1beta/files/big")), 1)

    def test_threshold_is_compared_in_raw_bytes(self):
        exactly_threshold = self.dir / "one-meg.m4a"
        exactly_threshold.write_bytes(b"a" * (1024 * 1024))
        self._install_interaction(text="inline transcript")

        result = self._provider(files_api_threshold_mb=1).transcribe(str(exactly_threshold))
        self.assertTrue(result["success"])
        self.assertEqual(self.server.calls("POST", "/upload/v1beta/files"), [])
        self.assertEqual(self.server.json_body(self.last_request)["input"][-1]["mime_type"], "audio/m4a")

    def test_missing_file_uri_is_an_error_but_still_cleans_up(self):
        big = self.dir / "no-uri.m4a"
        big.write_bytes(b"a" * (1_200_000))
        self.server.routes[("POST", "/upload/v1beta/files")] = (
            lambda record: (200, {"x-goog-upload-url": f"{self.server.base_url}/upload/session/10"}, {})
        )
        self.server.routes[("POST", "/upload/session/10")] = (
            lambda record: (200, {}, {"file": {"name": "files/no-uri", "state": "ACTIVE"}})
        )
        self.server.routes[("DELETE", "/v1beta/files/no-uri")] = lambda r: (200, {}, {})

        result = self._provider(files_api_threshold_mb=1).transcribe(str(big))
        self.assertFalse(result["success"])
        self.assertIn("file URI", result["error"])
        self.assertEqual(len(self.server.calls("DELETE", "/v1beta/files/no-uri")), 1)
        self.assertEqual(self.server.calls("POST", "/v1beta/interactions"), [])

    def test_is_available_tracks_the_key(self):
        provider = self._provider()
        self.assertTrue(provider.is_available())
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=False):
            self.assertFalse(provider.is_available())

    def test_catalog_surface(self):
        provider = self._provider()
        self.assertEqual(provider.default_model(), "gemini-3.8-flash")
        ids = {entry["id"] for entry in provider.list_models()}
        self.assertEqual(ids, {"gemini-3.8-flash", "gemini-3.5-transcribe"})
        schema = provider.get_setup_schema()
        self.assertEqual(schema["env_vars"][0]["key"], "GEMINI_API_KEY")


if __name__ == "__main__":
    unittest.main()
