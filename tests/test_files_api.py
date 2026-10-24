from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ._load import load_plugin
from ._server import StubServer

load_plugin()
from gemini_stt_under_test import _files  # noqa: E402
from gemini_stt_under_test._transport import GeminiAPIError, GeminiTransport  # noqa: E402


class FilesAPITests(unittest.TestCase):
    def setUp(self):
        self.server = StubServer().start()
        self.addCleanup(self.server.stop)
        self.transport = GeminiTransport(
            api_key="secret-key", base_url=f"{self.server.base_url}/v1beta",
            api_root=self.server.base_url, timeout=10, max_retries=0, user_agent="test",
        )
        self.addCleanup(self.transport.close)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "voice.m4a"
        self.path.write_bytes(b"audio-bytes" * 100)
        self.poll_count = 0

    def _install_routes(self, *, state="ACTIVE"):
        self.server.routes[("POST", "/upload/v1beta/files")] = (
            lambda record: (200, {"x-goog-upload-url": f"{self.server.base_url}/upload/session/1"}, {})
        )

        def finalize(record):
            self.uploaded = record["body"]
            return 200, {}, {
                "file": {
                    "name": "files/abc",
                    "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc",
                    "state": "PROCESSING" if state != "ACTIVE" else "ACTIVE",
                }
            }

        self.server.routes[("POST", "/upload/session/1")] = finalize

        def poll(record):
            self.poll_count += 1
            return 200, {}, {"file": {"name": "files/abc", "state": "ACTIVE"}}

        self.server.routes[("GET", "/v1beta/files/abc")] = poll
        self.server.routes[("DELETE", "/v1beta/files/abc")] = lambda record: (200, {}, {})

    def test_resumable_upload_protocol(self):
        self._install_routes()
        file = _files.upload_audio(self.transport, str(self.path), "audio/m4a")

        start = self.server.calls("POST", "/upload/v1beta/files")[0]
        self.assertEqual(start["headers"]["x-goog-upload-protocol"], "resumable")
        self.assertEqual(start["headers"]["x-goog-upload-command"], "start")
        self.assertEqual(start["headers"]["x-goog-upload-header-content-type"], "audio/m4a")
        self.assertEqual(start["headers"]["x-goog-upload-header-content-length"], str(self.path.stat().st_size))

        finalize = self.server.calls("POST", "/upload/session/1")[0]
        self.assertEqual(finalize["body"], self.path.read_bytes())
        self.assertNotIn("x-goog-api-key", finalize["headers"])

        self.assertEqual(file["name"], "files/abc")
        self.assertEqual(file["state"], "ACTIVE")

    def test_waits_for_processing_file(self):
        self._install_routes(state="PROCESSING")
        file = _files.upload_audio(self.transport, str(self.path), "audio/m4a")
        resolved = _files.wait_until_active(self.transport, file)
        self.assertEqual(resolved["state"], "ACTIVE")
        self.assertEqual(self.poll_count, 1)

    def test_delete_is_sent(self):
        self._install_routes()
        _files.delete_file(self.transport, "files/abc")
        self.assertEqual(len(self.server.calls("DELETE", "/v1beta/files/abc")), 1)

    def test_delete_swallows_errors(self):
        _files.delete_file(self.transport, "files/missing")  # 404 route
        self.assertEqual(len(self.server.calls("DELETE", "/v1beta/files/missing")), 1)

    def test_missing_upload_url_is_an_error(self):
        self.server.routes[("POST", "/upload/v1beta/files")] = lambda record: (200, {}, {})
        with self.assertRaises(GeminiAPIError) as caught:
            _files.upload_audio(self.transport, str(self.path), "audio/m4a")
        self.assertIn("upload URL", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
