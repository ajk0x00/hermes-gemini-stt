from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from ._load import PLUGIN_DIR

HERE = Path(__file__).resolve().parent
HERMES_REPO = Path(os.environ.get("HERMES_REPO") or (Path.home() / ".hermes" / "hermes-agent"))


class TTSHermesEndToEndTests(unittest.TestCase):
    """Loads the plugin through Hermes' real discovery + TTS dispatch path."""

    maxDiff = None

    def _run(self, *args: str) -> dict:
        env = dict(os.environ)
        env["HERMES_REPO"] = str(HERMES_REPO)
        env.pop("HERMES_HOME", None)
        completed = subprocess.run(
            [sys.executable, str(HERE / "e2e_tts_hermes.py"), "--plugin-dir", str(PLUGIN_DIR), *args],
            capture_output=True, text=True, timeout=240, env=env,
        )
        if completed.returncode != 0:
            self.fail(
                f"tts e2e subprocess failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-4000:]}"
            )
        return json.loads(completed.stdout.strip().splitlines()[-1])

    def _assert_payload(self, payload: dict) -> None:
        result = payload["result"]
        self.assertTrue(result["success"], result)
        self.assertEqual(result["provider"], "gemini-tts")
        self.assertTrue(payload["exists"], payload)
        self.assertEqual(payload["prefix"], "RIFF", payload)
        self.assertEqual(payload["tag_requests"], 1, payload)
        self.assertEqual(payload["tts_requests"], 1, payload)
        self.assertIn("<laugh>", payload["tts_text"])

    def test_tts_dispatch(self):
        self._assert_payload(self._run())

    def test_tts_host_isolation_dispatch(self):
        self._assert_payload(self._run("--isolation", "host"))


if __name__ == "__main__":
    unittest.main()
