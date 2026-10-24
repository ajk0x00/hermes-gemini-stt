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


class HermesEndToEndTests(unittest.TestCase):
    """Loads the plugin through Hermes' real discovery + dispatch path in a temp HERMES_HOME."""

    maxDiff = None

    def _run(self, *args: str) -> dict:
        env = dict(os.environ)
        env["HERMES_REPO"] = str(HERMES_REPO)
        env.pop("HERMES_HOME", None)
        completed = subprocess.run(
            [sys.executable, str(HERE / "e2e_hermes.py"), "--plugin-dir", str(PLUGIN_DIR), *args],
            capture_output=True, text=True, timeout=240, env=env,
        )
        if completed.returncode != 0:
            self.fail(f"e2e subprocess failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-4000:]}")
        return json.loads(completed.stdout.strip().splitlines()[-1])

    def test_inline_dispatch(self):
        payload = self._run("--model", "gemini-3.8-flash")
        result = payload["result"]
        self.assertTrue(result["success"], result)
        self.assertEqual(result["transcript"], "hello from gemini")
        self.assertEqual(result["provider"], "gemini")
        self.assertEqual(payload["interaction_calls"], 1)
        self.assertFalse(payload["files_used"])

    def test_transcribe_model_dispatch(self):
        payload = self._run("--model", "gemini-3.5-transcribe")
        result = payload["result"]
        self.assertTrue(result["success"], result)
        self.assertEqual(result["transcript"], "transcribe model output")

    def test_files_api_dispatch(self):
        payload = self._run("--model", "gemini-3.8-flash", "--files")
        result = payload["result"]
        self.assertTrue(result["success"], result)
        self.assertTrue(payload["files_used"], payload)
        self.assertTrue(payload["deleted"], payload)

    def test_host_isolation_dispatch(self):
        payload = self._run("--model", "gemini-3.8-flash", "--isolation", "host")
        result = payload["result"]
        self.assertTrue(result["success"], result)
        self.assertEqual(result["transcript"], "hello from gemini")


if __name__ == "__main__":
    unittest.main()
