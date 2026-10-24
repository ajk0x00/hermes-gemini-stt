"""One real Hermes transcription dispatch against a local Gemini stub.

Run in a subprocess so ``HERMES_HOME`` is resolved cleanly. Prints one JSON line on stdout; the
unittest wrapper asserts on it. No network and no API key are required.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RECORDS: list[dict] = []


def build_wav(path: Path, seconds: float = 0.5, payload_size: int | None = None) -> None:
    frames = b"\x00\x00" * int(16_000 * seconds)
    if payload_size:
        frames = b"\x00\x00" * payload_size
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(frames)


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            return

        def do_GET(self):  # noqa: N802
            self._dispatch("GET")

        def do_POST(self):  # noqa: N802
            self._dispatch("POST")

        def do_DELETE(self):  # noqa: N802
            self._dispatch("DELETE")

        def _dispatch(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            path = self.path.split("?", 1)[0]
            RECORDS.append({"method": method, "path": path, "body_len": len(body)})

            status, headers, payload = 200, {}, {}
            if path == "/v1beta/interactions":
                request = json.loads(body or b"{}")
                text = "transcribe model output" if "-transcribe" in request.get("model", "") else "hello from gemini"
                payload = {
                    "id": "int_e2e",
                    "status": "completed",
                    "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}],
                }
            elif path == "/upload/v1beta/files":
                headers["x-goog-upload-url"] = f"{self.server.base_url}/upload/session/e2e"
                payload = {}
            elif path == "/upload/session/e2e":
                payload = {
                    "file": {
                        "name": "files/e2e",
                        "uri": "https://generativelanguage.googleapis.com/v1beta/files/e2e",
                        "state": "ACTIVE",
                    }
                }
                headers["Content-Type"] = "application/json"
            elif path == "/v1beta/files/e2e":
                if method == "DELETE":
                    payload = {}
                else:
                    payload = {"file": {"name": "files/e2e", "state": "ACTIVE"}}
            else:
                status, payload = 404, {"error": {"message": f"no route {method} {path}"}}

            data = json.dumps(payload).encode()
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-dir", required=True)
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--isolation", default="in_process", choices=["in_process", "host"])
    parser.add_argument("--files", action="store_true")
    args = parser.parse_args()

    repo = Path(os.environ.get("HERMES_REPO") or (Path.home() / ".hermes" / "hermes-agent")).resolve()
    sys.path.insert(0, str(repo))
    import logging
    logging.basicConfig(level=logging.DEBUG if os.environ.get("E2E_DEBUG") else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    home = Path(tempfile.mkdtemp(prefix="hermes-gemini-e2e-"))
    work = Path(tempfile.mkdtemp(prefix="hermes-gemini-audio-"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler())
    server.base_url = f"http://127.0.0.1:{server.server_address[1]}"
    server.daemon_threads = True
    import threading

    threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        plugin_target = home / "plugins" / "gemini-stt"
        shutil.copytree(
            args.plugin_dir, plugin_target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"),
        )
        (home / "plugins").mkdir(parents=True, exist_ok=True)

        wav = work / "voice.wav"
        build_wav(wav, payload_size=(1_050_000 if args.files else None))

        (home / "config.yaml").write_text(
            "stt:\n"
            "  enabled: true\n"
            "  provider: gemini\n"
            "  language: en\n"
            "  gemini:\n"
            f"    model: {args.model}\n"
            "    language: en\n"
            "plugins:\n"
            f"  isolation: {args.isolation}\n"
            "  enabled:\n"
            "    - gemini-stt\n"
            "  entries:\n"
            "    gemini-stt:\n"
            "      settings:\n"
            f"        base_url: {server.base_url}/v1beta\n"
            "        timeout: 30\n"
            "        max_retries: 0\n"
            + ("        files_api_threshold_mb: 1\n" if args.files else ""),
            encoding="utf-8",
        )

        os.environ["HERMES_HOME"] = str(home)
        os.environ["GEMINI_API_KEY"] = "e2e-test-key"
        # A brand-new HERMES_HOME would otherwise make hermes_bootstrap re-sync the checkout's
        # dependencies before the plugin host child can serve its first request (a test-only cost).
        os.environ["HERMES_DISABLE_LAZY_INSTALLS"] = "1"

        from tools.transcription_tools import transcribe_audio

        result = transcribe_audio(str(wav))

        requests = [r for r in RECORDS if r["path"] == "/v1beta/interactions"]
        print(json.dumps({
            "result": result,
            "requests": RECORDS,
            "interaction_calls": len(requests),
            "files_used": any(r["path"] == "/upload/v1beta/files" for r in RECORDS),
            "deleted": any(r["method"] == "DELETE" for r in RECORDS),
        }))
        return 0
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(home, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
