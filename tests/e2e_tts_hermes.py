"""Real-Hermes TTS dispatch against a local Gemini stub.

Runs in a subprocess so HERMES_HOME and PATH are clean. Prints one JSON line; the unittest
wrapper asserts on it. No network, no API key, and no ffmpeg are required.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RECORDS: list[dict] = []


def wav_b64(frames: int = 2400) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setframerate(24000)
        handle.setsampwidth(2)
        handle.writeframes(b"\x00\x00" * frames)
    return base64.b64encode(buf.getvalue()).decode()


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            return

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            path = self.path.split("?", 1)[0]
            request = json.loads(body or b"{}")
            RECORDS.append({"path": path, "body": request})

            if path == "/v1beta/models/gemini-3.1-flash-lite:generateContent":
                payload = {
                    "candidates": [{
                        "content": {"parts": [{
                            "text": "Honestly <laugh> I was not expecting that. The build passed "
                                    "on the first try and every test is green."
                        }]},
                        "finishReason": "STOP",
                    }]
                }
            elif path == "/v1beta/models/gemini-3.8-flash-tts:generateContent":
                payload = {
                    "candidates": [{
                        "content": {"parts": [{
                            "inlineData": {"mimeType": "audio/wav", "data": wav_b64()}
                        }]},
                        "finishReason": "STOP",
                    }]
                }
            else:
                self.send_response(404)
                data = json.dumps({"error": {"message": f"no route {path}"}}).encode()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-dir", required=True)
    parser.add_argument("--isolation", default="in_process", choices=["in_process", "host"])
    args = parser.parse_args()

    repo = Path(os.environ.get("HERMES_REPO") or (Path.home() / ".hermes" / "hermes-agent")).resolve()
    sys.path.insert(0, str(repo))
    import logging
    logging.basicConfig(level=logging.WARNING)

    home = Path(tempfile.mkdtemp(prefix="hermes-gemini-tts-e2e-"))
    work = Path(tempfile.mkdtemp(prefix="hermes-gemini-tts-audio-"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler())
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()

    original_path = os.environ.get("PATH", "")
    try:
        plugin_target = home / "plugins" / "gemini-stt"
        shutil.copytree(
            args.plugin_dir, plugin_target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests"),
        )
        out = work / "reply.wav"
        (home / "config.yaml").write_text(
            "tts:\n"
            "  provider: gemini-tts\n"
            "  model: gemini-3.8-flash-tts\n"
            "  voice: Kore\n"
            "  output_format: wav\n"
            "plugins:\n"
            f"  isolation: {args.isolation}\n"
            "  enabled:\n"
            "    - gemini-stt\n"
            "  entries:\n"
            "    gemini-stt:\n"
            "      settings:\n"
            f"        base_url: http://127.0.0.1:{server.server_address[1]}/v1beta\n"
            "        tts_timeout: 30\n"
            "        tts_max_retries: 0\n"
            "        tts_tag_timeout: 5\n"
            "        tts_audio_tags: true\n",
            encoding="utf-8",
        )

        os.environ["HERMES_HOME"] = str(home)
        os.environ["GEMINI_API_KEY"] = "e2e-test-key"
        os.environ["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
        # Keep the test deterministic: no bundled/system ffmpeg, so the provider returns WAV.
        os.environ["PATH"] = "/usr/bin:/bin"

        from tools.tts_tool import text_to_speech_tool

        raw = text_to_speech_tool(
            "Honestly I was not expecting that. The build passed on the first try and every "
            "test is green.",
            output_path=str(out),
        )
        result = json.loads(raw)
        path = Path(result.get("file_path") or out)
        prefix = path.read_bytes()[:4] if path.is_file() else b""
        print(json.dumps({
            "result": result,
            "path": str(path),
            "exists": path.is_file(),
            "prefix": prefix.decode("latin-1"),
            "tag_requests": sum(1 for r in RECORDS if "gemini-3.1-flash-lite" in r["path"]),
            "tts_requests": sum(1 for r in RECORDS if "gemini-3.8-flash-tts" in r["path"]),
            "tts_text": next((
                r["body"].get("contents", [{}])[0].get("parts", [{}])[0].get("text", "")
                for r in RECORDS if "gemini-3.8-flash-tts" in r["path"]
            ), ""),
        }))
        return 0
    finally:
        os.environ["PATH"] = original_path
        server.shutdown()
        server.server_close()
        shutil.rmtree(home, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
