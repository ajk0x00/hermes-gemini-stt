"""A tiny threaded HTTP stub with a route table; no external test dependencies."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class StubServer:
    def __init__(self) -> None:
        self.records: list[dict] = []
        self.routes: dict[tuple[str, str], object] = {}
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def _make_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:  # keep test output clean
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
                record = {
                    "method": method,
                    "path": path,
                    "headers": {key.lower(): value for key, value in self.headers.items()},
                    "body": body,
                }
                outer.records.append(record)
                route = outer.routes.get((method, path))
                if route is None:
                    status, headers, payload = 404, {}, {"error": {"message": f"no route for {method} {path}"}}
                else:
                    status, headers, payload = route(record)
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "StubServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def __enter__(self) -> "StubServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def json_body(self, record: dict) -> dict:
        return json.loads(record["body"] or b"{}")

    def calls(self, method: str, path: str) -> list[dict]:
        return [r for r in self.records if r["method"] == method and r["path"] == path]
