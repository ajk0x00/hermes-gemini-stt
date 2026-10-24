from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from ._load import load_plugin

load_plugin()
from gemini_stt_under_test._transport import GeminiAPIError, GeminiTransport  # noqa: E402


def _response(status: int, payload: dict, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status, json=payload, headers=headers or {},
        request=httpx.Request("POST", "https://example.test/v1beta/interactions"),
    )


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def close(self) -> None:
        return

    def request(self, *args, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _transport(outcomes, *, max_retries=2) -> tuple[GeminiTransport, FakeClient]:
    transport = GeminiTransport(
        api_key="secret-key", base_url="https://example.test/v1beta",
        api_root="https://example.test", timeout=30, max_retries=max_retries, user_agent="test",
    )
    client = FakeClient(outcomes)
    transport._client = client
    return transport, client


class TransportTests(unittest.TestCase):
    def test_success_first_attempt(self):
        transport, client = _transport([_response(200, {"ok": True})])
        response = transport.request("POST", "https://example.test/v1beta/interactions", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(client.calls, 1)
        transport.close()

    def test_retries_retryable_status(self):
        transport, client = _transport([
            _response(503, {"error": {"message": "busy"}}),
            _response(200, {"ok": True}),
        ])
        with patch("time.sleep"):
            response = transport.request("POST", "https://example.test/v1beta/interactions", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(client.calls, 2)
        transport.close()

    def test_does_not_retry_client_errors(self):
        transport, client = _transport([_response(400, {"error": {"message": "bad audio"}})])
        with self.assertRaises(GeminiAPIError) as caught:
            transport.request("POST", "https://example.test/v1beta/interactions", json={})
        self.assertEqual(client.calls, 1)
        self.assertIn("400", str(caught.exception))
        self.assertIn("bad audio", str(caught.exception))
        self.assertFalse(caught.exception.retryable)
        transport.close()

    def test_honours_retry_after(self):
        transport, client = _transport([
            _response(429, {"error": {"message": "slow down"}}, {"Retry-After": "0"}),
            _response(200, {"ok": True}),
        ])
        with patch("time.sleep") as sleeper:
            transport.request("POST", "https://example.test/v1beta/interactions", json={})
        sleeper.assert_called()
        self.assertEqual(client.calls, 2)
        transport.close()

    def test_read_timeout_is_not_retried(self):
        transport, client = _transport([httpx.ReadTimeout("slow")])
        with self.assertRaises(GeminiAPIError) as caught:
            transport.request("POST", "https://example.test/v1beta/interactions", json={})
        self.assertIn("did not respond", str(caught.exception))
        self.assertEqual(client.calls, 1)
        transport.close()

    def test_connect_error_is_retried(self):
        transport, client = _transport([
            httpx.ConnectError("refused"),
            _response(200, {"ok": True}),
        ])
        with patch("time.sleep"):
            response = transport.request("POST", "https://example.test/v1beta/interactions", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(client.calls, 2)
        transport.close()

    def test_no_retry_does_not_sleep_after_a_connect_error(self):
        transport, client = _transport([httpx.ConnectError("refused")], max_retries=5)
        with patch("time.sleep") as sleeper:
            with self.assertRaises(GeminiAPIError):
                transport.request("POST", "https://example.test/v1beta/interactions", json={}, retry=False)
        self.assertEqual(client.calls, 1)
        sleeper.assert_not_called()
        transport.close()

    def test_exhausted_retries_raise_last_error(self):
        transport, client = _transport([
            _response(503, {"error": {"message": "one"}}),
            _response(503, {"error": {"message": "two"}}),
        ], max_retries=1)
        with patch("time.sleep"):
            with self.assertRaises(GeminiAPIError) as caught:
                transport.request("POST", "https://example.test/v1beta/interactions", json={})
        self.assertEqual(client.calls, 2)
        self.assertIn("two", str(caught.exception))
        transport.close()

    def test_headers_carry_the_key_and_user_agent(self):
        recorded = {}

        class Capturing(FakeClient):
            def request(self, method, url, **kwargs):
                recorded.update(kwargs)
                return _response(200, {"ok": True})

        transport, _ = _transport([])
        transport._client = Capturing([])
        transport.request("POST", "https://example.test/v1beta/interactions", json={})
        headers = recorded["headers"]
        self.assertEqual(headers["x-goog-api-key"], "secret-key")
        self.assertEqual(headers["User-Agent"], "test")
        transport.close()

    def test_unauthenticated_request_omits_the_key(self):
        recorded = {}

        class Capturing(FakeClient):
            def request(self, method, url, **kwargs):
                recorded.update(kwargs)
                return _response(200, {"ok": True})

        transport, _ = _transport([])
        transport._client = Capturing([])
        transport.request("POST", "https://upload.test/session", authenticated=False)
        self.assertNotIn("x-goog-api-key", recorded["headers"])
        transport.close()


if __name__ == "__main__":
    unittest.main()
