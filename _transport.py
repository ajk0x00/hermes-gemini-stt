"""HTTP transport with the retry rules that matter for a paid, non-idempotent endpoint.

A read timeout on ``POST /interactions`` is deliberately NOT retried: the request may already
have been processed, and a retry silently doubles the bill. Connect failures, 429 and 5xx are
safe to retry because the server rejected or never accepted the work.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

logger = logging.getLogger("hermes.plugins.gemini_stt")

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_BACKOFF_SECONDS = 10.0
_CONNECT_TIMEOUT = 10.0


class GeminiAPIError(RuntimeError):
    """A transport-level failure carrying the pieces the caller needs for an envelope."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.retry_after = retry_after


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except ValueError:
        return None  # HTTP-date form: fall back to the normal backoff


class GeminiTransport:
    def __init__(
        self, *, api_key: str, base_url: str, api_root: str,
        timeout: float, max_retries: int, user_agent: str,
    ) -> None:
        import httpx

        self._httpx = httpx
        self.api_key = api_key
        self.base_url = base_url
        self.api_root = api_root
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_agent = user_agent
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=min(_CONNECT_TIMEOUT, timeout)),
            follow_redirects=False,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GeminiTransport":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _headers(self, extra: dict[str, str] | None = None, *, authenticated: bool = True) -> dict[str, str]:
        headers = {"User-Agent": self.user_agent}
        if authenticated:
            headers["x-goog-api-key"] = self.api_key
        if extra:
            headers.update(extra)
        return headers

    def _timeout(self, read_timeout: float | None):
        seconds = read_timeout or self.timeout
        return self._httpx.Timeout(seconds, connect=min(_CONNECT_TIMEOUT, seconds))

    def request(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
        read_timeout: float | None = None,
        authenticated: bool = True,
        retry: bool = True,
    ):
        """One HTTP exchange with bounded retries; raises :class:`GeminiAPIError` on failure."""
        attempts = (self.max_retries + 1) if retry else 1
        last_error: GeminiAPIError | None = None

        for attempt in range(attempts):
            try:
                response = self._client.request(
                    method, url, json=json, content=content,
                    headers=self._headers(headers, authenticated=authenticated),
                    timeout=self._timeout(read_timeout),
                )
            except (self._httpx.ConnectError, self._httpx.ConnectTimeout, self._httpx.PoolTimeout) as exc:
                last_error = GeminiAPIError(f"Could not connect to Gemini: {exc}", retryable=True)
            except self._httpx.ReadTimeout:
                raise GeminiAPIError(
                    f"Gemini did not respond within {read_timeout or self.timeout:g}s."
                ) from None
            except self._httpx.TimeoutException:
                raise GeminiAPIError(f"Gemini request timed out after {read_timeout or self.timeout:g}s.") from None
            except self._httpx.HTTPError as exc:
                raise GeminiAPIError(f"Gemini request failed: {exc}") from None
            else:
                if response.status_code < 400:
                    return response
                last_error = self._error_from_response(response)
                if response.status_code not in RETRYABLE_STATUS or attempt + 1 >= attempts:
                    raise last_error

            if attempt + 1 >= attempts:
                break
            delay = last_error.retry_after if last_error and last_error.retry_after is not None else (
                0.5 * (2 ** attempt) + random.uniform(0.0, 0.25)
            )
            time.sleep(min(delay, _MAX_BACKOFF_SECONDS))

        raise last_error or GeminiAPIError("Gemini request failed.", retryable=True)

    def _error_from_response(self, response) -> GeminiAPIError:
        detail = ""
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                detail = str(error.get("message") or error.get("status") or "")
            elif isinstance(error, str):
                detail = error
        if not detail:
            detail = " ".join((response.text or "").split())[:300]
        message = f"Gemini API error ({response.status_code})"
        if detail:
            message = f"{message}: {detail}"
        return GeminiAPIError(
            message,
            status=response.status_code,
            retryable=response.status_code in RETRYABLE_STATUS,
            retry_after=_retry_after_seconds(response.headers.get("Retry-After")),
        )
