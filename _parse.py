"""Tolerant parsing of an Interactions response.

The REST reference documents ``steps[]`` (``model_output`` -> ``content[]`` -> ``text``), the
Python/JS SDKs expose ``output_text``, and Google's own Files-API curl example parses
``outputs[]``. All three are accepted so a surface change cannot silently produce an empty
transcript.
"""

from __future__ import annotations

from typing import Any

from ._transport import GeminiAPIError

_TERMINAL_FAILURES = frozenset({"failed", "cancelled", "requires_action"})


def decode_json(response) -> dict[str, Any]:
    """Parse a 2xx response body, converting an unexpected shape into a transport error."""
    try:
        payload = response.json()
    except Exception:
        body = " ".join((response.text or "").split())[:300]
        raise GeminiAPIError(f"Gemini returned a non-JSON response: {body}") from None
    if not isinstance(payload, dict):
        raise GeminiAPIError("Gemini returned an unexpected response shape.")
    return payload


def status_of(payload: dict[str, Any]) -> str:
    status = payload.get("status") if isinstance(payload, dict) else None
    return str(status or "completed").strip().lower()


def is_terminal_failure(status: str) -> bool:
    return status in _TERMINAL_FAILURES


def extract_transcript(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""

    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    outputs: list[str] = []
    for block in payload.get("outputs") or []:
        if isinstance(block, dict) and block.get("type") in (None, "text"):
            text = block.get("text")
            if isinstance(text, str) and text:
                outputs.append(text)
    if outputs:
        return "\n".join(outputs)

    steps: list[str] = []
    for step in payload.get("steps") or []:
        if not isinstance(step, dict) or step.get("type") != "model_output":
            continue
        for block in step.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    steps.append(text)
    return "\n".join(steps)


def response_error(payload: dict[str, Any]) -> str | None:
    """Human-readable failure detail, or ``None`` for a completed interaction.

    An ``incomplete`` response may still carry usable partial text; the caller decides whether
    to surface it, so this function reports the error without discarding the transcript.
    """
    if not isinstance(payload, dict):
        return "Gemini returned an unreadable response."

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("status")
        if message:
            return str(message)
    elif isinstance(error, str) and error.strip():
        return error.strip()

    status = status_of(payload)
    if status == "incomplete":
        return "Gemini stopped before finishing the transcript (incomplete response)."
    if is_terminal_failure(status):
        return f"Gemini interaction ended with status '{status}'."
    return None
