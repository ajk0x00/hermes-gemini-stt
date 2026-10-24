from __future__ import annotations

import unittest

from ._load import load_plugin

load_plugin()
from gemini_stt_under_test import _parse  # noqa: E402
from gemini_stt_under_test._transport import GeminiAPIError  # noqa: E402


class ExtractTests(unittest.TestCase):
    def test_output_text(self):
        self.assertEqual(_parse.extract_transcript({"output_text": "hello"}), "hello")

    def test_outputs_array(self):
        payload = {"outputs": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}
        self.assertEqual(_parse.extract_transcript(payload), "one\ntwo")

    def test_steps_model_output(self):
        payload = {
            "steps": [
                {"type": "thought", "summary": [{"type": "text", "text": "ignore me"}]},
                {"type": "model_output", "content": [{"type": "text", "text": "the transcript"}]},
            ]
        }
        self.assertEqual(_parse.extract_transcript(payload), "the transcript")

    def test_precedence_prefers_output_text(self):
        payload = {"output_text": "direct", "steps": [{"type": "model_output", "content": [{"type": "text", "text": "other"}]}]}
        self.assertEqual(_parse.extract_transcript(payload), "direct")

    def test_empty_and_odd_shapes(self):
        self.assertEqual(_parse.extract_transcript({}), "")
        self.assertEqual(_parse.extract_transcript(None), "")

    def test_ignores_non_text_blocks(self):
        payload = {"steps": [{"type": "model_output", "content": [{"type": "image", "uri": "x"}]}]}
        self.assertEqual(_parse.extract_transcript(payload), "")


class ErrorTests(unittest.TestCase):
    def test_completed_has_no_error(self):
        self.assertIsNone(_parse.response_error({"status": "completed"}))

    def test_error_object(self):
        self.assertEqual(
            _parse.response_error({"error": {"message": "bad key"}}), "bad key",
        )

    def test_failed_status(self):
        message = _parse.response_error({"status": "failed"})
        self.assertIn("failed", message)

    def test_incomplete_is_reported(self):
        self.assertIn("incomplete", _parse.response_error({"status": "incomplete"}))

    def test_terminal_failures(self):
        self.assertTrue(_parse.is_terminal_failure("cancelled"))
        self.assertFalse(_parse.is_terminal_failure("incomplete"))
        self.assertFalse(_parse.is_terminal_failure("completed"))


class DecodeTests(unittest.TestCase):
    def test_valid_json(self):
        class Response:
            def json(self):
                return {"status": "completed"}

        self.assertEqual(_parse.decode_json(Response()), {"status": "completed"})

    def test_non_json_raises_transport_error(self):
        class Response:
            text = "<html>gateway error</html>"

            def json(self):
                raise ValueError("not json")

        with self.assertRaises(GeminiAPIError) as caught:
            _parse.decode_json(Response())
        self.assertIn("non-JSON", str(caught.exception))

    def test_non_dict_raises(self):
        class Response:
            text = "[]"

            def json(self):
                return []

        with self.assertRaises(GeminiAPIError):
            _parse.decode_json(Response())


if __name__ == "__main__":
    unittest.main()
