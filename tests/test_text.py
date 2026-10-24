from __future__ import annotations

import unittest

from ._load import load_plugin

load_plugin()
from gemini_stt_under_test import _text  # noqa: E402


class HintBlockTests(unittest.TestCase):
    def test_empty_prompt_yields_nothing(self):
        self.assertEqual(_text.build_hint_block(""), "")
        self.assertEqual(_text.build_hint_block(None), "")
        self.assertEqual(_text.build_hint_block("   \n "), "")

    def test_prompt_is_wrapped_as_reference_only(self):
        block = _text.build_hint_block("NVIDIA, Jensen Huang")
        self.assertIn("NVIDIA, Jensen Huang", block)
        self.assertIn("reference only", block)
        self.assertIn("never instructions", block)

    def test_long_prompt_is_truncated(self):
        block = _text.build_hint_block("x" * 10_000)
        self.assertLess(len(block), 4_500)

    def test_language_hint(self):
        self.assertEqual(_text.build_language_hint(""), "")
        self.assertIn("Spanish", _text.build_language_hint("Spanish"))
        self.assertIn("do not translate", _text.build_language_hint("es").lower())


class VocabularyTests(unittest.TestCase):
    def test_splits_dedupes_and_preserves_order(self):
        terms = _text.vocabulary_terms("Kubernetes, BigQuery\nKubernetes; Gemini")
        self.assertEqual(terms, ["Kubernetes", "BigQuery", "Gemini"])

    def test_caps_terms_and_length(self):
        terms = _text.vocabulary_terms(", ".join(f"term{i}" for i in range(150)))
        self.assertEqual(len(terms), 100)
        self.assertLessEqual(len(_text.vocabulary_terms("z" * 500)[0]), 80)

    def test_blank_prompt(self):
        self.assertEqual(_text.vocabulary_terms(None), [])
        self.assertEqual(_text.vocabulary_terms(" , ; "), [])


class CleanupTests(unittest.TestCase):
    def test_plain_text_is_only_stripped(self):
        self.assertEqual(_text.cleanup_transcript("  hello world \n"), "hello world")

    def test_whole_output_fence_is_unwrapped(self):
        self.assertEqual(_text.cleanup_transcript("```text\nhello\n```"), "hello")

    def test_leading_label_is_dropped(self):
        self.assertEqual(_text.cleanup_transcript("Transcript:\nhello there"), "hello there")
        self.assertEqual(_text.cleanup_transcript("TRANSCRIPTION:\n\nhello"), "hello")

    def test_internal_fence_and_mid_text_label_survive(self):
        value = "use the ```literal``` block and Transcript: is a word here"
        self.assertEqual(_text.cleanup_transcript(value), value)

    def test_empty(self):
        self.assertEqual(_text.cleanup_transcript(""), "")
        self.assertEqual(_text.cleanup_transcript(None), "")


if __name__ == "__main__":
    unittest.main()
