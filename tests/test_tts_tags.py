from __future__ import annotations

import unittest

from ._load import load_plugin

load_plugin()
from gemini_stt_under_test import _tts_tags  # noqa: E402


class TagStyleTests(unittest.TestCase):
    def test_models(self):
        self.assertEqual(_tts_tags.tag_style("gemini-3.8-flash-tts"), "angle")
        self.assertEqual(_tts_tags.tag_style("models/gemini-3.8-flash-lite-tts"), "angle")
        self.assertEqual(_tts_tags.tag_style("gemini-3.1-flash-tts-preview"), "square")
        self.assertIsNone(_tts_tags.tag_style("gemini-2.5-flash-preview-tts"))
        self.assertIsNone(_tts_tags.tag_style("gemini-3.8-flash"))

    def test_has_tags_and_word_count(self):
        self.assertTrue(_tts_tags.has_tags("hello <laugh> there"))
        self.assertTrue(_tts_tags.has_tags("hello [laughs] there"))
        self.assertFalse(_tts_tags.has_tags("hello [1] there"))
        self.assertTrue(_tts_tags.has_tags("hello <laugh> there", style="angle"))
        self.assertTrue(_tts_tags.has_tags("hello [laughs] there", style="angle"))
        self.assertFalse(_tts_tags.has_tags("hello <variable> there", style="angle"))
        self.assertFalse(_tts_tags.has_tags("hello [excited] there", style="angle"))
        self.assertEqual(_tts_tags.word_count("one two three"), 3)
        self.assertEqual(_tts_tags.word_count(""), 0)


class SanitizerTests(unittest.TestCase):
    SOURCE = (
        "Honestly I was not expecting that. The build passed on the first try and every "
        "test is green. Let's ship it."
    )

    def test_accepts_angle_tags_and_keeps_words(self):
        tagged = "Honestly <laugh> I was not expecting that. The build passed on the first try and every test is green. <short pause> Let's ship it."
        result = _tts_tags.sanitize_tagged_text(self.SOURCE, tagged, style="angle")
        self.assertIsNotNone(result)
        self.assertIn("<laugh>", result)
        self.assertEqual(_tts_tags.word_count(result.replace("<laugh>", "")),
                         _tts_tags.word_count(self.SOURCE))

    def test_strips_closing_and_unknown_tags(self):
        tagged = "<laugh> Honestly I was not expecting that. </laugh> <banana> The build passed on the first try and every test is green. Let's ship it."
        result = _tts_tags.sanitize_tagged_text(self.SOURCE, tagged, style="angle")
        self.assertIsNotNone(result)
        self.assertIn("<laugh>", result)
        self.assertNotIn("</", result)
        self.assertNotIn("banana", result)

    def test_converts_square_tags_to_angle(self):
        tagged = "Honestly [laughs] I was not expecting that. The build passed on the first try and every test is green. Let's ship it."
        result = _tts_tags.sanitize_tagged_text(self.SOURCE, tagged, style="angle")
        self.assertIsNotNone(result)
        self.assertIn("<", result)
        self.assertNotIn("[", result)

    def test_rejects_added_or_removed_words(self):
        tagged = "Honestly <laugh> I was not expecting that. The build passed and every test is green. Let's ship it."
        self.assertIsNone(_tts_tags.sanitize_tagged_text(self.SOURCE, tagged, style="angle"))

    def test_rejects_a_rewrite_without_tags(self):
        self.assertIsNone(_tts_tags.sanitize_tagged_text(self.SOURCE, self.SOURCE, style="angle"))

    def test_limits_tags_per_sentence(self):
        tagged = (
            "<laugh> Honestly <gasp> I was not expecting that. The build passed on the first try "
            "and every test is green. Let's ship it."
        )
        result = _tts_tags.sanitize_tagged_text(self.SOURCE, tagged, style="angle")
        self.assertIsNotNone(result)
        self.assertEqual(result.count("<"), 1)

    def test_square_style_round_trip(self):
        tagged = "Honestly [laughs] I was not expecting that. The build passed on the first try and every test is green. Let's ship it."
        result = _tts_tags.sanitize_tagged_text(self.SOURCE, tagged, style="square")
        self.assertIsNotNone(result)
        self.assertIn("[laughs]", result)
        self.assertNotIn("<", result)

    def test_self_closing_tags_are_removed_or_canonicalised(self):
        tagged = "<laugh/> Honestly I was not expecting that. The build passed on the first try and every test is green. Let's ship it."
        result = _tts_tags.sanitize_tagged_text(self.SOURCE, tagged, style="angle")
        self.assertIsNotNone(result)
        self.assertIn("<laugh>", result)
        self.assertNotIn("/>", result)

    def test_manual_square_tags_become_angle(self):
        text = "Honestly [laughs] I was not expecting that."
        self.assertEqual(
            _tts_tags.canonicalize_tags(text, style="angle"),
            "Honestly <laugh> I was not expecting that.",
        )

    def test_manual_canonicalization_preserves_natural_brackets(self):
        text = "Set <variable> and [laughs] in the template."
        self.assertEqual(
            _tts_tags.canonicalize_tags(text, style="angle"),
            "Set <variable> and <laugh> in the template.",
        )

    def test_payload_contains_style_rules(self):
        payload = _tts_tags.build_tag_payload("hello world", style="angle")
        self.assertIn("angle-bracket", payload["system_instruction"]["parts"][0]["text"])
        self.assertEqual(payload["contents"][0]["parts"][0]["text"], "hello world")

    def test_extract_tagged_text(self):
        payload = {"candidates": [{"content": {"parts": [{"text": "hello <laugh>"}]}}]}
        self.assertEqual(_tts_tags.extract_tagged_text(payload), "hello <laugh>")
        self.assertEqual(_tts_tags.extract_tagged_text({}), "")
        self.assertEqual(
            _tts_tags.extract_tagged_text({"promptFeedback": {"blockReason": "SAFETY"}}), "",
        )


if __name__ == "__main__":
    unittest.main()
