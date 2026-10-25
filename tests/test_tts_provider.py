from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from ._load import FakeContext, load_plugin
from ._server import StubServer

plugin = load_plugin()


def _wav_bytes(frames: int = 2400) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setframerate(24000)
        handle.setsampwidth(2)
        handle.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


def _audio_response(raw: bytes, mime: str) -> dict:
    return {
        "candidates": [{
            "content": {"parts": [{"inlineData": {"mimeType": mime, "data": base64.b64encode(raw).decode()}}]},
            "finishReason": "STOP",
        }]
    }


class TTSProviderTests(unittest.TestCase):
    def setUp(self):
        self.server = StubServer().start()
        self.addCleanup(self.server.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)

    def _provider(self, **settings):
        values = {
            "base_url": f"{self.server.base_url}/v1beta",
            "tts_timeout": 20,
            "tts_max_retries": 0,
            "tts_tag_timeout": 2,
            "tts_audio_tags": False,
        }
        values.update(settings)
        return plugin.GeminiTTSProvider(FakeContext(values))

    def _install_tts(self, *, raw=None, mime="audio/wav", status=200, payload=None):
        raw = _wav_bytes() if raw is None else raw

        def handler(record):
            if payload is not None:
                return status, {}, payload
            return 200, {}, _audio_response(raw, mime)

        self.server.routes[("POST", "/v1beta/models/gemini-3.8-flash-tts:generateContent")] = handler
        return handler

    def _install_tag(self, text: str, status: int = 200):
        def handler(record):
            if status != 200:
                return status, {}, {"error": {"message": "tagger exploded"}}
            return 200, {}, {
                "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]
            }

        self.server.routes[
            ("POST", "/v1beta/models/gemini-3.1-flash-lite:generateContent")
        ] = handler

    def _out(self, suffix=".wav") -> Path:
        return self.dir / f"out{suffix}"

    def test_wav_response_is_written_without_double_wrap(self):
        raw = _wav_bytes()
        self._install_tts(raw=raw)
        result = self._provider().synthesize(
            "hello", str(self._out()), model="gemini-3.8-flash-tts", format="wav",
        )
        self.assertEqual(Path(result).read_bytes(), raw)
        with wave.open(result, "rb") as handle:
            self.assertEqual(handle.getnframes(), 2400)
            self.assertEqual(handle.getframerate(), 24000)
            self.assertEqual(handle.getnchannels(), 1)

        body = self.server.json_body(self.server.calls("POST", "/v1beta/models/gemini-3.8-flash-tts:generateContent")[0])
        self.assertEqual(body["contents"][0]["parts"][0]["text"], "hello")
        self.assertEqual(body["generationConfig"]["responseModalities"], ["AUDIO"])
        self.assertEqual(
            body["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"],
            "Kore",
        )

    def test_l16_response_is_wrapped(self):
        pcm = b"\x01\x00" * 2400
        self._install_tts(raw=pcm, mime="audio/l16; rate=24000; channels=1")
        result = self._provider().synthesize(
            "hello", str(self._out()), model="gemini-3.8-flash-tts", format="wav",
        )
        with wave.open(result, "rb") as handle:
            self.assertEqual(handle.getnframes(), 2400)
            self.assertEqual(handle.getframerate(), 24000)
            self.assertEqual(handle.readframes(2400), pcm)

    def test_style_is_sent_as_speech_metadata(self):
        self._install_tts()
        self._provider(tts_style="whispering").synthesize(
            "hello", str(self._out()), model="gemini-3.8-flash-tts", format="wav",
        )
        body = self.server.json_body(self.server.calls("POST", "/v1beta/models/gemini-3.8-flash-tts:generateContent")[0])
        self.assertEqual(body["contents"][0]["parts"][0]["speech_metadata"], {"style": "whispering"})

    def test_style_is_not_sent_to_legacy_model(self):
        self.server.routes[("POST", "/v1beta/models/gemini-2.5-flash-preview-tts:generateContent")] = (
            lambda record: (200, {}, _audio_response(_wav_bytes(), "audio/l16; rate=24000; channels=1"))
        )
        self._provider(tts_style="whispering").synthesize(
            "hello", str(self._out()), model="gemini-2.5-flash-preview-tts", format="wav",
        )
        body = self.server.json_body(self.server.calls("POST", "/v1beta/models/gemini-2.5-flash-preview-tts:generateContent")[0])
        self.assertNotIn("speech_metadata", body["contents"][0]["parts"][0])

    def test_missing_key_raises(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=False):
            with self.assertRaises(ValueError) as ctx:
                self._provider().synthesize("hello", str(self._out()))
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_api_error_is_redacted(self):
        self.server.routes[("POST", "/v1beta/models/gemini-3.8-flash-tts:generateContent")] = (
            lambda record: (401, {}, {"error": {"message": "bad key test-key"}})
        )
        with self.assertRaises(RuntimeError) as ctx:
            self._provider().synthesize("hello", str(self._out()))
        self.assertIn("401", str(ctx.exception))
        self.assertNotIn("test-key", str(ctx.exception))

    def test_auto_tagging_passes_tagged_text(self):
        source = (
            "Honestly I was not expecting that. The build passed on the first try and every "
            "test is green."
        )
        tagged = (
            "Honestly <laugh> I was not expecting that. The build passed on the first try and "
            "every test is green."
        )
        self._install_tts()
        self._install_tag(tagged)
        self._provider(tts_audio_tags=True).synthesize(
            source, str(self._out()), model="gemini-3.8-flash-tts", format="wav",
        )
        body = self.server.json_body(self.server.calls("POST", "/v1beta/models/gemini-3.8-flash-tts:generateContent")[0])
        self.assertIn("<laugh>", body["contents"][0]["parts"][0]["text"])
        self.assertEqual(
            len(self.server.calls("POST", "/v1beta/models/gemini-3.1-flash-lite:generateContent")), 1,
        )

    def test_tagging_failure_falls_back_to_untagged(self):
        source = (
            "Honestly I was not expecting that. The build passed on the first try and every "
            "test is green."
        )
        self._install_tts()
        self._install_tag("", status=500)
        self._provider(tts_audio_tags=True).synthesize(
            source, str(self._out()), model="gemini-3.8-flash-tts", format="wav",
        )
        body = self.server.json_body(self.server.calls("POST", "/v1beta/models/gemini-3.8-flash-tts:generateContent")[0])
        self.assertEqual(body["contents"][0]["parts"][0]["text"], source)
        self.assertEqual(
            len(self.server.calls("POST", "/v1beta/models/gemini-3.1-flash-lite:generateContent")), 1,
        )

    def test_short_text_and_manual_tags_skip_tagger(self):
        self._install_tts()
        provider = self._provider(tts_audio_tags=True)
        provider.synthesize("Hello there.", str(self._out()), model="gemini-3.8-flash-tts", format="wav")
        provider.synthesize(
            "Hello there. [laughs] This text already has a tag and enough words to be considered.",
            str(self._out()), model="gemini-3.8-flash-tts", format="wav",
        )
        self.assertEqual(
            self.server.calls("POST", "/v1beta/models/gemini-3.1-flash-lite:generateContent"), [],
        )
        body = self.server.json_body(self.server.calls("POST", "/v1beta/models/gemini-3.8-flash-tts:generateContent")[-1])
        self.assertIn("<laugh>", body["contents"][0]["parts"][0]["text"])
        self.assertNotIn("[laughs]", body["contents"][0]["parts"][0]["text"])

    def test_tag_cache_avoids_second_tagger_call(self):
        source = (
            "Honestly I was not expecting that. The build passed on the first try and every "
            "test is green."
        )
        tagged = (
            "Honestly <laugh> I was not expecting that. The build passed on the first try and "
            "every test is green."
        )
        self._install_tts()
        self._install_tag(tagged)
        provider = self._provider(tts_audio_tags=True)
        provider.synthesize(source, str(self._out()), model="gemini-3.8-flash-tts", format="wav")
        provider.synthesize(source, str(self._out()), model="gemini-3.8-flash-tts", format="wav")
        self.assertEqual(
            len(self.server.calls("POST", "/v1beta/models/gemini-3.1-flash-lite:generateContent")), 1,
        )

    def test_ffmpeg_fallback_writes_wav(self):
        self._install_tts()
        provider = self._provider(tts_audio_tags=False)
        with patch.object(provider, "_find_ffmpeg", return_value=None):
            result = provider.synthesize(
                "hello", str(self._out(".mp3")), model="gemini-3.8-flash-tts", format="mp3",
            )
            self.assertFalse(provider.voice_compatible)
        self.assertTrue(result.endswith(".wav"))
        self.assertEqual(Path(result).read_bytes()[:4], b"RIFF")

    def test_ffmpeg_conversion_uses_the_binary_and_returns_target(self):
        self._install_tts()
        fake = self.dir / "ffmpeg"
        fake.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "open(sys.argv[-1], 'wb').write(b'converted-audio')\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        provider = self._provider(tts_audio_tags=False)
        with patch.object(provider, "_find_ffmpeg", return_value=str(fake)):
            result = provider.synthesize(
                "hello", str(self._out(".mp3")), model="gemini-3.8-flash-tts", format="mp3",
            )
            self.assertTrue(provider.voice_compatible)
        self.assertTrue(result.endswith(".mp3"))
        self.assertEqual(Path(result).read_bytes(), b"converted-audio")

    def test_real_ffmpeg_produces_ogg_opus(self):
        ffmpeg = self._provider()._find_ffmpeg()
        if not ffmpeg:
            self.skipTest("no ffmpeg available")
        provider = self._provider(tts_ffmpeg_path=ffmpeg)
        result = provider._write_audio(
            _wav_bytes(), "audio/wav", str(self._out(".ogg")), "ogg",
        )
        self.assertTrue(result.endswith(".ogg"))
        self.assertEqual(Path(result).read_bytes()[:4], b"OggS")

    def test_stream_applies_tags(self):
        source = (
            "Honestly I was not expecting that. The build passed on the first try and every "
            "test is green."
        )
        tagged = (
            "Honestly <laugh> I was not expecting that. The build passed on the first try and "
            "every test is green."
        )
        self._install_tag(tagged)
        pcm = b"\x01\x00" * 480
        self.server.routes[("POST", "/v1beta/models/gemini-3.8-flash-tts:streamGenerateContent")] = (
            lambda record: (200, {"Content-Type": "text/event-stream"}, (
                "data: " + json.dumps({
                    "candidates": [{"content": {"parts": [{"inlineData": {
                        "mimeType": "audio/l16; rate=24000; channels=1",
                        "data": base64.b64encode(pcm).decode(),
                    }}]}}]
                }) + "\n\n"
            ).encode())
        )
        provider = self._provider(tts_audio_tags=True)
        self.assertEqual(b"".join(provider.stream(source, model="gemini-3.8-flash-tts")), pcm)
        body = self.server.json_body(
            self.server.calls("POST", "/v1beta/models/gemini-3.8-flash-tts:streamGenerateContent")[0]
        )
        self.assertIn("<laugh>", body["contents"][0]["parts"][0]["text"])

    def test_stream_yields_pcm_from_sse(self):
        pcm1 = b"\x01\x00" * 480
        pcm2 = b"\x02\x00" * 480
        def sse(chunk):
            return (
                "data: " + json.dumps({
                    "candidates": [{"content": {"parts": [{"inlineData": {
                        "mimeType": "audio/l16; rate=24000; channels=1",
                        "data": base64.b64encode(chunk).decode(),
                    }}]}}]
                }) + "\n\n"
            ).encode()

        self.server.routes[("POST", "/v1beta/models/gemini-3.8-flash-tts:streamGenerateContent")] = (
            lambda record: (200, {"Content-Type": "text/event-stream"}, sse(pcm1) + sse(pcm2))
        )
        provider = self._provider(tts_audio_tags=False)
        chunks = list(provider.stream("hello", model="gemini-3.8-flash-tts", format="pcm"))
        self.assertEqual(b"".join(chunks), pcm1 + pcm2)

    def test_stream_calls_are_thread_safe(self):
        from concurrent.futures import ThreadPoolExecutor

        pcm = b"\x03\x00" * 480
        self.server.routes[("POST", "/v1beta/models/gemini-3.8-flash-tts:streamGenerateContent")] = (
            lambda record: (200, {"Content-Type": "text/event-stream"}, (
                "data: " + json.dumps({
                    "candidates": [{"content": {"parts": [{"inlineData": {
                        "mimeType": "audio/l16; rate=24000; channels=1",
                        "data": base64.b64encode(pcm).decode(),
                    }}]}}]
                }) + "\n\n"
            ).encode())
        )
        provider = self._provider(tts_audio_tags=False)
        with ThreadPoolExecutor(max_workers=3) as pool:
            outputs = list(pool.map(
                lambda _: b"".join(provider.stream("hello", model="gemini-3.8-flash-tts")),
                range(3),
            ))
        self.assertEqual(outputs, [pcm, pcm, pcm])

    def test_stream_tagging_can_be_disabled(self):
        source = (
            "Honestly I was not expecting that. The build passed on the first try and every "
            "test is green."
        )
        pcm = b"\x01\x00" * 480
        self.server.routes[("POST", "/v1beta/models/gemini-3.8-flash-tts:streamGenerateContent")] = (
            lambda record: (200, {"Content-Type": "text/event-stream"}, (
                "data: " + json.dumps({
                    "candidates": [{"content": {"parts": [{"inlineData": {
                        "mimeType": "audio/l16; rate=24000; channels=1",
                        "data": base64.b64encode(pcm).decode(),
                    }}]}}]
                }) + "\n\n"
            ).encode())
        )
        provider = self._provider(tts_audio_tags=True, tts_audio_tags_streaming=False)
        self.assertEqual(b"".join(provider.stream(source, model="gemini-3.8-flash-tts")), pcm)
        self.assertEqual(
            self.server.calls("POST", "/v1beta/models/gemini-3.1-flash-lite:generateContent"), [],
        )

    def test_stream_error_raises(self):
        self.server.routes[("POST", "/v1beta/models/gemini-3.8-flash-tts:streamGenerateContent")] = (
            lambda record: (400, {}, {"error": {"message": "bad stream"}})
        )
        provider = self._provider(tts_audio_tags=False)
        with self.assertRaises(RuntimeError) as ctx:
            list(provider.stream("hello", model="gemini-3.8-flash-tts", format="pcm"))
        self.assertIn("400", str(ctx.exception))

    def test_catalog_surface(self):
        provider = self._provider(tts_model="gemini-3.8-flash-lite-tts", tts_voice="Puck")
        self.assertTrue(provider.is_available())
        self.assertEqual(provider.default_model(), "gemini-3.8-flash-lite-tts")
        self.assertEqual(provider.default_voice(), "Puck")
        self.assertEqual({m["id"] for m in provider.list_models() if m.get("default")}, {"gemini-3.8-flash-tts"})
        voices = {v["id"] for v in provider.list_voices()}
        self.assertIn("Kore", voices)
        self.assertIn("Puck", voices)
        self.assertEqual(len(voices), 30)
        self.assertEqual(provider.get_setup_schema()["env_vars"][0]["key"], "GEMINI_API_KEY")

    def test_is_available_tracks_the_key(self):
        provider = self._provider()
        self.assertTrue(provider.is_available())
        with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=False):
            self.assertFalse(provider.is_available())

    def test_voice_compatible_tracks_ffmpeg(self):
        provider = self._provider()
        with patch.object(provider, "_find_ffmpeg", return_value="/usr/bin/ffmpeg"):
            self.assertTrue(provider.voice_compatible)
        with patch.object(provider, "_find_ffmpeg", return_value=None):
            self.assertFalse(provider.voice_compatible)


if __name__ == "__main__":
    unittest.main()
