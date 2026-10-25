# Hermes Gemini Voice

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Hermes](https://img.shields.io/badge/Hermes-%E2%89%A50.22.2-6f42c1.svg)](https://github.com/NousResearch/hermes-agent)

A production-grade Google Gemini voice plugin for [Hermes](https://github.com/NousResearch/hermes-agent):

- **STT:** Gemini speech-to-text through `stt.provider: gemini`
- **TTS:** Gemini 3.8 speech synthesis through `tts.provider: gemini-tts`
- **Audio tags:** automatic 3.8 angle-bracket tagging, on by default

The plugin registers through Hermes' native `TranscriptionProvider` and `TTSProvider` extension
points. It makes **no Hermes core changes**.

> The repository and plugin are named `gemini-stt` for historical reasons. The plugin now covers
> both directions of the voice pipeline. The TTS provider is `gemini-tts` because Hermes reserves
> the name `gemini` for its built-in TTS implementation.

Validated against Hermes `0.22.2` in both `in_process` and `plugins.isolation: host` modes, with
real API runs for STT, the Files API, TTS unary synthesis, TTS streaming, and the full Hermes
dispatch path.

---

## Features

### Speech-to-text

- **Two model families.** `gemini-3.8-flash` for general multimodal transcription, or
  `gemini-3.5-transcribe` for ASR-tuned language hints, vocabulary biasing, and diarization.
- **Correct 20 MB inline handling.** Base64 inflates audio by 4/3, so the plugin computes a
  raw-byte budget and switches to the Files API above `files_api_threshold_mb`.
- **Files API lifecycle built in.** Resumable upload, `ACTIVE` polling, and deletion in a
  `finally` block so uploads never linger.
- **No double billing.** Connect failures, `429`, and `5xx` are retried; a read timeout on the
  paid POST is not.
- **Privacy-first defaults.** Every STT request sets `store: false`; uploads are deleted
  immediately; keys, prompts, and transcripts never reach the logs.
- **Never raises.** `transcribe()` always returns the Hermes envelope:
  `{"success", "transcript", "provider", "error"?}`.

### Text-to-speech

- **Gemini 3.8 native.** Handles the 3.8 `audio/wav` unary response directly instead of
  double-wrapping it as PCM.
- **Automatic audio tags.** Every eligible TTS call is tagged by default with `<laugh>`, `<sigh>`,
  `<short pause>`, and other 3.8 vocal events. No configuration required.
- **Word-safe tagging.** The tagger may only add tags; the plugin verifies the word sequence is
  unchanged and falls back to the original text on any mismatch or API failure.
- **Fast tagger.** Uses `gemini-3.1-flash-lite` for tagging. Gemini 3.8 Flash text was measured at
  ~21 s for the same rewrite and is too slow for live speech.
- **Streaming.** Implements the sentence-streaming voice path with 24 kHz mono PCM over
  `streamGenerateContent?alt=sse`.
- **Sustained style.** `tts_style` maps to Gemini 3.8 `speech_metadata.style`.
- **Graceful delivery.** Writes WAV directly; uses ffmpeg for MP3/OGG; falls back to WAV when
  ffmpeg is unavailable rather than mislabeling the container.

## Requirements

| Requirement | Version / notes |
|---|---|
| Hermes | `>= 0.22.2` |
| Python | 3.11–3.14 (the range Hermes supports) |
| Gemini API key | `GEMINI_API_KEY` (or `GOOGLE_API_KEY` as fallback) |
| Network | `https://generativelanguage.googleapis.com` by default |
| ffmpeg | Optional. Needed for MP3/OGG output and native voice bubbles |

No additional Python packages are required. `httpx` is already a Hermes dependency.

## Installation

```bash
git clone https://github.com/ajk0x00/hermes-gemini-stt.git ~/.hermes/plugins/gemini-stt
hermes plugins enable gemini-stt
hermes plugins validate ~/.hermes/plugins/gemini-stt
```

Add the key to `~/.hermes/.env`:

```bash
GEMINI_API_KEY=your-key-here
```

Alternatively, open **Settings → Plugins → Gemini STT** in the Hermes Desktop and paste the key
there; the form writes it to `GEMINI_API_KEY` rather than to `config.yaml`.

Restart the gateway so provider discovery reruns:

```bash
hermes gateway restart
```

## Configuration

A complete setup for both directions:

```yaml
stt:
  enabled: true
  provider: gemini
  gemini:
    model: gemini-3.8-flash
    language: en

tts:
  provider: gemini-tts
  model: gemini-3.8-flash-tts
  voice: Kore
  output_format: mp3

plugins:
  enabled:
    - gemini-stt
  entries:
    gemini-stt:
      settings:
        tts_audio_tags: true
        tts_style: "warm and conversational"
```

`tts.model` and `tts.voice` are read by Hermes' plugin dispatch and override the plugin defaults.
The plugin settings below are used when those top-level keys are absent.

### STT settings

| Setting | Default | Purpose |
|---|---|---|
| `timeout` | `180` | Read timeout for one STT request, in seconds (10–900) |
| `max_retries` | `2` | Retries for connect failures, `429`, and `5xx` (0–5) |
| `thinking_level` | `low` | `low`, `medium`, or `high` for general models |
| `base_url` | `""` | Gemini-compatible HTTPS endpoint (or `GEMINI_BASE_URL`) |
| `files_api_threshold_mb` | `12` | Raw audio size above which the Files API is used |
| `transcription_mode` | `verbatim` | `verbatim` or `smart` (transcribe model) |
| `diarize` | `false` | Speaker labels (transcribe model; forces `verbatim`) |
| `language_mode` | `hint` | `hint` sends the language; `auto` detects it |
| `keep_uploaded_files` | `false` | Keep Files API uploads for 48 h instead of deleting |
| `api_key` | `""` | Optional override; takes precedence over both env vars |

> `gemini-3.8-flash` rejects `thinking_level: minimal` with HTTP 400. The plugin only accepts
> `low`, `medium`, and `high`, and defaults to `low` for latency.

### TTS settings

| Setting | Default | Purpose |
|---|---|---|
| `tts_model` | `gemini-3.8-flash-tts` | Default TTS model when `tts.model` is absent |
| `tts_voice` | `Kore` | Default prebuilt voice, custom voice ID, or `voice_...` persona |
| `tts_style` | `""` | Sustained 3.8 style, sent as `speech_metadata.style` |
| `tts_audio_tags` | `true` | Automatically insert 3.8 audio tags |
| `tts_audio_tags_streaming` | `true` | Apply tagging on the streaming voice path too |
| `tts_tag_model` | `gemini-3.1-flash-lite` | Fast model used only for tag insertion |
| `tts_tag_timeout` | `5.0` | Maximum seconds to wait for the tagger (1–30) |
| `tts_timeout` | `120` | Read timeout for one TTS request, in seconds (10–900) |
| `tts_max_retries` | `2` | Retries for TTS connect failures, `429`, and `5xx` (0–5) |
| `tts_ffmpeg_path` | `""` | Optional ffmpeg path; empty auto-detects PATH or the bundled binary |

### Key precedence

1. `plugins.entries.gemini-stt.settings.api_key`
2. `GEMINI_API_KEY`
3. `GOOGLE_API_KEY`

## Usage

### Speech-to-text

Once `stt.provider` is `gemini`, voice messages on connected platforms are transcribed
automatically and handed to your normal Hermes agent. Gemini only converts audio to text; the
configured main model still handles reasoning, tools, memory, and the reply.

```python
from tools.transcription_tools import transcribe_audio

result = transcribe_audio("/path/to/voice.ogg")
# {'success': True, 'transcript': '...', 'provider': 'gemini'}
```

Override the model for one call:

```python
result = transcribe_audio("/path/to/voice.ogg", model="gemini-3.5-transcribe")
```

### Text-to-speech

Once `tts.provider` is `gemini-tts`, the `text_to_speech` tool and voice surfaces use Gemini:

```python
from tools.tts_tool import text_to_speech_tool

result = text_to_speech_tool("The build passed and every test is green.")
# {"success": true, "file_path": "...", "provider": "gemini-tts", ...}
```

The plugin registers the provider under the name `gemini-tts`. Hermes chooses it from
`tts.provider`; callers normally use `text_to_speech_tool` rather than constructing it directly.

Streaming voice surfaces receive raw 24 kHz mono PCM from the same provider through Hermes'
sentence-streaming path.

## How it works

```
voice note
   │
   ▼
Hermes ──► GeminiTranscriptionProvider ──► Gemini Interactions API
   │               │
   │               └──► Files API (audio above the inline budget)
   ▼
transcript ──► Hermes agent (your main model) ──► reply
                                                     │
                                                     ▼
                                      GeminiTTSProvider
                                       │              │
                                       │              └──► Gemini 3.8 TTS
                                       └──► auto tagger (3.1 Flash Lite)
```

| Path | Flow |
|---|---|
| **STT inline** | `POST /v1beta/interactions` with base64 audio in `input[]` |
| **STT Files API** | resumable upload → poll until `ACTIVE` → pass `file.uri` → `DELETE` |
| **TTS unary** | `generateContent` with `responseModalities: ["AUDIO"]` |
| **TTS streaming** | `streamGenerateContent?alt=sse` → base64 L16 PCM chunks |

### 3.8 output handling

Gemini 3.8 changed the unary audio format:

| Model family | Unary response | Plugin handling |
|---|---|---|
| Gemini 3.8 TTS | `audio/wav` with a RIFF header | Written directly; not wrapped again |
| Gemini 3.1 / 2.5 TTS | `audio/l16` PCM | Wrapped in a WAV container |
| All TTS streaming | Headerless `audio/l16` | Yielded as 24 kHz mono PCM |

## Models

### STT

| Model | Best for | Observed latency on a 13 s clip |
|---|---|---|
| `gemini-3.8-flash` | General audio understanding; the plugin default | 20 s to 2+ min, demand-sensitive |
| `gemini-3.5-transcribe` | Voice notes, cleaner ASR, vocabulary and diarization | 4–7 s inline; ~22 s via Files API |

### TTS

| Model | Best for | Notes |
|---|---|---|
| `gemini-3.8-flash-tts` | Flagship creative speech; the plugin default | Angle-bracket tags; WAV unary |
| `gemini-3.8-flash-lite-tts` | Faster, cheaper read-aloud | Same 3.8 tag schema |
| `gemini-3.1-flash-tts-preview` | Legacy expressive voice | Square-bracket tags |
| `gemini-2.5-flash-preview-tts` | Hermes' legacy default | No 3.8 tag support |
| `gemini-2.5-pro-preview-tts` | Legacy pro voice | No 3.8 tag support |

Latency varies with Google-side load. In live checks, 3.8 unary TTS took about 5 s for a short
two-sentence clip; streaming delivered first audio in about 1.3 s before tagging, and about 2.3 s
with automatic tagging enabled.

## Audio tags

Gemini 3.8 changed the tag model completely:

| Model | Tag syntax | Sustained style |
|---|---|---|
| Gemini 3.8 | `<laugh>`, `<sigh>`, `<short pause>` | `speech_metadata.style` |
| Gemini 3.1 | `[laughs]`, `[whispers]` | Inline direction |
| Gemini 2.5 | None | Inline direction |

The plugin's automatic tagger is **on by default** and produces the correct syntax for the selected
model. It uses `gemini-3.1-flash-lite`, enforces one tag per sentence and a density ceiling, and
never changes the words. Manual tags are preserved; on 3.8, `[laughs]` is translated to `<laugh>`.

The tagger is best-effort:

- timeout after `tts_tag_timeout` seconds → speak the original text untagged
- API error or blocked response → speak the original text untagged
- word-sequence mismatch or malformed output → speak the original text untagged
- text shorter than 12 words → no automatic tag
- repeated text is cached in memory

Tagging adds one small text-model request per TTS chunk or streamed sentence. The tagger sees the
same text as the TTS model, so enabling tags sends the reply to Google twice.

## Sustained style

For 3.8 models, set a persistent delivery style:

```yaml
plugins:
  entries:
    gemini-stt:
      settings:
        tts_style: "warm and conversational"
```

This becomes `speech_metadata.style` for every request. It is separate from point-in-time tags such
as `<laugh>`. The built-in `tts.provider: gemini` does not do this for 3.8; use `gemini-tts`.

## Audio delivery

- **WAV** is written directly.
- **MP3/OGG** is converted with ffmpeg.
- The plugin checks `tts_ffmpeg_path`, then `PATH`, then the bundled Hermes binary under
  `~/.hermes/tools/*/bin/ffmpeg`.
- If no ffmpeg is available, the plugin writes a `.wav` and returns that path rather than pretending
  WAV bytes are MP3/OGG.
- OGG output is forced to Opus for native voice bubbles.

Hermes caps plugin TTS input at 4,000 characters by default. To allow longer chunks:

```yaml
tts:
  gemini-tts:
    max_text_length: 16000
```

Gemini 3.8 TTS has an 8,192-token input limit, so extremely long chunks can still be rejected by
the API.

## Privacy and security

- Audio, TTS text, and optional hints are sent to Google (or your `base_url`) with your API key.
  The plugin does not write transcripts or speech text to disk.
- Automatic tagging sends the TTS text to the tagging model before synthesis. Disable
  `tts_audio_tags` if that is not acceptable.
- Every STT request sets `store: false`.
- Files API uploads are deleted immediately unless `keep_uploaded_files` is enabled.
- Logs include model, voice, byte count, tag count, status, and latency — never the text,
  transcript, request body, prompt, or key. Error text is redacted before it reaches an envelope or
  log line.
- `base_url` must use HTTPS unless it points at loopback; cleartext endpoints are refused so a key
  cannot leak over the network.

## Troubleshooting

### STT

| Symptom | Fix |
|---|---|
| `Gemini is not configured` | Add `GEMINI_API_KEY` to `~/.hermes/.env` and restart the gateway |
| `Gemini API error (401)` / `(403)` | Re-issue the key at [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `Gemini API error (404)` | The model is not available to this key; try `gemini-3.8-flash` |
| `Gemini API error (400)` mentioning `custom_vocabulary` | Turn off `diarize`; vocabulary and diarization are mutually exclusive |
| `Gemini API error (400)` mentioning `THINKING_LEVEL_MINIMAL` | Use `thinking_level: low` (the default) |
| `did not respond within 180s` | Raise `timeout`, or switch to `gemini-3.5-transcribe` |
| Empty transcript | The audio may be silent; check gateway logs for the returned status |
| `hermes tools` does not list Gemini | Known Hermes gap: the STT picker renders built-ins only. Set `stt.provider: gemini` manually |
| Works in-process but not with `plugins.isolation: host` | Run `hermes plugins validate` once and restart; the plugin is host-isolation audited and E2E-tested |

### TTS

| Symptom | Fix |
|---|---|
| Audio tags do nothing | Use `tts.provider: gemini-tts`, not the built-in `gemini` |
| `ffmpeg not found` | Install ffmpeg or let the plugin use the bundled Hermes binary; WAV fallback still plays |
| Voice bubble is a rectangular audio file | ffmpeg is missing or could not produce Opus; install it or set `tts_ffmpeg_path` |
| First streaming audio starts later than expected | Auto-tagging adds ~1 s; set `tts_audio_tags_streaming: false` for faster start |
| Tagger warning in logs | The tagger timed out or was unavailable; synthesis continued untagged |
| `Gemini API error (404)` for a TTS model | The model is not available to this key; try `gemini-3.8-flash-tts` |
| `Gemini API error (400)` for a voice | Use a prebuilt voice such as `Kore`, or a valid `voice_...` ID |
| `Gemini API error (429)` / quota exceeded | Wait for the free-tier reset or enable billing; 3.8 TTS has a low free-tier request cap |
| `tts.provider: gemini-tts` is not found | Enable the plugin, validate it, and restart the gateway |

## Development

No network or API key is needed for the standard suite: it uses `unittest` and a local HTTP stub.

```bash
cd ~/.hermes/plugins/gemini-stt
PYTHONPATH=~/.hermes/hermes-agent python -m unittest discover -s tests -t . -v
```

Use the interpreter of the venv Hermes runs under if `python` is not it. Set `HERMES_REPO` if
Hermes lives somewhere other than `~/.hermes/hermes-agent`.

The suite covers STT request builders, retries and timeouts, the Files API lifecycle, TTS unary and
streaming paths, 3.8 WAV vs L16 handling, audio-tag sanitization and fallback, ffmpeg conversion and
fallback, and real-Hermes end-to-end runs for both STT and TTS in-process and under
`plugins.isolation: host`.

### Live tests

Live STT checks call the real API and cost a few cents at most:

```bash
set -a; source ~/.hermes/.env; set +a
HERMES_TEST_LIVE_GEMINI=1 python -m unittest tests.test_live -v
```

Because `gemini-3.8-flash` is slower and demand-sensitive, its STT live test is gated separately:

```bash
HERMES_TEST_LIVE_GEMINI=1 HERMES_TEST_LIVE_GEMINI_SLOW=1 python -m unittest tests.test_live -v
```

Live TTS checks cover unary WAV, streaming PCM, and 3.8 angle-tag generation:

```bash
set -a; source ~/.hermes/.env; set +a
HERMES_TEST_LIVE_GEMINI_TTS=1 python -m unittest tests.test_live.LiveTTSTests -v
```

Set `HERMES_NEUTTS_SAMPLE` to a speech WAV if the bundled Hermes sample is unavailable.

### Validation

```bash
hermes plugins validate ~/.hermes/plugins/gemini-stt
```

This checks the manifest, config schema, capability probe, security scan, core-override policy, and
host-isolation audit.

## Contributing

Issues and pull requests are welcome. Keep the plugin free of Hermes core modifications, add a
regression test for behavior changes, and run the full suite plus `hermes plugins validate` before
opening a PR.

## License

MIT — see [LICENSE](LICENSE).
