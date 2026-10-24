# Hermes Gemini STT

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Hermes](https://img.shields.io/badge/Hermes-%E2%89%A50.22.2-6f42c1.svg)](https://github.com/NousResearch/hermes-agent)

A production-grade speech-to-text provider that plugs Google Gemini into
[Hermes](https://github.com/NousResearch/hermes-agent). It registers through the native
`TranscriptionProvider` extension point — **no Hermes core changes** — and works for gateway voice
messages, voice mode, and any caller of `transcribe_audio()`.

- **Default model:** `gemini-3.8-flash` — general multimodal, prompt-driven verbatim transcription
- **ASR-tuned model:** `gemini-3.5-transcribe` — language hints, custom vocabulary, diarization
- **Transport:** Gemini Interactions API over HTTPS, using `httpx` (already a Hermes dependency)

Validated against Hermes `0.22.2` in both `in_process` and `plugins.isolation: host` modes, with
real API runs for inline audio, the Files API, and the full Hermes dispatch path.

---

## Features

- **Two model families, one provider.** Pick the general model for broader audio understanding or
  the transcribe model for cleaner, faster ASR.
- **Handles the 20 MB inline limit correctly.** Base64 inflates audio by 4/3, so the plugin
  computes a raw-byte budget and switches to the Files API above `files_api_threshold_mb`.
- **Files API lifecycle built in.** Resumable upload, `ACTIVE` polling, and deletion in a
  `finally` block so uploads never linger.
- **Retries that do not double-bill.** Connect failures, `429`, and `5xx` are retried with
  exponential backoff and `Retry-After` support; a read timeout on the paid POST is *not* retried.
- **Privacy-first defaults.** Every request sets `store: false`; uploads are deleted immediately;
  keys, prompts, and transcripts never reach the logs.
- **Never raises.** `transcribe()` always returns the Hermes envelope:
  `{"success", "transcript", "provider", "error"?}`.
- **Bounded, tested code.** Standard library plus `httpx`; 99 unit/E2E tests, including real
  Hermes plugin discovery and dispatch.

## Requirements

| Requirement | Version / notes |
|---|---|
| Hermes | `>= 0.22.2` |
| Python | 3.11–3.14 (the range Hermes supports) |
| Gemini API key | `GEMINI_API_KEY` (or `GOOGLE_API_KEY` as fallback) |
| Network | `https://generativelanguage.googleapis.com` by default |

No additional Python packages are required.

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

Minimal `~/.hermes/config.yaml`:

```yaml
stt:
  enabled: true
  provider: gemini
  gemini:
    model: gemini-3.8-flash
    language: en
```

For interactive voice notes, `gemini-3.5-transcribe` is usually the better trade-off:

```yaml
stt:
  provider: gemini
  gemini:
    model: gemini-3.5-transcribe
    language: en
```

Model and language are resolved by the Hermes dispatcher from `stt.gemini.*` and passed to the
provider per call. Everything else lives under `plugins.entries.gemini-stt.settings` and is
rendered by the Desktop settings form.

### Plugin settings

| Setting | Default | Purpose |
|---|---|---|
| `timeout` | `180` | Read timeout for one Gemini request, in seconds (10–900) |
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

### Key precedence

1. `plugins.entries.gemini-stt.settings.api_key`
2. `GEMINI_API_KEY`
3. `GOOGLE_API_KEY`

## Usage

Once `stt.provider` is `gemini`, voice messages on connected platforms are transcribed
automatically and handed to your normal Hermes agent. Gemini only converts audio to text; the
configured main model still handles reasoning, tools, memory, and the reply.

You can also call the provider directly:

```python
from tools.transcription_tools import transcribe_audio

result = transcribe_audio("/path/to/voice.ogg")
# {'success': True, 'transcript': '...', 'provider': 'gemini'}
```

Override the model for one call:

```python
result = transcribe_audio("/path/to/voice.ogg", model="gemini-3.5-transcribe")
```

The return envelope is stable:

```python
{"success": True, "transcript": "...", "provider": "gemini"}
```

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
```

| Path | Flow |
|---|---|
| **Inline** (default) | `POST /v1beta/interactions` with base64 audio in `input[]` |
| **Files API** | resumable upload → poll until `ACTIVE` → pass `file.uri` → `DELETE` the file |

Both paths set `store: false`. The request shape depends on the model:

- `gemini-3.5-transcribe` — audio input plus `generation_config.transcription_config`
  (`language_codes`, `mode`, optional `custom_vocabulary`).
- `gemini-3.8-flash` — a strict verbatim `system_instruction`, an optional reference-only hint
  block, and `thinking_level`.

Responses are parsed tolerantly from `output_text`, `outputs[]`, or `steps[].content[]`, so an API
surface change cannot silently produce an empty transcript.

## Models

| Model | Best for | Latency observed on a 13 s clip |
|---|---|---|
| `gemini-3.8-flash` | General audio understanding; the plugin's default | 20 s to 2+ min, demand-sensitive |
| `gemini-3.5-transcribe` | Voice notes, cleaner ASR, vocabulary and diarization | 4–7 s inline; ~22 s via Files API |

Latency varies with Google-side load; the numbers above are observations, not guarantees.

### Diarization and vocabulary

`diarize: true` uses the API's object-form verbatim mode and forces `verbatim` even if
`transcription_mode: smart` is set (smart mode rejects diarization). Diarization and
`custom_vocabulary` are mutually exclusive in the Gemini API, so the plugin drops the vocabulary
with a warning when diarization is enabled.

### Multilingual audio

Hermes seeds `stt.language: "en"` and passes it to every provider. To let Gemini detect the
language instead, set:

```yaml
plugins:
  entries:
    gemini-stt:
      settings:
        language_mode: auto
```

### Vocabulary hints

`stt.prompt` (and any `pre_transcription` hook result) is forwarded to the provider. With
`gemini-3.5-transcribe` it becomes `custom_vocabulary` (deduped, capped at 100 terms); with general
models it is wrapped as a reference block the model is told never to treat as instructions.

## Audio handling

The inline limit is **20 MB of total request size**; base64 adds 4/3 overhead. The plugin converts
that limit into a raw-byte budget, so `files_api_threshold_mb: 12` means 12 MB of actual audio.
Thresholds above roughly 15 MB are capped by the request limit.

- **Files API:** resumable upload → `ACTIVE` → `file.uri` → delete in `finally`. Uploads otherwise
  persist for 48 hours.
- **Hermes limits:** Hermes caps a single upload at 25 MB and splits longer recordings before
  calling the provider.
- **Formats:** `.mp3`, `.m4a`, `.mp4`, `.mpeg`, `.mpga`, `.wav`, `.webm`, `.ogg`, `.oga`, `.opus`,
  `.aac`, `.flac`. `.mp4` is sent as `audio/m4a` (same ISO-BMFF container); when `ffprobe` is
  available and finds a real video stream, the plugin refuses early with an actionable error.

## Privacy and security

- Audio and the optional hints are sent to Google (or your `base_url`) with your API key.
  Transcripts are returned in memory and never written to disk by this plugin.
- `stt.prompt`, `stt.language`, and `pre_transcription` hints are part of the request. Do not put
  secrets in a prompt.
- Every request sets `store: false`.
- Files API uploads are deleted immediately unless `keep_uploaded_files` is enabled.
- Logs include model, byte count, transport, status, and latency — never the transcript, request
  body, prompt, or key. Error text is redacted before it reaches an envelope or log line.
- `base_url` must use HTTPS unless it points at loopback; cleartext endpoints are refused so a key
  cannot leak over the network.

## Troubleshooting

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

## Development

No network or API key is needed for the standard suite: it uses `unittest` and a local HTTP stub.

```bash
cd ~/.hermes/plugins/gemini-stt
PYTHONPATH=~/.hermes/hermes-agent python -m unittest discover -s tests -t . -v
```

Use the interpreter of the venv Hermes runs under if `python` is not it. Set `HERMES_REPO` if
Hermes lives somewhere other than `~/.hermes/hermes-agent`.

The suite covers request builders, retries and timeouts, the Files API lifecycle, envelope errors,
and two real-Hermes end-to-end runs: one in-process and one under `plugins.isolation: host`.

### Live tests

An opt-in live check calls the real API and costs a few cents at most:

```bash
set -a; source ~/.hermes/.env; set +a
HERMES_TEST_LIVE_GEMINI=1 python -m unittest tests.test_live -v
```

Because `gemini-3.8-flash` is slower and demand-sensitive, its live test is gated separately:

```bash
HERMES_TEST_LIVE_GEMINI=1 HERMES_TEST_LIVE_GEMINI_SLOW=1 python -m unittest tests.test_live -v
```

Set `HERMES_NEUTTS_SAMPLE` to a speech WAV if the bundled Hermes sample is unavailable.

### Validation

```bash
hermes plugins validate ~/.hermes/plugins/gemini-stt
```

This checks the manifest, config schema, capability probe, security scan, core-override policy, and
host-isolation audit.

## Contributing

Issues and pull requests are welcome. Keep the provider free of Hermes core modifications, add a
regression test for behavior changes, and run the full suite plus `hermes plugins validate` before
opening a PR.

## License

MIT — see [LICENSE](LICENSE).
