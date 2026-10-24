# Gemini STT for Hermes

A Hermes speech-to-text provider backed by Google Gemini. It plugs into the native
`TranscriptionProvider` extension point — no Hermes core changes — and works for gateway voice
messages, voice mode, and any caller of `transcribe_audio()`.

- **Default model:** `gemini-3.8-flash` (general multimodal, prompt-driven verbatim transcription)
- **Alternative:** `gemini-3.5-transcribe` (ASR-tuned: language hints, vocabulary biasing, diarization)
- **Transport:** Gemini Interactions API over HTTPS; no `google-genai` dependency

## Install

```bash
git clone <repo> ~/.hermes/plugins/gemini-stt
hermes plugins enable gemini-stt
hermes plugins validate ~/.hermes/plugins/gemini-stt
```

Add the key to `~/.hermes/.env` (or run `hermes auth`):

```bash
GEMINI_API_KEY=...
```

`GOOGLE_API_KEY` is accepted as a fallback. Then restart the gateway so plugin discovery reruns:

```bash
hermes gateway restart
```

## Configure

```yaml
stt:
  enabled: true
  provider: gemini
  gemini:
    model: gemini-3.8-flash
    language: en            # BCP-47 hint; "" is not an opt-out (stt.language fills it)
```

Provider settings live under `plugins.entries.gemini-stt.settings` and are exposed in the
Desktop **Settings → Plugins** form:

| Setting | Default | Purpose |
|---|---|---|
| `timeout` | `180` | Read timeout for one Gemini request (10–900s) |
| `max_retries` | `2` | Retries for connect failures, 429 and 5xx (0–5) |
| `thinking_level` | `low` | Latency control for general models (`low`, `medium`, `high`; 3.8 Flash rejects `minimal`) |
| `base_url` | `""` | Gemini-compatible HTTPS endpoint (or `GEMINI_BASE_URL` in `.env`) |
| `files_api_threshold_mb` | `12` | Raw size above which the Files API is used |
| `transcription_mode` | `verbatim` | `verbatim` or `smart` (3.5-transcribe) |
| `diarize` | `false` | Speaker labels (3.5-transcribe; forces `verbatim`) |
| `language_mode` | `hint` | `hint` sends the language; `auto` detects it |
| `keep_uploaded_files` | `false` | Keep Files API uploads for 48h |
| `api_key` | `""` | Optional override; the Desktop form writes it to `GEMINI_API_KEY` |

The Desktop settings form stores `api_key` through the `.env` credential route, so it lands in
`GEMINI_API_KEY` rather than in `config.yaml`. Writing `plugins.entries.gemini-stt.settings.api_key`
by hand still works and takes precedence over both environment variables.

### Picking the model

`gemini-3.8-flash` is a general multimodal model: the plugin sends a strict verbatim system
instruction, `thinking_level`, and strips a whole-output fence or a lone `Transcript:` label.
`gemini-3.5-transcribe` is purpose-built for ASR and produces cleaner transcripts with less prompt
sensitivity. Switch with `stt.gemini.model: gemini-3.5-transcribe`.

#### Latency

Observed on a 13-second speech clip, from the same machine and key:

| Model | Time | Notes |
|---|---|---|
| `gemini-3.5-transcribe` | 4–7 s inline, ~22 s with a Files API upload | Stable; intended for voice notes |
| `gemini-3.8-flash`, `thinking_level: low` | 20 s to 2+ min | Highly variable with Google-side demand |

The default model is `gemini-3.8-flash` because that is what this plugin was asked to expose. For
interactive voice messages, `gemini-3.5-transcribe` is the better latency/quality trade-off; keep
the general model if you want its broader audio understanding or already have access to it.

`diarize: true` uses the API's object-form verbatim mode and therefore forces `verbatim` even if
`transcription_mode: smart` is set (smart mode rejects diarization). Diarization and
`custom_vocabulary` are also mutually exclusive in the Gemini API, so the vocabulary is dropped
with a warning when diarization is enabled.

### Multilingual audio

Hermes seeds `stt.language: "en"`, which the dispatcher passes to every provider. If you speak
other languages and do not want the hint applied, set `language_mode: auto` in the plugin
settings — the provider then sends `language_codes: []` and lets Gemini detect the language.

### Vocabulary hints

`stt.prompt` (and any `pre_transcription` hook result) is forwarded to the provider. For
`gemini-3.5-transcribe` it becomes `custom_vocabulary` (deduped, max 100 terms). For general
models it is wrapped as a reference block that the model is told never to treat as instructions.
Vocabulary and diarization are mutually exclusive in the Gemini API; the plugin drops the
vocabulary with a warning when `diarize: true`.

## Size and format handling

The inline limit is **20 MB of total request size**, and base64 inflates raw audio by 4/3. The
plugin converts that request limit into a raw-byte budget and compares the file size directly;
the default `files_api_threshold_mb: 12` is therefore 12 MB of actual audio, not 12 MB of base64.
Thresholds above roughly 15 MB are capped by the 20 MB request limit.

- Files API: resumable upload → wait for `ACTIVE` → reference `file.uri` → delete the upload in a
  `finally` block. Uploads otherwise live for 48 hours.
- Hermes caps a single upload at 25 MB and splits longer recordings before calling the provider,
  so one plugin call never exceeds 25 MB.
- Supported suffixes: `.mp3`, `.m4a`, `.mp4`, `.mpeg`, `.mpga`, `.wav`, `.webm`, `.ogg`, `.oga`,
  `.opus`, `.aac`, `.flac`. `.mp4` is sent as `audio/m4a` (same ISO-BMFF container); if `ffprobe`
  is available and finds a video stream, the plugin refuses early with an actionable error.

## Privacy

- Audio is sent to `generativelanguage.googleapis.com` (or your configured `base_url`) with your
  API key. Transcripts are returned in memory and never written to disk by this plugin.
- `stt.prompt`, `stt.language` and any `pre_transcription` hint are also sent to Google as part of
  the request (or, for `gemini-3.5-transcribe`, as `custom_vocabulary` / `language_codes`). Do not
  put secrets in a prompt.
- Every request sets `store: false`, so Google does not retain the interaction.
- Files API uploads are deleted immediately after transcription unless `keep_uploaded_files` is on.
- Logs contain model, byte count, transport, status and latency — never the transcript, the
  request body, the prompt, or the key.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Gemini is not configured` | Add `GEMINI_API_KEY` to `~/.hermes/.env` and restart the gateway |
| `Gemini API error (401)` or `(403)` | Re-issue the key at aistudio.google.com |
| `Gemini API error (404)` | The model is not available to this key; try `gemini-3.8-flash` |
| `Gemini API error (400)` and `custom_vocabulary` | Turn off `diarize` (vocabulary and diarization are mutually exclusive) |
| `did not respond within 180s` | `thinking_level: low` is already the default; raise `timeout`, or use `gemini-3.5-transcribe` for much lower latency |
| Voice note returns an empty transcript | The audio may be silent; check gateway logs for the returned status |
| `hermes tools` does not list Gemini | Known Hermes gap: the STT picker only renders built-ins. Set `stt.provider: gemini` manually. |
| Works in-process but not with `plugins.isolation: host` | Install/validate at least once and restart; the plugin is host-isolation audited and covered by the E2E suite |

## Development

The plugin uses only the Python standard library plus `httpx` (already a Hermes dependency).
Tests use `unittest` and a local HTTP stub; no network or API key is required.

```bash
cd ~/.hermes/plugins/gemini-stt
PYTHONPATH=~/.hermes/hermes-agent python -m unittest discover -s tests -t . -v
```

Use the interpreter of the venv Hermes runs under if `python` is not it (a source install keeps
one under `~/.hermes/installs/.../venv/bin/python`). Set `HERMES_REPO` if Hermes lives somewhere
other than `~/.hermes/hermes-agent`.

The suite covers the request builders, retries/timeouts, Files API lifecycle, envelope errors, and
two real-Hermes end-to-end runs: one in-process and one under `plugins.isolation: host`.

An opt-in live check calls the real API and costs a few cents at most:

```bash
set -a; source ~/.hermes/.env; set +a
HERMES_TEST_LIVE_GEMINI=1 python -m unittest tests.test_live -v
```

`gemini-3.8-flash` is slower and demand-sensitive, so its live test is gated separately:

```bash
HERMES_TEST_LIVE_GEMINI=1 HERMES_TEST_LIVE_GEMINI_SLOW=1 python -m unittest tests.test_live -v
```

Set `HERMES_NEUTTS_SAMPLE` to a speech WAV if the bundled Hermes sample is unavailable.

## License

MIT
