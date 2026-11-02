# ASR Dashboard

[中文](README.md)

A desktop speech-recognition app (Python + PyQt6 Fluent UI). Drop in audio/video files or a whole folder and get transcripts with timestamps and speaker names, exported as TXT / SRT / VTT / JSON / ASS / Markdown.

- **Local recognition**: only **Qwen3-ASR-1.7B** and **Qwen3-ASR-0.6B**, plus **Qwen3-ForcedAligner-0.6B** for word timestamps. 30 languages + 22 Chinese dialects (auto-detected). Every parameter is adjustable in the UI.
- **Context prompts (hot words)**: global default context + named glossaries + per-task context, merged and de-duplicated.
- **Speaker diarization**: pyannote Community-1 (default), DiariZen (isolated environment), WavLM fallback. Works for local and API engines.
- **Voiceprint library**: enrol voices with names; `SPEAKER_00` is replaced by the real name automatically. Voiceprints never leave the machine.
- **Batch processing**: strictly sequential, resumable after a crash, mirrors sub-folders, writes reports.
- **API engines**: OpenAI-compatible transcriptions, OpenAI-compatible chat audio (remote Qwen3-ASR), Alibaba Cloud Model Studio (DashScope), custom HTTP.
- **Smart post-processing**: configure an LLM API (validated on save) for transcript clean-up; rule-based processing otherwise.
- **Realtime transcription**: connect to a self-hosted vLLM or DashScope streaming service; refines the recording afterwards.

## Installation

1. Install **uv**: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` (Windows) or `curl -LsSf https://astral.sh/uv/install.sh | sh`.
2. Install **FFmpeg** (required): `winget install Gyan.FFmpeg` / `sudo apt install ffmpeg`. The app checks it at start-up.
3. Install dependencies with exactly one PyTorch flavour (Python 3.11–3.13):
   ```bash
   uv sync --extra cu128   # NVIDIA (CUDA 12.8)
   uv sync --extra rocm    # AMD ROCm 6.4 (Linux; shows up as "cuda" in PyTorch)
   uv sync --extra cpu     # CPU only
   ```
   Optional: `--extra flash-attn`, `--extra vllm` (Linux), `--extra dashscope`.

   **qwen-asr and native Transformers cannot share an environment**: `qwen-asr` pins `transformers==4.57.6`, while native Qwen3-ASR support (the `-hf` checkpoints) starts at transformers **5.13**. They are mutually exclusive dependency groups: default group `qwen` (recommended) or `uv sync --extra cu128 --no-group qwen --group hf`. Unavailable backends are greyed out in Settings with the install command.
4. Run: `uv run asr-dashboard` (or `uv run python main.py`; on Windows double-click `ASR-Dashboard.bat`).

`uv.lock` must be generated once with `uv lock` on a machine that can reach `download.pytorch.org`.

## Models

- **Qwen3-ASR**: Settings → Local model. Source: HuggingFace / ModelScope / local folder; cache dir sets `HF_HOME` / `MODELSCOPE_CACHE`. Buttons: download (with progress), pre-warm (the log prints the final `from_pretrained` kwargs), cache listing / deletion.
- **pyannote Community-1** (gated, CC-BY-4.0): create a HF token, accept the terms on <https://huggingface.co/pyannote/speaker-diarization-community-1>, click *Download*. Afterwards it loads offline from the local folder with no token.
- **DiariZen** (weights **CC BY-NC 4.0, non-commercial only**): it ships a modified pyannote-audio and pins torch 2.1.1 / Python 3.10, so the app installs it into `~/.asr-dashboard/envs/diarizen/` with uv + git and runs it as a subprocess (JSON over stdin/stdout). Accept the licence checkbox first.

## Usage

Single file → pick/drop a file, choose an engine (grouped *Local* / *API*, ⚙ opens its settings), language, glossaries and task context, options, start. Results page: table / plain text / diff views, inline editing, speaker rename, *save to voiceprint library* (right-click), click-to-play, exports, warnings and timings. History lives in `~/.asr-dashboard/history.json`.

Timestamps: audio is always chunked first (hybrid: target 60 s, cut at the longest silence within ±10 s). With forced alignment, word timestamps are re-split into sentences (final punctuation / pause > 0.6 s / > 15 s); chunks are capped at 180 s (qwen-asr's own aligner input limit). Chunks whose language the aligner does not support fall back to chunk boundaries with a warning. Context is a soft bias passed as the system prompt (`context` in qwen-asr, `prompt` in Transformers).

## Voiceprint library

Add people, then samples from a file (optional time range), from the microphone (level meter) or from a result (longest clean, non-overlapping stretches, ~30 s). After diarization each speaker's mean embedding is compared with every person (cosine), assigned one-to-one with the Hungarian algorithm, and named if ≥ threshold (default 0.6; *calibrate* in Settings). All vectors must come from the same model (pyannote Community-1 embedding by default, or WavLM); recompute after switching. Import/export as zip.

## API models

Create from a preset or a provider type, fill URL / key / model, **Test connection** (real call with a built-in 3 s clip; shows latency, raw response and parsed result), save. Providers: `openai_transcriptions` (verbose_json → json fallback, `prompt` for context, Azure preset), `openai_chat_audio` (`audio_url` data URL, parses `language X<asr_text>…`), `dashscope_qwen` (temporary OSS upload, async submit + poll), `custom_http` (templates + JSONPath + optional polling). Shared behaviour: automatic splitting/compression above size/duration limits, exponential back-off on 429/5xx/timeouts honouring `Retry-After`, bounded concurrency, instant cancel, key redaction.

## Batch & resume

Choose a folder, scan options and sort (natural: `2.mp3` before `10.mp3`), reorder by drag-and-drop, start. One worker, models loaded once. Failed files are recorded and skipped; OOM retries once with half the batch size. Pause (after the current file) / resume / cancel / retry failed. `batch_manifest.json` is rewritten atomically after every file; at start-up the app offers to continue unfinished batches. Completed = status done + unchanged size/mtime + outputs exist. Reports: `batch_report.csv` / `.json`. Windows sleep is inhibited during a batch.

## Smart post-processing (LLM)

Settings → Smart post-processing: base URL, key, model → **Save & validate** (tiny `{"ok": true}` request, 20 s; distinguishes network / auth / unknown model / unparsable; detects `response_format=json_object`). Examples: DeepSeek `https://api.deepseek.com/v1` + `deepseek-chat`; DashScope `https://dashscope.aliyuncs.com/compatible-mode/v1` + `qwen-plus`; Ollama `http://localhost:11434/v1` + `qwen3:8b`; LM Studio `http://localhost:1234/v1`. Guard rails: ≤ 1500-char chunks with two read-only previous items, JSON in/out, only `text` may change, ids must match, ≤ 40 % edit ratio per item, otherwise that chunk falls back to rules. Original text is kept for diff/undo. Optional summary and speaker-identity suggestions.

## Realtime service deployment

```bash
uv venv && source .venv/bin/activate && uv pip install "vllm[audio]>=0.32"
./server/start_realtime.sh 1.7b 8000 0.8    # vllm serve ... --hf-overrides '{"architectures": ["Qwen3ASRRealtimeGeneration"]}'
cd server && MODEL=Qwen/Qwen3-ASR-0.6B docker compose up -d
```
vLLM only exposes `/v1/realtime` for models implementing its realtime interface; for Qwen3-ASR that is `Qwen3ASRRealtimeGeneration`, hence the override. The endpoint accepts only the model name (no prompt / language), so glossaries apply only to the refinement pass. Windows: run the server inside WSL2 (`wsl --install -d Ubuntu`, NVIDIA Windows driver, `ws://localhost:8000/v1/realtime`; open the firewall / `netsh portproxy` for LAN access). Sharing one GPU with the desktop app: lower `--gpu-memory-utilization` and prefer 0.6B for streaming. DashScope: `wss://dashscope.aliyuncs.com/api-ws/v1/realtime`, model `qwen3-asr-flash-realtime`.

## Config & secrets

`config.yaml` is validated with pydantic; invalid or missing values fall back to defaults (see `config.example.yaml`). API keys and the HF token are stored in the system keyring, or in `~/.asr-dashboard/secrets.yaml` (plaintext, the UI warns) when no keyring is available. Logs are redacted.

## Packaging

`uv pip install pyinstaller && uv run python build_exe.py`. The bundle supports CPU or CUDA only and does not include the DiariZen environment, vLLM or model weights.

## FAQ

Out of memory → smaller batch size, 0.6B, aligner on CPU, enable CPU fallback. FlashAttention → requires `flash-attn` and fp16/bf16. vLLM on Windows → not supported; use WSL2. DiariZen licence → non-commercial only; use pyannote Community-1 for commercial work.

## Development

```bash
uv sync --extra cpu && uv run ruff check . && uv run pytest
```
