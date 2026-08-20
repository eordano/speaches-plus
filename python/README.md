# speaches-plus-python

The multimedia nano-vLLM. One process, one OpenAI-compatible HTTP surface, multimodal in (text + audio + image + video) -> text + audio out.

Bundles five planes:

- **Qwen3-TTS** -- small text-to-speech with three modes (preset voice, voice design, voice cloning).
- **Qwen3-Omni** -- multimodal chat (any-to-any), transcription, translation. With talker -> speaks back.
- **Qwen3-ForcedAligner** -- word-level timestamps, drives SRT / VTT subtitles.
- **Gemma 4** -- alternative chat path (image + audio + video in, text out).
- **Kokoro TTS** -- 82M ONNX TTS, runs on CPU at real-time, 55 voices across 8 languages.

Plus an embedded **nano-vllm** engine (CUDA only) for vLLM-class throughput on the autoregressive paths.

## Quick start

One-time, after clone -- pin every model into `nix/models.nix` and pre-populate the offline HF cache:

```bash
bash scripts/fetch-models.sh
```

Then:

```bash
nix develop --command bash -c 'PORT=18329 ./test_e2e.sh'
```

This boots the server three times against three Qwen3-TTS variants and asserts each produces a valid WAV.

For interactive use:

```bash
nix run                                    # starts the default CPU wrapper on :8091
QWEN3_OMNI_MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct nix run
GEMMA_MODEL=google/gemma-4-E4B-it nix run
KOKORO_ENABLE=1 nix run
```

`nix develop` exports `HF_HUB_CACHE`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` pointing at a hub cache assembled from every model the source code references (speaches-plus-python's planes plus speaches-plus's audio models); the runtime never touches the network. Full list and `nix-hug-lib.buildCache` wiring: [IMPLEMENTATION.md](IMPLEMENTATION.md#bundled-huggingface-assets).

## OpenAI-compatible endpoints

| Endpoint | Routing |
|---|---|
| `POST /v1/chat/completions` | Omni (default), Gemma when `model: "gemma-..."` |
| `POST /v1/audio/transcriptions` | Omni / Gemma; `response_format=srt|vtt` adds the aligner |
| `POST /v1/audio/translations` | Omni / Gemma |
| `POST /v1/audio/speech` | Qwen3-TTS (default), Kokoro when `task_type="Kokoro"` or `model: "kokoro-..."` |
| `GET /v1/audio/voices` | Lists voices across both TTS engines |
| `POST /v1/voice-profiles` | Cache a Qwen3-TTS Base voice prompt |
| `GET /health` | Per-plane status |

Multimodal chat parts accepted on the wire: `text`, `input_audio` / `audio`, `image_url` / `image`, `video_url` / `video`. Each accepts a `data:` URL, an `http(s)://` URL, a `file://` URL, a path, or bare base64. See [IMPLEMENTATION.md](IMPLEMENTATION.md#audio--multimodal-input).

## Architecture

[IMPLEMENTATION.md](IMPLEMENTATION.md) is the single source of architectural truth -- planes, routing, env vars, the engine port contract, and the tested-with version table.

For provenance and per-vendor fork notes see [NOTICE.md](NOTICE.md).

## Repo layout

```
tts/qwen3/     Vendored fork of Qwen3-TTS (Alibaba)
tts/kokoro/    In-tree rewrite of kokoro-onnx
omni/qwen3/    Façade over transformers' Qwen3-Omni-Moe
omni/gemma/    Façade over transformers' Gemma 4
aligner/       Vendored fork of Qwen3-ForcedAligner (Alibaba)
audio/         Multimodal input loaders (audio + image + video)
nano_vllm/     Vendored fork of GeeeekExplorer/nano-vllm (engine, CUDA-only)
server.py      The FastAPI app
flake.nix      Build + bundled HF assets via nix-hug
```

## License

Apache 2.0. Bundled upstream code retains its original licenses; see [NOTICE.md](NOTICE.md).
