# NOTICE

This project is licensed under Apache 2.0 (see `LICENSE`). It bundles code from upstream projects that retain their original licenses.

## Vendored

### `tts/qwen3/` -- Qwen3-TTS

| | |
|---|---|
| Upstream | https://github.com/QwenLM/Qwen3-TTS |
| Commit | `022e286b98fbec7e1e916cb940cdf532cd9f488e` |
| License | Apache 2.0 |
| Forked | 2026-05-09 |

**Kept**: `tts/qwen3/inference/{qwen3_tts_model,qwen3_tts_tokenizer,streaming}.py`, `tts/qwen3/core/models/*`, `tts/qwen3/core/tokenizer_12hz/*`.

**Removed**: `tts/qwen3/cli/`, `tts/qwen3/__main__.py` (gradio demo with analytics-on-by-default), `tts/qwen3/core/tokenizer_25hz/` (V1 tokenizer; we serve only 12 Hz), `assets/`, `examples/`, `finetuning/`, the `qwen-tts-demo` script entry point, and the in-package `pyproject.toml` / `MANIFEST.in`.

**Patches** (transformers 4.x -> 5.x compat):

- `core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` -- `check_model_inputs` shim accepting both 4.x's factory convention and 5.x's direct-decorator convention.
- `core/models/modeling_qwen3_tts.py` -- `pad_token_id` access via `getattr` fallback (5.x configs no longer auto-provide it).
- `core/models/modeling_qwen3_tts.py` -- RoPE `'default'` init shim registered (5.x dropped the registry entry).
- `inference/qwen3_tts_model.py` -- `fix_mistral_regex=True` removed from the `AutoProcessor.from_pretrained` call site (5.x rejects it as a duplicate kwarg).

### `aligner/` -- Qwen3-ForcedAligner

| | |
|---|---|
| Upstream | https://github.com/QwenLM/Qwen3-ASR (`qwen_asr/inference/qwen3_forced_aligner.py` + `qwen_asr/core/transformers_backend/`) |
| Commit | `c17a131fe028b2e428b6e80a33d30bb4fa57b8df` |
| License | Apache 2.0 |
| Forked | 2026-05-09 |

**Kept**: aligner wrapper, audio normalization helpers, model class (`Qwen3ASRForConditionalGeneration` + config + processor).

**Removed**:

- The full `Qwen3ASRModel` autoregressive path -- we only call `model.thinker(...)` for one forward pass.
- `nagisa` (Japanese tokenizer), `soynlp` + `assets/korean_dict_jieba.dict` (Korean tokenizer + 276 KB Jieba dict). `encode_timestamp` raises `ValueError` for `language="japanese"` or `"korean"`. The other 9 of 11 supported languages tokenize via whitespace + CJK char splitting.
- The vllm backend.
- Three duplicate `Qwen3ASRThinkerText{MLP,RMSNorm,Attention}` classes -- defined upstream but never instantiated; the actually-used decoder layer references the `Qwen3ASRText*` (non-Thinker) variants.
- `Qwen3ASRThinkerTextPreTrainedModel` -- defined but never inherited.
- `Qwen3ASRForConditionalGeneration.generate()` + `Qwen3ASRThinkerForConditionalGeneration.prepare_inputs_for_generation()` + `GenerationMixin` inheritance -- we only call `thinker(...).logits` directly.
- `loss = self.loss_function(...)` branch in forward (`labels` is always `None`).
- All `@auto_docstring` decorators + the import (validate docstrings at class-define time, which we removed).
- All upstream docstrings.
- Per-pipeline helpers in `inference/utils.py`: `parse_asr_output`, `detect_and_fix_repetitions`, `merge_languages`, `split_audio_into_chunks`, `AudioChunk`, `chunk_list`, `normalize_language_name`, `validate_language`, `SUPPORTED_LANGUAGES`. Kept only `ensure_list`, `normalize_audios` and its transitive helpers.
- `Qwen3ASRProcessor.get_chunked_index` (full-pipeline helper) and `apply_chat_template` override (no chat path on the aligner).
- `Qwen3ForceAlignProcessor.tokenize_chinese_mixed` (defined but never called) and `Qwen3ForcedAligner.get_supported_languages` (server hardcodes `ALIGNER_LANGUAGES`).

**Patches** (upstream `qwen_asr` -> local `aligner` rename + transformers 4.x -> 5.x compat):

- `@check_model_inputs()` (factory) -> `@check_model_inputs` (direct decorator).
- `Qwen3ASRConfig.__init__` sets `self.thinker_config` *before* calling `super().__init__()` (5.x's `validate_token_ids` reads it during super init).
- RoPE `'default'` init shim registered.
- `pad_token_id` access via `getattr(..., None)` fallback.
- `Qwen3ASRThinkerTextRotaryEmbedding.compute_default_rope_parameters` added as alias of the `'default'` shim (5.x's `_init_weights` calls this method directly when `rope_type == "default"`).

Total: 2992 -> 1816 lines (-39%).

### `nano_vllm/` -- nano-vllm engine

| | |
|---|---|
| Upstream | https://github.com/GeeeekExplorer/nano-vllm |
| Commit | `bb823b3` |
| License | MIT |
| Forked | 2026-05-10 |

**Kept**: everything functional -- engine (`engine/`), all custom layers (`layers/`), the loader, the Qwen3 model, samplers.

**Patches**:

- `nanovllm` -> `nano_vllm` package rename (so multiple nano-vllm-derived projects can coexist on `sys.path`). Same rename applied to the `SharedMemory(name=...)` IPC ring.
- `nano_vllm/layers/attention.py`: wrapped `import flash_attn` + `import triton` in `try/except ImportError` and gated the `@triton.jit store_kvcache_kernel` definition behind the same flag. `Attention.forward` raises a clear error pointing at the `[gpu]` extra when called without CUDA. Lets the package stay importable on Mac.

## Façades (no fork)

### `omni/qwen3/`

Thin wrapper over stock `transformers.Qwen3OmniMoeForConditionalGeneration` + `Qwen3OmniMoeProcessor`. Reuses `audio` for input loading. Includes an MPS shim (`_mps_compat.py`) that patches `torch.histc` for integer dtypes -- required for transformers' MoE expert-counting on Apple Silicon.

Reference: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct

### `omni/gemma/`

Thin wrapper over stock `transformers.Gemma4ForConditionalGeneration` + `Gemma4Processor`. Reuses `audio`. Auto-picks `flash_attention_2` (CUDA + flash-attn installed) or `sdpa` (everywhere else). `GEMMA_COMPILE=1` opts into `torch.compile`.

References:
- https://ai.google.dev/gemma/docs/core/model_card_4
- https://huggingface.co/blog/gemma4

### `audio/`

In-house. No upstream. Centralizes `load_audio` / `load_image` / `load_video` / `normalize_parts` / `process_mm_info`. Used by both `omni/qwen3` and `omni/gemma`.

## In-tree rewrites

### `tts/kokoro/`

In-tree rewrite (~280 lines) of [`thewh1teagle/kokoro-onnx`](https://github.com/thewh1teagle/kokoro-onnx) (MIT). The model is [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) (Apache 2.0); we use the ONNX export at [`onnx-community/Kokoro-82M-v1.0-ONNX`](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX).

**Why rewrite vs vendor**: upstream `kokoro-onnx` requires `phonemizer-fork` and `espeakng-loader`, neither in nixpkgs. Upstream `phonemizer` (no `-fork`) IS in nixpkgs and exposes the identical `phonemize()` API. `espeakng-loader` only does shared-library / data-path discovery -- `pkgs.espeak-ng` + `PHONEMIZER_ESPEAK_LIBRARY` env var cover the same job. Rewrite uses only `numpy`, `onnxruntime`, `phonemizer` (upstream), and `huggingface_hub`.

**Reference fidelity**: the 178-entry IPA -> token-id vocab is transcribed verbatim from upstream's `config.json`. The `_split_phonemes` chunker, `_run` ONNX-input shaping (`input_ids` vs `tokens` + `style` + `speed`), and the `voice_style[len(tokens)]` indexing match upstream byte-for-byte. We don't ship the librosa-based silence-trim pass -- it's ~600 lines for marginal output quality on sentence-boundary-chunked audio.
