from __future__ import annotations

import json
import threading
from typing import Any

import torch
import xgrammar
from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

from audio import normalize_parts, process_mm_info

def move_inputs_to(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    target_device = model.device
    target_dtype = model.dtype
    moved: dict[str, Any] = {}
    for name, value in batch.items():
        if not hasattr(value, "to"):
            moved[name] = value
            continue
        moved_value = value.to(target_device)
        if moved_value.dtype.is_floating_point:
            moved_value = moved_value.to(target_dtype)
        moved[name] = moved_value
    return moved

def resolve_text_tokenizer(processor: Any) -> Any:
    return getattr(processor, "tokenizer", None) or processor

def resolve_vocab_size(*sources: Any, fallback: Any = None) -> int:
    for source in sources:
        if source is None:
            continue
        size = getattr(source, "vocab_size", None)
        if size:
            return int(size)
    if fallback is not None:
        return int(getattr(fallback, "vocab_size", len(fallback)))
    return 0

def build_processor_kwargs(
    processor: Any,
    conversation: list[dict[str, Any]],
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    templated = normalize_parts(conversation)
    templated_text = processor.apply_chat_template(
        templated, add_generation_prompt=True, tokenize=False,
    )
    audios, images, videos = process_mm_info(conversation)
    kwargs: dict[str, Any] = {
        "text": templated_text,
        "return_tensors": "pt",
        "padding": True,
    }
    if extra:
        kwargs.update(extra)
    if audios:
        kwargs["audio"] = audios
    if images:
        kwargs["images"] = images
    if videos:
        kwargs["videos"] = videos
    return kwargs

class GrammarCompilerCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._compiler: xgrammar.GrammarCompiler | None = None

    def get_or_init(
        self,
        tokenizer: Any,
        vocab_size: int,
    ) -> xgrammar.GrammarCompiler:
        if self._compiler is not None:
            return self._compiler
        with self._lock:
            if self._compiler is not None:
                return self._compiler
            info = xgrammar.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
            self._compiler = xgrammar.GrammarCompiler(info)
        return self._compiler

def compile_guided_json(
    compiler: xgrammar.GrammarCompiler,
    guided_json: dict[str, Any] | str,
) -> xgrammar.CompiledGrammar:
    if isinstance(guided_json, dict):
        if not guided_json:
            return compiler.compile_builtin_json_grammar()
        schema = json.dumps(guided_json)
    else:
        schema = guided_json
    return compiler.compile_json_schema(schema)

def transcribe_prompt(language: str | None) -> str:
    if language and language.lower() != "auto":
        return f"Transcribe this audio in {language}."
    return "Transcribe this audio."

def transcribe_conversation(audio_spec: str, prompt: str) -> list[dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_spec},
            {"type": "text", "text": prompt},
        ],
    }]

def xgrammar_logits_processor(compiled: xgrammar.CompiledGrammar) -> XGrammarLogitsProcessor:
    return XGrammarLogitsProcessor(compiled)
