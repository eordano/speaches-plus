from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import torch
import xgrammar

import env
from .._common import (
    GrammarCompilerCache,
    build_processor_kwargs,
    compile_guided_json,
    move_inputs_to,
    resolve_text_tokenizer,
    resolve_vocab_size,
    transcribe_conversation,
    transcribe_prompt,
    xgrammar_logits_processor,
)

DEFAULT_TEST_MODEL = "google/gemma-4-E4B-it"
DEFAULT_PROD_MODEL = "google/gemma-4-31B-it"
INPUT_AUDIO_SR = 16000

DEFAULT_CHAT_MAX_NEW_TOKENS = 512
DEFAULT_TRANSCRIBE_MAX_NEW_TOKENS = 512
DEFAULT_DEVICE_FOR_AUTO_MAP = "cuda:0"
ATTN_FLASH = "flash_attention_2"
ATTN_SDPA = "sdpa"
COMPILE_MODE_REDUCE_OVERHEAD = "reduce-overhead"

@dataclass
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

def _flash_attention_2_available() -> bool:
    return importlib.util.find_spec("flash_attn") is not None

def _pick_attn_implementation(device: str) -> str:
    override = env.read_str_or_none(env.GEMMA_ATTN_IMPL)
    if override:
        return override
    if device.startswith("cuda") and _flash_attention_2_available():
        return ATTN_FLASH
    return ATTN_SDPA

def _opt_in_compile(model: Any, device: str) -> Any:
    if not env.read_bool(env.GEMMA_COMPILE, default=False) or device == "mps":
        return model
    return torch.compile(model, mode=COMPILE_MODE_REDUCE_OVERHEAD, fullgraph=False)

def _output_embedding_weight(model: Any) -> Any:
    embedding = getattr(model, "get_output_embeddings", lambda: None)()
    return getattr(embedding, "weight", None)

def _resolve_gemma_vocab_size(model: Any, tokenizer: Any) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) if config is not None else None
    size = resolve_vocab_size(text_config, config, fallback=None)
    if size:
        return size
    weight = _output_embedding_weight(model)
    if weight is not None:
        return int(weight.shape[0])
    return int(getattr(tokenizer, "vocab_size", len(tokenizer)))

class Gemma4Wrapper:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self._grammar = GrammarCompilerCache()

    def _ensure_grammar_compiler(self) -> xgrammar.GrammarCompiler:
        tokenizer = resolve_text_tokenizer(self.processor)
        vocab_size = _resolve_gemma_vocab_size(self.model, tokenizer)
        return self._grammar.get_or_init(tokenizer, vocab_size)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict[str, Any] | None = "auto",
        attn_implementation: str | None = None,
    ) -> Gemma4Wrapper:
        from transformers import Gemma4ForConditionalGeneration, Gemma4Processor

        device = device_map if isinstance(device_map, str) else DEFAULT_DEVICE_FOR_AUTO_MAP
        if attn_implementation is None:
            attn_implementation = _pick_attn_implementation(device)

        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "attn_implementation": attn_implementation,
            "low_cpu_mem_usage": True,
        }
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        model = Gemma4ForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        model = _opt_in_compile(model, device)
        processor = Gemma4Processor.from_pretrained(model_id)
        return cls(model, processor)

    def _build_inputs(self, conversation: list[dict[str, Any]]) -> dict[str, Any]:
        kwargs = build_processor_kwargs(self.processor, conversation)
        return move_inputs_to(self.model, self.processor(**kwargs))

    @torch.inference_mode()
    def chat(
        self,
        conversation: list[dict[str, Any]],
        *,
        max_new_tokens: int = DEFAULT_CHAT_MAX_NEW_TOKENS,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
        guided_json: dict[str, Any] | str | None = None,
    ) -> ChatResult:
        inputs = self._build_inputs(conversation)
        prompt_length = inputs["input_ids"].shape[1]

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }
        for key, val in (
            ("temperature", temperature),
            ("top_p", top_p),
            ("do_sample", do_sample),
        ):
            if val is not None:
                generate_kwargs[key] = val
        if guided_json is not None:
            compiled = compile_guided_json(self._ensure_grammar_compiler(), guided_json)
            generate_kwargs.setdefault("logits_processor", []).append(
                xgrammar_logits_processor(compiled),
            )

        output_ids = self.model.generate(**inputs, **generate_kwargs)
        completion_ids = output_ids[:, prompt_length:]
        decoded = self.processor.batch_decode(
            completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        return ChatResult(
            text=(decoded[0] if decoded else "").strip(),
            prompt_tokens=int(prompt_length),
            completion_tokens=int(completion_ids.shape[1]),
        )

    @torch.inference_mode()
    def transcribe(
        self,
        audio_spec: str,
        *,
        language: str | None = None,
        prompt: str | None = None,
        max_new_tokens: int = DEFAULT_TRANSCRIBE_MAX_NEW_TOKENS,
    ) -> str:
        return self.chat(
            transcribe_conversation(audio_spec, prompt or transcribe_prompt(language)),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        ).text
