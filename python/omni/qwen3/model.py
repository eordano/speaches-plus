from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import xgrammar

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

OUTPUT_AUDIO_SR = 24000
INPUT_AUDIO_SR = 16000
DEFAULT_SPEAKER = "Ethan"
SUPPORTED_SPEAKERS = ("Ethan", "Chelsie", "Aiden")

DEFAULT_THINKER_MAX_NEW_TOKENS = 512
DEFAULT_TALKER_MAX_NEW_TOKENS = 4096
DEFAULT_TRANSCRIBE_MAX_NEW_TOKENS = 512

@dataclass
class ChatResult:
    text: str
    audio: np.ndarray | None
    sample_rate: int = OUTPUT_AUDIO_SR
    prompt_tokens: int = 0
    completion_tokens: int = 0

def _split_generate_output(out: Any) -> tuple[Any, Any]:
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, None

def _waveform_from_talker_output(audio: Any) -> np.ndarray | None:
    if audio is None:
        return None
    return audio.reshape(-1).detach().to(torch.float32).cpu().numpy()

def _thinker_text_config(model: Any) -> Any:
    config = getattr(getattr(model, "thinker", None), "config", None)
    return getattr(config, "text_config", None) if config is not None else None

def _thinker_config(model: Any) -> Any:
    return getattr(getattr(model, "thinker", None), "config", None)

class Qwen3OmniWrapper:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self._grammar = GrammarCompilerCache()

    def _ensure_grammar_compiler(self) -> xgrammar.GrammarCompiler:
        tokenizer = resolve_text_tokenizer(self.processor)
        vocab_size = resolve_vocab_size(
            _thinker_text_config(self.model),
            _thinker_config(self.model),
            fallback=tokenizer,
        )
        return self._grammar.get_or_init(tokenizer, vocab_size)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict[str, Any] | None = "auto",
        attn_implementation: str | None = None,
        disable_talker: bool = False,
    ) -> Qwen3OmniWrapper:
        from transformers import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
        )

        load_kwargs: dict[str, Any] = {"dtype": dtype, "low_cpu_mem_usage": True}
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation

        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        if disable_talker and hasattr(model, "disable_talker"):
            model.disable_talker()
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_id)
        return cls(model, processor)

    def _build_inputs(
        self, conversation: list[dict[str, Any]], *, use_audio_in_video: bool,
    ) -> dict[str, Any]:
        kwargs = build_processor_kwargs(
            self.processor,
            conversation,
            extra={"use_audio_in_video": use_audio_in_video},
        )
        return move_inputs_to(self.model, self.processor(**kwargs))

    @torch.inference_mode()
    def chat(
        self,
        conversation: list[dict[str, Any]],
        *,
        return_audio: bool = False,
        speaker: str = DEFAULT_SPEAKER,
        thinker_max_new_tokens: int = DEFAULT_THINKER_MAX_NEW_TOKENS,
        talker_max_new_tokens: int = DEFAULT_TALKER_MAX_NEW_TOKENS,
        thinker_temperature: float | None = None,
        thinker_top_p: float | None = None,
        thinker_do_sample: bool | None = None,
        use_audio_in_video: bool = False,
        guided_json: dict[str, Any] | str | None = None,
    ) -> ChatResult:
        inputs = self._build_inputs(conversation, use_audio_in_video=use_audio_in_video)
        prompt_length = inputs["input_ids"].shape[1]

        generate_kwargs: dict[str, Any] = {
            "speaker": speaker,
            "return_audio": return_audio,
            "thinker_max_new_tokens": thinker_max_new_tokens,
            "talker_max_new_tokens": talker_max_new_tokens,
            "thinker_return_dict_in_generate": True,
            "use_audio_in_video": use_audio_in_video,
        }
        for key, val in (
            ("thinker_temperature", thinker_temperature),
            ("thinker_top_p", thinker_top_p),
            ("thinker_do_sample", thinker_do_sample),
        ):
            if val is not None:
                generate_kwargs[key] = val
        if guided_json is not None:
            compiled = compile_guided_json(self._ensure_grammar_compiler(), guided_json)
            generate_kwargs["thinker_logits_processor"] = [xgrammar_logits_processor(compiled)]

        text_ids_or_dict, audio_tensor = _split_generate_output(
            self.model.generate(**inputs, **generate_kwargs),
        )
        sequences = (
            text_ids_or_dict.sequences
            if hasattr(text_ids_or_dict, "sequences")
            else text_ids_or_dict
        )
        completion_ids = sequences[:, prompt_length:]
        decoded = self.processor.batch_decode(
            completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        return ChatResult(
            text=(decoded[0] if decoded else "").strip(),
            audio=_waveform_from_talker_output(audio_tensor),
            sample_rate=OUTPUT_AUDIO_SR,
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
            return_audio=False,
            thinker_max_new_tokens=max_new_tokens,
        ).text
