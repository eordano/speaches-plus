import unicodedata
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

from aligner.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)

from .utils import (
    AudioLike,
    ensure_list,
    normalize_audios,
)

_UNSUPPORTED_LANGUAGES = frozenset({"japanese", "korean"})

class Qwen3ForceAlignProcessor:
    def __init__(self):
        pass

    def is_kept_char(self, ch: str) -> bool:
        if ch == "'":
            return True
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            return True
        return False

    def clean_token(self, token: str) -> str:
        return "".join(ch for ch in token if self.is_kept_char(ch))

    def is_cjk_char(self, ch: str) -> bool:
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F
            or 0x2B740 <= code <= 0x2B81F
            or 0x2B820 <= code <= 0x2CEAF
            or 0xF900 <= code <= 0xFAFF
        )

    def split_segment_with_chinese(self, seg: str) -> list[str]:
        tokens: list[str] = []
        buf: list[str] = []

        def flush_buf():
            nonlocal buf
            if buf:
                tokens.append("".join(buf))
                buf = []

        for ch in seg:
            if self.is_cjk_char(ch):
                flush_buf()
                tokens.append(ch)
            else:
                buf.append(ch)

        flush_buf()
        return tokens

    def tokenize_space_lang(self, text: str) -> list[str]:
        tokens: list[str] = []
        for seg in text.split():
            cleaned = self.clean_token(seg)
            if cleaned:
                tokens.extend(self.split_segment_with_chinese(cleaned))
        return tokens

    def fix_timestamp(self, data) -> list[int]:
        data = data.tolist()
        n = len(data)

        dp = [1] * n
        parent = [-1] * n

        for i in range(1, n):
            for j in range(i):
                if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j

        max_length = max(dp)
        max_idx = dp.index(max_length)

        lis_indices = []
        idx = max_idx
        while idx != -1:
            lis_indices.append(idx)
            idx = parent[idx]
        lis_indices.reverse()

        is_normal = [False] * n
        for idx in lis_indices:
            is_normal[idx] = True

        result = data.copy()
        i = 0

        while i < n:
            if not is_normal[i]:
                j = i
                while j < n and not is_normal[j]:
                    j += 1

                anomaly_count = j - i

                if anomaly_count <= 2:
                    left_val = None
                    for k in range(i - 1, -1, -1):
                        if is_normal[k]:
                            left_val = result[k]
                            break

                    right_val = None
                    for k in range(j, n):
                        if is_normal[k]:
                            right_val = result[k]
                            break

                    for k in range(i, j):
                        if left_val is None:
                            result[k] = right_val
                        elif right_val is None:
                            result[k] = left_val
                        else:
                            result[k] = left_val if (k - (i - 1)) <= ((j) - k) else right_val

                else:
                    left_val = None
                    for k in range(i - 1, -1, -1):
                        if is_normal[k]:
                            left_val = result[k]
                            break

                    right_val = None
                    for k in range(j, n):
                        if is_normal[k]:
                            right_val = result[k]
                            break

                    if left_val is not None and right_val is not None:
                        step = (right_val - left_val) / (anomaly_count + 1)
                        for k in range(i, j):
                            result[k] = left_val + step * (k - i + 1)
                    elif left_val is not None:
                        for k in range(i, j):
                            result[k] = left_val
                    elif right_val is not None:
                        for k in range(i, j):
                            result[k] = right_val

                i = j
            else:
                i += 1

        return [int(res) for res in result]

    def encode_timestamp(self, text: str, language: str) -> list[str]:
        language = language.lower()

        if language in _UNSUPPORTED_LANGUAGES:
            raise ValueError(
                f"language={language!r} requires the upstream qwen-asr "
                "Japanese/Korean tokenizer dependencies (nagisa, soynlp + "
                "korean_dict_jieba.dict), which speaches-plus-python doesn't ship. "
                "Other 9 supported languages work."
            )

        word_list = self.tokenize_space_lang(text)

        input_text = "<timestamp><timestamp>".join(word_list) + "<timestamp><timestamp>"
        input_text = "<|audio_start|><|audio_pad|><|audio_end|>" + input_text

        return word_list, input_text

    def parse_timestamp(self, word_list, timestamp):
        timestamp_output = []

        timestamp_fixed = self.fix_timestamp(timestamp)
        for i, word in enumerate(word_list):
            start_time = timestamp_fixed[i * 2]
            end_time = timestamp_fixed[i * 2 + 1]
            timestamp_output.append({
                "text": word,
                "start_time": start_time,
                "end_time": end_time
            })

        return timestamp_output

@dataclass(frozen=True)
class ForcedAlignItem:
    text: str
    start_time: int
    end_time: int

@dataclass(frozen=True)
class ForcedAlignResult:
    items: list[ForcedAlignItem]

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> ForcedAlignItem:
        return self.items[idx]

class Qwen3ForcedAligner:

    def __init__(
        self,
        model: Qwen3ASRForConditionalGeneration,
        processor: Qwen3ASRProcessor,
        aligner_processor: Qwen3ForceAlignProcessor,
    ):
        self.model = model
        self.processor = processor
        self.aligner_processor = aligner_processor

        self.device = getattr(model, "device", None)
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

        self.timestamp_token_id = int(model.config.timestamp_token_id)
        self.timestamp_segment_time = float(model.config.timestamp_segment_time)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> "Qwen3ForcedAligner":
        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
        AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if not isinstance(model, Qwen3ASRForConditionalGeneration):
            raise TypeError(
                f"AutoModel returned {type(model)}, expected Qwen3ASRForConditionalGeneration."
            )

        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True)
        aligner_processor = Qwen3ForceAlignProcessor()

        return cls(model=model, processor=processor, aligner_processor=aligner_processor)

    def _to_structured_items(self, timestamp_output: list[dict[str, Any]]) -> ForcedAlignResult:
        items: list[ForcedAlignItem] = []
        for it in timestamp_output:
            items.append(
                ForcedAlignItem(
                    text=str(it.get("text", "")),
                    start_time=float(it.get("start_time", 0)),
                    end_time=float(it.get("end_time", 0)),
                )
            )
        return ForcedAlignResult(items=items)

    @torch.inference_mode()
    def align(
        self,
        audio: AudioLike | list[AudioLike],
        text: str | list[str],
        language: str | list[str],
    ) -> list[ForcedAlignResult]:
        texts = ensure_list(text)
        languages = ensure_list(language)
        audios = normalize_audios(audio)

        if len(languages) == 1 and len(audios) > 1:
            languages = languages * len(audios)

        if not (len(audios) == len(texts) == len(languages)):
            raise ValueError(
                f"Batch size mismatch: audio={len(audios)}, text={len(texts)}, language={len(languages)}"
            )

        word_lists = []
        aligner_input_texts = []
        for t, lang in zip(texts, languages):
            word_list, aligner_input_text = self.aligner_processor.encode_timestamp(t, lang)
            word_lists.append(word_list)
            aligner_input_texts.append(aligner_input_text)

        inputs = self.processor(
            text=aligner_input_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        logits = self.model.thinker(**inputs).logits
        output_ids = logits.argmax(dim=-1)

        results: list[ForcedAlignResult] = []
        for input_id, output_id, word_list in zip(inputs["input_ids"], output_ids, word_lists):
            masked_output_id = output_id[input_id == self.timestamp_token_id]
            timestamp_ms = (masked_output_id * self.timestamp_segment_time).to("cpu").numpy()
            timestamp_output = self.aligner_processor.parse_timestamp(word_list, timestamp_ms)
            for it in timestamp_output:
                it['start_time'] = round(it['start_time'] / 1000.0, 3)
                it['end_time'] = round(it['end_time'] / 1000.0, 3)
            results.append(self._to_structured_items(timestamp_output))

        return results
