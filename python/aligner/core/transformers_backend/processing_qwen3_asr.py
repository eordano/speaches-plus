
import re

import numpy as np

from transformers.audio_utils import AudioInput
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import TextInput

class Qwen3ASRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        },
        "audio_kwargs": {
            "sampling_rate": 16000,
            "padding": True,
            "return_attention_mask": True,
        },
    }

def _get_feat_extract_output_lengths(input_lengths):
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13

class Qwen3ASRProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, feature_extractor=None, tokenizer=None, chat_template=None):
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)
        self.audio_token = self.tokenizer.audio_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token

    def __call__(
        self,
        text: TextInput = None,
        audio: AudioInput = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You need to specify a `text` input.")

        output_kwargs = self._merge_kwargs(
            Qwen3ASRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if audio is not None:
            output_kwargs["audio_kwargs"]["padding"] = True
            output_kwargs["audio_kwargs"]["truncation"] = False
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")
            audio_inputs["input_features"] = audio_inputs.pop("input_features")
            audio_lengths = iter(_get_feat_extract_output_lengths(audio_inputs["feature_attention_mask"].sum(-1)))
        else:
            audio_inputs = {}
            audio_lengths = iter([])

        if not isinstance(text, list):
            text = [text]

        text = self.replace_multimodal_special_tokens(text, audio_lengths)
        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**texts_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def replace_multimodal_special_tokens(self, text, audio_lengths):
        processed_text = []
        for sample in text:
            special_tokens = [re.escape(tok) for tok in [self.audio_token]]
            pattern = "|".join(special_tokens)
            positions = sorted(
                (match.start(), match.group()) for match in re.finditer(pattern, sample)
            )
            for _, special_token in positions:
                if special_token == self.audio_token:
                    sample = sample.replace(
                        self.audio_token,
                        "<|audio_placeholder|>" * next(audio_lengths),
                        1,
                    )
            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            processed_text.append(sample)
        return processed_text

    @property
    def model_input_names(self):
        return list(
            dict.fromkeys(
                self.tokenizer.model_input_names
                + self.feature_extractor.model_input_names
                + ["feature_attention_mask"]
            )
        )

__all__ = ["Qwen3ASRProcessor"]
