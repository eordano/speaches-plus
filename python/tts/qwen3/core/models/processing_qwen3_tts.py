
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin

class Qwen3TTSProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "padding_side": "left",
        }
    }

class Qwen3TTSProcessor(ProcessorMixin):

    attributes = ["tokenizer"]
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self, tokenizer=None, chat_template=None
    ):
        super().__init__(tokenizer, chat_template=chat_template)

    def __call__(self, text=None, **kwargs) -> BatchFeature:

        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            Qwen3TTSProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if not isinstance(text, list):
            text = [text]

        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**texts_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )

    def batch_decode(self, *args, **kwargs):

        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):

        return self.tokenizer.decode(*args, **kwargs)

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        return super().apply_chat_template(conversations, chat_template, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        return list(
            dict.fromkeys(
                tokenizer_input_names
            )
        )

__all__ = ["Qwen3TTSProcessor"]
