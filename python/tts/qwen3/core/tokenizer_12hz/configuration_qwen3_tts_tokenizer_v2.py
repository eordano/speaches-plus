
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from transformers import MimiConfig

logger = logging.get_logger(__name__)

class Qwen3TTSTokenizerV2DecoderConfig(PretrainedConfig):

    def __init__(
        self,
        codebook_size=2048,
        hidden_size=1024,
        latent_dim=1024,
        max_position_embeddings=8000,
        rope_theta=10000,
        num_attention_heads=16,
        num_key_value_heads=16,
        attention_bias=False,
        sliding_window=72,
        intermediate_size=3072,
        hidden_act="silu",
        layer_scale_initial_scale=0.01,
        rms_norm_eps=1e-5,
        num_hidden_layers=8,
        num_quantizers=16,
        upsample_rates=(8, 5, 4, 3),
        upsampling_ratios=(2, 2),
        decoder_dim=1536,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.codebook_size = codebook_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_bias = attention_bias
        self.sliding_window = sliding_window
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.layer_scale_initial_scale = layer_scale_initial_scale
        self.rms_norm_eps = rms_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.num_quantizers = num_quantizers
        self.upsample_rates = upsample_rates
        self.upsampling_ratios = upsampling_ratios
        self.decoder_dim = decoder_dim
        self.attention_dropout = attention_dropout

    @property
    def layer_types(self):

        return ["sliding_attention"] * self.num_hidden_layers

class Qwen3TTSTokenizerV2Config(PretrainedConfig):

    model_type = "qwen3_tts_tokenizer_12hz"
    sub_configs = {
        "encoder_config": MimiConfig,
        "decoder_config": Qwen3TTSTokenizerV2DecoderConfig,
    }

    def __init__(
        self,
        encoder_config=None,
        decoder_config=None,
        encoder_valid_num_quantizers=16,
        input_sample_rate=24000,
        output_sample_rate=24000,
        decode_upsample_rate=1920,
        encode_downsample_rate=1920,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if encoder_config is None:
            encoder_config = {}
            logger.info("encoder_config is None. Initializing encoder with default values")
        if decoder_config is None:
            decoder_config = {}
            logger.info("decoder_config is None. Initializing decoder with default values")

        self.encoder_config = MimiConfig(**encoder_config)
        self.decoder_config = Qwen3TTSTokenizerV2DecoderConfig(**decoder_config)

        self.encoder_valid_num_quantizers = encoder_valid_num_quantizers
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.decode_upsample_rate = decode_upsample_rate
        self.encode_downsample_rate = encode_downsample_rate

__all__ = ["Qwen3TTSTokenizerV2Config", "Qwen3TTSTokenizerV2DecoderConfig"]
