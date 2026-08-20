
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Qwen3ASRAudioEncoderConfig(PretrainedConfig):

    model_type = "qwen3_asr_audio_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        d_model=1280,
        dropout=0,
        attention_dropout=0,
        activation_function="gelu",
        activation_dropout=0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        n_window=100,
        output_dim=3584,
        n_window_infer=400,
        conv_chunksize=500,
        downsample_hidden_size=480,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size

class Qwen3ASRTextConfig(PretrainedConfig):

    model_type = "qwen3_asr_text"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=128000,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=5000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

class Qwen3ASRThinkerConfig(PretrainedConfig):

    model_type = "qwen3_asr_thinker"

    attribute_map = {}
    sub_configs = {
        "audio_config": Qwen3ASRAudioEncoderConfig,
        "text_config": Qwen3ASRTextConfig,
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=151646,
        audio_start_token_id=151647,
        user_token_id=872,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_token_id = user_token_id
        self.audio_start_token_id = audio_start_token_id
        self.initializer_range = initializer_range

        if isinstance(audio_config, dict):
            audio_config = Qwen3ASRAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen3ASRAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = Qwen3ASRTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3ASRTextConfig()
        self.text_config = text_config
        self.audio_token_id = audio_token_id

class Qwen3ASRConfig(PretrainedConfig):

    model_type = "qwen3_asr"
    sub_configs = {
        "thinker_config": Qwen3ASRThinkerConfig,
    }

    def __init__(
        self,
        thinker_config=None,
        support_languages=None,
        **kwargs,
    ):

        if thinker_config is None:
            thinker_config = {}
        self.thinker_config = Qwen3ASRThinkerConfig(**thinker_config)
        self.support_languages = support_languages

        super().__init__(**kwargs)

    def get_text_config(self, decoder=False) -> "PretrainedConfig":

        return self.thinker_config.get_text_config()

__all__ = ["Qwen3ASRConfig", "Qwen3ASRThinkerConfig", "Qwen3ASRAudioEncoderConfig"]
