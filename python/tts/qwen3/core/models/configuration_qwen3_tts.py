
from transformers.configuration_utils import PretrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Qwen3TTSSpeakerEncoderConfig(PretrainedConfig):

    def __init__(
        self,
        mel_dim=128,
        enc_dim=1024,
        enc_channels=[512, 512, 512, 512, 1536],
        enc_kernel_sizes=[5, 3, 3, 3, 1],
        enc_dilations=[1, 2, 3, 4, 1],
        enc_attention_channels=128,
        enc_res2net_scale=8,
        enc_se_channels=128,
        sample_rate=24000,
    ):
        self.mel_dim = mel_dim
        self.enc_dim = enc_dim
        self.enc_channels = enc_channels
        self.enc_kernel_sizes = enc_kernel_sizes
        self.enc_dilations = enc_dilations
        self.enc_attention_channels = enc_attention_channels
        self.enc_res2net_scale = enc_res2net_scale
        self.enc_se_channels = enc_se_channels
        self.sample_rate = sample_rate

class Qwen3TTSTalkerCodePredictorConfig(PretrainedConfig):

    model_type = "qwen3_tts_talker_code_predictor"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=2048,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=5,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=0.000001,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        layer_types=None,
        attention_dropout=0,
        num_code_groups=32,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

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
        rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types)
        self.num_code_groups = num_code_groups

class Qwen3TTSTalkerConfig(PretrainedConfig):

    model_type = "qwen3_tts_talker"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    sub_configs = {"code_predictor_config": Qwen3TTSTalkerCodePredictorConfig}

    def __init__(
        self,
        code_predictor_config=None,
        vocab_size=3072,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=0.000001,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=4096,
        attention_dropout=0,
        num_code_groups=32,
        text_hidden_size=2048,
        codec_eos_token_id=4198,
        codec_think_id=4202,
        codec_nothink_id=4203,
        codec_think_bos_id=4204,
        codec_think_eos_id=4205,
        codec_pad_id=4196,
        codec_bos_id=4197,
        spk_id=None,
        spk_is_dialect=None,
        codec_language_id=None,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None

        self.num_key_value_heads = num_key_value_heads
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

        if code_predictor_config is None:
            code_predictor_config = {}
            self.code_predictor_config = Qwen3TTSTalkerCodePredictorConfig()
            logger.info("code_predictor_config is None. Initializing code_predictor model with default values")
        elif isinstance(code_predictor_config, Qwen3TTSTalkerCodePredictorConfig):
            self.code_predictor_config = code_predictor_config
        else:
            self.code_predictor_config = Qwen3TTSTalkerCodePredictorConfig(**code_predictor_config)
        self.num_code_groups = num_code_groups
        self.text_hidden_size = text_hidden_size
        self.codec_eos_token_id = codec_eos_token_id
        self.codec_think_id = codec_think_id
        self.codec_language_id = codec_language_id
        self.codec_nothink_id = codec_nothink_id
        self.codec_think_bos_id = codec_think_bos_id
        self.codec_think_eos_id = codec_think_eos_id
        self.codec_pad_id = codec_pad_id
        self.codec_bos_id = codec_bos_id
        self.spk_id = spk_id
        self.spk_is_dialect = spk_is_dialect

class Qwen3TTSConfig(PretrainedConfig):

    model_type = "qwen3_tts"
    sub_configs = {
        "talker_config": Qwen3TTSTalkerConfig,
        "speaker_encoder_config": Qwen3TTSSpeakerEncoderConfig,
    }

    def __init__(
        self,
        talker_config=None,
        speaker_encoder_config=None,
        tokenizer_type=None,
        tts_model_size=None,
        tts_model_type=None,
        im_start_token_id=151644,
        im_end_token_id=151645,
        tts_pad_token_id=151671,
        tts_bos_token_id=151672,
        tts_eos_token_id=151673,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if talker_config is None:
            talker_config = {}
            logger.info("talker_config is None. Initializing talker model with default values")
        if speaker_encoder_config is None:
            speaker_encoder_config = {}
            logger.info("speaker_encoder_config is None. Initializing talker model with default values")

        self.talker_config = Qwen3TTSTalkerConfig(**talker_config)
        self.speaker_encoder_config = Qwen3TTSSpeakerEncoderConfig(**speaker_encoder_config)

        self.tokenizer_type = tokenizer_type
        self.tts_model_size = tts_model_size
        self.tts_model_type = tts_model_type

        self.im_start_token_id = im_start_token_id
        self.im_end_token_id = im_end_token_id
        self.tts_pad_token_id = tts_pad_token_id
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id

__all__ = ["Qwen3TTSConfig", "Qwen3TTSTalkerConfig", "Qwen3TTSSpeakerEncoderConfig"]
