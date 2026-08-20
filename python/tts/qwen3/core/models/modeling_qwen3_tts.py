
from __future__ import annotations

from typing import Optional

import torch
from transformers.generation import GenerationMixin
from transformers.utils import logging
from transformers.utils.hub import cached_file

from .configuration_qwen3_tts import (Qwen3TTSConfig)

logger = logging.get_logger(__name__)
import json
import os

from ...inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
from ._attention import Qwen3TTSAttention, Qwen3TTSTalkerAttention
from ._base import (
    Qwen3TTSPreTrainedModel,
    Qwen3TTSTalkerTextPreTrainedModel,
)
from ._code_predictor import (
    Qwen3TTSTalkerCodePredictorModel,
    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
)
from ._decoder_layers import (
    Qwen3TTSDecoderLayer,
    Qwen3TTSTalkerDecoderLayer,
)
from ._mlp import Qwen3TTSTalkerResizeMLP, Qwen3TTSTalkerTextMLP
from ._norms_rope import (
    Qwen3TTSRMSNorm,
    Qwen3TTSRotaryEmbedding,
    Qwen3TTSTalkerRotaryEmbedding,
)
from ._outputs import (
    Qwen3TTSTalkerCodePredictorOutputWithPast,
    Qwen3TTSTalkerOutputWithPast,
)
from ._speaker_encoder import Qwen3TTSSpeakerEncoder, mel_spectrogram
from ._talker import Qwen3TTSTalkerForConditionalGeneration, Qwen3TTSTalkerModel
from ._utils import download_weights_from_hf_specific

__all__ = [
    "Qwen3TTSForConditionalGeneration",
    "Qwen3TTSTalkerForConditionalGeneration",
    "Qwen3TTSTalkerModel",
    "Qwen3TTSPreTrainedModel",
    "Qwen3TTSTalkerTextPreTrainedModel",
    "Qwen3TTSAttention",
    "Qwen3TTSTalkerAttention",
    "Qwen3TTSDecoderLayer",
    "Qwen3TTSTalkerDecoderLayer",
    "Qwen3TTSRMSNorm",
    "Qwen3TTSRotaryEmbedding",
    "Qwen3TTSTalkerRotaryEmbedding",
    "Qwen3TTSSpeakerEncoder",
    "Qwen3TTSTalkerCodePredictorModel",
    "Qwen3TTSTalkerCodePredictorModelForConditionalGeneration",
    "Qwen3TTSTalkerResizeMLP",
    "Qwen3TTSTalkerTextMLP",
    "Qwen3TTSTalkerOutputWithPast",
    "Qwen3TTSTalkerCodePredictorOutputWithPast",
    "mel_spectrogram",
    "download_weights_from_hf_specific",
]

class Qwen3TTSForConditionalGeneration(Qwen3TTSPreTrainedModel, GenerationMixin):
    config_class = Qwen3TTSConfig

    def __init__(self, config: Qwen3TTSConfig):
        super().__init__(config)
        self.config = config

        self.talker = Qwen3TTSTalkerForConditionalGeneration(self.config.talker_config)

        if config.tts_model_type == "base":
            self.speaker_encoder = Qwen3TTSSpeakerEncoder(self.config.speaker_encoder_config)
        else:
            self.speaker_encoder = None

        self.speech_tokenizer = None
        self.generate_config = None

        self.supported_speakers = self.config.talker_config.spk_id.keys()
        self.supported_languages = ["auto"] + [
            lang for lang in self.config.talker_config.codec_language_id.keys()
            if "dialect" not in lang
        ]

        self.speaker_encoder_sample_rate = self.config.speaker_encoder_config.sample_rate
        self.tokenizer_type = self.config.tokenizer_type
        self.tts_model_size = self.config.tts_model_size
        self.tts_model_type = self.config.tts_model_type

        self.post_init()

    def load_speech_tokenizer(self, speech_tokenizer):
        self.speech_tokenizer = speech_tokenizer

    def load_generate_config(self, generate_config):
        self.generate_config = generate_config

    def get_supported_speakers(self):
        return self.supported_speakers

    def get_supported_languages(self):
        return self.supported_languages

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config=None,
        cache_dir=None,
        ignore_mismatched_sizes=False,
        force_download=False,
        local_files_only=False,
        token=None,
        revision="main",
        use_safetensors=None,
        weights_only=True,
        **kwargs,
    ):

        requested_attn_implementation = kwargs.pop("attn_implementation", None)
        if requested_attn_implementation is None and config and config._attn_implementation:
            requested_attn_implementation = config._attn_implementation

        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            attn_implementation=requested_attn_implementation,
            **kwargs,
        )

        if not local_files_only and not os.path.isdir(pretrained_model_name_or_path):
            download_weights_from_hf_specific(
                pretrained_model_name_or_path,
                cache_dir=kwargs.get("cache_dir", cache_dir),
                allow_patterns=["speech_tokenizer/*"],
                revision=kwargs.get("revision", revision),
            )
        speech_tokenizer_path = cached_file(
            pretrained_model_name_or_path,
            "speech_tokenizer/config.json",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        if speech_tokenizer_path is None:
            raise ValueError(f"""{pretrained_model_name_or_path}/{speech_tokenizer_path} not exists""")
        speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            os.path.dirname(speech_tokenizer_path),
            *model_args,
            **kwargs,
        )
        model.load_speech_tokenizer(speech_tokenizer)

        generate_config_path = cached_file(
            pretrained_model_name_or_path,
            "generation_config.json",
            subfolder=kwargs.pop("subfolder", None),
            cache_dir=kwargs.pop("cache_dir", None),
            force_download=kwargs.pop("force_download", False),
            proxies=kwargs.pop("proxies", None),
            resume_download=kwargs.pop("resume_download", None),
            local_files_only=kwargs.pop("local_files_only", False),
            token=kwargs.pop("use_auth_token", None),
            revision=kwargs.pop("revision", None),
        )
        with open(generate_config_path, "r", encoding="utf-8") as f:
            model.load_generate_config(json.load(f))

        return model

    @torch.inference_mode()
    def extract_speaker_embedding(self, audio, sr):
        assert sr == 24000, "Only support 24kHz audio"
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        return self.speaker_encoder(mels.to(self.device).to(self.dtype))[0]

    @torch.inference_mode()
    def generate_speaker_prompt(self, voice_clone_prompt: list[dict]):
        return [
            ref_spk_embedding.to(self.talker.device).to(self.talker.dtype)
            for ref_spk_embedding in voice_clone_prompt["ref_spk_embedding"]
        ]

    def generate_icl_prompt(
        self,
        text_id: torch.Tensor,
        ref_id: torch.Tensor,
        ref_code: torch.Tensor,
        tts_pad_embed: torch.Tensor,
        tts_eos_embed: torch.Tensor,
        non_streaming_mode: bool,
    ):

        text_embed = self.talker.text_projection(
            self.talker.get_text_embeddings()(torch.cat([ref_id, text_id], dim=-1))
        )
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)

        codec_embed_per_group = []
        for group_idx in range(self.talker.config.num_code_groups):
            if group_idx == 0:
                codec_embed_per_group.append(self.talker.get_input_embeddings()(ref_code[:, :1]))
            else:
                codec_embed_per_group.append(
                    self.talker.code_predictor.get_input_embeddings()[group_idx - 1](
                        ref_code[:, group_idx:group_idx + 1]
                    )
                )
        codec_embed = torch.cat(codec_embed_per_group, dim=1).sum(1).unsqueeze(0)
        codec_bos_embed = self.talker.get_input_embeddings()(
            torch.tensor(
                [[self.config.talker_config.codec_bos_id]],
                device=self.talker.device,
                dtype=text_id.dtype,
            )
        )
        codec_embed = torch.cat([codec_bos_embed, codec_embed], dim=1)

        text_lens = text_embed.shape[1]
        codec_lens = codec_embed.shape[1]

        if non_streaming_mode:
            codec_pad_embed = self.talker.get_input_embeddings()(
                torch.tensor(
                    [[self.config.talker_config.codec_pad_id] * text_lens],
                    device=self.talker.device,
                    dtype=text_id.dtype,
                )
            )
            icl_input_embed = torch.cat(
                [text_embed + codec_pad_embed, codec_embed + tts_pad_embed], dim=1
            )
            return icl_input_embed, tts_pad_embed

        if text_lens > codec_lens:
            return text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]

        text_embed = torch.cat([text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1)
        return text_embed + codec_embed, tts_pad_embed

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[list[torch.Tensor]] = None,
        instruct_ids: Optional[list[torch.Tensor]] = None,
        ref_ids: Optional[list[torch.Tensor]] = None,
        voice_clone_prompt: list[dict] = None,
        languages: list[str] = None,
        speakers: list[str] = None,
        non_streaming_mode=False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.05,
        **kwargs,
    ):
        talker_cfg = self.config.talker_config

        suppress_tokens = [
            i for i in range(talker_cfg.vocab_size - 1024, talker_cfg.vocab_size)
            if i != talker_cfg.codec_eos_token_id
        ]
        talker_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 2,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "subtalker_dosample": subtalker_dosample,
            "subtalker_top_k": subtalker_top_k,
            "subtalker_top_p": subtalker_top_p,
            "subtalker_temperature": subtalker_temperature,
            "eos_token_id": eos_token_id if eos_token_id is not None else talker_cfg.codec_eos_token_id,
            "repetition_penalty": repetition_penalty,
            "suppress_tokens": suppress_tokens,
            "output_hidden_states": getattr(kwargs, "output_hidden_states", True),
            "return_dict_in_generate": getattr(kwargs, "return_dict_in_generate", True),
        }

        batch_size = len(input_ids)
        per_sample_embeds: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]

        voice_clone_spk_embeds = None
        if voice_clone_prompt is not None:
            voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)

        if instruct_ids is not None:
            for index, instruct_id in enumerate(instruct_ids):
                if instruct_id is not None:
                    per_sample_embeds[index].append(
                        self.talker.text_projection(self.talker.get_text_embeddings()(instruct_id))
                    )

        trailing_text_hiddens = []
        if speakers is None:
            speakers = [None] * batch_size

        for index, (input_id, language, speaker) in enumerate(zip(input_ids, languages, speakers)):

            if voice_clone_spk_embeds is None:
                if speaker == "" or speaker is None:
                    speaker_embed = None
                else:
                    if speaker.lower() not in talker_cfg.spk_id:
                        raise NotImplementedError(f"Speaker {speaker} not implemented")
                    speaker_embed = self.talker.get_input_embeddings()(
                        torch.tensor(
                            talker_cfg.spk_id[speaker.lower()],
                            device=self.talker.device,
                            dtype=input_id.dtype,
                        )
                    )
            elif voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt["icl_mode"][index]:
                speaker_embed = voice_clone_spk_embeds[index]
            else:
                speaker_embed = None

            assert language is not None

            if language.lower() == "auto":
                language_id = None
            elif language.lower() not in talker_cfg.codec_language_id:
                raise NotImplementedError(f"Language {language} not implemented")
            else:
                language_id = talker_cfg.codec_language_id[language.lower()]

            if (
                language.lower() in ["chinese", "auto"]
                and speaker != "" and speaker is not None
                and talker_cfg.spk_is_dialect[speaker.lower()] is not False
            ):
                language_id = talker_cfg.codec_language_id[talker_cfg.spk_is_dialect[speaker.lower()]]

            tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
                self.talker.get_text_embeddings()(
                    torch.tensor(
                        [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
                        device=self.talker.device,
                        dtype=input_id.dtype,
                    )
                )
            ).chunk(3, dim=1)

            if language_id is None:
                think_tag_ids = [[
                    talker_cfg.codec_nothink_id,
                    talker_cfg.codec_think_bos_id,
                    talker_cfg.codec_think_eos_id,
                ]]
            else:
                think_tag_ids = [[
                    talker_cfg.codec_think_id,
                    talker_cfg.codec_think_bos_id,
                    language_id,
                    talker_cfg.codec_think_eos_id,
                ]]

            codec_think_embed = self.talker.get_input_embeddings()(
                torch.tensor(think_tag_ids, device=self.talker.device, dtype=input_id.dtype)
            )
            codec_pad_bos_embed = self.talker.get_input_embeddings()(
                torch.tensor(
                    [[talker_cfg.codec_pad_id, talker_cfg.codec_bos_id]],
                    device=self.talker.device,
                    dtype=input_id.dtype,
                )
            )
            if speaker_embed is None:
                codec_prefix_embed = torch.cat([codec_think_embed, codec_pad_bos_embed], dim=1)
            else:
                codec_prefix_embed = torch.cat(
                    [codec_think_embed, speaker_embed.view(1, 1, -1), codec_pad_bos_embed], dim=1
                )

            role_text_embed = self.talker.text_projection(
                self.talker.get_text_embeddings()(input_id[:, :3])
            )
            text_aligned_to_codec = torch.cat(
                (
                    tts_pad_embed.expand(-1, codec_prefix_embed.shape[1] - 2, -1),
                    tts_bos_embed,
                ),
                dim=1,
            ) + codec_prefix_embed[:, :-1]
            talker_input_embed = torch.cat((role_text_embed, text_aligned_to_codec), dim=1)

            if (
                voice_clone_prompt is not None
                and voice_clone_prompt["ref_code"] is not None
                and voice_clone_prompt["icl_mode"][index]
            ):
                icl_input_embed, trailing_text_hidden = self.generate_icl_prompt(
                    text_id=input_id[:, 3:-5],
                    ref_id=ref_ids[index][:, 3:-2],
                    ref_code=voice_clone_prompt["ref_code"][index].to(self.talker.device),
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=non_streaming_mode,
                )
                talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
            else:

                first_text_embed = self.talker.text_projection(
                    self.talker.get_text_embeddings()(input_id[:, 3:4])
                ) + codec_prefix_embed[:, -1:]
                talker_input_embed = torch.cat([talker_input_embed, first_text_embed], dim=1)

                if non_streaming_mode:

                    talker_input_embed = talker_input_embed[:, :-1]
                    text_body_len = input_id[:, 3:-5].shape[1]
                    full_text_with_eos = torch.cat(
                        (
                            self.talker.text_projection(
                                self.talker.get_text_embeddings()(input_id[:, 3:-5])
                            ),
                            tts_eos_embed,
                        ),
                        dim=1,
                    )
                    codec_pad_run = self.talker.get_input_embeddings()(
                        torch.tensor(
                            [[talker_cfg.codec_pad_id] * (text_body_len + 1)],
                            device=self.talker.device,
                            dtype=input_id.dtype,
                        )
                    )
                    codec_bos_embed = self.talker.get_input_embeddings()(
                        torch.tensor(
                            [[talker_cfg.codec_bos_id]],
                            device=self.talker.device,
                            dtype=input_id.dtype,
                        )
                    )
                    talker_input_embed = torch.cat(
                        [
                            talker_input_embed,
                            full_text_with_eos + codec_pad_run,
                            tts_pad_embed + codec_bos_embed,
                        ],
                        dim=1,
                    )
                    trailing_text_hidden = tts_pad_embed
                else:

                    trailing_text_hidden = torch.cat(
                        (
                            self.talker.text_projection(
                                self.talker.get_text_embeddings()(input_id[:, 4:-5])
                            ),
                            tts_eos_embed,
                        ),
                        dim=1,
                    )
            per_sample_embeds[index].append(talker_input_embed)
            trailing_text_hiddens.append(trailing_text_hidden)

        talker_input_embeds = [
            torch.cat([part for part in parts if part is not None], dim=1)
            for parts in per_sample_embeds
        ]

        original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds])
        sequences_reversed = [t.squeeze(0).flip(dims=[0]) for t in talker_input_embeds]
        padded_reversed = torch.nn.utils.rnn.pad_sequence(
            sequences_reversed, batch_first=True, padding_value=0.0
        )
        talker_input_embeds = padded_reversed.flip(dims=[1])

        batch_size, max_len = talker_input_embeds.shape[0], talker_input_embeds.shape[1]
        indices = torch.arange(max_len).expand(batch_size, -1)
        num_pads = max_len - original_lengths
        talker_attention_mask = (indices >= num_pads.unsqueeze(1)).long().to(talker_input_embeds.device)

        trailing_sequences = [t.squeeze(0) for t in trailing_text_hiddens]
        trailing_lengths = [s.shape[0] for s in trailing_sequences]
        padded_trailing = torch.nn.utils.rnn.pad_sequence(
            trailing_sequences, batch_first=True, padding_value=0.0
        )
        trailing_pad_mask = (
            torch.arange(max(trailing_lengths), device=padded_trailing.device)
            .expand(len(trailing_lengths), -1)
            >= torch.tensor(trailing_lengths, device=padded_trailing.device).unsqueeze(1)
        )
        padded_trailing[trailing_pad_mask] = tts_pad_embed.squeeze()
        trailing_text_hiddens = padded_trailing

        talker_result = self.talker.generate(
            inputs_embeds=talker_input_embeds,
            attention_mask=talker_attention_mask,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            **talker_kwargs,
        )

        talker_codes = torch.stack(
            [step[-1] for step in talker_result.hidden_states if step[-1] is not None], dim=1
        )
        talker_hidden_states = torch.cat(
            [step[0][-1][:, -1:] for step in talker_result.hidden_states], dim=1
        )[:, :-1]

        first_codebook = talker_codes[:, :, 0]
        is_stop_token = first_codebook == talker_cfg.codec_eos_token_id
        stop_indices = torch.argmax(is_stop_token.int(), dim=1)
        has_stop_token = is_stop_token.any(dim=1)
        effective_lengths = torch.where(has_stop_token, stop_indices, talker_codes.shape[1])

        talker_codes_list = [talker_codes[i, :length] for i, length in enumerate(effective_lengths)]
        talker_hidden_states_list = [
            talker_hidden_states[i, :length, :] for i, length in enumerate(effective_lengths)
        ]

        return talker_codes_list, talker_hidden_states_list

__all__ = [
    "Qwen3TTSForConditionalGeneration",
    "Qwen3TTSTalkerForConditionalGeneration",
    "Qwen3TTSPreTrainedModel",
    "Qwen3TTSTalkerModel",
]
