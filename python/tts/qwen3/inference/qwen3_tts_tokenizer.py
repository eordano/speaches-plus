import base64
import io
import urllib.request
from typing import Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel

from ..core import (
    Qwen3TTSTokenizerV2Config,
    Qwen3TTSTokenizerV2Model,
)

AudioInput = Union[
    str,
    np.ndarray,
    list[str],
    list[np.ndarray],
]

class Qwen3TTSTokenizer:

    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.config = None
        self.device = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "Qwen3TTSTokenizer":

        inst = cls()

        AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
        AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)

        inst.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
        inst.model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        inst.config = inst.model.config

        inst.device = getattr(inst.model, "device", None)
        if inst.device is None:

            try:
                inst.device = next(inst.model.parameters()).device
            except StopIteration:
                inst.device = torch.device("cpu")

        return inst

    def _is_probably_base64(self, s: str) -> bool:
        if s.startswith("data:audio"):
            return True

        if ("/" not in s and "\\" not in s) and len(s) > 256:
            return True
        return False

    def _is_url(self, s: str) -> bool:
        try:
            u = urlparse(s)
            return u.scheme in ("http", "https") and bool(u.netloc)
        except Exception:
            return False

    def _decode_base64_to_wav_bytes(self, b64: str) -> bytes:

        if "," in b64 and b64.strip().startswith("data:"):
            b64 = b64.split(",", 1)[1]
        return base64.b64decode(b64)

    def load_audio(
        self,
        x: str,
        target_sr: int,
    ) -> np.ndarray:

        if self._is_url(x):
            with urllib.request.urlopen(x) as resp:
                audio_bytes = resp.read()
            with io.BytesIO(audio_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        elif self._is_probably_base64(x):
            wav_bytes = self._decode_base64_to_wav_bytes(x)
            with io.BytesIO(wav_bytes) as f:
                audio, sr = sf.read(f, dtype="float32", always_2d=False)
        else:
            audio, sr = librosa.load(x, sr=None, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        if sr != target_sr:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)

        return audio.astype(np.float32)

    def _normalize_audio_inputs(
        self,
        audios: AudioInput,
        sr: int | None,
    ) -> list[np.ndarray]:

        target_sr = int(self.feature_extractor.sampling_rate)

        if isinstance(audios, (str, np.ndarray)):
            audios = [audios]

        if len(audios) == 0:
            return []

        if isinstance(audios[0], str):

            return [self.load_audio(x, target_sr=target_sr) for x in audios]

        if sr is None:
            raise ValueError("For numpy waveform input, you must provide `sr` (original sampling rate).")

        out: list[np.ndarray] = []
        for a in audios:
            if not isinstance(a, np.ndarray):
                raise TypeError("Mixed input types are not supported. Use all paths/base64 or all numpy arrays.")
            if a.ndim > 1:
                a = np.mean(a, axis=-1)
            if int(sr) != target_sr:
                a = librosa.resample(y=a.astype(np.float32), orig_sr=int(sr), target_sr=target_sr)
            out.append(a.astype(np.float32))
        return out

    def encode(
        self,
        audios: AudioInput,
        sr: int | None = None,
        return_dict: bool = True,
    ):

        wavs = self._normalize_audio_inputs(audios, sr=sr)

        inputs = self.feature_extractor(
            raw_audio=wavs,
            sampling_rate=int(self.feature_extractor.sampling_rate),
            return_tensors="pt",
        )
        inputs = inputs.to(self.device).to(self.model.dtype)

        with torch.inference_mode():

            enc = self.model.encode(
                inputs["input_values"].squeeze(1),
                inputs["padding_mask"].squeeze(1),
                return_dict=return_dict,
            )
        return enc

    def decode(
        self,
        encoded,
    ) -> tuple[list[np.ndarray], int]:

        model_type = self.model.get_model_type()

        def _to_tensor(x, dtype=None):
            if isinstance(x, torch.Tensor):
                return x
            x = np.asarray(x)
            t = torch.from_numpy(x)
            if dtype is not None:
                t = t.to(dtype)
            return t

        if hasattr(encoded, "audio_codes"):

            audio_codes_list = encoded.audio_codes
            xvectors_list = getattr(encoded, "xvectors", None)
            ref_mels_list = getattr(encoded, "ref_mels", None)
        elif isinstance(encoded, dict):
            audio_codes_list = encoded["audio_codes"]
            xvectors_list = encoded.get("xvectors", None)
            ref_mels_list = encoded.get("ref_mels", None)
        elif isinstance(encoded, list):

            audio_codes_list = [e["audio_codes"] for e in encoded]
            xvectors_list = [e["xvectors"] for e in encoded] if ("xvectors" in encoded[0]) else None
            ref_mels_list = [e["ref_mels"] for e in encoded] if ("ref_mels" in encoded[0]) else None
        else:
            raise TypeError("`encoded` must be an encode output, a dict, or a list of dicts.")

        if isinstance(audio_codes_list, torch.Tensor):

            t = audio_codes_list
            if t.dim() == 1:

                t = t.unsqueeze(0)
            elif t.dim() == 2:

                t = t.unsqueeze(0)
            audio_codes_padded = t.to(self.device)
        else:

            audio_codes_list = [_to_tensor(c, dtype=torch.long) for c in audio_codes_list]
            audio_codes_padded = pad_sequence(audio_codes_list, batch_first=True, padding_value=-1).to(self.device)

        with torch.inference_mode():

            if model_type != "qwen3_tts_tokenizer_12hz":
                raise ValueError(
                    f"speaches-plus-python only supports the 12 Hz tokenizer; got {model_type!r}"
                )
            dec = self.model.decode(audio_codes_padded, return_dict=True)
            wav_tensors = dec.audio_values

        wavs = [w.to(torch.float32).detach().cpu().numpy() for w in wav_tensors]
        return wavs, int(self.model.get_output_sample_rate())

    def get_model_type(self) -> str:

        return self.model.get_model_type()

    def get_input_sample_rate(self) -> int:

        return int(self.model.get_input_sample_rate())

    def get_output_sample_rate(self) -> int:

        return int(self.model.get_output_sample_rate())

    def get_encode_downsample_rate(self) -> int:

        return int(self.model.get_encode_downsample_rate())

    def get_decode_upsample_rate(self) -> int:

        return int(self.model.get_decode_upsample_rate())
