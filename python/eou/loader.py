from __future__ import annotations

import math
from dataclasses import dataclass, field

import env as env_keys

from . import constants
from .fusion import FusionRule
from .types import AudioPadAlignment, Eagerness, EouKind

@dataclass
class EouConfig:
    kind: EouKind = EouKind.VAD
    p_threshold: float = constants.P_THRESHOLD
    min_delay_ms: int = constants.MIN_DELAY_MS
    max_delay_ms: int = constants.MAX_DELAY_MS

    silence_hard_cap_ms: int = constants.SILENCE_HARD_CAP_MS
    inference_timeout_ms: int = constants.INFERENCE_TIMEOUT_MS

    context_turns: int = constants.CONTEXT_TURNS
    audio_window_ms: int = constants.AUDIO_WINDOW_MS
    audio_pad_alignment: AudioPadAlignment = AudioPadAlignment.LEADING

    thresholds: dict[str, float] = field(default_factory=dict)

    eagerness: Eagerness | None = None

    min_speech_for_response_ms: int = constants.MIN_SPEECH_FOR_RESPONSE_MS

    eager_p_threshold: float = constants.EAGER_P_THRESHOLD
    eager_max_inflight: int = constants.EAGER_MAX_INFLIGHT
    eager_periodic_enabled: bool = constants.EAGER_PERIODIC_ENABLED
    eager_interval_ms: int = constants.EAGER_INTERVAL_MS
    predicted_token_buffer_cap: int = constants.PREDICTED_TOKEN_BUFFER_CAP

    eot_threshold: float = constants.EOT_THRESHOLD
    eager_eot_threshold: float = constants.EAGER_EOT_THRESHOLD

    fusion_rule: FusionRule = field(
        default_factory=lambda: (
            FusionRule.parse(constants.FUSION_RULE) or FusionRule.NOISY_OR
        )
    )
    fusion_weight_text: float = constants.FUSION_WEIGHT_TEXT

    curve_k: float = constants.CURVE_K

    failure_p_default: float = constants.FAILURE_P_DEFAULT

    failure_delay_max: bool = False

    @classmethod
    def from_env(cls) -> "EouConfig":
        cfg = cls()
        legacy_enabled = _env_bool(env_keys.EOU_ENABLED)

        eagerness_raw = _env_str(env_keys.EOU_EAGERNESS)
        eagerness = Eagerness.parse(eagerness_raw) if eagerness_raw is not None else None
        if eagerness is not None:
            p, mn, mx = eagerness.triple()
            cfg.p_threshold = p
            cfg.min_delay_ms = int(mn)
            cfg.max_delay_ms = int(mx)
            cfg.eagerness = eagerness
        else:
            p = _env_unit_f32(env_keys.EOU_P_THRESHOLD)
            if p is not None:
                cfg.p_threshold = p
            n = _env_parse_int(env_keys.EOU_MIN_DELAY_MS)
            if n is not None:
                cfg.min_delay_ms = n
            n = _env_parse_int(env_keys.EOU_MAX_DELAY_MS)
            if n is not None:
                cfg.max_delay_ms = n

        v = _env_str(env_keys.EOU_THRESHOLDS)
        if v is not None:
            for entry in v.split(","):
                if ":" not in entry:
                    continue
                lang, score = entry.split(":", 1)
                lang = lang.strip()
                try:
                    s_val = float(score.strip())
                except ValueError:
                    continue
                if not math.isfinite(s_val) or not lang:
                    continue
                if s_val < 0.0:
                    s_val = 0.0
                elif s_val > 1.0:
                    s_val = 1.0
                cfg.thresholds[lang] = s_val

        n = _env_parse_int(env_keys.MIN_SPEECH_FOR_RESPONSE_MS)
        if n is not None:
            cfg.min_speech_for_response_ms = n
        else:
            n = _env_parse_int(env_keys.MIN_SPEECH_FOR_COMMIT_MS)
            if n is not None:
                cfg.min_speech_for_response_ms = n

        p = _env_unit_f32(env_keys.EOU_EAGER_P_THRESHOLD)
        if p is not None:
            cfg.eager_p_threshold = p
        n = _env_parse_int(env_keys.EOU_EAGER_MAX_INFLIGHT)
        if n is not None:
            cfg.eager_max_inflight = n
        b = _env_bool(env_keys.EOU_EAGER_PERIODIC)
        if b is not None:
            cfg.eager_periodic_enabled = b

        kraw = _env_str(env_keys.EOU_KIND)
        kind = EouKind.parse(kraw) if kraw is not None else None
        if kind is not None:
            cfg.kind = kind
        elif legacy_enabled is True:
            cfg.kind = EouKind.TEXT
        if legacy_enabled is False:
            cfg.kind = EouKind.VAD

        n = _env_parse_int(env_keys.EOU_SILENCE_HARD_CAP_MS)
        if n is not None:
            cfg.silence_hard_cap_ms = n
        n = _env_parse_int(env_keys.EOU_INFERENCE_TIMEOUT_MS)
        if n is not None:
            cfg.inference_timeout_ms = n
        n = _env_parse_int(env_keys.EOU_CONTEXT_TURNS)
        if n is not None:
            cfg.context_turns = n
        n = _env_parse_int(env_keys.EOU_AUDIO_WINDOW_MS)
        if n is not None:
            cfg.audio_window_ms = n

        araw = _env_str(env_keys.EOU_AUDIO_PAD_ALIGNMENT)
        align = AudioPadAlignment.parse(araw) if araw is not None else None
        if align is not None:
            cfg.audio_pad_alignment = align

        n = _env_parse_int(env_keys.EOU_EAGER_INTERVAL_MS)
        if n is not None:
            cfg.eager_interval_ms = n
        n = _env_parse_int(env_keys.EOU_PREDICTED_TOKEN_BUFFER_CAP)
        if n is not None:
            cfg.predicted_token_buffer_cap = n

        p = _env_unit_f32(env_keys.EOU_EOT_THRESHOLD)
        if p is not None:
            cfg.eot_threshold = p
        p = _env_unit_f32(env_keys.EOU_EAGER_EOT_THRESHOLD)
        if p is not None:
            cfg.eager_eot_threshold = p

        rraw = _env_str(env_keys.EOU_FUSION_RULE)
        rule = FusionRule.parse(rraw) if rraw is not None else None
        if rule is not None:
            cfg.fusion_rule = rule
        w = _env_unit_f32(env_keys.EOU_FUSION_WEIGHT_TEXT)
        if w is not None:
            cfg.fusion_weight_text = w
        return cfg

    def threshold_for_language(self, lang: str | None) -> float:
        if lang is not None:
            v = self.thresholds.get(lang)
            if v is not None:
                return v
        return self.p_threshold

    def eager_disabled(self) -> bool:
        if not math.isfinite(self.eager_p_threshold):
            return True
        return self.eager_p_threshold >= 1.0

def Load(cfg: EouConfig | None = None) -> EouConfig:
    return cfg if cfg is not None else EouConfig.from_env()

def _env_str(name: str) -> str | None:
    return env_keys.read_str_or_none(name)

def _env_parse_int(name: str) -> int | None:
    raw = _env_str(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None

def _env_unit_f32(name: str) -> float | None:
    raw = _env_str(name)
    if raw is None:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v

def _env_bool(name: str) -> bool | None:
    raw = _env_str(name)
    if raw is None:
        return None
    return raw.lower() in ("1", "true", "yes", "on")
