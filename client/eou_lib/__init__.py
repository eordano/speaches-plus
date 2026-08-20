"""Shared Python helpers for the EOU pipeline test/train tooling.

Submodules:

  smart_turn -- mel preprocessing + onnxruntime wrapper for smart-turn-v3
               (mirrors go/internal/eou/audio.go and rust/.../eou_audio.rs)
  heuristic  -- rule-based text scorer (mirrors go/internal/eou/heuristic.go
               and rust/.../eou.rs::HeuristicEouModel)
  features   -- gated-fusion feature extraction (mirrors
               go/internal/eou/gated_fusion.go::ExtractGatedFusionFeatures
               and rust/.../eou.rs::extract_gated_fusion_features)
  gate       -- gated-fusion combine + trained weights (mirrors
               FuseScoresGated / combine_fusion_gated on both sides)

These are the canonical Python ports of the production gates so test/train
tooling doesn't need a running server or a Go/Rust toolchain.
"""

from .features import GatedFusionFeatures, extract_gated_fusion_features
from .gate import (
    DEFAULT_GATED_FUSION_WEIGHTS,
    GatedFusionWeights,
    combine_fusion,
    combine_fusion_gated,
    combine_fusion_with_features,
)
from .heuristic import heuristic_score

def __getattr__(name):
    if name in {"SmartTurn", "prepare_audio", "log_mel"}:
        from . import smart_turn as _st
        return getattr(_st, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "GatedFusionFeatures",
    "extract_gated_fusion_features",
    "DEFAULT_GATED_FUSION_WEIGHTS",
    "GatedFusionWeights",
    "combine_fusion",
    "combine_fusion_gated",
    "combine_fusion_with_features",
    "heuristic_score",
    "SmartTurn",
    "prepare_audio",
    "log_mel",
]
