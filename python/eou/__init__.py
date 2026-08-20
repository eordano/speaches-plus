from __future__ import annotations

from .fusion import (
    DEFAULT_GATED_FUSION_WEIGHTS,
    FusionEouModel,
    FusionRule,
    GatedFusionFeatures,
    GatedFusionWeights,
    combine_fusion,
    combine_fusion_gated,
    combine_fusion_with_features,
    extract_gated_fusion_features,
    is_garbage_prob,
)
from .heuristic import HeuristicEouModel
from .integrated import (
    FakeIntegratedBackend,
    IntegratedEouBackend,
    IntegratedVerdict,
)
from .loader import EouConfig, Load
from .types import (
    AudioPadAlignment,
    Eagerness,
    EouKind,
    EouModel,
    HardCapRace,
    StubEouModel,
    sigmoid_lerp,
)

__all__ = [
    "AudioPadAlignment",
    "DEFAULT_GATED_FUSION_WEIGHTS",
    "Eagerness",
    "EouConfig",
    "EouKind",
    "EouModel",
    "FakeIntegratedBackend",
    "FusionEouModel",
    "FusionRule",
    "GatedFusionFeatures",
    "GatedFusionWeights",
    "HardCapRace",
    "HeuristicEouModel",
    "IntegratedEouBackend",
    "IntegratedVerdict",
    "Load",
    "StubEouModel",
    "combine_fusion",
    "combine_fusion_gated",
    "combine_fusion_with_features",
    "extract_gated_fusion_features",
    "is_garbage_prob",
    "sigmoid_lerp",
]
