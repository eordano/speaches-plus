
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import (ModelOutput)
from transformers.utils import logging

logger = logging.get_logger(__name__)

@dataclass
class Qwen3TTSTalkerCodePredictorOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    generation_steps: Optional[int] = None

@dataclass
class Qwen3TTSTalkerOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    past_hidden: Optional[torch.FloatTensor] = None
    generation_step: Optional[int] = None
    trailing_text_hidden: Optional[torch.FloatTensor] = None
    tts_pad_embed: Optional[torch.FloatTensor] = None
