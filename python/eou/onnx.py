from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

import env as env_keys

from . import constants
from . import byte_map
from .bpe import Tokenizer
from .chat_template import Turn, format_qwen_chat, rolling_history
from .types import EouModel

class TextEouModel(EouModel):
    def __init__(
        self,
        session,
        tokenizer: Tokenizer,
        max_ctx_tokens: int,
    ) -> None:
        self._session = session
        self._lock = threading.Lock()
        self._tokenizer = tokenizer
        self._max_ctx_tokens = (
            int(constants.MAX_CONTEXT_TOKENS)
            if max_ctx_tokens == 0
            else int(max_ctx_tokens)
        )
        self._im_end_id = tokenizer.im_end_id()
        self._has_attention_mask = True

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
    ) -> "TextEouModel":
        return cls.load_with_capacity(
            model_path,
            tokenizer_path,
            int(constants.MAX_CONTEXT_TOKENS),
        )

    @classmethod
    def load_with_capacity(
        cls,
        model_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
        max_ctx_tokens: int,
    ) -> "TextEouModel":
        import onnxruntime as ort

        tok = Tokenizer.load_from_path(tokenizer_path)
        if tok.im_end_id() < 0:
            raise ValueError("tokenizer has no <|im_end|> token")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        return cls(session, tok, max_ctx_tokens)

    @classmethod
    def from_parts(
        cls,
        session,
        tokenizer: Tokenizer,
        max_ctx_tokens: int,
    ) -> "TextEouModel":
        return cls(session, tokenizer, max_ctx_tokens)

    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def score_with_turns(self, turns: list[Turn], partial: str) -> float:
        prompt = format_qwen_chat(turns, partial)
        return self.score_prompt(prompt)

    def score_prompt(self, prompt: str) -> float:
        ids = self._tokenizer.encode(prompt)
        if not ids:
            return float(constants.FAILURE_P_DEFAULT)
        if len(ids) > self._max_ctx_tokens:
            truncated = ids[len(ids) - self._max_ctx_tokens :]
        else:
            truncated = list(ids)
        try:
            p = self._run_inference(truncated)
        except Exception:
            return float("nan")
        if math.isfinite(p) and 0.0 <= p <= 1.0:
            return p
        return float("nan")

    def _run_inference(self, ids: list[int]) -> float:
        n = len(ids)
        input_ids = np.asarray(ids, dtype=np.int64).reshape(1, n)
        mask = np.ones((1, n), dtype=np.int64)
        with self._lock:
            feed = {constants.INPUT_IDS: input_ids}
            if self._has_attention_mask:
                feed[constants.ATTENTION_MASK] = mask
            outputs = self._session.run([constants.OUTPUT_LOGITS], feed)
        first = np.asarray(outputs[0])
        shape = list(first.shape)
        logits = first.reshape(-1).astype(np.float32, copy=False)
        return extract_im_end_prob(logits, shape, self._im_end_id)

    def score(self, context: str) -> float:
        return self.score_with_turns([], context)

def extract_im_end_prob(logits, shape, im_end_id: int) -> float:
    arr = np.asarray(logits, dtype=np.float32).reshape(-1)
    if len(shape) < 2 or arr.size == 0:
        raise ValueError(f"empty logits (shape={shape})")
    vocab = int(shape[-1])
    if vocab == 0 or im_end_id < 0 or int(im_end_id) >= vocab:
        raise ValueError(f"im_end_id {im_end_id} out of vocab {vocab}")
    if arr.size < vocab:
        raise ValueError(f"logits length {arr.size} < vocab {vocab}")
    last_start = arr.size - vocab
    row = arr[last_start : last_start + vocab]
    max_logit = float(np.max(row))
    shifted = (row.astype(np.float64) - float(max_logit))
    exps = np.exp(shifted)
    s = float(np.sum(exps))
    if s <= 0.0:
        raise ValueError("degenerate logits (sum<=0)")
    p = float(exps[int(im_end_id)]) / s
    return float(p)

@dataclass
class TextEouPaths:
    model_path: Path
    tokenizer_path: Path

def resolve_text_eou_paths() -> Optional[TextEouPaths]:
    raw = env_keys.read_str_or_none(env_keys.EOU_MODEL_PATH)
    if raw is None:
        return None
    model_path = Path(raw)
    if not model_path.exists():
        return None
    raw_tok = env_keys.read_str_or_none(env_keys.EOU_TOKENIZER_PATH)
    if raw_tok is not None:
        tokenizer_path = Path(raw_tok)
    else:
        parent = model_path.parent if str(model_path.parent) else Path(".")
        tokenizer_path = parent / "tokenizer.json"
    return TextEouPaths(model_path=model_path, tokenizer_path=tokenizer_path)

def build_mock_tokenizer_json() -> str:
    table = byte_map.byte_to_char_table()
    vocab: list[tuple[str, int]] = []
    seen: set[str] = set()
    tid = 0
    for ch in table:
        s = ch
        if s not in seen:
            seen.add(s)
            vocab.append((s, tid))
            tid += 1
    extras = ["Ġh", "Ġhe", "Ġhel", "Ġhell", "Ġhello", "Ġworld"]
    for tok in extras:
        if tok not in seen:
            seen.add(tok)
            vocab.append((tok, tid))
            tid += 1
    im_start_id = tid
    vocab.append((constants.IM_START, im_start_id))
    tid += 1
    im_end_id = tid
    vocab.append((constants.IM_END, im_end_id))
    merges = [
        "Ġ h",
        "Ġh e",
        "Ġhe l",
        "Ġhel l",
        "Ġhell o",
        "Ġ w",
        "Ġw o",
        "Ġwo r",
        "Ġwor l",
        "Ġworl d",
    ]
    vocab_obj = {k: v for k, v in vocab}
    added = [
        {"id": im_start_id, "content": constants.IM_START, "special": True},
        {"id": im_end_id, "content": constants.IM_END, "special": True},
    ]
    doc = {
        "added_tokens": added,
        "model": {"type": "BPE", "vocab": vocab_obj, "merges": merges},
    }
    return json.dumps(doc)

_SHARED_TEXT_EOU_LOCK = threading.Lock()
_SHARED_TEXT_EOU: Optional[TextEouModel] = None
_SHARED_TEXT_EOU_INITIALIZED = False

def shared_text_eou_model() -> Optional[TextEouModel]:
    global _SHARED_TEXT_EOU, _SHARED_TEXT_EOU_INITIALIZED
    with _SHARED_TEXT_EOU_LOCK:
        if _SHARED_TEXT_EOU_INITIALIZED:
            return _SHARED_TEXT_EOU
        _SHARED_TEXT_EOU_INITIALIZED = True
        paths = resolve_text_eou_paths()
        if paths is None:
            _SHARED_TEXT_EOU = None
            return None
        max_ctx = env_keys.read_int(env_keys.EOU_MAX_CONTEXT_TOKENS, int(constants.MAX_CONTEXT_TOKENS))
        try:
            _SHARED_TEXT_EOU = TextEouModel.load_with_capacity(
                paths.model_path, paths.tokenizer_path, max_ctx
            )
        except Exception:
            _SHARED_TEXT_EOU = None
        return _SHARED_TEXT_EOU

__all__ = [
    "TextEouModel",
    "TextEouPaths",
    "Tokenizer",
    "Turn",
    "extract_im_end_prob",
    "format_qwen_chat",
    "rolling_history",
    "build_mock_tokenizer_json",
    "resolve_text_eou_paths",
    "shared_text_eou_model",
]
