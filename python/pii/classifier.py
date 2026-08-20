from __future__ import annotations

import os
import threading
from typing import List

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from pii.spans import PiiSpan, assemble_spans
from pii.viterbi import viterbi_decode

MODEL_ID = os.environ.get("REDACT_MODEL_ID", "openai/privacy-filter")
DEVICE = os.environ.get(
    "REDACT_DEVICE",
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
)
DTYPE = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}[os.environ.get("REDACT_DTYPE", "fp16" if DEVICE != "cpu" else "fp32")]
MAX_BATCH = int(os.environ.get("REDACT_MAX_BATCH", "32"))

class PiiClassifier:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        mdl = AutoModelForTokenClassification.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
        ).to(DEVICE).eval()
        id2label = mdl.config.id2label
        labels = [id2label[i] for i in range(len(id2label))]
        self._tokenizer = tok
        self._model = mdl
        self._labels = labels

    def classify_one(self, text: str) -> List[PiiSpan]:
        if not text.strip():
            return []
        with self._lock:
            enc = self._tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            attn = enc["attention_mask"][0].tolist()
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.inference_mode():
                logits = self._model(**enc).logits[0].float().cpu().numpy()
        path = viterbi_decode(logits, self._labels)
        label_names = [self._labels[int(i)] for i in path]
        return assemble_spans(label_names, offsets, attn)

    def classify_batch(self, texts: List[str]) -> List[List[PiiSpan]]:
        if not texts:
            return []
        with self._lock:
            enc = self._tokenizer(
                texts,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                padding=True,
            )
            offsets = enc.pop("offset_mapping").tolist()
            attn = enc["attention_mask"].tolist()
            inputs = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.inference_mode():
                logits = self._model(**inputs).logits.float().cpu().numpy()

        out: List[List[PiiSpan]] = []
        for i, text in enumerate(texts):
            if not text.strip():
                out.append([])
                continue
            T = sum(attn[i])
            path = viterbi_decode(np.asarray(logits[i, :T]), self._labels)
            label_names = [self._labels[int(j)] for j in path]
            out.append(assemble_spans(label_names, offsets[i][:T], attn[i][:T]))
        return out
