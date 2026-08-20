
from __future__ import annotations

from transformers.utils import logging

logger = logging.get_logger(__name__)
import huggingface_hub
from huggingface_hub import snapshot_download

def download_weights_from_hf_specific(
    model_name_or_path: str,
    cache_dir: str | None,
    allow_patterns: list[str],
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:

    assert len(allow_patterns) > 0
    offline = huggingface_hub.constants.HF_HUB_OFFLINE

    for pattern in allow_patterns:
        snapshot_dir = snapshot_download(
            model_name_or_path,
            allow_patterns=pattern,
            ignore_patterns=ignore_patterns,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=offline,
        )
    return snapshot_dir
