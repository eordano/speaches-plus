from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from . import task
from .models import ListModelsResponse, Model

def hf_owner(model_id: str) -> str:
    if "/" not in model_id or model_id.startswith(("/", ".")):
        return _stem(model_id) or model_id
    return model_id.split("/", 1)[0]

def _stem(model_id: str) -> str:
    return PurePosixPath(model_id).stem

def filter_by_task(models: list[Model], task_filter: str | None) -> list[Model]:
    if not task_filter:
        return list(models)
    return [m for m in models if m.task == task_filter]

def list_models_response(models: list[Model], task_filter: str | None) -> dict[str, Any]:
    return ListModelsResponse(data=filter_by_task(models, task_filter)).to_dict()
