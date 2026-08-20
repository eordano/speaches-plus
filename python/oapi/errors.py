from __future__ import annotations

from typing import Any

from fastapi import HTTPException

def openai_error(
    message: str,
    err_type: str,
    param: str | None = None,
    code: str | None = None,
) -> dict[str, dict[str, Any]]:
    return {"error": {"message": message, "type": err_type, "param": param, "code": code}}

def raise_openai_error(
    status_code: int,
    message: str,
    err_type: str,
    param: str | None = None,
    code: str | None = None,
) -> None:
    raise HTTPException(
        status_code=status_code,
        detail=openai_error(message, err_type, param, code),
    )

def missing_field(loc: list[str]) -> dict[str, Any]:
    return {"type": "missing", "loc": loc, "msg": "Field required"}

def fastapi_validation_error(entries: list[dict[str, Any]]) -> HTTPException:
    return HTTPException(status_code=422, detail=entries)

def raise_fastapi_validation_error(entries: list[dict[str, Any]]) -> None:
    raise fastapi_validation_error(entries)
