#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi>=0.115",
#   "uvicorn>=0.30",
# ]
# ///

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

def extract_user_text(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"text", "input_text"} and part.get("text"):
                    parts.append(part["text"])
            if parts:
                return " ".join(parts).strip()
    return ""

def build_app() -> FastAPI:
    app = FastAPI(title="fake-llm")
    state: dict[str, Any] = {
        "response_text": "acknowledged",
        "received": [],
        "fail_status": 0,
        "fail_body": "",
        "delay_ms": 0,
    }
    app.state.fake = state

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/test/state")
    async def get_state() -> dict[str, Any]:
        return {
            "response_text": state["response_text"],
            "received_count": len(state["received"]),
            "received": state["received"],
        }

    @app.post("/test/configure")
    async def configure(req: Request) -> dict[str, Any]:
        body = await req.json()
        if body.get("reset"):
            state["received"].clear()
            state["fail_status"] = 0
            state["fail_body"] = ""
            state["delay_ms"] = 0
        if "response_text" in body:
            state["response_text"] = body["response_text"]
        if "fail_status" in body:
            state["fail_status"] = int(body["fail_status"])
        if "fail_body" in body:
            state["fail_body"] = str(body["fail_body"])
        if "delay_ms" in body:
            state["delay_ms"] = int(body["delay_ms"])
        return {
            "response_text": state["response_text"],
            "fail_status": state["fail_status"],
            "delay_ms": state["delay_ms"],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request) -> Any:
        body = await req.json()
        messages = body.get("messages", [])
        user_text = extract_user_text(messages)
        state["received"].append(
            {
                "ts": time.time(),
                "user_text": user_text,
                "messages": messages,
                "model": body.get("model"),
                "stream": bool(body.get("stream")),
            }
        )
        logging.info(f"fake-llm <- model={body.get('model')!r} user_text={user_text!r}")

        if state["delay_ms"] > 0:
            await asyncio.sleep(state["delay_ms"] / 1000.0)
        if state["fail_status"] > 0:
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(
                state["fail_body"] or f"simulated fail_status={state['fail_status']}",
                status_code=state["fail_status"],
            )

        response_text = state["response_text"]
        chunk_id = "chatcmpl-" + uuid.uuid4().hex[:12]
        created = int(time.time())
        model_name = body.get("model") or "fake-llm"

        if not body.get("stream"):
            return {
                "id": chunk_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        async def generate() -> Any:
            words = response_text.split()
            for i, word in enumerate(words):
                content = word if i == 0 else " " + word
                chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }
                    ],
                }
                if i == 0:
                    chunk["choices"][0]["delta"]["role"] = "assistant"
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.01)

            done = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    return app

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8801)
    p.add_argument("--response-text", default="acknowledged")
    p.add_argument("--log-level", default="warning")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s fake-llm: %(message)s",
        datefmt="%H:%M:%S",
    )

    import uvicorn

    app = build_app()
    app.state.fake["response_text"] = args.response_text
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)

if __name__ == "__main__":
    main()
