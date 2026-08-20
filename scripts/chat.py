#!/usr/bin/env python3
"""Interactive multi-turn chat against a running speaches-plus server.

Usage:
    python3 scripts/chat.py                          # defaults to localhost:8000, qwen3.6
    SPEACHES_URL=http://host:18080 python3 scripts/chat.py
    python3 scripts/chat.py --model gemma4 --temp 0

Commands inside the chat:
    /clear          reset conversation history
    /model <name>   switch model mid-conversation
    /system <text>  set system prompt (clears history)
    /temp <float>   change temperature
    /raw            show raw JSON of last response
    /tts <text>     synthesize speech and save to /tmp/tts_last.wav
    /transcribe <f> transcribe a WAV file
    /embed <text>   get text embedding (prints first 8 dims + norm)
"""

import json, os, sys, struct, time

BASE = os.environ.get("SPEACHES_URL", "http://127.0.0.1:8000")
MODEL = "qwen3.6"
TEMP = 0.7
MAX_TOKENS = 512

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

messages = []
system_prompt = None
last_raw = None

def stream_chat(msgs, model, temp):
    global last_raw
    body = {
        "model": model,
        "messages": msgs,
        "max_tokens": MAX_TOKENS,
        "temperature": temp,
        "stream": True,
    }
    resp = requests.post(f"{BASE}/v1/chat/completions",
                         json=body, stream=True, timeout=120)
    resp.raise_for_status()
    try:
        sys.stdout.reconfigure(write_through=True)
    except Exception:
        pass
    full = ""
    chunks = []
    in_think = False
    think_open = False
    hide_thinking = os.environ.get("HIDE_THINKING", "0") == "1"
    first_token_time = None
    delta_count = 0
    completion_tokens = None
    for line in resp.iter_lines(decode_unicode=True, chunk_size=64):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
            chunks.append(chunk)
        except json.JSONDecodeError:
            continue
        usage = chunk.get("usage") or {}
        if usage.get("completion_tokens") is not None:
            completion_tokens = usage["completion_tokens"]
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        if not delta:
            continue
        if first_token_time is None:
            first_token_time = time.perf_counter()
        delta_count += 1
        full += delta
        if hide_thinking:
            visible = []
            i = 0
            while i < len(delta):
                if not in_think and delta[i:].startswith("<think>"):
                    in_think = True
                    i += len("<think>")
                    continue
                if in_think and delta[i:].startswith("</think>"):
                    in_think = False
                    i += len("</think>")
                    continue
                if not in_think:
                    visible.append(delta[i])
                i += 1
            piece = "".join(visible)
            if piece:
                print(piece, end="", flush=True)
        else:
            i = 0
            while i < len(delta):
                if delta[i:].startswith("<think>"):
                    print("\033[90m", end="", flush=True)
                    think_open = True
                    i += len("<think>")
                    continue
                if delta[i:].startswith("</think>"):
                    print("\033[0m", end="", flush=True)
                    think_open = False
                    i += len("</think>")
                    continue
                print(delta[i], end="", flush=True)
                i += 1
    if think_open:
        print("\033[0m", end="", flush=True)
    print()
    last_raw = chunks
    elapsed = (time.perf_counter() - first_token_time) if first_token_time else 0.0
    tokens = completion_tokens if completion_tokens is not None else delta_count
    if elapsed > 0 and tokens > 0:
        print(f"\033[90m[{tokens} tok / {elapsed:.2f}s = {tokens/elapsed:.1f} tok/s]\033[0m")
    return strip_think(full) if hide_thinking else full

def strip_think(text):
    import re
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

def do_tts(text):
    resp = requests.post(f"{BASE}/v1/audio/speech",
                         json={"input": text, "voice": "default"},
                         timeout=300)
    resp.raise_for_status()
    path = "/tmp/tts_last.wav"
    with open(path, "wb") as f:
        f.write(resp.content)
    n = len(resp.content)
    print(f"saved {n} bytes to {path}")

def do_transcribe(path):
    with open(path, "rb") as f:
        resp = requests.post(f"{BASE}/v1/audio/transcriptions",
                             files={"file": f},
                             data={"response_format": "json"},
                             timeout=60)
    resp.raise_for_status()
    print(resp.json().get("text", resp.text))

def do_embed(text):
    resp = requests.post(f"{BASE}/v1/audio/embeddings",
                         json={"input": text, "model": "text-embedding"},
                         timeout=60)
    resp.raise_for_status()
    data = resp.json()
    emb = data.get("data", [{}])[0].get("embedding", [])
    norm = sum(x*x for x in emb) ** 0.5
    dims = emb[:8]
    print(f"dim={len(emb)}  norm={norm:.4f}  first8={[round(x,4) for x in dims]}")

def main():
    global MODEL, TEMP, MAX_TOKENS, BASE, messages, system_prompt

    import argparse
    p = argparse.ArgumentParser(description="Chat with speaches-plus")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--temp", type=float, default=TEMP)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--url", default=BASE)
    args = p.parse_args()

    MODEL = args.model
    TEMP = args.temp
    MAX_TOKENS = args.max_tokens
    BASE = args.url

    print(f"chat: {MODEL} @ {BASE}  (ctrl-d to quit)")
    print(f"commands: /clear /model /system /temp /raw /tts /transcribe /embed")
    print(f"thinking: {'hidden (HIDE_THINKING=1)' if os.environ.get('HIDE_THINKING') == '1' else 'shown in grey'}")
    print()

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not line:
            continue

        if line == "/clear":
            messages.clear()
            print("(cleared)")
            continue
        if line == "/raw":
            print(json.dumps(last_raw, indent=2)[:2000] if last_raw else "(no last response)")
            continue
        if line.startswith("/model "):
            MODEL = line[7:].strip()
            print(f"(model -> {MODEL})")
            continue
        if line.startswith("/system "):
            system_prompt = line[8:].strip()
            messages.clear()
            print(f"(system prompt set, history cleared)")
            continue
        if line.startswith("/temp "):
            try:
                TEMP = float(line[6:].strip())
                print(f"(temp -> {TEMP})")
            except ValueError:
                print("(bad float)")
            continue
        if line.startswith("/tts "):
            try:
                do_tts(line[5:].strip())
            except Exception as e:
                print(f"error: {e}")
            continue
        if line.startswith("/transcribe "):
            try:
                do_transcribe(line[12:].strip())
            except Exception as e:
                print(f"error: {e}")
            continue
        if line.startswith("/embed "):
            try:
                do_embed(line[7:].strip())
            except Exception as e:
                print(f"error: {e}")
            continue

        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend(messages)
        msgs.append({"role": "user", "content": line})

        print("bot> ", end="", flush=True)
        try:
            reply = stream_chat(msgs, MODEL, TEMP)
        except Exception as e:
            print(f"\nerror: {e}")
            continue

        messages.append({"role": "user", "content": line})
        if reply:
            messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
