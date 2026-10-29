#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys

MAX_STATE_CHARS = 16000
LATEST_USER_CHARS = 4000
MAX_EVENT_CHARS = 5000
HISTORY_EVENTS = 48
NONE = "no_tool"
QUESTION = "Which tool, if any, should the assistant call next? Choose no_tool when it should answer without calling a tool."
NESTED_TOOL = re.compile(r"\btools\.([A-Za-z_][A-Za-z_0-9]*)\s*\(")

def text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(text(item.get("text", item.get("content", ""))) if isinstance(item, dict) else text(item) for item in value)
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)

def clipped(value, max_event_chars=MAX_EVENT_CHARS):
    return text(value).strip()[:max_event_chars]

def codex_events(raw, max_event_chars=MAX_EVENT_CHARS):
    for row in raw:
        if row.get("type") != "response_item":
            continue
        payload = row.get("payload", {})
        kind = payload.get("type")
        if kind == "message":
            role = payload.get("role")
            if role == "user":
                yield "user", clipped(payload.get("content"), max_event_chars)
            elif role == "assistant" and (payload.get("phase") == "final_answer" or payload.get("channel") == "final"):
                yield "decision", NONE
            elif role == "assistant":
                message = clipped(payload.get("content"), max_event_chars)
                if message:
                    yield "assistant", message
        elif kind in ("function_call", "custom_tool_call"):
            name = payload.get("name")
            if name:
                arguments = payload.get("arguments", payload.get("input"))
                nested = set(NESTED_TOOL.findall(text(arguments))) if name == "exec" else set()
                chosen = next(iter(nested)) if len(nested) == 1 else name if not nested else None
                if chosen:
                    yield "decision", chosen
                yield "tool", f"tool {name}: {clipped(payload.get('arguments', payload.get('input')), max_event_chars)}"
        elif kind in ("function_call_output", "custom_tool_call_output"):
            yield "result", clipped(payload.get("output"), max_event_chars)

def claude_events(raw, max_event_chars=MAX_EVENT_CHARS):
    for row in raw:
        if row.get("isSidechain"):
            continue
        kind = row.get("type")
        content = row.get("message", {}).get("content", [])
        if kind == "user":
            if isinstance(content, str):
                yield "user", clipped(content, max_event_chars)
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "tool_result":
                        yield "result", clipped(item.get("content"), max_event_chars)
                    elif item.get("type") == "text":
                        yield "user", clipped(item.get("text"), max_event_chars)
        elif kind == "assistant" and isinstance(content, list):
            calls = [item for item in content if item.get("type") == "tool_use" and item.get("name")]
            if len(calls) == 1:
                yield "decision", calls[0]["name"]
            elif not calls and row.get("message", {}).get("stop_reason") == "end_turn":
                yield "decision", NONE
            for item in content:
                if item.get("type") == "text":
                    message = clipped(item.get("text"), max_event_chars)
                    if message:
                        yield "assistant", message
                elif item.get("type") == "tool_use" and item.get("name"):
                    yield "tool", f"tool {item['name']}: {clipped(item.get('input'), max_event_chars)}"

def source_files(roots):
    for root in roots:
        root = Path(root)
        if root.is_file():
            yield root.parent, root
        else:
            for path in sorted(root.rglob("*.jsonl")):
                yield root, path

def agent_for(path):
    parts = path.parts
    if "mirror-codex" in parts or "sessions" in parts or "archived_sessions" in parts:
        return "codex"
    if "mirror-claude" in parts or "projects" in parts:
        return "claude"
    return None

def examples(roots, max_state_chars=MAX_STATE_CHARS, latest_user_chars=LATEST_USER_CHARS, max_event_chars=MAX_EVENT_CHARS, history_events=HISTORY_EVENTS):
    seen = set()
    for root, path in source_files(roots):
        agent = agent_for(path)
        if agent is None or path in seen:
            continue
        seen.add(path)
        history = []
        file_examples = []
        try:
            with path.open() as stream:
                raw = (json.loads(line) for line in stream if line.strip())
                events = codex_events(raw, max_event_chars) if agent == "codex" else claude_events(raw, max_event_chars)
                for kind, value in events:
                    if kind == "decision":
                        users = [item for item in history if item.startswith("user: ")]
                        if users:
                            latest = users[-1][-latest_user_chars:]
                            recent = "\n".join(history)[-(max_state_chars - len(latest)):]
                            state = f"agent: {agent}\nlatest {latest}\nrecent:\n{recent}"
                            file_examples.append((state, value))
                    elif value:
                        history.append(f"{kind}: {value}")
                        if len(history) > history_events:
                            history.pop(0)
        except (OSError, ValueError) as exc:
            print(f"skip {path}: {exc}", file=sys.stderr)
            continue
        if file_examples:
            session = hashlib.sha256(f"{agent}/{path.relative_to(root)}".encode()).hexdigest()[:16]
            yield agent, session, file_examples

def dataset(roots, out, eval_percent, format="laya", train_limit=None, max_state_chars=MAX_STATE_CHARS, latest_user_chars=LATEST_USER_CHARS, max_event_chars=MAX_EVENT_CHARS, history_events=HISTORY_EVENTS):
    tools = {agent: set() for agent in ("claude", "codex")}
    sessions = 0
    for agent, _, pairs in examples(roots, max_state_chars, latest_user_chars, max_event_chars, history_events):
        tools[agent].update(label for _, label in pairs if label != NONE)
        sessions += 1
    tools = {agent: sorted(names) for agent, names in tools.items()}
    if not any(tools.values()):
        raise ValueError("no tool decisions found in cccp logs")
    out.mkdir(parents=True, exist_ok=True)
    unique = {}
    raw_counts = {"train": 0, "eval": 0}
    for agent, session, pairs in examples(roots, max_state_chars, latest_user_chars, max_event_chars, history_events):
        options = {NONE: "Answer without a tool call"} | {name: None for name in tools[agent]}
        if len(options) < 2:
            continue
        split = "eval" if int(session[:8], 16) % 100 < eval_percent else "train"
        for state, label in pairs:
            raw_counts[split] += 1
            if format == "kev":
                questions = {"tool": {"type": "choice", "instructions": QUESTION, "criteria": options, "label": label}}
            else:
                questions = {"call_tool": {"type": "noul", "instructions": "Should the assistant call a tool next?", "answer": label != NONE}, "tool": {"type": "choice", "instructions": QUESTION, "criteria": options, "answer": label}}
            line = json.dumps({"state": state, "questions": questions}, ensure_ascii=False) + "\n"
            prior = unique.get(state)
            if prior is None and state not in unique:
                unique[state] = (line, label, split)
            elif prior is not None:
                if prior[1] != label:
                    unique[state] = None
                elif split == "eval":
                    unique[state] = (line, label, "eval")
    counts = {"train": 0, "eval": 0}
    train_seen = 0
    train_rows = []
    rng = random.Random(0)
    with (out / "train.jsonl").open("w") as train, (out / "eval.jsonl").open("w") as evaluation:
        for record in unique.values():
            if record is None:
                continue
            line, _, split = record
            if split == "eval":
                evaluation.write(line)
                counts["eval"] += 1
            elif train_limit is None:
                train.write(line)
                counts["train"] += 1
            else:
                train_seen += 1
                if len(train_rows) < train_limit:
                    train_rows.append(line)
                else:
                    index = rng.randrange(train_seen)
                    if index < train_limit:
                        train_rows[index] = line
        if train_limit is not None:
            train.writelines(train_rows)
            counts["train"] = len(train_rows)
    (out / "manifest.json").write_text(json.dumps({"counts": counts, "raw_counts": raw_counts, "conflicted_states": sum(value is None for value in unique.values()), "train_available": train_seen if train_limit is not None else counts["train"], "format": format, "tools": tools, "eval_percent": eval_percent, "sessions": sessions, "context": {"max_state_chars": max_state_chars, "latest_user_chars": latest_user_chars, "max_event_chars": max_event_chars, "history_events": history_events}}, indent=2) + "\n")
    if min(counts.values()) == 0:
        raise ValueError(f"need both training and held-out sessions; got {counts}")
    return counts

def main():
    parser = argparse.ArgumentParser(description="Build and train Laya System One tool decisions from cccp mirrors")
    parser.add_argument("--claude", type=Path, help="cccp mirror-claude/projects root")
    parser.add_argument("--codex", type=Path, help="cccp mirror-codex/sessions root")
    parser.add_argument("--prepared", type=Path, help="existing directory with train.jsonl and eval.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-percent", type=int, default=20)
    parser.add_argument("--format", choices=("laya", "kev"), default="laya")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--max-state-chars", type=int, default=MAX_STATE_CHARS)
    parser.add_argument("--latest-user-chars", type=int, default=LATEST_USER_CHARS)
    parser.add_argument("--max-event-chars", type=int, default=MAX_EVENT_CHARS)
    parser.add_argument("--history-events", type=int, default=HISTORY_EVENTS)
    parser.add_argument("--base", type=Path, help="Laya checkpoint; enables training")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.eval_percent <= 50:
        parser.error("--eval-percent must be between 1 and 50")
    if args.train_limit is not None and args.train_limit < 1:
        parser.error("--train-limit must be positive")
    if not 0 < args.latest_user_chars < args.max_state_chars or min(args.max_event_chars, args.history_events) < 1:
        parser.error("context limits must be positive and latest-user-chars must be smaller than max-state-chars")
    if args.format == "kev" and args.base:
        parser.error("--base is for Laya training; train Kev-format data with kev.train")
    if args.prepared and (args.claude or args.codex or not args.base):
        parser.error("--prepared requires --base and cannot be combined with mirror roots")
    if not args.prepared and not (args.claude and args.codex):
        parser.error("provide both --claude and --codex, or --prepared with --base")
    os.umask(0o077)
    data_dir = args.prepared or args.out
    if args.prepared:
        for name in ("train.jsonl", "eval.jsonl", "manifest.json"):
            if not (data_dir / name).is_file():
                parser.error(f"--prepared is missing {name}")
        args.out.mkdir(parents=True, exist_ok=True)
    else:
        counts = dataset([args.claude, args.codex], args.out, args.eval_percent, args.format, args.train_limit, args.max_state_chars, args.latest_user_chars, args.max_event_chars, args.history_events)
        print(json.dumps(counts))
    if args.base:
        root = Path(__file__).resolve().parents[1]
        command = [str(root / "rust/scripts/nvk.sh"), "run", "--release", "--bin", "nvk-laya-lora-train", "--", "--base", str(args.base.resolve()), "--data", str((data_dir / "train.jsonl").resolve()), "--eval-data", str((data_dir / "eval.jsonl").resolve()), "--out", str((args.out / "adapter").resolve()), "--merged-out", str((args.out / "merged").resolve()), "--steps", str(args.steps), "--batch", str(args.batch)]
        env = os.environ | {"NVK_PKG": "nv-train", "NVK_FEATURES": os.environ.get("NVK_FEATURES", "cuda")}
        with (args.out / "training.log").open("w") as log:
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"training complete: {args.out / 'training.log'}")

if __name__ == "__main__":
    main()
