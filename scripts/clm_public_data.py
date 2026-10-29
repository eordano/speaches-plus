"""Download public training splits as labeled System One decisions for clm_prepare."""

import argparse
import json
import random
import re
import fcntl

NONE = "no_tool"
NONE_DESC = "Answer without a tool call"
INSTRUCTIONS = ("Which tool, if any, should the assistant call next? "
                "Choose no_tool when it should answer without calling a tool.")
DESC_CHARS = 512
TURN_CHARS = 32000
HISTORY_TURNS = 48
NAME = re.compile(r'"name"\s*:\s*"([^"]+)"')

def short(desc):
    desc = " ".join(str(desc or "").split())
    return re.split(r"(?<=[.:;])\s", desc, maxsplit=1)[0][:DESC_CHARS].rstrip(" .") or None

def clip(s):
    s = " ".join(str(s).split())
    return s if len(s) <= TURN_CHARS else s[:TURN_CHARS - 1] + "…"

def tool_map(tools):
    out = {}
    for t in tools:
        if isinstance(t, str):
            try:
                t = json.loads(t)
            except ValueError:
                continue
        if isinstance(t, dict) and "function" in t and isinstance(t["function"], dict):
            t = t["function"]
        if isinstance(t, dict) and t.get("name"):
            out[str(t["name"])] = short(t.get("description"))
    return out

def json_objects(text):
    dec, i, out = json.JSONDecoder(), 0, []
    while True:
        j = min([k for k in (text.find("{", i), text.find("[", i)) if k >= 0], default=-1)
        if j < 0:
            return out
        try:
            obj, end = dec.raw_decode(text, j)
        except ValueError:
            i = j + 1
            continue
        out.append(obj)
        i = end

def record(source, history, tools, label):
    if not tools or (label != NONE and label not in tools):
        return None
    lines = [f"{who}: {clip(text)}" for who, text in history[-(2 * HISTORY_TURNS):]]
    return {"state": "\n".join(lines), "_meta": {"source": source},
            "questions": {"tool": {"type": "choice", "instructions": INSTRUCTIONS,
                                   "criteria": {NONE: NONE_DESC, **tools}, "label": label}}}

def when2call(ds):
    for r in ds:
        tools = tool_map(r["tools"])
        msgs = r["messages"]
        history = []
        for i, m in enumerate(msgs):
            if m["role"] == "user" and i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
                reply = msgs[i + 1]["content"] or ""
                marked = "TOOLCALL" in reply or reply.lstrip().startswith(("[{", "{"))
                call = NAME.search(reply) if marked else None
                history.append(("User", m["content"]))
                yield record("when2call", history, tools, call.group(1) if call else (None if marked else NONE))
                history.append(("Assistant", reply))

def when2call_mcq(ds):
    for r in ds:
        tools = tool_map(r["tools"])
        label = NONE
        if r["correct_answer"] == "tool_call":
            target = NAME.search(str(r.get("target_tool") or "")) or NAME.search(
                str((r.get("answers") or {}).get("tool_call") or ""))
            label = target.group(1) if target else None
        if label:
            yield record("when2call-mcq", [("User", r["question"])], tools, label)

def xlam_irrelevance(ds):
    for r in ds:
        tools = tool_map(json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"])
        answers = json.loads(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
        label = answers[0]["name"] if answers else NONE
        yield record("xlam-irr", [("User", r["query"])], tools, label)

def toolace(ds):
    for r in ds:
        tools = {}
        for obj in json_objects(r["system"].split("invoke:", 1)[-1]):
            tools.update(tool_map(obj if isinstance(obj, list) else [obj]))
        conv, history = r["conversations"], []
        for i, m in enumerate(conv):
            if m["from"] == "user":
                history.append(("User", m["value"]))
                nxt = conv[i + 1] if i + 1 < len(conv) else None
                if nxt and nxt["from"] == "assistant":
                    call = re.match(r"\s*\[\s*([^\(\[\]]+?)\s*\(", nxt["value"] or "")
                    yield record("toolace", history, tools, call.group(1) if call else NONE)
            elif m["from"] == "assistant":
                call = re.match(r"\s*\[\s*([^\(\[\]]+?)\s*\(", m["value"] or "")
                history.append(("Assistant", f"(called {call.group(1)})" if call else m["value"]))
            elif m["from"] in {"tool", "function"}:
                history.append(("Tool", m["value"]))

def glaive(ds):
    for r in ds:
        tools = {}
        for obj in json_objects(r["system"]):
            tools.update(tool_map([obj] if isinstance(obj, dict) else obj))
        if not tools:
            continue
        parts = re.split(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)", r["chat"])
        turns = [(parts[i].rstrip(":"), parts[i + 1].replace("<|endoftext|>", "").strip())
                 for i in range(1, len(parts) - 1, 2)]
        history = []
        for i, (who, text) in enumerate(turns):
            if who == "USER":
                history.append(("User", text))
                if i + 1 < len(turns) and turns[i + 1][0] == "ASSISTANT":
                    reply = turns[i + 1][1]
                    call = NAME.search(reply) if "<functioncall>" in reply else None
                    yield record("glaive", history, tools, call.group(1) if call else (None if "<functioncall>" in reply else NONE))
            elif who == "ASSISTANT":
                call = NAME.search(text) if "<functioncall>" in text else None
                history.append(("Assistant", f"(called {call.group(1)})" if call else text))
            elif who == "FUNCTION RESPONSE":
                history.append(("Tool", text))

def hermes(ds):
    for r in ds:
        tools = tool_map(json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"])
        conv = r["conversations"]
        for i, m in enumerate(conv):
            if m["from"] == "human" and i + 1 < len(conv) and conv[i + 1]["from"] == "gpt":
                reply = conv[i + 1]["value"] or ""
                call = NAME.search(reply.split("<tool_call>", 1)[1]) if "<tool_call>" in reply else None
                yield record("hermes", [("User", m["value"])], tools, call.group(1) if call else (None if "<tool_call>" in reply else NONE))
                break

SOURCES = {
    "when2call": ("nvidia/When2Call", "train_sft", "train", when2call),
    "when2call-mcq": ("nvidia/When2Call", "test", "mcq", when2call_mcq),
    "xlam-irr": ("MadeAgents/xlam-irrelevance-7.5k", None, "train", xlam_irrelevance),
    "toolace": ("Team-ACE/ToolACE", None, "train", toolace),
    "glaive": ("glaiveai/glaive-function-calling-v2", None, "train", glaive),
    "hermes": ("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train", hermes),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory for <source>.jsonl")
    ap.add_argument("--cap", default="when2call=5000,xlam-irr=7500,toolace=20000,glaive=70000,hermes=10000",
                    help="records kept per training source, sampled by --seed; when2call-mcq is kept whole")
    ap.add_argument("--only", help="comma-separated subset of " + ",".join(SOURCES))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from collections import Counter
    import hashlib
    from pathlib import Path
    from datasets import load_dataset
    from huggingface_hub import HfApi
    Path(a.out).mkdir(parents=True, exist_ok=True)
    lock = open(Path(a.out, '.download.lock'), 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if Path(a.out, 'complete.json').exists():
        print('public dataset conversion already complete', flush=True)
        return
    manifest = {}
    caps = {k: int(v) for k, v in (x.split("=") for x in a.cap.split(",") if x)}
    for name, (repo, config, split, parse) in SOURCES.items():
        if a.only and name not in a.only.split(","):
            continue
        if name == "when2call-mcq":
            continue
        revision = HfApi().dataset_info(repo).sha
        ds = load_dataset(repo, config, split=split, revision=revision)
        recs, seen = [], set()
        rejected = 0
        for original in ds:
            group = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
            for r in parse([original]):
                key = r and (r["state"], json.dumps(r["questions"]["tool"]["criteria"], sort_keys=True))
                if r and key not in seen:
                    seen.add(key)
                    r["_meta"].update(group=group, dataset=repo, revision=revision, split=split)
                    recs.append(r)
                else:
                    rejected += 1
        random.Random(a.seed).shuffle(recs)
        if name in caps:
            recs = recs[:caps[name]]
        with open(f"{a.out}/{name}.jsonl", "w") as o:
            for r in recs:
                o.write(json.dumps(r, ensure_ascii=False) + "\n")
        labels = Counter(r["questions"]["tool"]["label"] == NONE for r in recs)
        manifest[name] = {"repository": repo, "revision": revision, "split": split,
                          "available_rows": len(ds), "decisions": len(recs), "rejected_or_duplicate": rejected,
                          "none": labels[True], "tools": labels[False]}
        Path(a.out, "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{name}: {len(recs)} records, no_tool {labels[True]}, tool {labels[False]}", flush=True)
    Path(a.out, 'complete.json').write_text(json.dumps(manifest, indent=2) + '\n')

if __name__ == "__main__":
    main()
