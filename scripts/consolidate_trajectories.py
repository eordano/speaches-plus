#!/usr/bin/env python3
"""Distill served trajectories into a LoRA adapter, gate it, and roll it out.

One consolidation round: read NV_TRAJECTORY_DIR/trajectories.jsonl, keep rows
produced by the teacher model (--model, required) that finished naturally and
carry no reasoning split, no stop-sequence truncation, no multimodal prompt and
no "bad" label, optionally judge-score the survivors with a local model,
dedupe by whitespace/case-normalized content fingerprint, draw a leakage-free
holdout, train via POST /v1/fine_tuning/jobs?wait=true, and decide the
roll-over from paired per-example CE. The independence unit is the merged
prompt group: rows sharing a normalized prompt, plus multi-turn continuations
whose rendered prompt extends another row's prompt verbatim, form one held-out
window, never straddle the train/holdout split, and contribute one delta (the
mean over their rows) to the paired verdict. Rows whose prompt leaves the
512-token eval window no scorable completion (prompt_tokens at the cap, or
prompt bytes at the one-byte-per-token floor when prompt_tokens is absent)
train but never sit in the holdout, so the trainer's loader cannot silently
shrink the validation set out from under the pairing. A verdict needs >= 9
independent held-out windows (docs/book/08.2-quality-verdicts.md; fewer
cannot adjudicate and the round exits 3 UNRESOLVED without training), a mean
window delta under -margin and a paired t at or past -2. --null-check
additionally trains a same-shape adapter on label-shuffled rows through the
identical harness and requires the arm to beat its own null by an arm/null
|t| ratio; a ratio near 1 means the instrument cannot resolve this contrast
at any corpus size. On PASS the adapter is promoted and every older adapter
of the same suffix -- including any orphaned -nullcheck calibration adapter
-- unloads so exactly one rolling adapter serves. A server without runtime
adapter routes still gets an honest exit: the unmounted-load path reports
UNRESOLVED (3) when the round is unresolved instead of claiming a pass.
Adapter dirs stay on disk as the rollback artifacts. Exit codes: 0 pass,
1 error, 2 gate refused, 3 unresolved.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
import urllib.error
import urllib.request

HOLDOUT_NEEDS_NINE_WINDOWS_BECAUSE_08_2_SAYS_FEWER_INDEPENDENT_WINDOWS_CANNOT_ADJUDICATE = 9
PAIRED_T_MUST_REACH_2_BECAUSE_08_2_CALLS_SMALLER_PAIRED_MEANS_UNRESOLVED = 2.0
EVAL_ROWS_TRUNCATE_AT_512_TOKENS_SO_512_IS_THE_VERDICTS_N_CTX = 512
PROMPT_TOKEN_SLACK_OF_8_COVERS_SERVING_VS_TRAINING_TOKENIZER_DRIFT_SO_A_COMPLETION_TOKEN_STAYS_SCORABLE = 8

def read_trajectories(path):
    files = (
        sorted(glob.glob(os.path.join(path, "trajectories*.jsonl")))
        if os.path.isdir(path)
        else [path]
    )
    rows = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"skipping unparseable {fp}:{n}: {e}", file=sys.stderr)
    if not files:
        print(f"no trajectory files under {path}", file=sys.stderr)
    return rows

def read_label_patches(path):
    if not os.path.isdir(path):
        return {}
    patches = {}
    for fp in sorted(glob.glob(os.path.join(path, "labels-*.jsonl"))):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not p.get("label"):
                    continue
                for key in (p.get("id"), p.get("tag")):
                    if key:
                        patches[key] = p["label"]
    return patches

def whitespace_and_case_normalized(text):
    return " ".join(text.split()).casefold()

def content_fingerprint(prompt, completion):
    return hashlib.sha256(
        (
            whitespace_and_case_normalized(prompt)
            + "\x00"
            + whitespace_and_case_normalized(completion)
        ).encode()
    ).hexdigest()

def select_rows(rows, label_patches, args):
    seen = set()
    out = []
    dropped = {}

    def drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    for r in rows:
        model = r.get("model") or ""
        if model.startswith("ft:"):
            drop("adapter-produced")
            continue
        if model != args.model:
            drop("other-model")
            continue
        if r.get("finish_reason") != "stop":
            drop("finish!=stop")
            continue
        if r.get("stopped_by"):
            drop("stop-sequence-truncated")
            continue
        if r.get("multimodal"):
            drop("multimodal")
            continue
        if r.get("reasoning") and not args.keep_reasoning_rows:
            drop("reasoning-split")
            continue
        label = (
            label_patches.get(r.get("id"))
            or label_patches.get(r.get("tag"))
            or r.get("label")
        )
        if label == "bad":
            drop("labeled-bad")
            continue
        if args.require_label and label != "good":
            drop("unlabeled")
            continue
        prompt = r.get("prompt") or ""
        completion = r.get("completion") or ""
        if len(completion) < args.min_completion_chars:
            drop("too-short")
            continue
        key = content_fingerprint(prompt, completion)
        if key in seen:
            drop("duplicate-after-normalization")
            continue
        seen.add(key)
        out.append(
            {
                "prompt": prompt,
                "completion": completion,
                "prompt_tokens": r.get("prompt_tokens"),
            }
        )
    for reason, n in sorted(dropped.items()):
        print(f"  dropped {n:5d}  {reason}")
    return out[-args.max_records:]

def payload_rows(rows):
    return [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]

def holdout_eligible_because_the_eval_window_keeps_a_completion_token_scorable(row):
    cap = EVAL_ROWS_TRUNCATE_AT_512_TOKENS_SO_512_IS_THE_VERDICTS_N_CTX
    pt = row.get("prompt_tokens")
    if isinstance(pt, (int, float)) and not isinstance(pt, bool):
        return (
            pt + PROMPT_TOKEN_SLACK_OF_8_COVERS_SERVING_VS_TRAINING_TOKENIZER_DRIFT_SO_A_COMPLETION_TOKEN_STAYS_SCORABLE
            < cap
        )
    prompt_bytes_bound_tokens_because_every_token_spans_at_least_one_byte = len(
        row["prompt"].encode()
    )
    return prompt_bytes_bound_tokens_because_every_token_spans_at_least_one_byte < cap

def merged_prompt_groups_because_multiturn_continuations_embed_their_history(data):
    rows_by_pf, norm_by_pf = {}, {}
    for row in data:
        norm = whitespace_and_case_normalized(row["prompt"])
        pf = hashlib.sha256(norm.encode()).hexdigest()
        rows_by_pf.setdefault(pf, []).append(row)
        norm_by_pf[pf] = norm
    parent = {pf: pf for pf in rows_by_pf}

    def find(pf):
        while parent[pf] != pf:
            parent[pf] = parent[parent[pf]]
            pf = parent[pf]
        return pf

    open_prefixes = []
    for pf in sorted(rows_by_pf, key=norm_by_pf.get):
        norm = norm_by_pf[pf]
        if not norm:
            continue
        while open_prefixes and not norm.startswith(norm_by_pf[open_prefixes[-1]]):
            open_prefixes.pop()
        if open_prefixes:
            parent[find(pf)] = find(open_prefixes[-1])
        open_prefixes.append(pf)

    members = {}
    for pf in rows_by_pf:
        members.setdefault(find(pf), []).append(pf)
    groups = {}
    for pfs in members.values():
        rows = []
        for pf in sorted(pfs):
            rows.extend(rows_by_pf[pf])
        groups[min(pfs)] = rows
    return groups

def split_holdout_without_prompt_leakage(data, holdout_frac, seed):
    groups = merged_prompt_groups_because_multiturn_continuations_embed_their_history(data)
    keyed = sorted(
        groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()
    )
    min_windows = HOLDOUT_NEEDS_NINE_WINDOWS_BECAUSE_08_2_SAYS_FEWER_INDEPENDENT_WINDOWS_CANNOT_ADJUDICATE
    target_rows = min(
        len(data) - 1,
        max(min_windows, int(len(data) * holdout_frac)),
    )
    holdout, train, holdout_group_sizes = [], [], []
    for g in keyed:
        rows = groups[g]
        wants_more = len(holdout) < target_rows or len(holdout_group_sizes) < min_windows
        if wants_more and all(
            holdout_eligible_because_the_eval_window_keeps_a_completion_token_scorable(r)
            for r in rows
        ):
            holdout.extend(rows)
            holdout_group_sizes.append(len(rows))
        else:
            train.extend(rows)
    digest = hashlib.sha256(json.dumps([str(seed), keyed]).encode()).hexdigest()[:16]
    return holdout, train, digest, holdout_group_sizes

def paired_delta_stats(ce_adapter, ce_base, group_sizes=None):
    if not ce_adapter or len(ce_adapter) != len(ce_base):
        raise ValueError(
            f"per-example CE vectors unusable: adapter has {len(ce_adapter)} rows, base {len(ce_base)}"
        )
    if not all(math.isfinite(x) for x in ce_adapter + ce_base):
        raise ValueError("per-example CE vectors carry a non-finite value")
    row_deltas = [a - b for a, b in zip(ce_adapter, ce_base)]
    if group_sizes is None:
        group_sizes = [1] * len(row_deltas)
    if sum(group_sizes) != len(row_deltas):
        raise ValueError(
            f"holdout group sizes cover {sum(group_sizes)} rows but the eval scored {len(row_deltas)}"
        )
    deltas, start = [], 0
    for size in group_sizes:
        deltas.append(sum(row_deltas[start : start + size]) / size)
        start += size
    n = len(deltas)
    mean = sum(deltas) / n
    se = (
        math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1) / n)
        if n > 1
        else math.inf
    )
    if se == 0:
        t = 0.0 if mean == 0 else math.copysign(math.inf, mean)
    else:
        t = mean / se
    wins = sum(1 for d in deltas if d < 0)
    losses = sum(1 for d in deltas if d > 0)
    return {
        "n": n,
        "mean_delta": mean,
        "se": se,
        "t": t,
        "wins": wins,
        "losses": losses,
        "ties": n - wins - losses,
    }

def paired_eval_vectors(metrics, expected_n):
    ce_a, ce_b = metrics.get("eval_ce_adapter"), metrics.get("eval_ce_base")
    if not isinstance(ce_a, list) or not isinstance(ce_b, list):
        raise ValueError(
            "job metrics carry no per-example eval_ce_adapter/eval_ce_base vectors"
        )
    if len(ce_a) != expected_n or len(ce_b) != expected_n:
        raise ValueError(
            f"per-example CE length adapter={len(ce_a)} base={len(ce_b)} != {expected_n} held-out rows"
        )
    return [float(x) for x in ce_a], [float(x) for x in ce_b]

def paired_gate_decision(stats, gate_margin):
    return (
        stats["mean_delta"] < -gate_margin
        and stats["t"] <= -PAIRED_T_MUST_REACH_2_BECAUSE_08_2_CALLS_SMALLER_PAIRED_MEANS_UNRESOLVED
    )

def verdict_line(kind, arm, stats, gate_margin, extra=""):
    return (
        f"verdict: {kind} arm={arm} class=lora-adapter "
        f"n={stats['n']} paired held-out windows "
        f"n_ctx<={EVAL_ROWS_TRUNCATE_AT_512_TOKENS_SO_512_IS_THE_VERDICTS_N_CTX} "
        f"mean_delta={stats['mean_delta']:+.4f} se={stats['se']:.4f} t={stats['t']:+.2f} "
        f"signs={stats['wins']}-{stats['losses']}-{stats['ties']} "
        f"margin={gate_margin}{extra}"
    )

def unresolved_exit(ft_model, reason):
    print(
        f"UNRESOLVED: no resolving held-out verdict for {ft_model} ({reason}); "
        f"no promote, no older adapter unloaded, the previously gated boot "
        f"default stands",
        file=sys.stderr,
    )
    return 3

def label_shuffled_null_rows(train, seed):
    if len(train) < 2:
        raise ValueError("label shuffle needs at least 2 train rows")
    order = list(range(len(train)))
    random.Random(f"null:{seed}").shuffle(order)
    n = len(order)
    return [
        {
            "prompt": train[order[i]]["prompt"],
            "completion": train[order[(i + 1) % n]]["completion"],
        }
        for i in range(n)
    ]

def http_json(url, payload=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body

def chat_once(server, model, prompt_text, timeout):
    body = http_json(
        f"{server}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": 128,
        },
        timeout,
    )
    return body["choices"][0]["message"]["content"]

JUDGE_PROMPT = (
    "You are grading one AI response for use as fine-tuning data. Reply with ONLY an "
    "integer 1-10, where 10 means correct, helpful and well-formed, and 1 means wrong, "
    "harmful or degenerate.\n\nResponse to grade:\n{completion}"
)

def judge_score(server, judge_model, completion, timeout):
    reply = chat_once(
        server, judge_model, JUDGE_PROMPT.format(completion=completion[:4000]), timeout
    )
    m = re.search(r"\d+", reply or "")
    return int(m.group()) if m else None

def judge_filter(data, args):
    kept = []
    for i, row in enumerate(data):
        try:
            score = judge_score(args.server, args.judge_model, row["completion"], args.timeout)
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError) as e:
            print(f"judge call failed on row {i}: {e}", file=sys.stderr)
            score = None
        if score is not None and score >= args.judge_threshold:
            kept.append(row)
    print(f"judge kept {len(kept)}/{len(data)} rows (threshold {args.judge_threshold})")
    return kept

def list_models(server, timeout):
    body = http_json(f"{server}/v1/models", timeout=timeout)
    return [m.get("id", "") for m in body.get("data", [])]

def unload_adapter(server, name, timeout):
    return http_json(
        f"{server}/v1/unload_lora_adapter", {"lora_name": name}, timeout
    )

def train_job(server, payload, timeout):
    job = http_json(f"{server}/v1/fine_tuning/jobs?wait=true", payload, timeout)
    if not isinstance(job, dict) or job.get("status") != "succeeded":
        raise RuntimeError(f"fine-tuning job did not succeed: {job}")
    return job

def null_calibration_stats(args, train, holdout, hyperparameters, holdout_group_sizes):
    payload = {
        "model": args.base,
        "training_data": label_shuffled_null_rows(train, args.split_seed),
        "validation_data": payload_rows(holdout),
        "suffix": f"{args.suffix}-nullcheck",
        "hyperparameters": hyperparameters,
    }
    job = train_job(args.server, payload, args.timeout)
    metrics = job.get("metrics") or {}
    ce_a, ce_b = paired_eval_vectors(metrics, len(holdout))
    null_model = job.get("fine_tuned_model")
    if null_model:
        try:
            unload_adapter(args.server, null_model, args.timeout)
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"null adapter {null_model} unload failed (kept loaded): {e}", file=sys.stderr)
    return paired_delta_stats(ce_a, ce_b, holdout_group_sizes)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--trajectories",
        required=True,
        help="NV_TRAJECTORY_DIR (daily trajectories-*.jsonl files) or one jsonl file",
    )
    ap.add_argument("--base", required=True, help="base model dir the server also serves")
    ap.add_argument(
        "--model",
        required=True,
        help="teacher: only rows this served model produced become training data",
    )
    ap.add_argument("--server", default="http://127.0.0.1:1327")
    ap.add_argument("--min-completion-chars", type=int, default=16)
    ap.add_argument("--max-records", type=int, default=512)
    ap.add_argument("--keep-reasoning-rows", action="store_true")
    ap.add_argument("--require-label", action="store_true")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-threshold", type=int, default=7)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument(
        "--split-seed",
        default="holdout-v1",
        help="keys the deterministic prompt-grouped holdout ordering; change it to redraw the split",
    )
    ap.add_argument(
        "--gate-margin",
        type=float,
        default=0.0,
        help="the mean per-window paired CE delta (adapter minus base) must undercut zero by this much",
    )
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument(
        "--null-check",
        action="store_true",
        help="after a gate pass, train a same-shape adapter on label-shuffled rows through the "
        "identical harness and require arm/null |t| >= --null-ratio-min; a ratio near 1 means "
        "the instrument cannot resolve this contrast at any corpus size (08.2 sect 2d), so the "
        "round exits 3 UNRESOLVED instead of promoting",
    )
    ap.add_argument(
        "--null-ratio-min",
        type=float,
        default=2.0,
        help="minimum arm-to-null |t| ratio for the paired verdict to count as resolved",
    )
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--suffix", default="traj")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--eval-prompt", action="append", default=[])
    args = ap.parse_args(argv)

    rows = read_trajectories(args.trajectories)
    label_patches = read_label_patches(args.trajectories)
    if label_patches:
        print(f"{len(label_patches)} retro label patches (thumbs) joined by response id")
    data = select_rows(rows, label_patches, args)
    print(f"{len(rows)} trajectory records -> {len(data)} training rows after filtering")
    if args.judge_model and data and not args.dry_run:
        data = judge_filter(data, args)
    if not data:
        print("nothing to train on", file=sys.stderr)
        return 1
    if args.dry_run:
        print(json.dumps(data[0], indent=2)[:2000])
        return 0

    min_windows = HOLDOUT_NEEDS_NINE_WINDOWS_BECAUSE_08_2_SAYS_FEWER_INDEPENDENT_WINDOWS_CANNOT_ADJUDICATE
    if len(data) > min_windows:
        holdout, train, split_digest, holdout_group_sizes = (
            split_holdout_without_prompt_leakage(data, args.holdout_frac, args.split_seed)
        )
    else:
        holdout, train, split_digest, holdout_group_sizes = [], data, "no-split", []
    holdout_ineligible = sum(
        1
        for r in data
        if not holdout_eligible_because_the_eval_window_keeps_a_completion_token_scorable(r)
    )
    if holdout_ineligible:
        print(
            f"{holdout_ineligible} rows train-only: their prompt leaves the "
            f"{EVAL_ROWS_TRUNCATE_AT_512_TOKENS_SO_512_IS_THE_VERDICTS_N_CTX}-token "
            f"eval window no scorable completion"
        )
    if not args.no_gate and len(holdout_group_sizes) < min_windows:
        print(
            f"UNRESOLVED: {len(data)} deduped rows seat only {len(holdout_group_sizes)} "
            f"independent held-out windows ({len(holdout)} rows), under the {min_windows}-window "
            f"minimum (08.2: fewer independent windows cannot adjudicate); no training run, "
            f"no promote, no roll-over",
            file=sys.stderr,
        )
        return 3
    if not train:
        print(
            f"UNRESOLVED: every kept row shares one merged prompt group, so no "
            f"leakage-free train/holdout split exists; no training run, no promote, no roll-over",
            file=sys.stderr,
        )
        return 3
    print(
        f"{len(train)} train rows, {len(holdout)} held-out rows in "
        f"{len(holdout_group_sizes)} independent windows (split seed {args.split_seed!r}, "
        f"digest {split_digest}, prompt groups and their continuations never straddle the split)"
    )

    hyperparameters = {"rank": args.rank, "steps": args.steps, "lr": args.lr}
    payload = {
        "model": args.base,
        "training_data": payload_rows(train),
        "suffix": args.suffix,
        "hyperparameters": hyperparameters,
    }
    if holdout:
        payload["validation_data"] = payload_rows(holdout)
    try:
        job = train_job(args.server, payload, args.timeout)
    except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as e:
        print(f"fine-tuning request failed: {e}", file=sys.stderr)
        return 1
    print(json.dumps(job, indent=2))

    ft_model = job.get("fine_tuned_model")
    metrics = job.get("metrics") or {}
    if not ft_model:
        print("job succeeded but reported no fine_tuned_model", file=sys.stderr)
        return 1

    gate_passed = False
    unresolved_reason = None
    if holdout and not args.no_gate:
        try:
            ce_a, ce_b = paired_eval_vectors(metrics, len(holdout))
            stats = paired_delta_stats(ce_a, ce_b, holdout_group_sizes)
        except ValueError as e:
            print(f"gate: {e}; refusing roll-over", file=sys.stderr)
            return 1
        summed = (
            f"summed eval_loss adapter={metrics.get('eval_loss_adapter')} "
            f"base={metrics.get('eval_loss_base')}"
        )
        if not paired_gate_decision(stats, args.gate_margin):
            print(
                verdict_line("REFUSED", ft_model, stats, args.gate_margin)
                + f"; pass needs mean_delta < -margin and t <= "
                f"-{PAIRED_T_MUST_REACH_2_BECAUSE_08_2_CALLS_SMALLER_PAIRED_MEANS_UNRESOLVED} "
                f"({summed}); adapter stays loaded for inspection, no older adapter is unloaded",
                file=sys.stderr,
            )
            return 2
        print(verdict_line("PASS", ft_model, stats, args.gate_margin) + f" ({summed})")
        gate_passed = True
        if args.null_check:
            try:
                null_stats = null_calibration_stats(
                    args, train, holdout, hyperparameters, holdout_group_sizes
                )
            except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, ValueError) as e:
                print(f"null calibration failed: {e}", file=sys.stderr)
                return 1
            ratio = abs(stats["t"]) / max(abs(null_stats["t"]), 1e-9)
            print(
                f"null calibration: label-shuffled adapter t={null_stats['t']:+.2f} "
                f"signs={null_stats['wins']}-{null_stats['losses']}-{null_stats['ties']} "
                f"vs arm t={stats['t']:+.2f}; arm/null ratio {ratio:.2f} "
                f"(threshold {args.null_ratio_min})"
            )
            if ratio < args.null_ratio_min:
                gate_passed = False
                unresolved_reason = (
                    f"arm/null |t| ratio {ratio:.2f} < {args.null_ratio_min}: both scale "
                    f"sqrt(windows), so no corpus size separates this arm from its own null "
                    f"(08.2 sect 2d) and the instrument cannot resolve the contrast"
                )

    pending_reason = unresolved_reason or (
        "--no-gate" if args.no_gate else "empty holdout"
    )

    if not metrics.get("hot_loaded"):
        adapter_dir = (job.get("result_files") or [None])[0]
        if not adapter_dir:
            print("no adapter dir in result_files; cannot load", file=sys.stderr)
            return 1
        try:
            print(
                http_json(
                    f"{args.server}/v1/load_lora_adapter",
                    {"lora_name": ft_model, "lora_path": adapter_dir},
                    args.timeout,
                )
            )
        except urllib.error.HTTPError as e:
            if e.code in (404, 405):
                print(
                    f"runtime load unmounted ({e.code}); {ft_model} serves after the "
                    f"next restart once {adapter_dir} is in NV_LORA_ADAPTER_DIRS",
                    file=sys.stderr,
                )
                if not gate_passed:
                    return unresolved_exit(ft_model, pending_reason)
                return 0
            print(f"runtime load failed: {e}", file=sys.stderr)
            return 1
        except urllib.error.URLError as e:
            print(f"server unreachable during load: {e}", file=sys.stderr)
            return 1

    for prompt_text in args.eval_prompt or ["Reply with one short sentence: who are you?"]:
        try:
            adapted = chat_once(args.server, ft_model, prompt_text, args.timeout)
            base = chat_once(args.server, "", prompt_text, args.timeout)
            print(f"--- eval prompt: {prompt_text!r}")
            print(f"base   : {base[:400]}")
            print(f"adapter: {adapted[:400]}")
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError) as e:
            print(f"smoke eval failed for {ft_model}: {e}", file=sys.stderr)
            return 1

    if not gate_passed:
        return unresolved_exit(ft_model, pending_reason)

    try:
        promoted = http_json(
            f"{args.server}/v1/adapters/promote",
            {"lora_name": ft_model},
            args.timeout,
        )
        print(f"promoted to boot default: {promoted}")
    except urllib.error.HTTPError as e:
        if e.code in (404, 405):
            print(
                f"promote route unmounted ({e.code}); {ft_model} serves now but a "
                f"restart boots the previously promoted adapter",
                file=sys.stderr,
            )
        else:
            print(
                f"promote failed ({e}); {ft_model} serves now but a restart boots "
                f"the previously promoted adapter",
                file=sys.stderr,
            )
    except urllib.error.URLError as e:
        print(f"server unreachable during promote: {e}", file=sys.stderr)

    stale_pat = re.compile(rf"^ft:.+:{re.escape(args.suffix)}(-nullcheck)?:")
    try:
        stale = [
            m for m in list_models(args.server, args.timeout)
            if stale_pat.match(m) and m != ft_model
        ]
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f"could not list models to roll over: {e}", file=sys.stderr)
        return 1
    for old in stale:
        try:
            print(f"unloading previous adapter {old}: {unload_adapter(args.server, old, args.timeout)}")
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"unload of {old} failed (kept loaded): {e}", file=sys.stderr)
    print(f"rolling adapter is now {ft_model}")
    return 0

def self_test():
    import tempfile
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    checks = 0

    def ok(cond, what):
        nonlocal checks
        assert cond, f"self-test failed: {what}"
        checks += 1

    fp_a = content_fingerprint("What  is\tRust?", "A language.\n")
    fp_b = content_fingerprint("what is rust?", "a   LANGUAGE.")
    fp_c = content_fingerprint("what is rust?", "a compiler.")
    ok(fp_a == fp_b, "whitespace/case variants share a fingerprint")
    ok(fp_a != fp_c, "different completions split fingerprints")

    args = argparse.Namespace(
        model="teacher",
        keep_reasoning_rows=False,
        require_label=False,
        min_completion_chars=1,
        max_records=512,
    )
    raw = [
        {"model": "teacher", "finish_reason": "stop", "prompt": "P one", "completion": "C one"},
        {"model": "teacher", "finish_reason": "stop", "prompt": "P  ONE", "completion": "c one\n"},
        {"model": "teacher", "finish_reason": "stop", "prompt": "P one", "completion": "C other"},
    ]
    kept = select_rows(raw, {}, args)
    ok(len(kept) == 2, "near-duplicate retry deduped, shared-prompt distinct completion kept")

    eligible = holdout_eligible_because_the_eval_window_keeps_a_completion_token_scorable
    ok(eligible({"prompt": "short", "completion": "c", "prompt_tokens": 100}), "short prompt_tokens eligible")
    ok(not eligible({"prompt": "short", "completion": "c", "prompt_tokens": 504}), "prompt_tokens at cap-minus-slack is train-only")
    ok(eligible({"prompt": "short", "completion": "c", "prompt_tokens": 503}), "prompt_tokens under cap-minus-slack eligible")
    ok(not eligible({"prompt": "x" * 600, "completion": "c"}), "600-byte prompt without prompt_tokens is train-only at the one-byte-per-token floor")
    ok(eligible({"prompt": "short", "completion": "c"}), "short prompt without prompt_tokens eligible")

    convo = [
        {"prompt": "User: hi\nAssistant:", "completion": "hello there friend"},
        {
            "prompt": "User: hi\nAssistant: hello there friend\nUser: more\nAssistant:",
            "completion": "sure thing",
        },
        {"prompt": "User: unrelated\nAssistant:", "completion": "ok"},
    ]
    merged = merged_prompt_groups_because_multiturn_continuations_embed_their_history(convo)
    ok(len(merged) == 2, "a continuation merges into its conversation root's group")
    sizes = sorted(len(v) for v in merged.values())
    ok(sizes == [1, 2], "the merged conversation group holds both turns")

    data = [
        {"prompt": f"unique prompt {i} end", "completion": f"completion {i}"} for i in range(34)
    ] + [
        {"prompt": "SHARED  prompt", "completion": f"variant {i}"} for i in range(6)
    ]
    holdout, train, digest, group_sizes = split_holdout_without_prompt_leakage(data, 0.1, "seedA")
    ok(len(holdout) + len(train) == 40, "split partitions all rows")
    ok(len(group_sizes) >= 9, "holdout seats at least 9 independent windows")
    ok(sum(group_sizes) == len(holdout), "group sizes cover the holdout exactly")
    shared_in_holdout = sum(1 for r in holdout if r["prompt"] == "SHARED  prompt")
    ok(shared_in_holdout in (0, 6), "a shared prompt group never straddles the split")
    holdout2, train2, digest2, group_sizes2 = split_holdout_without_prompt_leakage(data, 0.1, "seedA")
    ok(
        (holdout, train, digest, group_sizes) == (holdout2, train2, digest2, group_sizes2),
        "split is deterministic under one seed",
    )
    _, _, digest3, _ = split_holdout_without_prompt_leakage(data, 0.1, "seedB")
    ok(digest != digest3, "a different seed redraws the ordering")
    tiny_holdout, tiny_train, _, tiny_sizes = split_holdout_without_prompt_leakage(data[:12], 0.1, "seedA")
    ok(len(tiny_sizes) >= 9 and len(tiny_train) >= 1, "small corpora still seat the 9-window floor")

    mix = [
        {"prompt": f"distinct prompt {i} end", "completion": f"completion {i}"} for i in range(12)
    ] + [
        {"prompt": f"long {i} " + "x" * 600, "completion": f"c{i}"} for i in range(5)
    ]
    long_h, long_t, _, long_sizes = split_holdout_without_prompt_leakage(mix, 0.1, "seedA")
    ok(all(len(r["prompt"]) < 600 for r in long_h), "over-cap prompts never sit in the holdout")
    ok(len(long_sizes) >= 9, "eligible rows still seat the window floor around train-only rows")

    stats = paired_delta_stats([0.9] * 9 + [1.1], [1.0] * 10)
    ok(stats["n"] == 10, "stats count rows when every window is one row")
    ok(abs(stats["mean_delta"] + 0.08) < 1e-12, "paired mean delta")
    ok(abs(stats["t"] + 4.0) < 1e-9, "t statistic on the known fixture")
    ok((stats["wins"], stats["losses"], stats["ties"]) == (9, 1, 0), "sign counts")
    grouped = paired_delta_stats([0.9, 0.9, 0.9, 1.2], [1.0, 1.0, 1.0, 1.0], [3, 1])
    ok(grouped["n"] == 2, "grouped stats count windows, not rows")
    ok(abs(grouped["mean_delta"] - 0.05) < 1e-9, "window deltas average the group's rows")
    ok((grouped["wins"], grouped["losses"], grouped["ties"]) == (1, 1, 0), "grouped sign counts")
    try:
        paired_delta_stats([1.0] * 3, [1.0] * 3, [2])
        ok(False, "group sizes must cover every eval row")
    except ValueError:
        ok(True, "uncovered eval rows raise")
    flat = paired_delta_stats([0.9] * 9, [1.0] * 9)
    ok(flat["t"] == -math.inf, "constant negative deltas drive t to -inf, not a crash")
    ok(paired_gate_decision(stats, 0.0), "clear win passes the paired gate")
    ok(not paired_gate_decision(stats, 0.1), "a margin above the mean delta refuses")
    weak = paired_delta_stats([0.99, 1.01, 0.99, 1.01, 0.99, 1.01, 0.99, 1.01, 0.98], [1.0] * 9)
    ok(not paired_gate_decision(weak, 0.0), "a sub-2-t mean refuses even when negative")
    try:
        paired_delta_stats([1.0, math.nan], [1.0, 1.0])
        ok(False, "nan CE must raise")
    except ValueError:
        ok(True, "nan CE raises")
    try:
        paired_eval_vectors({"eval_loss_adapter": 1.0}, 9)
        ok(False, "missing vectors must raise")
    except ValueError:
        ok(True, "missing per-example vectors raise")

    train_rows = [{"prompt": f"p{i}", "completion": f"c{i}"} for i in range(11)]
    null_rows = label_shuffled_null_rows(train_rows, "seedA")
    ok(len(null_rows) == 11, "null keeps the row count")
    ok(
        sorted(r["prompt"] for r in null_rows) == sorted(r["prompt"] for r in train_rows)
        and sorted(r["completion"] for r in null_rows) == sorted(r["completion"] for r in train_rows),
        "null preserves the prompt and completion multisets",
    )
    ok(
        all(r["completion"] != f"c{r['prompt'][1:]}" for r in null_rows),
        "no prompt keeps its own completion",
    )
    ok(
        null_rows == label_shuffled_null_rows(train_rows, "seedA"),
        "null shuffle is deterministic under one seed",
    )

    line = verdict_line("PASS", "ft:base:traj:x", stats, 0.05)
    ok("n=10" in line and "margin=0.05" in line and "n_ctx<=512" in line, "verdict names n, margin, n_ctx")

    calls = []
    scenario = {"gate": "pass", "null": "flat", "load_code": None, "served": []}

    class Stub(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            return

        def reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            calls.append(("GET", path, None))
            if path == "/v1/models":
                self.reply(200, {"data": [{"id": m} for m in scenario["served"]]})
            else:
                self.reply(404, {})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            path = self.path.split("?")[0]
            calls.append(("POST", path, payload))
            if path == "/v1/fine_tuning/jobs":
                val = payload.get("validation_data") or []
                is_null = str(payload.get("suffix", "")).endswith("-nullcheck")
                mode = scenario["null"] if is_null else scenario["gate"]
                base_ce = [1.0] * len(val)
                if mode == "pass":
                    ce_a = [1.0 - 0.1 + 0.001 * (-1) ** i for i in range(len(val))]
                elif mode == "refuse":
                    ce_a = [1.0 + 0.1 - 0.001 * (-1) ** i for i in range(len(val))]
                else:
                    ce_a = [1.0 + 0.05 * (-1) ** i for i in range(len(val))]
                short = "nul10000" if is_null else "arm10000"
                self.reply(
                    200,
                    {
                        "status": "succeeded",
                        "fine_tuned_model": f"ft:base:{payload.get('suffix', 'x')}:{short}",
                        "result_files": ["/nonexistent/adapter"],
                        "metrics": {
                            "eval_ce_adapter": ce_a,
                            "eval_ce_base": base_ce,
                            "eval_loss_adapter": sum(ce_a),
                            "eval_loss_base": sum(base_ce),
                            "hot_loaded": scenario["load_code"] is None,
                        },
                    },
                )
            elif path == "/v1/load_lora_adapter":
                if scenario["load_code"]:
                    self.reply(scenario["load_code"], {"error": "unmounted"})
                else:
                    self.reply(200, {"ok": True})
            elif path == "/v1/chat/completions":
                self.reply(200, {"choices": [{"message": {"content": "ok"}}]})
            elif path in ("/v1/adapters/promote", "/v1/unload_lora_adapter"):
                self.reply(200, {"ok": True})
            else:
                self.reply(404, {})

    def posts(path):
        return [c for c in calls if c[0] == "POST" and c[1] == path]

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    server_url = f"http://127.0.0.1:{srv.server_port}"

    with tempfile.TemporaryDirectory() as td:
        traj = os.path.join(td, "trajectories.jsonl")
        with open(traj, "w", encoding="utf-8") as f:
            for i in range(40):
                f.write(
                    json.dumps(
                        {
                            "model": "teacher",
                            "finish_reason": "stop",
                            "prompt": f"prompt {i} end",
                            "completion": f"a completion long enough {i}",
                        }
                    )
                    + "\n"
                )
        base_argv = [
            "--trajectories", traj,
            "--base", td,
            "--model", "teacher",
            "--server", server_url,
            "--min-completion-chars", "1",
            "--eval-prompt", "who?",
        ]

        scenario.update(
            gate="pass",
            null="flat",
            load_code=None,
            served=[
                "ft:base:traj:arm10000",
                "ft:base:traj:old00000",
                "ft:base:traj-nullcheck:dead0000",
            ],
        )
        calls.clear()
        ok(main(base_argv) == 0, "e2e: passing round exits 0")
        ok(len(posts("/v1/adapters/promote")) == 1, "e2e: passing round promotes")
        unloaded = [c[2]["lora_name"] for c in posts("/v1/unload_lora_adapter")]
        ok(
            "ft:base:traj:old00000" in unloaded
            and "ft:base:traj-nullcheck:dead0000" in unloaded,
            "e2e: rollover reaps the stale adapter and the orphaned nullcheck adapter",
        )

        scenario.update(gate="refuse")
        calls.clear()
        ok(main(base_argv) == 2, "e2e: refused gate exits 2")
        ok(not posts("/v1/adapters/promote"), "e2e: refused gate never promotes")

        scenario.update(gate="pass", null="pass")
        calls.clear()
        ok(main(base_argv + ["--null-check"]) == 3, "e2e: arm matching its own null exits 3")
        ok(not posts("/v1/adapters/promote"), "e2e: null-flipped round never promotes")
        ok(len(posts("/v1/fine_tuning/jobs")) == 2, "e2e: null check trains a second job")
        null_unloads = [c[2]["lora_name"] for c in posts("/v1/unload_lora_adapter")]
        ok("ft:base:traj-nullcheck:nul10000" in null_unloads, "e2e: null adapter unloads after scoring")

        scenario.update(null="flat")
        calls.clear()
        ok(main(base_argv + ["--null-check"]) == 0, "e2e: resolved null-check round exits 0")
        ok(len(posts("/v1/adapters/promote")) == 1, "e2e: resolved null-check promotes")

        scenario.update(load_code=404)
        calls.clear()
        ok(main(base_argv + ["--no-gate"]) == 3, "e2e: unmounted load on an ungated round exits 3, not 0")
        ok(not posts("/v1/adapters/promote"), "e2e: unmounted ungated round never promotes")
        scenario.update(load_code=None)

        small = os.path.join(td, "small.jsonl")
        with open(small, "w", encoding="utf-8") as f:
            for i in range(6):
                f.write(
                    json.dumps(
                        {
                            "model": "teacher",
                            "finish_reason": "stop",
                            "prompt": f"tiny {i} end",
                            "completion": f"tiny completion {i}",
                        }
                    )
                    + "\n"
                )
        calls.clear()
        ok(
            main(["--trajectories", small] + base_argv[2:]) == 3,
            "e2e: an under-9-window corpus exits 3 before training",
        )
        ok(not posts("/v1/fine_tuning/jobs"), "e2e: no training job before an unseatable verdict")

        longs = os.path.join(td, "longs.jsonl")
        with open(longs, "w", encoding="utf-8") as f:
            for i in range(40):
                f.write(
                    json.dumps(
                        {
                            "model": "teacher",
                            "finish_reason": "stop",
                            "prompt": f"conversation {i} end",
                            "completion": f"a completion long enough {i}",
                            "prompt_tokens": 600,
                        }
                    )
                    + "\n"
                )
        calls.clear()
        ok(
            main(["--trajectories", longs] + base_argv[2:]) == 3,
            "e2e: an all-over-cap corpus cannot seat scorable windows and exits 3",
        )
        ok(not posts("/v1/fine_tuning/jobs"), "e2e: no training job when no window is scorable")

    srv.shutdown()

    print(f"self-test: {checks} assertions passed")
    return 0

if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    sys.exit(main())
