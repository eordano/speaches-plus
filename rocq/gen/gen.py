#!/usr/bin/env python3
"""rocq/gen -- automated theorem generation for speaches-plus.

Maps (a) the model config.json, (b) machine constants, (c) a per-kernel
byte-traffic manifest derived from the code (kernels.json, with file:line
citations), and (d) the CURRENT benchmark measurements into shape-specialized
Rocq files:

  GenTraffic.v  -- the traffic arithmetic as Z definitions (the formulas ARE
                  the Rocq terms), with every total re-proved by vm_compute
                  and cross-checked against an INDEPENDENT scan of the
                  safetensors headers (a mismatch fails the build).
  GenRoofline.v -- lower-bound theorems instantiated to this GPU, proved
                  against the core Roofline development's physics hypotheses,
                  plus one efficiency-ratio corollary per phase:
                  measured <= r * theoretical_min ("within r x of light"),
                  and consistency tripwires theoretical_min <= measured
                  (these FAIL THE BUILD if the manifest ever overcounts).

Every future kernel/config change re-derives its efficiency bound by
re-running gen/run.sh instead of trusting a stale analysis.

Inputs are pinned in machine.json / measured.json / kernels.json next to this
script.  No third-party python dependencies; safetensors headers are read
directly (8-byte length + JSON header).
"""

import json
import math
import os
import re
import struct
import sys
import datetime
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
ROCQ = os.path.dirname(HERE)

def _resolve_dir(env_var, candidates, what):
    d = os.environ.get(env_var)
    if d:
        return d
    for c in candidates:
        if os.path.exists(os.path.join(c, "config.json")):
            return c
    print(
        f"[gen] FATAL: no {what} snapshot found; set {env_var}. Tried:\n  "
        + "\n  ".join(candidates),
        file=sys.stderr,
    )
    sys.exit(1)

HF_HUB_CACHE_DEFAULT = os.environ.get(
    "HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")
)

VERIFIER_DIR = _resolve_dir(
    "NV_GEMMA4_VERIFIER_DIR",
    [
        HF_HUB_CACHE_DEFAULT
        + "/models--nvidia--Gemma-4-31B-IT-NVFP4/"
        "snapshots/e5ef03afa233c35cb000323ff098d4291e1dd07c",
    ],
    "verifier",
)
DRAFT_DIR = _resolve_dir(
    "NV_EAGLE3_DRAFT_DIR",
    [
        HF_HUB_CACHE_DEFAULT
        + "/models--RedHatAI--gemma-4-31B-it-speculator.eagle3/snapshots/"
        "28a1c8b4bb64dbaee883ba35341841138bdf1fe3",
    ],
    "eagle3 draft",
)

def die(msg):
    print(f"[gen] FATAL: {msg}", file=sys.stderr)
    sys.exit(1)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def scan_safetensors(d):
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".safetensors"):
            continue
        with open(os.path.join(d, fn), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            out[k] = (v["dtype"], tuple(v["shape"]), b - a)
    return out

def eval_formula(formula, env):
    return eval(formula, {"__builtins__": {}, "min": min}, dict(env))

def coq_formula(formula):
    s = formula
    s = re.sub(r"min\(([^,()]+),\s*([^()]+)\)", r"(Z.min \1 (\2))", s)
    s = s.replace("//", "/")
    return s

def frac_lit(fr):
    fr = Fraction(fr)
    if fr.denominator == 1:
        return str(fr.numerator)
    return f"{fr.numerator} / {fr.denominator}"

def floor_frac(fr, denom):
    return Fraction(math.floor(Fraction(fr) * denom), denom)

def ceil_frac(fr, denom):
    return Fraction(math.ceil(Fraction(fr) * denom), denom)

def emit_phase_defs(L, phases):
    for ph, val in phases:
        L.append(f"(* {ph['kernel']}")
        for c in ph["cite"]:
            L.append(f"   - {c}")
        L.append("*)")
        L.append(f"Definition {ph['coq_name']} : Z := {coq_formula(ph['formula'])}.")
        L.append(f"Lemma {ph['coq_name']}_val : {ph['coq_name']} = {val}.")
        L.append("Proof. vm_compute. reflexivity. Qed.\n")

SUMMARY_RE = re.compile(
    r"GRAPHED SUMMARY gpu_accept=\w+ rounds=(\d+) emitted=(\d+) "
    r"drafts_accepted=(\d+) tokens/round=([\d.]+) draft-accept=([\d.]+) "
    r"tok_?/s=([\d.]+) ms_per_tok=([\d.]+) draft_ms/round=([\d.]+) "
    r"verify_ms/round=([\d.]+)"
)

def parse_measured(measured_cfg):
    log = measured_cfg["server_log"]
    min_rounds = measured_cfg["min_rounds"]
    rows = []
    kmodes = set()
    with open(log, errors="replace") as f:
        for line in f:
            m = SUMMARY_RE.search(line)
            if m:
                r = dict(
                    rounds=int(m.group(1)),
                    tpr=Fraction(m.group(4)),
                    tok_s=Fraction(m.group(6)),
                    ms_per_tok=Fraction(m.group(7)),
                    draft_ms=Fraction(m.group(8)),
                    verify_ms=Fraction(m.group(9)),
                )
                r["host_ms"] = r["tpr"] * r["ms_per_tok"] - r["draft_ms"] - r["verify_ms"]
                rows.append(r)
            km = re.search(r"cond_mode=(\w+) k=(\d+)", line)
            if km:
                kmodes.add((km.group(1), int(km.group(2))))
    if not rows:
        die(f"no GRAPHED SUMMARY lines in {log}")
    kept = [r for r in rows if r["rounds"] >= min_rounds]
    if not kept:
        die(f"no summaries with rounds >= {min_rounds} in {log}")
    if kmodes != {("shift", measured_cfg["k_chain"])}:
        die(f"cond_mode/k mismatch: log has {kmodes}, expected shift/k={measured_cfg['k_chain']}")

    bj = load_json(measured_cfg["bench_json"])
    tot_tok, tot_s = 0, Fraction(0)
    for rnd in bj["rounds"]:
        for p in rnd["per_prompt"]:
            tot_tok += p["completion_tokens"]
            tot_s += Fraction(str(p["decode_wall_s"]))
    agg = Fraction(tot_tok) / tot_s

    return dict(
        n_summaries=len(kept),
        n_all=len(rows),
        verify_lo=min(r["verify_ms"] for r in kept) / 1000,
        verify_hi=max(r["verify_ms"] for r in kept) / 1000,
        draft_lo=min(r["draft_ms"] for r in kept) / 1000,
        draft_hi=max(r["draft_ms"] for r in kept) / 1000,
        host_lo=max(min(r["host_ms"] for r in kept), Fraction(0)) / 1000,
        host_hi=max(max(r["host_ms"] for r in kept), Fraction(0)) / 1000,
        tpr_lo=min(r["tpr"] for r in kept),
        tpr_hi=max(r["tpr"] for r in kept),
        tok_s_lo=min(r["tok_s"] for r in kept),
        tok_s_hi=max(r["tok_s"] for r in kept),
        agg_tok_s=agg,
        agg_tokens=tot_tok,
        log=log,
        bench=measured_cfg["bench_json"],
        k=measured_cfg["k_chain"],
    )

def main():
    machine = load_json(os.path.join(HERE, "machine.json"))
    measured_cfg = load_json(os.path.join(HERE, "measured.json"))
    manifest = load_json(os.path.join(HERE, "kernels.json"))

    vcfg = load_json(os.path.join(VERIFIER_DIR, "config.json"))
    dcfg = load_json(os.path.join(DRAFT_DIR, "config.json"))
    tc = vcfg["text_config"]

    layer_types = tc["layer_types"]
    env = dict(
        h=tc["hidden_size"],
        inter=tc["intermediate_size"],
        nlayers=tc["num_hidden_layers"],
        n_slide=sum(1 for t in layer_types if t == "sliding_attention"),
        n_full=sum(1 for t in layer_types if t == "full_attention"),
        nh=tc["num_attention_heads"],
        hd=tc["head_dim"],
        nkv=tc["num_key_value_heads"],
        ghd=tc["global_head_dim"],
        gnkv=tc["num_global_key_value_heads"],
        window=tc["sliding_window"],
        vocab=tc["vocab_size"],
        draft_vocab=dcfg["draft_vocab_size"],
    )
    assert env["n_slide"] + env["n_full"] == env["nlayers"]

    moe_keys = [
        k for k in ("num_local_experts", "num_experts", "n_routed_experts",
                    "num_experts_per_tok", "moe_intermediate_size",
                    "shared_expert_intermediate_size")
        if tc.get(k) or vcfg.get(k)
    ]
    if moe_keys:
        die(
            "MoE checkpoint detected (config declares "
            + ", ".join(moe_keys)
            + "): every mlp_* formula in kernels.json is a DENSE per-layer MLP, "
            "so regenerating here would emit weight-traffic constants that are "
            "wrong by roughly the expert count, and the Rocq roofline floors "
            "would be proved from them. Add expert-aware phases (routed stack + "
            "shared expert, with witness selectors over experts.*) before "
            "pointing this generator at a MoE model."
        )

    assert tc["attention_k_eq_v"] is True, "qkv_full formula assumes k_eq_v"
    assert vcfg.get("tie_word_embeddings", True), "lm_head formula assumes tied embeddings"
    ignore = vcfg["quantization_config"]["ignore"]
    assert "lm_head" in ignore
    assert all(
        f"model.language_model.layers.{i}.self_attn*" in ignore
        for i in range(env["nlayers"])
    ), "attention bf16 assumption broken: some self_attn not in quant ignore list"

    vt = scan_safetensors(VERIFIER_DIR)
    lang_bytes = sum(sz for k, (_, _, sz) in vt.items() if k.startswith("model.language_model."))
    non_lang = {k for k in vt if not k.startswith("model.language_model.")}
    unexpected = {k for k in non_lang if "vision" not in k}
    if unexpected:
        print(f"[gen] note: non-language non-vision tensors excluded from W: {sorted(unexpected)}")

    dt = scan_safetensors(DRAFT_DIR)
    draft_subset_names = [
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
        "lm_head.weight",
        "layers.0.hidden_norm.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "norm.weight",
    ]
    draft_subset = sum(dt[n][2] for n in draft_subset_names)

    layer_re = re.compile(r"^model\.language_model\.layers\.(\d+)\.")

    def layer_class_of(name):
        m = layer_re.match(name)
        if not m:
            return None
        return layer_types[int(m.group(1))]

    def witness_names(w):
        if "exact" in w:
            return {w["exact"]} & set(vt)
        want = w["layers"]
        rx = re.compile(w["name_re"])
        out = set()
        for name in vt:
            lc = layer_class_of(name)
            if lc is None:
                continue
            if want == "sliding" and lc != "sliding_attention":
                continue
            if want == "full" and lc != "full_attention":
                continue
            if rx.search(name):
                out.add(name)
        return out

    vphases = []
    claimed = {}
    for ph in manifest["verify_weight_phases"]:
        val = eval_formula(ph["formula"], env)
        vphases.append((ph, val))
        w = ph.get("witness")
        if not w:
            die(f"phase {ph['coq_name']} has no witness selector")
        names = witness_names(w)
        if not names:
            die(f"phase {ph['coq_name']}: witness selector matched no tensors")
        actual = sum(vt[n][2] for n in names)
        if actual != val:
            die(
                f"phase {ph['coq_name']}: formula {val} != witnessed bytes {actual} "
                f"over {len(names)} tensors -- the manifest disagrees with the checkpoint"
            )
        for n in names:
            if n in claimed:
                die(
                    f"tensor {n} claimed by both {claimed[n]} and {ph['coq_name']}: "
                    "witness selectors must be disjoint or bytes are double counted"
                )
            claimed[n] = ph["coq_name"]
        print(f"[gen] witness {ph['coq_name']:22s} {val:>14,d} B over {len(names):4d} tensors  OK")

    big_sum = sum(v for _, v in vphases)
    misc = lang_bytes - big_sum
    unmatched = sum(
        sz for k, (_, _, sz) in vt.items()
        if k.startswith("model.language_model.") and k not in claimed
    )
    if unmatched != misc:
        die(
            f"unmatched tensor bytes {unmatched} != misc {misc}: the phase "
            "selectors and the formula sum disagree about what is left over"
        )
    if not (0 <= misc <= 16 * 2**20):
        die(
            f"misc_small_bytes out of range: lang={lang_bytes} formulas={big_sum} "
            f"misc={misc} -- the manifest disagrees with the checkpoint"
        )
    print(f"[gen] witness residual (norms etc) {misc:,d} B over "
          f"{len([k for k in vt if k.startswith('model.language_model.') and k not in claimed])} tensors")
    wv = lang_bytes

    dphases = []
    for ph in manifest["draft_step_weight_phases"]:
        val = eval_formula(ph["formula"], env)
        dphases.append((ph, val))
    wd_formula = sum(v for _, v in dphases)
    if wd_formula != draft_subset:
        die(f"draft formulas {wd_formula} != drafter safetensors subset {draft_subset}")
    wd = wd_formula

    kv = manifest["verify_kv_read"]
    kv_full_f = kv["formula_full"]
    kv_slide_f = kv["formula_sliding"]

    def kv_read(ctx):
        e = dict(env)
        e["ctx"] = ctx
        return eval_formula(kv_full_f, e) + eval_formula(kv_slide_f, e)

    CTX_FULL = tc["max_position_embeddings"]
    min_traffic_256 = wv + kv_read(256)
    min_traffic_full = wv + kv_read(CTX_FULL)

    BW = Fraction(machine["hbm_bw_bytes_per_s"])
    L2 = Fraction(machine["l2_resident_bytes"])
    ms = parse_measured(measured_cfg)
    K = ms["k"]

    BW_MEAS = Fraction(machine.get("measured_read_bw_bytes_per_s", 0))
    if BW_MEAS > BW:
        die(f"measured read bandwidth {float(BW_MEAS):.3e} exceeds the BW hypothesis "
            f"{float(BW):.3e}: the roofline floors would be unsound")

    floor_v = (wv - L2) / BW
    floor_v_full = (min_traffic_full - L2) / BW
    floor_d = (wd - L2) / BW
    tmin_serial = floor_v + K * floor_d

    tpr_hi = ms["tpr_hi"]
    ceiling = tpr_hi * BW / (wv - L2)
    serial_ceiling = tpr_hi / tmin_serial

    NS = 10**9
    floor_v_lo = floor_frac(floor_v, NS)
    floor_v_full_lo = floor_frac(floor_v_full, NS)
    floor_d_lo = floor_frac(floor_d, NS)
    tmin_lo = floor_frac(tmin_serial, NS)
    ceiling_up = ceil_frac(ceiling, 1)
    serial_ceiling_up = ceil_frac(serial_ceiling, 1)

    r_verify = ceil_frac(ms["verify_hi"] / floor_v, 100)
    r_draft = ceil_frac(ms["draft_hi"] / (K * floor_d), 100)
    agg_lit = floor_frac(ms["agg_tok_s"], 100)
    r_e2e = ceil_frac(ceil_frac(ceiling, 1) / agg_lit, 100)

    if not floor_v <= ms["verify_lo"]:
        die(f"verify floor {float(floor_v)} > measured lo {float(ms['verify_lo'])}: manifest overcounts")
    if not K * floor_d <= ms["draft_lo"]:
        die(f"draft floor {float(K*floor_d)} > measured lo {float(ms['draft_lo'])}: manifest overcounts")
    if not ms["agg_tok_s"] <= ceiling:
        die("measured aggregate exceeds speed-of-light ceiling: constants are wrong")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    stamp = (
        f"AUTO-GENERATED by rocq/gen/gen.py on {now} -- DO NOT EDIT; re-run gen/run.sh.\n"
        f"   verifier config: {re.sub(r'/nix/store/[a-z0-9]{{32}}-', '/nix/store/<hash>-', VERIFIER_DIR)}/config.json\n"
        f"   drafter  config: {re.sub(r'/nix/store/[a-z0-9]{{32}}-', '/nix/store/<hash>-', DRAFT_DIR)}/config.json\n"
        f"   machine:  {machine['hbm_bw_bytes_per_s']} B/s HBM, "
        f"{machine['l2_resident_bytes']} B L2-residency credit ({machine['hbm_bw_provenance']})\n"
        f"   measured: {ms['log']}\n"
        f"             ({ms['n_summaries']}/{ms['n_all']} summaries with rounds>={measured_cfg['min_rounds']}; "
        f"verify {float(ms['verify_lo']*1000):.2f}-{float(ms['verify_hi']*1000):.2f} ms, "
        f"draft {float(ms['draft_lo']*1000):.2f}-{float(ms['draft_hi']*1000):.2f} ms/round, "
        f"tokens/round {float(ms['tpr_lo']):.2f}-{float(ms['tpr_hi']):.2f}, K={K} shift)\n"
        f"             {ms['bench']} aggregate {float(ms['agg_tok_s']):.2f} tok/s over {ms['agg_tokens']} tokens"
    )

    L = []
    L.append(f"(* {stamp} *)\n")
    L.append("From Stdlib Require Import ZArith Lia.")
    L.append("Open Scope Z_scope.\n")
    L.append("(* Shape environment, derived from config.json. *)")
    L.append("Module S.")
    for name, meta in manifest["shapes"].items():
        L.append(f"  (* {meta['cite']} *)")
        L.append(f"  Definition {name} : Z := {env[name]}.")
    L.append("End S.")
    L.append("Import S.\n")
    L.append("Lemma layer_mix : n_slide + n_full = nlayers.")
    L.append("Proof. vm_compute. reflexivity. Qed.\n")

    L.append("(* ---- verify-path weight traffic, one forward pass ---- *)")
    unfold_names = " ".join(f"S.{n}" for n in manifest["shapes"])
    emit_phase_defs(L, vphases)

    L.append("(* Residual small tensors read each forward (RMSNorm gammas, layer_scalars,")
    L.append(f"   per-proj scale scalars): safetensors language_model total minus the GEMM")
    L.append(f"   formulas above; asserted < 16 MiB at generation ({misc} B). *)")
    L.append(f"Definition misc_small_bytes : Z := {misc}.\n")
    L.append("Definition verify_weight_bytes : Z :=")
    L.append("  " + " + ".join(ph["coq_name"] for ph, _ in vphases) + " + misc_small_bytes.\n")
    L.append("(* Independent inventory: sum of every model.language_model.* tensor's byte")
    L.append("   size from the safetensors headers (vision tower excluded -- never touched")
    L.append("   on the text decode path; tied embed counted once and read in full by the")
    L.append("   dense lm_head GEMM). The equality below is the manifest<->checkpoint")
    L.append("   cross-check, re-proved inside the kernel. *)")
    L.append(f"Definition safetensors_language_model_bytes : Z := {lang_bytes}.")
    L.append("Lemma verify_weight_crosscheck :")
    L.append("  verify_weight_bytes = safetensors_language_model_bytes.")
    L.append("Proof. vm_compute. reflexivity. Qed.")
    L.append(f"Lemma verify_weight_bytes_val : verify_weight_bytes = {wv}.")
    L.append("Proof. vm_compute. reflexivity. Qed.\n")

    L.append("(* ---- committed-KV read traffic (unique-byte lower bound) ---- *)")
    L.append(f"(* {kv['kernel']}")
    for c in kv["cite"]:
        L.append(f"   - {c}")
    L.append(f"   NOTE: {kv['note']}")
    L.append("*)")
    L.append("Definition kv_read_bytes (ctx : Z) : Z :=")
    L.append(f"  {coq_formula(kv_full_f)}")
    L.append(f"  + {coq_formula(kv_slide_f)}.\n")
    L.append("Lemma kv_read_bytes_mono :")
    L.append("  forall a b, 0 <= a -> a <= b -> kv_read_bytes a <= kv_read_bytes b.")
    L.append(f"Proof. intros; unfold kv_read_bytes; cbv [{unfold_names}]; lia. Qed.\n")
    L.append("Definition verify_min_traffic (ctx : Z) : Z :=")
    L.append("  verify_weight_bytes + kv_read_bytes ctx.\n")
    L.append(f"Lemma verify_min_traffic_at_256 : verify_min_traffic 256 = {min_traffic_256}.")
    L.append("Proof. vm_compute. reflexivity. Qed.")
    L.append(f"Lemma verify_min_traffic_at_{CTX_FULL} :")
    L.append(f"  verify_min_traffic {CTX_FULL} = {min_traffic_full}.")
    L.append("Proof. vm_compute. reflexivity. Qed.\n")

    L.append("(* ---- drafter chain-step weight traffic ---- *)")
    emit_phase_defs(L, dphases)
    excl = manifest["excluded_from_step"]
    L.append("(* Excluded from the per-step floor (all exclusions only WEAKEN the bound):")
    for k, v in excl.items():
        L.append(f"   - {k}: {v}")
    L.append("*)")
    L.append("Definition draft_step_weight_bytes : Z :=")
    L.append("  " + " + ".join(ph["coq_name"] for ph, _ in dphases) + ".\n")
    L.append("(* Cross-check vs the drafter checkpoint (same tensors, safetensors scan). *)")
    L.append(f"Definition safetensors_draft_step_bytes : Z := {draft_subset}.")
    L.append("Lemma draft_step_crosscheck :")
    L.append("  draft_step_weight_bytes = safetensors_draft_step_bytes.")
    L.append("Proof. vm_compute. reflexivity. Qed.")
    L.append(f"Lemma draft_step_weight_bytes_val : draft_step_weight_bytes = {wd}.")
    L.append("Proof. vm_compute. reflexivity. Qed.")

    with open(os.path.join(ROCQ, "GenTraffic.v"), "w") as f:
        f.write("\n".join(L) + "\n")

    R = []
    R.append(f"(* {stamp} *)\n")
    R.append("From Stdlib Require Import Reals Lra Psatz ZArith.")
    R.append("From SpeachesPlus Require Import Roofline GenTraffic.")
    R.append("Open Scope R_scope.\n")
    R.append("(* Machine constants (gen/machine.json). *)")
    R.append(f"Definition BW : R := {machine['hbm_bw_bytes_per_s']}.")
    R.append(f"Definition L2 : R := {machine['l2_resident_bytes']}.")
    R.append("Lemma BW_pos : 0 < BW. Proof. unfold BW; lra. Qed.\n")
    R.append("(* Weight traffic constants, imported from the checked Z development. *)")
    R.append("Definition Wv : R := IZR GenTraffic.verify_weight_bytes.")
    R.append(f"Lemma Wv_val : Wv = {wv}.")
    R.append("Proof. unfold Wv; rewrite GenTraffic.verify_weight_bytes_val; lra. Qed.")
    R.append("Lemma Wv_gt_L2 : L2 < Wv. Proof. rewrite Wv_val; unfold L2; lra. Qed.\n")
    R.append("Definition Wd : R := IZR GenTraffic.draft_step_weight_bytes.")
    R.append(f"Lemma Wd_val : Wd = {wd}.")
    R.append("Proof. unfold Wd; rewrite GenTraffic.draft_step_weight_bytes_val; lra. Qed.")
    R.append("Lemma Wd_gt_L2 : L2 < Wd. Proof. rewrite Wd_val; unfold L2; lra. Qed.\n")
    R.append("(* Traffic lower-bound hypothesis for phases whose compulsory bytes exceed")
    R.append("   the weight stream (verify at large ctx adds the committed-KV read). Same")
    R.append("   physics as Roofline.weight_traffic_lb, stated over total bytes. *)")
    R.append("Definition traffic_lb (B : R) (f : forward_exec) : Prop := B <= hbm_bytes f.\n")

    R.append("(* =============== (ii) lower bounds on THIS GPU =============== *)\n")
    R.append(f"(* Verify weight floor: {float(floor_v)*1000:.3f} ms exact; stated at ns precision. *)")
    R.append("Theorem gen_verify_weight_floor :")
    R.append("  forall f, respects_bandwidth BW f -> weight_traffic_lb L2 Wv f ->")
    R.append(f"  {frac_lit(floor_v_lo)} <= wall_time f.")
    R.append("Proof.")
    R.append("  intros f H1 H2.")
    R.append("  pose proof (verify_time_lower_bound BW L2 Wv f H1 H2) as H.")
    R.append("  rewrite Wv_val in H; unfold BW, L2 in *; lra.")
    R.append("Qed.\n")
    R.append(f"(* Verify floor at full context {CTX_FULL}: compulsory bytes = weights +")
    R.append("   clamped committed-KV unique reads, credited with the FULL L2 residency")
    R.append(f"   (weights and KV are disjoint allocations, so one shared credit is the")
    R.append(f"   weakest safe hypothesis): {float(floor_v_full)*1000:.3f} ms exact. *)")
    R.append(f"Theorem gen_verify_floor_full_ctx :")
    R.append("  forall f, respects_bandwidth BW f ->")
    R.append(f"  traffic_lb (IZR (GenTraffic.verify_min_traffic {CTX_FULL}) - L2) f ->")
    R.append(f"  {frac_lit(floor_v_full_lo)} <= wall_time f.")
    R.append("Proof.")
    R.append("  intros f H1 H2; unfold respects_bandwidth in H1; unfold traffic_lb in H2.")
    R.append(f"  rewrite GenTraffic.verify_min_traffic_at_{CTX_FULL} in H2.")
    R.append("  unfold BW, L2 in *; lra.")
    R.append("Qed.\n")
    R.append(f"(* Drafter step floor: {float(floor_d)*1e6:.1f} us exact. *)")
    R.append("Theorem gen_draft_step_floor :")
    R.append("  forall f, respects_bandwidth BW f -> weight_traffic_lb L2 Wd f ->")
    R.append(f"  {frac_lit(floor_d_lo)} <= wall_time f.")
    R.append("Proof.")
    R.append("  intros f H1 H2.")
    R.append("  pose proof (verify_time_lower_bound BW L2 Wd f H1 H2) as H.")
    R.append("  rewrite Wd_val in H; unfold BW, L2 in *; lra.")
    R.append("Qed.\n")

    R.append("(* Speed-of-light ceiling at the measured best acceptance")
    R.append(f"   (tokens/round <= {float(tpr_hi):.2f}, {ms['log']}):")
    R.append(f"   r <= tpr*BW/(Wv-L2) = {float(ceiling):.1f} tok/s exact. NOTE: this supersedes")
    R.append("   core Roofline.Measured.token_rate_ceiling_218 -- that instance used the")
    R.append("   road-to-50 W=18.3 GB estimate, which undercounted the bf16 attention")
    R.append("   stream; both are sound, this one is ~1.6x tighter. *)")
    R.append("Theorem gen_token_rate_ceiling :")
    R.append("  forall f T tpr r,")
    R.append("    respects_bandwidth BW f -> weight_traffic_lb L2 Wv f ->")
    R.append("    wall_time f <= T -> 0 <= tpr ->")
    R.append(f"    tpr <= {frac_lit(tpr_hi)} -> r * T = tpr ->")
    R.append(f"    r <= {frac_lit(ceiling_up)}.")
    R.append("Proof.")
    R.append("  intros f T tpr r H1 H2 HT Htpr Hub Hr.")
    R.append("  pose proof (speed_of_light BW L2 Wv f T tpr r BW_pos Wv_gt_L2 H1 H2 HT Htpr Hr) as Hsol.")
    R.append("  rewrite Wv_val in Hsol; unfold BW, L2 in *; lra.")
    R.append("Qed.\n")

    kv_full = kv_read(CTX_FULL)
    B_MID = 16
    ceil_b = ceil_frac(
        Fraction(B_MID) * tpr_hi * BW / (Fraction(wv) - L2 + Fraction(B_MID) * kv_full), 1
    )
    ceil_sat = ceil_frac(tpr_hi * BW / Fraction(kv_full), 1)
    R.append("(* BATCHED ceilings at full ctx. The B=1 hypothesis r*T = tpr in")
    R.append("   gen_token_rate_ceiling fixes a SINGLE stream; a batched forward")
    R.append("   amortizes the weight stream over B sequences while per-sequence KV")
    R.append("   does not amortize. Wkv = kv_read_bytes at full ctx. *)")
    R.append(f"Definition Wkv : R := IZR (GenTraffic.kv_read_bytes {CTX_FULL}).")
    R.append(f"Lemma Wkv_val : Wkv = {kv_full}.")
    R.append("Proof. unfold Wkv, GenTraffic.kv_read_bytes; cbv; lra. Qed.")
    R.append("Lemma Wkv_nonneg : 0 <= Wkv. Proof. rewrite Wkv_val; lra. Qed.")
    R.append("Lemma Wkv_pos : 0 < Wkv. Proof. rewrite Wkv_val; lra. Qed.\n")
    R.append(f"Theorem gen_token_rate_ceiling_batched_{B_MID} :")
    R.append("  forall f T tpr r,")
    R.append("    respects_bandwidth BW f ->")
    R.append(f"    batched_traffic_lb L2 Wv Wkv {B_MID} f ->")
    R.append("    wall_time f <= T -> 0 <= tpr ->")
    R.append(f"    tpr <= {frac_lit(tpr_hi)} -> r * T = {B_MID} * tpr ->")
    R.append(f"    r <= {frac_lit(ceil_b)}.")
    R.append("Proof.")
    R.append("  intros f T tpr r H1 H2 HT Htpr Hub Hr.")
    R.append(f"  pose proof (speed_of_light_batched BW L2 Wv Wkv {B_MID} f T tpr r")
    R.append("                BW_pos Wv_gt_L2 Wkv_nonneg (ltac:(lra)) H1 H2 HT Htpr Hr) as Hsol.")
    R.append("  rewrite Wv_val, Wkv_val in Hsol; unfold BW, L2 in *; lra.")
    R.append("Qed.\n")
    R.append("(* No batch size beats this: per-sequence KV traffic is the wall. *)")
    R.append("Theorem gen_token_rate_ceiling_any_batch :")
    R.append("  forall B f T tpr r,")
    R.append("    1 <= B ->")
    R.append("    respects_bandwidth BW f ->")
    R.append("    batched_traffic_lb L2 Wv Wkv B f ->")
    R.append("    wall_time f <= T -> 0 <= tpr ->")
    R.append(f"    tpr <= {frac_lit(tpr_hi)} -> r * T = B * tpr ->")
    R.append(f"    r <= {frac_lit(ceil_sat)}.")
    R.append("Proof.")
    R.append("  intros B f T tpr r HB H1 H2 HT Htpr Hub Hr.")
    R.append("  pose proof (batched_ceiling_saturates BW L2 Wv Wkv B f T tpr r")
    R.append("                BW_pos Wv_gt_L2 Wkv_pos HB H1 H2 HT Htpr Hr) as Hsat.")
    R.append("  rewrite Wkv_val in Hsat; unfold BW in *; lra.")
    R.append("Qed.\n")
    print(f"[gen] batched ceilings: B={B_MID} -> {float(ceil_b):.0f} tok/s, "
          f"any B -> {float(ceil_sat):.0f} tok/s (single-stream {float(ceiling_up):.0f})")

    R.append("(* Round-serial ceiling: our engine runs draft (K steps) then verify serially")
    R.append(f"   (chat_engine round loop), so T >= T_verify + {K}*T_draft_step. *)")
    R.append("Definition k_chain : R := " + str(K) + ".")
    R.append("Theorem gen_round_serial_ceiling :")
    R.append("  forall fv fd T tpr r,")
    R.append("    respects_bandwidth BW fv -> weight_traffic_lb L2 Wv fv ->")
    R.append("    respects_bandwidth BW fd -> weight_traffic_lb L2 Wd fd ->")
    R.append("    wall_time fv + k_chain * wall_time fd <= T ->")
    R.append(f"    0 <= tpr -> tpr <= {frac_lit(tpr_hi)} -> r * T = tpr ->")
    R.append(f"    r <= {frac_lit(serial_ceiling_up)}.")
    R.append("Proof.")
    R.append("  intros fv fd T tpr r H1 H2 H3 H4 HT Htpr Hub Hr.")
    R.append("  pose proof (verify_time_lower_bound BW L2 Wv fv H1 H2) as Hv.")
    R.append("  pose proof (verify_time_lower_bound BW L2 Wd fd H3 H4) as Hd.")
    R.append("  rewrite Wv_val in Hv; rewrite Wd_val in Hd.")
    R.append("  unfold BW, L2, k_chain in *.")
    R.append(f"  assert (HTmin : {frac_lit(tmin_lo)} <= T) by lra.")
    R.append("  assert (HTpos : 0 < T) by lra.")
    R.append("  assert (Hr0 : 0 <= r) by nra.")
    R.append(f"  assert (Hkey : r * ({frac_lit(tmin_lo)}) <= tpr) by nra.")
    R.append("  lra.")
    R.append("Qed.\n")

    R.append("(* =============== (iii) measured efficiency ratios =============== *)\n")
    R.append(f"(* Measured envelope: {ms['log']},")
    R.append(f"   {ms['n_summaries']} steady-state summaries (rounds >= {measured_cfg['min_rounds']}). *)")
    R.append(f"Definition measured_verify_lo_s : R := {frac_lit(ms['verify_lo'])}.")
    R.append(f"Definition measured_verify_hi_s : R := {frac_lit(ms['verify_hi'])}.")
    R.append(f"Definition measured_draft_lo_s  : R := {frac_lit(ms['draft_lo'])}.")
    R.append(f"Definition measured_draft_hi_s  : R := {frac_lit(ms['draft_hi'])}.")
    R.append(f"(* {ms['bench']}: {ms['agg_tokens']} tokens, aggregate decode rate (rounded down). *)")
    R.append(f"Definition measured_agg_tok_s   : R := {frac_lit(agg_lit)}.\n")

    R.append("(* Consistency tripwires: the physical floor must sit BELOW every measurement.")
    R.append("   If a manifest edit ever overcounts compulsory bytes, these two lemmas (and")
    R.append("   generation itself) fail -- the manifest cannot silently inflate. *)")
    R.append("Lemma gen_verify_floor_consistent : (Wv - L2) / BW <= measured_verify_lo_s.")
    R.append("Proof. rewrite Wv_val; unfold L2, BW, measured_verify_lo_s; lra. Qed.")
    R.append("Lemma gen_draft_floor_consistent :")
    R.append("  k_chain * ((Wd - L2) / BW) <= measured_draft_lo_s.")
    R.append("Proof. rewrite Wd_val; unfold L2, BW, k_chain, measured_draft_lo_s; lra. Qed.\n")

    rv_pct = float(r_verify)
    R.append(f"(* VERIFY is within {rv_pct:.2f}x of light: measured worst-case verify wall")
    R.append("   time vs the weight-stream floor. Further verify acceleration beyond this")
    R.append(f"   factor must either CUT BYTES (algorithm: e.g. quantize the bf16 attention")
    R.append("   stream, fp8 KV) or RAISE BW (hardware) -- no schedule change can pass it. *)")
    R.append("Theorem gen_verify_within_light_ratio :")
    R.append(f"  measured_verify_hi_s <= {frac_lit(r_verify)} * ((Wv - L2) / BW).")
    R.append("Proof. rewrite Wv_val; unfold measured_verify_hi_s, L2, BW; lra. Qed.\n")
    R.append(f"(* DRAFT chain (K={K}) is within {float(r_draft):.2f}x of light. *)")
    R.append("Theorem gen_draft_within_light_ratio :")
    R.append(f"  measured_draft_hi_s <= {frac_lit(r_draft)} * (k_chain * ((Wd - L2) / BW)).")
    R.append("Proof. rewrite Wd_val; unfold measured_draft_hi_s, k_chain, L2, BW; lra. Qed.\n")
    R.append(f"(* END-TO-END: the speed-of-light ceiling ({float(ceiling):.1f} tok/s at measured")
    R.append(f"   acceptance) is within {float(r_e2e):.2f}x of the measured aggregate decode rate")
    R.append(f"   ({float(agg_lit):.2f} tok/s) -- i.e. the whole engine sits within that factor")
    R.append("   of the bandwidth roof. *)")
    R.append("Theorem gen_e2e_within_light_ratio :")
    R.append(f"  {frac_lit(ceiling_up)} <= {frac_lit(r_e2e)} * measured_agg_tok_s.")
    R.append("Proof. unfold measured_agg_tok_s; lra. Qed.\n")

    host_hi = ms["host_hi"]
    round_hi = ms["verify_hi"] + ms["draft_hi"] + host_hi
    floor_round = floor_v + K * floor_d
    verify_gap = ms["verify_hi"] - floor_v
    draft_gap = ms["draft_hi"] - K * floor_d
    total_gap = verify_gap + draft_gap + host_hi
    verify_share_pct = math.floor(verify_gap / total_gap * 100)
    if BW_MEAS > 0:
        R.append("(* =============== bandwidth-hypothesis consistency =============== *)\n")
        R.append(f"(* The physics hypothesis BW = {float(BW)/1e12:.2f} TB/s must upper-bound anything a")
        R.append("   kernel actually achieves.  Best measured contiguous-read bandwidth on this")
        R.append(f"   GPU (in-tree sol_roofline harness, {machine.get('measured_read_bw_provenance','')}):")
        R.append(f"   {float(BW_MEAS)/1e9:.0f} GB/s.  Generation FAILS if this ever exceeds BW. *)")
        R.append(f"Definition measured_read_bw : R := {frac_lit(BW_MEAS)}.")
        R.append("Lemma bw_hypothesis_consistent : measured_read_bw <= BW.")
        R.append("Proof. unfold measured_read_bw, BW; lra. Qed.\n")
    R.append("(* =============== utilization headline =============== *)\n")
    R.append(f"(* Round-phase envelope (worst steady-state summary): verify {float(ms['verify_hi']*1000):.2f} ms +")
    R.append(f"   draft {float(ms['draft_hi']*1000):.2f} ms + host residue {float(host_hi*1000):.2f} ms")
    R.append(f"   (round wall = tokens/round x ms/token, minus the GPU phases) = {float(round_hi*1000):.2f} ms.")
    R.append(f"   Physical floor of the same round shape: verify {float(floor_v*1000):.2f} + K x draft-step")
    R.append(f"   {float(K*floor_d*1000):.2f} = {float(floor_round*1000):.2f} ms. *)")
    R.append(f"Definition measured_host_hi_s : R := {frac_lit(host_hi)}.")
    R.append("Definition measured_round_hi_s : R :=")
    R.append("  measured_verify_hi_s + measured_draft_hi_s + measured_host_hi_s.")
    R.append("Definition floor_round_s : R := (Wv - L2) / BW + k_chain * ((Wd - L2) / BW).\n")
    R.append("(* WHERE THE GAP LIVES -- an exact accounting, not an estimate. The three")
    R.append("   phase gaps sum to the whole round gap by construction; their measured")
    R.append(f"   values: verify-in-kernel {float(verify_gap*1000):.2f} ms ({float(verify_gap/total_gap*100):.1f}%), draft chain")
    R.append(f"   {float(draft_gap*1000):.2f} ms ({float(draft_gap/total_gap*100):.1f}%), host {float(host_hi*1000):.2f} ms ({float(host_hi/total_gap*100):.1f}%). *)")
    R.append("Definition verify_gap_s : R := measured_verify_hi_s - (Wv - L2) / BW.")
    R.append("Definition draft_gap_s  : R := measured_draft_hi_s - k_chain * ((Wd - L2) / BW).")
    R.append("Theorem gen_gap_decomposition :")
    R.append("  measured_round_hi_s - floor_round_s =")
    R.append("  verify_gap_s + draft_gap_s + measured_host_hi_s.")
    R.append("Proof.")
    R.append("  unfold measured_round_hi_s, floor_round_s, verify_gap_s, draft_gap_s. lra.")
    R.append("Qed.\n")
    R.append(f"(* The verify in-kernel gap is at least {verify_share_pct}% of everything separating")
    R.append("   this engine from its bandwidth floor: closing anything else is bounded")
    R.append(f"   by the remaining {100-verify_share_pct}%. *)")
    R.append("Theorem gen_gap_lives_in_verify :")
    R.append(f"  {verify_share_pct} / 100 * (measured_round_hi_s - floor_round_s) <= verify_gap_s.")
    R.append("Proof.")
    R.append("  rewrite gen_gap_decomposition.")
    R.append("  unfold verify_gap_s, draft_gap_s, measured_host_hi_s,")
    R.append("    measured_verify_hi_s, measured_draft_hi_s, k_chain, L2, BW.")
    R.append("  rewrite Wv_val, Wd_val. lra.")
    R.append("Qed.\n")
    R.append("(* THE HEADLINE, machine-checked: any schedule of this model on this GPU at")
    R.append(f"   the measured acceptance is capped at {frac_lit(ceiling_up)} tok/s, and that cap is at most")
    R.append(f"   {frac_lit(r_e2e)}x the {float(agg_lit):.2f} tok/s the merged engine already delivers -- we are")
    R.append(f"   within {frac_lit(r_e2e)}x of light, and gen_gap_lives_in_verify locates >= {verify_share_pct}% of")
    R.append("   the remaining distance in verify in-kernel efficiency. *)")
    R.append("Theorem gen_util_headline :")
    R.append("  forall f T tpr r,")
    R.append("    respects_bandwidth BW f -> weight_traffic_lb L2 Wv f ->")
    R.append("    wall_time f <= T -> 0 <= tpr ->")
    R.append(f"    tpr <= {frac_lit(tpr_hi)} -> r * T = tpr ->")
    R.append(f"    r <= {frac_lit(r_e2e)} * measured_agg_tok_s.")
    R.append("Proof.")
    R.append("  intros f T tpr r H1 H2 HT Htpr Hub Hr.")
    R.append("  pose proof (gen_token_rate_ceiling f T tpr r H1 H2 HT Htpr Hub Hr).")
    R.append("  pose proof gen_e2e_within_light_ratio. lra.")
    R.append("Qed.")

    with open(os.path.join(ROCQ, "GenRoofline.v"), "w") as f:
        f.write("\n".join(R) + "\n")

    summary = {
        "generated": now,
        "verify_weight_bytes": wv,
        "verify_weight_bytes_formula": big_sum,
        "misc_small_bytes": misc,
        "draft_step_weight_bytes": wd,
        "verify_min_traffic_at_256": min_traffic_256,
        "verify_min_traffic_at_full_ctx": min_traffic_full,
        "floor_verify_ms": float(floor_v * 1000),
        "floor_verify_full_ctx_ms": float(floor_v_full * 1000),
        "floor_draft_step_us": float(floor_d * 10**6),
        "token_ceiling_tok_s": float(ceiling),
        "round_serial_ceiling_tok_s": float(serial_ceiling),
        "measured": {
            "verify_ms": [float(ms["verify_lo"] * 1000), float(ms["verify_hi"] * 1000)],
            "draft_ms_round": [float(ms["draft_lo"] * 1000), float(ms["draft_hi"] * 1000)],
            "tokens_per_round": [float(ms["tpr_lo"]), float(ms["tpr_hi"])],
            "agg_tok_s": float(ms["agg_tok_s"]),
        },
        "ratios": {
            "verify_vs_light": float(r_verify),
            "draft_vs_light": float(r_draft),
            "e2e_ceiling_vs_measured": float(r_e2e),
        },
        "gap_ms": {
            "round_hi": float(round_hi * 1000),
            "floor_round": float(floor_round * 1000),
            "verify_gap": float(verify_gap * 1000),
            "draft_gap": float(draft_gap * 1000),
            "host_hi": float(host_hi * 1000),
            "verify_share_pct": verify_share_pct,
        },
    }
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    with open(os.path.join(HERE, "out", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[gen] === roofline table (regenerated) ===")
    print(f"[gen] verify weight stream      {wv/1e9:10.3f} GB  -> floor {float(floor_v)*1000:7.3f} ms")
    print(f"[gen] verify traffic @262144    {min_traffic_full/1e9:10.3f} GB  -> floor {float(floor_v_full)*1000:7.3f} ms")
    print(f"[gen] draft step weight stream  {wd/1e9:10.3f} GB  -> floor {float(floor_d)*1e6:7.1f} us")
    print(f"[gen] measured verify           {float(ms['verify_lo'])*1000:.2f}-{float(ms['verify_hi'])*1000:.2f} ms   within {float(r_verify):.2f}x of light")
    print(f"[gen] measured draft (K={K})      {float(ms['draft_lo'])*1000:.2f}-{float(ms['draft_hi'])*1000:.2f} ms     within {float(r_draft):.2f}x of light")
    print(f"[gen] token ceiling @tpr<={float(tpr_hi):.2f}  {float(ceiling):8.1f} tok/s (serial {float(serial_ceiling):.1f}); measured agg {float(ms['agg_tok_s']):.2f}")
    print("[gen] wrote GenTraffic.v GenRoofline.v gen/out/summary.json")

if __name__ == "__main__":
    main()
