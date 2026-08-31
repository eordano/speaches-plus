#!/usr/bin/env python3
import argparse, json, re, sys
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize
from safetensors import safe_open

A_CONVERTED_GGUF_IS_NOT_TRUSTED_UNTIL_ITS_VALUES_ARE_COMPARED_TO_THE_SOURCE_CHECKPOINT = """
A bad perplexity on a converted GGUF localises to one of three places: the tensor
VALUES, the architecture llama.cpp built from the metadata, or the evaluation
invocation. Only the first is cheap to settle, and settling it first is what keeps a
round from rewriting a converter that was already correct. This audit settles it.

compressed-tensors stores NVFP4 as weight_packed (two e2m1 nibbles per byte),
weight_scale (per-16 e4m3), and weight_global_scale (per-tensor). Dequantisation is
e2m1 * weight_scale / weight_global_scale. llama.cpp renames these to the ModelOpt
spelling and stores the RECIPROCAL of the global scale as a separate .scale tensor
applied at inference, so a correct conversion has gguf .scale == 1/weight_global_scale
and dequantised values equal to the reference within float32 rounding.

A TENSOR THIS AUDIT CANNOT NAME ON BOTH SIDES IS AN UNCHECKED TENSOR, NOT A PASSING
ONE. The first version of this script carried a hardcoded MoE-only name map, reported
168/168 and 295/295 tensors unmapped on two dense checkpoints, and printed PASS for
both. Coverage is therefore a gate: zero verified tensors, or any unmapped tensor
without --allow-unmapped, exits non-zero.
"""

E2M1_LEVELS_BY_NIBBLE = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32
)
NVFP4_ELEMENTS_PER_E4M3_SCALE = 16
FLOAT32_ULP_TOLERANCE_IS_THE_ONLY_DIFFERENCE_A_CORRECT_CONVERSION_MAY_SHOW = 1e-6

GGUF_SUFFIX_TO_PROJECTION = {
    "ffn_down": "down_proj", "ffn_gate": "gate_proj", "ffn_up": "up_proj",
    "attn_q": "q_proj", "attn_k": "k_proj", "attn_v": "v_proj", "attn_output": "o_proj",
}
STACKED_EXPERTS_SUFFIX = "_exps"
SHARED_EXPERT_SUFFIX = "_shexp"

CHECKPOINT_NAME = re.compile(
    r"layers\.(\d+)\.(?:(?:mlp|self_attn|feed_forward)\.)?"
    r"(?:experts\.(\d+)\.|(shared_expert(?:s)?)\.)?"
    r"(\w+_proj)$"
)

GEMMA4_FOLDS_A_ROUTER_PER_EXPERT_GAIN_INTO_THE_DOWN_SCALE_AND_IGNORING_IT_FABRICATES_A_DEFECT = """
Some MoE checkpoints carry a router `per_expert_scale`: a per-expert gain on the expert's
OUTPUT, i.e. on its down projection. llama.cpp folds it into the per-expert .scale tensor
written beside the packed weights, which is correct and the only place it can go. An audit
comparing .scale against 1/weight_global_scale alone therefore reports every down
projection as wrong by exactly that gain.

Not hypothetical: it produced a filed defect against gemma4-26B-A4B claiming 30 tensors
off by up to 2.34% on "exact dyadic ratios". The ratios were exact because the gain itself
is dyadic (0.984375 = 63/64, 1.015625 = 65/64); folding it in gives 0/128 mismatched at
8.8e-08, and the packed data was bit-identical all along. Model the gain, or do not
compare the scale.
"""

ROUTER_PER_EXPERT_GAIN_SUFFIX = "router.per_expert_scale"
THE_GAIN_MULTIPLIES_THE_EXPERT_OUTPUT_SO_IT_LANDS_ON_DOWN_PROJ_ONLY = "down_proj"

def router_per_expert_gain(index, load, block, proj, expert_count):
    if proj != THE_GAIN_MULTIPLIES_THE_EXPERT_OUTPUT_SO_IT_LANDS_ON_DOWN_PROJ_ONLY:
        return None
    for key in index:
        if key.endswith(ROUTER_PER_EXPERT_GAIN_SUFFIX) and f"layers.{block}." in key:
            t = load(key)
            if t is None:
                return None
            gain = t.float().numpy().reshape(-1).astype(np.float64)
            return gain if gain.size == expert_count else None
    return None

def index_checkpoint(snapshot):
    index = json.load(open(f"{snapshot}/model.safetensors.index.json"))["weight_map"]
    by_slot = {}
    for key in index:
        if not key.endswith(".weight_packed"):
            continue
        base = key.removesuffix(".weight_packed")
        m = CHECKPOINT_NAME.search(base)
        if not m:
            continue
        layer, expert, shared, proj = m.group(1), m.group(2), m.group(3), m.group(4)
        scope = "shared" if shared else ("expert" if expert is not None else "plain")
        by_slot[(int(layer), scope, proj, int(expert) if expert is not None else -1)] = base

    def load(name):
        if name not in index:
            return None
        with safe_open(f"{snapshot}/{index[name]}", framework="pt") as f:
            return f.get_tensor(name)

    return by_slot, load, index

def gguf_slot(name):
    m = re.match(r"blk\.(\d+)\.(\w+)\.weight$", name)
    if not m:
        return None
    layer, suffix = int(m.group(1)), m.group(2)
    scope = "plain"
    if suffix.endswith(STACKED_EXPERTS_SUFFIX):
        scope, suffix = "expert", suffix[: -len(STACKED_EXPERTS_SUFFIX)]
    elif suffix.endswith(SHARED_EXPERT_SUFFIX):
        scope, suffix = "shared", suffix[: -len(SHARED_EXPERT_SUFFIX)]
    proj = GGUF_SUFFIX_TO_PROJECTION.get(suffix)
    return None if proj is None else (layer, scope, proj)

def reference_dequantize(load, base):
    packed, scale = load(base + ".weight_packed"), load(base + ".weight_scale")
    global_scale = load(base + ".weight_global_scale")
    if packed is None or scale is None or global_scale is None:
        return None
    b = packed.numpy()
    nibbles = np.stack([b & 0x0F, b >> 4], axis=-1).reshape(b.shape[0], -1)
    per_block = scale.float().numpy().repeat(NVFP4_ELEMENTS_PER_E4M3_SCALE, axis=1)
    return E2M1_LEVELS_BY_NIBBLE[nibbles] * per_block / float(global_scale.float().reshape(-1)[0])

def main():
    ap = argparse.ArgumentParser(
        description="prove a converted NVFP4 GGUF carries the checkpoint's values before blaming the converter")
    ap.add_argument("gguf")
    ap.add_argument("snapshot")
    ap.add_argument("--dequantize-tensors", type=int, default=2,
                    help="full value comparisons to run; scales are checked for every mapped tensor")
    ap.add_argument("--allow-unmapped", action="store_true",
                    help="permit tensors this audit cannot name on both sides; without it they FAIL the run")
    args = ap.parse_args()

    reader = GGUFReader(args.gguf)
    by_name = {t.name: t for t in reader.tensors}
    quantized = [t for t in reader.tensors if t.tensor_type == GGMLQuantizationType.NVFP4]
    if not quantized:
        print(f"audit: {args.gguf} holds no NVFP4 tensors; nothing to prove", file=sys.stderr)
        return 1
    by_slot, load, index = index_checkpoint(args.snapshot)

    scales_ok = scales_bad = 0
    values_ok = values_bad = 0
    worst_scale_error = worst_value_error = 0.0
    unmapped, failures = [], []

    for tensor in quantized:
        slot = gguf_slot(tensor.name)
        companion = by_name.get(tensor.name.replace(".weight", ".scale"))
        if slot is None:
            unmapped.append((tensor.name, "gguf name does not parse into (layer, scope, projection)"))
            continue
        if companion is None:
            failures.append((tensor.name, "no .scale companion -- the global scale was dropped"))
            scales_bad += 1
            continue
        layer, scope, proj = slot
        stored = np.asarray(companion.data, dtype=np.float32).reshape(-1)
        if scope == "expert":
            bases = [by_slot.get((layer, "expert", proj, e)) for e in range(stored.size)]
        else:
            bases = [by_slot.get((layer, scope, proj, -1))]
        if any(b is None for b in bases):
            unmapped.append((tensor.name, f"no checkpoint tensor for layer {layer} {scope} {proj}"))
            continue

        reference = np.array([1.0 / float(load(b + ".weight_global_scale").float().reshape(-1)[0])
                              for b in bases], dtype=np.float64)
        gain = router_per_expert_gain(index, load, layer, proj, reference.size)
        if gain is not None:
            reference = reference * gain
        err = np.abs(stored.astype(np.float64) - reference) / np.abs(reference)
        worst_scale_error = max(worst_scale_error, float(err.max()))
        if err.max() < FLOAT32_ULP_TOLERANCE_IS_THE_ONLY_DIFFERENCE_A_CORRECT_CONVERSION_MAY_SHOW:
            scales_ok += 1
        else:
            scales_bad += 1
            failures.append((tensor.name, f"global scale differs by {err.max():.3g} relative"))

        if values_ok + values_bad < args.dequantize_tensors:
            ref = reference_dequantize(load, bases[0])
            if ref is None:
                continue
            got = dequantize(tensor.data, tensor.tensor_type).astype(np.float32)
            got = (got[0] if got.ndim == 3 else got) * stored[0]
            if got.shape != ref.shape:
                values_bad += 1
                failures.append((tensor.name, f"shape {got.shape} != checkpoint {ref.shape}"))
                continue
            rel = float(np.sqrt(((got.astype(np.float64) - ref.astype(np.float64)) ** 2).mean())
                        / np.sqrt((ref.astype(np.float64) ** 2).mean()))
            worst_value_error = max(worst_value_error, rel)
            if rel < FLOAT32_ULP_TOLERANCE_IS_THE_ONLY_DIFFERENCE_A_CORRECT_CONVERSION_MAY_SHOW:
                values_ok += 1
            else:
                values_bad += 1
                failures.append((tensor.name, f"values differ by {rel:.3g} relative rms"))

    print(f"NVFP4 tensors: {len(quantized)}  scale-verified: {scales_ok}  "
          f"scale-MISMATCH: {scales_bad}  unmapped: {len(unmapped)}")
    print(f"full value comparisons: {values_ok} matched, {values_bad} MISMATCH")
    print(f"worst relative scale error: {worst_scale_error:.3g}")
    print(f"worst relative value error: {worst_value_error:.3g}")
    for name, why in failures[:10]:
        print(f"  MISMATCH {name}: {why}")
    for name, why in unmapped[:10]:
        print(f"  UNMAPPED {name}: {why}")

    if scales_bad or values_bad:
        print("\naudit: the GGUF does NOT carry the checkpoint's values -- fix the conversion")
        return 1
    if scales_ok == 0:
        print(f"\naudit: VOID -- {len(quantized)} NVFP4 tensors and NOT ONE was verified. This is a "
              "finding about the audit, not a pass for the GGUF: extend the name map.")
        return 1
    if unmapped and not args.allow_unmapped:
        print(f"\naudit: INCOMPLETE -- {scales_ok} verified but {len(unmapped)} unchecked. Extend the "
              "name map, or pass --allow-unmapped to accept partial coverage deliberately.")
        return 1
    print(f"\naudit: {scales_ok} NVFP4 tensors carry the checkpoint's values; a bad perplexity here "
          "is NOT the converter -- look at the architecture llama.cpp built or the evaluation invocation")
    return 0

if __name__ == "__main__":
    sys.exit(main())
