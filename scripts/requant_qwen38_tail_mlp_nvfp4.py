import argparse
import json
import os
import re
import struct
import sys

import numpy as np

NVFP4_BLOCK = 16
E2M1_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E2M1_LOWER_ON_TIE_MIDPOINTS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
UE4M3_SUBNORMAL_STEP = np.float32(0.001953125)
STORED_GLOBAL_IS_448_TIMES_6_OVER_AMAX = 448.0 * 6.0
MLP_PROJ_RE = re.compile(r"^(model\.language_model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj))\.weight$")
COPY_CHUNK = 1 << 26


def decode_ue4m3_lut():
    lut = np.zeros(256, dtype=np.float32)
    for byte in range(0x7F):
        exp = (byte >> 3) & 0x0F
        mant = float(byte & 0x07)
        if exp == 0:
            lut[byte] = mant * UE4M3_SUBNORMAL_STEP
        else:
            lut[byte] = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
    return lut


DEC_UE4M3 = decode_ue4m3_lut()
UE4M3_CODES = np.arange(0x7F, dtype=np.uint8)
UE4M3_SORT = np.argsort(DEC_UE4M3[:0x7F], kind="stable")
UE4M3_SORTED_VALS = DEC_UE4M3[:0x7F][UE4M3_SORT]
UE4M3_SORTED_CODES = UE4M3_CODES[UE4M3_SORT]
UE4M3_MIDPOINTS = (UE4M3_SORTED_VALS[:-1] + UE4M3_SORTED_VALS[1:]) / 2.0

SCALE_GRID_IS_THE_NONZERO_UE4M3_CODES_A_ZERO_SCALE_KILLS_THE_BLOCK = UE4M3_SORTED_VALS > 0.0
SCALE_GRID_VALS = UE4M3_SORTED_VALS[SCALE_GRID_IS_THE_NONZERO_UE4M3_CODES_A_ZERO_SCALE_KILLS_THE_BLOCK]
SCALE_GRID_CODES = UE4M3_SORTED_CODES[SCALE_GRID_IS_THE_NONZERO_UE4M3_CODES_A_ZERO_SCALE_KILLS_THE_BLOCK]
SCALE_GRID_MIDPOINTS = (SCALE_GRID_VALS[:-1] + SCALE_GRID_VALS[1:]) / 2.0
CLOSED_FORM_STEPS_BELOW_RTN_BECAUSE_CLIPPING_THE_BLOCK_AMAX_BUYS_INTERIOR_RESOLUTION = 6
CLOSED_FORM_STEPS_ABOVE_RTN_BECAUSE_THE_E2M1_LADDER_STEPS_HALF_BELOW_TWO_AND_TWO_ABOVE_FOUR_SO_PARKING_THE_BLOCK_AMAX_ON_SIX_IS_THE_WORST_PLACE_FOR_IT = 10
GLOBAL_SWEEP_MUST_REACH_ONE_HALF_BECAUSE_FOUR_FORTY_EIGHT_OVER_AMAX_PINS_THE_TOP_BLOCK_AT_THE_GRID_CEILING_AND_ITS_OPTIMUM_IS_COARSER = (
    1.0,
    0.9,
    0.8,
    0.7,
    0.6,
    0.5,
)
CLOSED_FORM_ROW_CHUNK = 2048


def encode_ue4m3_nearest(x):
    x = np.asarray(x, dtype=np.float32)
    clamped = np.clip(np.nan_to_num(x, nan=0.0, posinf=448.0, neginf=0.0), 0.0, 448.0)
    idx = np.searchsorted(UE4M3_MIDPOINTS, clamped, side="right")
    return UE4M3_SORTED_CODES[idx]


def decode_fp8_e4m3fn_lut():
    lut = np.zeros(256, dtype=np.float32)
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exp = (byte >> 3) & 0x0F
        mant = float(byte & 0x07)
        if exp == 0:
            val = mant / 8.0 * (2.0 ** -6)
        elif exp == 15 and (byte & 0x07) == 0x07:
            val = np.nan
        else:
            val = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
        lut[byte] = sign * val
    return lut


DEC_FP8 = decode_fp8_e4m3fn_lut()


def bf16_bytes_to_f32(buf):
    u16 = np.frombuffer(buf, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)


def encode_e2m1_lower_on_tie(scaled):
    mag = np.abs(scaled)
    code = np.searchsorted(E2M1_LOWER_ON_TIE_MIDPOINTS, mag, side="left").astype(np.uint8)
    sign = (np.signbit(scaled)).astype(np.uint8) << 3
    return sign | code


def quantize_rows_nvfp4(w_f32, stored_global):
    rows, cols = w_f32.shape
    assert cols % NVFP4_BLOCK == 0, f"cols {cols} not multiple of {NVFP4_BLOCK}"
    blocks = w_f32.reshape(rows, cols // NVFP4_BLOCK, NVFP4_BLOCK)
    amax = np.max(np.abs(blocks), axis=2)
    local = np.where(amax == 0.0, np.float32(1.0), amax / np.float32(6.0))
    scale_bytes = encode_ue4m3_nearest(np.float32(stored_global) * local)
    decoded = DEC_UE4M3[scale_bytes]
    inv = np.where(decoded == 0.0, np.float32(1.0), np.float32(stored_global) / decoded)
    scaled = np.clip(blocks * inv[:, :, None], -6.0, 6.0)
    nib = encode_e2m1_lower_on_tie(scaled)
    flat = nib.reshape(rows, cols)
    packed = (flat[:, 0::2] | (flat[:, 1::2] << 4)).astype(np.uint8)
    return packed, scale_bytes.astype(np.uint8)


def pack_nibbles(nib, rows, cols):
    flat = nib.reshape(rows, cols)
    return (flat[:, 0::2] | (flat[:, 1::2] << 4)).astype(np.uint8)


def dequantize_blocks_under_scale(blocks, decoded, stored_global):
    inv = np.where(decoded == 0.0, np.float32(1.0), np.float32(stored_global) / decoded)
    scaled = np.clip(blocks * inv[..., None], -6.0, 6.0)
    nib = encode_e2m1_lower_on_tie(scaled)
    mag = E2M1_VALUES[nib & 0x07]
    signed = np.where(nib & 0x08, -mag, mag)
    return nib, signed * (decoded[..., None] / np.float32(stored_global))


def rtn_scale_grid_index(blocks, stored_global):
    amax = np.max(np.abs(blocks), axis=-1)
    base = np.float32(stored_global) * np.where(amax == 0.0, np.float32(1.0), amax / np.float32(6.0))
    return np.searchsorted(SCALE_GRID_MIDPOINTS, np.clip(base, 0.0, 448.0), side="right")


def closed_form_scale_grid_index(blocks, stored_global):
    idx0 = rtn_scale_grid_index(blocks, stored_global)
    best_idx = None
    best_err = None
    lo = -CLOSED_FORM_STEPS_BELOW_RTN_BECAUSE_CLIPPING_THE_BLOCK_AMAX_BUYS_INTERIOR_RESOLUTION
    hi = CLOSED_FORM_STEPS_ABOVE_RTN_BECAUSE_THE_E2M1_LADDER_STEPS_HALF_BELOW_TWO_AND_TWO_ABOVE_FOUR_SO_PARKING_THE_BLOCK_AMAX_ON_SIX_IS_THE_WORST_PLACE_FOR_IT
    for step in range(lo, hi + 1):
        idx = np.clip(idx0 + step, 0, len(SCALE_GRID_VALS) - 1)
        _, deq = dequantize_blocks_under_scale(blocks, SCALE_GRID_VALS[idx], stored_global)
        err = np.sum((deq.astype(np.float64) - blocks.astype(np.float64)) ** 2, axis=-1)
        if best_err is None:
            best_err, best_idx = err, idx
            continue
        take = err < best_err
        best_err = np.where(take, err, best_err)
        best_idx = np.where(take, idx, best_idx)
    return best_idx, best_err


def closed_form_block_error(w_f32, stored_global):
    rows, cols = w_f32.shape
    total = 0.0
    for start in range(0, rows, CLOSED_FORM_ROW_CHUNK):
        chunk = w_f32[start : start + CLOSED_FORM_ROW_CHUNK]
        blocks = chunk.reshape(chunk.shape[0], cols // NVFP4_BLOCK, NVFP4_BLOCK)
        _, err = closed_form_scale_grid_index(blocks, stored_global)
        total += float(np.sum(err))
    return total


def quantize_rows_nvfp4_closed_form(w_f32, stored_global):
    rows, cols = w_f32.shape
    assert cols % NVFP4_BLOCK == 0, f"cols {cols} not multiple of {NVFP4_BLOCK}"
    packed_parts = []
    scale_parts = []
    for start in range(0, rows, CLOSED_FORM_ROW_CHUNK):
        chunk = w_f32[start : start + CLOSED_FORM_ROW_CHUNK]
        n = chunk.shape[0]
        blocks = chunk.reshape(n, cols // NVFP4_BLOCK, NVFP4_BLOCK)
        idx, _ = closed_form_scale_grid_index(blocks, stored_global)
        nib, _ = dequantize_blocks_under_scale(blocks, SCALE_GRID_VALS[idx], stored_global)
        packed_parts.append(pack_nibbles(nib, n, cols))
        scale_parts.append(SCALE_GRID_CODES[idx].astype(np.uint8))
    return np.concatenate(packed_parts, axis=0), np.concatenate(scale_parts, axis=0)


def quantize_rows_nvfp4_grid_offset(w_f32, stored_global, offset):
    rows, cols = w_f32.shape
    packed_parts = []
    scale_parts = []
    for start in range(0, rows, CLOSED_FORM_ROW_CHUNK):
        chunk = w_f32[start : start + CLOSED_FORM_ROW_CHUNK]
        n = chunk.shape[0]
        blocks = chunk.reshape(n, cols // NVFP4_BLOCK, NVFP4_BLOCK)
        idx = np.clip(
            rtn_scale_grid_index(blocks, stored_global) + offset, 0, len(SCALE_GRID_VALS) - 1
        )
        nib, _ = dequantize_blocks_under_scale(blocks, SCALE_GRID_VALS[idx], stored_global)
        packed_parts.append(pack_nibbles(nib, n, cols))
        scale_parts.append(SCALE_GRID_CODES[idx].astype(np.uint8))
    return np.concatenate(packed_parts, axis=0), np.concatenate(scale_parts, axis=0)


def choose_global_scale(w_f32, amax_tensor, mode, sample_rows):
    g0 = STORED_GLOBAL_IS_448_TIMES_6_OVER_AMAX / amax_tensor
    if mode != "closed-form":
        return g0, 1.0, None
    stride = max(1, w_f32.shape[0] // sample_rows)
    probe = np.ascontiguousarray(w_f32[::stride])
    best = None
    table = []
    for factor in GLOBAL_SWEEP_MUST_REACH_ONE_HALF_BECAUSE_FOUR_FORTY_EIGHT_OVER_AMAX_PINS_THE_TOP_BLOCK_AT_THE_GRID_CEILING_AND_ITS_OPTIMUM_IS_COARSER:
        err = closed_form_block_error(probe, g0 * factor)
        table.append((factor, err))
        if best is None or err < best[1]:
            best = (factor, err)
    return g0 * best[0], best[0], table


def roundtrip_check(w_f32, packed, scale_bytes, stored_global, module):
    rows, cols = w_f32.shape
    lo = packed & 0x0F
    hi = packed >> 4
    nib = np.empty((rows, cols), dtype=np.uint8)
    nib[:, 0::2] = lo
    nib[:, 1::2] = hi
    vals = E2M1_VALUES[nib & 0x07] * np.where(nib & 0x08, np.float32(-1.0), np.float32(1.0))
    deq = vals.reshape(rows, cols // NVFP4_BLOCK, NVFP4_BLOCK) * (
        DEC_UE4M3[scale_bytes][:, :, None] / np.float32(stored_global)
    )
    deq = deq.reshape(rows, cols)
    ref_rms = float(np.sqrt(np.mean(w_f32.astype(np.float64) ** 2)))
    err_rms = float(np.sqrt(np.mean((deq.astype(np.float64) - w_f32.astype(np.float64)) ** 2)))
    rel = err_rms / ref_rms if ref_rms > 0 else 0.0
    print(f"  {module}: nvfp4 rms(dW)/rms(W)={rel:.4f}", flush=True)
    assert rel < 0.30, f"{module}: requant relative rms error {rel} implausibly high"
    return rel


class SrcShard:
    def __init__(self, path):
        self.f = open(path, "rb")
        header_len = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(header_len))
        self.data_base = 8 + header_len

    def raw(self, name):
        off = self.header[name]["data_offsets"]
        self.f.seek(self.data_base + off[0])
        return self.f.read(off[1] - off[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--scale-fit", choices=("rtn", "closed-form", "grid-offset"), default="rtn")
    ap.add_argument("--global-sweep-rows", type=int, default=1024)
    ap.add_argument(
        "--grid-offset-detunes-the-block-scale-to-price-ppl-against-weight-mse", type=int, default=1
    )
    args = ap.parse_args()
    src, dst = args.src, args.dst
    os.makedirs(dst, exist_ok=True)

    shard = SrcShard(os.path.join(src, "model.safetensors"))
    hdr = shard.header
    names = [k for k in hdr if k != "__metadata__"]

    fp8_modules = []
    nvfp4_input_globals = {"gate_proj": [], "up_proj": [], "down_proj": []}
    for name in names:
        m = MLP_PROJ_RE.match(name)
        if not m:
            continue
        module, layer, proj = m.group(1), int(m.group(2)), m.group(3)
        if hdr[name]["dtype"] == "F8_E4M3" and hdr.get(f"{module}.weight_scale", {}).get("dtype") == "BF16":
            fp8_modules.append((module, layer, proj))
    for name in names:
        m = re.match(r"^model\.language_model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.input_global_scale$", name)
        if m:
            nvfp4_input_globals[m.group(1)].append(struct.unpack("<f", shard.raw(name))[0])

    fp8_layers = sorted({l for _, l, _ in fp8_modules})
    assert len(fp8_modules) == 24 and len(fp8_layers) == 8, (
        f"expected exactly 8 fp8 MLP layers x 3 projections, found layers={fp8_layers} modules={len(fp8_modules)}"
    )
    print(f"fp8 MLP layers to requantize: {fp8_layers}", flush=True)

    synthesized_input_global = {
        proj: min(vals) for proj, vals in nvfp4_input_globals.items()
    }
    print(f"synthesized input_global_scale (min over {len(nvfp4_input_globals['gate_proj'])} nvfp4 layers, "
          f"smallest stored global assumes the largest activation amax seen anywhere): "
          f"{synthesized_input_global}", flush=True)

    replaced = {}
    dropped = set()
    for module, layer, proj in fp8_modules:
        wname = f"{module}.weight"
        sname = f"{module}.weight_scale"
        rows, cols = hdr[wname]["shape"]
        w_fp8 = np.frombuffer(shard.raw(wname), dtype=np.uint8).reshape(rows, cols)
        row_scale = bf16_bytes_to_f32(shard.raw(sname)).reshape(rows, 1)
        w = DEC_FP8[w_fp8] * row_scale
        assert np.isfinite(w).all(), f"{module}: non-finite dequantized fp8 weights"
        amax = float(np.max(np.abs(w)))
        assert amax > 0.0, f"{module}: all-zero weight"
        stored_global, factor, sweep = choose_global_scale(
            w, amax, args.scale_fit, args.global_sweep_rows
        )
        if sweep is not None:
            best = min(sweep, key=lambda t: t[1])[1]
            print(
                f"  {module}: global sweep "
                + " ".join(f"{f:.3f}:{e / best:.5f}" for f, e in sweep)
                + f" -> factor {factor:.3f}",
                flush=True,
            )
        if args.scale_fit != "rtn":
            if args.scale_fit == "closed-form":
                packed, scale_bytes = quantize_rows_nvfp4_closed_form(w, stored_global)
            else:
                packed, scale_bytes = quantize_rows_nvfp4_grid_offset(
                    w,
                    stored_global,
                    args.grid_offset_detunes_the_block_scale_to_price_ppl_against_weight_mse,
                )
            rtn_packed, rtn_scales = quantize_rows_nvfp4(
                w, STORED_GLOBAL_IS_448_TIMES_6_OVER_AMAX / amax
            )
            rtn_rel = roundtrip_check(
                w,
                rtn_packed,
                rtn_scales,
                STORED_GLOBAL_IS_448_TIMES_6_OVER_AMAX / amax,
                f"{module} [rtn-reference]",
            )
            del rtn_packed, rtn_scales
        else:
            packed, scale_bytes = quantize_rows_nvfp4(w, stored_global)
            rtn_rel = None
        rel = roundtrip_check(w, packed, scale_bytes, stored_global, module)
        if rtn_rel is not None:
            print(
                f"  {module}: {args.scale_fit} rel_rms {rel:.6f} vs rtn {rtn_rel:.6f} "
                f"mse_ratio {(rel / rtn_rel) ** 2:.5f}",
                flush=True,
            )
        replaced[wname] = [
            (f"{module}.weight_packed", "U8", [rows, cols // 2], packed.tobytes()),
            (f"{module}.weight_scale", "F8_E4M3", [rows, cols // NVFP4_BLOCK], scale_bytes.tobytes()),
            (f"{module}.weight_global_scale", "F32", [1], struct.pack("<f", stored_global)),
            (f"{module}.input_global_scale", "F32", [1], struct.pack("<f", synthesized_input_global[proj])),
        ]
        dropped.add(sname)
        del w, w_fp8, packed, scale_bytes

    out_entries = []
    for name in names:
        if name in dropped:
            continue
        if name in replaced:
            for new_name, dtype, shape, data in replaced[name]:
                out_entries.append((new_name, dtype, shape, data))
        else:
            nbytes = hdr[name]["data_offsets"][1] - hdr[name]["data_offsets"][0]
            out_entries.append((name, hdr[name]["dtype"], hdr[name]["shape"], nbytes))

    new_hdr = {}
    if "__metadata__" in hdr:
        new_hdr["__metadata__"] = hdr["__metadata__"]
    cursor = 0
    for name, dtype, shape, data in out_entries:
        nbytes = data if isinstance(data, int) else len(data)
        new_hdr[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + nbytes]}
        cursor += nbytes

    hdr_json = json.dumps(new_hdr, separators=(",", ":")).encode()
    pad = (8 - len(hdr_json) % 8) % 8
    hdr_json += b" " * pad

    out_path = os.path.join(dst, "model.safetensors")
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(hdr_json)))
        out.write(hdr_json)
        for name, dtype, shape, data in out_entries:
            if isinstance(data, int):
                off = hdr[name]["data_offsets"]
                shard.f.seek(shard.data_base + off[0])
                remaining = off[1] - off[0]
                while remaining > 0:
                    buf = shard.f.read(min(COPY_CHUNK, remaining))
                    assert buf, f"short read copying {name}"
                    out.write(buf)
                    remaining -= len(buf)
            else:
                out.write(data)
    new_model_size = os.path.getsize(out_path)
    print(f"wrote {out_path} ({new_model_size} bytes, was {os.path.getsize(os.path.join(src, 'model.safetensors'))})", flush=True)

    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    new_wm = {}
    for name, f in wm.items():
        if f != "model.safetensors":
            new_wm[name] = f
            continue
        if name in dropped:
            continue
        if name in replaced:
            for new_name, _, _, _ in replaced[name]:
                new_wm[new_name] = "model.safetensors"
        else:
            new_wm[name] = "model.safetensors"
    mtp_size = os.path.getsize(os.path.join(src, "model_mtp.safetensors"))
    idx["weight_map"] = new_wm
    idx["metadata"]["total_size"] = new_model_size + mtp_size
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump(idx, f, indent=2)

    cfg = json.load(open(os.path.join(src, "config.json")))
    qc = cfg.get("quantization_config", {})
    layer_alt = "|".join(str(l) for l in fp8_layers)
    tail_mlp_fp8_target = f"re:.*layers\\.({layer_alt})\\.mlp\\.(gate|up|down)_proj$"
    for group in qc.get("config_groups", {}).values():
        targets = group.get("targets")
        if targets and tail_mlp_fp8_target in targets:
            targets.remove(tail_mlp_fp8_target)
            print(f"config.json: removed fp8 target {tail_mlp_fp8_target}", flush=True)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    for entry in sorted(os.listdir(src)):
        if entry in ("model.safetensors", "model.safetensors.index.json", "config.json"):
            continue
        target = os.path.realpath(os.path.join(src, entry))
        link = os.path.join(dst, entry)
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(target, link)
        print(f"linked {entry}", flush=True)

    saved = os.path.getsize(os.path.join(src, "model.safetensors")) - new_model_size
    print(f"DONE saved_bytes={saved} ({saved / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
