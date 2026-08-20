#!/usr/bin/env python3
"""Enumerate the CUDA/wgpu kernel surface from disk and gate every citation in
docs/book/05.2-kernel-parity-matrix.md.

    kernel-parity-census.py           print the generated region
    kernel-parity-census.py --write   rewrite the generated region in the matrix
    kernel-parity-census.py --check   exit 1 on a stale citation, a stale
                                      generated region, an unclassified
                                      one-sided kernel, or a risen count of
                                      CUDA host fns no test names
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

DOC = "docs/book/05.2-kernel-parity-matrix.md"
BEGIN = "<!-- kernel-parity-census: BEGIN GENERATED -->"
END = "<!-- kernel-parity-census: END GENERATED -->"

REGION_IS_GENERATED_EDIT_THE_SCRIPT_NOT_THE_DOC = (
    "Everything between the census markers is written by "
    "`rust/scripts/kernel-parity-census.py` from the tree. Editing it by hand is "
    "editing a build product: `--check` overwrites the edit's meaning at the next "
    "run and reports the doc as stale. Change the script, then `--write`."
)

A_WGPU_PATH_CANNOT_EXPRESS_THIS_TODAY = "capability gap"
THE_FUNCTION_RUNS_ON_WGPU_UNFUSED = "fusion gap"
THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES = "not a gap"
BOTH_BACKENDS_HAVE_IT_ONLY_ONE_IS_GATED = "missing test"
WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY = "wgpu-native by design"
THE_MODEL_ITSELF_HAS_NO_CUDA_PORT_SO_EVERY_SHADER_OF_IT_IS_ONE_SIDED = "no cuda model"
NO_ENTRY_POINTS_AND_NOTHING_REFERENCES_IT = "dead weight"
THE_FILE_IS_A_FRAGMENT_OTHER_SHADERS_INCLUDE = "fragment"
THE_ENTRY_PROBES_THE_BACKEND_NOT_THE_MODEL = "backend probe"

ONE_SIDED_VERDICTS: dict[str, tuple[str, str]] = {
    "laguna_attn": (
        WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY,
        "CUDA laguna gets attention from FlashAttention-2, not from a kernel in this tree: laguna_fa2.rs is a cfg(cuda) FFI shim declaring FlashFwdParams and laguna_step_graph.rs drives it. wgpu has no FA2 to bind, so the attention is spelled here",
    ),
    "laguna_common": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "four entry points and CUDA reaches all four elsewhere: lgw_silu_mul against silu.cu, lgw_argmax_stage1/stage2 against the argmax in graph_decode.cu and dflash_accept.cu, and lgw_gather_embed against candle's own gather -- no CUDA file in this tree spells an embedding gather because nothing needs to",
    ),
    "laguna_gemv_bf16": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "gemv_bf16.cu carries the bf16 gemv family this splits out",
    ),
    "laguna_gemv_i8": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "gemv_bf16.cu's gemv_i8_normed_kernel is the same int8 gemv",
    ),
    "laguna_gemv_nvfp4": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "gemv_nvfp4.cu's nvfp4_gemv_bf16_kernel is the same nvfp4 gemv under a transposed name -- the census reads it as one-sided only because the words are ordered the other way",
    ),
    "laguna_moe": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "the CUDA MoE path is five files -- moe_route.cu, moe_permute.cu, moe_gemv.cu, moe_grouped_fp4_gemv.cu and moe_unpermute_scatter.cu -- against one shader here",
    ),
    "laguna_quant_rows": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "gemv_nvfp4.cu's nvfp4_quantize_row_bf16_kernel quantises the same rows",
    ),
    "gow_attn": (
        THE_MODEL_ITSELF_HAS_NO_CUDA_PORT_SO_EVERY_SHADER_OF_IT_IS_ONE_SIDED,
        "gpt-oss is served only on wgpu: nv-models has gpt_oss_wgpu.rs and no CUDA sibling, and chat_engine/build.rs's ModelFamily enumerates Qwen3, Gemma4, Gemma4E4b, Gemma4Moe, Qwen3_5Moe and Laguna with no GptOss variant. Nothing is missing from the CUDA side of this kernel because there is no CUDA side of this model",
    ),
    "gow_gemv": (
        THE_MODEL_ITSELF_HAS_NO_CUDA_PORT_SO_EVERY_SHADER_OF_IT_IS_ONE_SIDED,
        "gpt-oss is served only on wgpu: nv-models has gpt_oss_wgpu.rs and no CUDA sibling, and chat_engine/build.rs's ModelFamily enumerates Qwen3, Gemma4, Gemma4E4b, Gemma4Moe, Qwen3_5Moe and Laguna with no GptOss variant. Nothing is missing from the CUDA side of this kernel because there is no CUDA side of this model",
    ),
    "gow_moe": (
        THE_MODEL_ITSELF_HAS_NO_CUDA_PORT_SO_EVERY_SHADER_OF_IT_IS_ONE_SIDED,
        "gpt-oss is served only on wgpu: nv-models has gpt_oss_wgpu.rs and no CUDA sibling, and chat_engine/build.rs's ModelFamily enumerates Qwen3, Gemma4, Gemma4E4b, Gemma4Moe, Qwen3_5Moe and Laguna with no GptOss variant. Nothing is missing from the CUDA side of this kernel because there is no CUDA side of this model",
    ),
    "gow_mx": (
        THE_MODEL_ITSELF_HAS_NO_CUDA_PORT_SO_EVERY_SHADER_OF_IT_IS_ONE_SIDED,
        "gpt-oss is served only on wgpu: nv-models has gpt_oss_wgpu.rs and no CUDA sibling, and chat_engine/build.rs's ModelFamily enumerates Qwen3, Gemma4, Gemma4E4b, Gemma4Moe, Qwen3_5Moe and Laguna with no GptOss variant. Nothing is missing from the CUDA side of this kernel because there is no CUDA side of this model",
    ),
    "gow_prefill": (
        THE_MODEL_ITSELF_HAS_NO_CUDA_PORT_SO_EVERY_SHADER_OF_IT_IS_ONE_SIDED,
        "gpt-oss is served only on wgpu: nv-models has gpt_oss_wgpu.rs and no CUDA sibling, and chat_engine/build.rs's ModelFamily enumerates Qwen3, Gemma4, Gemma4E4b, Gemma4Moe, Qwen3_5Moe and Laguna with no GptOss variant. Nothing is missing from the CUDA side of this kernel because there is no CUDA side of this model",
    ),
    "flash_decode_fold_epilogue": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "CUDA has no __global__ fold at all: flash_decode.cu folds its splits inside the splitk kernel with fan_in and atomics (47 fan_in, 6 atomicAdd, 12 __threadfence). wgpu has no equivalent primitive, so the same reduction becomes its own pass. Same math, decomposed differently because the backend forces it",
    ),
    "flash_decode_fold_head": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "CUDA has no __global__ fold at all: flash_decode.cu folds its splits inside the splitk kernel with fan_in and atomics (47 fan_in, 6 atomicAdd, 12 __threadfence). wgpu has no equivalent primitive, so the same reduction becomes its own pass. Same math, decomposed differently because the backend forces it",
    ),
    "flash_decode_fold_reduce_sg": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "CUDA has no __global__ fold at all: flash_decode.cu folds its splits inside the splitk kernel with fan_in and atomics (47 fan_in, 6 atomicAdd, 12 __threadfence). wgpu has no equivalent primitive, so the same reduction becomes its own pass. Same math, decomposed differently because the backend forces it",
    ),
    "flash_decode_fold_reduce_wb": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "CUDA has no __global__ fold at all: flash_decode.cu folds its splits inside the splitk kernel with fan_in and atomics (47 fan_in, 6 atomicAdd, 12 __threadfence). wgpu has no equivalent primitive, so the same reduction becomes its own pass. Same math, decomposed differently because the backend forces it",
    ),
    "flash_decode_fold_rounds": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "CUDA has no __global__ fold at all: flash_decode.cu folds its splits inside the splitk kernel with fan_in and atomics (47 fan_in, 6 atomicAdd, 12 __threadfence). wgpu has no equivalent primitive, so the same reduction becomes its own pass. Same math, decomposed differently because the backend forces it",
    ),
    "gemv_bf16_sg": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "a subgroup is what CUDA gets for free as a warp -- gemv_bf16.cu's gemv_bf16_kernel already reduces with __shfl_xor_sync, so it needs no separately named subgroup variant",
    ),
    "gemv_bf16_sg_pk": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "a subgroup is what CUDA gets for free as a warp -- gemv_bf16.cu's gemv_bf16_kernel already reduces with __shfl_xor_sync, so it needs no separately named subgroup variant",
    ),
    "quantize_nvfp4_bf16_act_grid": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "quantize_nvfp4_bf16.cu quantizes the same bf16 -> nvfp4 activations through nv_kernels_quantize_nvfp4_bf16; only the grid-stride spelling is wgpu's",
    ),
    "capture_copy": (
        A_WGPU_PATH_CANNOT_EXPRESS_THIS_TODAY,
        "the two ops are trivial; what does not transfer is their reason for existing -- "
        "they feed CUDA graph capture, and wgpu has no graph/replay analog",
    ),
    "dflash_accept": (
        A_WGPU_PATH_CANNOT_EXPRESS_THIS_TODAY,
        "wgpu spec-decode runs the accept chain on the host -- `nv-specdecode/src/wgpu_spec.rs` "
        "calls the CPU `accept_prefix_argmax` -- so it pays a logits readback per step",
    ),
    "dflash_gate": (
        A_WGPU_PATH_CANNOT_EXPRESS_THIS_TODAY,
        "cheapest on the list: one elementwise softplus gate, and WGSL already spells the "
        "function as `gdn_softplus_safe` in `gdn_gating.wgsl`",
    ),
    "gdn_decode": (
        A_WGPU_PATH_CANNOT_EXPRESS_THIS_TODAY,
        "and the interesting half is the other direction: `gdn_gating.wgsl` / `gdn_recurrent.wgsl` "
        "are gated A by `parity_gdn.rs` and no wgpu model wires them -- `nv-layers/src/linear_attn.rs` "
        "dispatches CUDA or falls back to candle on the host",
    ),
    "gqa_verify_hd512": (
        THE_FUNCTION_RUNS_ON_WGPU_UNFUSED,
        "head_dim 512 is reachable on wgpu (`MAX_HEAD_DIM` in `kernels/flash_decode.rs`, "
        "`FD_MAX_HD` in `flash_decode.wgsl`); only the multi-token verify specialisation is CUDA-only",
    ),
    "kv_ring": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "ring KV is folded inside the wgpu kernels rather than standing alone, and "
        "`splitk_bf16kv_ring_wrap_matches_the_linearized_cache` in `wgpu_flash_decode.rs` gates it",
    ),
    "laguna_step": (
        THE_FUNCTION_RUNS_ON_WGPU_UNFUSED,
        "`nv-models/src/laguna_wgpu/` is a real wgpu Laguna decoder composed of generic kernels",
    ),
    "moe_gemv": (
        THE_FUNCTION_RUNS_ON_WGPU_UNFUSED,
        "`moe_grouped_gemm.wgsl` covers grouped MoE matmul; the fused swiglu epilogue is what is missing",
    ),
    "moe_grouped_fp4_gemv": (
        THE_FUNCTION_RUNS_ON_WGPU_UNFUSED,
        "as `moe_gemv`, plus the `gemv_nvfp4*` family for the fp4 half",
    ),
    "moe_route": (
        A_WGPU_PATH_CANNOT_EXPRESS_THIS_TODAY,
        "routing runs on the host: `plan_routing` in `nv-layers/src/moe_wgpu.rs` takes `topk_ids` as a "
        "host slice, and no WGSL entry computes a topk -- `moe_permute.wgsl` consumes `mp_topk_ids`, "
        "it does not produce them",
    ),
    "rmsnorm_quantize_nvfp4_bf16": (
        THE_FUNCTION_RUNS_ON_WGPU_UNFUSED,
        "`rmsnorm.wgsl` and `quantize_nvfp4_bf16.wgsl` both exist and are gated; composing them costs one pass",
    ),
    "cutlass_probe": (
        NO_ENTRY_POINTS_AND_NOTHING_REFERENCES_IT,
        "zero `__global__`, its wgsl was deleted, and its module is a pair of constants no test names",
    ),
    "marlin": (
        A_WGPU_PATH_CANNOT_EXPRESS_THIS_TODAY,
        "undecided: still no recorded will-not-port decision, and the largest unexamined surface in the tree",
    ),
    "cuda_sm120": (
        A_WGPU_PATH_CANNOT_EXPRESS_THIS_TODAY,
        "structural: CUTLASS/sm_120 tensor-core paths, mostly template-instantiated rather than "
        "`__global__`-declared, and WGSL has no equivalent surface -- `coop_mat` is not one",
    ),
    "assistant_drafter": (
        WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY,
        "a whole draft model inlined as shaders; CUDA drafts through the normal model path",
    ),
    "attn_decode_small_m": (WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY, "small-m specialisation CUDA does not need"),
    "attn_decode_small_m_fp8": (WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY, "small-m specialisation CUDA does not need"),
    "attn_decode_small_m_v2": (WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY, "WGSL emitted in Rust, invisible to any `ls wgsl/` census"),
    "dequant": (WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY, "a WGSL fn library with no entry points; CUDA dequant lives in `include/` headers"),
    "fused_attn_chain": (WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY, "the wgpu answer to CUDA graph capture"),
    "fused_norm_chain": (WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY, "the wgpu answer to CUDA graph capture"),
    "gemm_nvfp4": (WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY, "CUDA uses CUTLASS (`cuda_sm120/cutlass_fp4_gemm.cu`)"),
    "gemm_bf16_small_m": (WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY, "WGSL emitted by `writeln!` in the module; CUDA reaches this through cuBLASLt"),
    "gemm_coop_f16": (WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY, "WGSL emitted by `writeln!` in the module; CUDA reaches this through cuBLASLt"),
    "gemm_w4a16_small_m": (WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY, "WGSL emitted by `writeln!` in the module; CUDA reaches this through cuBLASLt"),
    "gemv_nvfp4_lin": (WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY, "layout/routing variant with no CUDA analog"),
    "gemv_nvfp4_v2": (WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY, "layout/routing variant with no CUDA analog"),
    "moe_grouped_gemm": (WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY, "CUDA uses `moe_gemv.cu` plus `cuda_sm120/moe_grouped_fp4_gemm.cu`"),
    "quant_gemv": (WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY, "CUDA reaches these through cuBLASLt"),
    "na": (WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY, "prefill-shaped GEMM, WGSL generated in Rust, module outside `kernels/`"),
    "na_attn": (WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY, "prefill-shaped attention, WGSL generated in Rust, module outside `kernels/`"),
    "na_bf16": (WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY, "prefill-shaped bf16 GEMM, WGSL generated in Rust, module outside `kernels/`"),
    "g4w_fp8_pk_bindings": (
        THE_FILE_IS_A_FRAGMENT_OTHER_SHADERS_INCLUDE,
        "no entry points; the shared fp8 binding block `gemma4_wgpu.rs` textually composes into the g4w fp8 pk shaders",
    ),
    "g4w_gemv4_pk_sg": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`gemv_nvfp4.cu`'s `nvfp4_gemv_bf16_kernel` is the same nvfp4 gemv; _sg is the subgroup-reduction spelling for adapters that have subgroups",
    ),
    "g4w_gemv4_pk_tree": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "the same nvfp4 gemv against `gemv_nvfp4.cu`; _tree is the no-subgroup reduction, an adapter split CUDA does not need because warp shuffles always exist",
    ),
    "g4w_gemv_q8_pk_sg_legacy": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`gemv_bf16.cu`'s `gemv_i8_normed_kernel` family carries the int8 gemv; legacy is the pre-template spelling kept for adapters the template path rejects",
    ),
    "g4w_gemv_q8_pk_sg_prologue": (
        THE_FILE_IS_A_FRAGMENT_OTHER_SHADERS_INCLUDE,
        "no entry points; the shared prologue `gemma4_wgpu.rs` prepends when instantiating the q8 sg template",
    ),
    "g4w_gemv_q8_pk_sg_template": (
        THE_FILE_IS_A_FRAGMENT_OTHER_SHADERS_INCLUDE,
        "its entries are `g4w_gemv_TAG_pk` placeholders -- `gemma4_wgpu.rs` substitutes TAG at pipeline build, so the census sees a kernel that never runs under this name",
    ),
    "g4w_gemv_q8_pk_tree_legacy": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "the same int8 gemv as the sg legacy file against `gemv_bf16.cu`, with the tree reduction for adapters without subgroups",
    ),
    "g4w_gemv_q8_pk_tree_prologue": (
        THE_FILE_IS_A_FRAGMENT_OTHER_SHADERS_INCLUDE,
        "no entry points; the tree-variant prologue `gemma4_wgpu.rs` prepends when instantiating the q8 tree template",
    ),
    "g4w_gemv_q8_pk_tree_template": (
        THE_FILE_IS_A_FRAGMENT_OTHER_SHADERS_INCLUDE,
        "TAG-placeholder entries substituted by `gemma4_wgpu.rs` at pipeline build, same as the sg template",
    ),
    "g4w_glue": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "gather2_bf16 and gather2_bf16_mk are `gather_rows_bf16.cu`'s `gather_rows_bf16_kernel`; pack_lo16 exists because wgpu buffers are u32-word addressed where CUDA reads bf16 pointers directly",
    ),
    "g4w_head_prep": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "computes the per-head kind/index table the pk attention dispatch consumes; CUDA passes head geometry as kernel launch arguments and needs no table",
    ),
    "g4w_mk_params": (
        THE_FILE_IS_A_FRAGMENT_OTHER_SHADERS_INCLUDE,
        "no entry points; the mk params struct block composed into the mk shaders by `gemma4_wgpu.rs` and `gemma4_e4b_wgpu.rs`",
    ),
    "g4w_norm_chain": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`rmsnorm_residual.cu`'s `rmsnorm_residual_kernel_bf16` and `rmsnorm.cu`'s `rmsnorm_kernel` are the same fused norm chains",
    ),
    "g4w_quant_pk": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`gemv_bf16.cu`'s `rowquant_i8_kernel` quantises the same rows",
    ),
    "q3m_attn": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "norm+rope reach `rmsnorm.cu` and `rope.cu` through nv-layers; `attention_fp8_decode.cu` is the decode attention; the output gate (q3w_attn_gate) is candle sigmoid on CUDA in `qwen3_5_moe.rs`",
    ),
    "q3m_delta": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "the delta chain is `depthwise_conv1d_silu_bf16.cu` (conv), `gdn_gating.cu` (gating) and `gdn_decode.cu`/`gdn_recurrent.cu` (recurrent); the u4..u32/l32 entries are wgpu unroll specialisations of one CUDA kernel",
    ),
    "q3m_gemv_bf16": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`gemv_bf16.cu` carries the bf16 gemv family, with the u4/u8 entries as wgpu unroll variants",
    ),
    "q3m_gemv_nvfp4": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`gemv_nvfp4.cu`'s `nvfp4_gemv_bf16_kernel` is the same nvfp4 gemv",
    ),
    "q3m_gemv_nvfp4_v2": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "the same nvfp4 gemv; fmlut/fdec/warp are three wgpu fp4-decode strategies, a search CUDA never needed because it decodes fp4 in hardware paths `gemv_nvfp4.cu` already uses",
    ),
    "q3m_moe": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`moe_route.cu`'s `moe_route_topk_kernel` (topk), `silu.cu`'s `silu_mul_kernel`, `moe_unpermute_scatter.cu` (combine) and `graph_decode.cu`'s `argmax_bf16_stage1_kernel`/`argmax_bf16_stage2_kernel`; gather_embed is candle gather on CUDA",
    ),
    "q3m_q8e": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`moe_gemv.cu` carries the per-expert gemv; `gemv_bf16.cu`'s `gemv_i8_normed_kernel` is the int8 form",
    ),
    "q3m_quant_lane_head_plain": (
        THE_FILE_IS_A_FRAGMENT_OTHER_SHADERS_INCLUDE,
        "no entry points; the plain head `qwen3_5_moe_wgpu.rs` prepends when instantiating the quant lane template",
    ),
    "q3m_quant_lane_head_silu": (
        THE_FILE_IS_A_FRAGMENT_OTHER_SHADERS_INCLUDE,
        "no entry points; the silu-fused head for the same template",
    ),
    "q3m_quant_lane_template": (
        THE_FILE_IS_A_FRAGMENT_OTHER_SHADERS_INCLUDE,
        "its entry is the QL_ENTRY_POINT placeholder -- `qwen3_5_moe_wgpu.rs` substitutes it at pipeline build",
    ),
    "q3m_quant_rows": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`gemv_bf16.cu`'s `rowquant_i8_kernel` quantises the same rows; the silu-fused spelling lives in `quantize_nvfp4_bf16.cu`",
    ),
    "q3m_router_rank_template": (
        THE_FILE_IS_A_FRAGMENT_OTHER_SHADERS_INCLUDE,
        "its entry is the RR_ENTRY_POINT placeholder substituted by `qwen3_5_moe_wgpu.rs`; the routing math it instantiates pairs against `moe_route.cu`",
    ),
    "e4b_axpby": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`residual_scale.cu`'s `residual_add_scale_bf16_kernel` is the same a*x+b*y",
    ),
    "e4b_flash2_pk_mk": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`flash_decode.cu`'s `flash_splitk_fused_fp8_kernel` and `flash_splitk_fused_fp8_mk_kernel` carry the fp8 splitk attention; `gemma4_e4b.rs` drives the fused flash decode entry points on CUDA",
    ),
    "e4b_fncu_prelude": (
        THE_FILE_IS_A_FRAGMENT_OTHER_SHADERS_INCLUDE,
        "no entry points; a prelude `gemma4_e4b_wgpu.rs` prepends to shaders needing the fncu helpers",
    ),
    "e4b_gatemul": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`gelu_tanh_mul.cu`'s `gelu_tanh_mul_bf16_kernel` is the same gelu-gate multiply",
    ),
    "e4b_gather": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`gather_rows_bf16.cu`'s `gather_rows_bf16_kernel` gathers the same rows",
    ),
    "e4b_gemv_pk_mk": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`gemv_bf16.cu`'s `gemm_bf16_mk_kernel` is the same multi-row gemm",
    ),
    "e4b_gemv_w4_pk": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`gemv_w4a16.cu` and the marlin w4a16 kernels carry the same 4-bit gemv/gemm family; block/v4 and pk/pk3 are wgpu packing variants",
    ),
    "e4b_lmhead_i8": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`gemv_bf16.cu`'s `gemv_i8_normed_kernel` over vocab rows is the same int8 lm head",
    ),
    "e4b_lora_cvt": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`lora_fused.cu` and `lora_grouped.cu` consume bf16 factors directly; widen/repack exist because wgpu stores lora factors as packed u32 words",
    ),
    "e4b_smk_tripwire": (
        THE_ENTRY_PROBES_THE_BACKEND_NOT_THE_MODEL,
        "smk_trip_clean/smk_trip_poisoned detect an adapter whose subgroup-matrix path miscompiles, at pipeline-build time in `gemma4_e4b_wgpu.rs`; they compute no model math and CUDA has nothing equivalent to probe",
    ),
    "e4b_verify_cap": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`residual_scale.cu`'s `tanh_softcap_bf16_to_f32_kernel` is the same logit softcap, driven from `gemma4.rs`",
    ),
    "g4m_attn": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "norm+rope reach `rmsnorm.cu` and `rope.cu` through nv-layers; the kv write and decode attention are candle ops in `gemma4_moe.rs`, which composes the CUDA forward from nv-layers and candle rather than in-tree attention kernels",
    ),
    "g4m_gemv_bf16": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "nv-layers `linear.rs` routes this family's projections to `gemv_bf16.cu` on CUDA",
    ),
    "g4m_gemv_bf16_legacy": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "the same bf16 gemv; legacy is the no-subgroup reduction spelling",
    ),
    "g4m_gemv_i8": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`gemv_bf16.cu`'s `gemv_i8_normed_kernel` is the same int8 gemv",
    ),
    "g4m_gemv_i8_v4": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "the same int8 gemv, vec4 spelling",
    ),
    "g4m_gemv_w4e": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "the 4-bit expert gemv: CUDA holds Gemma4Moe experts as nvfp4 and reaches `gemv_nvfp4.cu` through `linear.rs`, wgpu packs them w4a16 -- same function, different 4-bit container",
    ),
    "g4m_moe": (
        WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY,
        "the CUDA Gemma4Moe forward composes routing, gelu, combine and argmax from candle ops -- `gemma4_moe.rs` imports no nv-kernels -- so these shaders exist because wgpu-candle has no usable equivalents, not because CUDA is missing them",
    ),
    "g4m_prefill_head": (
        WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY,
        "CUDA prefill for this family is candle matmuls (cuBLAS); the pm_* fused prefill pipeline exists because wgpu has no cuBLAS to lean on",
    ),
    "g4m_prop_norm": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`rmsnorm.cu` and `rmsnorm_residual.cu` via nv-layers RmsNorm",
    ),
    "q3d_attn": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "dense layers share the qwen3.5 CUDA path: `rmsnorm.cu`/`rope.cu` via nv-layers, `attention_fp8_decode.cu` for decode, candle sigmoid for the output gate",
    ),
    "q3d_delta": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "same chain as the moe file: `depthwise_conv1d_silu_bf16.cu`, `gdn_gating.cu`, `gdn_decode.cu`/`gdn_recurrent.cu`",
    ),
    "q3d_gemv_bf16": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`gemv_bf16.cu`",
    ),
    "q3d_gemv_i8": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`gemv_bf16.cu`'s `gemv_i8_normed_kernel`; i8g is the grouped-scale form of the same gemv",
    ),
    "q3d_misc": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`silu.cu`'s `silu_mul_kernel`, candle gather for the embedding, and `graph_decode.cu`'s argmax stage pair",
    ),
    "q3d_prefill": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "the _m entries are batched spellings of the decode entries above; CUDA prefill reaches the same math through the same kernels plus cuBLAS GEMMs",
    ),
    "g4shared_flash2_pk": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`flash_decode.cu`'s `flash_splitk_stage2_kernel` is the same splitk stage-2 reduction",
    ),
    "g4shared_gemv_pk": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`gemv_bf16.cu`'s `gemv_bf16_kernel`; vec8 is a wgpu vectorisation width",
    ),
    "g4shared_rope_f32": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`rope.cu`'s `rope_kernel` and `rope_bf16.cu` are the same rotation",
    ),
    "g4a_unpack_hidden": (
        THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES,
        "`graph_decode.cu`'s `cast_bf16_f32_kernel` is the same bf16-to-f32 widen; the packed-u32 staging exists only because wgpu buffers are word-addressed, and the CUDA drafter in nv-specdecode reads backbone hidden states directly",
    ),
}

ENTRY_POINT_VERDICTS: dict[str, tuple[str, str]] = {
    "flash_decode.cu::flash_splitk_fused_fp8_derivev_kernel": (
        A_WGPU_PATH_CANNOT_EXPRESS_THIS_TODAY,
        "reconstructs V from the cached K on `attention_k_eq_v` layers; no wgsl entry does it, "
        "so the wgpu path reads a V it does not need to",
    ),
    "flash_decode.cu::flash_splitk_fused_mk_kernel": (
        BOTH_BACKENDS_HAVE_IT_ONLY_ONE_IS_GATED,
        "wgpu has six `_mk` entries in `flash_decode.wgsl` gated by `wgpu_flash_stage1_mk_unroll.rs` "
        "and `wgpu_flash_stage2_unroll.rs`; the CUDA side is the ungated one, so the fix is a "
        "CUDA-vs-CUDA gate and not a port",
    ),
    "flash_decode.cu::flash_splitk_fused_fp8_mk_kernel": (
        BOTH_BACKENDS_HAVE_IT_ONLY_ONE_IS_GATED,
        "same as its bf16 sibling",
    ),
    "laguna_step.cu::softplus_gate_exact_bf16_kernel": (
        A_WGPU_PATH_CANNOT_EXPRESS_THIS_TODAY,
        "one op spelled twice on CUDA, with `dflash_gate.cu::softplus_gate_bf16_kernel`, and neither on wgpu",
    ),
}

DELETED_BY_9c74aedd2_THE_PARITY_CULL = "deleted by `9c74aedd2` (cull 157 -> 87 files); its numbers are quarantined in the historical table"
DELETED_BY_0b11ba5f5_SUITE_CONSOLIDATION = "deleted by `0b11ba5f5`, which merged its assertions into a survivor named in the same row"
DELETED_BY_c7277262d_STRAY_ARTIFACT = "deleted by `c7277262d` as a stray committed artifact"
MOVED_INTO_THE_BOOK_BY_48de116aa = "the pre-`48de116aa` path of a page that now lives under `docs/book/`; named so a reader following an old link learns where it went"
THE_DOC_ASSERTS_THIS_RESOLVES_NOWHERE = "the matrix claims this symbol is absent from the tree; if it comes back the claim is false and this gate says so"
A_SYMBOL_THAT_LIVED_ONLY_IN_A_DELETED_SUITE = "a test fn or constant that lived in a suite the parity cull deleted"
PRINTED_OUTPUT_OF_A_DELETED_SUITE = "a token from the console output of a deleted suite, quoted in the historical table"
A_SUITE_FAMILY_THE_CULL_REMOVED_ENTIRELY = "a whole naming family the cull removed; nothing in the tree ends with it any more"

DEAD_CITATIONS: dict[str, str] = {
    "parity_gemm_nvfp4": DELETED_BY_9c74aedd2_THE_PARITY_CULL,
}

WGPU_BACKEND_MODULES_THAT_ARE_KERNELS = {"na", "na_attn", "na_bf16", "dequant"}
WGPU_BACKEND_MODULES_THAT_ARE_INFRASTRUCTURE = {"buffer", "device", "dispatch", "qualify", "mod", "pack"}

ONE_WGPU_MODULE_COVERING_SEVERAL_CU_FILES = {"lora": ("lora_fused", "lora_grouped")}

CUDA_HOST_FNS_NO_TEST_NAMES_RATCHET = 37

A_VENDORED_FILE_STEM_SHORTER_THAN_THIS_MATCHES_EVERY_TEST = 10

CUDA_ENTRY_POINT = re.compile(r"__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s+)?([A-Za-z0-9_]+)")
CUDA_HOST_FN = re.compile(r'extern\s+"C"\s+int\s+nv_kernels_([A-Za-z0-9_]+)\s*\(')
WGSL_ENTRY_POINT = re.compile(r"@compute[^\n]*\n(?:[^\n]*\n){0,2}?\s*fn\s+([A-Za-z0-9_]+)")

def root() -> Path:
    return Path(__file__).resolve().parents[2]

def kernels_dir() -> Path:
    return root() / "rust/crates/nv-kernels"

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

class Tree:
    def __init__(self) -> None:
        k = kernels_dir()
        self.cuda = {p.stem: p for p in sorted((k / "cuda").glob("*.cu"))}
        self.marlin = sorted((k / "cuda/marlin").rglob("*.cu"))
        self.sm120 = sorted((k / "cuda_sm120").glob("*.cu"))
        self.wgsl = {p.stem: p for p in sorted((k / "wgsl").glob("*.wgsl"))}
        self.kernel_mods = {p.stem: p for p in sorted((k / "src/wgpu_backend/kernels").glob("*.rs")) if p.stem != "mod"}
        self.backend_mods = {p.stem: p for p in sorted((k / "src/wgpu_backend").glob("*.rs"))}
        self.nvk_tests = sorted((k / "tests").glob("*.rs"))
        self.all_tests = sorted((root() / "rust/crates").glob("*/tests/*.rs"))
        self.cuda_eps = {n: CUDA_ENTRY_POINT.findall(read(p)) for n, p in self.cuda.items()}
        self.cuda_hosts = {n: CUDA_HOST_FN.findall(read(p)) for n, p in self.cuda.items()}
        self.wgsl_eps = {n: WGSL_ENTRY_POINT.findall(read(p)) for n, p in self.wgsl.items()}
        self.test_text = {p: read(p) for p in self.all_tests}
        self.src_text = {}
        for p in (root() / "rust/crates").glob("*/src/**/*.rs"):
            if "/nv-kernels/" in str(p):
                continue
            self.src_text[p] = read(p)

    def wgpu_kernel_modules(self) -> set[str]:
        extra = {n for n in self.backend_mods if n in WGPU_BACKEND_MODULES_THAT_ARE_KERNELS}
        return set(self.kernel_mods) | extra

    def wgpu_side_names(self) -> set[str]:
        names = set(self.wgsl) | self.wgpu_kernel_modules()
        for module, covered in ONE_WGPU_MODULE_COVERING_SEVERAL_CU_FILES.items():
            if module in names:
                names.discard(module)
                names.update(covered)
        return names

    def tests_naming(self, needles: list[str], as_prefix: bool = False) -> list[str]:
        tail = "" if as_prefix else r"\b"
        hits = []
        for p, text in self.test_text.items():
            if any(re.search(r"\b" + re.escape(n) + tail, text) for n in needles):
                hits.append(p.parents[1].name + "/tests/" + p.name)
        return sorted(hits)

    def crates_calling(self, needles: list[str]) -> list[str]:
        hits = set()
        for p, text in self.src_text.items():
            if any(re.search(r"\b" + re.escape(n) + r"\b", text) for n in needles):
                hits.add(p.parents[1].name if p.parents[1].name.startswith("nv-") else p.parts[-4])
        return sorted(hits)

def one_sided(t: Tree) -> tuple[list[str], list[str]]:
    wgpu = t.wgpu_side_names()
    cuda_only = [n for n in sorted(t.cuda) if n not in wgpu]
    wgpu_only = [n for n in sorted(wgpu) if n not in t.cuda]
    return cuda_only, wgpu_only

def untested_cuda_host_fns(t: Tree) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name in sorted(t.cuda):
        for h in sorted(set(t.cuda_hosts[name])):
            if not t.tests_naming([h, "nv_kernels_" + h]):
                out.setdefault(name + ".cu", []).append(h)
    return out

def cell(items: list[str], cap: int = 3) -> str:
    if not items:
        return "none"
    shown = ", ".join("`" + i + "`" for i in items[:cap])
    return shown if len(items) <= cap else shown + f", +{len(items) - cap} more"

def verdict_of(name: str, kind: str, failures: list[str]) -> tuple[str, str]:
    if name not in ONE_SIDED_VERDICTS:
        failures.append(
            f"UNCLASSIFIED {kind} kernel `{name}`: it is one-sided on disk and no verdict is recorded "
            f"for it in ONE_SIDED_VERDICTS. A one-sided kernel that nobody has classified is exactly "
            f"the reader-misleading state this census exists to prevent -- decide whether it is a "
            f"{A_WGPU_PATH_CANNOT_EXPRESS_THIS_TODAY}, a {THE_FUNCTION_RUNS_ON_WGPU_UNFUSED}, "
            f"{THE_FUNCTION_IS_PRESENT_UNDER_OTHER_ENTRY_NAMES}, or "
            f"{WGPU_NATIVE_BECAUSE_CUDA_GETS_THIS_SHAPE_FROM_A_LIBRARY}, and record why."
        )
        return ("UNCLASSIFIED", "")
    return ONE_SIDED_VERDICTS[name]

def render(t: Tree, failures: list[str]) -> str:
    cuda_only, wgpu_only = one_sided(t)
    cu_eps = sum(len(v) for v in t.cuda_eps.values())
    wgsl_eps = sum(len(v) for v in t.wgsl_eps.values())
    marlin_eps = sum(len(CUDA_ENTRY_POINT.findall(read(p))) for p in t.marlin)
    sm120_eps = sum(len(CUDA_ENTRY_POINT.findall(read(p))) for p in t.sm120)
    untested = untested_cuda_host_fns(t)
    untested_total = sum(len(v) for v in untested.values())
    host_total = sum(len(set(v)) for v in t.cuda_hosts.values())

    L = [BEGIN, ""]
    L.append("### Inventory, counted from disk")
    L.append("")
    L.append("| surface | count |")
    L.append("|---|---|")
    L.append(f"| top-level `cuda/*.cu` | {len(t.cuda)} ({cu_eps} `__global__`) |")
    L.append(f"| vendored `cuda/marlin/**/*.cu` | {len(t.marlin)} ({marlin_eps} `__global__`) |")
    L.append(f"| `cuda_sm120/*.cu` | {len(t.sm120)} ({sm120_eps} `__global__`; CUTLASS device code is template-instantiated, so this undercounts what runs) |")
    L.append(f"| `wgsl/*.wgsl` | {len(t.wgsl)} ({wgsl_eps} `@compute`) |")
    L.append(f"| modules under `src/wgpu_backend/kernels/` | {len(t.kernel_mods)} |")
    L.append(f"| kernel modules one level up in `src/wgpu_backend/` | {len(t.wgpu_kernel_modules()) - len(t.kernel_mods)} ({', '.join('`' + n + '`' for n in sorted(WGPU_BACKEND_MODULES_THAT_ARE_KERNELS))}) |")
    L.append(f"| nv-kernels test files | {len(t.nvk_tests)} |")
    L.append("")
    L.append("Entry-point totals are `__global__` and `@compute` counts. A kernel written in Rust")
    L.append("with `writeln!` has no `.wgsl` file and is counted as a module, not as a shader.")
    L.append("")
    L.append("### CUDA-only: a `.cu` with no `.wgsl` and no wgpu module")
    L.append("")
    L.append("| kernel | EP | entry points | tests naming it | called from | classification |")
    L.append("|---|---|---|---|---|---|")
    for n in cuda_only:
        eps = t.cuda_eps[n]
        hosts = t.cuda_hosts[n]
        v, why = verdict_of(n, "CUDA-only", failures)
        L.append(
            f"| `{n}.cu` | {len(eps)} | {cell(eps, 7)} | {cell(t.tests_naming(eps + hosts))} | "
            f"{cell(t.crates_calling(hosts), 4)} | **{v}** -- {why} |"
        )
    for n, files, eps in (
        ("marlin", t.marlin, marlin_eps),
        ("cuda_sm120", t.sm120, sm120_eps),
    ):
        v, why = verdict_of(n, "CUDA-only", failures)
        names = cell([p.name for p in files], 4)
        needles = [n] + [p.stem for p in files if len(p.stem) >= A_VENDORED_FILE_STEM_SHORTER_THAN_THIS_MATCHES_EVERY_TEST]
        L.append(f"| `{n}/` | {eps} | {names} | {cell(t.tests_naming(needles, as_prefix=True))} | -- | **{v}** -- {why} |")
    L.append("")
    L.append("### wgpu-only: a `.wgsl` or a wgpu module with no `.cu`")
    L.append("")
    L.append("| module / wgsl | wgsl EP | tests naming it | classification |")
    L.append("|---|---|---|---|")
    for n in wgpu_only:
        eps = t.wgsl_eps.get(n, [])
        v, why = verdict_of(n, "wgpu-only", failures)
        shader = f"{len(eps)}" if n in t.wgsl else "generated in Rust"
        L.append(f"| `{n}` | {shader} | {cell(t.tests_naming([n] + eps), 3)} | **{v}** -- {why} |")
    L.append("")
    L.append("### CUDA host entry fns no test in `rust/crates/*/tests/` names")
    L.append("")
    L.append("The `extern \"C\" int nv_kernels_*` fns are the CUDA API surface: one per callable path,")
    L.append("and what a test actually invokes. A file-level census reports the kernels below as")
    L.append("covered, because a sibling fn in the same `.cu` is tested. These are not.")
    L.append("")
    L.append("| file | host fns no test names |")
    L.append("|---|---|")
    for f in sorted(untested):
        L.append(f"| `{f}` | {', '.join('`nv_kernels_' + e + '`' for e in untested[f])} |")
    L.append("")
    L.append(
        f"{untested_total} of {host_total} host fns; the ratchet in the script is "
        f"{CUDA_HOST_FNS_NO_TEST_NAMES_RATCHET}. This is a *missing test* count, not a capability "
        f"gap count -- both backends may well have the kernel."
    )
    L.append("")
    L.append("Entry points with a recorded verdict, because a reader sent at them would otherwise guess:")
    L.append("")
    L.append("| entry point | classification |")
    L.append("|---|---|")
    for key in sorted(ENTRY_POINT_VERDICTS):
        v, why = ENTRY_POINT_VERDICTS[key]
        L.append(f"| `{key}` | **{v}** -- {why} |")
    L.append("")
    L.append("### Citations this file makes that must NOT resolve")
    L.append("")
    L.append("Every other `.rs` / `.cu` / `.wgsl` / `.md` name and every backticked identifier in this")
    L.append("document must resolve in the tree, or `--check` fails. These are the recorded exceptions:")
    L.append("a name here that comes back to life fails the check too, because the sentence around it")
    L.append("would then be wrong in the other direction.")
    L.append("")
    L.append("| name | why it is named here |")
    L.append("|---|---|")
    for n in sorted(DEAD_CITATIONS):
        L.append(f"| `{n}` | {DEAD_CITATIONS[n]} |")
    L.append("")
    L.append(END)
    return "\n".join(L)

def strip_fences(doc: str) -> str:
    out, fence = [], False
    for line in doc.splitlines():
        if line.startswith("```"):
            fence = not fence
            continue
        if not fence:
            out.append(line)
    return "\n".join(out)

def tree_identifiers() -> set[str]:
    out = subprocess.run(
        [
            "git",
            "grep",
            "-h",
            "--untracked",
            "-oE",
            "[A-Za-z_][A-Za-z0-9_]*",
            "--",
            "rust/",
            ":!rust/scripts/kernel-parity-census.py",
        ],
        cwd=root(),
        capture_output=True,
        text=True,
    ).stdout
    words = set(out.split())
    own = Path(__file__).read_text(encoding="utf-8")
    words |= set(re.findall(r"^([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]*)?=", own, re.M))
    words |= set(re.findall(r"^def ([A-Za-z_][A-Za-z0-9_]*)", own, re.M))
    for dirpath, dirnames, filenames in os.walk(root() / "rust/crates"):
        if "target" in Path(dirpath).parts:
            dirnames[:] = [d for d in dirnames if d != "target"]
            continue
        for f in filenames:
            words.add(Path(f).stem)
    return words

def tree_paths() -> list[str]:
    paths = []
    for base in ("rust", "docs", "conformance", "scripts", "python", "go"):
        for dirpath, dirnames, filenames in os.walk(root() / base):
            if "target" in Path(dirpath).parts or ".git" in Path(dirpath).parts:
                dirnames[:] = [d for d in dirnames if d not in ("target", ".git")]
                continue
            for f in filenames:
                paths.append(str(Path(dirpath).relative_to(root()) / f))
    return paths

def check_citations(doc: str, failures: list[str]) -> None:
    body = strip_fences(doc)
    hand_written = strip_fences(doc.split(BEGIN)[0] + doc.split(END)[-1]) if BEGIN in doc and END in doc else body
    words = tree_identifiers()
    paths = tree_paths()
    suffixes = set()
    for p in paths:
        parts = p.split("/")
        for i in range(len(parts)):
            suffixes.add("/".join(parts[i:]))

    cited_files = set(re.findall(r"[A-Za-z0-9_./-]*[A-Za-z0-9_-]+\.(?:rs|cu|wgsl|md)\b", body))
    cited_idents = set()
    for line in body.splitlines():
        for span in re.findall(r"`([^`\n]+)`", line):
            span = re.sub(r"[A-Za-z0-9_]+\.(?:rs|cu|wgsl|md|h|py|sh)\b", "", span)
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", span):
                if "_" in tok and len(tok) >= 6:
                    cited_idents.add(tok)

    for f in sorted(cited_files):
        if f in DEAD_CITATIONS:
            continue
        if f in suffixes:
            continue
        failures.append(
            f"STALE CITATION `{f}`: this document names it and no file in the tree has that path "
            f"suffix. Repoint the row at the file that carries the assertions now, or -- if the file "
            f"is deliberately gone -- add it to DEAD_CITATIONS with the commit that removed it."
        )

    for tok in sorted(cited_idents):
        if tok in DEAD_CITATIONS or tok in words:
            continue
        if tok.startswith("_") and any(w.endswith(tok) for w in words):
            continue
        if any(w.startswith(tok) for w in words):
            continue
        failures.append(
            f"STALE SYMBOL `{tok}`: this document cites it and it appears nowhere under `rust/`. "
            f"Symbols are how this file cites code, because line numbers rot in hours here -- a "
            f"symbol that stopped resolving makes the sentence around it a claim nobody can check. "
            f"Repoint it, or add it to DEAD_CITATIONS if the document's point is that it is gone."
        )

    for name in sorted(DEAD_CITATIONS):
        resolves = name in suffixes or name in words
        if resolves:
            failures.append(
                f"RESURRECTED `{name}`: DEAD_CITATIONS says this document names it as absent, and it "
                f"is present in the tree. The sentence citing it now reads backwards. Drop it from "
                f"DEAD_CITATIONS and rewrite the row that mentions it."
            )
        if not re.search(re.escape(name), hand_written):
            failures.append(
                f"UNCITED DEAD ENTRY `{name}`: it is listed in DEAD_CITATIONS and the hand-written "
                f"part of this document no longer mentions it, so the exception protects nothing and "
                f"would silently permit a future stale citation of the same name. The generated "
                f"region does not count -- it prints the registry, so it would vouch for every entry. "
                f"Remove it from DEAD_CITATIONS."
            )

def check_module_classification(t: Tree, failures: list[str]) -> None:
    known = WGPU_BACKEND_MODULES_THAT_ARE_KERNELS | WGPU_BACKEND_MODULES_THAT_ARE_INFRASTRUCTURE
    for n in sorted(t.backend_mods):
        if n not in known:
            failures.append(
                f"UNSORTED MODULE `src/wgpu_backend/{n}.rs`: it is neither in "
                f"WGPU_BACKEND_MODULES_THAT_ARE_KERNELS nor in "
                f"WGPU_BACKEND_MODULES_THAT_ARE_INFRASTRUCTURE. The wgpu-only enumeration counts the "
                f"first set and ignores the second, so an unsorted module is a kernel this census "
                f"either invents or hides."
            )

def check_verdict_targets(t: Tree, failures: list[str]) -> None:
    cuda_only, wgpu_only = one_sided(t)
    live = set(cuda_only) | set(wgpu_only) | {"marlin", "cuda_sm120", "cutlass_probe"}
    for n in sorted(ONE_SIDED_VERDICTS):
        if n not in live:
            failures.append(
                f"STALE VERDICT `{n}`: ONE_SIDED_VERDICTS classifies it as one-sided and it is not "
                f"one-sided on disk any more. If it was ported, that is a gap closing and the matrix "
                f"has to say so; drop the entry and move the row into the two-sided table."
            )
    for key in sorted(ENTRY_POINT_VERDICTS):
        f, ep = key.split("::")
        name = f[:-3]
        if name not in t.cuda_eps or ep not in t.cuda_eps[name]:
            failures.append(
                f"STALE ENTRY-POINT VERDICT `{key}`: no `__global__` by that name exists in that file. "
                f"An entry point renamed out from under a verdict leaves the verdict describing "
                f"nothing, which reads as coverage."
            )

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    t = Tree()
    failures: list[str] = []
    generated = render(t, failures)
    doc_path = root() / DOC
    doc = read(doc_path)

    if args.write:
        if BEGIN not in doc or END not in doc:
            print(f"{DOC} has no census markers; add {BEGIN} / {END} around the region", file=sys.stderr)
            return 1
        head = doc.split(BEGIN)[0]
        tail = doc.split(END, 1)[1]
        doc_path.write_text(head + generated + tail, encoding="utf-8")
        print(f"wrote the generated region of {DOC}")
        return 0

    if not args.check:
        print(generated)
        return 0

    check_module_classification(t, failures)
    check_verdict_targets(t, failures)
    check_citations(doc, failures)

    untested = sum(len(v) for v in untested_cuda_host_fns(t).values())
    if untested > CUDA_HOST_FNS_NO_TEST_NAMES_RATCHET:
        failures.append(
            f"UNTESTED HOST FNS ROSE: {untested} CUDA host fns are named by no test, ratchet is "
            f"{CUDA_HOST_FNS_NO_TEST_NAMES_RATCHET}. A new `extern \"C\"` entry that no test names is "
            f"the cheap moment to gate it; raising the ratchet instead is a decision to ship a "
            f"callable path uncovered and belongs in the same commit as the reason."
        )
    elif untested < CUDA_HOST_FNS_NO_TEST_NAMES_RATCHET:
        failures.append(
            f"UNTESTED HOST FNS FELL: {untested} vs ratchet {CUDA_HOST_FNS_NO_TEST_NAMES_RATCHET}. "
            f"Lower CUDA_HOST_FNS_NO_TEST_NAMES_RATCHET to {untested} in the commit that earned it, "
            f"so the next regression is caught against the tighter floor."
        )

    if BEGIN not in doc or END not in doc:
        failures.append(
            f"NO CENSUS MARKERS in {DOC}: {BEGIN} / {END} are gone, so the enumerated section is "
            f"hand-written again and nothing regenerates it."
        )
    else:
        current = BEGIN + doc.split(BEGIN, 1)[1].split(END, 1)[0] + END
        if current.strip() != generated.strip():
            failures.append(
                f"GENERATED REGION IS STALE in {DOC}: the tree and the document disagree. Run "
                f"`rust/scripts/kernel-parity-census.py --write` and read the diff -- it is the list "
                f"of kernels that moved without the matrix noticing. "
                f"{REGION_IS_GENERATED_EDIT_THE_SCRIPT_NOT_THE_DOC}"
            )

    if failures:
        print(f"kernel parity census FAILED: {len(failures)} problem(s)", file=sys.stderr)
        for f in failures:
            print("  - " + f, file=sys.stderr)
        return 1
    print(f"kernel parity census OK: {DOC} agrees with the tree and every citation resolves")
    return 0

if __name__ == "__main__":
    sys.exit(main())
