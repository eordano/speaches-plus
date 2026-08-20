from __future__ import annotations

from typing import Any

import numpy
import torch

SCALE_PERM_OUTER = 8
SCALE_PERM_INNER = 8
SCALE_PERM_SINGLE_OUTER = 4
SCALE_PERM_SINGLE_INNER_PAIRS = (0, 1, 8, 9, 16, 17, 24, 25)
SCALE_PERM_SINGLE_STEP = 2
GROUP_SIZE_CHANNELWISE = -1

NUM_BITS_4 = 4
NUM_BITS_8 = 8
INT32_BITS = 32

INTERLEAVE_4BIT = (0, 2, 4, 6, 1, 3, 5, 7)
INTERLEAVE_8BIT = (0, 2, 1, 3)

WORKSPACE_DTYPE = torch.int32
WORKSPACE_FALLBACK_SMS = 132
WORKSPACE_DEFAULT_BLOCKS_PER_SM = 4

MARLIN_GEMM_USE_ATOMIC_ADD = False
MARLIN_GEMM_USE_FP32_REDUCE = True
MARLIN_GEMM_IS_ZP_FLOAT = False
MARLIN_GEMM_IS_K_FULL = True

WEIGHT_PARAM = "weight_packed"
SCALE_PARAM = "weight_scale"
ZERO_POINT_PARAM = "weight_zero_point"
WEIGHT_SHAPE_PARAM = "weight_shape"
WORKSPACE_ATTR = "marlin_workspace"
G_IDX_ATTR = "marlin_g_idx"
G_IDX_SORT_INDICES_ATTR = "marlin_g_idx_sort_indices"
WTYPE_ATTR = "marlin_wtype"

try:
    import vllm._custom_ops as _custom_ops
    _HAS_MARLIN_KERNELS = True
except ImportError:
    _custom_ops = None
    _HAS_MARLIN_KERNELS = False

try:
    from vllm.scalar_type import scalar_types as _scalar_types
    _HAS_SCALAR_TYPES = True
    UINT4 = _scalar_types.uint4
    UINT4B8 = _scalar_types.uint4b8
except ImportError:
    _scalar_types = None
    _HAS_SCALAR_TYPES = False
    UINT4 = None
    UINT4B8 = None

def _require_kernels() -> None:
    if not _HAS_MARLIN_KERNELS or not _HAS_SCALAR_TYPES:
        raise RuntimeError(
            "Marlin kernels not available. Install the vllm package on a CUDA host "
            "or use a non-quantized checkpoint."
        )

def get_pack_factor(num_bits: int) -> int:
    assert INT32_BITS % num_bits == 0
    return INT32_BITS // num_bits

def _scale_perms() -> tuple[list[int], list[int]]:
    perm_grouped: list[int] = []
    for outer_index in range(SCALE_PERM_OUTER):
        perm_grouped.extend(
            [outer_index + SCALE_PERM_INNER * inner_index for inner_index in range(SCALE_PERM_INNER)]
        )
    perm_single: list[int] = []
    for single_outer_index in range(SCALE_PERM_SINGLE_OUTER):
        perm_single.extend(
            [SCALE_PERM_SINGLE_STEP * single_outer_index + offset for offset in SCALE_PERM_SINGLE_INNER_PAIRS]
        )
    return perm_grouped, perm_single

def marlin_make_workspace_new(
    device: torch.device,
    max_blocks_per_sm: int = WORKSPACE_DEFAULT_BLOCKS_PER_SM,
) -> torch.Tensor:

    if device.type != "cuda":
        sms = WORKSPACE_FALLBACK_SMS
    else:
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        sms = getattr(properties, "multi_processor_count", WORKSPACE_FALLBACK_SMS)
    return torch.zeros(
        sms * max_blocks_per_sm,
        dtype=WORKSPACE_DTYPE,
        device=device,
        requires_grad=False,
    )

def marlin_permute_scales(
    scales: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
) -> torch.Tensor:

    perm_grouped, perm_single = _scale_perms()
    if group_size < size_k and group_size != GROUP_SIZE_CHANNELWISE:
        permuted = scales.reshape((-1, len(perm_grouped)))[:, perm_grouped]
    else:
        permuted = scales.reshape((-1, len(perm_single)))[:, perm_single]
    return permuted.reshape((-1, size_n)).contiguous()

def _pack_cols(values: torch.Tensor, num_bits: int, size_k: int, size_n: int) -> torch.Tensor:
    pack_factor = get_pack_factor(num_bits)
    assert values.shape == (size_k, size_n)
    assert size_n % pack_factor == 0
    original_device = values.device
    as_uint = values.cpu().numpy().astype(numpy.uint32)
    packed = numpy.zeros((size_k, size_n // pack_factor), dtype=numpy.uint32)
    for slot_index in range(pack_factor):
        packed |= as_uint[:, slot_index::pack_factor] << (num_bits * slot_index)
    packed_tensor = torch.from_numpy(packed.astype(numpy.int32)).to(original_device)
    return packed_tensor.contiguous()

def _unpack_cols(packed: torch.Tensor, num_bits: int, size_k: int, size_n: int) -> torch.Tensor:
    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0
    assert packed.shape == (size_k, size_n // pack_factor)
    original_device = packed.device
    as_uint = packed.cpu().numpy().astype(numpy.uint32).copy()
    unpacked = numpy.zeros((size_k, size_n), dtype=numpy.uint32)
    mask = (1 << num_bits) - 1
    for slot_index in range(pack_factor):
        unpacked[:, slot_index::pack_factor] = as_uint & mask
        as_uint >>= num_bits
    return torch.from_numpy(unpacked.astype(numpy.int32)).to(original_device).contiguous()

def _marlin_zero_points(
    zero_points: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    perm_grouped, _ = _scale_perms()
    permuted = zero_points.reshape((-1, len(perm_grouped)))[:, perm_grouped]
    if num_bits == NUM_BITS_4:
        interleave = numpy.array(INTERLEAVE_4BIT)
    elif num_bits == NUM_BITS_8:
        interleave = numpy.array(INTERLEAVE_8BIT)
    else:
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")
    permuted = permuted.reshape((-1, len(interleave)))[:, interleave].ravel()
    permuted = permuted.reshape((-1, size_n)).contiguous()
    return _pack_cols(permuted, num_bits, size_k, size_n)

def awq_to_marlin_zero_points(
    packed_zero_points: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:

    unpacked = _unpack_cols(packed_zero_points, num_bits, size_k, size_n)
    if num_bits == NUM_BITS_4:
        undo_interleave = numpy.argsort(numpy.array(INTERLEAVE_4BIT))
    elif num_bits == NUM_BITS_8:
        undo_interleave = numpy.argsort(numpy.array(INTERLEAVE_8BIT))
    else:
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")
    undone = unpacked.reshape((-1, len(undo_interleave)))[:, undo_interleave].ravel()
    undone = undone.reshape((-1, size_n)).contiguous()
    return _marlin_zero_points(undone, size_k, size_n, num_bits)

def _empty_int_buffer(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.int32, device=device)

def prepare_gptq_layer_for_marlin(
    layer: torch.nn.Module,
    *,
    num_bits: int,
    group_size: int,
    has_zero_point: bool,
    input_size_per_partition: int,
    output_size_per_partition: int,
) -> None:

    _require_kernels()
    weight_packed = getattr(layer, WEIGHT_PARAM).data.contiguous()
    weight_scale = getattr(layer, SCALE_PARAM).data.contiguous()
    device = weight_packed.device

    repacked_weight = _custom_ops.awq_marlin_repack(
        weight_packed,
        size_k=input_size_per_partition,
        size_n=output_size_per_partition,
        num_bits=num_bits,
    )

    permuted_scales = marlin_permute_scales(
        weight_scale,
        size_k=input_size_per_partition,
        size_n=output_size_per_partition,
        group_size=group_size,
    )

    if has_zero_point:
        zero_points = getattr(layer, ZERO_POINT_PARAM).data.contiguous()
        grouped_k = (
            input_size_per_partition // group_size
            if group_size != GROUP_SIZE_CHANNELWISE
            else 1
        )
        repacked_zero_points = awq_to_marlin_zero_points(
            zero_points,
            size_k=grouped_k,
            size_n=output_size_per_partition,
            num_bits=num_bits,
        )
        wtype = UINT4
    else:
        repacked_zero_points = _empty_int_buffer(device)
        wtype = UINT4B8

    delattr(layer, WEIGHT_PARAM)
    delattr(layer, SCALE_PARAM)
    if hasattr(layer, ZERO_POINT_PARAM):
        delattr(layer, ZERO_POINT_PARAM)

    layer.register_parameter(
        WEIGHT_PARAM,
        torch.nn.Parameter(repacked_weight, requires_grad=False),
    )
    layer.register_parameter(
        SCALE_PARAM,
        torch.nn.Parameter(permuted_scales, requires_grad=False),
    )
    layer.register_buffer(ZERO_POINT_PARAM, repacked_zero_points, persistent=False)
    layer.register_buffer(WORKSPACE_ATTR, marlin_make_workspace_new(device), persistent=False)
    layer.register_buffer(G_IDX_ATTR, _empty_int_buffer(device), persistent=False)
    layer.register_buffer(G_IDX_SORT_INDICES_ATTR, _empty_int_buffer(device), persistent=False)
    setattr(layer, WTYPE_ATTR, wtype)

def apply_gptq_marlin_linear(
    *,
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor,
    workspace: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    wtype: Any,
    output_size_per_partition: int,
    input_size_per_partition: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:

    _require_kernels()
    reshaped = input_tensor.reshape(-1, input_tensor.shape[-1])
    output_shape = input_tensor.shape[:-1] + (output_size_per_partition,)
    output = _custom_ops.marlin_gemm(
        reshaped,
        None,
        weight,
        bias,
        weight_scale,
        None,
        None,
        weight_zero_point,
        g_idx,
        g_idx_sort_indices,
        workspace,
        wtype,
        size_m=reshaped.shape[0],
        size_n=output_size_per_partition,
        size_k=input_size_per_partition,
        is_k_full=MARLIN_GEMM_IS_K_FULL,
        use_atomic_add=MARLIN_GEMM_USE_ATOMIC_ADD,
        use_fp32_reduce=MARLIN_GEMM_USE_FP32_REDUCE,
        is_zp_float=MARLIN_GEMM_IS_ZP_FLOAT,
    )
    return output.reshape(output_shape)

__all__ = [
    "G_IDX_ATTR",
    "G_IDX_SORT_INDICES_ATTR",
    "SCALE_PARAM",
    "UINT4",
    "UINT4B8",
    "WEIGHT_PARAM",
    "WEIGHT_SHAPE_PARAM",
    "WORKSPACE_ATTR",
    "WTYPE_ATTR",
    "ZERO_POINT_PARAM",
    "apply_gptq_marlin_linear",
    "awq_to_marlin_zero_points",
    "get_pack_factor",
    "marlin_make_workspace_new",
    "marlin_permute_scales",
    "prepare_gptq_layer_for_marlin",
]
