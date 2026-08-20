from __future__ import annotations

import torch

INTEGER_DTYPES = frozenset({
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
})
MPS_DEVICE_TYPE = "mps"

_already_patched = False

def install() -> None:
    global _already_patched
    if _already_patched:
        return

    original_histc = torch.histc

    def histc_with_int_promotion(input_tensor, *args, **kwargs):
        on_mps = input_tensor.device.type == MPS_DEVICE_TYPE
        is_integer = input_tensor.dtype in INTEGER_DTYPES
        if on_mps and is_integer:
            return original_histc(input_tensor.float(), *args, **kwargs)
        return original_histc(input_tensor, *args, **kwargs)

    torch.histc = histc_with_int_promotion
    _already_patched = True
