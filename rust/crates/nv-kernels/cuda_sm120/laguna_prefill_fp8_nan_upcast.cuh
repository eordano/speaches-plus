#pragma once
#include <flashinfer/vec_dtypes.cuh>
#include <cstdint>

namespace flashinfer {

template <>
FLASHINFER_INLINE void vec_cast<nv_bfloat16, __nv_fp8_e4m3>::cast<8>(
    nv_bfloat16* dst, const __nv_fp8_e4m3* src) {
    uint32_t packed[2] = {reinterpret_cast<const uint32_t*>(src)[0],
                          reinterpret_cast<const uint32_t*>(src)[1]};
    fast_dequant_f8f16x4<__nv_fp8_e4m3, nv_bfloat16>(&packed[0],
        reinterpret_cast<uint2*>(dst));
    fast_dequant_f8f16x4<__nv_fp8_e4m3, nv_bfloat16>(&packed[1],
        reinterpret_cast<uint2*>(dst + 4));
    const uint32_t nan0 = ((packed[0] & 0x7f7f7f7fu) + 0x01010101u) & 0x80808080u;
    const uint32_t nan1 = ((packed[1] & 0x7f7f7f7fu) + 0x01010101u) & 0x80808080u;
    if ((nan0 | nan1) != 0) {
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t code = (packed[i / 4] >> ((i % 4) * 8)) & 0xffu;
            if ((code & 0x7fu) == 0x7fu)
                reinterpret_cast<uint16_t*>(dst)[i] = uint16_t(0x7fc0u | ((code & 0x80u) << 8));
        }
    }
}

}
