#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../rmsnorm_add_scale_row.cuh"

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

namespace MARLIN_NAMESPACE_NAME {

enum MarlinM1TailKind {
  kMarlinM1TailNone = 0,
  kMarlinM1TailRmsnormAddScaleThenOptionalNextNormChain = 1,
  kMarlinM1TailGeluTimesPerLayerInput = 2,
};

struct MarlinM1Tail {
  int kind;
  const __nv_bfloat16* res;
  const __nv_bfloat16* w;
  __nv_bfloat16* h_out;
  float* rstd_out;
  const __nv_bfloat16* next_w;
  __nv_bfloat16* normed_out;
  const float* pli;
  __nv_bfloat16* gelu_out;
  float eps;
  float scale;
  float eps_next;
  unsigned* counter;
};

constexpr int kMarlinM1TailTicketFloatsBeforeTheReduceScratch = 8;

template <int threads>
__device__ __forceinline__ void marlin_m1_tail_in_the_last_block_to_finish(
    const MarlinM1Tail& t, const __nv_bfloat16* c, int n, float* dyn_smem) {
  float* s_red = dyn_smem + kMarlinM1TailTicketFloatsBeforeTheReduceScratch;
  float* s_of = s_red + nvk_norm_row::kRmsnormRowReduceScratchFloats;
  if (t.kind == kMarlinM1TailRmsnormAddScaleThenOptionalNextNormChain) {
    nvk_norm_row::rmsnorm_add_scale_row<128, threads>(
        c, t.w, t.res, t.h_out, t.rstd_out, t.next_w, t.normed_out, s_red, s_of, n, t.eps,
        t.scale, t.eps_next);
  } else if (t.kind == kMarlinM1TailGeluTimesPerLayerInput) {
    for (int i = threadIdx.x; i < n; i += threads) {
      float g = __bfloat162float(__ldcg(c + i));
      const float k = 0.7978845608028654f;
      float th = tanhf(k * (g + 0.044715f * g * g * g));
      float gelu = 0.5f * g * (1.0f + th);
      t.gelu_out[i] = __float2bfloat16(gelu * t.pli[i]);
    }
  }
}

}
