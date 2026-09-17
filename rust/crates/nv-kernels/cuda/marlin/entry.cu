
#include <cuda_runtime.h>

#include "marlin.cuh"
#include "scalar_type.h"
#include "kernel.h"

#ifndef TORCH_CHECK
#define TORCH_CHECK(cond, ...) do { if (!(cond)) {} } while (0)
#endif

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

namespace marlin {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};
using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

thread_config_t small_batch_thread_configs[] = {

    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};

thread_config_t large_batch_thread_configs[] = {

    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

int get_scales_cache_size(thread_config_t const& th_config, int prob_m,
                          int prob_n, int prob_k, int num_bits, int group_size,
                          bool has_act_order, bool is_k_full, int stages) {
  bool cache_scales_chunk = has_act_order && !is_k_full;

  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else if (group_size == 0) {
    tb_groups = div_ceil(tb_k, 32);
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  if (cache_scales_chunk) {
    int load_groups =
        tb_groups * stages * 2;
    load_groups = max(load_groups, 32);
    return load_groups * tb_n * 2;
  } else {
    int tb_scales = tb_groups * tb_n * 2;

    return tb_scales * stages;
  }
}

int get_kernel_cache_size(thread_config_t const& th_config, int thread_m_blocks,
                          int prob_m, int prob_n, int prob_k, int num_bits,
                          int group_size, bool has_act_order, bool is_k_full,
                          int has_zp, bool is_zp_float, bool is_a_8bit,
                          int stages) {
  int pack_factor = 32 / num_bits;

  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;
  int sh_a_size = stages * (tb_m * tb_k) * (is_a_8bit ? 1 : 2);
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size =
      (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size =
      get_scales_cache_size(th_config, prob_m, prob_n, prob_k, num_bits,
                            group_size, has_act_order, is_k_full, stages);
  int sh_g_idx_size = has_act_order && !is_k_full ? stages * tb_k / 4 : 0;
  int sh_zp_size = 0;
  if (has_zp) {
    if (is_zp_float)
      sh_zp_size = sh_s_size;
    else if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  int total_size =
      tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_g_idx_size;

  return total_size;
}

bool is_valid_config(thread_config_t const& th_config, int thread_m_blocks,
                     int prob_m, int prob_n, int prob_k, int num_bits,
                     int group_size, bool has_act_order, bool is_k_full,
                     int has_zp, bool is_zp_float, bool is_a_8bit, int stages,
                     int max_shared_mem) {
  if (th_config.thread_k == -1 || th_config.thread_n == -1 ||
      th_config.num_threads == -1) {
    return false;
  }

  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }

  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }

  if (th_config.num_threads < 128) {
    return false;
  }

  int cache_size = get_kernel_cache_size(
      th_config, thread_m_blocks, prob_m, prob_n, prob_k, num_bits, group_size,
      has_act_order, is_k_full, has_zp, is_zp_float, is_a_8bit, stages);
  return cache_size <= max_shared_mem;
}

MarlinFuncPtr get_marlin_kernel(
    const vllm::ScalarType a_type, const vllm::ScalarType b_type,
    const vllm::ScalarType c_type, const vllm::ScalarType s_type,
    int thread_m_blocks, int thread_n_blocks, int thread_k_blocks,
    bool m_block_size_8, bool has_act_order, bool has_zp, int group_blocks,
    int threads, bool is_zp_float, int stages) {
  int num_bits = b_type.size_bits();
  auto kernel = MarlinDefault;

  #include "kernel_selector.h"

  return kernel;
}

exec_config_t determine_exec_config(
    const vllm::ScalarType& a_type, const vllm::ScalarType& b_type,
    const vllm::ScalarType& c_type, const vllm::ScalarType& s_type, int prob_m,
    int prob_n, int prob_k, int thread_m_blocks, bool m_block_size_8,
    int num_bits, int group_size, bool has_act_order, bool is_k_full,
    bool has_zp, bool is_zp_float, int is_a_8bit, int stages,
    int max_shared_mem, int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1
                                        ? large_batch_thread_configs
                                        : small_batch_thread_configs;
  int thread_configs_size =
      thread_m_blocks > 1
          ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
          : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(th_config, thread_m_blocks, prob_m, prob_n, prob_k,
                         num_bits, group_size, has_act_order, is_k_full, has_zp,
                         is_zp_float, is_a_8bit, stages,
                         max_shared_mem - 512)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(th_config, thread_m_blocks, prob_m,
                                           prob_n, prob_k, num_bits, group_size,
                                           has_act_order, is_k_full, has_zp,
                                           is_zp_float, is_a_8bit, stages);

    int group_blocks = 0;
    if (!has_act_order) {
      group_blocks = group_size == -1 ? -1 : group_size / 16;
    }

    auto kernel =
        get_marlin_kernel(a_type, b_type, c_type, s_type, thread_m_blocks,
                          th_config.thread_n / 16, th_config.thread_k / 16,
                          m_block_size_8, has_act_order, has_zp, group_blocks,
                          th_config.num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) continue;

    return {1, th_config};
  }

  return exec_cfg;
}

static int marlin_mm_raw(
    const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
    void* a_s, void* b_s, void* g_s, void* zp, void* g_idx, void* perm,
    void* a_tmp, int prob_m, int prob_n, int prob_k, int lda, void* workspace,
    vllm::ScalarType const& a_type, vllm::ScalarType const& b_type,
    vllm::ScalarType const& c_type, vllm::ScalarType const& s_type,
    bool has_bias, bool has_act_order, bool is_k_full, bool has_zp,
    int num_groups, int group_size, int dev, cudaStream_t stream,
    int thread_k_init, int thread_n_init, int sms, bool use_atomic_add,
    bool use_fp32_reduce, bool is_zp_float,
    const MarlinM1Tail* m1_tail) {
  bool is_a_8bit = a_type.size_bits() == 8;
  MarlinM1Tail tail_by_value = {};
  if (m1_tail != nullptr) {
    if (prob_m != 1) return -1;
    tail_by_value = *m1_tail;
  }
  if (!(prob_m > 0 && prob_n > 0 && prob_k > 0)) return -1;

  int group_blocks = 0;
  if (has_act_order) {
    if (is_k_full) {
      if (!(group_size != -1)) return -1;
      group_blocks = group_size / 16;
      if (!(prob_k % group_blocks == 0)) return -1;
    } else {
      if (!(group_size == 0)) return -1;
      group_blocks = 0;
    }
  } else {
    if (group_size == -1) {
      group_blocks = -1;
    } else {
      group_blocks = group_size / 16;
      if (!(prob_k % group_blocks == 0)) return -1;
    }
  }

  int num_bits = b_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;

  const int4* bias_ptr = (const int4*)b_bias;
  const float* a_s_ptr = (const float*)a_s;
  const int4* b_s_ptr = (const int4*)b_s;
  const float* g_s_ptr = (const float*)g_s;

  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = (const int*)g_idx;
  const int* perm_ptr = (const int*)perm;
  int4* a_tmp_ptr = (int4*)a_tmp;
  int* locks = (int*)workspace;

  if (has_act_order) return -1;

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (!(max_shared_mem > 0)) return -1;

  int major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         dev);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         dev);
  if (!(major_capability * 10 + minor_capability >= 75)) return -1;
  int stages = 4;
  if (major_capability == 7 && minor_capability == 5) {
    stages = 2;
    if (!(a_type == vllm::kFloat16 || a_type == vllm::kS8)) return -1;
  }

  int max_par = 16;
  if (prob_n <= 4096) max_par = 16 * 8;
  int max_shared_mem_new = max_shared_mem;
  int rest_m = prob_m;
  int max_thread_m_blocks = 4;
  while (rest_m) {
    int par_count = rest_m / (max_thread_m_blocks * 16);
    if (par_count > max_par) par_count = max_par;
    int prob_m_split =
        par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

    int thread_k = thread_k_init;
    int thread_n = thread_n_init;

    int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);
    int m_block_size_8 = prob_m_split <= 8 && a_type.size_bits() == 16;

    exec_config_t exec_cfg;
    thread_config_t thread_tfg;
    if (thread_k != -1 && thread_n != -1) {
      thread_tfg = thread_config_t{thread_k, thread_n, default_threads};
      exec_cfg = exec_config_t{1, thread_tfg};
      if (!(prob_n % thread_n == 0)) return -1;
      if (!(prob_k % thread_k == 0)) return -1;
    } else {
      exec_cfg = determine_exec_config(
          a_type, b_type, c_type, s_type, prob_m_split, prob_n, prob_k,
          thread_m_blocks, m_block_size_8, num_bits, group_size, has_act_order,
          is_k_full, has_zp, is_zp_float, is_a_8bit, stages, max_shared_mem,
          sms);
      thread_tfg = exec_cfg.tb_cfg;
      if (thread_tfg.thread_n != -1) {
        if (prob_n / thread_tfg.thread_n *
                div_ceil(prob_m_split, thread_m_blocks * 16) * 4 <=
            sms) {
          if (is_valid_config({128, 64, 128}, thread_m_blocks, prob_m_split,
                              prob_n, prob_k, num_bits, group_size,
                              has_act_order, is_k_full, has_zp, is_zp_float,
                              is_a_8bit, stages, max_shared_mem_new)) {
            thread_tfg = {128, 64, 128};
            exec_cfg = {1, thread_tfg};
          }
        }
      }

      if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1) {
        max_thread_m_blocks--;
        continue;
      }
    }

    int num_threads = thread_tfg.num_threads;
    thread_k = thread_tfg.thread_k;
    thread_n = thread_tfg.thread_n;
    int blocks = sms * exec_cfg.blocks_per_sm;
    if (exec_cfg.blocks_per_sm > 1)
      max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    if (!is_valid_config(thread_tfg, thread_m_blocks, prob_m_split, prob_n,
                         prob_k, num_bits, group_size, has_act_order, is_k_full,
                         has_zp, is_zp_float, is_a_8bit, stages,
                         max_shared_mem_new))
      return -1;

    auto kernel = get_marlin_kernel(
        a_type, b_type, c_type, s_type, thread_m_blocks, thread_n_blocks,
        thread_k_blocks, m_block_size_8, has_act_order, has_zp, group_blocks,
        num_threads, is_zp_float, stages);

    if (kernel == MarlinDefault) return -1;

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         max_shared_mem_new);

    bool part_use_atomic_add =
        use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

    // clang-format off
    kernel<<<blocks, num_threads, max_shared_mem_new, stream>>>(
        A_ptr, B_ptr, C_ptr, C_tmp_ptr, bias_ptr, a_s_ptr, b_s_ptr, g_s_ptr, zp_ptr,
        g_idx_ptr, num_groups,
        prob_m_split, prob_n, prob_k, lda, locks, has_bias, part_use_atomic_add,
        use_fp32_reduce, max_shared_mem_new, tail_by_value);
    // clang-format on

    A_ptr += prob_m_split * (lda / (is_a_8bit ? 16 : 8));
    a_s_ptr += prob_m_split;
    C_ptr += prob_m_split * (prob_n / 8);
    rest_m -= prob_m_split;
  }
  return 0;
}

template <int const num_threads, int const num_bits>
__global__ void gptq_marlin_repack_w4_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr,
    uint32_t* __restrict__ out_ptr, int size_k, int size_n) {
  constexpr int pack_factor = 32 / num_bits;

  constexpr int target_tile_n_size = tile_n_size;
  constexpr int target_tile_k_size = tile_k_size;
  int k_tiles = size_k / target_tile_k_size;
  int n_tiles = size_n / target_tile_n_size;
  int block_k_tiles = div_ceil(k_tiles, gridDim.x);

  auto start_k_tile = blockIdx.x * block_k_tiles;
  if (start_k_tile >= k_tiles) {
    return;
  }

  int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

  auto wait_for_stage = [&]() {
    cp_async_wait<repack_stages - 2>();
    __syncthreads();
  };

  extern __shared__ int4 sh[];

  int4* sh_pipe_ptr = sh;

  constexpr int tile_ints = target_tile_k_size / pack_factor;

  constexpr int stage_n_threads = target_tile_n_size / 4;
  constexpr int stage_k_threads = tile_ints;
  constexpr int stage_size = stage_k_threads * stage_n_threads;

  auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      cp_async_fence();
      return;
    }

    int first_n = n_tile_id * target_tile_n_size;

    int4* sh_ptr = sh_pipe_ptr + stage_size * pipe;

    if (threadIdx.x < stage_size) {
      auto k_id = threadIdx.x / stage_n_threads;
      auto n_id = threadIdx.x % stage_n_threads;

      int first_k = k_tile_id * target_tile_k_size;
      int first_k_packed = first_k / pack_factor;

      cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                reinterpret_cast<int4 const*>(
                    &(b_q_weight_ptr[(first_k_packed + k_id) * size_n +
                                     first_n + (n_id * 4)])));
    }

    cp_async_fence();
  };

  auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      return;
    }

    auto warp_id = threadIdx.x / 32;
    auto th_id = threadIdx.x % 32;

    if (warp_id >= 4) {
      return;
    }

    int tc_col = th_id / 4;
    int tc_row = (th_id % 4) * 2;

    constexpr int tc_offsets[4] = {0, 1, 8, 9};

    int cur_n = warp_id * 16 + tc_col;

    constexpr int sh_stride = target_tile_n_size;
    constexpr uint32_t mask = (1 << num_bits) - 1;

    int4* sh_stage_ptr = sh_pipe_ptr + stage_size * pipe;
    uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);

    uint32_t vals[8];

    uint32_t b1_vals[tile_ints];
    uint32_t b2_vals[tile_ints];

#pragma unroll
    for (int i = 0; i < tile_ints; i++) {
      b1_vals[i] = sh_stage_int_ptr[cur_n + sh_stride * i];
      b2_vals[i] = sh_stage_int_ptr[cur_n + 8 + sh_stride * i];
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
      int cur_elem = tc_row + tc_offsets[i];
      int cur_int = cur_elem / pack_factor;
      int cur_pos = cur_elem % pack_factor;

      vals[i] = (b1_vals[cur_int] >> (cur_pos * num_bits)) & mask;
      vals[4 + i] = (b2_vals[cur_int] >> (cur_pos * num_bits)) & mask;
    }

    constexpr int rtile_size =
        target_tile_k_size * target_tile_n_size / pack_factor;
    int out_offset = (k_tile_id * n_tiles + n_tile_id) * rtile_size;

    if constexpr (num_bits == 4) {
      int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};

      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4);
      }

      out_ptr[out_offset + th_id * 4 + warp_id] = res;
    } else {
      constexpr int pack_idx[4] = {0, 2, 1, 3};

      uint32_t res1 = 0;
      uint32_t res2 = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        res1 |= vals[pack_idx[i]] << (i * 8);
        res2 |= vals[4 + pack_idx[i]] << (i * 8);
      }

      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;
      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
    }
  };

  auto start_pipes = [&](int k_tile_id, int n_tile_id) {
#pragma unroll
    for (int pipe = 0; pipe < repack_stages - 1; pipe++) {
      fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
    }

    wait_for_stage();
  };
#pragma unroll
  for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
    int n_tile_id = 0;

    start_pipes(k_tile_id, n_tile_id);

    while (n_tile_id < n_tiles) {
#pragma unroll
      for (int pipe = 0; pipe < repack_stages; pipe++) {
        fetch_to_shared((pipe + repack_stages - 1) % repack_stages, k_tile_id,
                        n_tile_id + pipe + repack_stages - 1);
        repack_tile(pipe, k_tile_id, n_tile_id + pipe);
        wait_for_stage();
      }
      n_tile_id += repack_stages;
    }
  }
}

template <int const num_bits>
static int gptq_marlin_repack_w4_raw(uint32_t const* b_q_weight_ptr,
                                     uint32_t* out_ptr, int size_k, int size_n,
                                     int dev, cudaStream_t stream) {
  if (!(size_k % tile_k_size == 0)) return -1;
  if (!(size_n % tile_n_size == 0)) return -1;

  int blocks;
  cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev);

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (!(max_shared_mem > 0)) return -1;

  cudaFuncSetAttribute(gptq_marlin_repack_w4_kernel<repack_threads, num_bits>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       max_shared_mem);
  gptq_marlin_repack_w4_kernel<repack_threads, num_bits>
      <<<blocks, repack_threads, max_shared_mem, stream>>>(b_q_weight_ptr,
                                                           out_ptr, size_k,
                                                           size_n);
  return 0;
}

template <int const num_threads>
__global__ void gptq_marlin_repack_w4a8_kernel(
    uint32_t const* __restrict__ b_q_weight_ptr,
    uint32_t* __restrict__ out_ptr, int size_k, int size_n) {
  constexpr int num_bits = 4;
  constexpr int pack_factor = 32 / num_bits;

  constexpr int target_tile_n_size = tile_n_size / 2;
  constexpr int target_tile_k_size = tile_k_size * 2;
  int k_tiles = size_k / target_tile_k_size;
  int n_tiles = size_n / target_tile_n_size;
  int block_k_tiles = div_ceil(k_tiles, gridDim.x);

  auto start_k_tile = blockIdx.x * block_k_tiles;
  if (start_k_tile >= k_tiles) {
    return;
  }

  int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

  auto wait_for_stage = [&]() {
    cp_async_wait<repack_stages - 2>();
    __syncthreads();
  };

  extern __shared__ int4 sh[];

  int4* sh_pipe_ptr = sh;

  constexpr int tile_ints = target_tile_k_size / pack_factor;

  constexpr int stage_n_threads = target_tile_n_size / 4;
  constexpr int stage_k_threads = tile_ints;
  constexpr int stage_size = stage_k_threads * stage_n_threads;

  auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      cp_async_fence();
      return;
    }

    int first_n = n_tile_id * target_tile_n_size;

    int4* sh_ptr = sh_pipe_ptr + stage_size * pipe;

    if (threadIdx.x < stage_size) {
      auto k_id = threadIdx.x / stage_n_threads;
      auto n_id = threadIdx.x % stage_n_threads;

      int first_k = k_tile_id * target_tile_k_size;
      int first_k_packed = first_k / pack_factor;

      cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                reinterpret_cast<int4 const*>(
                    &(b_q_weight_ptr[(first_k_packed + k_id) * size_n +
                                     first_n + (n_id * 4)])));
    }

    cp_async_fence();
  };

  auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      return;
    }

    auto warp_id = threadIdx.x / 32;
    auto th_id = threadIdx.x % 32;

    if (warp_id >= 4) {
      return;
    }

    int tc_col = th_id / 4;
    int tc_row = (th_id % 4) * 4;

    int cur_n = (warp_id / 2) * 16 + tc_col;

    constexpr int sh_stride = target_tile_n_size;
    constexpr uint32_t mask = (1 << num_bits) - 1;

    int4* sh_stage_ptr = sh_pipe_ptr + stage_size * pipe;
    uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);

    uint32_t vals[8];

    uint32_t b1_vals[tile_ints];

#pragma unroll
    for (int i = 0; i < tile_ints; i++) {
      b1_vals[i] = sh_stage_int_ptr[cur_n + sh_stride * i + (warp_id % 2) * 8];
    }

#pragma unroll
    for (int i = 0; i < 4; i++) {
      int cur_elem = tc_row + i;
      int cur_int = cur_elem / pack_factor;
      int cur_pos = cur_elem % pack_factor;

      vals[i] = (b1_vals[cur_int] >> (cur_pos * num_bits)) & mask;
      vals[4 + i] =
          (b1_vals[cur_int + tile_ints / 2] >> (cur_pos * num_bits)) & mask;
    }

    constexpr int rtile_size =
        target_tile_k_size * target_tile_n_size / pack_factor;
    int out_offset = (k_tile_id * n_tiles + n_tile_id) * rtile_size;

    int pack_idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};

    uint32_t res = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      res |= vals[pack_idx[i]] << (i * 4);
    }

    out_ptr[out_offset + th_id * 4 + warp_id] = res;
  };

  auto start_pipes = [&](int k_tile_id, int n_tile_id) {
#pragma unroll
    for (int pipe = 0; pipe < repack_stages - 1; pipe++) {
      fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
    }

    wait_for_stage();
  };
#pragma unroll
  for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
    int n_tile_id = 0;

    start_pipes(k_tile_id, n_tile_id);

    while (n_tile_id < n_tiles) {
#pragma unroll
      for (int pipe = 0; pipe < repack_stages; pipe++) {
        fetch_to_shared((pipe + repack_stages - 1) % repack_stages, k_tile_id,
                        n_tile_id + pipe + repack_stages - 1);
        repack_tile(pipe, k_tile_id, n_tile_id + pipe);
        wait_for_stage();
      }
      n_tile_id += repack_stages;
    }
  }
}

static int gptq_marlin_repack_w4a8_raw(uint32_t const* b_q_weight_ptr,
                                       uint32_t* out_ptr, int size_k,
                                       int size_n, int dev,
                                       cudaStream_t stream) {
  if (!(size_k % (tile_k_size * 2) == 0)) return -1;
  if (!(size_n % tile_n_size == 0)) return -1;

  int blocks;
  cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev);

  int max_shared_mem = 0;
  cudaDeviceGetAttribute(&max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  if (!(max_shared_mem > 0)) return -1;

  cudaFuncSetAttribute(gptq_marlin_repack_w4a8_kernel<repack_threads>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       max_shared_mem);
  gptq_marlin_repack_w4a8_kernel<repack_threads>
      <<<blocks, repack_threads, max_shared_mem, stream>>>(b_q_weight_ptr,
                                                           out_ptr, size_k,
                                                           size_n);
  return 0;
}

__global__ void marlin_int4_fp8_preprocess_u4b8_kernel(
    const int32_t* __restrict__ qweight, int32_t* __restrict__ output) {
  int32_t val = qweight[blockIdx.x * 32 + threadIdx.x];
  int32_t new_val = 0;

#pragma unroll
  for (int32_t i = 0; i < 8; i++) {
    int32_t single_val = val & 0xF;
    single_val = single_val >= 8 ? single_val - 8 : 15 - single_val;
    new_val |= single_val << (i * 4);
    val >>= 4;
  }

  output[blockIdx.x * 32 + threadIdx.x] = new_val;
}

}

extern "C" {

int nv_kernels_marlin_workspace_elems(int* out_elems) {
  if (out_elems == nullptr) return -1;
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return -2;
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
      cudaSuccess)
    return -2;
  *out_elems = sms;
  return 0;
}

static int marlin_gemm_w4a16_impl(void* stream, const void* a_bf16,
                                  const void* b_q_marlin, const void* b_scales,
                                  void* c_out, void* workspace, int m, int n,
                                  int k, int group_size, int a_is_bf16,
                                  int c_prezeroed,
                                  const marlin::MarlinM1Tail* m1_tail) {
  if (m == 0) return 0;
  if (a_bf16 == nullptr || b_q_marlin == nullptr || b_scales == nullptr ||
      c_out == nullptr || workspace == nullptr)
    return -1;
  if (m < 0 || n <= 0 || k <= 0) return -1;

  if (k % marlin::tile_size != 0) return -1;
  if (n % marlin::min_thread_n != 0) return -1;
  if (group_size != -1) {
    if (group_size <= 0 || k % group_size != 0) return -1;
  }

  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return -2;
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
      cudaSuccess)
    return -2;

  vllm::ScalarType a_type = a_is_bf16 ? vllm::kBFloat16 : vllm::kFloat16;
  vllm::ScalarType c_type = a_type;
  vllm::ScalarType s_type = a_type;
  vllm::ScalarType b_type = vllm::kU4B8;

  int num_groups = (group_size == -1) ? 1 : (k / group_size);

  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);

  size_t c_bytes = (size_t)m * (size_t)n * 2;
  if (!c_prezeroed) cudaMemsetAsync(c_out, 0, c_bytes, cu_stream);

  int rc = marlin::marlin_mm_raw(
      a_bf16, b_q_marlin, c_out, nullptr,
      nullptr, nullptr, (void*)b_scales,
      nullptr, nullptr, nullptr, nullptr,
      nullptr, m, n, k, k, workspace, a_type, b_type, c_type,
      s_type, false, false, true,
      false, num_groups, group_size, dev, cu_stream,
      -1, -1, sms, true,
      false, false,
      m1_tail);

  if (rc == 0) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) rc = -3;
  }
  return rc;
}

int nv_kernels_marlin_gemm_w4a16(void* stream, const void* a_bf16,
                                 const void* b_q_marlin, const void* b_scales,
                                 void* c_out, void* workspace, int m, int n,
                                 int k, int group_size, int a_is_bf16) {
  return marlin_gemm_w4a16_impl(stream, a_bf16, b_q_marlin, b_scales, c_out,
                                workspace, m, n, k, group_size, a_is_bf16, 0,
                                nullptr);
}

int nv_kernels_marlin_gemm_w4a16_prezeroed(void* stream, const void* a_bf16,
                                           const void* b_q_marlin,
                                           const void* b_scales, void* c_out,
                                           void* workspace, int m, int n,
                                           int k, int group_size,
                                           int a_is_bf16) {
  return marlin_gemm_w4a16_impl(stream, a_bf16, b_q_marlin, b_scales, c_out,
                                workspace, m, n, k, group_size, a_is_bf16, 1,
                                nullptr);
}

int nv_kernels_marlin_gemm_w4a16_m1_tail(
    void* stream, const void* a_bf16, const void* b_q_marlin,
    const void* b_scales, void* c_out, void* workspace, int n, int k,
    int group_size, int kind, const uint16_t* res, const uint16_t* w,
    uint16_t* h_out, float* rstd_out, const uint16_t* next_w,
    uint16_t* normed_out, const float* pli, uint16_t* gelu_out, float eps,
    float scale, float eps_next, unsigned int* counter) {
  if (counter == nullptr) return -1;
  if (kind == marlin::kMarlinM1TailRmsnormAddScaleThenOptionalNextNormChain) {
    if (res == nullptr || w == nullptr || h_out == nullptr) return -1;
    if (normed_out != nullptr && next_w == nullptr) return -1;
  } else if (kind == marlin::kMarlinM1TailGeluTimesPerLayerInput) {
    if (pli == nullptr || gelu_out == nullptr) return -1;
  } else {
    return -1;
  }
  marlin::MarlinM1Tail tail = {
      kind,
      reinterpret_cast<const __nv_bfloat16*>(res),
      reinterpret_cast<const __nv_bfloat16*>(w),
      reinterpret_cast<__nv_bfloat16*>(h_out),
      rstd_out,
      reinterpret_cast<const __nv_bfloat16*>(next_w),
      reinterpret_cast<__nv_bfloat16*>(normed_out),
      pli,
      reinterpret_cast<__nv_bfloat16*>(gelu_out),
      eps,
      scale,
      eps_next,
      counter,
  };
  return marlin_gemm_w4a16_impl(stream, a_bf16, b_q_marlin, b_scales, c_out,
                                workspace, 1, n, k, group_size, 1, 1, &tail);
}

int nv_kernels_marlin_repack_w4a16(void* stream, const void* b_q_packed,
                                   void* b_q_marlin_out, int k, int n,
                                   int num_bits) {
  if (b_q_packed == nullptr || b_q_marlin_out == nullptr) return -1;
  if (num_bits != 4 && num_bits != 8) return -1;
  if (k <= 0 || n <= 0) return -1;
  if (k % marlin::tile_k_size != 0) return -1;
  if (n % marlin::tile_n_size != 0) return -1;

  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return -2;
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);

  int rc = num_bits == 8
      ? marlin::gptq_marlin_repack_w4_raw<8>(
            reinterpret_cast<uint32_t const*>(b_q_packed),
            reinterpret_cast<uint32_t*>(b_q_marlin_out), k, n, dev, cu_stream)
      : marlin::gptq_marlin_repack_w4_raw<4>(
            reinterpret_cast<uint32_t const*>(b_q_packed),
            reinterpret_cast<uint32_t*>(b_q_marlin_out), k, n, dev, cu_stream);

  if (rc == 0) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) rc = -3;
  }
  return rc;
}

}

extern "C" int nv_kernels_marlin_gemm_w4a16_ex(
    void* stream,
    const void* a_bf16,
    const void* b_q_marlin,
    const void* b_scales,
    void* c_out,
    void* c_tmp,
    void* workspace,
    int m,
    int n,
    int k,
    int group_size,
    int a_is_bf16,
    int use_atomic_add,
    int use_fp32_reduce
) {
  if (a_bf16 == nullptr || b_q_marlin == nullptr || b_scales == nullptr ||
      c_out == nullptr || workspace == nullptr)
    return -1;
  if (m < 0 || n <= 0 || k <= 0) return -1;
  if (k % marlin::tile_size != 0) return -1;
  if (n % marlin::min_thread_n != 0) return -1;
  if (group_size != -1) {
    if (group_size <= 0 || k % group_size != 0) return -1;
  }
  if (use_fp32_reduce && c_tmp == nullptr) return -1;

  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return -2;
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
      cudaSuccess)
    return -2;

  vllm::ScalarType a_type = a_is_bf16 ? vllm::kBFloat16 : vllm::kFloat16;
  vllm::ScalarType c_type = a_type;
  vllm::ScalarType s_type = a_type;
  vllm::ScalarType b_type = vllm::kU4B8;
  int num_groups = (group_size == -1) ? 1 : (k / group_size);
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);

  if (use_atomic_add) {
    size_t c_bytes = (size_t)m * (size_t)n * 2;
    cudaMemsetAsync(c_out, 0, c_bytes, cu_stream);
  }

  int rc = marlin::marlin_mm_raw(
      a_bf16, b_q_marlin, c_out, c_tmp,
      nullptr, nullptr, (void*)b_scales,
      nullptr, nullptr, nullptr, nullptr,
      nullptr, m, n, k, k, workspace, a_type, b_type, c_type,
      s_type, false, false, true,
      false, num_groups, group_size, dev, cu_stream,
      -1, -1, sms,
      use_atomic_add != 0, use_fp32_reduce != 0, false,
      nullptr);

  if (rc == 0) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) rc = -3;
  }
  return rc;
}

static const int kMarlinNvfp4CtmpFloatsPerBlock16RowsBy128ColsIsTheLargestSmallBatchTile = 16 * 128;

extern "C" int nv_kernels_marlin_gemm_fp8_w8a16(
    void* stream,
    const void* a_bf16,
    const void* b_q_marlin,
    const void* b_scales_bf16_per_channel,
    void* c_out,
    void* c_tmp_f32,
    long long c_tmp_floats,
    void* workspace,
    int m,
    int n,
    int k
) {
  if (a_bf16 == nullptr || b_q_marlin == nullptr || b_scales_bf16_per_channel == nullptr ||
      c_out == nullptr || c_tmp_f32 == nullptr || workspace == nullptr)
    return -1;
  if (m == 0) return 0;
  if (m < 0 || n <= 0 || k <= 0) return -1;
  if (k % marlin::tile_size != 0) return -1;
  if (n % marlin::min_thread_n != 0) return -1;

  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return -2;
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
      cudaSuccess)
    return -2;
  if (c_tmp_floats < (long long)sms * kMarlinNvfp4CtmpFloatsPerBlock16RowsBy128ColsIsTheLargestSmallBatchTile)
    return -5;

  vllm::ScalarType a_type = vllm::kBFloat16;
  vllm::ScalarType c_type = vllm::kBFloat16;
  vllm::ScalarType s_type = vllm::kBFloat16;
  vllm::ScalarType b_type = vllm::kFE4M3fn;
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);

  int rc = marlin::marlin_mm_raw(
      a_bf16, b_q_marlin, c_out, c_tmp_f32,
      nullptr, nullptr, (void*)b_scales_bf16_per_channel,
      nullptr, nullptr, nullptr, nullptr,
      nullptr, m, n, k, k, workspace, a_type, b_type, c_type,
      s_type, false, false, true,
      false, 1, -1, dev, cu_stream,
      -1, -1, sms,
      false, true, false,
      nullptr);

  if (rc == 0) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) rc = -3;
  }
  return rc;
}

extern "C" int nv_kernels_marlin_nvfp4_c_tmp_floats(long long* out_floats) {
  if (out_floats == nullptr) return -1;
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return -2;
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
      cudaSuccess)
    return -2;
  *out_floats = (long long)sms * kMarlinNvfp4CtmpFloatsPerBlock16RowsBy128ColsIsTheLargestSmallBatchTile;
  return 0;
}

extern "C" int nv_kernels_marlin_gemm_nvfp4_w4a16(
    void* stream,
    const void* a_bf16,
    const void* b_q_marlin,
    const void* b_scales_e5m3_bytes,
    const float* global_scale_dev,
    void* c_out,
    void* c_tmp_f32,
    long long c_tmp_floats,
    void* workspace,
    int m,
    int n,
    int k
) {
  if (a_bf16 == nullptr || b_q_marlin == nullptr || b_scales_e5m3_bytes == nullptr ||
      global_scale_dev == nullptr || c_out == nullptr || c_tmp_f32 == nullptr ||
      workspace == nullptr)
    return -1;
  if (m == 0) return 0;
  if (m < 0 || n <= 0 || k <= 0) return -1;
  if (k % marlin::tile_size != 0) return -1;
  if (n % marlin::min_thread_n != 0) return -1;
  const int group_size = 16;
  if (k % group_size != 0) return -1;

  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return -2;
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
      cudaSuccess)
    return -2;
  if (c_tmp_floats < (long long)sms * kMarlinNvfp4CtmpFloatsPerBlock16RowsBy128ColsIsTheLargestSmallBatchTile)
    return -5;

  vllm::ScalarType a_type = vllm::kBFloat16;
  vllm::ScalarType c_type = vllm::kBFloat16;
  vllm::ScalarType s_type = vllm::kFE4M3fn;
  vllm::ScalarType b_type = vllm::kFE2M1f;
  int num_groups = k / group_size;
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);

  int rc = marlin::marlin_mm_raw(
      a_bf16, b_q_marlin, c_out, c_tmp_f32,
      nullptr, nullptr, (void*)b_scales_e5m3_bytes,
      (void*)global_scale_dev, nullptr, nullptr, nullptr,
      nullptr, m, n, k, k, workspace, a_type, b_type, c_type,
      s_type, false, false, true,
      false, num_groups, group_size, dev, cu_stream,
      -1, -1, sms,
      false, true, false,
      nullptr);

  if (rc == 0) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) rc = -3;
  }
  return rc;
}

extern "C" int nv_kernels_marlin_int4_fp8_preprocess(void* stream,
                                                     const void* qweight,
                                                     void* out,
                                                     long long numel_i32) {
  if (qweight == nullptr || out == nullptr) return -1;
  if (numel_i32 <= 0 || numel_i32 % 32 != 0) return -1;
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);
  long long blocks = numel_i32 / 32;
  marlin::marlin_int4_fp8_preprocess_u4b8_kernel<<<(unsigned)blocks, 32, 0,
                                                   cu_stream>>>(
      reinterpret_cast<const int32_t*>(qweight),
      reinterpret_cast<int32_t*>(out));
  cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? 0 : -3;
}

extern "C" int nv_kernels_marlin_repack_w4a8(void* stream,
                                             const void* b_q_packed,
                                             void* b_q_marlin_out, int k,
                                             int n, int num_bits) {
  if (b_q_packed == nullptr || b_q_marlin_out == nullptr) return -1;
  if (num_bits != 4) return -1;
  if (k <= 0 || n <= 0) return -1;
  if (k % (marlin::tile_k_size * 2) != 0) return -1;
  if (n % marlin::tile_n_size != 0) return -1;

  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return -2;
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);

  int rc = marlin::gptq_marlin_repack_w4a8_raw(
      reinterpret_cast<uint32_t const*>(b_q_packed),
      reinterpret_cast<uint32_t*>(b_q_marlin_out), k, n, dev, cu_stream);

  if (rc == 0) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) rc = -3;
  }
  return rc;
}

extern "C" int nv_kernels_marlin_gemm_w4a8_fp8(
    void* stream, const void* a_fp8, const void* a_scales_f32,
    const void* b_q_marlin, const void* b_scales_bf16, void* c_out_bf16,
    void* workspace, int m, int n, int k, int group_size, int c_prezeroed) {
  if (m == 0) return 0;
  if (a_fp8 == nullptr || a_scales_f32 == nullptr || b_q_marlin == nullptr ||
      b_scales_bf16 == nullptr || c_out_bf16 == nullptr ||
      workspace == nullptr)
    return -1;
  if (m < 0 || n <= 0 || k <= 0) return -1;

  if (k % marlin::tile_size != 0) return -1;
  if (k % 16 != 0) return -1;
  if (n % marlin::min_thread_n != 0) return -1;
  if (group_size != -1) {
    if (group_size < 32 || k % group_size != 0) return -1;
  }

  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return -2;
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) !=
      cudaSuccess)
    return -2;

  vllm::ScalarType a_type = vllm::kFE4M3fn;
  vllm::ScalarType b_type = vllm::kU4B8;
  vllm::ScalarType c_type = vllm::kBFloat16;
  vllm::ScalarType s_type = vllm::kBFloat16;

  int num_groups = (group_size == -1) ? 1 : (k / group_size);

  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);

  size_t c_bytes = (size_t)m * (size_t)n * 2;
  if (!c_prezeroed) cudaMemsetAsync(c_out_bf16, 0, c_bytes, cu_stream);

  int rc = marlin::marlin_mm_raw(
      a_fp8, b_q_marlin, c_out_bf16, nullptr,
      nullptr, (void*)a_scales_f32, (void*)b_scales_bf16,
      nullptr, nullptr, nullptr, nullptr,
      nullptr, m, n, k, k, workspace, a_type, b_type, c_type,
      s_type, false, false, true,
      false, num_groups, group_size, dev, cu_stream,
      -1, -1, sms, true,
      false, false,
      nullptr);

  if (rc == 0) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) rc = -3;
  }
  return rc;
}
