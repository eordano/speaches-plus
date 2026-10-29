use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{bail, Result};
use nv_kernels::wgpu_backend::{na, WgpuContext};
use na::{storage_entry, uniform_entry};

pub const WG_THREADS: u32 = 128;
pub const HEAD_DIM: u32 = 128;

pub const THE_FUSED_VISION_ATTENTION_IS_STEEL_BECAUSE_THE_MISSING_OP_WAS_NAGAS_AND_THE_MPP_REWRITE_PRICED_ITS_OWN_CEILING: &str =
    "Every fused-attention form died in WGSL for one reason: naga exposes no per-element access \
     to a cooperative accumulator, so the online-softmax rescale between QK and PV cannot be \
     expressed. MSL exposes it, and two vehicles were priced at the vision shape (13600 patches, \
     hd 128, 12 heads, non-causal) against the 6.15s zmerge control: an mpp matmul2d kernel in \
     the in-tree na_attn idiom read 38.845s with head-dim-split simdgroups and 20.841s with \
     steel's own row-split geometry -- matmul2d at 8x32x128 tiles sustains ~2.6 TFLOP/s and no \
     geometry closes that -- while the vendored steel kernel itself \
     (attention<half,32,16,128,4,1>, third-party/mlx-steel-attn) reads 4.519s +-0.001, minus \
     26.5 percent, the first vision number below the rival's own 5.31s tower. It dispatches on \
     the same wgpu queue in the same pass list as every WGSL kernel, so the design pays zero \
     extra syncs; the slab, both softmax passes, the context GEMM and the merge never dispatch \
     under the arm. Numerics: closest of every arm to the cpu f32 twin (merged rms 4.4427e-2 \
     against the shipped default's own 4.5656e-2) and the corpus token gate holds all twelve \
     stable pages at exactly 0.0000 with every bistable on its standing trajectory and token \
     count.";

pub const STEEL_BQ: u32 = 32;
pub const STEEL_BK: u32 = 16;

pub type SteelAttnCfg = (u32, u32, u32, u32);

pub const STEEL_ATTN_DEFAULT_CFG: SteelAttnCfg = (STEEL_BQ, STEEL_BK, 4, 1);

pub const STEEL_ATTN_TILE_LADDER_CROWNS_THE_SHIPPED_TILE_EVERY_OTHER_ARM_LOSES_OR_WASHES_AT_13600_PATCHES: &[SteelAttnCfg] = &[
    STEEL_ATTN_DEFAULT_CFG,
    (32, 32, 4, 1),
    (32, 64, 4, 1),
    (64, 16, 8, 1),
    (64, 32, 8, 1),
    (16, 16, 2, 1),
    (8, 16, 1, 1),
];

pub const STEEL_ATTN_SUB_BQ32_ARMS_EXIST_BECAUSE_CROP_GRIDS_STARVE_THE_76_CORE_GPU: &str =
    "A crop encode dispatches ceil(patches/bq) x 12 threadgroups per attention pass, and the \
     gazette's region crops run 64-524 patches, so the shipped bq32 tile launches 24-192 \
     threadgroups against 76 GPU cores -- the same grid starvation the decode trunk measured \
     below m=128. bq16 and bq8 keep bk=16 and TQ==1, so they move only which workgroup owns a \
     query row and stay in the bitwise class; they exist for the crop-shape ladder and any \
     per-shape selection, never for the 13600-patch whole page where bq32 is the crowned tile.";

pub const STEEL_ATTN_CFGS_ALL_KEEP_TQ_1_AND_FIT_METALS_32768B_WORKGROUP_LIMIT: () = {
    let mut i = 0;
    let cfgs = STEEL_ATTN_TILE_LADDER_CROWNS_THE_SHIPPED_TILE_EVERY_OTHER_ARM_LOSES_OR_WASHES_AT_13600_PATCHES;
    while i < cfgs.len() {
        let (bq, bk, wm, wn) = cfgs[i];
        assert!(bq == 8 * wm * wn);
        assert!(bk % 8 == 0);
        let kv = if (bk + 8) * HEAD_DIM > bk * (HEAD_DIM + 8) {
            (bk + 8) * HEAD_DIM
        } else {
            bk * (HEAD_DIM + 8)
        };
        assert!((bq * (HEAD_DIM + 8) + kv) * 2 <= 32768);
        i += 1;
    }
};

pub const STEEL_IS_CONCATENATED_AND_CONSTANT_BAKED_BECAUSE_RUNTIME_MSL_HAS_NO_INCLUDE_PATH_AND_PASSTHROUGH_SETS_NO_FUNCTION_CONSTANTS: &str =
    "newLibraryWithSource resolves only system includes and the passthrough pipeline path calls      newFunctionWithName with no MTLFunctionConstantValues, so the vendored steel headers      (third-party/mlx-steel-attn, MIT, verbatim) are concatenated in dependency order with their      local includes stripped, the five function constants are baked as literals (non-causal,      unmasked, unaligned edges), and the instantiation is appended -- the same shape mlx ships      for head_dim 128, attention<half, 32, 16, 128, 4, 1>.";

const STEEL_FILES: &[&str] = &[
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/defines.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/utils/type_traits.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/utils/integral_constant.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/utils.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/gemm/params.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/attn/params.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/attn/transforms.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/attn/loader.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/attn/mma.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/attn/attn.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"),
];

pub const STEEL_LIMITS_SHIM_MATCHES_THE_FLOAT_SPECIALIZATION_IN_MLX_KERNELS_UTILS_H: &str = "
#include <metal_stdlib>
template <typename U>
struct Limits {
  static constexpr constant U max = metal::numeric_limits<U>::infinity();
  static constexpr constant U min = -metal::numeric_limits<U>::infinity();
  static constexpr constant U finite_max = metal::numeric_limits<U>::max();
  static constexpr constant U finite_min = -metal::numeric_limits<U>::max();
};
";

pub fn steel_entry(align_q: bool, align_k: bool) -> String {
    steel_entry_cfg(STEEL_ATTN_DEFAULT_CFG, align_q, align_k)
}

pub fn steel_entry_cfg(cfg: SteelAttnCfg, align_q: bool, align_k: bool) -> String {
    let (bq, bk, wm, wn) = cfg;
    format!(
        "dv_steel_attn_bq{bq}_bk{bk}_wm{wm}_wn{wn}_q{}k{}",
        u8::from(align_q),
        u8::from(align_k)
    )
}

pub fn steel_source(align_q: bool, align_k: bool) -> String {
    steel_source_cfg(STEEL_ATTN_DEFAULT_CFG, align_q, align_k)
}

pub fn steel_source_cfg(cfg: SteelAttnCfg, align_q: bool, align_k: bool) -> String {
    let _ = STEEL_IS_CONCATENATED_AND_CONSTANT_BAKED_BECAUSE_RUNTIME_MSL_HAS_NO_INCLUDE_PATH_AND_PASSTHROUGH_SETS_NO_FUNCTION_CONSTANTS;
    let mut b =
        String::from(STEEL_LIMITS_SHIM_MATCHES_THE_FLOAT_SPECIALIZATION_IN_MLX_KERNELS_UTILS_H);
    for f in STEEL_FILES {
        for line in f.lines() {
            let t = line.trim_start();
            if t.starts_with("#include \"mlx/") || t == "#pragma once" {
                continue;
            }
            b.push_str(line);
            b.push('\n');
        }
    }
    let aq: &str = if align_q {
        "constant bool align_Q = true;"
    } else {
        "constant bool align_Q = false;"
    };
    let ak: &str = if align_k {
        "constant bool align_K = true;"
    } else {
        "constant bool align_K = false;"
    };
    for (from, to) in [
        ("constant bool align_Q [[function_constant(200)]];", aq),
        ("constant bool align_K [[function_constant(201)]];", ak),
        (
            "constant bool has_mask [[function_constant(300)]];",
            "constant bool has_mask = false;",
        ),
        (
            "constant bool do_causal [[function_constant(301)]];",
            "constant bool do_causal = false;",
        ),
        (
            "constant bool has_sinks [[function_constant(302)]];",
            "constant bool has_sinks = false;",
        ),
        (", function_constant(has_mask)]]", "]]"),
        (", function_constant(has_sinks)]]", "]]"),
    ] {
        assert!(b.contains(from), "steel source lost the anchor {from}");
        b = b.replace(from, to);
    }
    let (bq, bk, wm, wn) = cfg;
    b.push_str(&format!(
        "\ntemplate [[host_name(\"{}\")]] [[kernel]] decltype(attention<half, {bq}, {bk}, 128, {wm}, {wn}, float, float>) attention<half, {bq}, {bk}, 128, {wm}, {wn}, float, float>;\n",
        steel_entry_cfg(cfg, align_q, align_k)
    ));
    b
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SteelAttnParams {
    pub b: i32,
    pub h: i32,
    pub d: i32,
    pub q_l: i32,
    pub k_l: i32,
    pub gqa_factor: i32,
    pub scale: f32,
    pub nq: i32,
    pub nk: i32,
    pub nq_aligned: i32,
    pub nk_aligned: i32,
    pub q_l_rem: i32,
    pub k_l_rem: i32,
    pub q_l_off: i32,
    pub q_strides: [i64; 3],
    pub k_strides: [i64; 3],
    pub v_strides: [i64; 3],
    pub o_strides: [i64; 3],
}

const _: () = assert!(std::mem::size_of::<SteelAttnParams>() == 152);

pub struct SteelPipelines {
    pub pipeline: Arc<wgpu::ComputePipeline>,
    pub layout: wgpu::BindGroupLayout,
}

type SteelCache = Mutex<
    Vec<(
        (wgpu::Device, SteelAttnCfg, bool, bool),
        std::result::Result<Arc<SteelPipelines>, String>,
    )>,
>;

fn steel_cache() -> &'static SteelCache {
    static CACHE: OnceLock<SteelCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn steel_build(
    ctx: &WgpuContext,
    cfg: SteelAttnCfg,
    align_q: bool,
    align_k: bool,
) -> Result<Arc<SteelPipelines>> {
    let entry: &'static str = Box::leak(steel_entry_cfg(cfg, align_q, align_k).into_boxed_str());
    let src: &'static str = Box::leak(steel_source_cfg(cfg, align_q, align_k).into_boxed_str());
    let module = na::msl_module(
        ctx,
        "dv-steel-attn",
        &[(entry, (cfg.2 * cfg.3 * 32, 1, 1))],
        src,
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let entries = [
        storage_entry(0, true),
        storage_entry(1, true),
        storage_entry(2, true),
        storage_entry(3, false),
        uniform_entry(4),
        uniform_entry(5),
        storage_entry(6, true),
        storage_entry(7, true),
    ];
    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dv-steel-attn"),
            entries: &entries,
        });
    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dv-steel-attn"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let pipeline = Arc::new(
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            }),
    );
    if let Some(err) = pollster::block_on(scope.pop()) {
        bail!("dv-steel-attn pipeline: {err}");
    }
    Ok(Arc::new(SteelPipelines {
        pipeline,
        layout: bgl,
    }))
}

pub fn steel_pipelines(
    ctx: &'static WgpuContext,
    align_q: bool,
    align_k: bool,
) -> Result<Arc<SteelPipelines>> {
    steel_pipelines_cfg(ctx, STEEL_ATTN_DEFAULT_CFG, align_q, align_k)
}

pub fn steel_pipelines_cfg(
    ctx: &'static WgpuContext,
    cfg: SteelAttnCfg,
    align_q: bool,
    align_k: bool,
) -> Result<Arc<SteelPipelines>> {
    let mut guard = steel_cache().lock().unwrap();
    for ((dev, c, aq, ak), entry) in guard.iter() {
        if *dev == ctx.device && *c == cfg && *aq == align_q && *ak == align_k {
            return entry.clone().map_err(|e| anyhow::anyhow!("{e}"));
        }
    }
    let built = steel_build(ctx, cfg, align_q, align_k);
    let stored = match &built {
        Ok(p) => Ok(p.clone()),
        Err(e) => Err(format!("{e:#}")),
    };
    guard.push(((ctx.device.clone(), cfg, align_q, align_k), stored));
    built
}

pub fn steel_grid(heads: u32, n_real: u32) -> (u32, u32, u32) {
    steel_grid_cfg(STEEL_ATTN_DEFAULT_CFG, heads, n_real)
}

pub fn steel_grid_cfg(cfg: SteelAttnCfg, heads: u32, n_real: u32) -> (u32, u32, u32) {
    (n_real.div_ceil(cfg.0), heads, 1)
}

pub const STEEL_GEMM_CONFIGS: &[(u32, u32, u32, u32, u32)] = &[
    (16, 32, 16, 2, 2),
    (16, 64, 16, 2, 2),
    (32, 32, 16, 2, 2),
    (32, 64, 16, 2, 2),
    (64, 32, 16, 2, 2),
    (64, 64, 16, 2, 2),
    (64, 64, 32, 2, 2),
    (64, 128, 16, 2, 2),
    (128, 64, 16, 2, 2),
    (64, 64, 64, 2, 2),
    (64, 128, 32, 2, 2),
    (128, 64, 32, 2, 2),
    (128, 128, 32, 2, 2),
];

pub const STEEL_GEMM_CONFIGS_ALL_FIT_METALS_32768B_WORKGROUP_LIMIT_BM_BK_PLUS_8_PLUS_BN_BK_PLUS_8_HALVES:
    () = {
    let mut i = 0;
    while i < STEEL_GEMM_CONFIGS.len() {
        let (bm, bn, bk, _, _) = STEEL_GEMM_CONFIGS[i];
        assert!((bm * (bk + 8) + bn * (bk + 8)) * 2 <= 32768);
        i += 1;
    }
};

pub const VISENC_CROP_SHAPES_STARVE_THE_GRID_SO_THE_CHAMPION_TILE_IS_PER_SHAPE: &str =
    "The vision cfg ladders re-run at the encode's real crop shapes (gazette crops 184-448 \
     patches median 208; big-fixture serial crops 192-2520) found the 13600-patch crowns \
     grid-starved exactly like the decode trunk below m=128: at m=208 the shipped 64x64x32 \
     projection tile launches 96-288 workgroups against 76 cores and loses 8-40 percent per \
     family to smaller instantiated tiles, and at n<=448 the bq32 attention tile loses up to \
     35 percent to bq16. Champions are two-stamp stable, every arm bit-identical (bk16 \
     attention arms and all gemm tiles), and the maps below mirror pf_steel_proj_cfg_for(m) \
     from the decode round: per-shape selection over the same instantiated, compile-asserted \
     config lists. The whole-page crowns are unchanged -- both maps return the shipped tiles \
     above their crossovers.";

pub fn steel_attn_cfg_for(n_real: u32) -> SteelAttnCfg {
    if n_real < 512 {
        (16, 16, 2, 1)
    } else {
        STEEL_ATTN_DEFAULT_CFG
    }
}

pub fn steel_gemm_cfg_for(rows: usize, n: usize) -> (u32, u32, u32, u32, u32) {
    let _ = VISENC_CROP_SHAPES_STARVE_THE_GRID_SO_THE_CHAMPION_TILE_IS_PER_SHAPE;
    match rows {
        0..=128 => {
            if n >= 4608 {
                (32, 32, 16, 2, 2)
            } else {
                (16, 32, 16, 2, 2)
            }
        }
        129..=320 => {
            if n >= 4608 {
                (32, 64, 16, 2, 2)
            } else if n >= 4224 {
                (32, 32, 16, 2, 2)
            } else {
                (16, 32, 16, 2, 2)
            }
        }
        321..=530 => {
            if n >= 4608 {
                (32, 64, 16, 2, 2)
            } else {
                (32, 32, 16, 2, 2)
            }
        }
        531..=690 => {
            if n <= 1536 {
                (32, 64, 16, 2, 2)
            } else {
                STEEL_GEMM_FUSED_CFG_IS_THE_FITTED_CHAMPION_ONLY_BECAUSE_EVERY_CFG_IS_BIT_IDENTICAL_AND_ONE_INSTANTIATION_KEEPS_THE_MODULE_LEAN
            }
        }
        _ => STEEL_GEMM_FUSED_CFG_IS_THE_FITTED_CHAMPION_ONLY_BECAUSE_EVERY_CFG_IS_BIT_IDENTICAL_AND_ONE_INSTANTIATION_KEEPS_THE_MODULE_LEAN,
    }
}

pub const STEEL_GEMM_RIDES_THE_SAME_DRIVER_PATTERNS_THE_ATTENTION_PORT_PROVED: &str =
    "MLX's own tower cast to f16 reads 4.081s against their shipped bf16 5.310s, which prices \
     our coop+splitk projections (1.85s of the tower by deleted work) about 0.4-0.5s behind \
     steel-gemm-at-f16. GEMMKernel::run in the vendored gemm.h does the whole job -- swizzle, \
     block mapping, ragged M/N via the safe loop, K tail -- so the driver kernel is one template \
     per transpose pattern instantiated at the ladder's configs with U=float so the output lands \
     in the f32 buffers every consumer already reads. The nt pattern is the projection layout \
     as-is: A the k_pad-strided f16 activations, B the k_pad-strided f16 weight rows. M is the \
     REAL row count, not the padded one -- steel guards its loads, so the coop kernel's \
     clamped-overshoot slack is never touched and padded rows are never computed. The bias pass \
     and every weight format but F16 stay on the incumbent path.";

const STEEL_GEMM_FILES: &[&str] = &[
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/defines.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/utils/type_traits.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/utils/integral_constant.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/utils.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/gemm/params.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/gemm/transforms.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/gemm/loader.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/gemm/mma.h"),
    include_str!("../../../../third-party/mlx-steel-attn/mlx/backend/metal/kernels/steel/gemm/gemm.h"),
];

pub fn steel_gemm_entry(bm: u32, bn: u32, bk: u32, wm: u32, wn: u32) -> String {
    format!("dv_steel_gemm_nt_{bm}_{bn}_{bk}_{wm}_{wn}")
}

pub const STEEL_GEMM_SPLITK_LIVES_IN_GRID_Z_BECAUSE_SLICE_DISPATCHES_DIE_ON_THE_SERIAL_DRAIN: &str =
    "The m=32 trunk families run 4.6x over their summed byte floors in-step (full 112-dispatch \
     chain 14.78 ms vs a 0.85 ms tile-machinery floor and a 0.57 ms trivial-launch floor), so \
     the wall is the k-walk inside starved 24-64-workgroup grids, not launch, drain, or bytes. \
     Splitting k into S separate dispatches recovers nothing -- Metal serial dispatch drains \
     each slice before the next launches, and the bracket read every such arm at or above the \
     single-dispatch time -- so the split lives in grid z of ONE dispatch: the wrapper offsets \
     A/B/D by tid.z times the batch strides GEMMParams already carries (k_slice, k_slice, m*n), \
     every slice runs concurrently, and one ascending-order reduce sums the S partials. The \
     vendored GEMMKernel::run never reads tid.z or the batch strides, so the wrapper adds \
     offsets only; instantiations stay on the narrow-class tiles where the starvation lives. \
     Partial sums reassociate the k-chain, so every splitk arm is tolerance-class: census \
     before serving, opt-in env, never a silent default.";

pub const STEEL_GEMM_COMPLEX_SHIM_EXISTS_SO_THE_NEVER_INSTANTIATED_COMPLEX_SPECIALIZATIONS_PARSE:
    &str = "
#include <metal_stdlib>
struct complex64_t {
  float real;
  float imag;
  complex64_t() thread = default;
  complex64_t(float r, float i) thread : real(r), imag(i) {}
};
inline complex64_t operator+(complex64_t a, complex64_t b) {
  return complex64_t(a.real + b.real, a.imag + b.imag);
}
inline complex64_t operator*(complex64_t a, complex64_t b) {
  return complex64_t(
      a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real);
}
";

fn steel_gemm_headers() -> String {
    let mut b = String::from(
        STEEL_GEMM_COMPLEX_SHIM_EXISTS_SO_THE_NEVER_INSTANTIATED_COMPLEX_SPECIALIZATIONS_PARSE,
    );
    for f in STEEL_GEMM_FILES {
        for line in f.lines() {
            let t = line.trim_start();
            if t.starts_with("#include \"mlx/") || t == "#pragma once" {
                continue;
            }
            b.push_str(line);
            b.push('\n');
        }
    }
    b
}

pub fn steel_gemm_source() -> String {
    let _ = STEEL_GEMM_RIDES_THE_SAME_DRIVER_PATTERNS_THE_ATTENTION_PORT_PROVED;
    let mut b = steel_gemm_headers();
    b.push_str(
        "
using namespace mlx::steel;

template <typename T, typename U, int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void dv_steel_gemm_nt(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device U* D [[buffer(2)]],
    const constant GEMMParams* params [[buffer(3)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  using gemm_kernel =
      GEMMKernel<T, U, BM, BN, BK, WM, WN, false, true, false, true, float>;
  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];
  gemm_kernel::run(
      A, B, D, params, As, Bs, simd_lane_id, simd_group_id, tid, lid);
}
",
    );
    for (bm, bn, bk, wm, wn) in STEEL_GEMM_CONFIGS {
        b.push_str(&format!(
            "template [[host_name(\"{}\")]] [[kernel]] decltype(dv_steel_gemm_nt<half, float, {bm}, {bn}, {bk}, {wm}, {wn}>) dv_steel_gemm_nt<half, float, {bm}, {bn}, {bk}, {wm}, {wn}>;\n",
            steel_gemm_entry(*bm, *bn, *bk, *wm, *wn)
        ));
    }
    b
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SteelGemmParams {
    pub m: i32,
    pub n: i32,
    pub k: i32,
    pub lda: i32,
    pub ldb: i32,
    pub ldd: i32,
    pub tiles_n: i32,
    pub tiles_m: i32,
    pub batch_stride_a: i64,
    pub batch_stride_b: i64,
    pub batch_stride_d: i64,
    pub swizzle_log: i32,
    pub gemm_k_iterations_aligned: i32,
    pub batch_ndim: i32,
    pub pad_tail: i32,
}

const _: () = assert!(std::mem::size_of::<SteelGemmParams>() == 72);

pub struct SteelGemmPipelines {
    pub by_config: Vec<((u32, u32, u32, u32, u32), Arc<wgpu::ComputePipeline>)>,
    pub layout: wgpu::BindGroupLayout,
}

impl SteelGemmPipelines {
    pub fn pick(&self, cfg: (u32, u32, u32, u32, u32)) -> Option<Arc<wgpu::ComputePipeline>> {
        self.by_config
            .iter()
            .find(|(c, _)| *c == cfg)
            .map(|(_, p)| p.clone())
    }
}

type SteelGemmCache = Mutex<
    Vec<(
        wgpu::Device,
        std::result::Result<Arc<SteelGemmPipelines>, String>,
    )>,
>;

fn steel_gemm_cache() -> &'static SteelGemmCache {
    static CACHE: OnceLock<SteelGemmCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn steel_gemm_build(ctx: &WgpuContext) -> Result<Arc<SteelGemmPipelines>> {
    let entries_meta: Vec<(&'static str, (u32, u32, u32))> = STEEL_GEMM_CONFIGS
        .iter()
        .map(|(bm, bn, bk, wm, wn)| {
            let e: &'static str =
                Box::leak(steel_gemm_entry(*bm, *bn, *bk, *wm, *wn).into_boxed_str());
            (e, (wm * wn * 32, 1, 1))
        })
        .collect();
    let src: &'static str = Box::leak(steel_gemm_source().into_boxed_str());
    let module = na::msl_module(ctx, "dv-steel-gemm", &entries_meta, src)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let entries = [
        storage_entry(0, true),
        storage_entry(1, true),
        storage_entry(2, false),
        uniform_entry(3),
    ];
    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dv-steel-gemm"),
            entries: &entries,
        });
    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dv-steel-gemm"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let mut by_config = Vec::new();
    for (cfg, (name, _)) in STEEL_GEMM_CONFIGS.iter().zip(entries_meta.iter()) {
        let pipeline = Arc::new(ctx.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            },
        ));
        by_config.push((*cfg, pipeline));
    }
    if let Some(err) = pollster::block_on(scope.pop()) {
        bail!("dv-steel-gemm pipelines: {err}");
    }
    Ok(Arc::new(SteelGemmPipelines {
        by_config,
        layout: bgl,
    }))
}

pub fn steel_gemm_pipelines(ctx: &'static WgpuContext) -> Result<Arc<SteelGemmPipelines>> {
    let mut guard = steel_gemm_cache().lock().unwrap();
    for (dev, entry) in guard.iter() {
        if *dev == ctx.device {
            return entry.clone().map_err(|e| anyhow::anyhow!("{e}"));
        }
    }
    let built = steel_gemm_build(ctx);
    let stored = match &built {
        Ok(p) => Ok(p.clone()),
        Err(e) => Err(format!("{e:#}")),
    };
    guard.push((ctx.device.clone(), stored));
    built
}

pub fn steel_gemm_params(
    m: usize,
    n: usize,
    k_pad: usize,
    ldd: usize,
    cfg: (u32, u32, u32, u32, u32),
) -> SteelGemmParams {
    let (bm, bn, bk, _, _) = cfg;
    SteelGemmParams {
        m: m as i32,
        n: n as i32,
        k: k_pad as i32,
        lda: k_pad as i32,
        ldb: k_pad as i32,
        ldd: ldd as i32,
        tiles_n: n.div_ceil(bn as usize) as i32,
        tiles_m: m.div_ceil(bm as usize) as i32,
        batch_stride_a: 0,
        batch_stride_b: 0,
        batch_stride_d: 0,
        swizzle_log: 0,
        gemm_k_iterations_aligned: (k_pad / bk as usize) as i32,
        batch_ndim: 1,
        pad_tail: 0,
    }
}

pub fn steel_gemm_grid(m: usize, n: usize, cfg: (u32, u32, u32, u32, u32)) -> (u32, u32, u32) {
    (
        n.div_ceil(cfg.1 as usize) as u32,
        m.div_ceil(cfg.0 as usize) as u32,
        1,
    )
}

pub fn steel_gemm_splitk_params(
    m: usize,
    n: usize,
    k_full: usize,
    slices: usize,
    cfg: (u32, u32, u32, u32, u32),
) -> SteelGemmParams {
    let _ = STEEL_GEMM_SPLITK_LIVES_IN_GRID_Z_BECAUSE_SLICE_DISPATCHES_DIE_ON_THE_SERIAL_DRAIN;
    let k_slice = k_full / slices;
    assert!(k_slice * slices == k_full && k_slice.is_multiple_of(cfg.2 as usize));
    let mut p = steel_gemm_params(m, n, k_slice, n, cfg);
    p.lda = k_full as i32;
    p.ldb = k_full as i32;
    p.batch_stride_a = k_slice as i64;
    p.batch_stride_b = k_slice as i64;
    p.batch_stride_d = (m * n) as i64;
    p
}

pub fn steel_gemm_splitk_grid(
    m: usize,
    n: usize,
    slices: usize,
    cfg: (u32, u32, u32, u32, u32),
) -> (u32, u32, u32) {
    let (x, y, _) = steel_gemm_grid(m, n, cfg);
    (x, y, slices as u32)
}

pub const STEEL_GEMM_FUSED_CFG_IS_THE_FITTED_CHAMPION_ONLY_BECAUSE_EVERY_CFG_IS_BIT_IDENTICAL_AND_ONE_INSTANTIATION_KEEPS_THE_MODULE_LEAN:
    (u32, u32, u32, u32, u32) = (64, 64, 32, 2, 2);

pub const STEEL_GEMM_FUSED_EPILOGUES_KEEP_THE_MMA_CHAIN_AND_REPLAY_THE_CRUMB_KERNELS_ROUNDING_VERBATIM: &str =
    "The vision crumb passes (residual add, silu-mul) are pure elementwise consumers of the steel \
     GEMM's f32 accumulators: the shipped chain stores the accumulator bits untransformed \
     (TransformNone f32->f32 is the identity) and the WGSL crumb re-reads exactly those bits, so \
     an epilogue that applies the same f32 expression to the live accumulator before the one \
     store reproduces the shipped rounding order by construction. The MMA reduction itself never \
     moves between kernels -- tile_matmad is a fixed ascending-k chain of simdgroup intrinsics \
     the compiler cannot reassociate, measured bit-identical across every cfg and even across \
     the WGSL coop implementation (0/5,222,400) -- so the simd_sum law's poison (a per-kernel \
     reduction order) has no surface here. resadd replays dv_add_f32: acc_new = x_old + acc, \
     x_old read first as the WGSL left operand; gsilu replays dv_silu_mul_f16: \
     g / (1.0 + exp(-g)) * u with u the live fc3 accumulator and g the fc1 output read from \
     memory, rounded to f16 once at the store exactly where the WGSL f16() cast sits. The \
     epilogue walkers guard rows AND columns on ragged tiles (the vendored apply_epilogue_safe \
     guards only columns and would read past the last real row of the operand buffer). The \
     kernels live in this wrapper, not in the vendored tree, and compile as a separate module so \
     a fused-source failure can never take down the shipped steel route.";

fn steel_gemm_fused_body() -> &'static str {
    "
using namespace mlx::steel;

template <typename mma_t>
METAL_FUNC void dv_epilogue_resadd(
    thread mma_t& mma_op,
    const device float* C,
    const int ldc,
    short2 dst_tile_dims,
    bool bounded) {
  C += mma_op.sm * ldc + mma_op.sn;
  dst_tile_dims -= short2(mma_op.sn, mma_op.sm);
  if (bounded && (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)) {
    return;
  }
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < mma_t::TM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < mma_t::TN; j++) {
      thread auto& accum = mma_op.Ctile.frag_at(i, j);
      const int offset_c = (i * mma_t::TM_stride) * ldc + (j * mma_t::TN_stride);
      constexpr short kelems = decltype(mma_op.Ctile)::kElemsPerFrag;
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < kelems; k++) {
        if (!bounded ||
            ((i * mma_t::TM_stride) < dst_tile_dims.y &&
             (j * mma_t::TN_stride + k) < dst_tile_dims.x)) {
          accum[k] = C[offset_c + k] + accum[k];
        }
      }
    }
  }
}

template <typename mma_t>
METAL_FUNC void dv_epilogue_gate_silu(
    thread mma_t& mma_op,
    const device float* G,
    const int ldg,
    short2 dst_tile_dims,
    bool bounded) {
  G += mma_op.sm * ldg + mma_op.sn;
  dst_tile_dims -= short2(mma_op.sn, mma_op.sm);
  if (bounded && (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)) {
    return;
  }
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < mma_t::TM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < mma_t::TN; j++) {
      thread auto& accum = mma_op.Ctile.frag_at(i, j);
      const int offset_g = (i * mma_t::TM_stride) * ldg + (j * mma_t::TN_stride);
      constexpr short kelems = decltype(mma_op.Ctile)::kElemsPerFrag;
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < kelems; k++) {
        if (!bounded ||
            ((i * mma_t::TM_stride) < dst_tile_dims.y &&
             (j * mma_t::TN_stride + k) < dst_tile_dims.x)) {
          float gv = G[offset_g + k];
          accum[k] = gv / (1.0f + metal::exp(-gv)) * accum[k];
        }
      }
    }
  }
}

template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void dv_steel_gemm_nt_resadd(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device float* D [[buffer(2)]],
    const constant GEMMParams* params [[buffer(3)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  using gemm_kernel =
      GEMMKernel<T, float, BM, BN, BK, WM, WN, false, true, false, true, float>;
  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];
  (void)lid;
  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;
  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }
  threadgroup_barrier(mem_flags::mem_none);
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);
  const device T* A_ptr = A + c_row_long * params->lda;
  const device T* B_ptr = B + c_col_long * params->ldb;
  device float* D_ptr = D + c_row_long * params->ldd + c_col_long;
  thread typename gemm_kernel::loader_a_t loader_a(
      A_ptr, params->lda, As, simd_group_id, simd_lane_id);
  thread typename gemm_kernel::loader_b_t loader_b(
      B_ptr, params->ldb, Bs, simd_group_id, simd_lane_id);
  thread typename gemm_kernel::mma_t mma_op(simd_group_id, simd_lane_id);
  const int gemm_k_iterations = params->gemm_k_iterations_aligned;
  short tgp_bm = min(BM, params->M - c_row);
  short tgp_bn = min(BN, params->N - c_col);
  short lbk = 0;
  if (tgp_bm == BM && tgp_bn == BN) {
    gemm_kernel::template gemm_loop<true, true, true>(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
    dv_epilogue_resadd(mma_op, D_ptr, params->ldd, short2(tgp_bn, tgp_bm), false);
    mma_op.store_result(D_ptr, params->ldd);
  } else if (tgp_bn == BN) {
    gemm_kernel::template gemm_loop<false, true, true>(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
    dv_epilogue_resadd(mma_op, D_ptr, params->ldd, short2(tgp_bn, tgp_bm), true);
    mma_op.store_result_safe(D_ptr, params->ldd, short2(tgp_bn, tgp_bm));
  } else if (tgp_bm == BM) {
    gemm_kernel::template gemm_loop<true, false, true>(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
    dv_epilogue_resadd(mma_op, D_ptr, params->ldd, short2(tgp_bn, tgp_bm), true);
    mma_op.store_result_safe(D_ptr, params->ldd, short2(tgp_bn, tgp_bm));
  } else {
    gemm_kernel::template gemm_loop<false, false, true>(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
    dv_epilogue_resadd(mma_op, D_ptr, params->ldd, short2(tgp_bn, tgp_bm), true);
    mma_op.store_result_safe(D_ptr, params->ldd, short2(tgp_bn, tgp_bm));
  }
}

template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void dv_steel_gemm_nt_gsilu(
    const device T* A [[buffer(0)]],
    const device T* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    const constant GEMMParams* params [[buffer(3)]],
    const device float* G [[buffer(4)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  using gemm_kernel =
      GEMMKernel<T, half, BM, BN, BK, WM, WN, false, true, false, true, float>;
  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];
  (void)lid;
  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;
  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }
  threadgroup_barrier(mem_flags::mem_none);
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);
  const device T* A_ptr = A + c_row_long * params->lda;
  const device T* B_ptr = B + c_col_long * params->ldb;
  device half* D_ptr = D + c_row_long * params->ldd + c_col_long;
  const device float* G_ptr = G + c_row_long * params->ldd + c_col_long;
  thread typename gemm_kernel::loader_a_t loader_a(
      A_ptr, params->lda, As, simd_group_id, simd_lane_id);
  thread typename gemm_kernel::loader_b_t loader_b(
      B_ptr, params->ldb, Bs, simd_group_id, simd_lane_id);
  thread typename gemm_kernel::mma_t mma_op(simd_group_id, simd_lane_id);
  const int gemm_k_iterations = params->gemm_k_iterations_aligned;
  short tgp_bm = min(BM, params->M - c_row);
  short tgp_bn = min(BN, params->N - c_col);
  short lbk = 0;
  if (tgp_bm == BM && tgp_bn == BN) {
    gemm_kernel::template gemm_loop<true, true, true>(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
    dv_epilogue_gate_silu(mma_op, G_ptr, params->ldd, short2(tgp_bn, tgp_bm), false);
    mma_op.store_result(D_ptr, params->ldd);
  } else if (tgp_bn == BN) {
    gemm_kernel::template gemm_loop<false, true, true>(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
    dv_epilogue_gate_silu(mma_op, G_ptr, params->ldd, short2(tgp_bn, tgp_bm), true);
    mma_op.store_result_safe(D_ptr, params->ldd, short2(tgp_bn, tgp_bm));
  } else if (tgp_bm == BM) {
    gemm_kernel::template gemm_loop<true, false, true>(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
    dv_epilogue_gate_silu(mma_op, G_ptr, params->ldd, short2(tgp_bn, tgp_bm), true);
    mma_op.store_result_safe(D_ptr, params->ldd, short2(tgp_bn, tgp_bm));
  } else {
    gemm_kernel::template gemm_loop<false, false, true>(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
    dv_epilogue_gate_silu(mma_op, G_ptr, params->ldd, short2(tgp_bn, tgp_bm), true);
    mma_op.store_result_safe(D_ptr, params->ldd, short2(tgp_bn, tgp_bm));
  }
}
"
}

pub fn steel_gemm_fused_entry(kind: &str, cfg: (u32, u32, u32, u32, u32)) -> String {
    let (bm, bn, bk, wm, wn) = cfg;
    format!("dv_steel_gemm_nt_{kind}_{bm}_{bn}_{bk}_{wm}_{wn}")
}

pub const STEEL_GEMM_FUSED_CFGS_COVER_THE_CROP_MAP_SO_PER_SHAPE_SELECTION_NEVER_TRADES_AWAY_THE_EPILOGUES:
    &[(u32, u32, u32, u32, u32)] = &[
    STEEL_GEMM_FUSED_CFG_IS_THE_FITTED_CHAMPION_ONLY_BECAUSE_EVERY_CFG_IS_BIT_IDENTICAL_AND_ONE_INSTANTIATION_KEEPS_THE_MODULE_LEAN,
    (16, 32, 16, 2, 2),
    (32, 32, 16, 2, 2),
    (32, 64, 16, 2, 2),
];

pub fn steel_gemm_fused_source() -> String {
    let _ = STEEL_GEMM_FUSED_EPILOGUES_KEEP_THE_MMA_CHAIN_AND_REPLAY_THE_CRUMB_KERNELS_ROUNDING_VERBATIM;
    let mut b = steel_gemm_headers();
    b.push_str(steel_gemm_fused_body());
    for cfg in STEEL_GEMM_FUSED_CFGS_COVER_THE_CROP_MAP_SO_PER_SHAPE_SELECTION_NEVER_TRADES_AWAY_THE_EPILOGUES {
        let (bm, bn, bk, wm, wn) = *cfg;
        for kind in ["resadd", "gsilu"] {
            b.push_str(&format!(
                "template [[host_name(\"{}\")]] [[kernel]] decltype(dv_steel_gemm_nt_{kind}<half, {bm}, {bn}, {bk}, {wm}, {wn}>) dv_steel_gemm_nt_{kind}<half, {bm}, {bn}, {bk}, {wm}, {wn}>;\n",
                steel_gemm_fused_entry(kind, *cfg)
            ));
        }
    }
    b
}

pub struct SteelGemmFusedPipelines {
    pub resadd_layout: wgpu::BindGroupLayout,
    pub gsilu_layout: wgpu::BindGroupLayout,
    pub by_config: Vec<(
        (u32, u32, u32, u32, u32),
        Arc<wgpu::ComputePipeline>,
        Arc<wgpu::ComputePipeline>,
    )>,
}

impl SteelGemmFusedPipelines {
    pub fn pick_resadd(
        &self,
        cfg: (u32, u32, u32, u32, u32),
    ) -> Option<Arc<wgpu::ComputePipeline>> {
        self.by_config
            .iter()
            .find(|(c, _, _)| *c == cfg)
            .map(|(_, r, _)| r.clone())
    }

    pub fn pick_gsilu(&self, cfg: (u32, u32, u32, u32, u32)) -> Option<Arc<wgpu::ComputePipeline>> {
        self.by_config
            .iter()
            .find(|(c, _, _)| *c == cfg)
            .map(|(_, _, g)| g.clone())
    }
}

type SteelGemmFusedCache = Mutex<
    Vec<(
        wgpu::Device,
        std::result::Result<Arc<SteelGemmFusedPipelines>, String>,
    )>,
>;

fn steel_gemm_fused_cache() -> &'static SteelGemmFusedCache {
    static CACHE: OnceLock<SteelGemmFusedCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn steel_gemm_fused_build(ctx: &WgpuContext) -> Result<Arc<SteelGemmFusedPipelines>> {
    let cfgs = STEEL_GEMM_FUSED_CFGS_COVER_THE_CROP_MAP_SO_PER_SHAPE_SELECTION_NEVER_TRADES_AWAY_THE_EPILOGUES;
    let entries_meta: Vec<(&'static str, (u32, u32, u32))> = cfgs
        .iter()
        .flat_map(|cfg| {
            ["resadd", "gsilu"].map(|kind| {
                let e: &'static str =
                    Box::leak(steel_gemm_fused_entry(kind, *cfg).into_boxed_str());
                (e, (cfg.3 * cfg.4 * 32, 1, 1))
            })
        })
        .collect();
    let src: &'static str = Box::leak(steel_gemm_fused_source().into_boxed_str());
    let module = na::msl_module(ctx, "dv-steel-gemm-fused", &entries_meta, src)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mk_layout = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            });
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        (bgl, layout)
    };
    let (resadd_layout, resadd_pl) = mk_layout(
        "dv-steel-gemm-resadd",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    );
    let (gsilu_layout, gsilu_pl) = mk_layout(
        "dv-steel-gemm-gsilu",
        &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
            storage_entry(4, true),
        ],
    );
    let mk_pipeline = |pl: &wgpu::PipelineLayout, entry: &'static str| {
        Arc::new(
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(pl),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        )
    };
    let mut by_config = Vec::new();
    for (cfg, pair) in cfgs.iter().zip(entries_meta.chunks(2)) {
        let r = mk_pipeline(&resadd_pl, pair[0].0);
        let g = mk_pipeline(&gsilu_pl, pair[1].0);
        by_config.push((*cfg, r, g));
    }
    if let Some(err) = pollster::block_on(scope.pop()) {
        bail!("dv-steel-gemm-fused pipelines: {err}");
    }
    Ok(Arc::new(SteelGemmFusedPipelines {
        resadd_layout,
        gsilu_layout,
        by_config,
    }))
}

pub fn steel_gemm_fused_pipelines(
    ctx: &'static WgpuContext,
) -> Result<Arc<SteelGemmFusedPipelines>> {
    let mut guard = steel_gemm_fused_cache().lock().unwrap();
    for (dev, entry) in guard.iter() {
        if *dev == ctx.device {
            return entry.clone().map_err(|e| anyhow::anyhow!("{e}"));
        }
    }
    let built = steel_gemm_fused_build(ctx);
    let stored = match &built {
        Ok(p) => Ok(p.clone()),
        Err(e) => Err(format!("{e:#}")),
    };
    guard.push((ctx.device.clone(), stored));
    built
}

pub const STEEL_GEMM_CVT_FUSIONS_ARE_BITWISE_BECAUSE_BOTH_CONVERSIONS_ARE_ELEMENTWISE_AND_THE_MMA_CHAIN_NEVER_MOVES: &str =
    "The decode trunk's mirror passes are pure elementwise conversions around the steel GEMMs: \
     dg_scvt_q re-rounds nothing (bf16 bits shifted to f32 exactly, then one f32->f16 RNE at \
     pack2x16float), and dg_cvt is one f32->bf16e RNE per element on the stored f32 accumulator \
     identity. The abf16 loader replays dg_scvt_q's conversion at As-staging time, so the MMA \
     consumes half values bit-identical to the mirror it replaces; the bf16o store walker \
     replays dg_cvt's bf16e word-pack on the live accumulator pair exactly where store_result \
     wrote the f32 identity. Every 8x8 MMA fragment owns an even-aligned column pair \
     (kElemsPerFrag == 2, sn always even), so each fragment is exactly one packed word, and \
     n stays even at every trunk shape by assertion. The replicated dv_gemm_loop_abf16 copies \
     the vendored gemm_loop's K-aligned body verbatim -- the reduction lives in mma_op.mma, \
     untouched. Kernels live in this wrapper as a separate module so a fused-source failure \
     can never take down the shipped steel route.";

pub const STEEL_GEMM_CVT_CFGS_ARE_THE_TRUNK_MAP_TILES: &[(u32, u32, u32, u32, u32)] =
    &[(32, 32, 16, 2, 2), (64, 64, 32, 2, 2)];

pub const STEEL_GEMM_CVT_SK_CFG_IS_THE_ONLY_TILE_SPLITK_ENGAGES: (u32, u32, u32, u32, u32) =
    (32, 32, 16, 2, 2);

pub const STEEL_GEMM_CVT_CFGS_ARE_A_SUBSET_OF_THE_COMPILE_ASSERTED_STEEL_GEMM_CONFIGS: () = {
    let mut i = 0;
    while i < STEEL_GEMM_CVT_CFGS_ARE_THE_TRUNK_MAP_TILES.len() {
        let want = STEEL_GEMM_CVT_CFGS_ARE_THE_TRUNK_MAP_TILES[i];
        let mut j = 0;
        let mut found = false;
        while j < STEEL_GEMM_CONFIGS.len() {
            let have = STEEL_GEMM_CONFIGS[j];
            if want.0 == have.0
                && want.1 == have.1
                && want.2 == have.2
                && want.3 == have.3
                && want.4 == have.4
            {
                found = true;
            }
            j += 1;
        }
        assert!(found);
        let (_, _, _, wm, wn) = want;
        let tgp = wm * wn * 32;
        let a_elems = want.0 * want.2;
        assert!(a_elems.is_multiple_of(tgp) && (a_elems / tgp).is_multiple_of(2));
        i += 1;
    }
};

pub fn steel_gemm_cvt_entry(kind: &str, cfg: (u32, u32, u32, u32, u32)) -> String {
    let (bm, bn, bk, wm, wn) = cfg;
    format!("dv_steel_gemm_nt_{kind}_{bm}_{bn}_{bk}_{wm}_{wn}")
}

fn steel_gemm_cvt_body() -> &'static str {
    "
using namespace mlx::steel;

METAL_FUNC uint dv_bf16e(float x) {
  uint b = as_type<uint>(x);
  uint r = 0x7fffu + ((b >> 16) & 1u);
  return (x != x) ? 0x7fc0u : ((b + r) >> 16);
}

METAL_FUNC uint dv_bf16w(float lo, float hi) {
  return (dv_bf16e(lo) & 0xffffu) | (dv_bf16e(hi) << 16);
}

METAL_FUNC half dv_h_of_bf16lo(uint w) {
  return half(as_type<float>((w & 0xffffu) << 16));
}

METAL_FUNC half dv_h_of_bf16hi(uint w) {
  return half(as_type<float>(w & 0xffff0000u));
}

template <
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short n_reads = (BCOLS * BROWS) / (tgp_size),
    short TCOLS = BCOLS / n_reads,
    short TROWS = tgp_size / TCOLS>
struct DvBf16BlockLoader {
  static_assert(n_reads % 2 == 0, \"bf16 words pack element pairs\");
  STEEL_CONST short n_rows = (BROWS + TROWS - 1) / TROWS;
  STEEL_CONST short vec_size = n_reads;
  const int src_ld;
  const int tile_stride;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup half* dst;
  const device uint* src;
  int elem_off;

  METAL_FUNC DvBf16BlockLoader(
      const device uint* src_words,
      const int elem_base,
      const int src_ld_,
      threadgroup half* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) thread
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld_),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(thread_idx / TCOLS),
        bj(vec_size * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_words),
        elem_off(elem_base + bi * src_ld_ + bj) {}

  METAL_FUNC void load_unsafe() const thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      const int wo = (elem_off + i * src_ld) >> 1;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j += 2) {
        const uint w = src[wo + (j >> 1)];
        dst[i * dst_ld + j] = dv_h_of_bf16lo(w);
        dst[i * dst_ld + j + 1] = dv_h_of_bf16hi(w);
      }
    }
  }

  METAL_FUNC void load_safe(short2 src_tile_dim) const thread {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < BROWS; i += TROWS) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = half(0);
        }
      }
      return;
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BROWS; i += TROWS) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < vec_size; j++) {
        const bool ok = (i < src_tile_dim.y) && (j < src_tile_dim.x);
        const int idx = ok ? (i * src_ld + j) : 0;
        const uint w = src[(elem_off + idx) >> 1];
        const half v = ((idx & 1) == 0) ? dv_h_of_bf16lo(w) : dv_h_of_bf16hi(w);
        dst[i * dst_ld + j] = ok ? v : half(0);
      }
    }
  }

  METAL_FUNC void next() thread {
    elem_off += tile_stride;
  }
};

template <typename mma_t>
METAL_FUNC void dv_store_bf16(
    thread mma_t& mma_op,
    device uint* Dw,
    const int ldd,
    short2 dst_tile_dims,
    bool bounded) {
  static_assert(
      decltype(mma_op.Ctile)::kElemsPerFrag == 2,
      \"the bf16 word store rides the 2-element even-aligned fragment\");
  const int ldw = ldd >> 1;
  dst_tile_dims -= short2(mma_op.sn, mma_op.sm);
  if (bounded && (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)) {
    return;
  }
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < mma_t::TM; i++) {
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < mma_t::TN; j++) {
      thread auto& accum = mma_op.Ctile.frag_at(i, j);
      if (!bounded ||
          ((i * mma_t::TM_stride) < dst_tile_dims.y &&
           (j * mma_t::TN_stride + 1) < dst_tile_dims.x)) {
        Dw[(mma_op.sm + i * mma_t::TM_stride) * ldw +
           ((mma_op.sn + j * mma_t::TN_stride) >> 1)] =
            dv_bf16w(float(accum[0]), float(accum[1]));
      }
    }
  }
}

template <typename loader_a_t2, typename loader_b_t, typename mma_t, bool M_aligned, bool N_aligned>
METAL_FUNC void dv_gemm_loop_abf16(
    threadgroup half* As,
    threadgroup half* Bs,
    const int gemm_k_iterations,
    thread loader_a_t2& loader_a,
    thread loader_b_t& loader_b,
    thread mma_t& mma_op,
    short2 tile_dims_A,
    short2 tile_dims_B) {
  for (int k = 0; k < gemm_k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (M_aligned) {
      loader_a.load_unsafe();
    } else {
      loader_a.load_safe(tile_dims_A);
    }
    if (N_aligned) {
      loader_b.load_unsafe();
    } else {
      loader_b.load_safe(tile_dims_B);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    loader_a.next();
    loader_b.next();
  }
}

template <int BM, int BN, int BK, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void dv_steel_gemm_nt_bf16o(
    const device half* A [[buffer(0)]],
    const device half* B [[buffer(1)]],
    device uint* D [[buffer(2)]],
    const constant GEMMParams* params [[buffer(3)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  using gemm_kernel =
      GEMMKernel<half, float, BM, BN, BK, WM, WN, false, true, false, true, float>;
  threadgroup half As[gemm_kernel::tgp_mem_size_a];
  threadgroup half Bs[gemm_kernel::tgp_mem_size_b];
  (void)lid;
  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;
  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }
  threadgroup_barrier(mem_flags::mem_none);
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);
  const device half* A_ptr = A + c_row_long * params->lda;
  const device half* B_ptr = B + c_col_long * params->ldb;
  device uint* D_ptr = D + c_row_long * (params->ldd >> 1) + (c_col_long >> 1);
  thread typename gemm_kernel::loader_a_t loader_a(
      A_ptr, params->lda, As, simd_group_id, simd_lane_id);
  thread typename gemm_kernel::loader_b_t loader_b(
      B_ptr, params->ldb, Bs, simd_group_id, simd_lane_id);
  thread typename gemm_kernel::mma_t mma_op(simd_group_id, simd_lane_id);
  const int gemm_k_iterations = params->gemm_k_iterations_aligned;
  short tgp_bm = min(BM, params->M - c_row);
  short tgp_bn = min(BN, params->N - c_col);
  short lbk = 0;
  if (tgp_bm == BM && tgp_bn == BN) {
    gemm_kernel::template gemm_loop<true, true, true>(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
    dv_store_bf16(mma_op, D_ptr, params->ldd, short2(tgp_bn, tgp_bm), false);
  } else if (tgp_bn == BN) {
    gemm_kernel::template gemm_loop<false, true, true>(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
    dv_store_bf16(mma_op, D_ptr, params->ldd, short2(tgp_bn, tgp_bm), true);
  } else if (tgp_bm == BM) {
    gemm_kernel::template gemm_loop<true, false, true>(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
    dv_store_bf16(mma_op, D_ptr, params->ldd, short2(tgp_bn, tgp_bm), true);
  } else {
    gemm_kernel::template gemm_loop<false, false, true>(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tgp_bm, tgp_bn, lbk);
    dv_store_bf16(mma_op, D_ptr, params->ldd, short2(tgp_bn, tgp_bm), true);
  }
}

template <int BM, int BN, int BK, int WM, int WN, bool BF16O, bool SPLITK>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void dv_steel_gemm_nt_abf16(
    const device uint* A [[buffer(0)]],
    const device half* B [[buffer(1)]],
    device uint* D [[buffer(2)]],
    const constant GEMMParams* params [[buffer(3)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  using gemm_kernel =
      GEMMKernel<half, float, BM, BN, BK, WM, WN, false, true, false, true, float>;
  using loader_a_t2 = DvBf16BlockLoader<
      BM,
      BK,
      BK + gemm_kernel::tgp_padding_a,
      1,
      gemm_kernel::tgp_size>;
  threadgroup half As[gemm_kernel::tgp_mem_size_a];
  threadgroup half Bs[gemm_kernel::tgp_mem_size_b];
  (void)lid;
  const int tid_y = ((tid.y) << params->swizzle_log) +
      ((tid.x) & ((1 << params->swizzle_log) - 1));
  const int tid_x = (tid.x) >> params->swizzle_log;
  if (params->tiles_n <= tid_x || params->tiles_m <= tid_y) {
    return;
  }
  threadgroup_barrier(mem_flags::mem_none);
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_col_long = size_t(c_col);
  const int a_elem_base = c_row * params->lda +
      (SPLITK ? int(tid.z * params->batch_stride_a) : 0);
  const device half* B_ptr = B + c_col_long * params->ldb +
      (SPLITK ? size_t(tid.z * params->batch_stride_b) : size_t(0));
  device uint* D_ptr =
      D + size_t(c_row) * size_t(params->ldd >> 1) + (c_col_long >> 1);
  device float* Df_ptr = ((device float*)D) + size_t(c_row) * size_t(params->ldd) +
      c_col_long + (SPLITK ? size_t(tid.z * params->batch_stride_d) : size_t(0));
  thread loader_a_t2 loader_a(
      A, a_elem_base, params->lda, As, simd_group_id, simd_lane_id);
  thread typename gemm_kernel::loader_b_t loader_b(
      B_ptr, params->ldb, Bs, simd_group_id, simd_lane_id);
  thread typename gemm_kernel::mma_t mma_op(simd_group_id, simd_lane_id);
  const int gemm_k_iterations = params->gemm_k_iterations_aligned;
  short tgp_bm = min(BM, params->M - c_row);
  short tgp_bn = min(BN, params->N - c_col);
  const short2 tile_dims_A = short2(BK, tgp_bm);
  const short2 tile_dims_B = short2(tgp_bn, BK);
  if (tgp_bm == BM && tgp_bn == BN) {
    dv_gemm_loop_abf16<loader_a_t2, typename gemm_kernel::loader_b_t,
        typename gemm_kernel::mma_t, true, true>(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tile_dims_A, tile_dims_B);
    if (BF16O) {
      dv_store_bf16(mma_op, D_ptr, params->ldd, short2(tgp_bn, tgp_bm), false);
    } else {
      mma_op.store_result(Df_ptr, params->ldd);
    }
  } else if (tgp_bn == BN) {
    dv_gemm_loop_abf16<loader_a_t2, typename gemm_kernel::loader_b_t,
        typename gemm_kernel::mma_t, false, true>(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tile_dims_A, tile_dims_B);
    if (BF16O) {
      dv_store_bf16(mma_op, D_ptr, params->ldd, short2(tgp_bn, tgp_bm), true);
    } else {
      mma_op.store_result_safe(Df_ptr, params->ldd, short2(tgp_bn, tgp_bm));
    }
  } else if (tgp_bm == BM) {
    dv_gemm_loop_abf16<loader_a_t2, typename gemm_kernel::loader_b_t,
        typename gemm_kernel::mma_t, true, false>(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tile_dims_A, tile_dims_B);
    if (BF16O) {
      dv_store_bf16(mma_op, D_ptr, params->ldd, short2(tgp_bn, tgp_bm), true);
    } else {
      mma_op.store_result_safe(Df_ptr, params->ldd, short2(tgp_bn, tgp_bm));
    }
  } else {
    dv_gemm_loop_abf16<loader_a_t2, typename gemm_kernel::loader_b_t,
        typename gemm_kernel::mma_t, false, false>(
        As, Bs, gemm_k_iterations, loader_a, loader_b, mma_op, tile_dims_A, tile_dims_B);
    if (BF16O) {
      dv_store_bf16(mma_op, D_ptr, params->ldd, short2(tgp_bn, tgp_bm), true);
    } else {
      mma_op.store_result_safe(Df_ptr, params->ldd, short2(tgp_bn, tgp_bm));
    }
  }
}
"
}

pub fn steel_gemm_cvt_source() -> String {
    let _ = STEEL_GEMM_CVT_FUSIONS_ARE_BITWISE_BECAUSE_BOTH_CONVERSIONS_ARE_ELEMENTWISE_AND_THE_MMA_CHAIN_NEVER_MOVES;
    const _: () = STEEL_GEMM_CVT_CFGS_ARE_A_SUBSET_OF_THE_COMPILE_ASSERTED_STEEL_GEMM_CONFIGS;
    let mut b = steel_gemm_headers();
    b.push_str(steel_gemm_cvt_body());
    for cfg in STEEL_GEMM_CVT_CFGS_ARE_THE_TRUNK_MAP_TILES {
        let (bm, bn, bk, wm, wn) = *cfg;
        b.push_str(&format!(
            "template [[host_name(\"{}\")]] [[kernel]] decltype(dv_steel_gemm_nt_bf16o<{bm}, {bn}, {bk}, {wm}, {wn}>) dv_steel_gemm_nt_bf16o<{bm}, {bn}, {bk}, {wm}, {wn}>;\n",
            steel_gemm_cvt_entry("bf16o", *cfg)
        ));
        b.push_str(&format!(
            "template [[host_name(\"{}\")]] [[kernel]] decltype(dv_steel_gemm_nt_abf16<{bm}, {bn}, {bk}, {wm}, {wn}, false, false>) dv_steel_gemm_nt_abf16<{bm}, {bn}, {bk}, {wm}, {wn}, false, false>;\n",
            steel_gemm_cvt_entry("abf16", *cfg)
        ));
        b.push_str(&format!(
            "template [[host_name(\"{}\")]] [[kernel]] decltype(dv_steel_gemm_nt_abf16<{bm}, {bn}, {bk}, {wm}, {wn}, true, false>) dv_steel_gemm_nt_abf16<{bm}, {bn}, {bk}, {wm}, {wn}, true, false>;\n",
            steel_gemm_cvt_entry("abf16o", *cfg)
        ));
    }
    let sk = STEEL_GEMM_CVT_SK_CFG_IS_THE_ONLY_TILE_SPLITK_ENGAGES;
    let (bm, bn, bk, wm, wn) = sk;
    b.push_str(&format!(
        "template [[host_name(\"{}\")]] [[kernel]] decltype(dv_steel_gemm_nt_abf16<{bm}, {bn}, {bk}, {wm}, {wn}, false, true>) dv_steel_gemm_nt_abf16<{bm}, {bn}, {bk}, {wm}, {wn}, false, true>;\n",
        steel_gemm_cvt_entry("sk_abf16", sk)
    ));
    b
}

pub struct SteelGemmCvtPipelines {
    pub layout: wgpu::BindGroupLayout,
    by_kind: Vec<(
        (String, (u32, u32, u32, u32, u32)),
        Arc<wgpu::ComputePipeline>,
    )>,
}

impl SteelGemmCvtPipelines {
    pub fn pick(
        &self,
        kind: &str,
        cfg: (u32, u32, u32, u32, u32),
    ) -> Option<Arc<wgpu::ComputePipeline>> {
        self.by_kind
            .iter()
            .find(|((k, c), _)| k == kind && *c == cfg)
            .map(|(_, p)| p.clone())
    }
}

type SteelGemmCvtCache = Mutex<
    Vec<(
        wgpu::Device,
        std::result::Result<Arc<SteelGemmCvtPipelines>, String>,
    )>,
>;

fn steel_gemm_cvt_cache() -> &'static SteelGemmCvtCache {
    static CACHE: OnceLock<SteelGemmCvtCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn steel_gemm_cvt_build(ctx: &WgpuContext) -> Result<Arc<SteelGemmCvtPipelines>> {
    let mut wanted: Vec<(String, (u32, u32, u32, u32, u32), u32)> = Vec::new();
    for cfg in STEEL_GEMM_CVT_CFGS_ARE_THE_TRUNK_MAP_TILES {
        for kind in ["bf16o", "abf16", "abf16o"] {
            wanted.push((kind.to_string(), *cfg, cfg.3 * cfg.4 * 32));
        }
    }
    let sk = STEEL_GEMM_CVT_SK_CFG_IS_THE_ONLY_TILE_SPLITK_ENGAGES;
    wanted.push(("sk_abf16".to_string(), sk, sk.3 * sk.4 * 32));
    let entries_meta: Vec<(&'static str, (u32, u32, u32))> = wanted
        .iter()
        .map(|(kind, cfg, tgp)| {
            let e: &'static str = Box::leak(steel_gemm_cvt_entry(kind, *cfg).into_boxed_str());
            (e, (*tgp, 1, 1))
        })
        .collect();
    let src: &'static str = Box::leak(steel_gemm_cvt_source().into_boxed_str());
    let module = na::msl_module(ctx, "dv-steel-gemm-cvt", &entries_meta, src)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let entries = [
        storage_entry(0, true),
        storage_entry(1, true),
        storage_entry(2, false),
        uniform_entry(3),
    ];
    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dv-steel-gemm-cvt"),
            entries: &entries,
        });
    let pl = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dv-steel-gemm-cvt"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let mut by_kind = Vec::new();
    for ((kind, cfg, _), (name, _)) in wanted.iter().zip(entries_meta.iter()) {
        let pipeline = Arc::new(ctx.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            },
        ));
        by_kind.push(((kind.clone(), *cfg), pipeline));
    }
    if let Some(err) = pollster::block_on(scope.pop()) {
        bail!("dv-steel-gemm-cvt pipelines: {err}");
    }
    Ok(Arc::new(SteelGemmCvtPipelines {
        layout: bgl,
        by_kind,
    }))
}

pub fn steel_gemm_cvt_pipelines(ctx: &'static WgpuContext) -> Result<Arc<SteelGemmCvtPipelines>> {
    let mut guard = steel_gemm_cvt_cache().lock().unwrap();
    for (dev, entry) in guard.iter() {
        if *dev == ctx.device {
            return entry.clone().map_err(|e| anyhow::anyhow!("{e}"));
        }
    }
    let built = steel_gemm_cvt_build(ctx);
    let stored = match &built {
        Ok(p) => Ok(p.clone()),
        Err(e) => Err(format!("{e:#}")),
    };
    guard.push((ctx.device.clone(), stored));
    built
}

pub const STEEL_RESTAGE_MOVES_STAGING_AND_BARRIERS_NEVER_AN_FP_OP_SO_THE_ARM_IS_BITWISE_CLASS:
    &str =
    "The vendored attention keeps the score tile in simdgroup registers between its two MMAs \
     already; what it stages through threadgroup memory is K and V, ALIASED in one buffer, and \
     the alias forces 4 threadgroup barriers per k-block. The structure bracket priced that \
     serialization at ~7.5 ms of the invoice shape's 59.7 ms (barrier-free surgery walls run \
     23.4 TF/s, the steel GEMM class), priced the simdgroup fences as a measured WIN (+7 ms \
     when deleted -- they stay), and killed every staging-overlap alternative by measurement: \
     full K/V double-buffering pays +14 pct (21 KB smem halves threadgroup residency), V-only \
     double-buffering is a wash, direct-device V fragment reads pay +16 pct (4-warp redundant \
     strided reads), and the private async-copy builtin does not compile through the shipped \
     passthrough. The champion keeps exactly two barriers per k-block: K and V land in SEPARATE \
     smem buffers in one staging phase, and the Q fragments hoist to registers loaded once \
     straight from device memory (the same half words the smem path staged and re-read every \
     one of the 850 k-blocks), deleting Q_smem entirely -- 10.5 KB smem, 54.3 vs 59.7 ms at \
     13600 patches (-9.1 pct, three agreeing stamps), 0/20,889,600 output half words moved. \
     The QK mma chain, the scale multiply, the online-softmax order and the PV mma chain are \
     the vendored kernel's verbatim: the arm moves WHERE operands wait, never what arithmetic \
     runs, so it is bitwise-class by construction and gated as such. FLIPPED DEFAULT-ON under \
     the bitwise pre-authorization after the real-weight trunk gate held every one of the \
     5,222,400 merged f32 bits across both arms: unset serves the restage form at the bq32 \
     whole-page cfg -- at bq16 crop grids the win does not transfer (+1.1 pct at n=448), so \
     crops keep stock at every polarity.";

pub const STEEL_RESTAGE_SMEM_FITS_METALS_32768B_WORKGROUP_LIMIT: () = {
    let (_bq, bk, _wm, _wn) = STEEL_ATTN_DEFAULT_CFG;
    assert!(((bk + 8) * HEAD_DIM + bk * (HEAD_DIM + 8)) * 2 <= 32768);
};

const STEEL_RESTAGE_BODY: &str = r#"
template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void attention_restage(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5)]],
    const device float* mask [[buffer(6)]],
    const device T* sinks [[buffer(7)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  (void)lid;
  (void)mask_params;
  (void)mask;
  (void)sinks;

  ulong3 tidl{tid.x, tid.y, tid.z};

  Q += tidl.z * params->Q_strides[0] + tidl.y * params->Q_strides[1] +
      tidl.x * BQ * params->Q_strides[2];

  ulong kv_head_idx = int(tid.y) / params->gqa_factor;
  K += tidl.z * params->K_strides[0] + kv_head_idx * params->K_strides[1];
  V += tidl.z * params->V_strides[0] + kv_head_idx * params->V_strides[1];

  O += tidl.z * params->O_strides[0] + tidl.y * params->O_strides[1] +
      tidl.x * BQ * params->O_strides[2];

  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);
  constexpr short LDK_tgp = BK + padK;
  constexpr short LDV_tgp = BD + padV;
  constexpr short kmem = (BK + padK) * BD;
  constexpr short vmem = BK * (BD + padV);

  threadgroup T K_smem[kmem];
  threadgroup T V_smem[vmem];

  using KBlockLoader = BlockLoaderT<T, BK, BD, 1, LDK_tgp, 0, WM * WN * 32>;
  using VBlockLoader = BlockLoaderT<T, BK, BD, LDV_tgp, 1, 0, WM * WN * 32>;

  KBlockLoader loader_k(K, params->K_strides[2], K_smem, simd_group_id, simd_lane_id);
  VBlockLoader loader_v(V, params->V_strides[2], V_smem, simd_group_id, simd_lane_id);

  const AccumType scale = params->scale * M_LOG2E_F;

  constexpr short kFragSize = 8;
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>;

  constexpr int kNWarps = WM * WN;
  static_assert(
      BQ == kNWarps * kFragSize,
      "attention_restage hosts exactly one simdgroup matrix of Q rows per warp.");

  constexpr int TK = BK / kFragSize;
  constexpr int TD = BD / kFragSize;

  MMATile<AccumType, 1, TD, MMAFrag_acc_t> Qrtile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Stile;
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile;
  MMATile<AccumType, 1, 1, MMAFrag_acc_t> Vtile;
  MMATile<AccumType, 1, TD, MMAFrag_acc_t> Otile;

  Otile.clear();

  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * simd_group_id;

  const short Ks_offset = sm * LDK_tgp + sn;
  const short Vs_offset = sm * LDV_tgp + sn;
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp;

  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      MMAFrag_acc_t::load_safe(
          Qrtile.frag_at(0, dd),
          Q,
          int(params->Q_strides[2]),
          Int<1>{},
          params->qL_rem,
          BD,
          tm + sm,
          dd * kFragSize + sn);
    }
  } else {
    Qrtile.template load<T, 1, 1>(
        Q + (tm + sm) * params->Q_strides[2] + sn, int(params->Q_strides[2]));
  }

  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread;

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  for (int kb = 0; kb < params->NK; kb++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_K && kb == (params->NK_aligned)) {
      loader_k.load_safe(short2(BD, params->kL_rem));
      loader_v.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_k.load_unsafe();
      loader_v.load_unsafe();
    }
    loader_k.next();
    loader_v.next();

    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      Ktile.template load<T, 1, 1, LDK_tgp, 1>(&K_smem[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);

      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        MMAFrag_acc_t::mma(
            Stile.frag_at(0, ik),
            Qrtile.frag_at(0, dd),
            Ktile.frag_at(0, ik),
            Stile.frag_at(0, ik));
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < decltype(Stile)::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= scale;
    }

    if (!align_K && kb == (params->NK_aligned)) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= params->kL_rem) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    AccumType new_max[kRowsPT];
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    Stile.template row_reduce<MaxOp>(new_max);

    Stile.template row_bin_op<ExpSubOp>(new_max);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    Otile.template row_bin_op<MulOp>(factor);

    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TD; id++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        if (BD >= 128) {
          simdgroup_barrier(mem_flags::mem_none);
        }

        const short kk = ik * kFragSize;
        const short dd = id * kFragSize;

        Vtile.template load<T, 1, 1, LDV_tgp, 1>(&V_smem[Vs_offset + kk * LDV_tgp + dd]);

        if (BD >= 128) {
          simdgroup_barrier(mem_flags::mem_none);
        }

        MMAFrag_acc_t::mma(
            Otile.frag_at(0, id),
            Stile.frag_at(0, ik),
            Vtile.frag_at(0, 0),
            Otile.frag_at(0, id));
      }
    }
  }

  Otile.template row_bin_op<DivOp>(sum_score);
  threadgroup_barrier(mem_flags::mem_none);

  O += (tm + sm) * params->O_strides[2] + sn;

  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    auto dst_tile_dims = short2(BD - sn, params->qL_rem - (tm + sm));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0)
      return;

    Otile.template store_safe<T, 1, 1>(O, params->O_strides[2], dst_tile_dims);
  } else {
    Otile.template store<T, 1, 1>(O, params->O_strides[2]);
  }
}
"#;

pub fn steel_restage_entry_cfg(cfg: SteelAttnCfg, align_q: bool, align_k: bool) -> String {
    let (bq, bk, wm, wn) = cfg;
    format!(
        "dv_steel_attn_rs_bq{bq}_bk{bk}_wm{wm}_wn{wn}_q{}k{}",
        u8::from(align_q),
        u8::from(align_k)
    )
}

pub fn steel_restage_source_cfg(cfg: SteelAttnCfg, align_q: bool, align_k: bool) -> String {
    let _ = STEEL_RESTAGE_MOVES_STAGING_AND_BARRIERS_NEVER_AN_FP_OP_SO_THE_ARM_IS_BITWISE_CLASS;
    const _: () = STEEL_RESTAGE_SMEM_FITS_METALS_32768B_WORKGROUP_LIMIT;
    let (bq, bk, wm, wn) = cfg;
    let mut b = steel_source_cfg(cfg, align_q, align_k);
    b.push_str(STEEL_RESTAGE_BODY);
    b.push_str(&format!(
        "template [[host_name(\"{}\")]] [[kernel]] decltype(attention_restage<half, {bq}, {bk}, 128, {wm}, {wn}, float>) attention_restage<half, {bq}, {bk}, 128, {wm}, {wn}, float>;\n",
        steel_restage_entry_cfg(cfg, align_q, align_k)
    ));
    b
}

fn steel_restage_cache() -> &'static SteelCache {
    static CACHE: OnceLock<SteelCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

pub fn steel_restage_pipelines_cfg(
    ctx: &'static WgpuContext,
    cfg: SteelAttnCfg,
    align_q: bool,
    align_k: bool,
) -> Result<Arc<SteelPipelines>> {
    assert!(
        cfg == STEEL_ATTN_DEFAULT_CFG,
        "the restage arm is instantiated only at the crowned whole-page tile"
    );
    let mut guard = steel_restage_cache().lock().unwrap();
    for ((dev, c, aq, ak), entry) in guard.iter() {
        if *dev == ctx.device && *c == cfg && *aq == align_q && *ak == align_k {
            return entry.clone().map_err(|e| anyhow::anyhow!("{e}"));
        }
    }
    let entry: &'static str =
        Box::leak(steel_restage_entry_cfg(cfg, align_q, align_k).into_boxed_str());
    let src: &'static str =
        Box::leak(steel_restage_source_cfg(cfg, align_q, align_k).into_boxed_str());
    let built = (|| -> Result<Arc<SteelPipelines>> {
        let module = na::msl_module(
            ctx,
            "dv-steel-attn-rs",
            &[(entry, (cfg.2 * cfg.3 * 32, 1, 1))],
            src,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let entries = [
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, false),
            uniform_entry(4),
            uniform_entry(5),
            storage_entry(6, true),
            storage_entry(7, true),
        ];
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("dv-steel-attn-rs"),
                entries: &entries,
            });
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("dv-steel-attn-rs"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let pipeline = Arc::new(ctx.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            },
        ));
        if let Some(err) = pollster::block_on(scope.pop()) {
            bail!("dv-steel-attn-rs pipeline: {err}");
        }
        Ok(Arc::new(SteelPipelines {
            pipeline,
            layout: bgl,
        }))
    })();
    let stored = match &built {
        Ok(p) => Ok(p.clone()),
        Err(e) => Err(format!("{e:#}")),
    };
    guard.push(((ctx.device.clone(), cfg, align_q, align_k), stored));
    built
}

pub fn steel_pipelines_serving(
    ctx: &'static WgpuContext,
    cfg: SteelAttnCfg,
    align_q: bool,
    align_k: bool,
) -> Result<Arc<SteelPipelines>> {
    if cfg == STEEL_ATTN_DEFAULT_CFG {
        return steel_restage_pipelines_cfg(ctx, cfg, align_q, align_k);
    }
    steel_pipelines_cfg(ctx, cfg, align_q, align_k)
}
