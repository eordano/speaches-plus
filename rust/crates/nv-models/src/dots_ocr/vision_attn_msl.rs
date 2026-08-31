use std::sync::{Arc, Mutex, OnceLock};

use anyhow::Result;
use nv_kernels::wgpu_backend::{na, WgpuContext};

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


fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub const STEEL_BQ: u32 = 32;
pub const STEEL_BK: u32 = 16;

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
    format!(
        "dv_steel_attn_q{}k{}",
        u8::from(align_q),
        u8::from(align_k)
    )
}

pub fn steel_source(align_q: bool, align_k: bool) -> String {
    let _ = STEEL_IS_CONCATENATED_AND_CONSTANT_BAKED_BECAUSE_RUNTIME_MSL_HAS_NO_INCLUDE_PATH_AND_PASSTHROUGH_SETS_NO_FUNCTION_CONSTANTS;
    let mut b = String::from(STEEL_LIMITS_SHIM_MATCHES_THE_FLOAT_SPECIALIZATION_IN_MLX_KERNELS_UTILS_H);
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
        ("constant bool has_mask [[function_constant(300)]];", "constant bool has_mask = false;"),
        ("constant bool do_causal [[function_constant(301)]];", "constant bool do_causal = false;"),
        ("constant bool has_sinks [[function_constant(302)]];", "constant bool has_sinks = false;"),
        (", function_constant(has_mask)]]", "]]"),
        (", function_constant(has_sinks)]]", "]]"),
    ] {
        assert!(b.contains(from), "steel source lost the anchor {from}");
        b = b.replace(from, to);
    }
    b.push_str(&format!(
        "\ntemplate [[host_name(\"{}\")]] [[kernel]] decltype(attention<half, 32, 16, 128, 4, 1, float, float>) attention<half, 32, 16, 128, 4, 1, float, float>;\n",
        steel_entry(align_q, align_k)
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

type SteelCache =
    Mutex<Vec<((wgpu::Device, bool, bool), std::result::Result<Arc<SteelPipelines>, String>)>>;

fn steel_cache() -> &'static SteelCache {
    static CACHE: OnceLock<SteelCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn steel_build(ctx: &WgpuContext, align_q: bool, align_k: bool) -> Result<Arc<SteelPipelines>> {
    let entry: &'static str = Box::leak(steel_entry(align_q, align_k).into_boxed_str());
    let src: &'static str = Box::leak(steel_source(align_q, align_k).into_boxed_str());
    let module = na::msl_module(ctx, "dv-steel-attn", &[(entry, (128, 1, 1))], src)
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
        anyhow::bail!("dv-steel-attn pipeline: {err}");
    }
    Ok(Arc::new(SteelPipelines { pipeline, layout: bgl }))
}

pub fn steel_pipelines(
    ctx: &'static WgpuContext,
    align_q: bool,
    align_k: bool,
) -> Result<Arc<SteelPipelines>> {
    let mut guard = steel_cache().lock().unwrap();
    for ((dev, aq, ak), entry) in guard.iter() {
        if *dev == ctx.device && *aq == align_q && *ak == align_k {
            return entry.clone().map_err(|e| anyhow::anyhow!("{e}"));
        }
    }
    let built = steel_build(ctx, align_q, align_k);
    let stored = match &built {
        Ok(p) => Ok(p.clone()),
        Err(e) => Err(format!("{e:#}")),
    };
    guard.push(((ctx.device.clone(), align_q, align_k), stored));
    built
}

pub fn steel_grid(heads: u32, n_real: u32) -> (u32, u32, u32) {
    (n_real.div_ceil(STEEL_BQ), heads, 1)
}

pub const STEEL_GEMM_CONFIGS: &[(u32, u32, u32, u32, u32)] = &[
    (32, 32, 16, 2, 2),
    (32, 64, 16, 2, 2),
    (64, 32, 16, 2, 2),
    (64, 64, 16, 2, 2),
    (64, 64, 32, 2, 2),
    (64, 128, 16, 2, 2),
    (128, 64, 16, 2, 2),
];

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

pub const STEEL_GEMM_COMPLEX_SHIM_EXISTS_SO_THE_NEVER_INSTANTIATED_COMPLEX_SPECIALIZATIONS_PARSE: &str = "
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

pub fn steel_gemm_source() -> String {
    let _ = STEEL_GEMM_RIDES_THE_SAME_DRIVER_PATTERNS_THE_ATTENTION_PORT_PROVED;
    let mut b = String::from(STEEL_GEMM_COMPLEX_SHIM_EXISTS_SO_THE_NEVER_INSTANTIATED_COMPLEX_SPECIALIZATIONS_PARSE);
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

type SteelGemmCache =
    Mutex<Vec<(wgpu::Device, std::result::Result<Arc<SteelGemmPipelines>, String>)>>;

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
    for ((cfg, (name, _))) in STEEL_GEMM_CONFIGS.iter().zip(entries_meta.iter()) {
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
        anyhow::bail!("dv-steel-gemm pipelines: {err}");
    }
    Ok(Arc::new(SteelGemmPipelines { by_config, layout: bgl }))
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
