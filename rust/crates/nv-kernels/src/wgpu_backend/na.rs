use std::sync::{Arc, Mutex, OnceLock};

use super::device::WgpuContext;
use super::dispatch::profile;
use super::{Result, WgpuError};

pub const TILE_N: u32 = 64;
pub const TILE_M: u32 = 16;
pub const K_CHUNK: u32 = 32;
pub const WG_THREADS: u32 = 128;

pub const ENTRY_PK: &str = "na_gemm_w4a16_pk";
pub const ENTRY_PK3: &str = "na_gemm_w4a16_pk3";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NaStaticParams {
    pub n_rows: u32,
    pub k_elems: u32,
    pub gs: u32,
    pub scale_row_stride: u32,
    pub scale_elem_stride: u32,
    pub q_rows: u32,
    pub kv_rows: u32,
    pub v_off: u32,
}

pub const MSL: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

struct NaMkParams {
    uint m;
    uint x_stride_words;
    uint y_stride_words;
    uint dst_word_off;
};

struct NaStaticParams {
    uint n_rows;
    uint k_elems;
    uint gs;
    uint scale_row_stride;
    uint scale_elem_stride;
    uint q_rows;
    uint kv_rows;
    uint v_off;
};

constexpr constant int TN = 64;
constexpr constant int TM = 16;
constexpr constant int KC = 32;
constexpr constant int NSG = 4;

static inline uint na_bf16_encode(float x) {
    uint b = as_type<uint>(x);
    uint r = 0x7fffu + ((b >> 16u) & 1u);
    return (x != x) ? 0x7fc0u : ((b + r) >> 16u);
}

static void na_tile(
    device uint* wq,
    device ushort* ws,
    device bfloat* xb,
    constant NaMkParams& mk,
    constant NaStaticParams& np,
    int n0,
    uint sgid,
    uint lane,
    uint lid,
    threadgroup float* tg_scale,
    threadgroup float* tg_asum,
    threadgroup float* ftile)
{
    int K = int(np.k_elems);
    int NR = int(np.n_rows);
    int m = int(mk.m);

    auto tA = tensor(xb,
                     dextents<int32_t, 2>(K, m),
                     array<int, 2>{1, int(mk.x_stride_words) * 2});
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tW(
        reinterpret_cast<device uchar*>(wq), dextents<int32_t, 2>(K, NR));

    constexpr auto d = matmul2d_descriptor(TM, TN, KC, false, true, false,
                                           matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroup> op;

    auto sA0 = tA.slice(0, 0);
    auto sW0 = tW.slice(0, n0);
    auto acc = op.get_destination_cooperative_tensor<decltype(sA0), decltype(sW0), float>();
    auto cg = op.get_destination_cooperative_tensor<decltype(sA0), decltype(sW0), float>();
    constexpr int CAP = (TM * TN) / 32;
    uint16_t nloc[CAP];
    uint16_t mloc[CAP];
    #pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        acc[i] = 0.0f;
        auto idx = acc.get_multidimensional_index(i);
        nloc[i] = acc.is_valid_element(i) ? uint16_t(idx[0]) : uint16_t(0);
        mloc[i] = acc.is_valid_element(i) ? uint16_t(idx[1]) : uint16_t(0);
    }

    for (int k0 = int(sgid) * KC; k0 < K; k0 += NSG * KC) {
        uint g = uint(k0) / np.gs;
        for (uint i = lane; i < uint(TN); i += 32u) {
            int row = n0 + int(i);
            tg_scale[sgid * uint(TN) + i] = row < NR
                ? float(as_type<bfloat>(ws[(uint(row) * np.scale_row_stride + g) * np.scale_elem_stride]))
                : 0.0f;
        }
        if (lane < uint(m)) {
            device bfloat* xr = xb + lane * mk.x_stride_words * 2 + uint(k0);
            float s = 0.0f;
            for (int i = 0; i < KC; i++) {
                s += float(xr[i]);
            }
            tg_asum[sgid * uint(TM) + lane] = s;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto sA = tA.slice(k0, 0);
        auto sW = tW.slice(k0, n0);
        op.run(sA, sW, cg);
        #pragma unroll
        for (uint16_t i = 0; i < uint16_t(CAP); ++i) {
            acc[i] += tg_scale[sgid * uint(TN) + nloc[i]] * (cg[i] - 8.0f * tg_asum[sgid * uint(TM) + mloc[i]]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = lid; i < uint(TM * TN); i += 128u) {
        ftile[i] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 0; s < uint(NSG); s++) {
        if (sgid == s) {
            #pragma unroll
            for (uint16_t i = 0; i < uint16_t(CAP); ++i) {
                if (acc.is_valid_element(i)) {
                    ftile[uint(mloc[i]) * TN + uint(nloc[i])] += acc[i];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void na_gemm_w4a16_pk(
    device uint* wq [[buffer(0)]],
    device ushort* ws [[buffer(1)]],
    device bfloat* xb [[buffer(2)]],
    device uint* y [[buffer(3)]],
    constant NaMkParams& mk [[buffer(4)]],
    constant NaStaticParams& np [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint lid [[thread_index_in_threadgroup]])
{
    if (mk.m == 0u) {
        return;
    }
    threadgroup float tg_scale[NSG * TN];
    threadgroup float tg_asum[NSG * TM];
    threadgroup float ftile[TM * TN];
    int n0 = int(tgid.x) * TN;
    na_tile(wq, ws, xb, mk, np, n0, sgid, lane, lid, tg_scale, tg_asum, ftile);

    int NR = int(np.n_rows);
    uint words = uint(TN / 2) * mk.m;
    for (uint w = lid; w < words; w += 128u) {
        uint t = w / uint(TN / 2);
        uint rp = w % uint(TN / 2);
        int row = n0 + int(2 * rp);
        if (row >= NR) {
            continue;
        }
        uint lo = na_bf16_encode(ftile[t * TN + 2 * rp]) & 0xffffu;
        uint hi = (row + 1 < NR) ? na_bf16_encode(ftile[t * TN + 2 * rp + 1]) : 0u;
        y[mk.dst_word_off + t * mk.y_stride_words + uint(row >> 1)] = lo | (hi << 16u);
    }
}

kernel void na_gemm_w4a16_pk3(
    device uint* wq [[buffer(0)]],
    device ushort* ws [[buffer(1)]],
    device bfloat* xb [[buffer(2)]],
    device uint* yq [[buffer(3)]],
    device uint* yk [[buffer(4)]],
    device uint* yv [[buffer(5)]],
    constant NaMkParams& mk [[buffer(6)]],
    constant NaStaticParams& np [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint lid [[thread_index_in_threadgroup]])
{
    if (mk.m == 0u) {
        return;
    }
    threadgroup float tg_scale[NSG * TN];
    threadgroup float tg_asum[NSG * TM];
    threadgroup float ftile[TM * TN];
    int n0 = int(tgid.x) * TN;
    na_tile(wq, ws, xb, mk, np, n0, sgid, lane, lid, tg_scale, tg_asum, ftile);

    int NR = int(np.n_rows);
    uint words = uint(TN / 2) * mk.m;
    for (uint w = lid; w < words; w += 128u) {
        uint t = w / uint(TN / 2);
        uint rp = w % uint(TN / 2);
        uint row = uint(n0) + 2u * rp;
        if (int(row) >= NR) {
            continue;
        }
        uint lo = na_bf16_encode(ftile[t * TN + 2 * rp]) & 0xffffu;
        uint hi = (int(row) + 1 < NR) ? na_bf16_encode(ftile[t * TN + 2 * rp + 1]) : 0u;
        uint word = lo | (hi << 16u);
        if (row < np.q_rows) {
            yq[t * (np.q_rows >> 1u) + (row >> 1u)] = word;
        } else {
            uint kr = row - np.q_rows;
            if (kr < np.kv_rows) {
                yk[t * (np.kv_rows >> 1u) + (kr >> 1u)] = word;
            }
            if (row >= np.v_off) {
                uint vr = row - np.v_off;
                if (vr < np.kv_rows) {
                    yv[t * (np.kv_rows >> 1u) + (vr >> 1u)] = word;
                }
            }
        }
    }
}
"#;

pub fn supported(ctx: &WgpuContext) -> bool {
    ctx.info.backend == wgpu::Backend::Metal
        && ctx
            .device
            .features()
            .contains(wgpu::Features::PASSTHROUGH_SHADERS)
}

#[cfg(target_os = "macos")]
mod msl4 {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::OnceLock;

    use objc2::runtime::{AnyObject, Sel};
    use objc2::{msg_send, sel};

    const MSL_4_0: usize = 4 << 16;

    static ACTIVE: AtomicBool = AtomicBool::new(false);
    static ORIG_IMP: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C-unwind" fn init_hook(this: *mut AnyObject, cmd: Sel) -> *mut AnyObject {
        let orig: unsafe extern "C-unwind" fn(*mut AnyObject, Sel) -> *mut AnyObject =
            unsafe { std::mem::transmute(ORIG_IMP.load(Ordering::Acquire)) };
        let obj = unsafe { orig(this, cmd) };
        if !obj.is_null() && ACTIVE.load(Ordering::Acquire) {
            let current: usize = unsafe { msg_send![&*obj, languageVersion] };
            if current < MSL_4_0 {
                let _: () = unsafe { msg_send![&*obj, setLanguageVersion: MSL_4_0] };
            }
        }
        obj
    }

    fn install() -> bool {
        static INSTALLED: OnceLock<bool> = OnceLock::new();
        *INSTALLED.get_or_init(|| {
            let Some(facade) = objc2::runtime::AnyClass::get(c"MTLCompileOptions") else {
                return false;
            };
            let probe: *mut AnyObject = unsafe { msg_send![facade, new] };
            if probe.is_null() {
                return false;
            }
            let cls = unsafe { (*probe).class() };
            let _: () = unsafe { msg_send![&*probe, release] };
            let Some(inherited) = cls.instance_method(sel!(init)) else {
                return false;
            };
            let own = cls
                .instance_methods()
                .iter()
                .any(|m| m.name() == sel!(init));
            unsafe {
                let hook: unsafe extern "C-unwind" fn(*mut AnyObject, Sel) -> *mut AnyObject =
                    init_hook;
                let imp: unsafe extern "C-unwind" fn() = std::mem::transmute::<
                    unsafe extern "C-unwind" fn(*mut AnyObject, Sel) -> *mut AnyObject,
                    unsafe extern "C-unwind" fn(),
                >(hook);
                if own {
                    let orig = inherited.set_implementation(imp);
                    ORIG_IMP.store(orig as usize, Ordering::Release);
                    true
                } else {
                    ORIG_IMP.store(inherited.implementation() as usize, Ordering::Release);
                    let added = objc2::ffi::class_addMethod(
                        (cls as *const objc2::runtime::AnyClass).cast_mut(),
                        sel!(init),
                        imp,
                        c"@@:".as_ptr(),
                    );
                    added.as_bool()
                }
            }
        })
    }

    pub struct Guard(());

    impl Guard {
        pub fn activate() -> Option<Self> {
            if !install() {
                return None;
            }
            ACTIVE.store(true, Ordering::Release);
            Some(Guard(()))
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            ACTIVE.store(false, Ordering::Release);
        }
    }

    pub fn probe() -> Option<(usize, usize)> {
        let _guard = Guard::activate()?;
        let cls = objc2::runtime::AnyClass::get(c"MTLCompileOptions")?;
        let obj: *mut AnyObject = unsafe { msg_send![cls, new] };
        if obj.is_null() {
            return None;
        }
        let hooked: usize = unsafe { msg_send![&*obj, languageVersion] };
        drop(_guard);
        let obj2: *mut AnyObject = unsafe { msg_send![cls, new] };
        let plain: usize = unsafe { msg_send![&*obj2, languageVersion] };
        let _: () = unsafe { msg_send![&*obj, release] };
        let _: () = unsafe { msg_send![&*obj2, release] };
        Some((hooked, plain))
    }
}

#[cfg(target_os = "macos")]
pub fn msl4_probe() -> Option<(usize, usize)> {
    msl4::probe()
}

struct NaPipelines {
    pk: Arc<wgpu::ComputePipeline>,
    pk3: Arc<wgpu::ComputePipeline>,
}

type CacheEntry = (
    wgpu::Device,
    std::result::Result<Arc<NaPipelines>, WgpuError>,
);

fn cache() -> &'static Mutex<Vec<CacheEntry>> {
    static CACHE: OnceLock<Mutex<Vec<CacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

pub(crate) fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

pub(crate) fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

#[cfg(not(target_os = "macos"))]
pub fn msl_module(
    ctx: &WgpuContext,
    _label: &str,
    _entries: &[(&str, (u32, u32, u32))],
    _msl: &'static str,
) -> Result<wgpu::ShaderModule> {
    let _ = ctx;
    Err(WgpuError::Unsupported(
        "na tensor-ops are macOS-only".into(),
    ))
}

#[cfg(target_os = "macos")]
pub fn msl_module(
    ctx: &WgpuContext,
    label: &str,
    entries: &[(&str, (u32, u32, u32))],
    msl: &'static str,
) -> Result<wgpu::ShaderModule> {
    let _guard = msl4::Guard::activate().ok_or_else(|| {
        WgpuError::Unsupported("MTLCompileOptions language-version hook failed to install".into())
    })?;
    let internal = ctx.device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = unsafe {
        ctx.device
            .create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some(label),
                entry_points: std::borrow::Cow::Owned(
                    entries
                        .iter()
                        .map(
                            |&(name, workgroup_size)| wgpu::PassthroughShaderEntryPoint {
                                name: std::borrow::Cow::Owned(name.to_string()),
                                workgroup_size,
                            },
                        )
                        .collect(),
                ),
                msl: Some(std::borrow::Cow::Borrowed(msl)),
                ..Default::default()
            })
    };
    if let Some(err) = pollster::block_on(validation.pop()) {
        return Err(WgpuError::ShaderCompile(format!("{label}: {err}")));
    }
    if let Some(err) = pollster::block_on(internal.pop()) {
        return Err(WgpuError::ShaderCompile(format!("{label}: {err}")));
    }
    Ok(module)
}

fn build_pipelines(ctx: &WgpuContext) -> Result<Arc<NaPipelines>> {
    let module = msl_module(
        ctx,
        "nv-na-gemm-w4a16",
        &[
            (ENTRY_PK, (WG_THREADS, 1, 1)),
            (ENTRY_PK3, (WG_THREADS, 1, 1)),
        ],
        MSL,
    )?;
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mk_layout = |label: &str, storages: &[(u32, bool)], uniforms: &[u32]| {
        let mut entries: Vec<wgpu::BindGroupLayoutEntry> = storages
            .iter()
            .map(|&(b, ro)| storage_entry(b, ro))
            .collect();
        entries.extend(uniforms.iter().map(|&b| uniform_entry(b)));
        let bgl = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &entries,
            });
        ctx.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            })
    };
    let pk_layout = mk_layout(
        "nv-na-pk",
        &[(0, true), (1, true), (2, true), (3, false)],
        &[4, 5],
    );
    let pk3_layout = mk_layout(
        "nv-na-pk3",
        &[
            (0, true),
            (1, true),
            (2, true),
            (3, false),
            (4, false),
            (5, false),
        ],
        &[6, 7],
    );
    let mk_pipeline = |label: &str, layout: &wgpu::PipelineLayout, entry: &str| {
        Arc::new(
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        )
    };
    let pk = mk_pipeline("nv-na-gemm-w4-pk", &pk_layout, ENTRY_PK);
    let pk3 = mk_pipeline("nv-na-gemm-w4-pk3", &pk3_layout, ENTRY_PK3);
    if let Some(err) = pollster::block_on(scope.pop()) {
        return Err(WgpuError::ShaderCompile(format!("na passthrough: {err}")));
    }
    profile::name_pipeline(&pk, "na-gemm-w4:na_gemm_w4a16_pk");
    profile::name_pipeline(&pk3, "na-gemm-w4:na_gemm_w4a16_pk3");
    Ok(Arc::new(NaPipelines { pk, pk3 }))
}

fn pipelines(ctx: &WgpuContext) -> Result<Arc<NaPipelines>> {
    let mut guard = cache().lock().unwrap();
    for (dev, entry) in guard.iter() {
        if *dev == ctx.device {
            return entry.clone();
        }
    }
    let built = if supported(ctx) {
        build_pipelines(ctx)
    } else {
        Err(WgpuError::Unsupported(
            "na tensor-ops need the Metal backend with PASSTHROUGH_SHADERS".into(),
        ))
    };
    guard.push((ctx.device.clone(), built.clone()));
    built
}

pub fn pk_pipeline(ctx: &WgpuContext) -> Result<Arc<wgpu::ComputePipeline>> {
    pipelines(ctx).map(|p| p.pk.clone())
}

pub fn pk3_pipeline(ctx: &WgpuContext) -> Result<Arc<wgpu::ComputePipeline>> {
    pipelines(ctx).map(|p| p.pk3.clone())
}

pub fn available(ctx: &WgpuContext) -> bool {
    pipelines(ctx).is_ok()
}

pub fn pipeline_label(p: &wgpu::ComputePipeline) -> Option<&'static str> {
    let guard = cache().lock().ok()?;
    for (_, entry) in guard.iter() {
        if let Ok(pls) = entry {
            if pls.pk.as_ref() == p {
                return Some("na_gemm_w4");
            }
            if pls.pk3.as_ref() == p {
                return Some("na_gemm_w4_qkv");
            }
        }
    }
    None
}

pub fn grid_x(n_rows: u32) -> u32 {
    n_rows.div_ceil(TILE_N)
}
