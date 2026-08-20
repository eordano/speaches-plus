#![cfg(feature = "wgpu")]

mod common;
use common::d;
use common::require;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch::{self, GpuUniform};
use nv_kernels::wgpu_backend::kernels::gemv_w4a16 as gw;
use nv_kernels::wgpu_backend::na;
use common::LcgShift33W4a16Packs as Lcg;

struct StderrLog;

impl log::Log for StderrLog {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Error
    }
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
    }
    fn flush(&self) {}
}

fn init_log() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = log::set_logger(&StderrLog);
        log::set_max_level(log::LevelFilter::Error);
    });
}

fn ctx_or_skip(test: &str) -> Option<&'static WgpuContext> {
    init_log();
    match WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("{test}: {}", ctx.summary());
            Some(ctx)
        }
        Err(e) => {
            if require() {
                panic!(
                    "{test}: no wgpu adapter: {e}. This gate refuses to report \
                     success without running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to \
                     skip on purpose."
                );
            }
            eprintln!("{test}: SKIP no wgpu adapter: {e}");
            None
        }
    }
}

fn skip_row_out_of_scope_because_adapter_backend_is_not_metal(
    test: &str,
    ctx: &WgpuContext,
) -> bool {
    if ctx.info.backend == wgpu::Backend::Metal {
        return false;
    }
    eprintln!(
        "{test}: SKIP row not run: na tensor-ops are MSL-only (metal_tensor + \
         MetalPerformancePrimitives) and this adapter backend is {:?}, not Metal, so the \
         Metal-passthrough property is out of scope on this box; a Metal adapter still \
         panics loudly if na is unsupported there",
        ctx.info.backend
    );
    true
}

fn na_or_skip(test: &str) -> Option<&'static WgpuContext> {
    let ctx = ctx_or_skip(test)?;
    if !na::available(ctx) {
        if !require() {
            eprintln!(
                "{test}: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) na tensor-ops unavailable on \
                 this adapter"
            );
            return None;
        }

        if skip_row_out_of_scope_because_adapter_backend_is_not_metal(test, ctx) {
            return None;
        }
        if !na::supported(ctx) {
            panic!(
                "{test}: this IS a Metal backend yet na::supported() is false, meaning \
                 PASSTHROUGH_SHADERS is missing on a Metal adapter. That is the property this \
                 suite exists to check, so it must not be skipped past; set \
                 NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        panic!(
            "{test}: this IS a Metal backend with PASSTHROUGH_SHADERS, yet na::available() is \
             false -- the na pipelines FAILED TO COMPILE. That is a real defect and the property \
             this suite exists to check, so it must not be skipped past; set \
             NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
        );
    }
    Some(ctx)
}

fn qv(packed: &[u32], n: usize, k: usize, row_words: usize) -> f32 {
    let lin = n * row_words * 8 + k;
    (((packed[lin / 8] >> (4 * (lin % 8))) & 0xf) as i32 - 8) as f32
}

const MIN_MM_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

kernel void na_min_mm(device half* a [[buffer(0)]],
                      device half* b [[buffer(1)]],
                      device float* c [[buffer(2)]])
{
    auto tA = tensor(a, dextents<int32_t, 2>(16, 8));
    auto tB = tensor(b, dextents<int32_t, 2>(16, 16));
    auto tC = tensor(c, dextents<int32_t, 2>(16, 8));
    constexpr auto d = matmul2d_descriptor(8, 16, 16, false, false, false,
                                           matmul2d_descriptor::mode::multiply);
    matmul2d<d, execution_simdgroups<4>> op;
    op.run(tA, tB, tC);
}
"#;

#[test]
#[cfg(target_os = "macos")]
fn na_msl4_hook_bumps_default_language_version() {
    init_log();
    match na::msl4_probe() {
        Some((hooked, plain)) => {
            eprintln!("msl4 probe: hooked default {hooked:#x}, plain default {plain:#x}");
            assert!(hooked >= 4 << 16, "hook did not reach MSL 4.0: {hooked:#x}");
        }
        None => panic!("msl4 hook failed to install"),
    }
}

#[test]
fn na_passthrough_minimal_matmul2d() {
    let Some(ctx) = ctx_or_skip("na_passthrough_minimal_matmul2d") else {
        return;
    };
    if !na::supported(ctx) {
        if !require() {
            eprintln!(
                "na_passthrough_minimal_matmul2d: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) \
                 passthrough unsupported"
            );
            return;
        }
        if skip_row_out_of_scope_because_adapter_backend_is_not_metal(
            "na_passthrough_minimal_matmul2d",
            ctx,
        ) {
            return;
        }
        panic!(
            "na_passthrough_minimal_matmul2d: this IS a Metal backend yet na::supported() is \
             false, meaning PASSTHROUGH_SHADERS is missing on a Metal adapter, which is the \
             property under test; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
        );
    }
    let (m, n, k) = (8usize, 16usize, 16usize);
    let mut rng = Lcg::new(7);
    let a: Vec<half::f16> = (0..m * k)
        .map(|_| half::f16::from_f32(rng.next_f32()))
        .collect();
    let b: Vec<half::f16> = (0..k * n)
        .map(|_| half::f16::from_f32(rng.next_f32()))
        .collect();
    let a_bits: Vec<u16> = a.iter().map(|v| v.to_bits()).collect();
    let b_bits: Vec<u16> = b.iter().map(|v| v.to_bits()).collect();

    let module = na::msl_module(ctx, "na-min-mm", &[("na_min_mm", (128, 1, 1))], MIN_MM_MSL)
        .expect("runtime MSL compile of matmul2d kernel");
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..3)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: i < 2 },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("na-min-mm"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("na_min_mm"),
            compilation_options: Default::default(),
            cache: None,
        });
    if let Some(err) = pollster::block_on(scope.pop()) {
        panic!("passthrough pipeline failed: {err}");
    }

    let buf_a = dispatch::storage_from_slice(ctx, "na-a", &a_bits);
    let buf_b = dispatch::storage_from_slice(ctx, "na-b", &b_bits);
    let buf_c = dispatch::storage_zeroed(ctx, "na-c", (m * n * 4) as u64);
    dispatch::dispatch(
        ctx,
        &pipeline,
        &[(0, &buf_a), (1, &buf_b), (2, &buf_c)],
        (1, 1, 1),
    )
    .unwrap();
    let got: Vec<f32> = dispatch::read_back(ctx, &buf_c, m * n).unwrap();
    let mut max_err = 0f32;
    for mi in 0..m {
        for ni in 0..n {
            let mut r = 0f32;
            for ki in 0..k {
                r += a[mi * k + ki].to_f32() * b[ki * n + ni].to_f32();
            }
            max_err = max_err.max((got[mi * n + ni] - r).abs());
        }
    }
    eprintln!("na_passthrough_minimal_matmul2d: max abs err {max_err:.2e}");
    assert!(max_err < 1e-2, "matmul2d output diverged: {max_err}");
}

struct Case {
    n: usize,
    k: usize,
    m: usize,
    gs: usize,
    y_stride_words: usize,
    dst_word_off: usize,
}

fn cpu_ref(packed: &[u32], scales: &[u16], x: &[u16], c: &Case, xs_words: usize) -> Vec<f32> {
    let row_words = c.k / 8;
    let srs = c.k / c.gs;
    let mut out = vec![0f32; c.m * c.n];
    for t in 0..c.m {
        for n in 0..c.n {
            let mut acc = 0f64;
            for g in 0..srs {
                let mut dot = 0f64;
                for kk in g * c.gs..(g + 1) * c.gs {
                    dot += f64::from(d(x[t * xs_words * 2 + kk]))
                        * f64::from(qv(packed, n, kk, row_words));
                }
                acc += f64::from(d(scales[n * srs + g])) * dot;
            }
            out[t * c.n + n] = acc as f32;
        }
    }
    out
}

fn unpack_y(y: &[u32], c: &Case) -> Vec<f32> {
    let mut out = vec![0f32; c.m * c.n];
    for t in 0..c.m {
        for n in 0..c.n {
            let word = y[c.dst_word_off + t * c.y_stride_words + n / 2];
            let bits = if n % 2 == 0 {
                (word & 0xffff) as u16
            } else {
                (word >> 16) as u16
            };
            out[t * c.n + n] = d(bits);
        }
    }
    out
}

fn check_close(label: &str, got: &[f32], want: &[f32], rel_tol: f32) {
    let mut max_rel = 0f32;
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let denom = w.abs().max(1e-3);
        let rel = (g - w).abs() / denom;
        if rel > max_rel {
            max_rel = rel;
        }
        assert!(
            rel <= rel_tol,
            "{label}: element {i} got {g} want {w} rel {rel} > {rel_tol}"
        );
    }
    eprintln!(
        "{label}: max rel err {max_rel:.2e} over {} elements",
        got.len()
    );
}

fn run_case(ctx: &'static WgpuContext, c: &Case) {
    let mut rng = Lcg::new(0x5eed ^ (c.n as u64) << 20 ^ (c.k as u64));
    let row_words = c.k / 8;
    let srs = c.k / c.gs;
    let xs_words = c.k / 2;
    let packed = rng.packed(c.n * row_words);
    let scales = rng.scales(c.n * srs);
    let x = rng.bf16_words(c.m * c.k, 1.0);

    let scale_words = gw::pack_scale_words(&scales);
    let x_words: Vec<u32> = x
        .chunks(2)
        .map(|p| (p[0] as u32) | ((p[1] as u32) << 16))
        .collect();

    let buf_w = dispatch::storage_from_slice(ctx, "na-t-w", &packed);
    let buf_s = dispatch::storage_from_slice(ctx, "na-t-s", &scale_words);
    let buf_x = dispatch::storage_from_slice(ctx, "na-t-x", &x_words);
    let y_len = c.dst_word_off + c.m * c.y_stride_words;
    let buf_y_na = dispatch::storage_zeroed(ctx, "na-t-y", (y_len * 4) as u64);
    let buf_y_sg = dispatch::storage_zeroed(ctx, "na-t-y2", (y_len * 4) as u64);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MkP {
        m: u32,
        x_stride_words: u32,
        y_stride_words: u32,
        dst_word_off: u32,
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct SgP {
        n_rows: u32,
        k_elems: u32,
        gs: u32,
        w_row_words: u32,
        scale_row_stride: u32,
        groups_x: u32,
    }
    let mkp = GpuUniform::new(
        ctx,
        "na-t-mk",
        &MkP {
            m: c.m as u32,
            x_stride_words: (c.k / 2) as u32,
            y_stride_words: c.y_stride_words as u32,
            dst_word_off: c.dst_word_off as u32,
        },
    );
    let nap = GpuUniform::new(
        ctx,
        "na-t-np",
        &na::NaStaticParams {
            n_rows: c.n as u32,
            k_elems: c.k as u32,
            gs: c.gs as u32,
            scale_row_stride: srs as u32,
            scale_elem_stride: 1,
            q_rows: 0,
            kv_rows: 0,
            v_off: 0,
        },
    );

    let na_pipe = na::pk_pipeline(ctx).unwrap();
    dispatch::dispatch(
        ctx,
        &na_pipe,
        &[
            (0, &buf_w),
            (1, &buf_s),
            (2, &buf_x),
            (3, &buf_y_na),
            (4, mkp.raw()),
            (5, nap.raw()),
        ],
        (na::grid_x(c.n as u32), 1, 1),
    )
    .unwrap();

    let want = cpu_ref(&packed, &scales, &x, c, xs_words);
    let y_na: Vec<u32> = dispatch::read_back(ctx, &buf_y_na, y_len).unwrap();
    let got_na = unpack_y(&y_na, c);
    check_close(
        &format!("na vs cpu n={} k={} m={}", c.n, c.k, c.m),
        &got_na,
        &want,
        0.02,
    );

    if gw::sg_pk_supported(ctx.subgroup_width()) && c.m <= gw::SG_MK_MAX as usize {
        let sgp = GpuUniform::new(
            ctx,
            "na-t-sg",
            &SgP {
                n_rows: c.n as u32,
                k_elems: c.k as u32,
                gs: c.gs as u32,
                w_row_words: row_words as u32,
                scale_row_stride: srs as u32,
                groups_x: 0,
            },
        );
        let src = gw::sg_mk_source(c.m as u32);
        let groups = (c.n as u32).div_ceil(gw::SG_PK_ROWS);
        dispatch::run(
            ctx,
            "na-t-sgmk",
            &src,
            gw::SG_MK_PK_ENTRY,
            &[
                (1, &buf_s),
                (3, &buf_y_sg),
                (4, sgp.raw()),
                (6, &buf_w),
                (7, &buf_x),
                (35, mkp.raw()),
            ],
            (groups, 1, 1),
        )
        .unwrap();
        let y_sg: Vec<u32> = dispatch::read_back(ctx, &buf_y_sg, y_len).unwrap();
        let got_sg = unpack_y(&y_sg, c);
        check_close(
            &format!("na vs wgsl-sg-mk n={} k={} m={}", c.n, c.k, c.m),
            &got_na,
            &got_sg,
            0.02,
        );
    } else {
        eprintln!("na parity: sg_mk unavailable, cpu reference only");
    }
}

#[test]
fn na_gemm_w4a16_pk_matches_wgsl_and_cpu() {
    let Some(ctx) = na_or_skip("na_gemm_w4a16_pk_matches_wgsl_and_cpu") else {
        return;
    };
    for c in [
        Case {
            n: 64,
            k: 64,
            m: 2,
            gs: 32,
            y_stride_words: 32,
            dst_word_off: 0,
        },
        Case {
            n: 200,
            k: 256,
            m: 10,
            gs: 32,
            y_stride_words: 128,
            dst_word_off: 5,
        },
        Case {
            n: 1024,
            k: 512,
            m: 16,
            gs: 128,
            y_stride_words: 512,
            dst_word_off: 0,
        },
        Case {
            n: 320,
            k: 2560,
            m: 10,
            gs: 32,
            y_stride_words: 160,
            dst_word_off: 0,
        },
    ] {
        run_case(ctx, &c);
    }
}

#[test]
fn na_gemm_w4a16_pk3_splits_match_pk() {
    let Some(ctx) = na_or_skip("na_gemm_w4a16_pk3_splits_match_pk") else {
        return;
    };
    let (q_rows, kv_rows, k, m, gs) = (128usize, 64usize, 256usize, 10usize, 32usize);
    let n = q_rows + 2 * kv_rows;
    let v_off = q_rows + kv_rows;
    let srs = k / gs;
    let row_words = k / 8;
    let mut rng = Lcg::new(0xabcd);
    let packed = rng.packed(n * row_words);
    let scales = rng.scales(n * srs);
    let x = rng.bf16_words(m * k, 1.0);
    let scale_words = gw::pack_scale_words(&scales);
    let x_words: Vec<u32> = x
        .chunks(2)
        .map(|p| (p[0] as u32) | ((p[1] as u32) << 16))
        .collect();

    let buf_w = dispatch::storage_from_slice(ctx, "na-t3-w", &packed);
    let buf_s = dispatch::storage_from_slice(ctx, "na-t3-s", &scale_words);
    let buf_x = dispatch::storage_from_slice(ctx, "na-t3-x", &x_words);
    let buf_yq = dispatch::storage_zeroed(ctx, "na-t3-yq", (m * q_rows / 2 * 4) as u64);
    let buf_yk = dispatch::storage_zeroed(ctx, "na-t3-yk", (m * kv_rows / 2 * 4) as u64);
    let buf_yv = dispatch::storage_zeroed(ctx, "na-t3-yv", (m * kv_rows / 2 * 4) as u64);
    let buf_y = dispatch::storage_zeroed(ctx, "na-t3-y", (m * n / 2 * 4) as u64);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct MkP {
        m: u32,
        x_stride_words: u32,
        y_stride_words: u32,
        dst_word_off: u32,
    }
    let mkp = GpuUniform::new(
        ctx,
        "na-t3-mk",
        &MkP {
            m: m as u32,
            x_stride_words: (k / 2) as u32,
            y_stride_words: (n / 2) as u32,
            dst_word_off: 0,
        },
    );
    let nap = GpuUniform::new(
        ctx,
        "na-t3-np",
        &na::NaStaticParams {
            n_rows: n as u32,
            k_elems: k as u32,
            gs: gs as u32,
            scale_row_stride: srs as u32,
            scale_elem_stride: 1,
            q_rows: q_rows as u32,
            kv_rows: kv_rows as u32,
            v_off: v_off as u32,
        },
    );

    let pk = na::pk_pipeline(ctx).unwrap();
    dispatch::dispatch(
        ctx,
        &pk,
        &[
            (0, &buf_w),
            (1, &buf_s),
            (2, &buf_x),
            (3, &buf_y),
            (4, mkp.raw()),
            (5, nap.raw()),
        ],
        (na::grid_x(n as u32), 1, 1),
    )
    .unwrap();
    let pk3 = na::pk3_pipeline(ctx).unwrap();
    dispatch::dispatch(
        ctx,
        &pk3,
        &[
            (0, &buf_w),
            (1, &buf_s),
            (2, &buf_x),
            (3, &buf_yq),
            (4, &buf_yk),
            (5, &buf_yv),
            (6, mkp.raw()),
            (7, nap.raw()),
        ],
        (na::grid_x(n as u32), 1, 1),
    )
    .unwrap();

    let y: Vec<u32> = dispatch::read_back(ctx, &buf_y, m * n / 2).unwrap();
    let yq: Vec<u32> = dispatch::read_back(ctx, &buf_yq, m * q_rows / 2).unwrap();
    let yk: Vec<u32> = dispatch::read_back(ctx, &buf_yk, m * kv_rows / 2).unwrap();
    let yv: Vec<u32> = dispatch::read_back(ctx, &buf_yv, m * kv_rows / 2).unwrap();
    for t in 0..m {
        for w in 0..q_rows / 2 {
            assert_eq!(
                yq[t * q_rows / 2 + w],
                y[t * n / 2 + w],
                "q word t={t} w={w}"
            );
        }
        for w in 0..kv_rows / 2 {
            assert_eq!(
                yk[t * kv_rows / 2 + w],
                y[t * n / 2 + q_rows / 2 + w],
                "k word t={t} w={w}"
            );
            assert_eq!(
                yv[t * kv_rows / 2 + w],
                y[t * n / 2 + v_off / 2 + w],
                "v word t={t} w={w}"
            );
        }
    }
    eprintln!("na_gemm_w4a16_pk3_splits_match_pk: split outputs bit-match packed");
}
