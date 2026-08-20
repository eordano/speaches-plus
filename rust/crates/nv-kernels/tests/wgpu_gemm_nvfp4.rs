#![cfg(feature = "wgpu")]

mod common;
use common::wgpu_allow_skip;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::gemm_nvfp4::{self, GemmPath};
use nv_quant::nvfp4::{swizzle_scales, Nvfp4Tensor};

fn ctx(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(c) if c.qualify().qualified => {
            eprintln!("{test}: {}", c.summary());
            Some(c)
        }
        Ok(c) => {
            if !wgpu_allow_skip() {
                panic!(
                    "{test}: wgpu adapter not qualified: {:?}. This gate refuses to report \
                     success without running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on \
                     purpose.",
                    c.qualify().reason
                );
            }
            eprintln!(
                "{test}: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) not qualified: {:?}",
                c.qualify().reason
            );
            None
        }
        Err(e) => {
            if !wgpu_allow_skip() {
                panic!(
                    "{test}: no wgpu adapter: {e}. This gate refuses to report success \
                     without running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
                );
            }
            eprintln!("{test}: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) no adapter: {e}");
            None
        }
    }
}

struct Case {
    a: Nvfp4Tensor,
    b: Nvfp4Tensor,
    a_sf: Vec<u8>,
    b_sf: Vec<u8>,
    m: usize,
    n: usize,
    k: usize,
}

fn make_case(m: usize, n: usize, k: usize, seed: f32) -> Case {
    let a_rows: Vec<Vec<f32>> = (0..m)
        .map(|i| {
            (0..k)
                .map(|j| (((i * k + j) as f32) * 0.07 + seed).sin())
                .collect()
        })
        .collect();
    let b_rows: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..k)
                .map(|j| (((i * k + j) as f32) * 0.09 + seed).cos() * 1.5)
                .collect()
        })
        .collect();
    let a = Nvfp4Tensor::quantize_rows(&a_rows);
    let b = Nvfp4Tensor::quantize_rows(&b_rows);
    let a_sf = swizzle_scales(&a.scales, m, k / 16);
    let b_sf = swizzle_scales(&b.scales, n, k / 16);
    Case {
        a,
        b,
        a_sf,
        b_sf,
        m,
        n,
        k,
    }
}

fn cpu_oracle(case: &Case, alpha: f32) -> Vec<u16> {
    let a = case.a.dequantize();
    let b = case.b.dequantize();
    let mut out = vec![0u16; case.m * case.n];
    for i in 0..case.m {
        for j in 0..case.n {
            let mut acc = 0f32;
            for p in 0..case.k {
                acc += a[i][p] * b[j][p];
            }
            out[i * case.n + j] = bf16::from_f32(acc * alpha).to_bits();
        }
    }
    out
}

fn run(ctx: &WgpuContext, case: &Case, alpha: f32, path: GemmPath) -> (Vec<u16>, GemmPath) {
    let mut out = vec![0u16; case.m * case.n];
    let used = gemm_nvfp4::nvfp4_gemm_bf16(
        ctx,
        &case.a.data,
        &case.a_sf,
        &case.b.data,
        &case.b_sf,
        alpha,
        &mut out,
        case.m,
        case.n,
        case.k,
        path,
    )
    .unwrap_or_else(|e| panic!("gemm {}: {e}", path.label()));
    (out, used)
}

fn report(name: &str, got: &[u16], want: &[u16]) -> (usize, i32, f64) {
    let mut mismatch = 0usize;
    let mut max_ulp = 0i32;
    let mut sq = 0f64;
    let mut ref_sq = 0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        if g != w {
            mismatch += 1;
            max_ulp = max_ulp.max((*g as i32 - *w as i32).abs());
        }
        let gv = bf16::from_bits(*g).to_f32() as f64;
        let wv = bf16::from_bits(*w).to_f32() as f64;
        sq += (gv - wv) * (gv - wv);
        ref_sq += wv * wv;
    }
    let rel = (sq / ref_sq.max(1e-12)).sqrt();
    eprintln!(
        "{name}: {mismatch}/{} bf16 words differ, max_ulp={max_ulp}, rel_rms={rel:.3e}",
        got.len()
    );
    (mismatch, max_ulp, rel)
}

#[test]
fn coop_matrix_capability_is_reported() {
    let Some(ctx) = ctx("coop_caps") else {
        return;
    };
    for c in &ctx.caps.coop_configs {
        eprintln!("coop_caps: {c:?}");
    }
    match ctx.caps.coop_gemm_tile() {
        Some(t) => {
            assert!(ctx.caps.cooperative_matrix);
            assert_eq!(ctx.caps.coop_gemm_reason(), None);
            eprintln!("coop_caps: gemm tile {t:?}");
        }
        None => eprintln!(
            "coop_caps: no coop gemm tile: {:?}",
            ctx.caps.coop_gemm_reason()
        ),
    }
}

#[test]
fn scalar_gemm_matches_the_cpu_oracle() {
    let Some(ctx) = ctx("scalar_gemm") else {
        return;
    };
    for (m, n, k) in [(16usize, 16usize, 16usize), (33, 47, 64), (128, 128, 128)] {
        let case = make_case(m, n, k, 0.3);
        let want = cpu_oracle(&case, 1.0);
        let (got, used) = run(ctx, &case, 1.0, GemmPath::Scalar);
        assert_eq!(used, GemmPath::Scalar);
        let (_mm, max_ulp, rel) = report(&format!("scalar {m}x{n}x{k}"), &got, &want);
        assert!(max_ulp <= 1, "scalar {m}x{n}x{k} max_ulp={max_ulp}");
        assert!(rel < 1e-3, "scalar {m}x{n}x{k} rel_rms={rel:e}");
    }
}

#[test]
fn scalar_gemm_applies_the_global_scale() {
    let Some(ctx) = ctx("scalar_gemm_alpha") else {
        return;
    };
    let case = make_case(32, 32, 64, 1.1);
    let want = cpu_oracle(&case, 0.375);
    let (got, _) = run(ctx, &case, 0.375, GemmPath::Scalar);
    let (_mm, max_ulp, rel) = report("scalar alpha=0.375", &got, &want);
    assert!(max_ulp <= 1);
    assert!(rel < 1e-3);
}

#[test]
fn coop_gemm_matches_the_cpu_oracle_and_the_scalar_path() {
    let Some(ctx) = ctx("coop_gemm") else {
        return;
    };
    if let Some(why) = ctx.caps.coop_gemm_reason() {
        if wgpu_allow_skip() {
            eprintln!("coop_gemm: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) coop_mat unavailable: {why}");
            return;
        }
        panic!(
            "coop_gemm: coop_mat unavailable: {why}. wgpu_coop_matrix_probe treats this same \
             capability as fatal, so this oracle comparison must not report success without \
             it. Set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
        );
    }
    eprintln!("coop_gemm: tile {:?}", ctx.caps.coop_gemm_tile());
    for (m, n, k) in [
        (16usize, 16usize, 16usize),
        (33, 47, 64),
        (128, 128, 128),
        (129, 65, 256),
    ] {
        let case = make_case(m, n, k, 0.3);
        let want = cpu_oracle(&case, 1.0);
        let (coop, used) = run(ctx, &case, 1.0, GemmPath::CoopMat);
        assert_eq!(used, GemmPath::CoopMat);
        let (_m0, ulp_cpu, rel_cpu) = report(&format!("coop {m}x{n}x{k} vs cpu"), &coop, &want);
        assert!(ulp_cpu <= 1, "coop {m}x{n}x{k} vs cpu max_ulp={ulp_cpu}");
        assert!(
            rel_cpu < 1e-3,
            "coop {m}x{n}x{k} vs cpu rel_rms={rel_cpu:e}"
        );

        let (scalar, _) = run(ctx, &case, 1.0, GemmPath::Scalar);
        let (_m1, ulp_s, rel_s) = report(&format!("coop {m}x{n}x{k} vs scalar"), &coop, &scalar);
        assert!(ulp_s <= 1, "coop vs scalar {m}x{n}x{k} max_ulp={ulp_s}");
        assert!(rel_s < 1e-3, "coop vs scalar {m}x{n}x{k} rel_rms={rel_s:e}");
    }
}

#[test]
fn auto_picks_coop_when_the_device_reports_a_tile() {
    let Some(ctx) = ctx("auto_path") else {
        return;
    };
    if std::env::var("NV_KERNELS_WGPU_GEMM").is_ok() {
        if wgpu_allow_skip() {
            eprintln!(
                "auto_path: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) NV_KERNELS_WGPU_GEMM \
                 overrides the path this test asserts"
            );
            return;
        }
        panic!(
            "auto_path: NV_KERNELS_WGPU_GEMM is set, which overrides the dispatch choice \
             this test asserts -- the assertion would be voided while still reporting a \
             pass. Unset it, or set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
        );
    }
    let want = if ctx.caps.coop_gemm_tile().is_some() {
        GemmPath::CoopMat
    } else {
        GemmPath::Scalar
    };
    let case = make_case(16, 16, 32, 0.9);
    let (_out, used) = run(ctx, &case, 1.0, GemmPath::Auto);
    eprintln!("auto_path: executed {}", used.label());
    assert_eq!(used, want);
}

#[test]
#[ignore]
fn bench_coop_vs_scalar() {
    let Some(ctx) = ctx("bench") else {
        return;
    };
    let mut paths = vec![GemmPath::Scalar];
    if ctx.caps.coop_gemm_tile().is_some() {
        paths.push(GemmPath::CoopMat);
    } else {
        eprintln!("bench: coop unavailable: {:?}", ctx.caps.coop_gemm_reason());
    }
    let shapes = [
        (512usize, 512usize, 512usize),
        (1024, 1024, 1024),
        (1024, 1024, 4096),
        (2048, 2048, 1024),
    ];
    let iters = 5;
    for (m, n, k) in shapes {
        let case = make_case(m, n, k, 0.5);
        let moved = (m * k / 2 + n * k / 2 + m * n * 4) as f64;
        for p in paths.iter().copied() {
            if p == GemmPath::Scalar && m * n * k > 1 << 30 {
                continue;
            }
            let (_warm, used) = run(ctx, &case, 1.0, p);
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = run(ctx, &case, 1.0, p);
            }
            let per = t0.elapsed().as_secs_f64() / iters as f64;
            let flops = 2.0 * (m * n * k) as f64;
            eprintln!(
                "bench {m}x{n}x{k} {:>8}: {:7.2} ms/call, {:7.1} GFLOP/s, {:5.2} MB moved, {:5.2} GB/s effective",
                used.label(),
                per * 1e3,
                flops / per / 1e9,
                moved / 1e6,
                moved / per / 1e9
            );
        }
    }
}

#[test]
fn shape_errors_are_reported() {
    let Some(ctx) = ctx("shape_errors") else {
        return;
    };
    let case = make_case(16, 16, 16, 0.0);
    let mut out = vec![0u16; 16 * 16];
    let err = gemm_nvfp4::nvfp4_gemm_bf16(
        ctx,
        &case.a.data,
        &case.a_sf,
        &case.b.data,
        &case.b_sf,
        1.0,
        &mut out,
        16,
        16,
        24,
        GemmPath::Scalar,
    )
    .expect_err("K=24 is not a multiple of 16");
    assert!(format!("{err}").contains("not a multiple of 16"));
}
