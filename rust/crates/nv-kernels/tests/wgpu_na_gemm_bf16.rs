#![cfg(feature = "wgpu")]

mod common;
use common::wgpu_allow_skip;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::na;
use nv_kernels::wgpu_backend::na_bf16;

fn ctx_or_skip(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("{test}: {}", ctx.summary());
            Some(ctx)
        }
        Err(e) => {
            if !wgpu_allow_skip() {
                panic!(
                    "{test}: no wgpu adapter: {e}. This gate refuses to report success \
                     without running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
                );
            }
            eprintln!("{test}: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) no wgpu adapter: {e}");
            None
        }
    }
}

struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / 2147483648.0) - 1.0
    }
    fn bf16s(&mut self, n: usize, gain: f32) -> Vec<u16> {
        (0..n)
            .map(|_| bf16::from_f32(self.next_f32() * gain).to_bits())
            .collect()
    }
}

fn pack_u16(src: &[u16]) -> Vec<u32> {
    let mut out = vec![0u32; src.len().div_ceil(2)];
    for (i, v) in src.iter().enumerate() {
        out[i / 2] |= (*v as u32) << (16 * (i % 2));
    }
    out
}

fn bv(b: u16) -> f64 {
    f64::from(f32::from_bits((b as u32) << 16))
}

fn run_case(ctx: &'static WgpuContext, n: usize, k: usize, m_alloc: usize, m_live: usize) {
    let mut rng = Lcg(0x9e3779b97f4a7c15 ^ ((n * 31 + k * 7 + m_live) as u64));
    let w = rng.bf16s(n * k, 1.0);
    let x = rng.bf16s(m_alloc * k, 0.5);
    let x_stride_words = k / 2;
    let y_stride_words = n.div_ceil(2);

    let wb = dispatch::storage_from_slice(ctx, "nab-w", &pack_u16(&w));
    let xb = dispatch::storage_from_slice(ctx, "nab-x", &pack_u16(&x));
    let yb = dispatch::storage_zeroed(ctx, "nab-y", (m_alloc * y_stride_words * 4) as u64);
    let np = dispatch::uniform_from(
        ctx,
        "nab-np",
        &na_bf16::NaBf16Params {
            n_rows: n as u32,
            k_elems: k as u32,
            x_stride_words: x_stride_words as u32,
            y_stride_words: y_stride_words as u32,
            dst_word_off: 0,
            ..Default::default()
        },
    );
    let lp = dispatch::uniform_from(
        ctx,
        "nab-lp",
        &na_bf16::NaLiveParams {
            m_live: m_live as u32,
            base: 0,
            ..Default::default()
        },
    );
    let pipeline = na_bf16::pipeline(ctx).expect("na bf16 pipeline");
    dispatch::dispatch(
        ctx,
        &pipeline,
        &[(0, &wb), (1, &xb), (2, &yb), (3, &np), (4, &lp)],
        (na_bf16::grid_x(n as u32), 1, 1),
    )
    .expect("dispatch");
    let words: Vec<u32> = dispatch::read_back(ctx, &yb, m_alloc * y_stride_words).expect("read");

    let mut max_rel = 0f64;
    for t in 0..m_live {
        for r in 0..n {
            let mut acc = 0f64;
            for c in 0..k {
                acc += bv(w[r * k + c]) * bv(x[t * k + c]);
            }
            let word = words[t * y_stride_words + r / 2];
            let got = bv(if r % 2 == 0 {
                (word & 0xffff) as u16
            } else {
                (word >> 16) as u16
            });
            let denom = acc.abs().max(1e-3);
            max_rel = max_rel.max((got - acc).abs() / denom);
        }
    }
    for t in m_live..m_alloc {
        for wd in 0..y_stride_words {
            assert_eq!(
                words[t * y_stride_words + wd],
                0,
                "dead row {t} word {wd} written (m_live {m_live})"
            );
        }
    }
    eprintln!("na_gemm_bf16 n={n} k={k} m_live={m_live}/{m_alloc}: max rel err {max_rel:.2e}");
    assert!(
        max_rel < 0.02,
        "na_gemm_bf16 n={n} k={k} m={m_live}: rel err {max_rel} over 2% gate"
    );
}

#[test]
fn na_gemm_bf16_matches_cpu_reference() {
    let Some(ctx) = ctx_or_skip("na_gemm_bf16_matches_cpu_reference") else {
        return;
    };
    if !na_bf16::available(ctx) {
        if wgpu_allow_skip() {
            eprintln!(
                "na_gemm_bf16_matches_cpu_reference: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) na \
                 tensor-ops unavailable"
            );
            return;
        }
        if ctx.info.backend != wgpu::Backend::Metal {
            eprintln!(
                "na_gemm_bf16_matches_cpu_reference: SKIP row not run: na tensor-ops are \
                 MSL-only (metal_tensor + MetalPerformancePrimitives) and this adapter backend \
                 is {:?}, not Metal, so the Metal-passthrough property is out of scope on this \
                 box; a Metal adapter still panics loudly if na is unsupported there",
                ctx.info.backend
            );
            return;
        }
        if !na::supported(ctx) {
            panic!(
                "na_gemm_bf16_matches_cpu_reference: this IS a Metal adapter and \
                 na::supported() is still false -- PASSTHROUGH_SHADERS is missing on a Metal \
                 backend, which is the property under test failing, not an out-of-scope box."
            );
        }
        panic!(
            "na_gemm_bf16_matches_cpu_reference: this IS a Metal backend, yet \
             na_bf16::available() is false -- the na bf16 pipelines FAILED TO COMPILE, the very \
             thing this test scores. Set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
        );
    }
    for &(n, k, m_alloc, m_live) in &[
        (256usize, 512usize, 16usize, 16usize),
        (256, 512, 16, 1),
        (256, 512, 16, 5),
        (1024, 4096, 16, 16),
        (64, 4096, 16, 16),
        (254, 512, 16, 3),
    ] {
        run_case(ctx, n, k, m_alloc, m_live);
    }
}
