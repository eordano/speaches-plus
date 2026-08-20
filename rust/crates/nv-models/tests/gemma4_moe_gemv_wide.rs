
#![cfg(feature = "wgpu")]

mod common;
use common::bf16_bits;
use common::bf16_val;
use common::Rng;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::WgpuContext;
use nv_models::gemma4_moe_wgpu::{bench, gemv_wide_enabled, GemvBf16Params};

const SHAPES: &[(&str, usize, usize)] = &[
    ("attn q  (sliding)", 4096, 2816),
    ("attn k  (sliding)", 2048, 2816),
    ("attn o  (sliding)", 2816, 4096),
    ("attn q  (full)", 8192, 2816),
    ("attn k  (full)", 1024, 2816),
    ("attn o  (full)", 2816, 8192),
    ("mlp gate", 2112, 2816),
    ("mlp down", 2816, 2112),
    ("lm_head (tied embed)", 262144, 2816),
];

const MAX_ROWS: usize = 512;

fn pack_pairs(src: &[u16]) -> Vec<u32> {
    let words = src.len().div_ceil(2).max(1);
    let mut out = vec![0u32; words.next_multiple_of(4)];
    for (i, v) in src.iter().enumerate() {
        out[i / 2] |= (*v as u32) << (16 * (i % 2));
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Path {

    Scalar,

    Wide,

    Legacy,
}

fn gemv_raw(
    ctx: &WgpuContext,
    path: Path,
    w: &[u16],
    n: usize,
    k: usize,
    x: &[u16],
    out_f32: bool,
) -> Vec<u32> {
    assert_eq!(w.len(), n * k, "weight is not n*k");
    assert_eq!(x.len(), k, "activation is not k");
    let (src, entry) = match path {
        Path::Legacy => (
            bench::gemv_bf16_legacy_source(),
            bench::GEMV_BF16_LEGACY_ENTRY,
        ),
        _ => (bench::gemv_bf16_source(), bench::GEMV_BF16_ENTRY),
    };
    let wbuf = dispatch::storage_from_slice(ctx, "wideab-w", &pack_pairs(w));
    let xbuf = dispatch::storage_from_slice(ctx, "wideab-x", &pack_pairs(x));
    let out_words = if out_f32 { n } else { n.div_ceil(2) };
    let ybuf = dispatch::storage_zeroed(ctx, "wideab-y", (out_words * 4).max(16) as u64);
    let grid = dispatch::workgroup_count_1d(ctx, n.div_ceil(2) as u64, 1);
    let p = dispatch::uniform_from(
        ctx,
        "wideab-p",
        &GemvBf16Params {
            n_rows: n as u32,
            k_words: (k / 2) as u32,
            groups_x: grid.0,
            out_f32: u32::from(out_f32),
            w_row_words: (k / 2) as u32,
            x_off_words: 0,
            y_off_words: 0,
            wide: u32::from(path == Path::Wide),
            alpha: 1.0,
            ..Default::default()
        },
    );
    dispatch::run(
        ctx,
        "wideab",
        &src,
        entry,
        &[(0, &wbuf), (1, &xbuf), (2, &p), (3, &ybuf)],
        grid,
    )
    .unwrap_or_else(|e| panic!("dispatch {entry}: {e}"));
    dispatch::read_back(ctx, &ybuf, out_words).expect("read y")
}

fn gemv(ctx: &WgpuContext, w: &[u16], n: usize, k: usize, x: &[u16], wide: u32) -> Vec<f32> {
    let path = if wide == 1 { Path::Wide } else { Path::Scalar };
    gemv_raw(ctx, path, w, n, k, x, true)
        .into_iter()
        .map(f32::from_bits)
        .collect()
}

fn gemv_bf16_out(
    ctx: &WgpuContext,
    path: Path,
    w: &[u16],
    n: usize,
    k: usize,
    x: &[u16],
) -> Vec<f32> {
    gemv_raw(ctx, path, w, n, k, x, false)
        .into_iter()
        .flat_map(|word| {
            [
                bf16_val((word & 0xffff) as u16),
                bf16_val((word >> 16) as u16),
            ]
        })
        .take(n)
        .collect()
}

fn oracle(w: &[u16], n: usize, k: usize, x: &[u16]) -> Vec<f64> {
    (0..n)
        .map(|r| {
            (0..k)
                .map(|j| bf16_val(w[r * k + j]) as f64 * bf16_val(x[j]) as f64)
                .sum()
        })
        .collect()
}

fn assert_matches(label: &str, got: &[f32], reference: &[f64], tol: f64) -> f64 {
    let mag = reference.iter().fold(0f64, |a, b| a.max(b.abs()));
    assert!(
        mag > 1e-6,
        "{label}: the f64 reference is degenerate (max |ref| {mag:.3e}); this comparison \
         would pass on zeros"
    );
    let out_mag = got.iter().fold(0f64, |a, b| a.max(b.abs() as f64));
    assert!(
        out_mag > 1e-6,
        "{label}: the kernel wrote nothing but zeros (max |y| {out_mag:.3e})"
    );
    let mut worst = 0f64;
    let mut worst_at = 0usize;
    for (r, (g, rf)) in got.iter().zip(reference.iter()).enumerate() {
        let rel = (*g as f64 - *rf).abs() / mag;
        if rel > worst {
            worst = rel;
            worst_at = r;
        }
    }
    assert!(
        worst < tol,
        "{label}: row {worst_at} is {worst:.3e} off the f64 reference (tolerance {tol:.1e}); \
         this kernel does not compute the dot product it claims to"
    );
    worst
}

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect("this suite needs a wgpu adapter; there is no skip path")
}

fn fixture(seed: u64, n: usize, k: usize) -> (Vec<u16>, Vec<u16>) {
    let mut rw = Rng(seed);
    let w: Vec<u16> = (0..n * k).map(|_| bf16_bits(rw.next_f32() * 0.1)).collect();
    let mut rx = Rng(seed ^ 0x9e37_79b9_7f4a_7c15);
    let x: Vec<u16> = (0..k).map(|_| bf16_bits(rx.next_f32())).collect();
    (w, x)
}

const TOL: f64 = 2e-3;

#[test]
fn both_dense_gemv_loads_match_an_f64_oracle_at_every_26b_shape() {
    let ctx = ctx();
    eprintln!("[wide] adapter: {}", ctx.info.name);
    let mut worst_gap = 0f64;
    let mut worst_gap_at = "";
    let mut identical = 0usize;
    for &(name, n_full, k) in SHAPES {
        assert!(
            k.is_multiple_of(8),
            "{name}: k={k} cannot take the wide load; the 26B has no such shape and this \
             table has drifted from the checkpoint"
        );
        let n = n_full.min(MAX_ROWS);
        let (w, x) = fixture(0x51ed_270b_ee98_1c3f ^ (k as u64) << 20 ^ n as u64, n, k);
        let reference = oracle(&w, n, k, &x);

        let narrow = gemv(ctx, &w, n, k, &x, 0);
        let wide = gemv(ctx, &w, n, k, &x, 1);
        let e_narrow = assert_matches(&format!("{name} scalar"), &narrow, &reference, TOL);
        let e_wide = assert_matches(&format!("{name} wide"), &wide, &reference, TOL);

        let mag = reference.iter().fold(0f64, |a, b| a.max(b.abs()));
        let gap = narrow
            .iter()
            .zip(wide.iter())
            .fold(0f64, |a, (s, v)| a.max((*s as f64 - *v as f64).abs()))
            / mag;
        let bits = narrow
            .iter()
            .zip(wide.iter())
            .filter(|(s, v)| s.to_bits() != v.to_bits())
            .count();
        if bits == 0 {
            identical += 1;
        }
        if gap > worst_gap {
            worst_gap = gap;
            worst_gap_at = name;
        }
        eprintln!(
            "[wide] {name:<22} [{n},{k}]  scalar {e_narrow:.2e}  wide {e_wide:.2e}  \
             scalar-vs-wide {gap:.2e} over {bits}/{n} rows differing"
        );
    }
    eprintln!(
        "[wide] worst scalar-vs-wide disagreement {worst_gap:.3e} relative, at {worst_gap_at}; \
         both paths inside {TOL:.0e} of the f64 oracle at every shape"
    );
    assert!(
        worst_gap < 1e-4,
        "the two loads disagree by {worst_gap:.3e} relative at {worst_gap_at}; that is far \
         past f32 reassociation and means one of them is wrong"
    );
    assert_eq!(
        identical,
        0,
        "{identical} of {} shapes produced bit-identical output from the two loads -- the \
         `wide` uniform is not reaching the kernel and this suite is grading one path twice",
        SHAPES.len()
    );
}

#[test]
fn the_wide_load_holds_over_512_fixtures() {
    let ctx = ctx();
    const N: usize = 64;
    const K: usize = 2112;
    let mut worst_ref = 0f64;
    let mut worst_gap = 0f64;
    let mut all_identical = true;
    for f in 0..512u64 {
        let (w, x) = fixture(0xC0FF_EE00_0000_0000 ^ f.wrapping_mul(0x9e37_79b9), N, K);
        let reference = oracle(&w, N, K, &x);
        let wide = gemv(ctx, &w, N, K, &x, 1);
        worst_ref = worst_ref.max(assert_matches(
            &format!("fixture {f} wide"),
            &wide,
            &reference,
            TOL,
        ));
        if f % 64 == 0 {
            let narrow = gemv(ctx, &w, N, K, &x, 0);
            assert_matches(&format!("fixture {f} scalar"), &narrow, &reference, TOL);
            let mag = reference.iter().fold(0f64, |a, b| a.max(b.abs()));
            worst_gap = worst_gap.max(
                narrow
                    .iter()
                    .zip(wide.iter())
                    .fold(0f64, |a, (s, v)| a.max((*s as f64 - *v as f64).abs()))
                    / mag,
            );
            if narrow
                .iter()
                .zip(wide.iter())
                .any(|(s, v)| s.to_bits() != v.to_bits())
            {
                all_identical = false;
            }
        }
    }
    let mut flips = 0usize;
    let mut words = 0usize;
    for f in 0..512u64 {
        let (w, x) = fixture(0x51F7_0000_0000_0000 ^ f.wrapping_mul(0x9e37_79b9), N, K);
        let s = gemv_raw(ctx, Path::Scalar, &w, N, K, &x, false);
        let v = gemv_raw(ctx, Path::Wide, &w, N, K, &x, false);
        words += s.len() * 2;
        flips += s
            .iter()
            .zip(v.iter())
            .map(|(a, b)| usize::from(a & 0xffff != b & 0xffff) + usize::from(a >> 16 != b >> 16))
            .sum::<usize>();
    }
    eprintln!(
        "[wide] 512 fixtures at [{N},{K}]: worst wide-vs-f64 {worst_ref:.3e}, worst \
         wide-vs-scalar {worst_gap:.3e}; PACKED-BF16 outputs differ in {flips}/{words} words \
         ({:.2e} per word). A token issues ~8.1e5 dense-GEMV output words, so the wide load \
         flips of order {:.0} bf16 activations per forward pass -- that, not the 1e-7 dot \
         error, is what the routers downstream see.",
        flips as f64 / words as f64,
        flips as f64 / words as f64 * 8.09e5
    );
    assert!(
        flips > 0,
        "no packed-bf16 output word differed between the two loads over {words} words; \
         either the wide uniform is not reaching the kernel or this sample is too small to \
         say anything about what the graph sees"
    );
    assert!(
        !all_identical,
        "every sampled fixture gave bit-identical output from both loads; the `wide` uniform \
         is not reaching the kernel"
    );
}

#[test]
fn the_default_scalar_path_is_bit_identical_to_the_kernel_it_replaced() {
    let ctx = ctx();
    let mut checked = 0usize;
    let mut wide_differs = 0usize;
    for &(name, n_full, k) in SHAPES {
        let n = n_full.min(MAX_ROWS);
        let (w, x) = fixture(0x0BAD_5EED_0000_0001 ^ (k as u64) << 20 ^ n as u64, n, k);
        for out_f32 in [true, false] {
            let legacy = gemv_raw(ctx, Path::Legacy, &w, n, k, &x, out_f32);
            let shipped = gemv_raw(ctx, Path::Scalar, &w, n, k, &x, out_f32);
            let wide = gemv_raw(ctx, Path::Wide, &w, n, k, &x, out_f32);
            assert!(
                legacy.iter().any(|v| *v != 0),
                "{name} out_f32={out_f32}: the frozen kernel wrote nothing but zeros, so this \
                 comparison would pass on zeros"
            );
            let diff = legacy
                .iter()
                .zip(shipped.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                diff,
                0,
                "{name} out_f32={out_f32}: the shipped scalar branch differs from the frozen \
                 pre-change kernel in {diff} of {} output words -- the default path moved, \
                 and it was advertised as unchanged",
                legacy.len()
            );
            if wide.iter().zip(legacy.iter()).any(|(a, b)| a != b) {
                wide_differs += 1;
            }
            checked += 1;
        }
    }
    eprintln!(
        "[wide] {checked} (shape, epilogue) pairs: the shipped scalar branch is bit-identical \
         to the frozen pre-change kernel; the wide branch differs on {wide_differs} of them"
    );
    assert!(
        wide_differs >= checked / 2,
        "the wide branch matched the frozen kernel on {} of {checked} comparisons; this test \
         is not able to see a difference and its bit-identity claim is empty",
        checked - wide_differs
    );
}

#[test]
fn the_packed_bf16_epilogue_matches_an_f64_oracle_on_both_loads() {
    let ctx = ctx();
    const BF16_TOL: f64 = 6e-3;
    for &(name, n_full, k) in SHAPES {
        let n = n_full.min(MAX_ROWS);
        let (w, x) = fixture(0x00D5_1CE0_0000_0007 ^ (k as u64) << 20 ^ n as u64, n, k);
        let reference = oracle(&w, n, k, &x);
        for path in [Path::Scalar, Path::Wide] {
            let got = gemv_bf16_out(ctx, path, &w, n, k, &x);
            assert_matches(
                &format!("{name} packed {path:?}"),
                &got,
                &reference,
                BF16_TOL,
            );
        }
    }
    eprintln!("[wide] packed-bf16 epilogue verified against the f64 oracle on both loads");
}

#[test]
fn the_wide_load_is_off_unless_it_is_asked_for() {
    assert!(
        std::env::var("NV_G4MOE_GEMV_WIDE").is_err(),
        "NV_G4MOE_GEMV_WIDE is set on this runner, so the shipped default cannot be observed \
         here; unset it rather than letting this test grade the runner's opinion"
    );
    for label in [
        "g4m-at-qproj",
        "g4m-at-kproj",
        "g4m-at-vproj",
        "g4m-at-oproj",
        "g4m-mlp-gate",
        "g4m-mlp-up",
        "g4m-mlp-down",
        "g4m-lmhead",
    ] {
        assert!(
            !gemv_wide_enabled(label),
            "{label} takes the wide dense load by default; it is a 96.35%-greedy-agreement \
             change against the shipped kernel and must be reached through \
             NV_G4MOE_GEMV_WIDE=1"
        );
    }
    eprintln!("[wide] default is the scalar load at every dense dispatch class");
}

#[test]
fn the_wide_load_is_wrong_at_a_k_the_guard_refuses() {
    let ctx = ctx();
    const N: usize = 64;
    const K: usize = 2116;
    assert!(!K.is_multiple_of(8), "the whole point of this k");
    let (w, x) = fixture(0xBADC_0FFE_E0DD_F00D, N, K);
    let reference = oracle(&w, N, K, &x);
    let narrow = gemv(ctx, &w, N, K, &x, 0);
    assert_matches("guard k scalar", &narrow, &reference, TOL);

    let wide = gemv(ctx, &w, N, K, &x, 1);
    let mag = reference.iter().fold(0f64, |a, b| a.max(b.abs()));
    let worst = wide
        .iter()
        .zip(reference.iter())
        .fold(0f64, |a, (g, r)| a.max((*g as f64 - *r).abs() / mag));
    eprintln!(
        "[wide] guard control: forcing the wide load at k={K} reads {worst:.3e} off the f64 \
         oracle (the scalar load is exact there)"
    );
    assert!(
        worst > TOL,
        "forcing the wide load at k={K} still matched the oracle to {worst:.3e}; the \
         alignment guard in the graph is then untested by this suite and may be wrong"
    );
}
