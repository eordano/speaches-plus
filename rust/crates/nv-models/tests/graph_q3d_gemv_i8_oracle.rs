#![cfg(feature = "wgpu")]

mod common;
use common::pack_bf16;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use common::LcgOddSeedShift33SignedUnitPacks as Lcg;
use common::unpack_bf16_bits;

const GROUPED: &str = "q3d_gemv_i8g";
const ROWSCALE: &str = "q3d_gemv_i8";

fn source() -> String {
    let src = nv_models::qwen3_5_dense_wgpu::shipped_gemv_i8_source();
    for e in [GROUPED, ROWSCALE] {
        assert!(
            src.contains(&format!("fn {e}(")),
            "the shipped int8 GEMV source no longer declares {e}; the entry moved and this gate \
             is now testing nothing"
        );
    }
    src
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Q3q8Params {
    n_rows: u32,
    k_elems: u32,
    groups_x: u32,
    groups_per_row: u32,
    group_shift: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

fn pack_i8(w: &[i8]) -> Vec<u32> {
    assert!(w.len().is_multiple_of(4), "int8 rows pack 4 per word");
    let mut out = vec![0u32; w.len() / 4];
    for (i, v) in w.iter().enumerate() {
        out[i / 4] |= ((*v as i32 as u32) & 0xff) << (8 * (i % 4));
    }
    out
}

struct Case {
    label: &'static str,
    entry: &'static str,
    n: usize,
    k: usize,

    group: usize,
    w: Vec<i8>,
    x: Vec<f32>,
    s: Vec<f32>,
}

impl Case {
    fn params(&self) -> Q3q8Params {
        Q3q8Params {
            n_rows: self.n as u32,
            k_elems: self.k as u32,
            groups_x: self.n.div_ceil(8) as u32,
            groups_per_row: self.k.checked_div(self.group).unwrap_or(1) as u32,
            group_shift: if self.group > 0 {
                (self.group / 4).trailing_zeros()
            } else {
                0
            },
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }
    }

    fn reference(&self) -> Vec<(f64, f64)> {
        let p = self.params();
        let gshift = p.group_shift;
        let gpr = p.groups_per_row as usize;
        let words = self.k / 4;
        let mut out = Vec::with_capacity(self.n);
        for r in 0..self.n {
            let mut acc = 0f64;
            let mut l1 = 0f64;
            for i in (0..words).rev() {
                let mut d = 0f64;
                for e in (0..4).rev() {
                    let j = 4 * i + e;
                    d += self.w[r * self.k + j] as f64 * self.x[j] as f64;
                }
                if self.group > 0 {
                    let s = self.s[r * gpr + (i >> gshift)] as f64;
                    acc += s * d;
                    l1 += (s * d).abs();
                } else {
                    acc += d;
                    l1 += d.abs();
                }
            }
            if self.group == 0 {
                let s = self.s[r] as f64;
                acc *= s;
                l1 *= s.abs();
            }
            out.push((acc, l1));
        }
        out
    }
}

fn exact_cases() -> Vec<Case> {
    let mut out = Vec::new();
    for (label, entry, n, k, group) in [
        (
            "exact grouped g16 n=24 k=64",
            GROUPED,
            24usize,
            64usize,
            16usize,
        ),
        ("exact grouped g32 n=17 k=64", GROUPED, 17, 64, 32),
        ("exact rowscale n=24 k=64", ROWSCALE, 24, 64, 0),
        ("exact rowscale n=13 k=32", ROWSCALE, 13, 32, 0),
    ] {
        let mut r = Lcg::new(0x1_8000 ^ (n as u64) << 20 ^ (k as u64) << 8 ^ group as u64);
        let w: Vec<i8> = (0..n * k)
            .map(|_| if r.next_u32() & 1 == 0 { -1i8 } else { 1 })
            .collect();
        let x: Vec<f32> = (0..k)
            .map(|_| if r.next_u32() & 1 == 0 { -1.0f32 } else { 1.0 })
            .collect();
        let ns = if group > 0 { n * (k / group) } else { n };
        let s: Vec<f32> = (0..ns)
            .map(|_| if r.next_u32() & 1 == 0 { 1.0f32 } else { 2.0 })
            .collect();
        out.push(Case {
            label,
            entry,
            n,
            k,
            group,
            w,
            x,
            s,
        });
    }
    out
}

fn random_cases() -> Vec<Case> {
    let mut out = Vec::new();
    for (label, entry, n, k, group) in [
        (
            "random grouped g32 n=48 k=256",
            GROUPED,
            48usize,
            256usize,
            32usize,
        ),
        ("random grouped g128 n=33 k=256", GROUPED, 33, 256, 128),
        ("random rowscale n=48 k=256", ROWSCALE, 48, 256, 0),
        ("random rowscale n=9 k=64", ROWSCALE, 9, 64, 0),
    ] {
        let mut r = Lcg::new(0xbeef_0000 ^ (n as u64) << 20 ^ (k as u64) << 8 ^ group as u64);
        let w: Vec<i8> = (0..n * k)
            .map(|_| ((r.next_u32() % 255) as i32 - 127) as i8)
            .collect();
        let x: Vec<f32> = (0..k)
            .map(|_| half::bf16::from_f32(r.next_f32() * 0.35).to_f32())
            .collect();
        let ns = if group > 0 { n * (k / group) } else { n };
        let s: Vec<f32> = (0..ns)
            .map(|_| 0.002 + 0.02 * (r.next_u32() % 1000) as f32 / 1000.0)
            .collect();
        out.push(Case {
            label,
            entry,
            n,
            k,
            group,
            w,
            x,
            s,
        });
    }
    out
}

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_q3d_gemv_i8_oracle needs a real wgpu adapter; a skipped numeric gate reads as a \
         passed one, so this panics rather than returning early",
    )
}

fn require_subgroup32(ctx: &WgpuContext) {
    assert!(
        nv_kernels::wgpu_backend::kernels::gemv_nvfp4_v2::subgroup32_ok(ctx),
        "adapter {} reports no 32-wide subgroups; both int8 GEMV entries reduce with \
         subgroupShuffleXor over exactly 32 lanes and w8_enabled() gates on the same predicate. \
         This suite cannot run here and must not report success.",
        ctx.info.name
    );
}

fn dispatch_case(ctx: &WgpuContext, src: &str, c: &Case) -> Vec<u16> {
    let p = c.params();
    let w_b = dispatch::storage_from_slice(ctx, "w", &pack_i8(&c.w));
    let s_b = dispatch::storage_from_slice(ctx, "s", &c.s);
    let x_b = dispatch::storage_from_slice(ctx, "x", &pack_bf16(&c.x));
    let y_words = c.n.div_ceil(2);
    let y_b = dispatch::storage_zeroed(ctx, "y", (y_words * 4) as u64);
    let p_b = dispatch::uniform_from(ctx, "p", &p);

    dispatch::run(
        ctx,
        "q3d-gemv-i8-oracle",
        src,
        c.entry,
        &[(0, &w_b), (1, &s_b), (2, &x_b), (3, &y_b), (4, &p_b)],
        (p.groups_x, 1, 1),
    )
    .unwrap_or_else(|e| panic!("{}: dispatch {}: {e}", c.label, c.entry));

    let words: Vec<u32> = dispatch::read_back(ctx, &y_b, y_words).expect("read back gemv output");
    unpack_bf16_bits(&words, c.n)
}

fn assert_scales_are_not_uniform(c: &Case) {
    let lo = c.s.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = c.s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        hi > lo * 1.5,
        "{}: scale corpus spans only {lo}..{hi}; with near-uniform scales, dropping or \
         mis-indexing the scale load is unobservable and this case gates nothing",
        c.label
    );
}

#[test]
fn both_int8_gemv_entries_are_bit_exact_on_integer_data() {
    let ctx = ctx();
    eprintln!("[q3d-gemv-i8-oracle] adapter: {}", ctx.info.name);
    require_subgroup32(ctx);
    let src = source();

    let mut covered: Vec<&str> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    for c in exact_cases() {
        assert_scales_are_not_uniform(&c);
        let want = c.reference();

        let ref_max = want.iter().fold(0f64, |a, (v, _)| a.max(v.abs()));
        let nonzero = want.iter().filter(|(v, _)| *v != 0.0).count();
        assert!(
            ref_max >= 4.0 && nonzero * 2 >= c.n,
            "{}: reference is degenerate (max |out| {ref_max}, {nonzero}/{} rows nonzero); a \
             comparison against it would pass on a kernel that writes constants",
            c.label,
            c.n
        );

        let got = dispatch_case(ctx, &src, &c);

        let mut mismatch: Vec<String> = Vec::new();
        for (r, (v, _)) in want.iter().enumerate() {
            let want_bits = half::bf16::from_f32(*v as f32).to_bits();
            if got[r] != want_bits {
                mismatch.push(format!(
                    "row {r}: got {:?} (0x{:04x}) want {v} (0x{want_bits:04x})",
                    half::bf16::from_bits(got[r]).to_f32(),
                    got[r]
                ));
            }
        }
        eprintln!(
            "[q3d-gemv-i8-oracle] {} ({}): {} rows, ref_max={ref_max}, {} mismatched",
            c.label,
            c.entry,
            c.n,
            mismatch.len()
        );
        if !mismatch.is_empty() {
            failures.push(format!(
                "{} [{}]: {} of {} rows wrong\n  {}",
                c.label,
                c.entry,
                mismatch.len(),
                c.n,
                mismatch.join("\n  ")
            ));
        }
        if !covered.contains(&c.entry) {
            covered.push(c.entry);
        }
    }
    assert_eq!(
        covered.len(),
        2,
        "this suite must exercise BOTH int8 entries; it covered {covered:?}. \
         `q3d_gemv_i8` is the one the sweep recorded as NOT-REACHED and is the whole reason \
         this file exists"
    );
    assert!(
        failures.is_empty(),
        "the shipped int8 GEMV disagrees with the f64 host reference on data where every partial \
         sum is an integer below 2^8. f32 accumulates such values exactly in any order and bf16 \
         stores them exactly, so there is no rounding freedom here and every one of these is a \
         defect:\n{}",
        failures.join("\n")
    );
}

#[test]
fn both_int8_gemv_entries_track_the_f64_reference_on_random_data() {
    let ctx = ctx();
    require_subgroup32(ctx);
    let src = source();

    let mut worst_ratio = 0f64;
    let mut failures: Vec<String> = Vec::new();
    for c in random_cases() {
        assert_scales_are_not_uniform(&c);
        let want = c.reference();
        let ref_max = want.iter().fold(0f64, |a, (v, _)| a.max(v.abs()));
        assert!(
            ref_max > 1e-3,
            "{}: reference output is degenerate (max |out| {ref_max:e})",
            c.label
        );

        let got_bits = dispatch_case(ctx, &src, &c);
        let got: Vec<f64> = got_bits
            .iter()
            .map(|b| half::bf16::from_bits(*b).to_f32() as f64)
            .collect();
        let got_max = got.iter().fold(0f64, |a, v| a.max(v.abs()));
        assert!(
            got_max > 1e-3,
            "{}: kernel output is all-but-zero (max |out| {got_max:e}); the comparison would be \
             zeros against zeros",
            c.label
        );

        let words = c.k / 4;
        let mut over: Vec<String> = Vec::new();
        for (r, (v, l1)) in want.iter().enumerate() {
            let bound =
                v.abs() * f64::powi(2.0, -8) + 8.0 * l1 * words as f64 * f64::powi(2.0, -24);
            let err = (got[r] - v).abs();
            let ratio = err / bound.max(f64::MIN_POSITIVE);
            worst_ratio = worst_ratio.max(ratio);
            if err > bound {
                over.push(format!(
                    "row {r}: got {} want {v} (err {err:e}, {ratio:.1}x the bound {bound:e})",
                    got[r]
                ));
            }
        }
        eprintln!(
            "[q3d-gemv-i8-oracle] {} ({}): {} rows, {} over bound, ref_max={ref_max:.5}",
            c.label,
            c.entry,
            c.n,
            over.len()
        );
        if !over.is_empty() {
            failures.push(format!(
                "{} [{}]: {} of {} rows over bound\n  {}",
                c.label,
                c.entry,
                over.len(),
                c.n,
                over.join("\n  ")
            ));
        }
    }
    eprintln!("[q3d-gemv-i8-oracle] worst err/bound across random cases: {worst_ratio:.3}");
    assert!(
        failures.is_empty(),
        "the shipped int8 GEMV diverged from the f64 host reference by more than one bf16 ulp \
         plus the f32 accumulation slack. The output is packed bf16, so that bound IS the \
         format's own resolution, not a tolerance chosen to pass:\n{}",
        failures.join("\n")
    );
}

#[test]
fn the_grouped_corpus_varies_its_scale_within_a_row() {
    let mut worst = 0f32;
    for c in exact_cases().into_iter().chain(random_cases()) {
        if c.group == 0 {
            continue;
        }
        let gpr = c.k / c.group;
        assert!(
            gpr > 1,
            "{}: one group per row is a rowscale in disguise",
            c.label
        );
        for r in 0..c.n {
            let row = &c.s[r * gpr..(r + 1) * gpr];
            let lo = row.iter().cloned().fold(f32::INFINITY, f32::min);
            let hi = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            worst = worst.max(hi / lo);
        }
    }
    eprintln!("[q3d-gemv-i8-oracle] worst within-row scale ratio: {worst}");
    assert!(
        worst > 1.5,
        "no row in the grouped corpus varies its scale by more than {worst}x; with a flat scale \
         per row the group index is unobservable and the grouped arm degenerates into the \
         rowscale one"
    );
}
