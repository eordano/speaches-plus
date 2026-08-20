#![cfg(feature = "wgpu")]

mod common;
use common::pack_bf16;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use common::LcgOddSeedShift33SignedUnitPacks as Lcg;

const WIDTHS: [usize; 2] = [16, 3];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmMkParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    w_row_words: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    pad0: u32,
    pad1: u32,
}

struct Case {
    label: String,
    m: usize,
    n: usize,
    k: usize,

    w: Vec<f32>,

    x: Vec<f32>,
}

impl Case {
    fn params(&self) -> GemmMkParams {
        let k_words = (self.k / 2) as u32;
        GemmMkParams {
            n_rows: self.n as u32,
            k_words,
            groups_x: self.n.div_ceil(2) as u32,
            w_row_words: k_words,
            x_stride_words: k_words,
            y_stride_words: self.n.div_ceil(2) as u32,
            pad0: 0,
            pad1: 0,
        }
    }

    fn reference(&self) -> Vec<(f64, f64)> {
        let mut out = Vec::with_capacity(self.m * self.n);
        for mi in 0..self.m {
            for r in 0..self.n {
                let mut acc = 0f64;
                let mut l1 = 0f64;
                for j in (0..self.k).rev() {
                    let t = self.w[r * self.k + j] as f64 * self.x[mi * self.k + j] as f64;
                    acc += t;
                    l1 += t.abs();
                }
                out.push((acc, l1));
            }
        }
        out
    }
}

fn exact_cases() -> Vec<Case> {
    let mut out = Vec::new();
    for m in WIDTHS {
        for (n, k) in [(24usize, 64usize), (9, 64)] {
            let mut r = Lcg::new(0x9e11_0000 ^ (m as u64) << 24 ^ (n as u64) << 8 ^ k as u64);
            let pm1 = |r: &mut Lcg| if r.next_u32() & 1 == 0 { -1.0f32 } else { 1.0 };
            out.push(Case {
                label: format!("exact m={m} n={n} k={k}"),
                m,
                n,
                k,
                w: (0..n * k).map(|_| pm1(&mut r)).collect(),
                x: (0..m * k).map(|_| pm1(&mut r)).collect(),
            });
        }
    }
    out
}

fn random_cases() -> Vec<Case> {
    let mut out = Vec::new();
    for m in WIDTHS {
        for (n, k) in [(48usize, 256usize), (7, 128)] {
            let mut r = Lcg::new(0x51de_0000 ^ (m as u64) << 24 ^ (n as u64) << 8 ^ k as u64);
            let v = |r: &mut Lcg| half::bf16::from_f32(r.next_f32() * 0.4).to_f32();
            out.push(Case {
                label: format!("random m={m} n={n} k={k}"),
                m,
                n,
                k,
                w: (0..n * k).map(|_| v(&mut r)).collect(),
                x: (0..m * k).map(|_| v(&mut r)).collect(),
            });
        }
    }
    out
}

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_q3d_prefill_gemm_oracle needs a real wgpu adapter; a skipped numeric gate reads as \
         a passed one, so this panics rather than returning early",
    )
}

fn source_for(m: usize) -> (String, String) {
    let (src, entry) = nv_models::qwen3_5_dense_wgpu::shipped_prefill_gemm(m);
    assert!(
        src.contains(&format!("fn {entry}(")),
        "the generated prefill GEMM source for m={m} does not declare {entry}; the generator and \
         the entry-name helper disagree and this gate would be testing nothing"
    );
    assert!(
        src.contains(&format!("mi < {m}u")),
        "the generated prefill GEMM source for m={m} does not bake {m} into its row loop; the \
         width is no longer a compile-time literal and this gate's per-width corpus is moot"
    );
    (src, entry)
}

fn dispatch_case(ctx: &WgpuContext, c: &Case) -> Vec<f32> {
    let p = c.params();
    let (src, entry) = source_for(c.m);
    let w_b = dispatch::storage_from_slice(ctx, "w", &pack_bf16(&c.w));
    let x_b = dispatch::storage_from_slice(ctx, "x", &pack_bf16(&c.x));
    let y_words = c.m * p.y_stride_words as usize;
    let y_b = dispatch::storage_zeroed(ctx, "y", (y_words * 4) as u64);
    let p_b = dispatch::uniform_from(ctx, "p", &p);

    dispatch::run(
        ctx,
        "q3d-pf-gemm-oracle",
        &src,
        &entry,
        &[(0, &w_b), (1, &x_b), (2, &p_b), (3, &y_b)],
        (p.groups_x, 1, 1),
    )
    .unwrap_or_else(|e| panic!("{}: dispatch {entry}: {e}", c.label));

    let words: Vec<u32> = dispatch::read_back(ctx, &y_b, y_words).expect("read back gemm output");
    let mut out = Vec::with_capacity(c.m * c.n);
    for mi in 0..c.m {
        for r in 0..c.n {
            let word = words[mi * p.y_stride_words as usize + r / 2];
            let bits = if r.is_multiple_of(2) {
                (word & 0xffff) as u16
            } else {
                (word >> 16) as u16
            };
            out.push(half::bf16::from_bits(bits).to_f32());
        }
    }
    out
}

#[test]
fn the_prefill_gemm_is_bit_exact_on_integer_data() {
    let ctx = ctx();
    eprintln!("[q3d-pf-gemm-oracle] adapter: {}", ctx.info.name);

    let mut failures: Vec<String> = Vec::new();
    for c in exact_cases() {
        let want = c.reference();
        let ref_max = want.iter().fold(0f64, |a, (v, _)| a.max(v.abs()));
        let nonzero = want.iter().filter(|(v, _)| *v != 0.0).count();
        assert!(
            ref_max >= 4.0 && nonzero * 2 >= want.len(),
            "{}: reference is degenerate (max |out| {ref_max}, {nonzero}/{} entries nonzero); a \
             comparison against it would pass on a kernel that writes constants",
            c.label,
            want.len()
        );

        let got = dispatch_case(ctx, &c);
        let mut mismatch: Vec<String> = Vec::new();
        for (i, (v, _)) in want.iter().enumerate() {
            let want_bits = half::bf16::from_f32(*v as f32).to_bits();
            if half::bf16::from_f32(got[i]).to_bits() != want_bits {
                mismatch.push(format!(
                    "m-row {} row {}: got {} want {v}",
                    i / c.n,
                    i % c.n,
                    got[i]
                ));
            }
        }
        eprintln!(
            "[q3d-pf-gemm-oracle] {}: {} entries, ref_max={ref_max}, {} mismatched",
            c.label,
            want.len(),
            mismatch.len()
        );
        if !mismatch.is_empty() {
            failures.push(format!(
                "{}: {} of {} entries wrong\n  {}",
                c.label,
                mismatch.len(),
                want.len(),
                mismatch
                    .iter()
                    .take(24)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n  ")
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "the shipped M-row prefill GEMM disagrees with the f64 host reference on data where every \
         partial sum is an integer below 2^8. f32 accumulates such values exactly in any order \
         and bf16 stores them exactly, so there is no rounding freedom here and every one of \
         these is a defect:\n{}",
        failures.join("\n")
    );
}

#[test]
fn the_prefill_gemm_tracks_the_f64_reference_on_random_data() {
    let ctx = ctx();
    let mut worst_ratio = 0f64;
    let mut failures: Vec<String> = Vec::new();
    for c in random_cases() {
        let want = c.reference();
        let ref_max = want.iter().fold(0f64, |a, (v, _)| a.max(v.abs()));
        assert!(
            ref_max > 1e-3,
            "{}: reference output is degenerate (max |out| {ref_max:e})",
            c.label
        );
        let got = dispatch_case(ctx, &c);
        let got_max = got.iter().fold(0f32, |a, b| a.max(b.abs()));
        assert!(
            got_max > 1e-3,
            "{}: kernel output is all-but-zero (max |out| {got_max:e}); the comparison would be \
             zeros against zeros",
            c.label
        );

        let mut over: Vec<String> = Vec::new();
        for (i, (v, l1)) in want.iter().enumerate() {
            let bound = v.abs() * f64::powi(2.0, -8) + 8.0 * l1 * c.k as f64 * f64::powi(2.0, -24);
            let err = (got[i] as f64 - v).abs();
            worst_ratio = worst_ratio.max(err / bound.max(f64::MIN_POSITIVE));
            if err > bound {
                over.push(format!(
                    "m-row {} row {}: got {} want {v} (err {err:e} > bound {bound:e})",
                    i / c.n,
                    i % c.n,
                    got[i]
                ));
            }
        }
        eprintln!(
            "[q3d-pf-gemm-oracle] {}: {} entries, {} over bound, ref_max={ref_max:.5}",
            c.label,
            want.len(),
            over.len()
        );
        if !over.is_empty() {
            failures.push(format!(
                "{}: {} of {} entries over bound\n  {}",
                c.label,
                over.len(),
                want.len(),
                over.iter()
                    .take(24)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n  ")
            ));
        }
    }
    eprintln!("[q3d-pf-gemm-oracle] worst err/bound across random cases: {worst_ratio:.3}");
    assert!(
        failures.is_empty(),
        "the shipped M-row prefill GEMM diverged from the f64 host reference by more than one \
         bf16 ulp plus the f32 accumulation slack. The output is packed bf16, so that bound IS \
         the format's own resolution, not a tolerance chosen to pass:\n{}",
        failures.join("\n")
    );
}

#[test]
fn every_m_row_in_the_corpus_carries_different_activations() {
    let mut worst_pairs_equal = 0usize;
    let mut checked = 0usize;
    for c in exact_cases().into_iter().chain(random_cases()) {
        assert!(c.m > 1, "{}: an M block of one row is a GEMV", c.label);
        for a in 0..c.m {
            for b in (a + 1)..c.m {
                checked += 1;
                if c.x[a * c.k..(a + 1) * c.k] == c.x[b * c.k..(b + 1) * c.k] {
                    worst_pairs_equal += 1;
                }
            }
        }
    }
    eprintln!(
        "[q3d-pf-gemm-oracle] {worst_pairs_equal} of {checked} m-row pairs share an activation row"
    );
    assert_eq!(
        worst_pairs_equal, 0,
        "{worst_pairs_equal} pairs of M rows carry identical activations; on those rows the \
         per-row stride is unobservable"
    );
}

#[test]
fn every_declared_width_generates_its_own_entry() {
    let mut entries: Vec<String> = Vec::new();
    for m in WIDTHS {
        let (_, entry) = source_for(m);
        assert!(
            !entries.contains(&entry),
            "width {m} generated the entry name {entry}, which another width already claimed; \
             the two would share a pipeline-cache slot and one of them would never be compiled"
        );
        entries.push(entry);
    }
    eprintln!("[q3d-pf-gemm-oracle] widths {WIDTHS:?} -> entries {entries:?}");
    let covered: Vec<usize> = exact_cases()
        .iter()
        .chain(random_cases().iter())
        .map(|c| c.m)
        .collect();
    for m in WIDTHS {
        assert!(
            covered.contains(&m),
            "width {m} is declared but no case dispatches it"
        );
    }
}
