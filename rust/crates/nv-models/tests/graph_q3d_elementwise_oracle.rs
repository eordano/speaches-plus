#![cfg(feature = "wgpu")]

mod common;
use common::bf16;
use common::bf16_bits_from_f64 as bf16_bits;
use common::pack_bf16_from_f64 as pack_bf16;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use common::LcgOddSeedNextF64SignedUnit as Lcg;

const MISC_TAG: &str = "q3d:misc";
const SILU_ENTRY: &str = "q3w_silu_mul";
const GATHER_M_ENTRY: &str = "q3w_gather_embed_m";

const TIE_GUARD: f64 = 0.05;

const POISON: u32 = 0xdead_beef;

fn misc_source() -> String {
    let all = nv_models::qwen3_5_dense_wgpu::nozi_audit_sources();
    let hit = all
        .into_iter()
        .find(|(tag, _)| *tag == MISC_TAG)
        .unwrap_or_else(|| {
            panic!(
                "qwen3_5_dense_wgpu::nozi_audit_sources() no longer exposes {MISC_TAG}; this gate \
                 compiles the SHIPPED text and cannot fall back to a copy"
            )
        });
    let src = hit.1;
    assert!(
        src.contains(&format!("fn {SILU_ENTRY}(")),
        "{MISC_TAG} no longer declares {SILU_ENTRY}; the entry moved and this gate is now testing \
         nothing"
    );
    src
}

fn prefill_source() -> String {
    let src = nv_models::qwen3_5_dense_wgpu::shipped_prefill_source();
    assert!(
        src.contains(&format!("fn {GATHER_M_ENTRY}(")),
        "the shipped prefill source no longer declares {GATHER_M_ENTRY}; the entry moved and this \
         gate is now testing nothing"
    );
    src
}

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_q3d_elementwise_oracle needs a real wgpu adapter; a skipped numeric gate reads as a \
         passed one, so this panics rather than returning early",
    )
}

fn next_bf16(x: f64) -> f64 {
    let bits = half::bf16::from_f32(x as f32).to_bits();
    half::bf16::from_bits(bits.wrapping_add(1)).to_f32() as f64
}

fn tie_dist_ulps(v: f64) -> f64 {
    if v == 0.0 || !v.is_finite() {
        return 0.5;
    }
    let ulp = (v.abs().log2().floor() - 7.0).exp2();
    let r = v.abs() / ulp;
    ((r - r.floor()) - 0.5).abs()
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SmParams {
    n_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

struct SiluData {
    g: Vec<f64>,
    u: Vec<f64>,
    want: Vec<f64>,
    min_tie: f64,
}

fn silu_data(n_words: usize, seed: u64) -> SiluData {
    let n = n_words * 2;
    let mut r = Lcg::new(seed);
    let mut g = Vec::with_capacity(n);
    let mut u = Vec::with_capacity(n);
    let mut want = Vec::with_capacity(n);
    let mut min_tie = 0.5f64;
    for _ in 0..n {
        let mut gv = bf16(r.next() * 5.0);
        let mut guard = 0;
        while tie_dist_ulps(silu(gv)) < TIE_GUARD {
            gv = next_bf16(gv);
            guard += 1;
            assert!(guard < 10_000, "silu tie screen failed to converge");
        }
        let a = bf16(silu(gv));
        let mut uv = bf16(r.next() * 2.0 + 0.5);
        guard = 0;
        while tie_dist_ulps(a * uv) < TIE_GUARD {
            uv = next_bf16(uv);
            guard += 1;
            assert!(guard < 10_000, "product tie screen failed to converge");
        }
        min_tie = min_tie
            .min(tie_dist_ulps(silu(gv)))
            .min(tie_dist_ulps(a * uv));
        want.push(bf16(a * uv));
        g.push(gv);
        u.push(uv);
    }
    SiluData {
        g,
        u,
        want,
        min_tie,
    }
}

#[test]
fn q3w_silu_mul_matches_an_f64_host_reference() {
    let ctx = ctx();
    eprintln!("[q3d-elementwise-oracle] adapter: {}", ctx.info.name);
    let src = misc_source();
    for n_words in [96usize, 7] {
        let d = silu_data(n_words, 0x5117_0000 ^ n_words as u64);
        let g_b = dispatch::storage_from_slice(ctx, "sm-g", &pack_bf16(&d.g));
        let u_b = dispatch::storage_from_slice(ctx, "sm-u", &pack_bf16(&d.u));
        let y_b = dispatch::storage_from_slice(ctx, "sm-y", &vec![POISON; n_words + 4]);
        let p_b = dispatch::uniform_from(
            ctx,
            "sm-p",
            &SmParams {
                n_words: n_words as u32,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        );
        dispatch::run(
            ctx,
            "q3d-elementwise-oracle",
            &src,
            SILU_ENTRY,
            &[(10, &g_b), (11, &u_b), (12, &y_b), (13, &p_b)],
            dispatch::workgroup_count_1d(ctx, n_words as u64, 64),
        )
        .unwrap_or_else(|e| panic!("n_words {n_words}: dispatch {SILU_ENTRY}: {e}"));

        let words: Vec<u32> =
            dispatch::read_back(ctx, &y_b, n_words + 4).expect("read back silu output");
        let ref_max = d.want.iter().fold(0f64, |a, b| a.max(b.abs()));
        assert!(
            ref_max > 1e-3,
            "n_words {n_words}: the reference is degenerate (max |y| {ref_max:e})"
        );
        let mut got_max = 0f64;
        for (i, w) in words[..n_words].iter().enumerate() {
            for (half_i, bits) in [(0usize, (*w & 0xffff) as u16), (1, (*w >> 16) as u16)] {
                let e = 2 * i + half_i;
                let gv = half::bf16::from_bits(bits).to_f32() as f64;
                got_max = got_max.max(gv.abs());
                assert_eq!(
                    bits,
                    bf16_bits(d.want[e]),
                    "n_words {n_words}: {SILU_ENTRY} diverged at element {e}: kernel {gv} vs \
                     reference {}. This corpus is screened so every bf16 rounding the kernel \
                     performs sits at least {TIE_GUARD} of an ulp from a tie, which is ~1000x the \
                     f32/f64 disagreement in exp -- so the demand is bit-exactness. Contrast the \
                     graph-level fixture, which stayed green when this entry's output was \
                     QUARTERED.",
                    d.want[e]
                );
            }
        }
        assert!(
            got_max > 1e-3,
            "n_words {n_words}: {SILU_ENTRY} output is all-but-zero; the comparison would be \
             zeros against zeros"
        );
        assert!(
            words[n_words..].iter().all(|w| *w == POISON),
            "n_words {n_words}: {SILU_ENTRY} wrote past n_words, so the `w >= sm_p.n_words` guard \
             is not holding and a tail workgroup would scribble on the next buffer"
        );
        eprintln!(
            "[q3d-elementwise-oracle] silu n_words={n_words}: {} elements bit-exact, min tie \
             distance {:.4} ulp, {} tail words still poisoned",
            d.want.len(),
            d.min_tie,
            words.len() - n_words
        );
    }
}

#[test]
fn the_silu_corpus_gates_and_multiplies() {
    let d = silu_data(96, 0x5117_0000 ^ 96);
    let u_gap = d.u.iter().fold(0f64, |a, b| a.max((b - 1.0).abs()));
    let mut worst_silu_gap = 0f64;
    let mut negative = 0usize;
    for gv in &d.g {
        worst_silu_gap = worst_silu_gap.max((silu(*gv) - gv).abs());
        negative += usize::from(*gv < 0.0);
    }
    eprintln!(
        "[q3d-elementwise-oracle] worst |u - 1| {u_gap:.4}, worst |silu(g) - g| \
         {worst_silu_gap:.4}, {negative} of {} gate inputs negative",
        d.g.len()
    );
    assert!(
        u_gap > 0.25,
        "every u is within {u_gap} of 1; the `* bf16_lo(uw)` multiply would be an identity"
    );
    assert!(
        worst_silu_gap > 0.25,
        "silu(g) never departs from g by more than {worst_silu_gap}; replacing the activation \
         with a pass-through would be invisible"
    );
    assert!(
        negative * 4 > d.g.len(),
        "only {negative} of {} gate inputs are negative; silu is nearly linear on the positive \
         side and the saturating branch would go untested",
        d.g.len()
    );
    assert!(
        d.min_tie >= TIE_GUARD,
        "the screened corpus still contains a bf16 rounding {} ulp from a tie",
        d.min_tie
    );
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GeParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
}

struct GatherCase {
    label: &'static str,
    row_off: usize,
    n_rows: usize,
    vocab: usize,
    hidden_words: usize,
    tokens: Vec<i32>,
}

fn gather_cases() -> Vec<GatherCase> {
    vec![
        GatherCase {
            label: "unsharded vocab32 hw300",
            row_off: 0,
            n_rows: 32,
            vocab: 32,
            hidden_words: 300,
            tokens: vec![3, 0, 7, 5],
        },
        GatherCase {
            label: "shard rows 8..16 of vocab32",
            row_off: 8,
            n_rows: 8,
            vocab: 32,
            hidden_words: 130,
            tokens: vec![10, 2, 15, 40, -1],
        },
    ]
}

impl GatherCase {

    fn resolved(&self, t: usize) -> usize {
        let raw = self.tokens[t];
        let s = if raw > 0 { raw as usize } else { 0 };
        if s >= self.vocab {
            0
        } else {
            s
        }
    }
    fn writes(&self, t: usize) -> bool {
        let s = self.resolved(t);
        s >= self.row_off && s < self.row_off + self.n_rows
    }
    fn embed(&self) -> Vec<u32> {
        (0..self.n_rows * self.hidden_words)
            .map(|i| 0x0200_0000 + i as u32)
            .collect()
    }
}

#[test]
fn q3w_gather_embed_m_copies_only_the_rows_this_shard_owns() {
    let ctx = ctx();
    let src = prefill_source();
    for c in gather_cases() {
        let m = c.tokens.len();
        let emb = c.embed();
        let emb_b = dispatch::storage_from_slice(ctx, "pge-emb", &emb);
        let tok_b = dispatch::storage_from_slice(ctx, "pge-tok", &c.tokens);
        let out_b = dispatch::storage_from_slice(ctx, "pge-out", &vec![POISON; m * c.hidden_words]);
        let p_b = dispatch::uniform_from(
            ctx,
            "pge-p",
            &GeParams {
                row_off: c.row_off as u32,
                n_rows: c.n_rows as u32,
                hidden_words: c.hidden_words as u32,
                vocab: c.vocab as u32,
            },
        );
        let (gx, _, _) = dispatch::workgroup_count_1d(ctx, c.hidden_words as u64, 256);
        dispatch::run(
            ctx,
            "q3d-elementwise-oracle",
            &src,
            GATHER_M_ENTRY,
            &[(0, &emb_b), (1, &tok_b), (2, &out_b), (3, &p_b)],
            (gx, m as u32, 1),
        )
        .unwrap_or_else(|e| panic!("{}: dispatch {GATHER_M_ENTRY}: {e}", c.label));

        let got: Vec<u32> =
            dispatch::read_back(ctx, &out_b, m * c.hidden_words).expect("read back gathered rows");
        let mut wrote = 0usize;
        for t in 0..m {
            let at = t * c.hidden_words;
            if c.writes(t) {
                wrote += 1;
                let base = (c.resolved(t) - c.row_off) * c.hidden_words;
                for w in 0..c.hidden_words {
                    assert_eq!(
                        got[at + w],
                        emb[base + w],
                        "{}: {GATHER_M_ENTRY} gathered the wrong word for token {t} (raw {}, \
                         resolved row {}) at offset {w}: 0x{:08x} vs 0x{:08x}. Every embedding \
                         word in this corpus is distinct, so a row or a column offset that is off \
                         by one moves an identifiable value.",
                        c.label,
                        c.tokens[t],
                        c.resolved(t),
                        got[at + w],
                        emb[base + w]
                    );
                }
            } else {
                for w in 0..c.hidden_words {
                    assert_eq!(
                        got[at + w],
                        POISON,
                        "{}: {GATHER_M_ENTRY} wrote row {t} (raw token {}, resolved {}) even \
                         though it falls outside this shard's rows {}..{}; a spurious write on a \
                         sharded embedding silently replaces another shard's contribution",
                        c.label,
                        c.tokens[t],
                        c.resolved(t),
                        c.row_off,
                        c.row_off + c.n_rows
                    );
                }
            }
        }
        assert!(
            wrote > 0,
            "{}: no token in this case lands inside the shard, so the whole gather is an early \
             return and the copy is untested",
            c.label
        );
        eprintln!(
            "[q3d-elementwise-oracle] {}: {wrote}/{m} rows gathered exactly, {} rows correctly \
             skipped",
            c.label,
            m - wrote
        );
    }
}

#[test]
fn the_gather_corpus_covers_every_token_disposition() {
    let mut below = false;
    let mut above = false;
    let mut out_of_vocab = false;
    let mut nonpositive = false;
    let mut inside_sharded = false;
    let mut multi_workgroup = false;
    for c in gather_cases() {
        multi_workgroup |= c.hidden_words > 256;
        for t in 0..c.tokens.len() {
            let raw = c.tokens[t];
            let s = c.resolved(t);
            out_of_vocab |= raw > 0 && raw as usize >= c.vocab;
            nonpositive |= raw <= 0;
            if c.row_off > 0 {
                below |= s < c.row_off;
                above |= s >= c.row_off + c.n_rows;
                inside_sharded |= c.writes(t);
            }
        }
    }
    assert!(
        below && inside_sharded,
        "the sharded case must contain both a token this shard owns and one below its range; \
         without both, `if (s < pge_p.row_off) return;` is untested in one direction"
    );
    assert!(
        above || out_of_vocab,
        "no token lands above the shard's row range, so the upper bound check is untested"
    );
    assert!(
        out_of_vocab,
        "no token is out of vocabulary, so the `if (s >= vocab) s = 0;` clamp is untested and an \
         out-of-range gather would read past the embedding buffer unobserved"
    );
    assert!(
        nonpositive,
        "no token is <= 0, so the `if (pge_tok[t] > 0)` guard is untested"
    );
    assert!(
        multi_workgroup,
        "no case has hidden_words > 256, so `wid.x * 256u + lid.x` never advances past its first \
         workgroup and the `w >= hidden_words` tail guard is untested"
    );
}
