#![cfg(feature = "wgpu")]

mod common;
use common::bf16;
use common::bf16_bits_from_f64 as bf16_bits;
use common::CkParams;
use common::pack_bf16_from_f64 as pack_bf16;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use common::LcgOddSeedNextF64SignedUnit as Lcg;

const ATTN_TAG: &str = "q3d:attn";
const NR_ENTRY: &str = "q3w_attn_norm_rope";
const NR_M_ENTRY: &str = "q3w_attn_norm_rope_m";
const KV_ENTRY: &str = "q3w_kv_write";
const KV_M_ENTRY: &str = "q3w_kv_write_m";
const GATE_ENTRY: &str = "q3w_attn_gate";
const GATE_M_ENTRY: &str = "q3w_attn_gate_m";

const TIE_GUARD: f64 = 0.05;

fn attn_source() -> String {
    let all = nv_models::qwen3_5_dense_wgpu::nozi_audit_sources();
    let hit = all
        .into_iter()
        .find(|(tag, _)| *tag == ATTN_TAG)
        .unwrap_or_else(|| {
            panic!(
                "qwen3_5_dense_wgpu::nozi_audit_sources() no longer exposes {ATTN_TAG}; this gate \
                 compiles the SHIPPED attention text and cannot fall back to a copy"
            )
        });
    let src = hit.1;
    for e in [NR_ENTRY, KV_ENTRY, GATE_ENTRY] {
        assert!(
            src.contains(&format!("fn {e}(")),
            "{ATTN_TAG} no longer declares {e}; the entry moved and this gate is now testing \
             nothing"
        );
    }
    src
}

fn prefill_source() -> String {
    let src = nv_models::qwen3_5_dense_wgpu::shipped_prefill_source();
    for e in [NR_M_ENTRY, KV_M_ENTRY, GATE_M_ENTRY] {
        assert!(
            src.contains(&format!("fn {e}(")),
            "the shipped prefill source no longer declares {e}; the entry moved and this gate is \
             now testing nothing"
        );
    }
    src
}

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_q3d_attn_chain_oracle needs a real wgpu adapter; a skipped numeric gate reads as a \
         passed one, so this panics rather than returning early",
    )
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ArParams {
    n_rows: u32,
    head_dim: u32,
    src_stride: u32,
    rot_half: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ArmParams {
    n_rows: u32,
    head_dim: u32,
    src_stride: u32,
    rot_half: u32,
    x_row_elems: u32,
    y_row_elems: u32,
    pad0: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct KvParams {
    words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AgParams {
    n_words: u32,
    head_dim: u32,
    src_stride: u32,
    gate_off: u32,
    has_gate: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AgmParams {
    n_words: u32,
    head_dim: u32,
    src_stride: u32,
    gate_off: u32,
    has_gate: u32,
    x_row_elems: u32,
    pad0: u32,
    pad1: u32,
}

fn unpack_bf16_bits(words: &[u32]) -> Vec<u16> {
    let mut out = Vec::with_capacity(words.len() * 2);
    for w in words {
        out.push((*w & 0xffff) as u16);
        out.push((*w >> 16) as u16);
    }
    out
}

fn tie_dist_ulps(v: f64) -> f64 {
    if v == 0.0 || !v.is_finite() {
        return 0.5;
    }
    let e = v.abs().log2().floor();
    let ulp = (e - 7.0).exp2();
    let r = v.abs() / ulp;
    let frac = r - r.floor();
    (frac - 0.5).abs()
}

fn dyadic_row(hd: usize, salt: usize) -> Vec<f64> {
    assert!(
        hd.is_multiple_of(8),
        "the dyadic-exact construction needs head_dim divisible by 8, got {hd}"
    );
    let n4 = hd / 8;
    let n2 = 3 * hd / 8;
    let mut mags = Vec::with_capacity(hd);
    for i in 0..hd {
        let slot = (i + salt) % 8;
        mags.push(if slot == 0 {
            4.0
        } else if slot < 4 {
            2.0
        } else {
            1.0
        });
    }
    assert_eq!(
        mags.iter().filter(|m| **m == 4.0).count(),
        n4,
        "dyadic row lost its magnitude-4 count"
    );
    assert_eq!(
        mags.iter().filter(|m| **m == 2.0).count(),
        n2,
        "dyadic row lost its magnitude-2 count"
    );
    mags.iter()
        .enumerate()
        .map(|(i, m)| if (i + salt) % 3 == 0 { -*m } else { *m })
        .collect()
}

const NR_WEIGHTS: [f64; 8] = [1.0, 1.5, -1.0, 0.5, 2.0, -1.5, -0.5, -2.0];

const NR_COS: [f64; 6] = [0.75, -0.5, 0.25, 1.0, -0.75, 0.5];
const NR_SIN: [f64; 6] = [-0.25, 0.75, 0.5, -1.0, 0.25, -0.75];

struct NrCase {
    label: &'static str,
    n_rows: usize,
    hd: usize,
    src_stride: usize,
    rot_half: usize,
    pos: usize,
    m_live: usize,
    base: usize,
    salt: usize,
}

fn nr_cases() -> Vec<NrCase> {
    vec![
        NrCase {
            label: "hd32 gate-stride rows4 rot8",
            n_rows: 4,
            hd: 32,
            src_stride: 64,
            rot_half: 8,
            pos: 9,
            m_live: 3,
            base: 7,
            salt: 0,
        },
        NrCase {
            label: "hd64 tight rows2 rot32(full)",
            n_rows: 2,
            hd: 64,
            src_stride: 64,
            rot_half: 32,
            pos: 0,
            m_live: 1,
            base: 0,
            salt: 3,
        },
        NrCase {
            label: "hd128 gate-stride rows3 rot16",
            n_rows: 3,
            hd: 128,
            src_stride: 256,
            rot_half: 16,
            pos: 5,
            m_live: 4,
            base: 11,
            salt: 5,
        },
    ]
}

impl NrCase {
    fn x_row_elems(&self) -> usize {
        self.n_rows * self.src_stride + 16
    }
    fn y_row_elems(&self) -> usize {
        self.n_rows * self.hd + 8
    }
    fn rope_rows(&self) -> usize {
        self.pos.max(self.base + self.m_live) + 2
    }
    fn norm_w(&self) -> Vec<f64> {
        (0..self.hd)
            .map(|d| NR_WEIGHTS[(d + self.salt) % NR_WEIGHTS.len()])
            .collect()
    }
    fn cos(&self) -> Vec<f64> {
        (0..self.rope_rows() * self.rot_half)
            .map(|i| NR_COS[(i + self.salt) % NR_COS.len()])
            .collect()
    }
    fn sin(&self) -> Vec<f64> {
        (0..self.rope_rows() * self.rot_half)
            .map(|i| NR_SIN[(i * 2 + self.salt) % NR_SIN.len()])
            .collect()
    }

    fn src_row(&self, t: usize, r: usize) -> Vec<f64> {
        dyadic_row(self.hd, self.salt + 2 * r + 5 * t)
    }
}

fn nr_reference(c: &NrCase, t: usize, r: usize, p: usize, eps: f64) -> (Vec<f64>, f64) {
    let hd = c.hd;
    let v = c.src_row(t, r);
    let w = c.norm_w();
    let cos = c.cos();
    let sin = c.sin();
    let mut ss = 0f64;
    for d in (0..hd).rev() {
        ss += v[d] * v[d];
    }
    let rms = 1.0 / (ss / hd as f64 + eps).sqrt();
    let n: Vec<f64> = (0..hd).map(|d| bf16(v[d] * rms * w[d])).collect();
    let rh = c.rot_half;
    let out = (0..hd)
        .map(|d| {
            if d < rh {
                n[d] * cos[p * rh + d] - n[d + rh] * sin[p * rh + d]
            } else if d < 2 * rh {
                let i = d - rh;
                n[i] * sin[p * rh + i] + n[d] * cos[p * rh + i]
            } else {
                n[d]
            }
        })
        .collect();
    (out, rms)
}

const POISON: u32 = 0xdead_beef;

fn assert_bit_exact(label: &str, entry: &str, got: &[u16], want: &[f64]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: {entry} readback length {} != reference length {}",
        got.len(),
        want.len()
    );
    let ref_max = want.iter().fold(0f64, |a, b| a.max(b.abs()));
    assert!(
        ref_max > 1e-3,
        "{label}: reference for {entry} is degenerate (max |out| {ref_max:e}); a comparison \
         against it would pass on any kernel"
    );
    let got_max = got
        .iter()
        .fold(0f32, |a, b| a.max(half::bf16::from_bits(*b).to_f32().abs()));
    assert!(
        got_max > 1e-3,
        "{label}: {entry} output is all-but-zero (max |out| {got_max:e}); the comparison would be \
         zeros against zeros"
    );
    let distinct: std::collections::BTreeSet<u16> = got.iter().copied().collect();
    assert!(
        distinct.len() > 2,
        "{label}: {entry} produced only {} distinct bf16 values over {} lanes; a constant output \
         would satisfy an index-insensitive check",
        distinct.len(),
        got.len()
    );
    let mut wrong = 0usize;
    let mut first = None;
    for (i, (g, wv)) in got.iter().zip(want.iter()).enumerate() {
        let want_bits = bf16_bits(*wv);
        if *g != want_bits {
            wrong += 1;
            if first.is_none() {
                first = Some((i, *g, want_bits, *wv));
            }
        }
    }
    if wrong > 0 {
        let (i, g, wb, wv) = first.unwrap();
        panic!(
            "{label}: {entry} diverged from the f64 host reference on {wrong} of {} lanes; first \
             at lane {i}: kernel 0x{g:04x} ({}) vs reference 0x{wb:04x} ({wv:e}). This corpus is \
             dyadic-exact end to end -- every sum, the inverseSqrt, and both bf16 roundings land \
             on representable values -- so the demand is bit-exactness and there is no tolerance \
             to relax. Contrast the graph-level fixture, whose rel < 0.05 on 4-layer logits is \
             300x looser than the effect it would have to see.",
            got.len(),
            half::bf16::from_bits(g).to_f32()
        );
    }
}

#[test]
fn q3w_attn_norm_rope_matches_an_f64_host_reference() {
    let ctx = ctx();
    eprintln!("[q3d-attn-chain-oracle] adapter: {}", ctx.info.name);
    let src = attn_source();
    for c in nr_cases() {
        let hd = c.hd;
        let elems = c.n_rows * c.src_stride;
        let mut flat = vec![-7.5f64; elems];
        for r in 0..c.n_rows {
            let row = c.src_row(0, r);
            flat[r * c.src_stride..r * c.src_stride + hd].copy_from_slice(&row);
        }
        let mut want = Vec::with_capacity(c.n_rows * hd);
        let mut rms_seen = 0f64;
        for r in 0..c.n_rows {
            let (o, rms) = nr_reference(&c, 0, r, c.pos, 0.0);
            rms_seen = rms;
            want.extend(o);
        }
        assert!(
            (rms_seen - 0.5).abs() < 1e-12,
            "{}: the dyadic corpus no longer produces rms == 0.5 (got {rms_seen}); with rms == 1 \
             the normalization is an identity and dropping it would be invisible",
            c.label
        );

        let src_b = dispatch::storage_from_slice(ctx, "ar-src", &pack_bf16(&flat));
        let w_b = dispatch::storage_from_slice(ctx, "ar-w", &pack_bf16(&c.norm_w()));
        let cos_b = dispatch::storage_from_slice(
            ctx,
            "ar-cos",
            &c.cos().iter().map(|v| *v as f32).collect::<Vec<f32>>(),
        );
        let sin_b = dispatch::storage_from_slice(
            ctx,
            "ar-sin",
            &c.sin().iter().map(|v| *v as f32).collect::<Vec<f32>>(),
        );
        let pos_b = dispatch::storage_from_slice(ctx, "ar-pos", &[c.pos as i32]);
        let out_words = c.n_rows * hd / 2;
        let out_b = dispatch::storage_from_slice(ctx, "ar-out", &vec![POISON; out_words]);
        let par_b = dispatch::uniform_from(
            ctx,
            "ar-p",
            &ArParams {
                n_rows: c.n_rows as u32,
                head_dim: hd as u32,
                src_stride: c.src_stride as u32,
                rot_half: c.rot_half as u32,
                pad0: 0,
                pad1: 0,
                pad2: 0,
                eps: 0.0,
            },
        );

        dispatch::run(
            ctx,
            "q3d-attn-chain-oracle",
            &src,
            NR_ENTRY,
            &[
                (0, &src_b),
                (1, &w_b),
                (2, &cos_b),
                (3, &sin_b),
                (4, &pos_b),
                (5, &out_b),
                (6, &par_b),
            ],
            (c.n_rows as u32, 1, 1),
        )
        .unwrap_or_else(|e| panic!("{}: dispatch {NR_ENTRY}: {e}", c.label));

        let words: Vec<u32> =
            dispatch::read_back(ctx, &out_b, out_words).expect("read back norm+rope output");
        assert_bit_exact(c.label, NR_ENTRY, &unpack_bf16_bits(&words), &want);
        eprintln!(
            "[q3d-attn-chain-oracle] {} decode: {} lanes bit-exact (rms {rms_seen}, pos {})",
            c.label,
            want.len(),
            c.pos
        );
    }
}

#[test]
fn q3w_attn_norm_rope_m_matches_an_f64_host_reference() {
    let ctx = ctx();
    let src = prefill_source();
    let mut saw_multi = false;
    for c in nr_cases() {
        saw_multi |= c.m_live > 1;
        let hd = c.hd;
        let grid_m = c.m_live + 1;
        let mut flat = vec![-7.5f64; grid_m * c.x_row_elems()];
        for t in 0..grid_m {
            for r in 0..c.n_rows {
                let row = c.src_row(t, r);
                let at = t * c.x_row_elems() + r * c.src_stride;
                flat[at..at + hd].copy_from_slice(&row);
            }
        }
        let out_words = grid_m * c.y_row_elems() / 2;
        let out_b = dispatch::storage_from_slice(ctx, "par-out", &vec![POISON; out_words]);

        let src_b = dispatch::storage_from_slice(ctx, "par-src", &pack_bf16(&flat));
        let w_b = dispatch::storage_from_slice(ctx, "par-w", &pack_bf16(&c.norm_w()));
        let cos_b = dispatch::storage_from_slice(
            ctx,
            "par-cos",
            &c.cos().iter().map(|v| *v as f32).collect::<Vec<f32>>(),
        );
        let sin_b = dispatch::storage_from_slice(
            ctx,
            "par-sin",
            &c.sin().iter().map(|v| *v as f32).collect::<Vec<f32>>(),
        );
        let par_b = dispatch::uniform_from(
            ctx,
            "par-p",
            &ArmParams {
                n_rows: c.n_rows as u32,
                head_dim: hd as u32,
                src_stride: c.src_stride as u32,
                rot_half: c.rot_half as u32,
                x_row_elems: c.x_row_elems() as u32,
                y_row_elems: c.y_row_elems() as u32,
                pad0: 0,
                eps: 0.0,
            },
        );
        let ck_b = dispatch::uniform_from(
            ctx,
            "par-ck",
            &CkParams {
                m_live: c.m_live as u32,
                base: c.base as u32,
                pad0: 0,
                pad1: 0,
            },
        );

        dispatch::run(
            ctx,
            "q3d-attn-chain-oracle",
            &src,
            NR_M_ENTRY,
            &[
                (60, &src_b),
                (61, &w_b),
                (62, &cos_b),
                (63, &sin_b),
                (64, &out_b),
                (65, &par_b),
                (66, &ck_b),
            ],
            (c.n_rows as u32, grid_m as u32, 1),
        )
        .unwrap_or_else(|e| panic!("{}: dispatch {NR_M_ENTRY}: {e}", c.label));

        let words: Vec<u32> =
            dispatch::read_back(ctx, &out_b, out_words).expect("read back M-row norm+rope output");
        let bits = unpack_bf16_bits(&words);
        for t in 0..c.m_live {
            let mut want = Vec::with_capacity(c.n_rows * hd);
            for r in 0..c.n_rows {
                want.extend(nr_reference(&c, t, r, c.base + t, 0.0).0);
            }
            let at = t * c.y_row_elems();
            assert_bit_exact(c.label, NR_M_ENTRY, &bits[at..at + c.n_rows * hd], &want);
        }
        let dead = c.m_live * c.y_row_elems() / 2;
        assert!(
            words[dead..].iter().all(|w| *w == POISON),
            "{}: {NR_M_ENTRY} wrote past m_live -- the row at t == m_live == {} lost its poison, \
             so the `t >= par_ck.m_live` guard is not doing its job and a prefill chunk shorter \
             than the grid would corrupt the next chunk",
            c.label,
            c.m_live
        );
        eprintln!(
            "[q3d-attn-chain-oracle] {} prefill: {} live rows bit-exact, base {}, tail row \
             untouched",
            c.label, c.m_live, c.base
        );
    }
    assert!(
        saw_multi,
        "every M-row case has m_live == 1, where the token loop degenerates to the decode entry \
         and the x_row_elems / y_row_elems / base strides are all multiplied by zero"
    );
}

#[test]
fn the_norm_rope_corpus_rotates_and_normalizes() {
    let mut saw_partial = false;
    let mut saw_full = false;
    let mut saw_nontrivial_rope = false;
    let mut saw_wide_stride = false;
    let mut saw_nonzero_pos = false;
    let mut saw_nonzero_base = false;
    for c in nr_cases() {
        saw_partial |= 2 * c.rot_half < c.hd;
        saw_full |= 2 * c.rot_half == c.hd;
        saw_wide_stride |= c.src_stride > c.hd;
        saw_nonzero_pos |= c.pos > 0;
        saw_nonzero_base |= c.base > 0;
        for (cv, sv) in c.cos().iter().zip(c.sin().iter()) {
            saw_nontrivial_rope |= (*cv - 1.0).abs() > 1e-9 && sv.abs() > 1e-9;
        }
        let w = c.norm_w();
        assert!(
            w.iter().any(|v| (*v - w[0]).abs() > 1e-9),
            "{}: the norm weight vector is constant; a per-lane weight index bug would be \
             invisible",
            c.label
        );
        let v = c.src_row(0, 0);
        assert!(
            v.iter().any(|x| (*x - v[0]).abs() > 1e-9),
            "{}: the activation row is constant; a per-lane source index bug would be invisible",
            c.label
        );
    }
    assert!(
        saw_partial,
        "no case has 2 * rot_half < head_dim, so the `return ar_buf[d]` pass-through tail of \
         ar_rope_at is never executed and deleting it would go unnoticed"
    );
    assert!(
        saw_full,
        "no case rotates the whole head, so a rot_half that is silently halved would still pass"
    );
    assert!(
        saw_nontrivial_rope,
        "every rope table entry is cos == 1 / sin == 0, which makes the rotation an identity"
    );
    assert!(
        saw_wide_stride,
        "no case has src_stride > head_dim, so the attn_output_gate layout -- the one the shipped \
         dense config actually uses for q -- is never exercised and dropping src_stride would pass"
    );
    assert!(
        saw_nonzero_pos && saw_nonzero_base,
        "the rope position is 0 in every case; `ar_cos[p * rh + d]` would then be indistinguishable \
         from `ar_cos[d]`"
    );
}

struct KvCase {
    label: &'static str,
    words: usize,
    max_seq: usize,
    pos: usize,
    m_live: usize,
}

fn kv_cases() -> Vec<KvCase> {
    vec![
        KvCase {
            label: "words96 pos5 m4",
            words: 96,
            max_seq: 16,
            pos: 5,
            m_live: 4,
        },
        KvCase {
            label: "words32 pos0 m1",
            words: 32,
            max_seq: 8,
            pos: 0,
            m_live: 1,
        },
    ]
}

fn kv_payload(words: usize, rows: usize, salt: u32) -> Vec<u32> {
    (0..words * rows)
        .map(|i| 0x0100_0000 + salt * 0x0001_0000 + i as u32)
        .collect()
}

fn check_cache(label: &str, entry: &str, what: &str, got: &[u32], want: &[u32]) {
    assert_eq!(got.len(), want.len(), "{label}: {what} length mismatch");
    let distinct: std::collections::BTreeSet<u32> = want.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        want.len(),
        "{label}: the {what} reference has repeated words, so a scatter that lands on the wrong \
         slab could still compare equal"
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g, w,
            "{label}: {entry} wrote the wrong {what} at word {i}: 0x{g:08x} vs 0x{w:08x}. This is \
             a u32 scatter, so the demand is exact equality. A cache write that is one slab or one \
             word off leaves THIS token's logits intact and corrupts every later token, which is \
             exactly what the single-token graph fixture cannot see."
        );
    }
}

#[test]
fn q3w_kv_write_scatters_to_the_position_slab() {
    let ctx = ctx();
    let src = attn_source();
    for c in kv_cases() {
        let k = kv_payload(c.words, 1, 1);
        let v = kv_payload(c.words, 1, 2);
        let slab = c.max_seq * c.words;
        let kc_b = dispatch::storage_from_slice(ctx, "kw-kc", &vec![POISON; slab]);
        let vc_b = dispatch::storage_from_slice(ctx, "kw-vc", &vec![POISON; slab]);
        let k_b = dispatch::storage_from_slice(ctx, "kw-k", &k);
        let v_b = dispatch::storage_from_slice(ctx, "kw-v", &v);
        let pos_b = dispatch::storage_from_slice(ctx, "kw-pos", &[c.pos as i32]);
        let p_b = dispatch::uniform_from(
            ctx,
            "kw-p",
            &KvParams {
                words: c.words as u32,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        );
        dispatch::run(
            ctx,
            "q3d-attn-chain-oracle",
            &src,
            KV_ENTRY,
            &[
                (10, &k_b),
                (11, &v_b),
                (12, &kc_b),
                (13, &vc_b),
                (14, &pos_b),
                (15, &p_b),
            ],
            dispatch::workgroup_count_1d(ctx, c.words as u64, 64),
        )
        .unwrap_or_else(|e| panic!("{}: dispatch {KV_ENTRY}: {e}", c.label));

        let kc: Vec<u32> = dispatch::read_back(ctx, &kc_b, slab).expect("read back k cache");
        let vc: Vec<u32> = dispatch::read_back(ctx, &vc_b, slab).expect("read back v cache");
        let at = c.pos * c.words;
        check_cache(c.label, KV_ENTRY, "k cache", &kc[at..at + c.words], &k);
        check_cache(c.label, KV_ENTRY, "v cache", &vc[at..at + c.words], &v);
        for (name, cache) in [("k", &kc), ("v", &vc)] {
            for (i, w) in cache.iter().enumerate() {
                if i >= at && i < at + c.words {
                    continue;
                }
                assert_eq!(
                    *w, POISON,
                    "{}: {KV_ENTRY} clobbered {name} cache word {i}, outside the slab for \
                     position {}; a scatter that writes more than its own position destroys \
                     history that no later read can recover",
                    c.label, c.pos
                );
            }
        }
        eprintln!(
            "[q3d-attn-chain-oracle] {} decode kv: slab {} exact, {} words of history intact",
            c.label,
            c.pos,
            slab - c.words
        );
    }
}

#[test]
fn q3w_kv_write_m_scatters_every_live_token_and_stops() {
    let ctx = ctx();
    let src = prefill_source();
    for c in kv_cases() {
        let base = 3usize;
        let grid_m = c.m_live + 1;
        let k = kv_payload(c.words, grid_m, 1);
        let v = kv_payload(c.words, grid_m, 2);
        let slab = c.max_seq * c.words;
        assert!(
            base + grid_m <= c.max_seq,
            "{}: the cache is too small for base {base} plus {grid_m} grid rows",
            c.label
        );
        let kc_b = dispatch::storage_from_slice(ctx, "pkw-kc", &vec![POISON; slab]);
        let vc_b = dispatch::storage_from_slice(ctx, "pkw-vc", &vec![POISON; slab]);
        let k_b = dispatch::storage_from_slice(ctx, "pkw-k", &k);
        let v_b = dispatch::storage_from_slice(ctx, "pkw-v", &v);
        let p_b = dispatch::uniform_from(
            ctx,
            "pkw-p",
            &KvParams {
                words: c.words as u32,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        );
        let ck_b = dispatch::uniform_from(
            ctx,
            "pkw-ck",
            &CkParams {
                m_live: c.m_live as u32,
                base: base as u32,
                pad0: 0,
                pad1: 0,
            },
        );
        let (gx, _, _) = dispatch::workgroup_count_1d(ctx, c.words as u64, 64);
        dispatch::run(
            ctx,
            "q3d-attn-chain-oracle",
            &src,
            KV_M_ENTRY,
            &[
                (70, &k_b),
                (71, &v_b),
                (72, &kc_b),
                (73, &vc_b),
                (74, &p_b),
                (75, &ck_b),
            ],
            (gx, grid_m as u32, 1),
        )
        .unwrap_or_else(|e| panic!("{}: dispatch {KV_M_ENTRY}: {e}", c.label));

        let kc: Vec<u32> = dispatch::read_back(ctx, &kc_b, slab).expect("read back k cache");
        let vc: Vec<u32> = dispatch::read_back(ctx, &vc_b, slab).expect("read back v cache");
        for t in 0..c.m_live {
            let at = (base + t) * c.words;
            check_cache(
                c.label,
                KV_M_ENTRY,
                "k cache",
                &kc[at..at + c.words],
                &k[t * c.words..(t + 1) * c.words],
            );
            check_cache(
                c.label,
                KV_M_ENTRY,
                "v cache",
                &vc[at..at + c.words],
                &v[t * c.words..(t + 1) * c.words],
            );
        }
        let live_lo = base * c.words;
        let live_hi = (base + c.m_live) * c.words;
        for (name, cache) in [("k", &kc), ("v", &vc)] {
            for (i, w) in cache.iter().enumerate() {
                if i >= live_lo && i < live_hi {
                    continue;
                }
                assert_eq!(
                    *w,
                    POISON,
                    "{}: {KV_M_ENTRY} wrote {name} cache word {i} outside slabs \
                     {base}..{}; either the base offset or the `t >= pkw_ck.m_live` guard is \
                     wrong, and a chunk shorter than its grid would then overwrite live history",
                    c.label,
                    base + c.m_live
                );
            }
        }
        eprintln!(
            "[q3d-attn-chain-oracle] {} prefill kv: slabs {base}..{} exact, tail row skipped",
            c.label,
            base + c.m_live
        );
    }
}

#[test]
fn the_kv_corpus_writes_at_a_nonzero_position() {
    let mut worst = 0usize;
    let mut spans_two_workgroups = false;
    for c in kv_cases() {
        worst = worst.max(c.pos);
        spans_two_workgroups |= c.words > 64;
        assert!(
            c.pos + 1 <= c.max_seq && c.m_live >= 1,
            "{}: the case does not fit its own cache",
            c.label
        );
    }
    assert!(
        worst > 0,
        "every KV case writes at position 0, where the `p * kw_p.words` offset is zero and \
         dropping it entirely would still pass"
    );
    assert!(
        spans_two_workgroups,
        "no KV case has more than 64 words, so the whole row fits one workgroup and the \
         global_invocation_id x term is never exercised past its first block"
    );
}

struct GateCase {
    label: &'static str,
    n_words: usize,
    head_dim: usize,
    has_gate: bool,
    m: usize,
}

fn gate_cases() -> Vec<GateCase> {
    vec![
        GateCase {
            label: "hd32 4heads gated m3",
            n_words: 64,
            head_dim: 32,
            has_gate: true,
            m: 3,
        },
        GateCase {
            label: "hd16 2heads ungated m2",
            n_words: 16,
            head_dim: 16,
            has_gate: false,
            m: 2,
        },
    ]
}

impl GateCase {
    fn heads(&self) -> usize {
        self.n_words * 2 / self.head_dim
    }
    fn src_stride(&self) -> usize {
        2 * self.head_dim
    }
    fn gate_off(&self) -> usize {
        self.head_dim
    }
    fn x_row_elems(&self) -> usize {
        self.heads() * self.src_stride() + 16
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn next_bf16(x: f64) -> f64 {
    let bits = half::bf16::from_f32(x as f32).to_bits();
    half::bf16::from_bits(bits.wrapping_add(1)).to_f32() as f64
}

fn gate_pair(r: &mut Lcg) -> (f64, f64, f64) {
    let mut g = bf16(r.next() * 6.0);
    let mut guard = 0;
    while tie_dist_ulps(sigmoid(g)) < TIE_GUARD {
        g = next_bf16(g);
        guard += 1;
        assert!(guard < 10_000, "sigmoid tie screen failed to converge");
    }
    let s = bf16(sigmoid(g));
    let mut a = r.next() * 3.0 + 0.25;
    guard = 0;
    while tie_dist_ulps(a) < TIE_GUARD || tie_dist_ulps(bf16(a) * s) < TIE_GUARD {
        a += 0.0011;
        guard += 1;
        assert!(guard < 10_000, "activation tie screen failed to converge");
    }
    (a, g, s)
}

struct GateData {
    attn: Vec<f64>,
    qraw: Vec<f64>,
    want: Vec<f64>,
    min_tie: f64,
}

fn gate_data(c: &GateCase, m: usize, x_row_elems: usize) -> GateData {
    let mut r = Lcg::new(0x9a7e_0000 ^ (c.head_dim as u64) << 8 ^ m as u64);
    let elems = c.n_words * 2;
    let mut attn = vec![0f64; m * elems];
    let mut qraw = vec![-3.25f64; m * x_row_elems];
    let mut want = vec![0f64; m * elems];
    let mut min_tie = 0.5f64;
    for t in 0..m {
        for e in 0..elems {
            let (a, g, s) = gate_pair(&mut r);
            attn[t * elems + e] = a;
            let h = e / c.head_dim;
            let d = e % c.head_dim;
            qraw[t * x_row_elems + h * c.src_stride() + c.gate_off() + d] = g;
            min_tie = min_tie.min(tie_dist_ulps(a)).min(tie_dist_ulps(sigmoid(g)));
            want[t * elems + e] = if c.has_gate { bf16(a) * s } else { bf16(a) };
            min_tie = min_tie.min(tie_dist_ulps(want[t * elems + e]));
        }
    }
    GateData {
        attn,
        qraw,
        want,
        min_tie,
    }
}

#[test]
fn q3w_attn_gate_matches_an_f64_host_reference() {
    let ctx = ctx();
    let src = attn_source();
    for c in gate_cases() {
        let d = gate_data(&c, 1, c.x_row_elems());
        let attn_b = dispatch::storage_from_slice(
            ctx,
            "ag-attn",
            &d.attn.iter().map(|v| *v as f32).collect::<Vec<f32>>(),
        );
        let qraw_b = dispatch::storage_from_slice(ctx, "ag-qraw", &pack_bf16(&d.qraw));
        let out_b = dispatch::storage_from_slice(ctx, "ag-out", &vec![POISON; c.n_words]);
        let p_b = dispatch::uniform_from(
            ctx,
            "ag-p",
            &AgParams {
                n_words: c.n_words as u32,
                head_dim: c.head_dim as u32,
                src_stride: c.src_stride() as u32,
                gate_off: c.gate_off() as u32,
                has_gate: u32::from(c.has_gate),
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        );
        dispatch::run(
            ctx,
            "q3d-attn-chain-oracle",
            &src,
            GATE_ENTRY,
            &[(30, &attn_b), (31, &qraw_b), (32, &out_b), (33, &p_b)],
            dispatch::workgroup_count_1d(ctx, c.n_words as u64, 64),
        )
        .unwrap_or_else(|e| panic!("{}: dispatch {GATE_ENTRY}: {e}", c.label));
        let words: Vec<u32> =
            dispatch::read_back(ctx, &out_b, c.n_words).expect("read back gate output");
        assert_bit_exact(c.label, GATE_ENTRY, &unpack_bf16_bits(&words), &d.want);
        eprintln!(
            "[q3d-attn-chain-oracle] {} decode gate: {} lanes bit-exact, min tie distance \
             {:.4} ulp",
            c.label,
            d.want.len(),
            d.min_tie
        );
    }
}

#[test]
fn q3w_attn_gate_m_matches_an_f64_host_reference() {
    let ctx = ctx();
    let src = prefill_source();
    for c in gate_cases() {
        let xr = c.x_row_elems();
        let d = gate_data(&c, c.m, xr);
        let attn_b = dispatch::storage_from_slice(
            ctx,
            "pag-attn",
            &d.attn.iter().map(|v| *v as f32).collect::<Vec<f32>>(),
        );
        let qraw_b = dispatch::storage_from_slice(ctx, "pag-qraw", &pack_bf16(&d.qraw));
        let out_b = dispatch::storage_from_slice(ctx, "pag-out", &vec![POISON; c.m * c.n_words]);
        let p_b = dispatch::uniform_from(
            ctx,
            "pag-p",
            &AgmParams {
                n_words: c.n_words as u32,
                head_dim: c.head_dim as u32,
                src_stride: c.src_stride() as u32,
                gate_off: c.gate_off() as u32,
                has_gate: u32::from(c.has_gate),
                x_row_elems: xr as u32,
                pad0: 0,
                pad1: 0,
            },
        );
        let (gx, _, _) = dispatch::workgroup_count_1d(ctx, c.n_words as u64, 64);
        dispatch::run(
            ctx,
            "q3d-attn-chain-oracle",
            &src,
            GATE_M_ENTRY,
            &[(90, &attn_b), (91, &qraw_b), (92, &out_b), (93, &p_b)],
            (gx, c.m as u32, 1),
        )
        .unwrap_or_else(|e| panic!("{}: dispatch {GATE_M_ENTRY}: {e}", c.label));
        let words: Vec<u32> =
            dispatch::read_back(ctx, &out_b, c.m * c.n_words).expect("read back M-row gate output");
        assert_bit_exact(c.label, GATE_M_ENTRY, &unpack_bf16_bits(&words), &d.want);
        eprintln!(
            "[q3d-attn-chain-oracle] {} prefill gate: {} lanes over {} rows bit-exact, min tie \
             distance {:.4} ulp",
            c.label,
            d.want.len(),
            c.m,
            d.min_tie
        );
    }
}

#[test]
fn the_gate_corpus_covers_both_branches_and_the_gate_bites() {
    let mut gated = false;
    let mut ungated = false;
    let mut worst_bite = 0f64;
    for c in gate_cases() {
        gated |= c.has_gate;
        ungated |= !c.has_gate;
        if !c.has_gate {
            continue;
        }
        let d = gate_data(&c, 1, c.x_row_elems());
        for (i, w) in d.want.iter().enumerate() {
            let ungated_value = bf16(d.attn[i]);
            worst_bite = worst_bite.max((w - ungated_value).abs() / ungated_value.abs().max(1e-6));
        }
        assert!(
            d.min_tie >= TIE_GUARD,
            "{}: the gate corpus contains a bf16 rounding only {:.4} ulp from a tie, below the \
             {TIE_GUARD} screen; the f32 and f64 roundings could then disagree and this gate would \
             be flaky rather than bit-exact",
            c.label,
            d.min_tie
        );
    }
    eprintln!("[q3d-attn-chain-oracle] worst gate deflection: {worst_bite:.4} relative");
    assert!(gated && ungated, "the corpus must cover has_gate 0 and 1");
    assert!(
        worst_bite > 0.25,
        "the sigmoid gate never moves an output by more than {worst_bite} relative; with a gate \
         this close to 1 deleting the multiply would be invisible"
    );
}
