#![cfg(feature = "wgpu")]

mod common;
use common::pack_bf16_from_f64 as pack_bf16;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use nv_models::qwen3_5_moe_wgpu::delta_recurrent_kernel;

const HEAD_FUSED_ENTRY: &str = "q3w_delta_head_fused";
const ATTN_FUSED_ENTRY: &str = "q3w_attn_qk_norm_rope_qcast";
const GEMV_MERGED_ENTRY: &str = "q3w_gemv_dn_merged_fp8_qkv_fp8_z_bf16_ab";

const WHY_BIT_IDENTITY_IS_THE_GATE: &str = "each fused entry here replaces a dispatch chain of \
     shipped entries whose numerics are pinned by their own host-reference oracles \
     (graph_q3d_delta_decode_oracle, graph_q3d_delta_front_oracle, graph_q3d_attn_chain_oracle, \
     graph_q3d_gemv_i8_oracle's bf16 sibling coverage); the fusion contract is that per-element \
     arithmetic order is preserved, so the only acceptable relation to the chain it replaces is \
     BIT identity -- any tolerance would re-admit exactly the reassociation the contract forbids";

fn ctx() -> &'static WgpuContext {
    WgpuContext::shared().expect(
        "graph_q3d_fused_decode_identity needs a real wgpu adapter; a skipped identity gate reads \
         as a passed one, so this panics rather than returning early",
    )
}

fn shipped(tag: &str) -> String {
    nv_models::qwen3_5_dense_wgpu::nozi_audit_sources()
        .into_iter()
        .find(|(t, _)| *t == tag)
        .unwrap_or_else(|| panic!("nozi_audit_sources no longer exposes {tag}"))
        .1
}

fn assert_entry(src: &str, entry: &str) {
    assert!(
        src.contains(&format!("fn {entry}(")),
        "the shipped source no longer declares {entry}; this gate is now testing nothing"
    );
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 32) as u32) as f64 / (1u64 << 31) as f64) - 1.0
    }
    fn f32s(&mut self, n: usize, scale: f64) -> Vec<f32> {
        (0..n).map(|_| (self.next() * scale) as f32).collect()
    }
    fn f64s(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| self.next() * scale).collect()
    }
    fn e4m3_words_no_nan(&mut self, n_words: usize) -> Vec<u32> {
        (0..n_words)
            .map(|_| {
                let mut w = 0u32;
                for b in 0..4 {
                    self.0 = self
                        .0
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let mut byte = (self.0 >> 40) as u8;
                    if byte & 0x7f == 0x7f {
                        byte &= 0xbf;
                    }
                    w |= (byte as u32) << (8 * b);
                }
                w
            })
            .collect()
    }
}

fn assert_bits_eq(label: &str, what: &str, want: &[f32], got: &[f32]) {
    assert_eq!(want.len(), got.len(), "{label}: {what} length");
    for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
        assert_eq!(
            w.to_bits(),
            g.to_bits(),
            "{label}: {what}[{i}] differs between the fused entry and the chain it replaces \
             ({w} vs {g}). {WHY_BIT_IDENTITY_IS_THE_GATE}"
        );
    }
}

fn assert_words_eq(label: &str, what: &str, want: &[u32], got: &[u32]) {
    assert_eq!(want.len(), got.len(), "{label}: {what} length");
    for (i, (w, g)) in want.iter().zip(got.iter()).enumerate() {
        assert_eq!(
            w, g,
            "{label}: {what} word [{i}] differs between the fused entry and the chain it \
             replaces ({w:#010x} vs {g:#010x}). {WHY_BIT_IDENTITY_IS_THE_GATE}"
        );
    }
}

fn assert_nondegenerate_f32(label: &str, what: &str, v: &[f32]) {
    let nonzero = v.iter().filter(|x| **x != 0.0).count();
    assert!(
        nonzero * 2 > v.len(),
        "{label}: {what} is mostly zero ({nonzero}/{} nonzero); the identity would be vacuous",
        v.len()
    );
    assert!(
        v.iter().all(|x| x.is_finite()),
        "{label}: {what} contains non-finite values; the corpus left the operating range"
    );
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DqParams {
    n_v: u32,
    d_k: u32,
    d_v: u32,
    key_dim: u32,
    v_per_k: u32,
    pad0: u32,
    pad1: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DgParams {
    n_v: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DrParams {
    heads: u32,
    d_k: u32,
    d_v: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DoParams {
    n_v: u32,
    d_v: u32,
    pad0: u32,
    eps: f32,
}

struct HeadCase {
    label: &'static str,
    n_v: usize,
    v_per_k: usize,
    d_k: usize,
    d_v: usize,
}

const HEAD_STEPS_CARRIED_STATE_SEES_DECAY_AND_HISTORY_DEFECTS: usize = 3;

#[test]
fn q3w_delta_head_fused_is_bit_identical_to_the_split_gating_recurrent_out_chain() {
    let ctx = ctx();
    let src = shipped("q3d:delta");
    for e in [
        HEAD_FUSED_ENTRY,
        "q3w_delta_qkv",
        "q3w_delta_gating",
        "q3w_delta_out",
    ] {
        assert_entry(&src, e);
    }
    let (rec_entry, rec_lanes) = delta_recurrent_kernel();
    let cases = [
        HeadCase {
            label: "served nv48 vpk3 dk128 dv128 (qwen3.8 geometry)",
            n_v: 48,
            v_per_k: 3,
            d_k: 128,
            d_v: 128,
        },
        HeadCase {
            label: "ragged nv6 vpk3 dk32 dv48",
            n_v: 6,
            v_per_k: 3,
            d_k: 32,
            d_v: 48,
        },
        HeadCase {
            label: "tiny nv4 vpk1 dk16 dv16",
            n_v: 4,
            v_per_k: 1,
            d_k: 16,
            d_v: 16,
        },
    ];
    for c in cases {
        let key_dim = c.n_v / c.v_per_k * c.d_k;
        let conv_dim = 2 * key_dim + c.n_v * c.d_v;
        let mut r = Lcg::new(0xfa5e_d000 ^ ((c.n_v as u64) << 8) ^ c.d_k as u64);

        let dq_p = dispatch::uniform_from(
            ctx,
            "fh-dq-p",
            &DqParams {
                n_v: c.n_v as u32,
                d_k: c.d_k as u32,
                d_v: c.d_v as u32,
                key_dim: key_dim as u32,
                v_per_k: c.v_per_k as u32,
                pad0: 0,
                pad1: 0,
                scale: 1.0 / (c.d_k as f32).sqrt(),
            },
        );
        let dg_p = dispatch::uniform_from(
            ctx,
            "fh-dg-p",
            &DgParams {
                n_v: c.n_v as u32,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        );
        let dr_p = dispatch::uniform_from(
            ctx,
            "fh-dr-p",
            &DrParams {
                heads: c.n_v as u32,
                d_k: c.d_k as u32,
                d_v: c.d_v as u32,
                pad0: 0,
            },
        );
        let do_p = dispatch::uniform_from(
            ctx,
            "fh-do-p",
            &DoParams {
                n_v: c.n_v as u32,
                d_v: c.d_v as u32,
                pad0: 0,
                eps: 1e-6,
            },
        );
        let alog = dispatch::storage_from_slice(ctx, "fh-alog", &r.f32s(c.n_v, 0.5));
        let dt = dispatch::storage_from_slice(ctx, "fh-dt", &r.f32s(c.n_v, 1.0));
        let norm_w = dispatch::storage_from_slice(
            ctx,
            "fh-normw",
            &pack_bf16(&r.f64s(c.d_v, 0.5).iter().map(|x| 1.0 + x).collect::<Vec<_>>()),
        );

        let state_words = c.n_v * c.d_k * c.d_v;
        let state_a = dispatch::storage_zeroed(ctx, "fh-state-a", (state_words * 4) as u64);
        let state_b = dispatch::storage_zeroed(ctx, "fh-state-b", (state_words * 4) as u64);
        let qg = dispatch::storage_zeroed(ctx, "fh-q", (c.n_v * c.d_k * 4) as u64);
        let kg = dispatch::storage_zeroed(ctx, "fh-k", (c.n_v * c.d_k * 4) as u64);
        let vg = dispatch::storage_zeroed(ctx, "fh-v", (c.n_v * c.d_v * 4) as u64);
        let g = dispatch::storage_zeroed(ctx, "fh-g", (c.n_v * 4) as u64);
        let beta = dispatch::storage_zeroed(ctx, "fh-beta", (c.n_v * 4) as u64);
        let core = dispatch::storage_zeroed(ctx, "fh-core", (c.n_v * c.d_v * 4) as u64);
        let gated_words = c.n_v * c.d_v / 2;
        let gated_a = dispatch::storage_zeroed(ctx, "fh-gated-a", (gated_words * 4) as u64);
        let gated_b = dispatch::storage_zeroed(ctx, "fh-gated-b", (gated_words * 4) as u64);

        let mut prev_state: Option<Vec<f32>> = None;
        for step in 0..HEAD_STEPS_CARRIED_STATE_SEES_DECAY_AND_HISTORY_DEFECTS {
            let mixed = dispatch::storage_from_slice(ctx, "fh-mixed", &r.f32s(conv_dim, 1.2));
            let ab = dispatch::storage_from_slice(ctx, "fh-ab", &pack_bf16(&r.f64s(2 * c.n_v, 1.5)));
            let z = dispatch::storage_from_slice(ctx, "fh-z", &pack_bf16(&r.f64s(c.n_v * c.d_v, 1.0)));

            dispatch::run(
                ctx,
                "fh-split",
                &src,
                "q3w_delta_qkv",
                &[(10, &mixed), (11, &qg), (12, &kg), (13, &vg), (14, &dq_p)],
                (c.n_v as u32, 1, 1),
            )
            .expect("split");
            dispatch::run(
                ctx,
                "fh-gating",
                &src,
                "q3w_delta_gating",
                &[
                    (20, &ab),
                    (21, &alog),
                    (22, &dt),
                    (23, &g),
                    (24, &beta),
                    (25, &dg_p),
                ],
                dispatch::workgroup_count_1d(ctx, c.n_v as u64, 64),
            )
            .expect("gating");
            let rec_grid_y = if rec_lanes > 0 {
                (c.d_v as u32).div_ceil(rec_lanes)
            } else {
                1
            };
            dispatch::run(
                ctx,
                "fh-recurrent",
                &src,
                rec_entry,
                &[
                    (30, &qg),
                    (31, &kg),
                    (32, &vg),
                    (33, &g),
                    (34, &beta),
                    (35, &core),
                    (36, &state_a),
                    (37, &dr_p),
                ],
                (c.n_v as u32, rec_grid_y, 1),
            )
            .expect("recurrent");
            dispatch::run(
                ctx,
                "fh-out",
                &src,
                "q3w_delta_out",
                &[
                    (40, &core),
                    (41, &norm_w),
                    (42, &z),
                    (43, &gated_a),
                    (44, &do_p),
                ],
                (c.n_v as u32, 1, 1),
            )
            .expect("out");

            dispatch::run(
                ctx,
                "fh-fused",
                &src,
                HEAD_FUSED_ENTRY,
                &[
                    (10, &mixed),
                    (14, &dq_p),
                    (20, &ab),
                    (21, &alog),
                    (22, &dt),
                    (25, &dg_p),
                    (36, &state_b),
                    (41, &norm_w),
                    (42, &z),
                    (43, &gated_b),
                    (44, &do_p),
                ],
                (c.n_v as u32, 1, 1),
            )
            .expect("fused head");

            let ga: Vec<u32> = dispatch::read_back(ctx, &gated_a, gated_words).expect("gated a");
            let gb: Vec<u32> = dispatch::read_back(ctx, &gated_b, gated_words).expect("gated b");
            let sa: Vec<f32> = dispatch::read_back(ctx, &state_a, state_words).expect("state a");
            let sb: Vec<f32> = dispatch::read_back(ctx, &state_b, state_words).expect("state b");
            let what_g = format!("gated step {step}");
            let what_s = format!("state step {step}");
            assert_words_eq(c.label, &what_g, &ga, &gb);
            assert_bits_eq(c.label, &what_s, &sa, &sb);
            assert_nondegenerate_f32(c.label, &what_s, &sa);
            assert!(
                ga.iter().any(|w| *w != 0),
                "{}: gated output is all zero at step {step}; the identity would be vacuous",
                c.label
            );
            if let Some(prev) = prev_state.as_ref() {
                assert!(
                    prev.iter().zip(sa.iter()).any(|(a, b)| a.to_bits() != b.to_bits()),
                    "{}: the recurrent state did not move between steps, so the carried-state \
                     half of this identity ran nothing",
                    c.label
                );
            }
            prev_state = Some(sa);
        }
        eprintln!(
            "[q3d-fused-identity] {}: {HEAD_FUSED_ENTRY} bit-identical to the 4-dispatch chain \
             ({rec_entry}) over {} steps",
            c.label, HEAD_STEPS_CARRIED_STATE_SEES_DECAY_AND_HISTORY_DEFECTS
        );
    }
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
struct AfParams {
    n_q_rows: u32,
    n_k_rows: u32,
    head_dim: u32,
    q_src_stride: u32,
    k_src_stride: u32,
    rot_half: u32,
    pad0: u32,
    eps: f32,
}

struct AttnCase {
    label: &'static str,
    n_q: usize,
    n_kv: usize,
    hd: usize,
    gate: bool,
    rot_half: usize,
    pos: i32,
}

#[test]
fn q3w_attn_qk_norm_rope_qcast_is_bit_identical_to_the_norm_pair_plus_cast() {
    let ctx = ctx();
    let src = shipped("q3d:attn");
    for e in [ATTN_FUSED_ENTRY, "q3w_attn_norm_rope"] {
        assert_entry(&src, e);
    }
    let cases = [
        AttnCase {
            label: "served nh24 nkv4 hd256 gated rot32 pos37 (qwen3.8 geometry)",
            n_q: 24,
            n_kv: 4,
            hd: 256,
            gate: true,
            rot_half: 32,
            pos: 37,
        },
        AttnCase {
            label: "tiny nh12 nkv2 hd64 ungated rot8 pos0",
            n_q: 12,
            n_kv: 2,
            hd: 64,
            gate: false,
            rot_half: 8,
            pos: 0,
        },
        AttnCase {
            label: "full-rotation nh3 nkv1 hd128 gated rot64 pos5",
            n_q: 3,
            n_kv: 1,
            hd: 128,
            gate: true,
            rot_half: 64,
            pos: 5,
        },
    ];
    for c in cases {
        let stride_q = if c.gate { 2 * c.hd } else { c.hd };
        let max_pos = (c.pos as usize) + 1;
        let mut r = Lcg::new(0xa77e_0000 ^ ((c.n_q as u64) << 8) ^ c.hd as u64);
        let q_raw = dispatch::storage_from_slice(
            ctx,
            "af-qraw",
            &pack_bf16(&r.f64s(c.n_q * stride_q, 1.3)),
        );
        let k_raw =
            dispatch::storage_from_slice(ctx, "af-kraw", &pack_bf16(&r.f64s(c.n_kv * c.hd, 1.3)));
        let qn = dispatch::storage_from_slice(
            ctx,
            "af-qn",
            &pack_bf16(&r.f64s(c.hd, 0.5).iter().map(|x| 1.0 + x).collect::<Vec<_>>()),
        );
        let kn = dispatch::storage_from_slice(
            ctx,
            "af-kn",
            &pack_bf16(&r.f64s(c.hd, 0.5).iter().map(|x| 1.0 + x).collect::<Vec<_>>()),
        );
        let cos = dispatch::storage_from_slice(ctx, "af-cos", &r.f32s(max_pos * c.rot_half, 1.0));
        let sin = dispatch::storage_from_slice(ctx, "af-sin", &r.f32s(max_pos * c.rot_half, 1.0));
        let pos = dispatch::storage_from_slice(ctx, "af-pos", &[c.pos]);
        let q_words = c.n_q * c.hd / 2;
        let k_words = c.n_kv * c.hd / 2;
        let q_pair = dispatch::storage_zeroed(ctx, "af-q-pair", (q_words * 4) as u64);
        let k_pair = dispatch::storage_zeroed(ctx, "af-k-pair", (k_words * 4) as u64);
        let q_fused = dispatch::storage_zeroed(ctx, "af-q-fused", (q_words * 4) as u64);
        let k_fused = dispatch::storage_zeroed(ctx, "af-k-fused", (k_words * 4) as u64);
        let qf32 = dispatch::storage_zeroed(ctx, "af-qf32", (c.n_q * c.hd * 4) as u64);

        let qp = dispatch::uniform_from(
            ctx,
            "af-qp",
            &ArParams {
                n_rows: c.n_q as u32,
                head_dim: c.hd as u32,
                src_stride: stride_q as u32,
                rot_half: c.rot_half as u32,
                pad0: 0,
                pad1: 0,
                pad2: 0,
                eps: 1e-6,
            },
        );
        let kp = dispatch::uniform_from(
            ctx,
            "af-kp",
            &ArParams {
                n_rows: c.n_kv as u32,
                head_dim: c.hd as u32,
                src_stride: c.hd as u32,
                rot_half: c.rot_half as u32,
                pad0: 0,
                pad1: 0,
                pad2: 0,
                eps: 1e-6,
            },
        );
        dispatch::run(
            ctx,
            "af-qnorm",
            &src,
            "q3w_attn_norm_rope",
            &[
                (0, &q_raw),
                (1, &qn),
                (2, &cos),
                (3, &sin),
                (4, &pos),
                (5, &q_pair),
                (6, &qp),
            ],
            (c.n_q as u32, 1, 1),
        )
        .expect("qnorm");
        dispatch::run(
            ctx,
            "af-knorm",
            &src,
            "q3w_attn_norm_rope",
            &[
                (0, &k_raw),
                (1, &kn),
                (2, &cos),
                (3, &sin),
                (4, &pos),
                (5, &k_pair),
                (6, &kp),
            ],
            (c.n_kv as u32, 1, 1),
        )
        .expect("knorm");

        let afp = dispatch::uniform_from(
            ctx,
            "af-p",
            &AfParams {
                n_q_rows: c.n_q as u32,
                n_k_rows: c.n_kv as u32,
                head_dim: c.hd as u32,
                q_src_stride: stride_q as u32,
                k_src_stride: c.hd as u32,
                rot_half: c.rot_half as u32,
                pad0: 0,
                eps: 1e-6,
            },
        );
        dispatch::run(
            ctx,
            "af-fused",
            &src,
            ATTN_FUSED_ENTRY,
            &[
                (0, &q_raw),
                (1, &qn),
                (2, &cos),
                (3, &sin),
                (4, &pos),
                (5, &q_fused),
                (40, &k_raw),
                (41, &kn),
                (42, &k_fused),
                (43, &qf32),
                (44, &afp),
            ],
            ((c.n_q + c.n_kv) as u32, 1, 1),
        )
        .expect("fused qk norm");

        let qa: Vec<u32> = dispatch::read_back(ctx, &q_pair, q_words).expect("q pair");
        let qb: Vec<u32> = dispatch::read_back(ctx, &q_fused, q_words).expect("q fused");
        let ka: Vec<u32> = dispatch::read_back(ctx, &k_pair, k_words).expect("k pair");
        let kb: Vec<u32> = dispatch::read_back(ctx, &k_fused, k_words).expect("k fused");
        let qf: Vec<f32> = dispatch::read_back(ctx, &qf32, c.n_q * c.hd).expect("q f32");
        assert_words_eq(c.label, "q", &qa, &qb);
        assert_words_eq(c.label, "k", &ka, &kb);
        let cast_of_pair: Vec<f32> = qa
            .iter()
            .flat_map(|w| {
                [
                    f32::from_bits((w & 0xffff) << 16),
                    f32::from_bits((w >> 16) << 16),
                ]
            })
            .collect();
        assert_bits_eq(
            c.label,
            "q_f32 (cast_bf16_to_f32 is a pure bf16 decode of the q words)",
            &cast_of_pair,
            &qf,
        );
        assert_nondegenerate_f32(c.label, "q_f32", &qf);
        assert!(
            ka.iter().any(|w| *w != 0),
            "{}: k output is all zero; the identity would be vacuous",
            c.label
        );
        eprintln!(
            "[q3d-fused-identity] {}: {ATTN_FUSED_ENTRY} bit-identical to qnorm+knorm+qcast",
            c.label
        );
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GbParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_off_words: u32,
    y_off_words: u32,
    pad0: u32,
    alpha: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MgParams {
    qkv_pairs: u32,
    z_pairs: u32,
    ab_pairs: u32,
    qkv_rows: u32,
    z_rows: u32,
    ab_rows: u32,
    fp8_row_words: u32,
    bf16_row_words: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct MergedCase {
    label: &'static str,
    qkv_rows: usize,
    z_rows: usize,
    ab_rows: usize,
    k: usize,
}

#[test]
fn q3w_gemv_dn_merged_is_bit_identical_to_the_three_projection_dispatches() {
    let ctx = ctx();
    let src = shipped("q3d:gemv_bf16");
    for e in [GEMV_MERGED_ENTRY, "q3w_gemv_fp8_rowscale", "q3w_gemv_bf16"] {
        assert_entry(&src, e);
    }
    let cases = [
        MergedCase {
            label: "even rows qkv20 z12 ab6 k64",
            qkv_rows: 20,
            z_rows: 12,
            ab_rows: 6,
            k: 64,
        },
        MergedCase {
            label: "odd tails qkv7 z9 ab5 k32",
            qkv_rows: 7,
            z_rows: 9,
            ab_rows: 5,
            k: 32,
        },
    ];
    for c in cases {
        let mut r = Lcg::new(0x9e37_0000_u64.wrapping_add((c.qkv_rows as u64) << 8) ^ c.k as u64);
        let fp8_row_words = c.k / 4;
        let bf16_row_words = c.k / 2;
        let w_qkv =
            dispatch::storage_from_slice(ctx, "mg-wqkv", &r.e4m3_words_no_nan(c.qkv_rows * fp8_row_words));
        let w_z =
            dispatch::storage_from_slice(ctx, "mg-wz", &r.e4m3_words_no_nan(c.z_rows * fp8_row_words));
        let w_ab = dispatch::storage_from_slice(
            ctx,
            "mg-wab",
            &pack_bf16(&r.f64s(c.ab_rows * c.k, 1.0)),
        );
        let s_qkv_raw = r.f32s(c.qkv_rows, 0.02).iter().map(|x| x.abs() + 0.001f32).collect::<Vec<_>>();
        let s_z_raw = r.f32s(c.z_rows, 0.02).iter().map(|x| x.abs() + 0.001f32).collect::<Vec<_>>();
        let s_qkv = dispatch::storage_from_slice(
            ctx,
            "mg-sqkv",
            &nv_kernels::shift_decode_fold::fold_scales_for_e4m3_shift_decode(&s_qkv_raw),
        );
        let s_z = dispatch::storage_from_slice(
            ctx,
            "mg-sz",
            &nv_kernels::shift_decode_fold::fold_scales_for_e4m3_shift_decode(&s_z_raw),
        );
        let x = dispatch::storage_from_slice(ctx, "mg-x", &pack_bf16(&r.f64s(c.k, 1.0)));
        let mk_y = |label: &str, rows: usize| {
            dispatch::storage_zeroed(ctx, label, (rows.div_ceil(2) * 4) as u64)
        };
        let y_qkv_a = mk_y("mg-yqkv-a", c.qkv_rows);
        let y_z_a = mk_y("mg-yz-a", c.z_rows);
        let y_ab_a = mk_y("mg-yab-a", c.ab_rows);
        let y_qkv_b = mk_y("mg-yqkv-b", c.qkv_rows);
        let y_z_b = mk_y("mg-yz-b", c.z_rows);
        let y_ab_b = mk_y("mg-yab-b", c.ab_rows);

        let gb = |rows: usize, row_words: usize, groups_x: u32| GbParams {
            n_rows: rows as u32,
            k_words: row_words as u32,
            groups_x,
            out_f32: 0,
            w_row_words: row_words as u32,
            x_off_words: 0,
            y_off_words: 0,
            pad0: 0,
            alpha: 1.0,
            pad1: 0,
            pad2: 0,
            pad3: 0,
        };
        for (label, w, sc, y, rows, row_words, entry) in [
            ("mg-un-qkv", &w_qkv, Some(&s_qkv), &y_qkv_a, c.qkv_rows, fp8_row_words, "q3w_gemv_fp8_rowscale"),
            ("mg-un-z", &w_z, Some(&s_z), &y_z_a, c.z_rows, fp8_row_words, "q3w_gemv_fp8_rowscale"),
            ("mg-un-ab", &w_ab, None, &y_ab_a, c.ab_rows, bf16_row_words, "q3w_gemv_bf16"),
        ] {
            let grid = dispatch::workgroup_count_1d(ctx, rows.div_ceil(2) as u64, 1);
            let p = dispatch::uniform_from(ctx, label, &gb(rows, row_words, grid.0));
            let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![(0, w), (1, &x), (2, &p), (3, y)];
            if let Some(sc) = sc {
                binds.push((4, sc));
            }
            dispatch::run(ctx, label, &src, entry, &binds, grid).expect(label);
        }

        let qkv_pairs = c.qkv_rows.div_ceil(2);
        let z_pairs = c.z_rows.div_ceil(2);
        let ab_pairs = c.ab_rows.div_ceil(2);
        let grid =
            dispatch::workgroup_count_1d(ctx, (qkv_pairs + z_pairs + ab_pairs) as u64, 1);
        let mp = dispatch::uniform_from(
            ctx,
            "mg-p",
            &MgParams {
                qkv_pairs: qkv_pairs as u32,
                z_pairs: z_pairs as u32,
                ab_pairs: ab_pairs as u32,
                qkv_rows: c.qkv_rows as u32,
                z_rows: c.z_rows as u32,
                ab_rows: c.ab_rows as u32,
                fp8_row_words: fp8_row_words as u32,
                bf16_row_words: bf16_row_words as u32,
                groups_x: grid.0,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        );
        dispatch::run(
            ctx,
            "mg-merged",
            &src,
            GEMV_MERGED_ENTRY,
            &[
                (0, &w_qkv),
                (1, &x),
                (3, &y_qkv_b),
                (4, &s_qkv),
                (5, &w_z),
                (6, &s_z),
                (7, &y_z_b),
                (8, &w_ab),
                (9, &y_ab_b),
                (10, &mp),
            ],
            grid,
        )
        .expect("merged gemv");

        for (what, ya, yb, rows) in [
            ("y_qkv", &y_qkv_a, &y_qkv_b, c.qkv_rows),
            ("y_z", &y_z_a, &y_z_b, c.z_rows),
            ("y_ab", &y_ab_a, &y_ab_b, c.ab_rows),
        ] {
            let a: Vec<u32> = dispatch::read_back(ctx, ya, rows.div_ceil(2)).expect(what);
            let b: Vec<u32> = dispatch::read_back(ctx, yb, rows.div_ceil(2)).expect(what);
            assert_words_eq(c.label, what, &a, &b);
            assert!(
                a.iter().any(|w| *w != 0),
                "{}: {what} is all zero; the identity would be vacuous",
                c.label
            );
        }
        eprintln!(
            "[q3d-fused-identity] {}: {GEMV_MERGED_ENTRY} bit-identical to the three projection \
             dispatches",
            c.label
        );
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SmParams {
    n_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct SiluQuantCase {
    label: &'static str,
    k: usize,
    global: f32,
}

#[test]
fn q3w_silu_mul_quant_is_bit_identical_to_silu_mul_then_quant_rows() {
    use nv_models::qwen3_5_moe_wgpu::{QuantRowsParams, SiluPairParams};
    let ctx = ctx();
    let quant_src = nv_models::qwen3_5_moe_wgpu::nvfp4_quant_source();
    let misc_src = shipped("q3d:misc");
    for e in ["q3w_quant_rows", "q3w_silu_mul_quant", "q3w_silu_mul_quant_l32"] {
        assert_entry(&quant_src, e);
    }
    assert_entry(&misc_src, "q3w_silu_mul");
    let cases = [
        SiluQuantCase {
            label: "served k17408 (qwen3.8 intermediate_size)",
            k: 17408,
            global: 0.37,
        },
        SiluQuantCase {
            label: "small k512",
            k: 512,
            global: 1.0,
        },
    ];
    let subgroup32 = nv_kernels::wgpu_backend::kernels::gemv_nvfp4_v2::subgroup32_ok(ctx);
    for c in cases {
        let k_blocks = c.k / 16;
        assert_eq!(k_blocks % 4, 0, "{}: k_blocks must be a multiple of 4", c.label);
        let mut r = Lcg::new(0x51f0_u64 ^ ((c.k as u64) << 4));
        let y_gate = dispatch::storage_from_slice(ctx, "sq-gate", &pack_bf16(&r.f64s(c.k, 2.0)));
        let y_up = dispatch::storage_from_slice(ctx, "sq-up", &pack_bf16(&r.f64s(c.k, 1.0)));
        let sel = dispatch::storage_from_slice(ctx, "sq-sel", &[0u32]);
        let globals = dispatch::storage_from_slice(ctx, "sq-glob", &[c.global]);
        let act = dispatch::storage_zeroed(ctx, "sq-act", (c.k * 2) as u64);
        let smp = dispatch::uniform_from(
            ctx,
            "sq-smp",
            &SmParams {
                n_words: (c.k / 2) as u32,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        );
        dispatch::run(
            ctx,
            "sq-silu",
            &misc_src,
            "q3w_silu_mul",
            &[(10, &y_gate), (11, &y_up), (12, &act), (13, &smp)],
            dispatch::workgroup_count_1d(ctx, (c.k / 2) as u64, 64),
        )
        .expect("silu_mul");
        let qp = dispatch::uniform_from(
            ctx,
            "sq-qp",
            &QuantRowsParams {
                k_blocks: k_blocks as u32,
                n_slots: 1,
                use_sel: 0,
                x_slot_stride_elems: 0,
            },
        );
        let pp = dispatch::uniform_from(
            ctx,
            "sq-pp",
            &SiluPairParams {
                u_off_elems: 0,
                ..Default::default()
            },
        );
        let xq_words = c.k / 2;
        let xs_words = k_blocks.div_ceil(4);
        let xq_a = dispatch::storage_zeroed(ctx, "sq-xq-a", (xq_words * 4) as u64);
        let xs_a = dispatch::storage_zeroed(ctx, "sq-xs-a", (xs_words * 4) as u64);
        dispatch::run(
            ctx,
            "sq-quant",
            &quant_src,
            "q3w_quant_rows",
            &[
                (10, &act),
                (11, &qp),
                (12, &xq_a),
                (13, &xs_a),
                (14, &sel),
                (15, &globals),
            ],
            ((k_blocks as u32).div_ceil(256).max(1), 1, 1),
        )
        .expect("quant_rows");
        let want_xq: Vec<u32> = dispatch::read_back(ctx, &xq_a, xq_words).expect("xq a");
        let want_xs: Vec<u32> = dispatch::read_back(ctx, &xs_a, xs_words).expect("xs a");
        assert!(
            want_xq.iter().any(|w| *w != 0),
            "{}: reference codes are all zero; the identity would be vacuous",
            c.label
        );

        let mut arms: Vec<(&str, u32)> = vec![("q3w_silu_mul_quant", 256)];
        if subgroup32 {
            arms.push(("q3w_silu_mul_quant_l32", 4));
            arms.push(("q3w_silu_mul_quant_l256", 32));
        }
        for (entry, wg_blocks) in arms {
            let xq_b = dispatch::storage_zeroed(ctx, "sq-xq-b", (xq_words * 4) as u64);
            let xs_b = dispatch::storage_zeroed(ctx, "sq-xs-b", (xs_words * 4) as u64);
            dispatch::run(
                ctx,
                "sq-fused",
                &quant_src,
                entry,
                &[
                    (11, &qp),
                    (12, &xq_b),
                    (13, &xs_b),
                    (14, &sel),
                    (15, &globals),
                    (16, &y_gate),
                    (17, &y_up),
                    (18, &pp),
                ],
                ((k_blocks as u32).div_ceil(wg_blocks).max(1), 1, 1),
            )
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
            let got_xq: Vec<u32> = dispatch::read_back(ctx, &xq_b, xq_words).expect("xq b");
            let got_xs: Vec<u32> = dispatch::read_back(ctx, &xs_b, xs_words).expect("xs b");
            let what_q = format!("{entry} codes");
            let what_s = format!("{entry} scales");
            assert_words_eq(c.label, &what_q, &want_xq, &got_xq);
            assert_words_eq(c.label, &what_s, &want_xs, &got_xs);
        }
        eprintln!(
            "[q3d-fused-identity] {}: silu_mul_quant arms bit-identical to silu_mul + \
             quant_rows (subgroup32={subgroup32})",
            c.label
        );
    }
}

struct TwoWeightCase {
    label: &'static str,
    n: usize,
    k: usize,
}

#[test]
fn q3w_gemv_nvfp4_mrow2_2w_is_bit_identical_to_the_gate_and_up_mrow2_dispatches() {
    use nv_models::qwen3_5_moe_wgpu::GemvNvfp4Params;
    let ctx = ctx();
    let src = nv_models::qwen3_5_moe_wgpu::nvfp4_v2_mrow2_source_cfg128x2_matching_the_dense_decode_route();
    for e in ["q3w_gemv_nvfp4_mrow2", "q3w_gemv_nvfp4_mrow2_2w"] {
        assert_entry(&src, e);
    }
    assert!(
        nv_kernels::wgpu_backend::kernels::gemv_nvfp4_v2::subgroup32_ok(ctx),
        "this adapter cannot run the mrow2 route at all, so the dense decode path never reaches \
         the 2w entry either; run this gate on the served subgroup-32 adapter instead of reading \
         a skip as coverage"
    );
    let cases = [
        TwoWeightCase {
            label: "served n17408 k5120 (qwen3.8 mlp gate/up geometry)",
            n: 17408,
            k: 5120,
        },
        TwoWeightCase {
            label: "odd-tail n30 k96",
            n: 30,
            k: 96,
        },
        TwoWeightCase {
            label: "small n64 k1024",
            n: 64,
            k: 1024,
        },
    ];
    const MROW2_ROWS_PER_GROUP_AT_CFG128X2_IS_FOUR_SUBGROUPS_TIMES_TWO_ROWS: u32 = 8;
    for c in cases {
        let k_blocks = c.k / 16;
        assert_eq!(k_blocks % 2, 0, "{}: mrow2 walks block pairs", c.label);
        let mut r = Lcg::new(0x2b2b_0000 ^ ((c.n as u64) << 8) ^ c.k as u64);
        let row_vec4 = k_blocks / 2;
        let w_words = c.n * row_vec4 * 4;
        let ws_words = nv_kernels::wgpu_backend::kernels::gemv_nvfp4::swizzled_scale_len(c.n, k_blocks) / 4;
        let mk_w = |r: &mut Lcg, tag: &str| {
            let w: Vec<u32> = (0..w_words).map(|_| (r.next() * (u32::MAX as f64 / 2.0)) as i64 as u32).collect();
            dispatch::storage_from_slice(ctx, tag, &w)
        };
        let wa = mk_w(&mut r, "2w-wa");
        let wb = mk_w(&mut r, "2w-wb");
        let wsa = dispatch::storage_from_slice(ctx, "2w-wsa", &r.e4m3_words_no_nan(ws_words));
        let wsb = dispatch::storage_from_slice(ctx, "2w-wsb", &r.e4m3_words_no_nan(ws_words));
        let xq: Vec<u32> = (0..c.k / 8).map(|_| (r.next() * (u32::MAX as f64 / 2.0)) as i64 as u32).collect();
        let xq = dispatch::storage_from_slice(ctx, "2w-xq", &xq);
        let xs = dispatch::storage_from_slice(ctx, "2w-xs", &r.e4m3_words_no_nan(k_blocks.div_ceil(4)));
        let sel = dispatch::storage_from_slice(ctx, "2w-sel", &[0u32]);
        let alpha_a = 0.37f32;
        let alpha_b = 1.13f32;
        let y_words = c.n.div_ceil(2);
        let ya_single = dispatch::storage_zeroed(ctx, "2w-ya-s", (y_words * 4) as u64);
        let yb_single = dispatch::storage_zeroed(ctx, "2w-yb-s", (y_words * 4) as u64);
        let ya_fused = dispatch::storage_zeroed(ctx, "2w-ya-f", (y_words * 4) as u64);
        let yb_fused = dispatch::storage_zeroed(ctx, "2w-yb-f", (y_words * 4) as u64);
        let grid = dispatch::workgroup_count_1d(
            ctx,
            c.n as u64,
            MROW2_ROWS_PER_GROUP_AT_CFG128X2_IS_FOUR_SUBGROUPS_TIMES_TWO_ROWS,
        );
        let params = |alpha: f32| GemvNvfp4Params {
            alpha,
            n_rows: c.n as u32,
            k_blocks: k_blocks as u32,
            k_tiles: k_blocks.div_ceil(4) as u32,
            groups_x: grid.0,
            w_e_stride_vec2: 0,
            sf_e_stride_bytes: 0,
            x_slot_stride_vec2: 0,
            xsf_slot_stride_bytes: 0,
            y_slot_stride_words: (c.n / 2) as u32,
            per_expert_alpha: 0,
            m_slots_sharing_expert_zero: 0,
        };
        let dummy = dispatch::storage_from_slice(ctx, "2w-dummy", &[1.0f32]);
        for (tag, w, ws, y, a) in [
            ("2w-single-a", &wa, &wsa, &ya_single, alpha_a),
            ("2w-single-b", &wb, &wsb, &yb_single, alpha_b),
        ] {
            let p = dispatch::uniform_from(ctx, tag, &params(a));
            dispatch::run(
                ctx,
                tag,
                &src,
                "q3w_gemv_nvfp4_mrow2",
                &[
                    (18, w),
                    (11, ws),
                    (19, &xq),
                    (13, &xs),
                    (14, &p),
                    (15, y),
                    (16, &sel),
                    (17, &dummy),
                ],
                grid,
            )
            .expect(tag);
        }
        let alphas = dispatch::storage_from_slice(ctx, "2w-alphas", &[alpha_a, alpha_b]);
        let p = dispatch::uniform_from(ctx, "2w-p", &params(alpha_a));
        dispatch::run(
            ctx,
            "2w-fused",
            &src,
            "q3w_gemv_nvfp4_mrow2_2w",
            &[
                (18, &wa),
                (11, &wsa),
                (19, &xq),
                (13, &xs),
                (14, &p),
                (15, &ya_fused),
                (16, &sel),
                (17, &alphas),
                (21, &wb),
                (22, &wsb),
                (23, &yb_fused),
            ],
            (grid.0, 2, 1),
        )
        .expect("2w fused");
        assert_eq!(
            grid.1, 1,
            "{}: the 2w entry owns grid y as its gate/up selector, so this shape must fit \
             grid x alone (the model route falls back to the pair when it does not)",
            c.label
        );
        for (what, ys, yf) in [("y_gate", &ya_single, &ya_fused), ("y_up", &yb_single, &yb_fused)] {
            let a: Vec<u32> = dispatch::read_back(ctx, ys, y_words).expect(what);
            let b: Vec<u32> = dispatch::read_back(ctx, yf, y_words).expect(what);
            assert_words_eq(c.label, what, &a, &b);
            assert!(
                a.iter().filter(|w| **w != 0).count() * 2 > y_words,
                "{}: {what} is mostly zero ({} of {y_words} nonzero); the identity would be vacuous",
                c.label,
                a.iter().filter(|w| **w != 0).count()
            );
        }
        eprintln!(
            "[q3d-fused-identity] {}: q3w_gemv_nvfp4_mrow2_2w bit-identical to the two mrow2 \
             dispatches",
            c.label
        );
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct KvWriteP {
    words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct KvFp8P {
    n_tokens: u32,
    n_kv: u32,
    head_dim: u32,
    ring: u32,
    pairs: u32,
    start: u32,
    slots: u32,
    reserved: u32,
}

struct KvFoldCase {
    label: &'static str,
    n_kv: usize,
    hd: usize,
    max_seq: usize,
    positions: [i32; 2],
}

#[test]
fn quantize_kv_fp8_kv_write_bf16_is_bit_identical_to_kv_write_plus_the_quantize_pair() {
    let ctx = ctx();
    let attn_src = shipped("q3d:attn");
    let kvq_src = nv_kernels::wgpu_backend::compose(nv_kernels::wgpu_backend::kernels::kv_fp8::WGSL);
    assert_entry(&attn_src, "q3w_kv_write");
    for e in ["quantize_kv_fp8", "quantize_kv_fp8_kv_write_bf16"] {
        assert_entry(&kvq_src, e);
    }
    let cases = [
        KvFoldCase {
            label: "served nkv4 hd256 (qwen3.8 attention geometry)",
            n_kv: 4,
            hd: 256,
            max_seq: 64,
            positions: [0, 37],
        },
        KvFoldCase {
            label: "tiny nkv2 hd32",
            n_kv: 2,
            hd: 32,
            max_seq: 8,
            positions: [3, 4],
        },
        KvFoldCase {
            label: "single-head nkv1 hd64",
            n_kv: 1,
            hd: 64,
            max_seq: 4,
            positions: [0, 1],
        },
    ];
    for c in cases {
        let words = c.n_kv * c.hd / 2;
        let cache_words = c.max_seq * words;
        let fp8_words = c.max_seq * c.n_kv * c.hd / 4;
        let scale_elems = c.max_seq * c.n_kv;
        let mut r = Lcg::new(0x4bf0_1d00 ^ ((c.n_kv as u64) << 8) ^ c.hd as u64);
        let mk_cache = |tag: &str, w: usize| dispatch::storage_zeroed(ctx, tag, (w * 4) as u64);
        let kc_a = mk_cache("kvf-kc-a", cache_words);
        let vc_a = mk_cache("kvf-vc-a", cache_words);
        let kc_b = mk_cache("kvf-kc-b", cache_words);
        let vc_b = mk_cache("kvf-vc-b", cache_words);
        let kc8_a = mk_cache("kvf-kc8-a", fp8_words);
        let vc8_a = mk_cache("kvf-vc8-a", fp8_words);
        let kc8_b = mk_cache("kvf-kc8-b", fp8_words);
        let vc8_b = mk_cache("kvf-vc8-b", fp8_words);
        let ksc_a = mk_cache("kvf-ksc-a", scale_elems);
        let vsc_a = mk_cache("kvf-vsc-a", scale_elems);
        let ksc_b = mk_cache("kvf-ksc-b", scale_elems);
        let vsc_b = mk_cache("kvf-vsc-b", scale_elems);
        let kvp = dispatch::uniform_from(
            ctx,
            "kvf-kvp",
            &KvWriteP {
                words: words as u32,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        );
        let qp = dispatch::uniform_from(
            ctx,
            "kvf-qp",
            &KvFp8P {
                n_tokens: 1,
                n_kv: c.n_kv as u32,
                head_dim: c.hd as u32,
                ring: 0,
                pairs: c.n_kv as u32,
                start: 0,
                slots: c.max_seq as u32,
                reserved: 0,
            },
        );
        for pos in c.positions {
            let k = dispatch::storage_from_slice(ctx, "kvf-k", &pack_bf16(&r.f64s(c.n_kv * c.hd, 1.4)));
            let v = dispatch::storage_from_slice(ctx, "kvf-v", &pack_bf16(&r.f64s(c.n_kv * c.hd, 0.9)));
            let pos_buf = dispatch::storage_from_slice(ctx, "kvf-pos", &[pos]);
            dispatch::run(
                ctx,
                "kvf-write",
                &attn_src,
                "q3w_kv_write",
                &[(10, &k), (11, &v), (12, &kc_a), (13, &vc_a), (14, &pos_buf), (15, &kvp)],
                dispatch::workgroup_count_1d(ctx, words as u64, 64),
            )
            .expect("kv_write");
            for (tag, x, out, sc) in [
                ("kvf-qk", &k, &kc8_a, &ksc_a),
                ("kvf-qv", &v, &vc8_a, &vsc_a),
            ] {
                dispatch::run(
                    ctx,
                    tag,
                    &kvq_src,
                    "quantize_kv_fp8",
                    &[(0, x), (1, out), (2, sc), (3, &pos_buf), (4, &qp)],
                    (c.n_kv as u32, 1, 1),
                )
                .expect(tag);
            }
            dispatch::run(
                ctx,
                "kvf-fused",
                &kvq_src,
                "quantize_kv_fp8_kv_write_bf16",
                &[
                    (0, &k),
                    (1, &kc8_b),
                    (2, &ksc_b),
                    (3, &pos_buf),
                    (4, &qp),
                    (9, &v),
                    (10, &vc8_b),
                    (11, &vsc_b),
                    (12, &kc_b),
                    (13, &vc_b),
                ],
                (c.n_kv as u32, 2, 1),
            )
            .expect("fused kv write+quant");

            for (what, a, b, len) in [
                ("kc bf16", &kc_a, &kc_b, cache_words),
                ("vc bf16", &vc_a, &vc_b, cache_words),
                ("kc8 fp8", &kc8_a, &kc8_b, fp8_words),
                ("vc8 fp8", &vc8_a, &vc8_b, fp8_words),
            ] {
                let wa: Vec<u32> = dispatch::read_back(ctx, a, len).expect(what);
                let wb: Vec<u32> = dispatch::read_back(ctx, b, len).expect(what);
                let what_pos = format!("{what} after pos {pos}");
                assert_words_eq(c.label, &what_pos, &wa, &wb);
            }
            for (what, a, b) in [("k scales", &ksc_a, &ksc_b), ("v scales", &vsc_a, &vsc_b)] {
                let sa: Vec<f32> = dispatch::read_back(ctx, a, scale_elems).expect(what);
                let sb: Vec<f32> = dispatch::read_back(ctx, b, scale_elems).expect(what);
                let what_pos = format!("{what} after pos {pos}");
                assert_bits_eq(c.label, &what_pos, &sa, &sb);
                assert!(
                    sa.iter().any(|x| *x != 0.0),
                    "{}: {what_pos} all zero; the identity would be vacuous",
                    c.label
                );
            }
            let kc: Vec<u32> = dispatch::read_back(ctx, &kc_a, cache_words).expect("kc");
            assert!(
                kc.iter().any(|w| *w != 0),
                "{}: bf16 cache all zero at pos {pos}; the identity would be vacuous",
                c.label
            );
        }
        eprintln!(
            "[q3d-fused-identity] {}: quantize_kv_fp8_kv_write_bf16 bit-identical to \
             kv_write + the fp8 quantize pair over positions {:?}",
            c.label, c.positions
        );
    }
}
