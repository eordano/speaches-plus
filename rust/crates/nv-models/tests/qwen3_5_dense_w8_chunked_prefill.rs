#![cfg(feature = "wgpu")]

mod common;
use common::greedy_after_prefill;
use common::env_lock;
use common::nvfp4_dense_lin as nvfp4;
use common::tiny_config_qwen35_dense as tiny_config;

use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseConfig;
use nv_models::qwen3_5_moe::LayerType;
use nv_models::qwen3_5_moe_wgpu::HostBf16Lin;
use common::LcgOddSeedShift33SignedUnitPacks as Lcg;
use common::unpack_bf16_bits;

const CONTINUATION_TOKENS: usize = 12;

const PROMPT_LENGTHS_SPANNING_CHUNK_BOUNDARY: [usize; 3] = [33, 23, 5];

const W8_GROUPS_COVERING_BOTH_I8_ENTRIES: [usize; 2] = [32, 0];

struct W8Env;

const W8_BIT_IDENTITY_HOLDS_ONLY_ON_THE_LEGACY_ARMS_COOP_TILED_WY_ALL_REASSOCIATE: [&str; 4] = [
    "NV_Q3D_KV_FP8",
    "NV_Q3D_PF_COOP",
    "NV_Q3D_PF_ATTN_TILED",
    "NV_Q3D_PF_SCAN_WY",
];

impl W8Env {
    fn set(mode: &str, group: usize) -> Self {
        std::env::set_var("NV_Q3D_WGPU_W8", mode);
        std::env::set_var("NV_Q3D_WGPU_W8_GROUP", group.to_string());
        for e in W8_BIT_IDENTITY_HOLDS_ONLY_ON_THE_LEGACY_ARMS_COOP_TILED_WY_ALL_REASSOCIATE {
            std::env::set_var(e, "0");
        }
        Self
    }
}

impl Drop for W8Env {
    fn drop(&mut self) {
        std::env::remove_var("NV_Q3D_WGPU_W8");
        std::env::remove_var("NV_Q3D_WGPU_W8_GROUP");
        for e in W8_BIT_IDENTITY_HOLDS_ONLY_ON_THE_LEGACY_ARMS_COOP_TILED_WY_ALL_REASSOCIATE {
            std::env::remove_var(e);
        }
    }
}

fn have_gpu() -> bool {
    match WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("[q3d-w8-pf] adapter: {}", ctx.info.name);
            true
        }
        Err(e) => {
            if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
                panic!(
                    "no wgpu adapter ({e}). This gate covers the i8 M-row prefill arm and refuses \
                     to report success without running; set NV_MODELS_ALLOW_SKIP=1 to skip it on \
                     purpose."
                );
            }
            eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): no wgpu adapter: {e}");
            false
        }
    }
}

fn subgroups_ok() -> bool {
    let Ok(ctx) = WgpuContext::shared() else {
        return false;
    };
    let ok = nv_kernels::wgpu_backend::kernels::gemv_nvfp4_v2::subgroup32_ok(ctx);
    if !ok {
        eprintln!(
            "[skip] adapter has no 32-wide subgroups; w8_enabled() gates the int8 arm off here \
             and portable adapters keep the nvfp4 M-row path, so this suite has nothing to gate"
        );
    }
    ok
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Q3q8mParams {
    n_rows: u32,
    k_elems: u32,
    groups_x: u32,
    groups_per_row: u32,
    group_shift: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    pad0: u32,
}

fn pack_i8(w: &[i8]) -> Vec<u32> {
    assert!(w.len().is_multiple_of(4), "int8 rows pack 4 per word");
    let mut out = vec![0u32; w.len() / 4];
    for (i, v) in w.iter().enumerate() {
        out[i / 4] |= ((*v as i32 as u32) & 0xff) << (8 * (i % 4));
    }
    out
}

fn pack_bf16_bits(vals: &[u16]) -> Vec<u32> {
    assert!(
        vals.len().is_multiple_of(2),
        "packed bf16 needs even length"
    );
    vals.chunks(2)
        .map(|c| (c[0] as u32) | ((c[1] as u32) << 16))
        .collect()
}

struct MkCase {
    n: usize,
    k: usize,
    group: usize,
    m: usize,
}

#[test]
fn the_i8_m_row_gemm_is_bit_identical_to_the_per_token_i8_gemv_on_every_slot() {
    let _g = env_lock();
    if !have_gpu() || !subgroups_ok() {
        return;
    }
    let ctx = WgpuContext::shared().expect("adapter probed above");
    let per_token_src = q3d::shipped_gemv_i8_source();

    let cases = [
        MkCase {
            n: 24,
            k: 64,
            group: 16,
            m: 16,
        },
        MkCase {
            n: 17,
            k: 64,
            group: 32,
            m: 5,
        },
        MkCase {
            n: 48,
            k: 256,
            group: 128,
            m: 2,
        },
        MkCase {
            n: 24,
            k: 64,
            group: 0,
            m: 16,
        },
        MkCase {
            n: 13,
            k: 32,
            group: 0,
            m: 3,
        },
    ];
    let mut grouped_covered = 0usize;
    let mut rowscale_covered = 0usize;
    for c in &cases {
        let (mk_src, mk_grouped, mk_rowscale) = q3d::shipped_prefill_gemm_i8(c.m);
        for e in [&mk_grouped, &mk_rowscale] {
            assert!(
                mk_src.contains(&format!("fn {e}(")),
                "the generated i8 M-row source no longer declares {e}; the entry moved and this \
                 gate is now testing nothing"
            );
        }
        let mut r = Lcg::new(0x98_1717 ^ ((c.n as u64) << 24) ^ ((c.k as u64) << 8) ^ c.group as u64);
        let w: Vec<i8> = (0..c.n * c.k)
            .map(|_| ((r.next_u32() % 255) as i32 - 127) as i8)
            .collect();
        let ns = if c.group > 0 {
            c.n * (c.k / c.group)
        } else {
            c.n
        };
        let s: Vec<f32> = (0..ns)
            .map(|_| 0.002 + 0.02 * (r.next_u32() % 1000) as f32 / 1000.0)
            .collect();
        let x_rows: Vec<Vec<u16>> = (0..c.m).map(|_| r.bf16_vec(c.k, 0.35)).collect();

        let p1 = Q3q8Params {
            n_rows: c.n as u32,
            k_elems: c.k as u32,
            groups_x: c.n.div_ceil(8) as u32,
            groups_per_row: c.k.checked_div(c.group).unwrap_or(1) as u32,
            group_shift: if c.group > 0 {
                (c.group / 4).trailing_zeros()
            } else {
                0
            },
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let per_token_entry = if c.group > 0 {
            "q3d_gemv_i8g"
        } else {
            "q3d_gemv_i8"
        };
        let y_row_words = c.n.div_ceil(2);
        let w_b = dispatch::storage_from_slice(ctx, "w", &pack_i8(&w));
        let s_b = dispatch::storage_from_slice(ctx, "s", &s);
        let p1_b = dispatch::uniform_from(ctx, "p1", &p1);
        let mut want_rows: Vec<Vec<u16>> = Vec::new();
        for xr in &x_rows {
            let x_b = dispatch::storage_from_slice(ctx, "x1", &pack_bf16_bits(xr));
            let y_b = dispatch::storage_zeroed(ctx, "y1", (y_row_words * 4) as u64);
            dispatch::run(
                ctx,
                "q3d-w8-pf-single",
                &per_token_src,
                per_token_entry,
                &[(0, &w_b), (1, &s_b), (2, &x_b), (3, &y_b), (4, &p1_b)],
                (p1.groups_x, 1, 1),
            )
            .expect("per-token i8 gemv dispatch");
            let words: Vec<u32> =
                dispatch::read_back(ctx, &y_b, y_row_words).expect("read back per-token y");
            want_rows.push(unpack_bf16_bits(&words, c.n));
        }

        let pm = Q3q8mParams {
            n_rows: p1.n_rows,
            k_elems: p1.k_elems,
            groups_x: p1.groups_x,
            groups_per_row: p1.groups_per_row,
            group_shift: p1.group_shift,
            x_stride_words: (c.k / 2) as u32,
            y_stride_words: y_row_words as u32,
            pad0: 0,
        };
        let x_all: Vec<u16> = x_rows.iter().flatten().copied().collect();
        let x_b = dispatch::storage_from_slice(ctx, "xm", &pack_bf16_bits(&x_all));
        let y_b = dispatch::storage_zeroed(ctx, "ym", (c.m * y_row_words * 4) as u64);
        let pm_b = dispatch::uniform_from(ctx, "pm", &pm);
        let mk_entry = if c.group > 0 { &mk_grouped } else { &mk_rowscale };
        dispatch::run(
            ctx,
            "q3d-w8-pf-mk",
            &mk_src,
            mk_entry,
            &[(0, &w_b), (1, &s_b), (2, &x_b), (3, &y_b), (4, &pm_b)],
            (pm.groups_x, 1, 1),
        )
        .expect("i8 M-row gemm dispatch");
        let words: Vec<u32> =
            dispatch::read_back(ctx, &y_b, c.m * y_row_words).expect("read back M-row y");

        let mut mismatched = 0usize;
        for (mi, want) in want_rows.iter().enumerate() {
            let got = unpack_bf16_bits(&words[mi * y_row_words..(mi + 1) * y_row_words], c.n);
            for (rr, (g, wv)) in got.iter().zip(want.iter()).enumerate() {
                if g != wv {
                    mismatched += 1;
                    if mismatched <= 8 {
                        eprintln!(
                            "[q3d-w8-pf] n={} k={} group={} m={} slot {mi} row {rr}: got \
                             0x{g:04x} want 0x{wv:04x}",
                            c.n, c.k, c.group, c.m
                        );
                    }
                }
            }
        }
        eprintln!(
            "[q3d-w8-pf] n={} k={} group={} m={} ({mk_entry}): {mismatched} of {} output lanes \
             differ from the per-token entry",
            c.n,
            c.k,
            c.group,
            c.m,
            c.m * c.n
        );
        assert_eq!(
            mismatched, 0,
            "the i8 M-row entry {mk_entry} must be BIT-IDENTICAL to {per_token_entry} run once \
             per slot: both stride k/4 weight words by 32 lanes, chain the same four fma per \
             word, and reduce with the same subgroupShuffleXor ladder, so there is no rounding \
             freedom between them and any differing bit is a slot-stride or indexing defect"
        );
        if c.group > 0 {
            grouped_covered += 1;
        } else {
            rowscale_covered += 1;
        }
    }
    assert!(
        grouped_covered >= 2 && rowscale_covered >= 2,
        "this gate must exercise BOTH generated i8 M-row entries; covered grouped \
         {grouped_covered} rowscale {rowscale_covered}"
    );
}

fn bf16_lin(r: &mut Lcg, n: usize, k: usize, scale: f32) -> HostBf16Lin {
    HostBf16Lin {
        w: r.bf16_vec(n * k, scale),
        n,
        k,
    }
}

fn norm_vec(r: &mut Lcg, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| half::bf16::from_f32(1.0 + 0.1 * r.next_f32()).to_bits())
        .collect()
}

fn tiny_nvfp4_weights(cfg: &Qwen3_5DenseConfig, seed: u64) -> q3d::HostDenseWeights {
    let mut r = Lcg::new(seed);
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let hd = cfg.head_dim;
    let n_k = cfg.linear_num_key_heads;
    let n_v = cfg.linear_num_value_heads;
    let d_k = cfg.linear_key_head_dim;
    let d_v = cfg.linear_value_head_dim;
    let key_dim = n_k * d_k;
    let value_dim = n_v * d_v;
    let conv_dim = 2 * key_dim + value_dim;
    let ks = cfg.linear_conv_kernel_dim;

    let mut layers = Vec::new();
    for li in 0..cfg.num_hidden_layers {
        let mixer = match cfg.layer_types[li] {
            LayerType::LinearAttention => {
                q3d::HostDenseMixer::Delta(Box::new(nv_models::qwen3_5_moe_wgpu::HostDeltaNet {
                    in_proj_qkv: bf16_lin(&mut r, conv_dim, hidden, 0.12),
                    in_proj_z: bf16_lin(&mut r, value_dim, hidden, 0.12),
                    in_proj_ab: bf16_lin(&mut r, 2 * n_v, hidden, 0.12),
                    conv1d: r.f32_vec(conv_dim * ks, 0.4),
                    a_log: r.f32_vec(n_v, 0.5),
                    dt_bias: r.f32_vec(n_v, 0.5),
                    norm_w: norm_vec(&mut r, d_v),
                    out_proj: bf16_lin(&mut r, hidden, value_dim, 0.12),
                }))
            }
            LayerType::FullAttention => {
                let q_out = cfg.num_attention_heads * hd * 2;
                let kv_out = cfg.num_key_value_heads * hd;
                q3d::HostDenseMixer::Attn(Box::new(q3d::HostDenseAttention {
                    q: nvfp4(bf16_lin(&mut r, q_out, hidden, 0.12)),
                    k: nvfp4(bf16_lin(&mut r, kv_out, hidden, 0.12)),
                    v: nvfp4(bf16_lin(&mut r, kv_out, hidden, 0.12)),
                    o: nvfp4(bf16_lin(&mut r, hidden, cfg.num_attention_heads * hd, 0.12)),
                    q_norm: norm_vec(&mut r, hd),
                    k_norm: norm_vec(&mut r, hd),
                }))
            }
        };
        layers.push(q3d::HostDenseLayer {
            input_ln: norm_vec(&mut r, hidden),
            post_attn_ln: norm_vec(&mut r, hidden),
            mixer,
            delta_fp8: Default::default(),
            mlp: q3d::HostDenseMlp {
                gate: nvfp4(bf16_lin(&mut r, inter, hidden, 0.15)),
                up: nvfp4(bf16_lin(&mut r, inter, hidden, 0.15)),
                down: nvfp4(bf16_lin(&mut r, hidden, inter, 0.15)),
            },
        });
    }

    q3d::HostDenseWeights {
        embed: r.bf16_vec(cfg.vocab_size * hidden, 0.6),
        final_norm: norm_vec(&mut r, hidden),
        lm_head: r.bf16_vec(cfg.vocab_size * hidden, 0.2),
        layers,
    }
}

#[test]
fn w8_chunked_prefill_stays_enabled_and_reproduces_the_per_token_replay_bit_for_bit() {
    let _g = env_lock();
    if !have_gpu() || !subgroups_ok() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_nvfp4_weights(&cfg, 0xd15e_9b00_0098);

    for group in W8_GROUPS_COVERING_BOTH_I8_ENTRIES {
        let _e = W8Env::set("all", group);
        let mut gpu = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 96).expect("build wgpu model");

        let m = gpu.prefill_chunk_len();
        assert!(
            m >= 2,
            "NV_Q3D_WGPU_W8=all group={group} reported prefill_chunk_len()={m}: the i8 M-row arm \
             is absent again and the W8 graph fell back to replaying prompts one token per \
             command buffer -- the exact regression task #98 removed"
        );
        eprintln!(
            "[q3d-w8-pf] group={group} m={m}, prefill passes/chunk={}, decode passes/token={}",
            gpu.prefill_pass_count(),
            gpu.pass_count()
        );

        for len in PROMPT_LENGTHS_SPANNING_CHUNK_BOUNDARY {
            let tokens: Vec<u32> = (0..len as u32).map(|i| (i * 7 + 3) % 64).collect();
            let (ids_chunked, logits_chunked) = greedy_after_prefill(&mut gpu, &tokens, true, CONTINUATION_TOKENS);
            let (ids_replay, logits_replay) = greedy_after_prefill(&mut gpu, &tokens, false, CONTINUATION_TOKENS);
            let bit_diff = logits_chunked
                .iter()
                .zip(logits_replay.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let worst = logits_chunked
                .iter()
                .zip(logits_replay.iter())
                .fold(0f32, |a, (c, r)| a.max((c - r).abs()));
            eprintln!(
                "[q3d-w8-pf] group={group} prompt {len:>3} tokens ({} prefilled): {bit_diff} of \
                 {} logit lanes differ, max_abs {worst:.6}; chunked {ids_chunked:?} replay \
                 {ids_replay:?}",
                tokens.len() - 1,
                logits_chunked.len()
            );
            assert_eq!(
                ids_chunked, ids_replay,
                "W8 (group={group}) chunked prefill and per-token replay produced different \
                 greedy continuations at a {len}-token prompt"
            );
            assert_eq!(
                bit_diff, 0,
                "W8 (group={group}) chunked prefill and per-token replay must leave a \
                 BIT-IDENTICAL KV cache: the i8 M-row entries reproduce the per-token int8 \
                 accumulation order exactly (same 32-lane stride, same fma chain, same shuffle \
                 ladder) and every other pass in this graph already holds bit identity in the \
                 nvfp4 chunked-prefill gate. {bit_diff} of {} lanes differ at a {len}-token \
                 prompt, max_abs {worst}. Do not relax this to argmax: a slot-stride defect that \
                 makes every chunk row read token 0 leaves the argmax of this tiny model \
                 untouched and moves only these bits.",
                logits_chunked.len()
            );

            let mut st = q3d::RefState::new(&cfg);
            let mut want = Vec::new();
            for t in &tokens {
                want = q3d::reference_step(&cfg, &hw, &mut st, *t).expect("cpu reference step");
            }
            let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
            let rel = logits_chunked
                .iter()
                .zip(want.iter())
                .fold(0f32, |a, (g, w)| a.max((g - w).abs() / scale));
            eprintln!("[q3d-w8-pf] group={group} prompt {len:>3} tokens vs CPU reference: rel {rel:.4}");
            assert!(
                rel < 0.05,
                "W8 (group={group}) chunked prefill diverged from the CPU reference at {len} \
                 tokens (rel {rel})"
            );
        }
    }
}
