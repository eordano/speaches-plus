#![cfg(feature = "wgpu")]

mod common;
use common::LcgOddSeedShift33SignedUnit as Lcg;
use common::{bf16_lin, norm_vec};

use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_dense_wgpu::{Qwen3_5DenseConfig, Qwen3_5DenseWgpu};
use nv_models::qwen3_5_moe::LayerType;
use nv_models::qwen3_5_moe_wgpu as q3w;

const TINY_MAX_SEQ: usize = 64;

fn have_gpu() -> bool {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("[wgpu] adapter: {}", ctx.info.name);
            true
        }
        Err(e) => {
            eprintln!("[no-adapter] {e}");
            false
        }
    }
}

fn require_gpu() {
    assert!(
        have_gpu(),
        "no wgpu adapter: a skipped bit-identity proof reads as a passed one, so this suite \
         fails instead of skipping"
    );
}

fn ids_from(vocab: usize, n: usize, salt: u32) -> Vec<u32> {
    (0..n)
        .map(|i| ((i as u32 * 7 + salt * 13 + 1) % (vocab as u32 - 1)) + 1)
        .collect()
}

fn bit_diff(want: &[f32], got: &[f32]) -> usize {
    want.iter()
        .zip(got)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

fn tiny_config() -> Qwen3_5DenseConfig {
    Qwen3_5DenseConfig {
        intermediate_size: 96,
        max_position_embeddings: TINY_MAX_SEQ,
        ..common::tiny_config_qwen35_dense()
    }
}

fn tiny_weights(cfg: &Qwen3_5DenseConfig, seed: u64) -> q3d::HostDenseWeights {
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
                q3d::HostDenseMixer::Delta(Box::new(q3w::HostDeltaNet {
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
                    q: bf16_lin(&mut r, q_out, hidden, 0.12).into(),
                    k: bf16_lin(&mut r, kv_out, hidden, 0.12).into(),
                    v: bf16_lin(&mut r, kv_out, hidden, 0.12).into(),
                    o: bf16_lin(&mut r, hidden, cfg.num_attention_heads * hd, 0.12).into(),
                    q_norm: norm_vec(&mut r, hd),
                    k_norm: norm_vec(&mut r, hd),
                    qkv_bias_bf16: vec![],
                }))
            }
        };
        layers.push(q3d::HostDenseLayer {
            input_ln: norm_vec(&mut r, hidden),
            post_attn_ln: norm_vec(&mut r, hidden),
            mixer,
            mlp: q3d::HostDenseMlp {
                gate: bf16_lin(&mut r, inter, hidden, 0.15).into(),
                up: bf16_lin(&mut r, inter, hidden, 0.15).into(),
                down: bf16_lin(&mut r, hidden, inter, 0.15).into(),
            },
            delta_fp8: q3d::DeltaFp8::default(),
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
fn a_rewound_checkpoint_continues_bit_identical_to_a_cold_prefill_of_the_shared_prefix() {
    require_gpu();
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x9e37_79b9_7f4a);
    let prompt = ids_from(cfg.vocab_size, 9, 3);
    let generated = ids_from(cfg.vocab_size, 6, 7);
    let delta = ids_from(cfg.vocab_size, 5, 11);
    let tail = ids_from(cfg.vocab_size, 4, 17);

    let mut cold = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("build cold");
    for &t in &prompt {
        cold.prefill_step(t).expect("cold prefill step");
    }
    let want: Vec<(u32, Vec<f32>)> = delta
        .iter()
        .chain(&tail)
        .map(|&t| cold.decode_step_logits(t).expect("cold continuation step"))
        .collect();

    let mut warm = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("build warm");
    for &t in &prompt {
        warm.prefill_step(t).expect("warm prefill step");
    }
    assert!(
        warm.save_resume_checkpoint().expect("save checkpoint"),
        "a text-only decoder with no MTP head must accept a resume checkpoint"
    );
    assert_eq!(warm.resume_checkpoint_pos(), Some(prompt.len()));
    for &t in &generated {
        warm.decode_step(t).expect("warm simulated generation step");
    }
    assert_eq!(warm.current_pos(), prompt.len() + generated.len());

    let (_, contaminated) = warm
        .decode_step_logits(delta[0])
        .expect("pre-rewind divergence probe");
    assert!(
        bit_diff(&want[0].1, &contaminated) > 0,
        "stepping the delta WITHOUT the rewind matched the cold continuation bit for bit at the \
         divergence point, so this suite would pass with rewind_to as a no-op and certifies \
         nothing"
    );

    assert!(
        warm.rewind_to(prompt.len()).expect("rewind to checkpoint"),
        "rewind_to must accept exactly the saved position"
    );
    assert_eq!(warm.current_pos(), prompt.len());
    for (i, (want_tok, want_row)) in want.iter().enumerate() {
        let t = if i < delta.len() {
            delta[i]
        } else {
            tail[i - delta.len()]
        };
        let (got_tok, got_row) = warm.decode_step_logits(t).expect("rewound continuation step");
        assert_eq!(
            bit_diff(want_row, &got_row),
            0,
            "continuation step {i}: the rewound stream drifted from the cold prefill, so the \
             DeltaNet recurrent or conv state was not restored to the end-of-prefill snapshot. \
             {}",
            q3d::RESUME_CHECKPOINT_COPIES_DELTANET_STATE_BECAUSE_ONLY_ATTENTION_KV_REWINDS_BY_POS
        );
        assert_eq!(got_tok, *want_tok, "continuation step {i} argmax");
    }

    let generated2 = ids_from(cfg.vocab_size, 3, 23);
    for &t in &generated2 {
        warm.decode_step(t).expect("second simulated generation step");
    }
    assert!(
        warm.rewind_to(prompt.len()).expect("second rewind"),
        "one checkpoint must serve every later turn that still shares its prefix"
    );
    let (got_tok, got_row) = warm
        .decode_step_logits(delta[0])
        .expect("second rewound continuation step");
    assert_eq!(
        bit_diff(&want[0].1, &got_row),
        0,
        "the second rewind from the same checkpoint drifted: the restore consumed or mutated \
         the snapshot"
    );
    assert_eq!(got_tok, want[0].0, "second-rewind argmax");
}

#[test]
fn a_mid_grid_rewind_continues_bit_identical_to_a_cold_prefill_of_the_grid_boundary_prefix() {
    require_gpu();
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x0dd5_eed5_1234);
    let prompt = ids_from(cfg.vocab_size, 24, 3);
    let boundary = 16usize;
    let delta = ids_from(cfg.vocab_size, 5, 11);

    let mut cold = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("build cold");
    for &t in &prompt[..boundary] {
        cold.prefill_step(t).expect("cold prefill step");
    }
    let want: Vec<(u32, Vec<f32>)> = delta
        .iter()
        .map(|&t| cold.decode_step_logits(t).expect("cold continuation step"))
        .collect();

    let mut warm = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("build warm");
    for (i, &t) in prompt.iter().enumerate() {
        if i > 0 && i % 8 == 0 {
            assert!(
                warm.save_grid_checkpoint().expect("save grid checkpoint"),
                "a text-only decoder with no MTP head must accept a grid checkpoint"
            );
        }
        warm.prefill_step(t).expect("warm prefill step");
    }
    assert!(warm.save_resume_checkpoint().expect("save end checkpoint"));
    assert_eq!(warm.grid_checkpoint_at_or_below(7), None);
    assert_eq!(warm.grid_checkpoint_at_or_below(15), Some(8));
    assert_eq!(warm.grid_checkpoint_at_or_below(23), Some(16));
    for &t in &ids_from(cfg.vocab_size, 6, 7) {
        warm.decode_step(t).expect("warm simulated generation step");
    }

    let (_, contaminated) = warm
        .decode_step_logits(delta[0])
        .expect("pre-rewind divergence probe");
    assert!(
        bit_diff(&want[0].1, &contaminated) > 0,
        "stepping the delta WITHOUT the rewind matched the cold continuation bit for bit, so a \
         no-op rewind would pass and this gate certifies nothing"
    );

    assert!(
        warm.rewind_to(boundary).expect("rewind to grid boundary"),
        "rewind_to must accept a saved grid boundary"
    );
    assert_eq!(warm.current_pos(), boundary);
    assert_eq!(
        warm.resume_checkpoint_pos(),
        None,
        "{}",
        q3d::REWIND_DROPS_CHECKPOINTS_ABOVE_THE_TARGET_BECAUSE_THE_RESUMED_STREAM_DIVERGES_THERE
    );
    assert_eq!(
        warm.grid_checkpoint_at_or_below(TINY_MAX_SEQ),
        Some(boundary),
        "the boundary rewound to is still the live stream's state and must survive"
    );
    for (i, (want_tok, want_row)) in want.iter().enumerate() {
        let (got_tok, got_row) = warm
            .decode_step_logits(delta[i])
            .expect("rewound continuation step");
        assert_eq!(
            bit_diff(want_row, &got_row),
            0,
            "diverging-suffix step {i}: the mid-grid rewound stream drifted from the cold prefill \
             of the shared boundary prefix. {}",
            q3d::RESUME_CHECKPOINT_COPIES_DELTANET_STATE_BECAUSE_ONLY_ATTENTION_KV_REWINDS_BY_POS
        );
        assert_eq!(got_tok, *want_tok, "diverging-suffix step {i} argmax");
    }
    assert!(
        !warm.rewind_to(boundary + 8).expect("rewind above the earlier target"),
        "{}",
        q3d::REWIND_DROPS_CHECKPOINTS_ABOVE_THE_TARGET_BECAUSE_THE_RESUMED_STREAM_DIVERGES_THERE
    );
}

#[test]
fn the_grid_cap_evicts_the_lowest_boundary_and_reset_plus_new_stream_refuse_stale_entries() {
    require_gpu();
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x5eed_0ca9_0042);
    let cap =
        q3d::GRID_SLOT_CAP_BOUNDS_VRAM_AT_CAP_DELTANET_CLONES_AND_EVICTS_THE_LOWEST_BOUNDARY_BECAUSE_A_DIVERGING_SUFFIX_REWINDS_TO_THE_HIGHEST_BOUNDARY_UNDER_ITS_LCP;
    let prompt = ids_from(cfg.vocab_size, cap + 2, 5);

    let mut m = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("build");
    for &t in &prompt {
        m.prefill_step(t).expect("prefill step");
        m.save_grid_checkpoint().expect("save grid checkpoint");
    }
    assert_eq!(
        m.grid_checkpoint_at_or_below(TINY_MAX_SEQ),
        Some(cap + 2),
        "the newest boundary must always be resident"
    );
    assert_eq!(
        m.grid_checkpoint_at_or_below(2),
        None,
        "saving {} boundaries into {cap} slots must evict the lowest ones, and the eviction \
         must be loud, not a silent wrong-age hit",
        cap + 2
    );
    assert_eq!(m.grid_checkpoint_at_or_below(3), Some(3));
    assert!(m.rewind_to(3).expect("rewind to the oldest surviving boundary"));

    m.reset().expect("reset");
    assert_eq!(
        m.grid_checkpoint_at_or_below(TINY_MAX_SEQ),
        None,
        "reset replaces the whole stream, so a surviving grid entry would rewind into state \
         that no longer exists"
    );
    for &t in &ids_from(cfg.vocab_size, 8, 9) {
        m.prefill_step(t).expect("interleaved second-stream prefill step");
    }
    assert!(
        !m.rewind_to(3).expect("rewind into the forgotten stream"),
        "an interleaved different conversation must degrade to cold, never rewind into the \
         previous stream's grid"
    );
}

#[test]
fn a_checkpoint_is_refused_at_any_other_pos_and_forgotten_by_reset() {
    require_gpu();
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x51ee_d5ee_d001);
    let prompt = ids_from(cfg.vocab_size, 8, 5);

    let mut m = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("build");
    assert!(
        !m.rewind_to(0).expect("rewind with no checkpoint"),
        "with no checkpoint saved there is nothing to rewind to"
    );
    for &t in &prompt {
        m.prefill_step(t).expect("prefill step");
    }
    assert!(m.save_resume_checkpoint().expect("save"));
    let p = prompt.len();
    for &t in &ids_from(cfg.vocab_size, 4, 9) {
        m.decode_step(t).expect("decode step");
    }
    assert!(
        !m.rewind_to(p - 1).expect("rewind below the checkpoint"),
        "the recurrent state at {}-1 was never saved; admitting it would replay KV past the \
         divergence as if it belonged to this prompt",
        p
    );
    assert!(
        !m.rewind_to(p + 1).expect("rewind past the checkpoint"),
        "one token past the checkpoint was never saved either"
    );
    assert!(m.rewind_to(p).expect("rewind to the checkpoint"));
    assert_eq!(m.current_pos(), p);

    m.reset().expect("reset");
    assert_eq!(
        m.resume_checkpoint_pos(),
        None,
        "reset replaces the whole stream, so a surviving checkpoint would rewind into state \
         that no longer exists"
    );
    assert!(!m.rewind_to(p).expect("rewind after reset"));
}
