#![cfg(feature = "wgpu")]

mod common;
use common::bf16_lin;
use common::LcgOddSeedShift33SignedUnit as Lcg;
use common::norm_vec;
use common::nvfp4;
mod hub_snapshot;

use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_dense_wgpu::{Qwen3_5DenseConfig, Qwen3_5DenseWgpu};
use nv_models::qwen3_5_moe::{LayerType, Qwen3MoeConfig};
use nv_models::qwen3_5_moe_wgpu as q3w;
use nv_models::qwen3_5_moe_wgpu::{HostBf16Lin, Qwen3MoeWgpu};

static ENV_LOCK_THESE_TESTS_MUTATE_PROCESS_GLOBAL_ENV_SO_MUST_NOT_INTERLEAVE:
    std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK_THESE_TESTS_MUTATE_PROCESS_GLOBAL_ENV_SO_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

const DENSE_M_ENV: &str = "NV_WGPU_PREFILL_M";

const MOE_M_ENV: &str = "NV_QWEN35MOE_WGPU_PREFILL_M";

const VERIFY_ROWS_8_KEEPS_THE_TINY_MAX_SEQ_64_ABLE_TO_HOLD_A_PROMPT_PLUS_TWO_ROUNDS: usize = 8;

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

fn dense_tiny_config() -> Qwen3_5DenseConfig {
    Qwen3_5DenseConfig {
        intermediate_size: 96,
        max_position_embeddings: TINY_MAX_SEQ,
        ..common::tiny_config_qwen35_dense()
    }
}

fn dense_tiny_weights(cfg: &Qwen3_5DenseConfig, seed: u64) -> q3d::HostDenseWeights {
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

fn moe_tiny_config() -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        hidden_size: 128,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 32,
        moe_intermediate_size: 64,
        shared_expert_intermediate_size: 64,
        num_experts: 8,
        num_experts_per_tok: 2,
        vocab_size: 64,
        max_position_embeddings: TINY_MAX_SEQ,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        partial_rotary_factor: 0.5,
        bos_token_id: 0,
        eos_token_id: 1,
        layer_types: vec![
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
        ],
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        attn_output_gate: true,
        tie_word_embeddings: false,
    }
}

fn moe_tiny_weights(cfg: &Qwen3MoeConfig, seed: u64) -> q3w::HostWeights {
    let mut r = Lcg::new(seed);
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let sinter = cfg.shared_expert_intermediate_size;
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
            LayerType::LinearAttention => q3w::HostMixer::Delta(Box::new(q3w::HostDeltaNet {
                in_proj_qkv: bf16_lin(&mut r, conv_dim, hidden, 0.12),
                in_proj_z: bf16_lin(&mut r, value_dim, hidden, 0.12),
                in_proj_ab: bf16_lin(&mut r, 2 * n_v, hidden, 0.12),
                conv1d: r.f32_vec(conv_dim * ks, 0.4),
                a_log: r.f32_vec(n_v, 0.5),
                dt_bias: r.f32_vec(n_v, 0.5),
                norm_w: norm_vec(&mut r, d_v),
                out_proj: bf16_lin(&mut r, hidden, value_dim, 0.12),
            })),
            LayerType::FullAttention => {
                let q_out = cfg.num_attention_heads * hd * 2;
                let kv_out = cfg.num_key_value_heads * hd;
                q3w::HostMixer::Attn(Box::new(q3w::HostAttention {
                    q: nvfp4(&mut r, q_out, hidden, 0.12),
                    k: nvfp4(&mut r, kv_out, hidden, 0.12),
                    v: nvfp4(&mut r, kv_out, hidden, 0.12),
                    o: nvfp4(&mut r, hidden, cfg.num_attention_heads * hd, 0.12),
                    q_norm: norm_vec(&mut r, hd),
                    k_norm: norm_vec(&mut r, hd),
                }))
            }
        };
        let gates: Vec<_> = (0..cfg.num_experts)
            .map(|_| nvfp4(&mut r, inter, hidden, 0.15))
            .collect();
        let ups: Vec<_> = (0..cfg.num_experts)
            .map(|_| nvfp4(&mut r, inter, hidden, 0.15))
            .collect();
        let downs: Vec<_> = (0..cfg.num_experts)
            .map(|_| nvfp4(&mut r, hidden, inter, 0.15))
            .collect();
        layers.push(q3w::HostLayer {
            input_ln: norm_vec(&mut r, hidden),
            post_attn_ln: norm_vec(&mut r, hidden),
            mixer,
            moe: q3w::HostMoe {
                router: bf16_lin(&mut r, cfg.num_experts, hidden, 0.3),
                experts_gate: q3w::stack_nvfp4_host(&gates),
                experts_up: q3w::stack_nvfp4_host(&ups),
                experts_down: q3w::stack_nvfp4_host(&downs),
                shared_gate: nvfp4(&mut r, sinter, hidden, 0.15),
                shared_up: nvfp4(&mut r, sinter, hidden, 0.15),
                shared_down: nvfp4(&mut r, hidden, sinter, 0.15),
                shared_expert_gate: bf16_lin(&mut r, 1, hidden, 0.3),
            },
        });
    }

    q3w::HostWeights {
        embed: r.bf16_vec(cfg.vocab_size * hidden, 0.6),
        final_norm: norm_vec(&mut r, hidden),
        lm_head: r.bf16_vec(cfg.vocab_size * hidden, 0.2),
        layers,
    }
}

fn bit_diff(want: &[f32], got: &[f32]) -> usize {
    want.iter()
        .zip(got)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}

const WHY_ROW_LOGITS_AND_NOT_ARGMAX: &str =
    "a tiny model's argmax barely depends on context, so an argmax-only oracle passes while \
     the DeltaNet state and every KV row are wrong; the verify epilogue reuses the decode \
     head's own passes on one row at a time, so the contract is bit-identity of the logits";

#[test]
fn dense_verify_chain_row_logits_are_bit_identical_to_the_m1_decode_steps_it_replaces() {
    let _env = env_lock();
    require_gpu();
    std::env::set_var(
        DENSE_M_ENV,
        VERIFY_ROWS_8_KEEPS_THE_TINY_MAX_SEQ_64_ABLE_TO_HOLD_A_PROMPT_PLUS_TWO_ROUNDS.to_string(),
    );
    let cfg = dense_tiny_config();
    let hw = dense_tiny_weights(&cfg, 0x9e37_79b9_7f4a);
    let prompt = ids_from(cfg.vocab_size, 9, 3);

    for chain_len in [1usize, 3, 8] {
        let chain = ids_from(cfg.vocab_size, chain_len, 11 + chain_len as u32);

        let mut m1 = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("build m1");
        for &t in &prompt {
            m1.prefill_step(t).expect("m1 prefill step");
        }
        let want: Vec<(u32, Vec<f32>)> = chain
            .iter()
            .map(|&t| m1.decode_step_logits(t).expect("m1 decode step"))
            .collect();

        let mut vf = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("build verify");
        assert_eq!(
            vf.verify_max_rows(),
            VERIFY_ROWS_8_KEEPS_THE_TINY_MAX_SEQ_64_ABLE_TO_HOLD_A_PROMPT_PLUS_TWO_ROUNDS,
            "{DENSE_M_ENV} did not reach the M-row prefill graph, so verify_chain would compare \
             nothing"
        );
        for &t in &prompt {
            vf.prefill_step(t).expect("verify prefill step");
        }
        let (toks, logits) = vf.verify_chain_logits(&chain).expect("verify_chain");
        assert_eq!(toks.len(), chain_len, "one argmax per live row");
        assert_eq!(logits.len(), chain_len * cfg.vocab_size, "one logit row per live row");
        assert!(
            chain_len == 1 || want.windows(2).any(|w| bit_diff(&w[0].1, &w[1].1) > 0),
            "the M=1 reference rows are bit-identical to each other at chain_len={chain_len}, so \
             a verify epilogue that ignored the row index would pass this comparison"
        );
        for (r, (want_tok, want_row)) in want.iter().enumerate() {
            let got_row = &logits[r * cfg.vocab_size..(r + 1) * cfg.vocab_size];
            assert_eq!(
                bit_diff(want_row, got_row),
                0,
                "chain_len={chain_len} row {r}: verify_chain logits differ from the M=1 decode \
                 step at the same position. {WHY_ROW_LOGITS_AND_NOT_ARGMAX}"
            );
            assert_eq!(toks[r], *want_tok, "chain_len={chain_len} row {r} argmax");
        }
        vf.advance(chain_len).expect("advance the whole chain");
        assert_eq!(
            vf.current_pos(),
            m1.current_pos(),
            "a fully accepted chain must leave pos where {chain_len} decode steps left it"
        );
    }
    std::env::remove_var(DENSE_M_ENV);
}

#[test]
fn dense_full_and_partial_accepts_leave_a_stream_bit_identical_to_pure_m1_stepping() {
    let _env = env_lock();
    require_gpu();
    std::env::set_var(
        DENSE_M_ENV,
        VERIFY_ROWS_8_KEEPS_THE_TINY_MAX_SEQ_64_ABLE_TO_HOLD_A_PROMPT_PLUS_TWO_ROUNDS.to_string(),
    );
    let cfg = dense_tiny_config();
    let hw = dense_tiny_weights(&cfg, 0x51ee_d5ee_d001);
    let prompt = ids_from(cfg.vocab_size, 9, 5);
    let chain = ids_from(cfg.vocab_size, 6, 21);
    let tail = ids_from(cfg.vocab_size, 5, 31);
    let mut first_tail_row_per_accept: Vec<Vec<f32>> = Vec::new();

    for accepted in [0usize, 1, 3, 6] {
        let mut m1 = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("build m1");
        for &t in &prompt {
            m1.prefill_step(t).expect("m1 prefill step");
        }
        for &t in &chain[..accepted] {
            m1.decode_step(t).expect("m1 accepted step");
        }
        let accepted_pos = m1.current_pos();
        let want: Vec<(u32, Vec<f32>)> = tail
            .iter()
            .map(|&t| m1.decode_step_logits(t).expect("m1 tail step"))
            .collect();

        let mut vf = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("build verify");
        for &t in &prompt {
            vf.prefill_step(t).expect("verify prefill step");
        }
        vf.verify_chain(&chain).expect("verify_chain");
        vf.advance(accepted).expect("advance the accepted prefix");
        assert_eq!(
            vf.current_pos(),
            accepted_pos,
            "accepted={accepted}: pos after advance must equal the M=1 position"
        );
        for (i, (want_tok, want_row)) in want.iter().enumerate() {
            let (got_tok, got_row) = vf.decode_step_logits(tail[i]).expect("verify tail step");
            assert_eq!(
                bit_diff(want_row, &got_row),
                0,
                "accepted={accepted} tail step {i}: the continued stream drifted from pure M=1 \
                 stepping, so the DeltaNet recurrent state or the conv state was not rolled \
                 back to the accepted prefix"
            );
            assert_eq!(got_tok, *want_tok, "accepted={accepted} tail step {i} argmax");
        }
        first_tail_row_per_accept.push(want[0].1.clone());
    }
    assert!(
        first_tail_row_per_accept
            .windows(2)
            .any(|w| bit_diff(&w[0], &w[1]) > 0),
        "every accept length produced the same first tail logits, so this suite would pass with \
         no rollback at all: the recurrent state visibly does not depend on how many rows were \
         committed and the tiny weights cannot certify the rollback"
    );
    std::env::remove_var(DENSE_M_ENV);
}

#[test]
fn dense_verify_chain_refuses_a_second_round_before_advance_commits_or_rolls_back() {
    let _env = env_lock();
    require_gpu();
    std::env::set_var(
        DENSE_M_ENV,
        VERIFY_ROWS_8_KEEPS_THE_TINY_MAX_SEQ_64_ABLE_TO_HOLD_A_PROMPT_PLUS_TWO_ROUNDS.to_string(),
    );
    let cfg = dense_tiny_config();
    let hw = dense_tiny_weights(&cfg, 0x1234_5678);
    let mut vf = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("build verify");
    for &t in &ids_from(cfg.vocab_size, 4, 1) {
        vf.prefill_step(t).expect("prefill step");
    }
    let chain = ids_from(cfg.vocab_size, 3, 2);
    vf.verify_chain(&chain).expect("first verify_chain");
    let err = vf
        .verify_chain(&chain)
        .expect_err("a second verify_chain without advance must be refused");
    assert!(
        format!("{err}").contains("advance"),
        "the refusal must name advance() as the commit-or-rollback point, got: {err}"
    );
    let err = vf
        .advance(chain.len() + 1)
        .expect_err("advancing past the verified rows must be refused");
    assert!(
        format!("{err}").contains("beyond"),
        "the over-advance refusal must say so, got: {err}"
    );
    vf.advance(chain.len()).expect("advance after the refusals");
    let err = vf
        .advance(1)
        .expect_err("advance without a pending verify_chain must be refused");
    assert!(
        format!("{err}").contains("without a pending"),
        "got: {err}"
    );
    std::env::remove_var(DENSE_M_ENV);
}

#[test]
fn moe_verify_chain_row_logits_are_bit_identical_to_the_m1_decode_steps_it_replaces() {
    let _env = env_lock();
    require_gpu();
    std::env::set_var(
        MOE_M_ENV,
        VERIFY_ROWS_8_KEEPS_THE_TINY_MAX_SEQ_64_ABLE_TO_HOLD_A_PROMPT_PLUS_TWO_ROUNDS.to_string(),
    );
    let cfg = moe_tiny_config();
    let hw = moe_tiny_weights(&cfg, 0x9e37_79b9_7f4a);
    let prompt = ids_from(cfg.vocab_size, 9, 3);

    for chain_len in [1usize, 3, 8] {
        let chain = ids_from(cfg.vocab_size, chain_len, 11 + chain_len as u32);

        let mut m1 = Qwen3MoeWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("build m1");
        for &t in &prompt {
            m1.prefill_step(t).expect("m1 prefill step");
        }
        let want: Vec<(u32, Vec<f32>)> = chain
            .iter()
            .map(|&t| m1.decode_step_logits(t).expect("m1 decode step"))
            .collect();

        let mut vf = Qwen3MoeWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("build verify");
        assert_eq!(
            vf.verify_max_rows(),
            VERIFY_ROWS_8_KEEPS_THE_TINY_MAX_SEQ_64_ABLE_TO_HOLD_A_PROMPT_PLUS_TWO_ROUNDS,
            "{MOE_M_ENV} did not reach the M-row prefill list, so verify_chain would compare \
             nothing"
        );
        for &t in &prompt {
            vf.prefill_step(t).expect("verify prefill step");
        }
        let (toks, logits) = vf.verify_chain_logits(&chain).expect("verify_chain");
        assert_eq!(toks.len(), chain_len, "one argmax per live row");
        assert_eq!(logits.len(), chain_len * cfg.vocab_size, "one logit row per live row");
        assert!(
            chain_len == 1 || want.windows(2).any(|w| bit_diff(&w[0].1, &w[1].1) > 0),
            "the M=1 reference rows are bit-identical to each other at chain_len={chain_len}, so \
             a verify epilogue that ignored the row index would pass this comparison"
        );
        for (r, (want_tok, want_row)) in want.iter().enumerate() {
            let got_row = &logits[r * cfg.vocab_size..(r + 1) * cfg.vocab_size];
            assert_eq!(
                bit_diff(want_row, got_row),
                0,
                "chain_len={chain_len} row {r}: verify_chain logits differ from the M=1 decode \
                 step at the same position. {WHY_ROW_LOGITS_AND_NOT_ARGMAX}"
            );
            assert_eq!(toks[r], *want_tok, "chain_len={chain_len} row {r} argmax");
        }
        vf.advance(chain_len).expect("advance the whole chain");
        assert_eq!(
            vf.current_pos(),
            m1.current_pos(),
            "a fully accepted chain must leave pos where {chain_len} decode steps left it"
        );
    }
    std::env::remove_var(MOE_M_ENV);
}

#[test]
fn moe_full_and_partial_accepts_leave_a_stream_bit_identical_to_pure_m1_stepping() {
    let _env = env_lock();
    require_gpu();
    std::env::set_var(
        MOE_M_ENV,
        VERIFY_ROWS_8_KEEPS_THE_TINY_MAX_SEQ_64_ABLE_TO_HOLD_A_PROMPT_PLUS_TWO_ROUNDS.to_string(),
    );
    let cfg = moe_tiny_config();
    let hw = moe_tiny_weights(&cfg, 0x51ee_d5ee_d001);
    let prompt = ids_from(cfg.vocab_size, 9, 5);
    let full_width = VERIFY_ROWS_8_KEEPS_THE_TINY_MAX_SEQ_64_ABLE_TO_HOLD_A_PROMPT_PLUS_TWO_ROUNDS;
    let chain = ids_from(cfg.vocab_size, full_width, 21);
    let tail = ids_from(cfg.vocab_size, 5, 31);
    let mut first_tail_row_per_accept: Vec<Vec<f32>> = Vec::new();

    for accepted in [0usize, 1, 3, full_width] {
        let mut m1 = Qwen3MoeWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("build m1");
        for &t in &prompt {
            m1.prefill_step(t).expect("m1 prefill step");
        }
        for &t in &chain[..accepted] {
            m1.decode_step(t).expect("m1 accepted step");
        }
        let accepted_pos = m1.current_pos();
        let want: Vec<(u32, Vec<f32>)> = tail
            .iter()
            .map(|&t| m1.decode_step_logits(t).expect("m1 tail step"))
            .collect();

        let mut vf = Qwen3MoeWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("build verify");
        for &t in &prompt {
            vf.prefill_step(t).expect("verify prefill step");
        }
        vf.verify_chain(&chain).expect("verify_chain");
        vf.advance(accepted).expect("advance the accepted prefix");
        assert_eq!(
            vf.current_pos(),
            accepted_pos,
            "accepted={accepted}: pos after advance must equal the M=1 position"
        );
        for (i, (want_tok, want_row)) in want.iter().enumerate() {
            let (got_tok, got_row) = vf.decode_step_logits(tail[i]).expect("verify tail step");
            assert_eq!(
                bit_diff(want_row, &got_row),
                0,
                "accepted={accepted} tail step {i}: the continued stream drifted from pure M=1 \
                 stepping. The M-row list forwards all {full_width} rows, so every accept \
                 shorter than the full width must restore the pre-verify recurrent snapshot \
                 and replay the accepted prefix through M=1 stepping"
            );
            assert_eq!(got_tok, *want_tok, "accepted={accepted} tail step {i} argmax");
        }
        first_tail_row_per_accept.push(want[0].1.clone());
    }
    assert!(
        first_tail_row_per_accept
            .windows(2)
            .any(|w| bit_diff(&w[0], &w[1]) > 0),
        "every accept length produced the same first tail logits, so this suite would pass with \
         no rollback at all: the recurrent state visibly does not depend on how many rows were \
         committed and the tiny weights cannot certify the rollback"
    );
    std::env::remove_var(MOE_M_ENV);
}

#[test]
fn moe_verify_chain_needs_a_full_m_row_chunk_of_free_kv_rows() {
    let _env = env_lock();
    require_gpu();
    std::env::set_var(
        MOE_M_ENV,
        VERIFY_ROWS_8_KEEPS_THE_TINY_MAX_SEQ_64_ABLE_TO_HOLD_A_PROMPT_PLUS_TWO_ROUNDS.to_string(),
    );
    let cfg = moe_tiny_config();
    let hw = moe_tiny_weights(&cfg, 0xabc_def);
    let rows = VERIFY_ROWS_8_KEEPS_THE_TINY_MAX_SEQ_64_ABLE_TO_HOLD_A_PROMPT_PLUS_TWO_ROUNDS;
    let max_seq = 24;
    let mut vf = Qwen3MoeWgpu::new(cfg.clone(), &hw, max_seq).expect("build verify");
    assert_eq!(vf.verify_max_rows(), rows);
    for &t in &ids_from(cfg.vocab_size, max_seq - rows + 1, 7) {
        vf.prefill_step(t).expect("prefill step");
    }
    let err = vf
        .verify_chain(&ids_from(cfg.vocab_size, 2, 9))
        .expect_err("a chain that cannot fit the whole baked chunk must be refused");
    assert!(
        format!("{err}").contains("free kv rows"),
        "the refusal must name the whole-chunk kv requirement so the serving clamp can mirror \
         it, got: {err}"
    );
    std::env::remove_var(MOE_M_ENV);
}

fn real_gate(env: &str) -> bool {
    if std::env::var(env).as_deref() != Ok("1") {
        panic!("set {env}=1 to run this real-weights gate; it must never silently skip");
    }
    true
}

#[test]
#[ignore = "loads unsloth/Qwen3.8-27B-NVFP4 on wgpu; set NV_QWEN38_VERIFY_TEST=1 \
            (NV_QWEN38_DIR, NV_QWEN38_VERIFY_LAYERS, NV_QWEN38_VERIFY_M optional) -- \
            asserts verify_chain row logits are bit-identical to the decode steps they replace"]
fn real_weights_qwen38_dense_verify_chain_matches_decode_steps() {
    let _env = env_lock();
    real_gate("NV_QWEN38_VERIFY_TEST");
    require_gpu();
    let rows: usize = std::env::var("NV_QWEN38_VERIFY_M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    std::env::set_var(DENSE_M_ENV, rows.to_string());
    let dir = match std::env::var("NV_QWEN38_DIR") {
        Ok(d) if !d.is_empty() => std::path::PathBuf::from(d),
        _ => hub_snapshot::snapshot_of(
            "unsloth/Qwen3.8-27B-NVFP4",
            &["config.json", "tokenizer.json", "*.safetensors"],
        )
        .expect("no complete unsloth/Qwen3.8-27B-NVFP4 snapshot; set NV_QWEN38_DIR"),
    };
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let mut cfg =
        Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("the 27B config parses as dense");
    if let Ok(v) = std::env::var("NV_QWEN38_VERIFY_LAYERS") {
        let n: usize = v.parse().expect("NV_QWEN38_VERIFY_LAYERS is a usize");
        assert!(n >= 1 && n <= cfg.num_hidden_layers, "layer count out of range");
        cfg.num_hidden_layers = n;
        cfg.layer_types.truncate(n);
    }
    let max_seq = 64;
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let prompt: Vec<u32> = (0..9u32).map(|i| 1000 + i * 37).collect();
    let chain: Vec<u32> = (0..rows as u32).map(|i| 2000 + i * 53).collect();

    let mut m1 =
        Qwen3_5DenseWgpu::from_loader(cfg.clone(), &loader, max_seq).expect("build m1 from loader");
    for &t in &prompt {
        m1.prefill_step(t).expect("m1 prefill step");
    }
    let want: Vec<(u32, Vec<f32>)> = chain
        .iter()
        .map(|&t| m1.decode_step_logits(t).expect("m1 decode step"))
        .collect();
    drop(m1);

    let mut vf = Qwen3_5DenseWgpu::from_loader(cfg.clone(), &loader, max_seq)
        .expect("build verify from loader");
    drop(loader);
    assert_eq!(vf.verify_max_rows(), rows, "the M-row prefill graph must be live");
    for &t in &prompt {
        vf.prefill_step(t).expect("verify prefill step");
    }
    let (toks, logits) = vf.verify_chain_logits(&chain).expect("verify_chain");
    let vocab = cfg.vocab_size;
    for (r, (want_tok, want_row)) in want.iter().enumerate() {
        let got_row = &logits[r * vocab..(r + 1) * vocab];
        let diff = bit_diff(want_row, got_row);
        eprintln!("[q38-verify] row {r}: {diff} of {vocab} logits differ, tok {}", toks[r]);
        assert_eq!(
            diff, 0,
            "row {r}: real-weights verify_chain logits differ from the decode step they replace"
        );
        assert_eq!(toks[r], *want_tok, "row {r} argmax");
    }
    vf.advance(rows).expect("advance the whole chain");
    std::env::remove_var(DENSE_M_ENV);
}

#[test]
#[ignore = "loads RedHatAI/Qwen3.6-35B-A3B-NVFP4 on wgpu; set NV_QWEN36_VERIFY_TEST=1 \
            (NV_QWEN36_DIR, NV_QWEN36_VERIFY_M optional) -- asserts verify_chain row logits \
            are bit-identical to the decode steps they replace"]
fn real_weights_qwen36_moe_verify_chain_matches_decode_steps() {
    let _env = env_lock();
    real_gate("NV_QWEN36_VERIFY_TEST");
    require_gpu();
    let rows: usize = std::env::var("NV_QWEN36_VERIFY_M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    std::env::set_var(MOE_M_ENV, rows.to_string());
    let dir = match std::env::var("NV_QWEN36_DIR") {
        Ok(d) if !d.is_empty() => std::path::PathBuf::from(d),
        _ => hub_snapshot::snapshot_of(
            "RedHatAI/Qwen3.6-35B-A3B-NVFP4",
            &["config.json", "*.safetensors"],
        )
        .expect("no complete RedHatAI/Qwen3.6-35B-A3B-NVFP4 snapshot; set NV_QWEN36_DIR"),
    };
    let cfg = Qwen3MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
    let max_seq = 64;
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let prompt: Vec<u32> = (0..9u32).map(|i| 1000 + i * 37).collect();
    let chain: Vec<u32> = (0..rows as u32).map(|i| 2000 + i * 53).collect();

    let mut m1 =
        Qwen3MoeWgpu::from_loader(cfg.clone(), &loader, max_seq).expect("build m1 from loader");
    for &t in &prompt {
        m1.prefill_step(t).expect("m1 prefill step");
    }
    let want: Vec<(u32, Vec<f32>)> = chain
        .iter()
        .map(|&t| m1.decode_step_logits(t).expect("m1 decode step"))
        .collect();
    drop(m1);

    let mut vf =
        Qwen3MoeWgpu::from_loader(cfg.clone(), &loader, max_seq).expect("build verify from loader");
    drop(loader);
    assert_eq!(vf.verify_max_rows(), rows, "the M-row prefill list must be live");
    for &t in &prompt {
        vf.prefill_step(t).expect("verify prefill step");
    }
    let (toks, logits) = vf.verify_chain_logits(&chain).expect("verify_chain");
    let vocab = cfg.vocab_size;
    for (r, (want_tok, want_row)) in want.iter().enumerate() {
        let got_row = &logits[r * vocab..(r + 1) * vocab];
        let diff = bit_diff(want_row, got_row);
        eprintln!("[q36-verify] row {r}: {diff} of {vocab} logits differ, tok {}", toks[r]);
        assert_eq!(
            diff, 0,
            "row {r}: real-weights verify_chain logits differ from the decode step they replace"
        );
        assert_eq!(toks[r], *want_tok, "row {r} argmax");
    }
    vf.advance(rows).expect("advance the whole chain");
    std::env::remove_var(MOE_M_ENV);
}
