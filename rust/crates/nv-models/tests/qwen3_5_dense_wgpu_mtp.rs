#![cfg(feature = "wgpu")]

mod common;
use common::bf16_lin;
use common::LcgOddSeedShift33SignedUnit as Lcg;
use common::norm_vec;
use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_dense_wgpu::{MtpHostWeights, Qwen3_5DenseConfig, Qwen3_5DenseWgpu};
use nv_models::qwen3_5_moe::LayerType;
use nv_models::qwen3_5_moe_wgpu::HostBf16Lin;

static ENV_LOCK_THESE_TESTS_MUTATE_PROCESS_GLOBAL_ENV_SO_MUST_NOT_INTERLEAVE:
    std::sync::Mutex<()> = std::sync::Mutex::new(());

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK_THESE_TESTS_MUTATE_PROCESS_GLOBAL_ENV_SO_MUST_NOT_INTERLEAVE
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

const DENSE_M_ENV: &str = "NV_WGPU_PREFILL_M";

const VERIFY_ROWS_8_KEEPS_THE_TINY_MAX_SEQ_64_ABLE_TO_HOLD_A_PROMPT_PLUS_ROUNDS: usize = 8;

const TINY_MAX_SEQ: usize = 64;

fn require_gpu() {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => eprintln!("[wgpu] adapter: {}", ctx.info.name),
        Err(e) => panic!(
            "no wgpu adapter ({e}): a skipped mtp identity proof reads as a passed one, so this \
             suite fails instead of skipping"
        ),
    }
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
            delta_fp8: q3d::DeltaFp8::default(),
            mlp: q3d::HostDenseMlp {
                gate: bf16_lin(&mut r, inter, hidden, 0.15).into(),
                up: bf16_lin(&mut r, inter, hidden, 0.15).into(),
                down: bf16_lin(&mut r, hidden, inter, 0.15).into(),
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

fn mtp_tiny_weights(cfg: &Qwen3_5DenseConfig, seed: u64) -> MtpHostWeights {
    let mut r = Lcg::new(seed);
    let hidden = cfg.hidden_size;
    let hd = cfg.head_dim;
    let inter = cfg.intermediate_size;
    let q_out = cfg.num_attention_heads * hd * 2;
    let kv_out = cfg.num_key_value_heads * hd;
    MtpHostWeights {
        pre_fc_norm_embedding: norm_vec(&mut r, hidden),
        pre_fc_norm_hidden: norm_vec(&mut r, hidden),
        fc: bf16_lin(&mut r, hidden, 2 * hidden, 0.1),
        input_ln: norm_vec(&mut r, hidden),
        attn: q3d::HostDenseAttention {
            q: bf16_lin(&mut r, q_out, hidden, 0.12).into(),
            k: bf16_lin(&mut r, kv_out, hidden, 0.12).into(),
            v: bf16_lin(&mut r, kv_out, hidden, 0.12).into(),
            o: bf16_lin(&mut r, hidden, cfg.num_attention_heads * hd, 0.12).into(),
            q_norm: norm_vec(&mut r, hd),
            k_norm: norm_vec(&mut r, hd),
            qkv_bias_bf16: vec![],
        },
        post_attn_ln: norm_vec(&mut r, hidden),
        mlp: q3d::HostDenseMlp {
            gate: bf16_lin(&mut r, inter, hidden, 0.15).into(),
            up: bf16_lin(&mut r, inter, hidden, 0.15).into(),
            down: bf16_lin(&mut r, hidden, inter, 0.15).into(),
        },
        final_norm: norm_vec(&mut r, hidden),
    }
}

fn prefill_like_serving(m: &mut Qwen3_5DenseWgpu, prompt: &[u32]) -> u32 {
    let (last, rest) = prompt.split_last().expect("non-empty prompt");
    let done = m.prefill_tokens(rest).expect("chunked prefill");
    for &t in &rest[done..] {
        m.prefill_step(t).expect("prefill step");
    }
    m.decode_step(*last).expect("anchor decode step")
}

fn reference_stream(
    cfg: &Qwen3_5DenseConfig,
    hw: &q3d::HostDenseWeights,
    prompt: &[u32],
    n: usize,
) -> Vec<u32> {
    let mut m = Qwen3_5DenseWgpu::new(cfg.clone(), hw, TINY_MAX_SEQ).expect("build reference");
    let mut last = prefill_like_serving(&mut m, prompt);
    let mut out = vec![last];
    while out.len() < n {
        last = m.decode_step(last).expect("reference decode step");
        out.push(last);
    }
    out
}

enum DraftPolicy {
    MtpOwn,
    Junk,
    Clairvoyant,
}

const DRAFT_POLICY_INVARIANCE_IS_THE_BAR: &str =
    "every emitted token is gated by the trunk verify argmax and every commit path (full accept, \
     partial accept with snapshot+replay, drafter KV rewind+catch-up) must land in the state pure \
     M=1 stepping produces, so the served stream may not depend on WHAT was drafted -- junk \
     drafts, the mtp head's own drafts, and clairvoyant reference drafts must all emit the \
     byte-identical stream";

fn mtp_stream(
    cfg: &Qwen3_5DenseConfig,
    hw: &q3d::HostDenseWeights,
    mtp: &MtpHostWeights,
    prompt: &[u32],
    n: usize,
    k: usize,
    policy: DraftPolicy,
    reference: &[u32],
) -> (Vec<u32>, usize) {
    let mut m = Qwen3_5DenseWgpu::new(cfg.clone(), hw, TINY_MAX_SEQ).expect("build mtp engine");
    m.mtp_attach_host(mtp).expect("attach mtp head");
    let mut last = prefill_like_serving(&mut m, prompt);
    assert_eq!(
        m.mtp_len(),
        m.current_pos(),
        "prompt auto-sync must leave the drafter KV at the trunk committed length"
    );
    let mut out = vec![last];
    let mut accepted_total = 0usize;
    while out.len() < n {
        let rows = m.verify_max_rows();
        assert!(rows >= 2, "{DENSE_M_ENV} did not reach the M-row prefill graph");
        if m.current_pos() + rows.max(m.prefill_chunk_len()) > TINY_MAX_SEQ {
            last = m.decode_step(last).expect("tail decode step");
            out.push(last);
            continue;
        }
        let want = k.min(rows - 1);
        let own = m.mtp_draft_round(last, want).expect("mtp draft round");
        let drafts: Vec<u32> = match policy {
            DraftPolicy::MtpOwn => own,
            DraftPolicy::Junk => (0..want as u32)
                .map(|j| (j * 5 + 2) % cfg.vocab_size as u32)
                .collect(),
            DraftPolicy::Clairvoyant => {
                let mut d = Vec::with_capacity(want);
                for j in 0..want {
                    d.push(*reference.get(out.len() + j).unwrap_or(&own[j.min(own.len() - 1)]));
                }
                d
            }
        };
        let mut batch = vec![last];
        batch.extend_from_slice(&drafts);
        let amax = m.verify_chain(&batch).expect("verify chain");
        let mut acc = 0usize;
        while acc < drafts.len() && amax[acc] == drafts[acc] {
            acc += 1;
        }
        let bonus = amax[acc];
        m.advance(acc + 1).expect("advance the accepted prefix");
        m.mtp_post_verify(&batch[1..=acc]).expect("mtp post verify");
        assert_eq!(
            m.mtp_len(),
            m.current_pos(),
            "round catch-up must leave the drafter KV at the trunk committed length"
        );
        accepted_total += acc;
        out.extend_from_slice(&drafts[..acc]);
        out.push(bonus);
        last = bonus;
    }
    out.truncate(n);
    (out, accepted_total)
}

#[test]
fn mtp_rounds_emit_the_stream_pure_m1_stepping_emits_under_every_draft_policy() {
    let _env = env_lock();
    require_gpu();
    std::env::set_var(
        DENSE_M_ENV,
        VERIFY_ROWS_8_KEEPS_THE_TINY_MAX_SEQ_64_ABLE_TO_HOLD_A_PROMPT_PLUS_ROUNDS.to_string(),
    );
    let cfg = dense_tiny_config();
    let hw = dense_tiny_weights(&cfg, 0x9e37_79b9_7f4a);
    let mtp = mtp_tiny_weights(&cfg, 0xabcd_ef01_2345);
    let prompt = ids_from(cfg.vocab_size, 19, 3);
    let n = 24;
    let reference = reference_stream(&cfg, &hw, &prompt, n);

    let (own, _) = mtp_stream(&cfg, &hw, &mtp, &prompt, n, 3, DraftPolicy::MtpOwn, &reference);
    assert_eq!(own, reference, "mtp-drafted stream diverged. {DRAFT_POLICY_INVARIANCE_IS_THE_BAR}");

    let (junk, _) = mtp_stream(&cfg, &hw, &mtp, &prompt, n, 3, DraftPolicy::Junk, &reference);
    assert_eq!(junk, reference, "junk-drafted stream diverged. {DRAFT_POLICY_INVARIANCE_IS_THE_BAR}");

    let (clair, clair_accepted) = mtp_stream(
        &cfg,
        &hw,
        &mtp,
        &prompt,
        n,
        3,
        DraftPolicy::Clairvoyant,
        &reference,
    );
    assert_eq!(
        clair, reference,
        "clairvoyant-drafted stream diverged. {DRAFT_POLICY_INVARIANCE_IS_THE_BAR}"
    );
    assert!(
        clair_accepted > 0,
        "clairvoyant drafts from the reference stream never accepted, so the full-accept commit \
         and drafter catch-up paths were never exercised and this identity is vacuous"
    );
    std::env::remove_var(DENSE_M_ENV);
}

const VF_TINY_ARM_PINS_ONE_KERNEL_PER_STAGE_SO_THE_M16_AND_M8_GRAPHS_RECORD_IDENTICAL_ROUTES:
    [(&str, &str); 3] = [
    (q3d::PF_COOP_ENV_DEFAULT_ON_SET_0_FOR_THE_LEGACY_M16_GEMM_ARM, "0"),
    (q3d::PF_ATTN_TILED_ENV_DEFAULT_ON_SET_0_FOR_THE_SCORES_SLAB_ARM, "1"),
    (q3d::PF_SCAN_WY_ENV_DEFAULT_ON_SET_0_FOR_THE_TOKEN_SERIAL_SCAN, "1"),
];

fn vf_tiny_env_set() {
    std::env::set_var(DENSE_M_ENV, "16");
    std::env::set_var(
        q3d::VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM,
        "8",
    );
    for (k, v) in VF_TINY_ARM_PINS_ONE_KERNEL_PER_STAGE_SO_THE_M16_AND_M8_GRAPHS_RECORD_IDENTICAL_ROUTES {
        std::env::set_var(k, v);
    }
}

fn vf_tiny_env_clear() {
    std::env::remove_var(DENSE_M_ENV);
    std::env::remove_var(
        q3d::VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM,
    );
    for (k, _) in VF_TINY_ARM_PINS_ONE_KERNEL_PER_STAGE_SO_THE_M16_AND_M8_GRAPHS_RECORD_IDENTICAL_ROUTES {
        std::env::remove_var(k);
    }
}

#[test]
fn unset_verify_m_default_stays_on_the_padded_arm_beside_a_legacy_prefill_graph() {
    let _env = env_lock();
    require_gpu();
    std::env::remove_var(q3d::VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM);
    std::env::set_var(
        DENSE_M_ENV,
        VERIFY_ROWS_8_KEEPS_THE_TINY_MAX_SEQ_64_ABLE_TO_HOLD_A_PROMPT_PLUS_ROUNDS.to_string(),
    );
    let cfg = dense_tiny_config();
    let hw = dense_tiny_weights(&cfg, 0x7031_c0de_0002);
    let m = Qwen3_5DenseWgpu::new(cfg, &hw, TINY_MAX_SEQ).expect("build");
    assert!(
        !m.verify_chain_rides_dedicated_graph(),
        "{}",
        q3d::VERIFY_M_DEFAULT_RECORDS_ONLY_BESIDE_A_COOP_PREFILL_GRAPH
    );
    assert_eq!(
        m.verify_max_rows(),
        VERIFY_ROWS_8_KEEPS_THE_TINY_MAX_SEQ_64_ABLE_TO_HOLD_A_PROMPT_PLUS_ROUNDS,
        "the env-unset default must leave the legacy graph's verify capacity at the prefill m; \
         a shrunk capacity here breaks every caller of the historic rows == m contract"
    );
    std::env::remove_var(DENSE_M_ENV);
}

#[test]
fn dedicated_verify_graph_rows_bit_match_the_padded_prefill_graph_on_pinned_arms() {
    let _env = env_lock();
    require_gpu();
    vf_tiny_env_set();
    let cfg = dense_tiny_config();
    let hw = dense_tiny_weights(&cfg, 0x7031_c0de_0001);
    let mut m = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("build");
    assert!(
        m.verify_chain_rides_dedicated_graph(),
        "NV_Q3D_VERIFY_M=8 under NV_WGPU_PREFILL_M=16 must record the dedicated verify graph; \
         without it this A/B compares the padded graph to itself and proves nothing"
    );
    assert_eq!(m.verify_max_rows(), 8, "chain capacity must be the dedicated graph's rows");
    assert_eq!(m.prefill_chunk_len(), 16, "the prefill graph must keep its own M");
    let prompt = ids_from(cfg.vocab_size, 19, 7);
    let mut last = prefill_like_serving(&mut m, &prompt);
    for round in 0..3u32 {
        let mut batch = vec![last];
        batch.extend(ids_from(cfg.vocab_size, 3, 11 + round));
        let (a_tok, a_log) = m.verify_chain_logits(&batch).expect("dedicated-graph chain");
        m.advance(0).expect("rollback the dedicated-graph forward");
        let (b_tok, b_log) = m
            .debug_verify_chain_logits_via_the_padded_prefill_graph_for_test(&batch)
            .expect("padded-graph chain");
        assert_eq!(
            a_log.len(),
            b_log.len(),
            "both arms must return live-rows x vocab logits"
        );
        let mut max_abs = 0f32;
        let mut diff_bits = 0usize;
        for (x, y) in a_log.iter().zip(&b_log) {
            if x.to_bits() != y.to_bits() {
                diff_bits += 1;
                max_abs = max_abs.max((x - y).abs());
            }
        }
        assert_eq!(
            a_tok, b_tok,
            "round {round}: verify argmax rows diverged between the 8-row dedicated graph and \
             the 16-row padded graph on arms pinned kernel-identical; the pad rows are supposed \
             to be dead weight, so a divergence means a chunk-bracket-dependent kernel"
        );
        assert_eq!(
            diff_bits, 0,
            "round {round}: {diff_bits} logit values differ (max abs {max_abs:e}) between the \
             dedicated 8-row verify graph and the padded 16-row graph on arms pinned \
             kernel-identical; the rows are supposed to be bit-independent of the chunk M"
        );
        m.advance(batch.len()).expect("commit every verified row");
        last = *a_tok.last().expect("argmax per row");
    }
    vf_tiny_env_clear();
}

#[test]
fn mtp_rounds_on_the_dedicated_verify_graph_emit_the_pure_m1_stream() {
    let _env = env_lock();
    require_gpu();
    vf_tiny_env_set();
    let cfg = dense_tiny_config();
    let hw = dense_tiny_weights(&cfg, 0x9e37_79b9_7f4a);
    let mtp = mtp_tiny_weights(&cfg, 0xabcd_ef01_2345);
    let prompt = ids_from(cfg.vocab_size, 19, 3);
    let n = 24;
    let reference = reference_stream(&cfg, &hw, &prompt, n);
    let (own, _) = mtp_stream(&cfg, &hw, &mtp, &prompt, n, 3, DraftPolicy::MtpOwn, &reference);
    assert_eq!(
        own, reference,
        "mtp-drafted stream through the dedicated verify graph diverged. \
         {DRAFT_POLICY_INVARIANCE_IS_THE_BAR}"
    );
    let (clair, clair_accepted) = mtp_stream(
        &cfg,
        &hw,
        &mtp,
        &prompt,
        n,
        3,
        DraftPolicy::Clairvoyant,
        &reference,
    );
    assert_eq!(
        clair, reference,
        "clairvoyant-drafted stream through the dedicated verify graph diverged. \
         {DRAFT_POLICY_INVARIANCE_IS_THE_BAR}"
    );
    assert!(
        clair_accepted > 0,
        "clairvoyant drafts never accepted, so the dedicated graph's partial/full accept commit \
         and drafter catch-up paths were never exercised and this identity is vacuous"
    );
    vf_tiny_env_clear();
}

#[test]
fn mtp_round_bookkeeping_refuses_desync_and_double_rounds() {
    let _env = env_lock();
    require_gpu();
    std::env::set_var(
        DENSE_M_ENV,
        VERIFY_ROWS_8_KEEPS_THE_TINY_MAX_SEQ_64_ABLE_TO_HOLD_A_PROMPT_PLUS_ROUNDS.to_string(),
    );
    let cfg = dense_tiny_config();
    let hw = dense_tiny_weights(&cfg, 0x51ee_d5ee_d001);
    let mtp = mtp_tiny_weights(&cfg, 0x1234_5678);
    let mut m = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ).expect("build");
    m.mtp_attach_host(&mtp).expect("attach");
    let err = m
        .mtp_attach_host(&mtp)
        .expect_err("a second attach must be refused");
    assert!(format!("{err}").contains("already attached"), "got: {err}");
    let prompt = ids_from(cfg.vocab_size, 9, 5);
    let last = prefill_like_serving(&mut m, &prompt);
    m.mtp_draft_round(last, 3).expect("first draft round");
    let err = m
        .mtp_draft_round(last, 3)
        .expect_err("a second draft round without post_verify must be refused as a desync");
    assert!(format!("{err}").contains("desync"), "got: {err}");
    let err = Qwen3_5DenseWgpu::new(cfg.clone(), &hw, TINY_MAX_SEQ)
        .expect("fresh engine")
        .mtp_post_verify(&[])
        .expect_err("post_verify without a round must be refused");
    assert!(format!("{err}").contains("mtp_draft_round"), "got: {err}");
    std::env::remove_var(DENSE_M_ENV);
}
