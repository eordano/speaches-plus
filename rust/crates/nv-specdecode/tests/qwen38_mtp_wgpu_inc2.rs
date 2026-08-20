#![cfg(feature = "wgpu")]

use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_dense_wgpu::{MtpHostWeights, Qwen3_5DenseConfig, Qwen3_5DenseWgpu};
use nv_models::qwen3_5_moe::LayerType;
use nv_models::qwen3_5_moe_wgpu::{HostBf16Lin, HostDeltaNet};
use nv_specdecode::qwen38_mtp::run_mtp_verify_round;

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
            "no wgpu adapter ({e}): a skipped inc2 commit proof reads as a passed one, so this \
             suite fails instead of skipping"
        ),
    }
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32 - 1.0
    }

    fn bf16_vec(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_bits())
            .collect()
    }

    fn f32_vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| half::bf16::from_f32(self.next_f32() * scale).to_f32())
            .collect()
    }
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

fn ids_from(vocab: usize, n: usize, salt: u32) -> Vec<u32> {
    (0..n)
        .map(|i| ((i as u32 * 7 + salt * 13 + 1) % (vocab as u32 - 1)) + 1)
        .collect()
}

fn dense_tiny_config() -> Qwen3_5DenseConfig {
    Qwen3_5DenseConfig {
        hidden_size: 128,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 32,
        intermediate_size: 96,
        vocab_size: 64,
        max_position_embeddings: TINY_MAX_SEQ,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        partial_rotary_factor: 0.25,
        bos_token_id: None,
        eos_token_id: 1,
        layer_types: vec![
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
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

fn dense_tiny_weights(cfg: &Qwen3_5DenseConfig, seed: u64) -> q3d::HostDenseWeights {
    let mut r = Lcg::new(seed);
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let hd = cfg.head_dim;
    let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let ks = cfg.linear_conv_kernel_dim;

    let mut layers = Vec::new();
    for li in 0..cfg.num_hidden_layers {
        let mixer = match cfg.layer_types[li] {
            LayerType::LinearAttention => q3d::HostDenseMixer::Delta(Box::new(HostDeltaNet {
                in_proj_qkv: bf16_lin(&mut r, conv_dim, hidden, 0.12),
                in_proj_z: bf16_lin(&mut r, value_dim, hidden, 0.12),
                in_proj_ab: bf16_lin(&mut r, 2 * cfg.linear_num_value_heads, hidden, 0.12),
                conv1d: r.f32_vec(conv_dim * ks, 0.4),
                a_log: r.f32_vec(cfg.linear_num_value_heads, 0.5),
                dt_bias: r.f32_vec(cfg.linear_num_value_heads, 0.5),
                norm_w: norm_vec(&mut r, cfg.linear_value_head_dim),
                out_proj: bf16_lin(&mut r, hidden, value_dim, 0.12),
            })),
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

const INC2_INVARIANCE_BAR: &str =
    "increment 2 commits the batched verify rows (full accept in place, partial accept via \
     advance(0) rollback + batched prefix re-forward) instead of the increment-1 M=1 replay; on \
     the tiny synthetic config the M-row graph is bit-exact against M=1 stepping, so every draft \
     policy must still emit the byte-identical pure-M=1 stream and leave mtp_len == current_pos \
     after every round";

struct Inc2RunStats {
    accepted: usize,
    reforwards: usize,
}

fn mtp_stream_inc2(
    cfg: &Qwen3_5DenseConfig,
    hw: &q3d::HostDenseWeights,
    mtp: &MtpHostWeights,
    prompt: &[u32],
    n: usize,
    k: usize,
    policy: DraftPolicy,
    reference: &[u32],
    replay_commit: bool,
) -> (Vec<u32>, Inc2RunStats) {
    let mut m = Qwen3_5DenseWgpu::new(cfg.clone(), hw, TINY_MAX_SEQ).expect("build inc2 engine");
    m.mtp_attach_host(mtp).expect("attach mtp head");
    let mut last = prefill_like_serving(&mut m, prompt);
    let mut out = vec![last];
    let mut stats = Inc2RunStats {
        accepted: 0,
        reforwards: 0,
    };
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
            DraftPolicy::Clairvoyant => (0..want)
                .map(|j| *reference.get(out.len() + j).unwrap_or(&own[j.min(own.len() - 1)]))
                .collect(),
        };
        let r = run_mtp_verify_round(&mut m, last, &drafts, replay_commit).expect("inc2 round");
        m.mtp_post_verify(&r.batch[1..r.accept.commit_len])
            .expect("mtp post verify");
        assert_eq!(
            m.mtp_len(),
            m.current_pos(),
            "round catch-up desync after a {} commit. {INC2_INVARIANCE_BAR}",
            if r.prefix_reforwarded_batched {
                "batched prefix re-forward"
            } else {
                "in-place"
            }
        );
        stats.accepted += r.accept.draft_accepted;
        stats.reforwards += r.prefix_reforwarded_batched as usize;
        out.extend_from_slice(&r.emitted);
        last = *out.last().expect("emitted is never empty");
    }
    out.truncate(n);
    (out, stats)
}

#[test]
fn inc2_batched_commit_emits_the_pure_m1_stream_under_every_draft_policy() {
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

    let (own, own_stats) = mtp_stream_inc2(
        &cfg,
        &hw,
        &mtp,
        &prompt,
        n,
        3,
        DraftPolicy::MtpOwn,
        &reference,
        false,
    );
    assert_eq!(own, reference, "mtp-drafted inc2 stream diverged. {INC2_INVARIANCE_BAR}");

    let (junk, junk_stats) = mtp_stream_inc2(
        &cfg,
        &hw,
        &mtp,
        &prompt,
        n,
        3,
        DraftPolicy::Junk,
        &reference,
        false,
    );
    assert_eq!(junk, reference, "junk-drafted inc2 stream diverged. {INC2_INVARIANCE_BAR}");
    assert!(
        junk_stats.reforwards > 0,
        "junk drafts never took the partial-accept batched re-forward path, so the inc2 commit \
         mechanism under test never ran and this identity is vacuous"
    );

    let (clair, clair_stats) = mtp_stream_inc2(
        &cfg,
        &hw,
        &mtp,
        &prompt,
        n,
        3,
        DraftPolicy::Clairvoyant,
        &reference,
        false,
    );
    assert_eq!(
        clair, reference,
        "clairvoyant-drafted inc2 stream diverged. {INC2_INVARIANCE_BAR}"
    );
    assert!(
        clair_stats.accepted > 0,
        "clairvoyant drafts never accepted, so the full-accept in-place commit and drafter \
         catch-up were never exercised"
    );
    let _ = own_stats;
    std::env::remove_var(DENSE_M_ENV);
}

#[test]
fn inc2_replay_escape_emits_the_same_stream_through_the_increment1_commit() {
    let _env = env_lock();
    require_gpu();
    std::env::set_var(
        DENSE_M_ENV,
        VERIFY_ROWS_8_KEEPS_THE_TINY_MAX_SEQ_64_ABLE_TO_HOLD_A_PROMPT_PLUS_ROUNDS.to_string(),
    );
    let cfg = dense_tiny_config();
    let hw = dense_tiny_weights(&cfg, 0x51ee_d5ee_d001);
    let mtp = mtp_tiny_weights(&cfg, 0x1234_5678);
    let prompt = ids_from(cfg.vocab_size, 17, 9);
    let n = 20;
    let reference = reference_stream(&cfg, &hw, &prompt, n);
    let (replayed, replay_stats) = mtp_stream_inc2(
        &cfg,
        &hw,
        &mtp,
        &prompt,
        n,
        3,
        DraftPolicy::Junk,
        &reference,
        true,
    );
    assert_eq!(
        replayed, reference,
        "the NV_Q38_MTP_VERIFY_REPLAY escape must reproduce the increment-1 replay-commit stream"
    );
    assert_eq!(
        replay_stats.reforwards, 0,
        "replay mode must never take the batched prefix re-forward"
    );
    std::env::remove_var(DENSE_M_ENV);
}
