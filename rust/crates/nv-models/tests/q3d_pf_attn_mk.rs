#![cfg(feature = "wgpu")]

use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseConfig;
use nv_models::qwen3_5_moe::LayerType;
use nv_models::qwen3_5_moe_wgpu::HostBf16Lin;

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
        (((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32) - 1.0
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

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static L: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    L.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn tiny_config() -> Qwen3_5DenseConfig {
    Qwen3_5DenseConfig {
        hidden_size: 128,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 32,
        intermediate_size: 192,
        vocab_size: 64,
        max_position_embeddings: 64,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        partial_rotary_factor: 0.25,
        bos_token_id: None,
        eos_token_id: 1,
        layer_types: vec![
            LayerType::LinearAttention,
            LayerType::FullAttention,
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

const ARM_ENV: &str = q3d::PF_ATTN_MK_ENV_DEFAULT_OFF_UNTIL_THE_LADDER_AND_PPL_GATE_ARE_ON_RECORD;

const TILED_ENV: &str = q3d::PF_ATTN_TILED_ENV_DEFAULT_ON_SET_0_FOR_THE_SCORES_SLAB_ARM;

const M_ENV_PINNED_TO_16_BECAUSE_THESE_PROMPTS_AND_MAX_SEQ_ASSUME_THE_PRE_COOP_CHUNK: &str =
    "NV_WGPU_PREFILL_M";

struct M16Pin;

impl M16Pin {
    fn set() -> Self {
        std::env::set_var(
            M_ENV_PINNED_TO_16_BECAUSE_THESE_PROMPTS_AND_MAX_SEQ_ASSUME_THE_PRE_COOP_CHUNK,
            "16",
        );
        pin_kv_fp8_off_process_wide_because_the_scores_and_mk_arms_never_quantize_chunk_rows();
        M16Pin
    }
}

fn pin_kv_fp8_off_process_wide_because_the_scores_and_mk_arms_never_quantize_chunk_rows() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| std::env::set_var("NV_Q3D_KV_FP8", "0"));
}

impl Drop for M16Pin {
    fn drop(&mut self) {
        std::env::remove_var(
            M_ENV_PINNED_TO_16_BECAUSE_THESE_PROMPTS_AND_MAX_SEQ_ASSUME_THE_PRE_COOP_CHUNK,
        );
    }
}

const PROMPT_LENGTHS_COVERING_FULL_PARTIAL_AND_DEAD_ROW_GROUP_CHUNKS: [usize; 3] = [33, 23, 9];

const CONTINUATION_TOKENS: usize = 8;

const MK_VS_SCORES_TOLERANCE_IS_REDUCTION_ORDER_NOT_CORRECTNESS: f32 = 0.05;

const TILED_VS_SCORES_TOLERANCE_ADDS_THE_E4M3_KV_ROUND_TRIP_TO_REDUCTION_ORDER: f32 = 0.05;

fn gpu_present_or_explicit_skip() -> bool {
    if nv_kernels::wgpu_backend::WgpuContext::shared().is_ok() {
        return true;
    }
    if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() == Ok("1") {
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): no wgpu adapter");
        return false;
    }
    panic!("no wgpu adapter; this gate must never silently skip");
}

fn tiny_weights(cfg: &Qwen3_5DenseConfig, seed: u64) -> q3d::HostDenseWeights {
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
            LayerType::LinearAttention => {
                q3d::HostDenseMixer::Delta(Box::new(nv_models::qwen3_5_moe_wgpu::HostDeltaNet {
                    in_proj_qkv: bf16_lin(&mut r, conv_dim, hidden, 0.12),
                    in_proj_z: bf16_lin(&mut r, value_dim, hidden, 0.12),
                    in_proj_ab: bf16_lin(&mut r, 2 * cfg.linear_num_value_heads, hidden, 0.12),
                    conv1d: r.f32_vec(conv_dim * ks, 0.4),
                    a_log: r.f32_vec(cfg.linear_num_value_heads, 0.5),
                    dt_bias: r.f32_vec(cfg.linear_num_value_heads, 0.5),
                    norm_w: norm_vec(&mut r, cfg.linear_value_head_dim),
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
            delta_fp8: Default::default(),
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

fn scores_slab_bytes(gpu: &q3d::Qwen3_5DenseWgpu) -> u64 {
    gpu.vram_report()
        .by_class
        .iter()
        .find(|(c, _, _)| c == "pf-at-scores")
        .map(|(_, _, b)| *b)
        .expect("the pf graph always allocates the pf-at-scores class")
}

fn chunked_then_greedy(
    gpu: &mut q3d::Qwen3_5DenseWgpu,
    tokens: &[u32],
) -> (u32, Vec<f32>, Vec<u32>) {
    gpu.reset().expect("reset");
    let (last, rest) = tokens.split_last().expect("prompt non-empty");
    let done = gpu.prefill_tokens(rest).expect("prefill_tokens");
    assert!(
        done > 0 || rest.len() < gpu.prefill_chunk_len(),
        "chunked prefill consumed nothing on a {}-token prompt at m={}",
        rest.len(),
        gpu.prefill_chunk_len()
    );
    for t in &rest[done..] {
        gpu.prefill_step(*t).expect("tail prefill step");
    }
    let (arg, logits) = gpu.decode_step_logits(*last).expect("last prompt token");
    let mut ids = Vec::with_capacity(CONTINUATION_TOKENS);
    let mut next = arg;
    for _ in 0..CONTINUATION_TOKENS {
        ids.push(next);
        next = gpu.decode_step(next).expect("greedy decode step");
    }
    (arg, logits, ids)
}

#[test]
fn the_verify_row_instrument_observes_the_pf_attention_output_so_the_ab_gate_is_not_vacuous() {
    let _g = env_lock();
    if !gpu_present_or_explicit_skip() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0xa77b_3a1c_0003);
    let mut hw2 = tiny_weights(&cfg, 0xa77b_3a1c_0003);
    let mut flipped = 0usize;
    for layer in &mut hw2.layers {
        if let q3d::HostDenseMixer::Attn(a) = &mut layer.mixer {
            if let q3d::HostDenseLin::Bf16(l) = &mut a.o {
                for w in &mut l.w {
                    *w ^= 0x0100;
                    flipped += 1;
                }
            }
        }
    }
    assert!(flipped > 0, "no attention o-proj bf16 weights found to perturb");
    let batch: Vec<u32> = (0..6u32).map(|i| (i * 5 + 1) % 64).collect();
    let _m = M16Pin::set();
    std::env::remove_var(ARM_ENV);
    let run = |hww: &q3d::HostDenseWeights| {
        let mut gpu = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), hww, 96).expect("build");
        gpu.verify_chain_logits(&batch).expect("verify").1
    };
    let a = run(&hw);
    let b = run(&hw2);
    let diff = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    eprintln!(
        "[pf-attn-mk] o-proj perturbation moved {diff}/{} verify row-logit lanes",
        a.len()
    );
    assert!(
        diff > 0,
        "verify_chain row logits never observe the pf attention o-proj: with no decode warmup \
         the only o-proj consumer is the pf graph, so the A/B gate below would be vacuous"
    );
}

#[test]
fn pf_attn_mk_arm_matches_the_scores_arm_within_reduction_order_and_keeps_the_greedy_stream() {
    let _g = env_lock();
    if !gpu_present_or_explicit_skip() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0xa77b_3a1c_0001);

    let _m = M16Pin::set();
    std::env::remove_var(ARM_ENV);
    std::env::set_var(TILED_ENV, "0");
    let mut scores_arm =
        q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 96).expect("build scores arm");
    std::env::set_var(ARM_ENV, "1");
    let mut mk_arm = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 96).expect("build mk arm");
    std::env::remove_var(ARM_ENV);
    std::env::set_var(TILED_ENV, "1");
    let mut tiled_arm = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 96).expect("build tiled arm");
    std::env::remove_var(TILED_ENV);

    assert_eq!(
        scores_arm.prefill_chunk_len(),
        mk_arm.prefill_chunk_len(),
        "the mk arm must not move the chunk length at m<=16"
    );
    let (slab_scores, slab_mk) = (scores_slab_bytes(&scores_arm), scores_slab_bytes(&mk_arm));
    eprintln!(
        "[pf-attn-mk] a_scores slab bytes: scores arm {slab_scores}, mk arm {slab_mk}; prefill \
         passes: scores arm {}, mk arm {}",
        scores_arm.prefill_pass_count(),
        mk_arm.prefill_pass_count()
    );
    assert!(
        mk_arm.prefill_pass_count() > scores_arm.prefill_pass_count(),
        "the mk arm records 1 qcast + 2 passes per row group in place of the single scores \
         dispatch per attention layer; an equal pass count means the route never engaged \
         (scores {}, mk {})",
        scores_arm.prefill_pass_count(),
        mk_arm.prefill_pass_count()
    );
    assert!(
        slab_mk <= 4 && slab_scores > slab_mk,
        "NV_Q3D_PF_ATTN_MK=1 must gate the dead m x n_heads x max_seq f32 scores slab down to a \
         placeholder allocation (scores arm {slab_scores} B, mk arm {slab_mk} B)"
    );
    assert_eq!(
        scores_arm.prefill_chunk_len(),
        tiled_arm.prefill_chunk_len(),
        "the tiled arm must not move the chunk length at m<=16"
    );
    let slab_tiled = scores_slab_bytes(&tiled_arm);
    eprintln!(
        "[pf-attn-tiled] a_scores slab bytes: scores arm {slab_scores}, tiled arm {slab_tiled}; \
         prefill passes: scores arm {}, tiled arm {}",
        scores_arm.prefill_pass_count(),
        tiled_arm.prefill_pass_count()
    );
    assert!(
        tiled_arm.prefill_pass_count() > scores_arm.prefill_pass_count(),
        "the tiled arm records 2 fp8 KV quantizes + 1 qcast + tiled stage1 + stage2 per attention \
         layer in place of the single scores dispatch; an equal pass count means the route never \
         engaged (scores {}, tiled {})",
        scores_arm.prefill_pass_count(),
        tiled_arm.prefill_pass_count()
    );
    assert!(
        slab_tiled <= 4 && slab_scores > slab_tiled,
        "NV_Q3D_PF_ATTN_TILED=1 must gate the dead m x n_heads x max_seq f32 scores slab down to \
         a placeholder allocation (scores arm {slab_scores} B, tiled arm {slab_tiled} B)"
    );

    for len in PROMPT_LENGTHS_COVERING_FULL_PARTIAL_AND_DEAD_ROW_GROUP_CHUNKS {
        let tokens: Vec<u32> = (0..len as u32).map(|i| (i * 7 + 3) % 64).collect();
        let (arg_s, logits_s, ids_s) = chunked_then_greedy(&mut scores_arm, &tokens);
        let (arg_m, logits_m, ids_m) = chunked_then_greedy(&mut mk_arm, &tokens);
        let (arg_t, logits_t, ids_t) = chunked_then_greedy(&mut tiled_arm, &tokens);

        let bitdiff = logits_s
            .iter()
            .zip(logits_m.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let scale = logits_s.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
        let rel = logits_s
            .iter()
            .zip(logits_m.iter())
            .fold(0f32, |a, (s, m)| a.max((s - m).abs() / scale));
        eprintln!(
            "[pf-attn-mk] prompt {len:>3}: {bitdiff}/{} lanes differ, rel {rel:.6}, argmax \
             {arg_s}/{arg_m}, continuation {ids_s:?} vs {ids_m:?} \
             (bit-identity is legal: the attn gate rounds a_attn through bf16, absorbing \
             sub-bf16 reduction-order differences)",
            logits_s.len()
        );
        assert!(
            rel < MK_VS_SCORES_TOLERANCE_IS_REDUCTION_ORDER_NOT_CORRECTNESS,
            "prompt {len}: mk arm drifted rel {rel} past {MK_VS_SCORES_TOLERANCE_IS_REDUCTION_ORDER_NOT_CORRECTNESS} \
             of the scores arm. Both arms read the same bf16 KV rows and accumulate in f32; the \
             only licensed differences are reduction order (online exp2 flash softmax + split-k \
             merge vs two-pass exp over a materialized score row) and q pre-scaled by \
             1/sqrt(head_dim) in f32 before the dot instead of after it, all of which stay well \
             under this bound on a tiny model"
        );
        assert_eq!(
            arg_s, arg_m,
            "prompt {len}: the mk arm flipped the greedy argmax"
        );
        if len == PROMPT_LENGTHS_COVERING_FULL_PARTIAL_AND_DEAD_ROW_GROUP_CHUNKS[0] {
            let mut st = q3d::RefState::new(&cfg);
            let mut want = Vec::new();
            for t in &tokens {
                want = q3d::reference_step(&cfg, &hw, &mut st, *t).expect("cpu reference step");
            }
            let ref_scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
            let ref_rel = logits_m
                .iter()
                .zip(want.iter())
                .fold(0f32, |a, (g, w)| a.max((g - w).abs() / ref_scale));
            eprintln!("[pf-attn-mk] prompt {len:>3}: mk arm vs CPU reference rel {ref_rel:.6}");
            assert!(
                ref_rel < MK_VS_SCORES_TOLERANCE_IS_REDUCTION_ORDER_NOT_CORRECTNESS,
                "prompt {len}: the mk arm drifted rel {ref_rel} from the independent f32 CPU \
                 reference; agreement with the scores arm alone cannot certify a defect the \
                 arms share"
            );
            let ref_rel_t = logits_t
                .iter()
                .zip(want.iter())
                .fold(0f32, |a, (g, w)| a.max((g - w).abs() / ref_scale));
            eprintln!(
                "[pf-attn-tiled] prompt {len:>3}: tiled arm vs CPU reference rel {ref_rel_t:.6}"
            );
            assert!(
                ref_rel_t < TILED_VS_SCORES_TOLERANCE_ADDS_THE_E4M3_KV_ROUND_TRIP_TO_REDUCTION_ORDER,
                "prompt {len}: the tiled arm drifted rel {ref_rel_t} from the independent f32 CPU \
                 reference; the exact-f32 anchor bounds what fp8-vs-fp8 agreement between GPU \
                 arms cannot certify"
            );
        }
        assert_eq!(
            ids_s, ids_m,
            "prompt {len}: the greedy continuation diverged. The mid-stack attention output \
             reaches the carried DeltaNet/KV states, so the arms are not bit-equal at decode \
             time, but drift is bounded by the rel gate above; a greedy flip inside that bound \
             on this tiny model's argmax margins indicates a routing defect (wrong causal end, \
             wrong row group, stale fd uniform), not reduction order"
        );
        let rel_t = logits_s
            .iter()
            .zip(logits_t.iter())
            .fold(0f32, |a, (s, t)| a.max((s - t).abs() / scale));
        eprintln!(
            "[pf-attn-tiled] prompt {len:>3}: rel {rel_t:.6} vs scores, argmax {arg_s}/{arg_t}, \
             continuation {ids_s:?} vs {ids_t:?}"
        );
        assert!(
            rel_t < TILED_VS_SCORES_TOLERANCE_ADDS_THE_E4M3_KV_ROUND_TRIP_TO_REDUCTION_ORDER,
            "prompt {len}: tiled arm drifted rel {rel_t} past \
             {TILED_VS_SCORES_TOLERANCE_ADDS_THE_E4M3_KV_ROUND_TRIP_TO_REDUCTION_ORDER} of the \
             scores arm. Licensed differences are the tiled online softmax reduction order, q \
             pre-scaling via fd scaling, and the per-(position, head) e4m3 KV round trip the \
             chunk streams instead of bf16, all far under this bound on a tiny model"
        );
        assert_eq!(
            arg_s, arg_t,
            "prompt {len}: the tiled arm flipped the greedy argmax"
        );
        assert_eq!(
            ids_s, ids_t,
            "prompt {len}: the tiled arm diverged the greedy continuation; inside the rel bound \
             above a flip indicates a routing defect (wrong causal end, wrong tile row0, stale \
             tiled fd uniform, fp8 slots missing for rows written by M=1 steps), not fp8 noise"
        );
    }
}

#[test]
fn pf_attn_mk_arm_verify_chain_returns_the_scores_arm_tokens() {
    let _g = env_lock();
    if !gpu_present_or_explicit_skip() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0xa77b_3a1c_0002);
    let warmup: Vec<u32> = (0..5u32).map(|i| (i * 11 + 2) % 64).collect();
    let batch: Vec<u32> = (0..6u32).map(|i| (i * 5 + 1) % 64).collect();

    let run = |arm_env: Option<&str>| {
        let _m = M16Pin::set();
        std::env::remove_var(ARM_ENV);
        std::env::set_var(TILED_ENV, "0");
        if let Some(e) = arm_env {
            std::env::set_var(e, "1");
        }
        let mut gpu = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 96).expect("build arm");
        std::env::remove_var(ARM_ENV);
        std::env::remove_var(TILED_ENV);
        for t in &warmup {
            gpu.prefill_step(*t).expect("warmup step");
        }
        gpu.verify_chain_logits(&batch).expect("verify_chain")
    };
    let (toks_scores, rows_scores) = run(None);
    let (toks_mk, rows_mk) = run(Some(ARM_ENV));
    let (toks_tiled, rows_tiled) = run(Some(TILED_ENV));
    let bitdiff = rows_scores
        .iter()
        .zip(rows_mk.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let scale = rows_scores.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
    let rel = rows_scores
        .iter()
        .zip(rows_mk.iter())
        .fold(0f32, |a, (s, m)| a.max((s - m).abs() / scale));
    eprintln!(
        "[pf-attn-mk] verify_chain scores {toks_scores:?} mk {toks_mk:?}; row logits {bitdiff}/{} \
         lanes differ, rel {rel:.6}",
        rows_scores.len()
    );
    assert_eq!(toks_scores.len(), batch.len(), "verify_chain returned short");
    assert!(
        rel < MK_VS_SCORES_TOLERANCE_IS_REDUCTION_ORDER_NOT_CORRECTNESS,
        "verify_chain row logits drifted rel {rel} across arms; same bf16 KV, f32 accumulation, \
         only reduction order and q pre-scaling may differ"
    );
    assert_eq!(
        toks_scores, toks_mk,
        "verify_chain rides the same pf pass list; the mk arm's per-group fd uniforms must map \
         batch row r to causal end pos base+r+1 exactly as ck.m_live/base do on the scores arm"
    );
    let rel_t = rows_scores
        .iter()
        .zip(rows_tiled.iter())
        .fold(0f32, |a, (s, t)| a.max((s - t).abs() / scale));
    eprintln!(
        "[pf-attn-tiled] verify_chain scores {toks_scores:?} tiled {toks_tiled:?}; row logits \
         rel {rel_t:.6}"
    );
    assert!(
        rel_t < TILED_VS_SCORES_TOLERANCE_ADDS_THE_E4M3_KV_ROUND_TRIP_TO_REDUCTION_ORDER,
        "verify_chain row logits drifted rel {rel_t} between the tiled and scores arms; only \
         reduction order, fd q scaling, and the e4m3 KV round trip may differ, and the warmup \
         M=1 steps must have kept the fp8 cache in sync for the rows the chunk did not write"
    );
    assert_eq!(
        toks_scores, toks_tiled,
        "verify_chain rides the same pf pass list; the tiled arm's whole-chunk fd view must map \
         batch row r to causal end pos base+r+1 exactly as ck.m_live/base do on the scores arm"
    );
}

const FP8_ROUND_TRIP_REL_BAND_PROVES_THE_QUANTIZE_ENGAGED_NOT_A_BF16_BIT_COPY: (f32, f32) =
    (0.001, 0.5);

#[test]
fn the_tiled_arm_fp8_cache_holds_a_real_e4m3_round_trip_of_the_bf16_kv_rows() {
    let _g = env_lock();
    if !gpu_present_or_explicit_skip() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0xa77b_3a1c_0001);
    let max_seq = 96usize;
    let _m = M16Pin::set();
    std::env::set_var(TILED_ENV, "1");
    let mut gpu = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, max_seq).expect("build tiled arm");
    std::env::remove_var(TILED_ENV);
    let m = gpu.prefill_chunk_len();
    let tokens: Vec<u32> = (0..m as u32).map(|i| (i * 7 + 3) % 64).collect();
    let done = gpu.prefill_tokens(&tokens).expect("prefill");
    assert_eq!(done, m, "one full chunk must land through the pf list");

    let n_kv = cfg.num_key_value_heads;
    let hd = cfg.head_dim;
    let kc_bytes = (max_seq * n_kv * hd * 2) as u64;
    let kc8_bytes = (max_seq * n_kv * hd) as u64;
    let sc_bytes = (max_seq * n_kv * 4) as u64;
    let want = [kc_bytes, kc_bytes, kc8_bytes, kc8_bytes, sc_bytes, sc_bytes];
    let e4m3 = |b: u32| -> f32 {
        let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
        let e = ((b >> 3) & 0xf) as i32;
        let mant = (b & 7) as f32;
        if e == 0 {
            s * mant / 8.0 * 2f32.powi(-6)
        } else {
            s * (1.0 + mant / 8.0) * 2f32.powi(e - 7)
        }
    };
    let bufs = gpu.debug_state_buffer_words_for_test();
    let n_full_attn = cfg
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::FullAttention))
        .count();
    let mut groups = 0usize;
    let mut i = 0;
    while i + want.len() <= bufs.len() {
        if (0..want.len()).all(|j| bufs[i + j].0 == want[j]) {
            for (cache, cache8, scales, side) in
                [(&bufs[i].1, &bufs[i + 2].1, &bufs[i + 4].1, "k"),
                 (&bufs[i + 1].1, &bufs[i + 3].1, &bufs[i + 5].1, "v")]
            {
                let mut maxrel = 0f32;
                let mut nz = 0usize;
                for slot in 0..m {
                    for kvh in 0..n_kv {
                        let sc = f32::from_bits(scales[slot * n_kv + kvh]);
                        for d in 0..hd {
                            let e = (slot * n_kv + kvh) * hd + d;
                            let w = cache[e / 2];
                            let bf = f32::from_bits(if e % 2 == 0 {
                                (w & 0xffff) << 16
                            } else {
                                w & 0xffff_0000
                            });
                            let code = (cache8[e / 4] >> (8 * (e % 4))) & 0xff;
                            let q = e4m3(code) * sc;
                            if bf != 0.0 {
                                nz += 1;
                                maxrel = maxrel.max((q - bf).abs() / bf.abs().max(1e-6));
                            }
                        }
                    }
                }
                eprintln!(
                    "[pf-attn-tiled] attn layer group {groups} {side}: nz {nz}, max rel \
                     |fp8 - bf16| {maxrel:.6}"
                );
                assert_eq!(
                    nz,
                    m * n_kv * hd,
                    "every chunk KV element must be populated in both caches"
                );
                let (lo, hi) = FP8_ROUND_TRIP_REL_BAND_PROVES_THE_QUANTIZE_ENGAGED_NOT_A_BF16_BIT_COPY;
                assert!(
                    maxrel > lo && maxrel < hi,
                    "{side} cache round-trip max rel {maxrel} escaped ({lo}, {hi}): below means \
                     q3w_pf_quantize_kv_fp8_m never actually quantized (a bit-copy would make the \
                     tiled-vs-scores gate above vacuous, since bf16 residual adds absorb genuine \
                     e4m3 noise on this tiny model); above means the scales or byte packing are \
                     wrong"
                );
            }
            groups += 1;
            i += want.len();
        } else {
            i += 1;
        }
    }
    assert_eq!(
        groups, n_full_attn,
        "expected one bf16+fp8+scales state sextet per full-attention layer"
    );
}
