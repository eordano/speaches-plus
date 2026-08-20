#![cfg(feature = "cuda")]

mod hub_dirs;

use candle_core::Device;
use minijinja::{context, Environment, Value as JinjaValue};
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_specdecode::qwen38_mtp::{
    mtp_chain_depth_from_env, mtp_draft_dir_override_from_env, Qwen38DenseMtpHead,
    Qwen38MtpGraphedDecodeSession, Qwen38MtpSelfSpecEngine, MTP_WEIGHTS_FILE_NAME,
};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::Path;
use tokenizers::Tokenizer;
mod common;
use common::raise_exception;
use common::render_real_chat_template_no_thinking;
use common::strftime_now;
use common::GraphedRunRecord;
use common::generate_graphed;

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";

#[test]
fn real_mtp_shard_loads_the_full_head_on_cuda_without_the_trunk() {
    use nv_specdecode::qwen38_mtp::QWEN38_27B_MTP_GEOMETRY;
    let Some(dir) = hub_dirs::snapshot(NVFP4_REPO, &["config.json", MTP_WEIGHTS_FILE_NAME]) else {
        if std::env::var("NV_Q38_REQUIRE_MTP_SHARD").as_deref() == Ok("1") {
            panic!("NV_Q38_REQUIRE_MTP_SHARD=1 but no snapshot carries {MTP_WEIGHTS_FILE_NAME}");
        }
        eprintln!("SKIP: no {NVFP4_REPO} snapshot with {MTP_WEIGHTS_FILE_NAME}");
        return;
    };
    let Ok(device) = Device::new_cuda_with_stream(0) else {
        eprintln!("[skip] no cuda device; the real MTP shard load needs the card");
        return;
    };
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("config json");
    let t = &v["text_config"];
    let eps = t["rms_norm_eps"].as_f64().expect("rms_norm_eps");
    let head_dim = t["head_dim"].as_u64().expect("head_dim") as f64;
    let rotary_dim = (head_dim * t["partial_rotary_factor"].as_f64().expect("factor")) as usize;

    let map = candle_core::safetensors::load(dir.join(MTP_WEIGHTS_FILE_NAME), &device)
        .expect("load the real model_mtp.safetensors onto the device");
    assert_eq!(map.len(), 15, "the shipped MTP shard carries exactly 15 tensors");
    for (name, tensor) in &map {
        assert_eq!(
            tensor.dtype(),
            candle_core::DType::BF16,
            "{name}: the shipped MTP head is bf16 throughout"
        );
    }
    let head = Qwen38DenseMtpHead::from_map_for_geometry(
        &map,
        QWEN38_27B_MTP_GEOMETRY,
        eps,
        rotary_dim,
        candle_core::DType::BF16,
    )
    .expect("the real shard must construct the head through the same path serving uses");
    let cache = head
        .new_kv_cache(1024, &device)
        .expect("drafter kv cache for the real geometry");
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.max_seq(), 1024);
    eprintln!(
        "[q38-mtp-real-load] basis: checkpoint={} file={MTP_WEIGHTS_FILE_NAME} backend=cuda \
         tensors=15 dtype=bf16 rotary_dim={rotary_dim} eps={eps}",
        dir.display()
    );
}

#[test]
#[ignore = "loads the ~17 GB Qwen3.8-27B NVFP4 checkpoint; set NV_Q38_MTP=1; depends on the track-2 dense NVFP4 loader"]
fn qwen38_mtp_spec_matches_reference_and_reports_acceptance() {
    if std::env::var("NV_Q38_MTP").as_deref() != Ok("1") {
        panic!("set NV_Q38_MTP=1 to run (it must never silently skip)");
    }
    let dir = hub_dirs::snapshot(
        NVFP4_REPO,
        &[
            "config.json",
            "tokenizer.json",
            "chat_template.jinja",
            MTP_WEIGHTS_FILE_NAME,
        ],
    )
    .expect("Qwen3.8-27B-NVFP4 snapshot with the MTP shard not found in the hub cache");

    let question = std::env::var("NV_Q38_Q")
        .unwrap_or_else(|_| "What is the capital of France? Answer in one short sentence.".into());
    let prompt_text = render_real_chat_template_no_thinking(&dir, &question);
    assert!(
        prompt_text.contains("<|im_start|>user") && prompt_text.ends_with("<think>\n\n</think>\n\n"),
        "rendered template drifted from the no-thinking generation prompt: {prompt_text:?}"
    );

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let prompt: Vec<u32> = tok
        .encode(prompt_text.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();
    let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tok.token_to_id(s))
        .collect();
    assert!(!stop.is_empty(), "tokenizer lost the im_end/endoftext stop tokens");

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let base = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("track-2 dense NVFP4 loader must load Qwen3.8-27B before this A/B can run");

    let mtp = Qwen38DenseMtpHead::from_checkpoint(
        mtp_draft_dir_override_from_env().as_deref(),
        &dir,
        &base,
        &device,
    )
    .expect("MTP head loads from the shipped model_mtp.safetensors");

    const MAX_NEW: usize = 64;
    const MAX_SEQ: usize = 1024;
    let k = mtp_chain_depth_from_env();
    let eng = Qwen38MtpSelfSpecEngine::new(&base, &mtp, k)
        .expect("k fits the lm_head rows-per-call ceiling")
        .with_stop_ids(stop.clone());

    let t0 = std::time::Instant::now();
    let (ref_ids, _) = eng
        .generate_reference(&prompt, MAX_NEW, MAX_SEQ)
        .expect("reference");
    let ref_s = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let (spec_ids, stats) = eng.generate_greedy(&prompt, MAX_NEW, MAX_SEQ).expect("spec");
    let spec_s = t1.elapsed().as_secs_f64();

    let text = tok.decode(&spec_ids, false).unwrap_or_default();
    eprintln!(
        "Q38 MTP A/B basis=(model=unsloth/Qwen3.8-27B-NVFP4, drafter=shipped mtp head bf16, \
         k={k}, max_new={MAX_NEW}, prompt_toks={}, template=official no-thinking, greedy): \
         new_toks={} accept_rate={:.3} pos0_accept={:.3} tokens_per_round={:.2} draft_ms={:.1} \
         verify_ms={:.1} ref={:.2}s spec={:.2}s ratio={:.2}x text={text:?}",
        prompt.len(),
        spec_ids.len(),
        stats.accept_rate(),
        stats.pos0_accept_rate(),
        stats.tokens_per_round(),
        stats.draft_ms,
        stats.verify_ms,
        ref_s,
        spec_s,
        ref_s / spec_s.max(1e-9),
    );

    assert_eq!(
        spec_ids, ref_ids,
        "lossless bar: the speculative stream must equal the single-token greedy reference \
         exactly (the qwen3.6 #92 wiring bar); any divergence here is a rollback or reanchor bug"
    );
    assert!(
        spec_ids.last().is_some_and(|t| stop.contains(t)),
        "generation must stop on a stop token; got {spec_ids:?}"
    );
    assert!(
        spec_ids.iter().any(|t| !stop.contains(t)),
        "generation is empty of content tokens, the rates above are measured on nothing"
    );
    assert!(
        stats.accept_rate() > 0.0,
        "zero acceptance: the MTP head never drafted a token the base agreed with; check the \
         zero-centered norm convention and the fc concat order before blaming the checkpoint"
    );
}

const MROW_PARITY_ABS_TOL_0_02_BOUNDS_GEMV_VS_GEMM_PROJECTION_REORDER_ON_SMALL_RANDOM_WEIGHTS:
    f32 = 0.02;
const MROW_PARITY_WARM_STEPS_3_SO_PARITY_IS_MEASURED_FROM_A_NONZERO_STATE: usize = 3;
const MROW_PARITY_ROWS_4_MATCHES_A_K3_MTP_VERIFY_CHAIN: usize = 4;

#[test]
fn mrow_verify_matches_the_per_row_fused_loop_on_a_synthetic_gdn_layer() {
    use candle_core::{DType, Tensor};
    use nv_layers::linear::Linear;
    use nv_layers::linear_attn::{LinAttnState, LinearAttention, LinearAttentionConfig};

    let Ok(device) = Device::new_cuda_with_stream(0) else {
        eprintln!("[skip] no cuda device; the m-row GDN verify parity needs the card");
        return;
    };
    let cfg = LinearAttentionConfig {
        hidden_size: 256,
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_conv_kernel_dim: 4,
        mamba_ssm_dtype: DType::F32,
        rms_eps: 1e-6,
    };
    let hidden = cfg.hidden_size;
    let conv_dim = cfg.conv_dim();
    let value_dim = cfg.value_dim();
    let n_v = cfg.linear_num_value_heads;
    let bf = |t: Tensor| t.to_dtype(DType::BF16).expect("bf16 cast");
    let rnd = |shape: &[usize]| {
        bf(Tensor::randn(0f32, 0.05f32, shape, &device).expect("randn"))
    };
    let la = LinearAttention::new(
        cfg,
        Linear::new(rnd(&[conv_dim, hidden]), None).expect("qkv"),
        Linear::new(rnd(&[value_dim, hidden]), None).expect("z"),
        Linear::new(rnd(&[n_v, hidden]), None).expect("a"),
        Linear::new(rnd(&[n_v, hidden]), None).expect("b"),
        rnd(&[conv_dim, 1, cfg.linear_conv_kernel_dim]),
        rnd(&[n_v]),
        rnd(&[n_v]),
        rnd(&[cfg.linear_value_head_dim]),
        Linear::new(rnd(&[hidden, value_dim]), None).expect("out"),
    )
    .expect("synthetic GDN layer with the q38 head geometry");
    assert!(
        la.fused_decode_supported(),
        "the synthetic geometry must take the fused decode path or this parity proves nothing"
    );

    let st_a = la.new_fused_state(&device).expect("state A");
    let st_b = la.new_fused_state(&device).expect("state B");
    for step in 0..MROW_PARITY_WARM_STEPS_3_SO_PARITY_IS_MEASURED_FROM_A_NONZERO_STATE {
        let x = rnd(&[1, 1, hidden]);
        la.forward_decode_fused(&x, &st_a)
            .expect("warm A")
            .unwrap_or_else(|| panic!("warm step {step}: fused decode refused arm A"));
        la.forward_decode_fused(&x, &st_b)
            .expect("warm B")
            .unwrap_or_else(|| panic!("warm step {step}: fused decode refused arm B"));
    }
    device.synchronize().expect("sync after warm");

    let m = MROW_PARITY_ROWS_4_MATCHES_A_K3_MTP_VERIFY_CHAIN;
    let x_rows = rnd(&[1, m, hidden]);
    let ckpts_a: Vec<LinAttnState> = (0..m)
        .map(|_| la.new_fused_state(&device).expect("ckpt A"))
        .collect();
    let ckpts_b: Vec<LinAttnState> = la
        .new_fused_verify_ckpt_rows_off_one_slab_so_chunk_kernels_write_ckpts_in_place(&device, m)
        .expect("ckpt B slab matches what the serving verify lane preallocates");

    let mut outs_a: Vec<Tensor> = Vec::with_capacity(m);
    for j in 0..m {
        let xj = x_rows.narrow(1, j, 1).expect("row").copy().expect("copy");
        let out_j = la
            .forward_decode_fused(&xj, &st_a)
            .expect("per-row fused decode")
            .unwrap_or_else(|| panic!("row {j}: fused decode refused"));
        ckpts_a[j].copy_data_from(&st_a).expect("ckpt A copy");
        outs_a.push(out_j);
    }
    let refs: Vec<&Tensor> = outs_a.iter().collect();
    let out_a = Tensor::cat(&refs, 1).expect("cat A");

    let out_b = la
        .forward_verify_mrow_projections_once_because_per_row_fused_steps_reread_every_gdn_weight(
            &x_rows, &st_b, &ckpts_b,
        )
        .expect("m-row verify")
        .expect("the m-row primitive refused a geometry it was built for");
    device.synchronize().expect("sync after both arms");

    let host = |t: &Tensor| -> Vec<f32> {
        t.to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("host")
    };
    let tol = MROW_PARITY_ABS_TOL_0_02_BOUNDS_GEMV_VS_GEMM_PROJECTION_REORDER_ON_SMALL_RANDOM_WEIGHTS;
    let max_abs = |a: &[f32], b: &[f32]| -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    let d_out = max_abs(&host(&out_a), &host(&out_b));
    let d_live_rec = max_abs(
        &host(st_a.recurrent_state()),
        &host(st_b.recurrent_state()),
    );
    let d_live_conv = max_abs(&host(st_a.conv_state()), &host(st_b.conv_state()));
    let mut d_ck_rec = 0f32;
    let mut d_ck_conv = 0f32;
    for j in 0..m {
        d_ck_rec = d_ck_rec.max(max_abs(
            &host(ckpts_a[j].recurrent_state()),
            &host(ckpts_b[j].recurrent_state()),
        ));
        d_ck_conv = d_ck_conv.max(max_abs(
            &host(ckpts_a[j].conv_state()),
            &host(ckpts_b[j].conv_state()),
        ));
    }
    eprintln!(
        "Q38 mrow-verify parity basis=(synthetic q38 head geometry, warm {} steps, m={m}, \
         arm A=per-row forward_decode_fused loop, arm B=m-row projections-once chunk): \
         max_abs out={d_out:.5} live_rec={d_live_rec:.5} live_conv={d_live_conv:.5} \
         ckpt_rec={d_ck_rec:.5} ckpt_conv={d_ck_conv:.5} tol={tol}",
        MROW_PARITY_WARM_STEPS_3_SO_PARITY_IS_MEASURED_FROM_A_NONZERO_STATE
    );
    for (name, d) in [
        ("out", d_out),
        ("live_rec", d_live_rec),
        ("live_conv", d_live_conv),
        ("ckpt_rec", d_ck_rec),
        ("ckpt_conv", d_ck_conv),
    ] {
        assert!(
            d <= tol,
            "m-row verify diverged from the per-row fused loop on {name}: {d} > {tol}"
        );
    }
}

const V2_MAX_NEW_64_MATCHES_THE_LLAMA_CPP_SERVER_MTP_CHAT_N_PREDICT: usize = 64;
const V2_MIN_CONTENT_TOKENS_48_SO_THE_RATES_ARE_MEASURED_ON_REAL_GENERATION: usize = 48;
const V2_WARM_TOKENS_16_UNTIMED_BEFORE_EACH_ARM_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW: usize = 16;
const V2_MAX_SEQ: usize = 2048;

const V2_CREATIVE_PROMPTS_MATCH_THE_RECORD_BAR_HONESTY_BRACKET: [&str; 3] = [
    "Write a short story about a lighthouse keeper who discovers something unusual.",
    "Invent a new board game and explain its rules.",
    "Describe an imaginary city built inside a mountain.",
];

const V2_JUNK_DRAFT_TOKEN_ANY_POLICY_MUST_EMIT_THE_SAME_STREAM_IF_ROLLBACK_IS_LOSSLESS: u32 = 0;

fn generate_with_forced_junk_drafts_proving_draft_policy_invariance_of_the_emitted_stream(
    base: &Qwen3Moe,
    mtp: &Qwen38DenseMtpHead,
    k: usize,
    prompt: &[u32],
    max_new: usize,
    max_seq: usize,
    stop: &[u32],
) -> Vec<u32> {
    let mut session = nv_specdecode::qwen38_mtp::Qwen38MtpDecodeSession::start(
        base, mtp, k, prompt, max_seq,
    )
    .expect("junk-draft session start");
    let junk =
        vec![V2_JUNK_DRAFT_TOKEN_ANY_POLICY_MUST_EMIT_THE_SAME_STREAM_IF_ROLLBACK_IS_LOSSLESS; k];
    let anchor = session.anchor();
    let mut generated: Vec<u32> = vec![anchor];
    if stop.contains(&anchor) {
        return generated;
    }
    'rounds: while generated.len() < max_new && session.round_fits() {
        let emitted = session
            .round_with_drafts_from_a_clairvoyant_test_oracle(&junk)
            .expect("junk-draft round");
        for &t in emitted.iter() {
            generated.push(t);
            if stop.contains(&t) {
                break 'rounds;
            }
            if generated.len() >= max_new {
                break 'rounds;
            }
        }
    }
    generated
}

fn leading_agreement_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[test]
#[ignore = "loads the ~54 GB Qwen3.8-27B NVFP4 checkpoint; set NV_Q38_MTP=1 -- warm A/B v2: both arms get an untimed 16-token warmup generation before their timed run (v1 timed a cold reference against a warm spec arm), the prompts are the record-bar creative set eliciting >=48 content tokens (v1 asserted a stop token within 64, selecting for short deterministic answers), and steady-state ms/tok is (draft+verify)/new_toks for spec and sum(round_ms)/new_toks for the normal arm, both excluding prefill; correctness bar is DRAFT-POLICY INVARIANCE (junk drafts must emit the identical stream, isolating rollback from kernel numerics) because the m=k+1 batched verify and the m=1 reference decode run different kernels whose fp drift flips argmax at the near-ties creative text is full of -- the exact-match-vs-reference bar lives in v1 where large logit gaps keep it stable"]
fn qwen38_mtp_warm_ab_v2_creative_prompts_steady_state_ms_per_token() {
    if std::env::var("NV_Q38_MTP").as_deref() != Ok("1") {
        panic!("set NV_Q38_MTP=1 to run (it must never silently skip)");
    }
    let dir = hub_dirs::snapshot(
        NVFP4_REPO,
        &[
            "config.json",
            "tokenizer.json",
            "chat_template.jinja",
            MTP_WEIGHTS_FILE_NAME,
        ],
    )
    .expect("Qwen3.8-27B-NVFP4 snapshot with the MTP shard not found in the hub cache");

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tok.token_to_id(s))
        .collect();
    assert!(!stop.is_empty(), "tokenizer lost the im_end/endoftext stop tokens");

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let base = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("track-2 dense NVFP4 loader must load Qwen3.8-27B before this A/B can run");
    drop(weights);

    let mtp = Qwen38DenseMtpHead::from_checkpoint(
        mtp_draft_dir_override_from_env().as_deref(),
        &dir,
        &base,
        &device,
    )
    .expect("MTP head loads from the shipped model_mtp.safetensors");

    let k = mtp_chain_depth_from_env();
    let eng = Qwen38MtpSelfSpecEngine::new(&base, &mtp, k)
        .expect("k fits the lm_head rows-per-call ceiling")
        .with_stop_ids(stop.clone());

    let max_new = V2_MAX_NEW_64_MATCHES_THE_LLAMA_CPP_SERVER_MTP_CHAT_N_PREDICT;
    let warm_new = V2_WARM_TOKENS_16_UNTIMED_BEFORE_EACH_ARM_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW;

    for (pi, question) in V2_CREATIVE_PROMPTS_MATCH_THE_RECORD_BAR_HONESTY_BRACKET
        .iter()
        .enumerate()
    {
        let prompt_text = render_real_chat_template_no_thinking(&dir, question);
        let prompt: Vec<u32> = tok
            .encode(prompt_text.as_str(), false)
            .expect("encode")
            .get_ids()
            .to_vec();

        eng.generate_reference(&prompt, warm_new, V2_MAX_SEQ)
            .expect("untimed reference warmup arm");
        let t0 = std::time::Instant::now();
        let (ref_ids, ref_stats) = eng
            .generate_reference(&prompt, max_new, V2_MAX_SEQ)
            .expect("reference");
        let ref_s = t0.elapsed().as_secs_f64();

        eng.generate_greedy(&prompt, warm_new, V2_MAX_SEQ)
            .expect("untimed spec warmup arm");
        let t1 = std::time::Instant::now();
        let (spec_ids, stats) = eng
            .generate_greedy(&prompt, max_new, V2_MAX_SEQ)
            .expect("spec");
        let spec_s = t1.elapsed().as_secs_f64();

        let junk_ids = generate_with_forced_junk_drafts_proving_draft_policy_invariance_of_the_emitted_stream(
            &base, &mtp, k, &prompt, max_new, V2_MAX_SEQ, &stop,
        );
        assert_eq!(
            spec_ids, junk_ids,
            "draft-policy invariance broken on prompt {pi}: every emitted token is the batched \
             verify's own greedy continuation of the committed prefix, so the stream must not \
             depend on what the drafter proposed; a mismatch is a rollback/reanchor/lin-ckpt bug, \
             not numerics"
        );
        let ref_agreement_prefix = leading_agreement_prefix_len(&spec_ids, &ref_ids);
        let content = spec_ids.iter().filter(|t| !stop.contains(t)).count();
        assert!(
            content >= V2_MIN_CONTENT_TOKENS_48_SO_THE_RATES_ARE_MEASURED_ON_REAL_GENERATION,
            "prompt {pi} elicited only {content} content tokens of the >= {} this A/B \
             requires; a short deterministic answer flatters acceptance exactly the way \
             v1's stop-within-64 assert did",
            V2_MIN_CONTENT_TOKENS_48_SO_THE_RATES_ARE_MEASURED_ON_REAL_GENERATION
        );

        let new_toks = spec_ids.len();
        let spec_steady_ms_tok = (stats.draft_ms + stats.verify_ms) / new_toks as f64;
        let spec_steady_incl_commit_ms_tok =
            (stats.draft_ms + stats.verify_ms + stats.commit_ms) / new_toks as f64;
        let ref_new_toks = ref_ids.len();
        let ref_steady_ms_tok =
            ref_stats.round_ms.iter().sum::<f64>() / ref_new_toks.max(1) as f64;
        let text = tok.decode(&spec_ids, false).unwrap_or_default();
        eprintln!(
            "Q38 MTP A/B v2 basis=(model=unsloth/Qwen3.8-27B-NVFP4, drafter=shipped mtp head \
             bf16, k={k}, max_new={max_new}, prompt={pi} prompt_toks={}, template=official \
             no-thinking, greedy, both arms warmed {warm_new} untimed tokens, steady-state \
             excludes prefill, normal arm=eager forward_with_cache): new_toks={new_toks} \
             ref_agreement_prefix={ref_agreement_prefix} draft_policy_invariance=held \
             accept_rate={:.3} pos0_accept={:.3} tokens_per_round={:.2} \
             spec_steady_ms_tok={spec_steady_ms_tok:.2} \
             spec_steady_incl_commit_ms_tok={spec_steady_incl_commit_ms_tok:.2} \
             ref_steady_ms_tok={ref_steady_ms_tok:.2} draft_ms={:.1} verify_ms={:.1} \
             commit_ms={:.1} ref_wall_s={ref_s:.2} spec_wall_s={spec_s:.2} wall_ratio={:.2}x \
             text={text:?}",
            prompt.len(),
            stats.accept_rate(),
            stats.pos0_accept_rate(),
            stats.tokens_per_round(),
            stats.draft_ms,
            stats.verify_ms,
            stats.commit_ms,
            ref_s / spec_s.max(1e-9),
        );
    }
}

#[test]
#[ignore = "loads the ~54 GB Qwen3.8-27B NVFP4 checkpoint; set NV_Q38_MTP=1 -- warm A/B v3: the spec arm runs the GRAPHED m=k+1 verify (one CUDA graph replay per round, mk splitk attention on the fp8 kv, fused GDN sequential steps with device-side rollback checkpoints) against the same eager m=1 reference as v2; correctness bar is draft-policy invariance through the graphed path, junk drafts must emit the identical stream; steady-state is reported both including the first round (which pays graph capture after every prefill, the per-request serving cost in this harness) and excluding it (the hot-engine rate)"]
fn qwen38_mtp_warm_ab_v3_graphed_verify_creative_prompts_steady_state_ms_per_token() {
    if std::env::var("NV_Q38_MTP").as_deref() != Ok("1") {
        panic!("set NV_Q38_MTP=1 to run (it must never silently skip)");
    }
    let dir = hub_dirs::snapshot(
        NVFP4_REPO,
        &[
            "config.json",
            "tokenizer.json",
            "chat_template.jinja",
            MTP_WEIGHTS_FILE_NAME,
        ],
    )
    .expect("Qwen3.8-27B-NVFP4 snapshot with the MTP shard not found in the hub cache");

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tok.token_to_id(s))
        .collect();
    assert!(!stop.is_empty(), "tokenizer lost the im_end/endoftext stop tokens");

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = nv_models::qwen3_5_moe::Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let base = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("track-2 dense NVFP4 loader must load Qwen3.8-27B before this A/B can run");
    drop(weights);

    let mut engine = nv_models::graph_engine::GraphedQwen3Moe::new(base, &device, V2_MAX_SEQ)
        .expect("graphed engine over the dense arm");
    let mtp = Qwen38DenseMtpHead::from_checkpoint(
        mtp_draft_dir_override_from_env().as_deref(),
        &dir,
        engine.underlying(),
        &device,
    )
    .expect("MTP head loads from the shipped model_mtp.safetensors");

    let k = mtp_chain_depth_from_env();
    let max_new = V2_MAX_NEW_64_MATCHES_THE_LLAMA_CPP_SERVER_MTP_CHAT_N_PREDICT;
    let warm_new = V2_WARM_TOKENS_16_UNTIMED_BEFORE_EACH_ARM_BECAUSE_A_COLD_CLOCK_READS_2X_SLOW;

    for (pi, question) in V2_CREATIVE_PROMPTS_MATCH_THE_RECORD_BAR_HONESTY_BRACKET
        .iter()
        .enumerate()
    {
        let prompt_text = render_real_chat_template_no_thinking(&dir, question);
        let prompt: Vec<u32> = tok
            .encode(prompt_text.as_str(), false)
            .expect("encode")
            .get_ids()
            .to_vec();

        let (ref_ids, ref_steady_ms_tok, ref_s) = {
            let eng = Qwen38MtpSelfSpecEngine::new(engine.underlying(), &mtp, k)
                .expect("eager reference engine")
                .with_stop_ids(stop.clone());
            eng.generate_reference(&prompt, warm_new, V2_MAX_SEQ)
                .expect("untimed reference warmup arm");
            let t0 = std::time::Instant::now();
            let (ids, stats) = eng
                .generate_reference(&prompt, max_new, V2_MAX_SEQ)
                .expect("reference");
            let ref_s = t0.elapsed().as_secs_f64();
            let steady = stats.round_ms.iter().sum::<f64>() / ids.len().max(1) as f64;
            (ids, steady, ref_s)
        };

        generate_graphed(&mut engine, &mtp, k, &prompt, warm_new, &stop, None);
        let t1 = std::time::Instant::now();
        let spec = generate_graphed(&mut engine, &mtp, k, &prompt, max_new, &stop, None);
        let spec_s = t1.elapsed().as_secs_f64();

        let junk = generate_graphed(
            &mut engine,
            &mtp,
            k,
            &prompt,
            max_new,
            &stop,
            Some(V2_JUNK_DRAFT_TOKEN_ANY_POLICY_MUST_EMIT_THE_SAME_STREAM_IF_ROLLBACK_IS_LOSSLESS),
        );
        assert_eq!(
            spec.ids, junk.ids,
            "draft-policy invariance broken on prompt {pi} through the GRAPHED verify: every \
             emitted token is the batched verify's own greedy continuation of the committed \
             prefix; a mismatch is a rollback/reanchor/lin-ckpt/graph-replay bug, not numerics"
        );
        let ref_agreement_prefix = leading_agreement_prefix_len(&spec.ids, &ref_ids);
        let content = spec.ids.iter().filter(|t| !stop.contains(t)).count();
        assert!(
            content >= V2_MIN_CONTENT_TOKENS_48_SO_THE_RATES_ARE_MEASURED_ON_REAL_GENERATION,
            "prompt {pi} elicited only {content} content tokens of the >= {} this A/B requires",
            V2_MIN_CONTENT_TOKENS_48_SO_THE_RATES_ARE_MEASURED_ON_REAL_GENERATION
        );

        let stats = &spec.stats;
        let new_toks = spec.ids.len();
        let spec_steady_ms_tok = (stats.draft_ms + stats.verify_ms) / new_toks as f64;
        let spec_steady_incl_commit_ms_tok =
            (stats.draft_ms + stats.verify_ms + stats.commit_ms) / new_toks as f64;
        let (hot_ms, hot_toks) = spec
            .round_wall_ms
            .iter()
            .zip(spec.round_emitted.iter())
            .skip(1)
            .fold((0f64, 0usize), |(ms, tk), (w, e)| (ms + w, tk + e));
        let spec_hot_ms_tok = hot_ms / hot_toks.max(1) as f64;
        let capture_round_ms = spec.round_wall_ms.first().copied().unwrap_or(0.0);
        let text = tok.decode(&spec.ids, false).unwrap_or_default();
        eprintln!(
            "Q38 MTP A/B v3 basis=(model=unsloth/Qwen3.8-27B-NVFP4, drafter=shipped mtp head \
             bf16, k={k}, max_new={max_new}, prompt={pi} prompt_toks={}, template=official \
             no-thinking, greedy, both arms warmed {warm_new} untimed tokens, steady-state \
             excludes prefill, spec arm=graphed m=k+1 verify [mk splitk fp8 attention + fused \
             GDN sequential + device rollback ckpts], normal arm=eager forward_with_cache, \
             capture paid once per session inside round 0): new_toks={new_toks} \
             ref_agreement_prefix={ref_agreement_prefix} draft_policy_invariance=held \
             accept_rate={:.3} pos0_accept={:.3} tokens_per_round={:.2} \
             spec_steady_ms_tok={spec_steady_ms_tok:.2} \
             spec_steady_incl_commit_ms_tok={spec_steady_incl_commit_ms_tok:.2} \
             spec_hot_ms_tok_excl_round0={spec_hot_ms_tok:.2} \
             capture_round0_ms={capture_round_ms:.1} ref_steady_ms_tok={ref_steady_ms_tok:.2} \
             draft_ms={:.1} verify_ms={:.1} commit_ms={:.1} ref_wall_s={ref_s:.2} \
             spec_wall_s={spec_s:.2} wall_ratio={:.2}x text={text:?}",
            prompt.len(),
            stats.accept_rate(),
            stats.pos0_accept_rate(),
            stats.tokens_per_round(),
            stats.draft_ms,
            stats.verify_ms,
            stats.commit_ms,
            ref_s / spec_s.max(1e-9),
        );
    }
}
