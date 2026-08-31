#![cfg(feature = "cuda")]

mod hub_dirs;

use candle_core::{DType, Device, Tensor};
use nv_models::qwen3_5_moe::{
    Qwen3Moe, Qwen3MoeKvCache, Qwen3_5DenseConfig,
    NV_Q38_PREFILL_CHUNK_ENV_DEFAULT_4096_SET_0_FOR_THE_ONE_SHOT_FORWARD_WHOSE_TRANSIENTS_MEASURED_0_21_MIB_PER_TOKEN as CHUNK_ENV,
};
use nv_specdecode::qwen38_mtp::{
    mtp_chain_depth_from_env, mtp_draft_dir_override_from_env, Qwen38DenseMtpHead,
    Qwen38MtpSelfSpecEngine,
};
use nv_weights::{QuantizationConfig, WeightLoader};
use tokenizers::Tokenizer;

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";

const AB_CHUNK_4096_THE_RECOMMENDED_SERVE_DEFAULT: usize = 4096;

const AB_PROMPT_TARGET_TOKENS_12000_SPANS_TWO_FULL_CHUNKS_AND_A_TAIL: usize = 12_000;

const AB_TEACHER_FORCED_STEPS_64_EACH_READS_THE_PREFILLED_KV_AND_GDN_STATE: usize = 64;

const CONTRACT_IS_A_TOLERANCE_PAIR_NOT_BIT_IDENTITY_BECAUSE_NVFP4_AND_FLASH_KERNEL_CONFIGS_VARY_WITH_M_THE_SAME_DRIFT_CLASS_MTP_VERIFY_INC2_PINS:
    &str = "the tiny-model suite proves the chunk seam (state carry, KV writes, positions) \
            bitwise; on the real NVFP4 checkpoint a chunked prefill picks different kernel \
            configs for different M, so the gates here are a teacher-forced logit/NLL band and \
            acceptance preservation, not token equality";

const TEACHER_FORCED_ARGMAX_AGREEMENT_FLOOR_58_OF_64_MEASURED_63_OF_64_THE_ONE_FLIP_WAS_A_TIE: usize =
    58;

const TEACHER_FORCED_MEAN_NLL_DELTA_CEILING_0_03_MEASURED_0_0128_NATS_PER_TOKEN_ON_12K_PROMPT: f64 =
    0.03;

const ACCEPT_RATE_ABS_DELTA_CEILING_0_15_MEASURED_0_075_AT_K3_OVER_48_NEW_TOKENS: f64 = 0.15;

const WITHIN_ARM_TIE_FLIP_TAIL_ALLOWANCE_2_MEASURED_47_OF_48_THE_M4_VERIFY_VS_M1_DECODE_TIE_CLASS:
    usize = 2;

fn long_prompt_ids(tok: &Tokenizer) -> Vec<u32> {
    let filler = "The measurement ladder walks request depths in fixed steps, recording the \
                  device memory peak and the wall clock for every prefill it replays. ";
    let mut text = String::from("<|im_start|>user\nSummarize the following notes.\n");
    while tok
        .encode(text.as_str(), false)
        .expect("encode probe")
        .get_ids()
        .len()
        < AB_PROMPT_TARGET_TOKENS_12000_SPANS_TWO_FULL_CHUNKS_AND_A_TAIL
    {
        for _ in 0..64 {
            text.push_str(filler);
        }
    }
    text.push_str("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    tok.encode(text.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec()
}

fn host_row(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

fn nll_of(row: &[f32], token: u32) -> f64 {
    let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let z: f64 = row.iter().map(|&v| ((v as f64) - m).exp()).sum();
    m + z.ln() - row[token as usize] as f64
}

fn prefill_last_row(
    base: &Qwen3Moe,
    device: &Device,
    prompt: &[u32],
    cache: &mut Qwen3MoeKvCache,
) -> Tensor {
    match nv_models::qwen3_5_moe::q38_prefill_chunk_tokens_from_env()
        .filter(|&c| prompt.len() > c)
    {
        Some(chunk) => base
            .prefill_chunks_dispatched_serving_last_row_logits_bounding_transients_to_one_chunk(
                prompt, chunk, cache, None,
            )
            .expect("chunked solo prefill"),
        None => {
            let toks =
                Tensor::from_vec(prompt.to_vec(), (1usize, prompt.len()), device).expect("toks");
            let pos = Tensor::from_vec(
                (0..prompt.len() as i32).collect::<Vec<i32>>(),
                prompt.len(),
                device,
            )
            .expect("pos");
            base.forward_with_cache_dispatched_rows(&toks, &pos, cache, None, Some(1))
                .expect("one-shot solo prefill")
        }
    }
}

fn teacher_forced_rows(
    base: &Qwen3Moe,
    device: &Device,
    cache: &mut Qwen3MoeKvCache,
    first_row: Vec<f32>,
    forced: &[u32],
) -> Vec<Vec<f32>> {
    let mut rows = vec![first_row];
    for &tok in forced {
        let pos = cache.current_len() as i32;
        let t = Tensor::from_vec(vec![tok], (1usize, 1usize), device).unwrap();
        let p = Tensor::from_vec(vec![pos], 1usize, device).unwrap();
        let lg = base
            .forward_with_cache_dispatched_rows(&t, &p, cache, None, Some(1))
            .expect("teacher-forced decode step");
        rows.push(host_row(&lg));
    }
    rows
}

#[test]
#[ignore = "loads the ~17 GB Qwen3.8-27B NVFP4 checkpoint; set NV_Q38_CHUNK_AB=1"]
fn chunked_prefill_holds_the_teacher_forced_band_and_preserves_mtp_acceptance_on_the_real_checkpoint(
) {
    hub_dirs::require_env_flag(
        "qwen38_chunked_prefill_real_ab",
        "NV_Q38_CHUNK_AB",
        "It loads the 27B NVFP4 checkpoint and runs four full prefills.",
    );
    let dir = hub_dirs::snapshot(
        NVFP4_REPO,
        &["config.json", "tokenizer.json", "model.safetensors", "model_mtp.safetensors"],
    )
    .expect("Qwen3.8-27B-NVFP4 snapshot with the MTP shard not found in the hub cache");

    std::env::set_var(CHUNK_ENV, "0");

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let prompt = long_prompt_ids(&tok);
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let base =
        Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device).expect("dense loader");
    let mtp = Qwen38DenseMtpHead::from_checkpoint(
        mtp_draft_dir_override_from_env().as_deref(),
        &dir,
        &base,
        &device,
    )
    .expect("MTP head");

    let max_seq = prompt.len() + 256;
    let steps = AB_TEACHER_FORCED_STEPS_64_EACH_READS_THE_PREFILLED_KV_AND_GDN_STATE;

    let t0 = std::time::Instant::now();
    let mut cache_a = base.new_kv_cache(max_seq).expect("one-shot cache");
    let row_a0 = host_row(&prefill_last_row(&base, &device, &prompt, &mut cache_a));
    let one_shot_prefill_s = t0.elapsed().as_secs_f64();
    let mut forced: Vec<u32> = vec![argmax(&row_a0)];
    let mut probe_cache = base.new_kv_cache(max_seq).expect("probe cache");
    let _ = prefill_last_row(&base, &device, &prompt, &mut probe_cache);
    for _ in 1..steps {
        let pos = probe_cache.current_len() as i32;
        let t = Tensor::from_vec(vec![*forced.last().unwrap()], (1usize, 1usize), &device)
            .unwrap();
        let p = Tensor::from_vec(vec![pos], 1usize, &device).unwrap();
        let lg = base
            .forward_with_cache_dispatched_rows(&t, &p, &mut probe_cache, None, Some(1))
            .expect("forced-sequence probe step");
        forced.push(argmax(&host_row(&lg)));
    }
    drop(probe_cache);
    let rows_a = teacher_forced_rows(&base, &device, &mut cache_a, row_a0, &forced);
    drop(cache_a);

    std::env::set_var(
        CHUNK_ENV,
        AB_CHUNK_4096_THE_RECOMMENDED_SERVE_DEFAULT.to_string(),
    );
    let t1 = std::time::Instant::now();
    let mut cache_b = base.new_kv_cache(max_seq).expect("chunked cache");
    let row_b0 = host_row(&prefill_last_row(&base, &device, &prompt, &mut cache_b));
    let chunked_prefill_s = t1.elapsed().as_secs_f64();
    let rows_b = teacher_forced_rows(&base, &device, &mut cache_b, row_b0, &forced);
    drop(cache_b);

    let mut max_abs = 0f32;
    let mut argmax_agree = 0usize;
    let mut first_mismatch: Option<usize> = None;
    let mut nll_a = 0f64;
    let mut nll_b = 0f64;
    for (i, (ra, rb)) in rows_a.iter().zip(&rows_b).take(steps).enumerate() {
        let d = ra
            .iter()
            .zip(rb)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        max_abs = max_abs.max(d);
        if argmax(ra) == argmax(rb) {
            argmax_agree += 1;
        } else if first_mismatch.is_none() {
            first_mismatch = Some(i);
        }
        nll_a += nll_of(ra, forced[i]);
        nll_b += nll_of(rb, forced[i]);
    }
    let mean_nll_delta = ((nll_a - nll_b) / steps as f64).abs();

    const MAX_NEW: usize = 48;
    let k = mtp_chain_depth_from_env();
    let eng = Qwen38MtpSelfSpecEngine::new(&base, &mtp, k).expect("engine");

    let t2 = std::time::Instant::now();
    let (spec_ids_b, stats_b) = eng
        .generate_greedy(&prompt, MAX_NEW, max_seq)
        .expect("chunked mtp stream");
    let mtp_chunked_s = t2.elapsed().as_secs_f64();
    let (ref_ids_b, _) = eng
        .generate_reference(&prompt, MAX_NEW, max_seq)
        .expect("chunked reference stream");

    std::env::set_var(CHUNK_ENV, "0");
    let t3 = std::time::Instant::now();
    let (_spec_ids_a, stats_a) = eng
        .generate_greedy(&prompt, MAX_NEW, max_seq)
        .expect("one-shot mtp stream");
    let mtp_one_shot_s = t3.elapsed().as_secs_f64();
    let accept_delta = (stats_a.accept_rate() - stats_b.accept_rate()).abs() as f64;

    eprintln!(
        "Q38 chunked-prefill A/B basis=(model=unsloth/Qwen3.8-27B-NVFP4, backend=cuda, \
         chunk={}, prompt_toks={}, k={k}, teacher_forced_steps={steps}, mtp_max_new={MAX_NEW}, \
         greedy): prefill one-shot={:.2}s chunked={:.2}s; teacher-forced max_abs_logit_diff={:.4} \
         argmax_agree={argmax_agree}/{steps} first_mismatch={first_mismatch:?} \
         nll/token one-shot={:.5} chunked={:.5} |delta|={:.6}; mtp one-shot={:.2}s \
         accept_rate={:.3} accepted={} | chunked={:.2}s accept_rate={:.3} accepted={} \
         accept_delta={:.4}; contract: {}",
        AB_CHUNK_4096_THE_RECOMMENDED_SERVE_DEFAULT,
        prompt.len(),
        one_shot_prefill_s,
        chunked_prefill_s,
        max_abs,
        nll_a / steps as f64,
        nll_b / steps as f64,
        mean_nll_delta,
        mtp_one_shot_s,
        stats_a.accept_rate(),
        stats_a.accepted,
        mtp_chunked_s,
        stats_b.accept_rate(),
        stats_b.accepted,
        accept_delta,
        CONTRACT_IS_A_TOLERANCE_PAIR_NOT_BIT_IDENTITY_BECAUSE_NVFP4_AND_FLASH_KERNEL_CONFIGS_VARY_WITH_M_THE_SAME_DRIFT_CLASS_MTP_VERIFY_INC2_PINS,
    );

    let agree_prefix = spec_ids_b
        .iter()
        .zip(&ref_ids_b)
        .take_while(|(a, b)| a == b)
        .count();
    assert!(
        agree_prefix + WITHIN_ARM_TIE_FLIP_TAIL_ALLOWANCE_2_MEASURED_47_OF_48_THE_M4_VERIFY_VS_M1_DECODE_TIE_CLASS
            >= spec_ids_b.len().min(ref_ids_b.len()),
        "within the chunked arm the speculative stream agreed with its greedy reference only \
         through {agree_prefix} of {} tokens; more than the tie-flip tail allowance diverged, \
         which is a drafter carry-row or verify seam bug, not kernel drift \
         (spec={spec_ids_b:?} ref={ref_ids_b:?})",
        spec_ids_b.len()
    );
    assert!(
        argmax_agree
            >= TEACHER_FORCED_ARGMAX_AGREEMENT_FLOOR_58_OF_64_MEASURED_63_OF_64_THE_ONE_FLIP_WAS_A_TIE,
        "teacher-forced argmax agreement {argmax_agree}/{steps} fell under the floor"
    );
    assert!(
        mean_nll_delta
            < TEACHER_FORCED_MEAN_NLL_DELTA_CEILING_0_03_MEASURED_0_0128_NATS_PER_TOKEN_ON_12K_PROMPT,
        "teacher-forced |nll delta| {mean_nll_delta} nats/token breached the ceiling"
    );
    assert!(
        stats_b.accept_rate() > 0.0,
        "zero acceptance in the chunked arm: the A/B compared a broken stream"
    );
    assert!(
        accept_delta < ACCEPT_RATE_ABS_DELTA_CEILING_0_15_MEASURED_0_075_AT_K3_OVER_48_NEW_TOKENS,
        "acceptance rate moved by {accept_delta} between one-shot and chunked prefill"
    );
}
