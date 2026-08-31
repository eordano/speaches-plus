#![cfg(feature = "cuda")]

use candle_core::{Device, Tensor};
use nv_models::gemma4_moe::{moe_stall_prof, Gemma4Moe, Gemma4MoeConfig};
use std::path::PathBuf;
use std::time::Instant;

mod ctx_timing_common;

const PREFILL_CHUNK_256_MATCHING_THE_SERVING_LOOP: usize = 256;
const DEPTHS_DEFAULT_8192_32768_MATCHING_THE_DEPTH_HARNESS_ROWS: &[usize] = &[8192, 32768];
const KV_HEADROOM_8_ROWS_KEEPS_THE_CACHE_STRICTLY_LARGER_THAN_THE_PROMPT: usize = 8;
const SEAM_GATE_DEPTH_8192_CROSSES_31_CHUNK_SEAMS: usize = 8192;

const G4MOE_HOME_HUB_REPO: &str = "models--google--gemma-4-26B-A4B-it";

fn g4moe_snapshot_dir_env_override_then_home_hub() -> PathBuf {
    if let Ok(d) = std::env::var("NV_G4MOE_SNAPSHOT") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(&home)
        .join(".cache/huggingface/hub")
        .join(G4MOE_HOME_HUB_REPO)
        .join("snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("g4moe snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file())
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no gemma-4-26B-A4B-it snapshot under HOME hub; set NV_G4MOE_SNAPSHOT")
}

fn checkpoint_label(dir: &std::path::Path) -> String {
    let rev = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into());
    format!("google/gemma-4-26B-A4B-it@{rev}")
}

fn gate_env_or_allow_skip() -> bool {
    if std::env::var("NV_G4MOE_PREFILL_TEST").ok().as_deref() == Some("1") {
        return true;
    }
    if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
        panic!(
            "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_G4MOE_PREFILL_TEST=1, or \
             NV_MODELS_ALLOW_SKIP=1 to skip on purpose"
        );
    }
    eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_G4MOE_PREFILL_TEST=1 to run");
    false
}

fn depths_from_env() -> Vec<usize> {
    match std::env::var("NV_CTX_TOKENS") {
        Ok(v) => v
            .split(',')
            .map(|s| {
                let s = s.trim();
                let (num, mult) = match s.strip_suffix(['k', 'K']) {
                    Some(n) => (n, 1024usize),
                    None => (s, 1usize),
                };
                num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
            })
            .collect(),
        Err(_) => DEPTHS_DEFAULT_8192_32768_MATCHING_THE_DEPTH_HARNESS_ROWS.to_vec(),
    }
}

fn load_model(dir: &std::path::Path, device: &Device) -> Gemma4Moe {
    let config = Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config.json");
    let loader = nv_weights::WeightLoader::open_dir(dir, device).expect("weights");
    let m = Gemma4Moe::from_loader(config, &loader, device).expect("cuda model via from_loader");
    assert_eq!(
        m.experts_nvfp4_resident_layer_count(),
        0,
        "the serving default keeps every expert stack bf16; unset NV_G4MOE_CUDA_NVFP4 for \
         destall rows so the arm matches the published serving basis"
    );
    m
}

fn synthetic_ids(depth: usize) -> Vec<u32> {
    (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect()
}

fn chunked_prefill_returning_last_logits(
    m: &Gemma4Moe,
    device: &Device,
    ids: &[u32],
    mut per_chunk_logits: Option<&mut dyn FnMut(usize, &Tensor)>,
) -> Tensor {
    let depth = ids.len();
    let mut cache = m
        .new_kv_cache(depth + KV_HEADROOM_8_ROWS_KEEPS_THE_CACHE_STRICTLY_LARGER_THAN_THE_PROMPT)
        .expect("kv cache");
    let mut last = None;
    let mut off = 0usize;
    let mut chunk_i = 0usize;
    while off < depth {
        let c = (depth - off).min(PREFILL_CHUNK_256_MATCHING_THE_SERVING_LOOP);
        let tokens_t = Tensor::from_vec(ids[off..off + c].to_vec(), (1usize, c), device).unwrap();
        let pos: Vec<i32> = (off as i32..(off + c) as i32).collect();
        let pos_t = Tensor::from_vec(pos, c, device).unwrap();
        let logits = m
            .forward_with_cache(&tokens_t, &pos_t, &mut cache)
            .unwrap_or_else(|e| panic!("prefill chunk at {off}: {e:#}"));
        if let Some(f) = per_chunk_logits.as_mut() {
            f(chunk_i, &logits);
        }
        last = Some(logits);
        off += c;
        chunk_i += 1;
    }
    last.expect("non-empty prompt")
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn logits_fnv(logits: &Tensor) -> u64 {
    let v: Vec<f32> = logits
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap_or_else(|e| panic!("logits drain for hash: {e:#}"));
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    fnv1a64(&bytes)
}

#[test]
#[ignore = "loads the real ~26B-A4B MoE checkpoint (~48 GiB bf16) on cuda; set \
            NV_G4MOE_PREFILL_TEST=1 -- warm serving-chunked (256) prefill tok/s at depths \
            from NV_CTX_TOKENS (default 8192,32768), then an untimed pass with the \
            NV_G4MOE_MOE_PROF split so the CTX-PREFILL row never carries prof overhead: \
            drain = host blocked in the two routing to_vec1 drains, dispatch_host = host \
            enqueue time of the per-expert eager loop -- the vLLM v0.28 sync-stall class"]
fn g4moe_cuda_serving_chunked_prefill_tok_s_with_moe_stall_split() {
    if !gate_env_or_allow_skip() {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let dir = g4moe_snapshot_dir_env_override_then_home_hub();
    let label = checkpoint_label(&dir);
    let m = load_model(&dir, &device);

    for depth in depths_from_env() {
        let ids = synthetic_ids(depth);

        let _ = chunked_prefill_returning_last_logits(&m, &device, &ids, None);
        device.synchronize().expect("warmup sync");

        let t0 = Instant::now();
        let _ = chunked_prefill_returning_last_logits(&m, &device, &ids, None);
        device.synchronize().expect("timed sync");
        let prefill_s = t0.elapsed().as_secs_f64();
        let chunks = depth.div_ceil(PREFILL_CHUNK_256_MATCHING_THE_SERVING_LOOP);
        println!(
            "CTX-PREFILL g4moe-cuda depth={depth} prefill_s={prefill_s:.3} prefill_tok_s={:.1} \
             chunk=256 chunks={chunks} checkpoint={label}",
            depth as f64 / prefill_s,
        );

        std::env::set_var("NV_G4MOE_MOE_PROF", "1");
        let _ = moe_stall_prof::snapshot_and_reset_route_enqueue_drain_dispatch_forwards();
        let _ = moe_stall_prof::snapshot_and_reset_expert_pretranspose_bytes();
        let tp = Instant::now();
        let _ = chunked_prefill_returning_last_logits(&m, &device, &ids, None);
        device.synchronize().expect("prof sync");
        let prof_wall_ms = tp.elapsed().as_secs_f64() * 1e3;
        std::env::remove_var("NV_G4MOE_MOE_PROF");
        let [route_enqueue_ns, drain_ns, dispatch_ns, forwards] =
            moe_stall_prof::snapshot_and_reset_route_enqueue_drain_dispatch_forwards();
        let pretranspose_bytes = moe_stall_prof::snapshot_and_reset_expert_pretranspose_bytes();
        let stall_ms = (drain_ns + dispatch_ns) as f64 / 1e6;
        println!(
            "G4MOE-MOE-STALL depth={depth} chunks={chunks} moe_forwards={forwards} \
             route_enqueue_ms={:.1} drain_ms={:.1} dispatch_host_ms={:.1} \
             prof_wall_ms={prof_wall_ms:.1} drain_plus_dispatch_share_pct={:.1} \
             expert_pretranspose_gb={:.2} expert_pretranspose_gb_per_chunk={:.3}",
            route_enqueue_ns as f64 / 1e6,
            drain_ns as f64 / 1e6,
            dispatch_ns as f64 / 1e6,
            100.0 * stall_ms / prof_wall_ms,
            pretranspose_bytes as f64 / 1e9,
            pretranspose_bytes as f64 / 1e9 / chunks as f64,
        );
    }
}

const BT_ARM_PPL_REL_BAND_2PCT_MATCHING_THE_HOUSE_ARM_VS_CONTROL_READING: f64 = 0.02;
const CHAT_WRAPPED_CTX_256_CONT_512_MATCHING_THE_HOUSE_PPL_SLICE: (usize, usize) = (256, 512);
const PPL_SANITY_CEILING_1000_A_BROKEN_ROUTE_SCORES_SHUFFLED_CLASS_THOUSANDS: f64 = 1000.0;

mod ppl_common;

fn teacher_forced_nll_from_chunk_logits_f64(
    logits: &Tensor,
    chunk_start: usize,
    ids: &[u32],
    score_start: usize,
    nll_sum: &mut f64,
    scored: &mut usize,
) {
    let dims = logits.dims().to_vec();
    let vocab = *dims.last().unwrap();
    let positions: usize = dims[..dims.len() - 1].iter().product();
    let v: Vec<f32> = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for p in 0..positions {
        let g = chunk_start + p;
        if g + 1 >= ids.len() || g + 1 < score_start {
            continue;
        }
        let row = &v[p * vocab..(p + 1) * vocab];
        let mx = row.iter().fold(f32::NEG_INFINITY, |a, x| a.max(*x));
        let mut denom = 0.0f64;
        for x in row {
            denom += ((x - mx) as f64).exp();
        }
        let target = ids[g + 1] as usize;
        *nll_sum -= ((row[target] - mx) as f64) - denom.ln();
        *scored += 1;
    }
}

#[test]
#[ignore = "loads the real ~26B-A4B MoE checkpoint on cuda; set NV_G4MOE_PREFILL_TEST=1 and \
            NV_PPL_CORPUS -- model-level teacher-forced gate for the NV_G4MOE_EXPERT_BT_GEMM \
            route: the house ppl instrument scores seq-1 steps whose m=1 gemv is \
            arm-identical, so only chunked (256) prefill reaches the changed m>=2 kernels; \
            scores the house chat-wrapped continuation slice from the prefill chunk logits \
            for both arms in one process and bands the ppl delta (top-1 agreement on \
            synthetic ids is not a gate: near-flat logits plus 8k of KV divergence flip \
            ~1/3 of argmaxes under any benign numeric change)"]
fn g4moe_cuda_btgemm_arm_chunked_prefill_teacher_forced_ppl_within_band_of_default() {
    if !gate_env_or_allow_skip() {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let dir = g4moe_snapshot_dir_env_override_then_home_hub();
    let m = load_model(&dir, &device);
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let corpus = ppl_common::corpus_text_from_nv_ppl_corpus_env_failing_loudly();
    let (ctx, cont) = CHAT_WRAPPED_CTX_256_CONT_512_MATCHING_THE_HOUSE_PPL_SLICE;
    let (ids, score_start) =
        ppl_common::chat_wrapped_continuation_ids_and_copy_start_rendered_through_the_snapshot_jinja(
            &dir, &tok, &corpus, bos, ctx, cont,
        );
    let depth = ids.len();

    let mut arm_ppl = |bt: bool| -> (f64, usize) {
        if bt {
            std::env::set_var("NV_G4MOE_EXPERT_BT_GEMM", "1");
        }
        let mut nll_sum = 0.0f64;
        let mut scored = 0usize;
        {
            let mut collect = |chunk_i: usize, logits: &Tensor| {
                teacher_forced_nll_from_chunk_logits_f64(
                    logits,
                    chunk_i * PREFILL_CHUNK_256_MATCHING_THE_SERVING_LOOP,
                    &ids,
                    score_start,
                    &mut nll_sum,
                    &mut scored,
                );
            };
            let _ = chunked_prefill_returning_last_logits(&m, &device, &ids, Some(&mut collect));
        }
        if bt {
            std::env::remove_var("NV_G4MOE_EXPERT_BT_GEMM");
        }
        ((nll_sum / scored as f64).exp(), scored)
    };

    let (ppl_default, scored_default) = arm_ppl(false);
    let (ppl_bt, scored_bt) = arm_ppl(true);
    assert_eq!(scored_default, scored_bt, "arms scored different position counts");
    let rel = (ppl_bt - ppl_default).abs() / ppl_default;
    println!(
        "G4MOE-BTGEMM-ARM-GATE depth={depth} scored={scored_default} ppl_default={ppl_default:.4} \
         ppl_bt={ppl_bt:.4} rel_delta={rel:.5} checkpoint={}",
        checkpoint_label(&dir),
    );
    assert!(
        ppl_default.is_finite() && ppl_default < PPL_SANITY_CEILING_1000_A_BROKEN_ROUTE_SCORES_SHUFFLED_CLASS_THOUSANDS,
        "default-arm corpus ppl {ppl_default} is not plausible; the instrument is broken, \
         not the arm"
    );
    assert!(
        rel <= BT_ARM_PPL_REL_BAND_2PCT_MATCHING_THE_HOUSE_ARM_VS_CONTROL_READING,
        "bt arm ppl {ppl_bt:.4} vs default {ppl_default:.4}: relative delta {rel:.5} beyond \
         the 2% band -- the transposed-B expert route moved real-checkpoint prefill quality"
    );
}

#[test]
#[ignore = "loads the real ~26B-A4B MoE checkpoint on cuda; set NV_G4MOE_PREFILL_TEST=1 -- \
            bitwise gate for destall orchestration: hashes every chunk's full-vocab logits \
            over an 8192-token serving-chunked prefill on two consecutive default-path runs \
            and requires identical FNV64 per chunk; the printed per-chunk hashes also pin \
            the default path across builds (the within-noise overlap arm was removed per \
            knobs-sparingly after measuring +0.3% inside run noise)"]
fn g4moe_cuda_chunked_prefill_8192_logits_bitwise_across_seam_two_default_runs() {
    if !gate_env_or_allow_skip() {
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let dir = g4moe_snapshot_dir_env_override_then_home_hub();
    let m = load_model(&dir, &device);
    let ids = synthetic_ids(SEAM_GATE_DEPTH_8192_CROSSES_31_CHUNK_SEAMS);

    let mut default_hashes: Vec<u64> = Vec::new();
    {
        let mut collect = |_i: usize, logits: &Tensor| default_hashes.push(logits_fnv(logits));
        let _ = chunked_prefill_returning_last_logits(&m, &device, &ids, Some(&mut collect));
    }
    for (i, h) in default_hashes.iter().enumerate() {
        println!("G4MOE-SEAM-HASH arm=default chunk={i} fnv={h:016x}");
    }
    let mut all = Vec::with_capacity(default_hashes.len() * 8);
    for h in &default_hashes {
        all.extend_from_slice(&h.to_le_bytes());
    }
    println!(
        "G4MOE-SEAM-BITWISE depth={} chunks={} combined_fnv={:016x}",
        ids.len(),
        default_hashes.len(),
        fnv1a64(&all)
    );

    let mut rerun_hashes: Vec<u64> = Vec::new();
    {
        let mut collect = |_i: usize, logits: &Tensor| rerun_hashes.push(logits_fnv(logits));
        let _ = chunked_prefill_returning_last_logits(&m, &device, &ids, Some(&mut collect));
    }

    assert_eq!(
        default_hashes.len(),
        rerun_hashes.len(),
        "runs produced different chunk counts"
    );
    for i in 0..default_hashes.len() {
        assert_eq!(
            default_hashes[i], rerun_hashes[i],
            "chunk {i}: consecutive default-path runs diverged bitwise -- the destalled \
             orchestration must be deterministic"
        );
    }
    println!(
        "G4MOE-SEAM-BITWISE-GATE chunks={} run_vs_rerun=identical",
        default_hashes.len()
    );
}
