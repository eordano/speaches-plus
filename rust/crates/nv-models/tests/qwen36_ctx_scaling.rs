#![cfg(feature = "cuda")]

mod common;
mod ctx_timing_common;
use common::ctx_tokens_from_env_default_256_8k_168k;
use candle_core::{DType, Device, Tensor};
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use common::qwen36_nvfp4_snapshot_dir_env_override_then_home_hub;

const PREFILL_CHUNK_512_QWEN3MOE_KV_IS_FLAT_WITH_NO_RING_CAP_ONLY_MAX_SEQ_BOUNDS_WRITES_512_MATCHES_THE_PROVEN_PPL_BLOCK:
    usize = 512;
const TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN: usize = 64;
const WARMUP_DECODE_STEPS_8_SO_THE_FIRST_ALLOCS_DONT_COUNT: usize = 8;

fn checkpoint_label_repo_at_short_rev_so_runs_jsonl_rows_carry_their_basis(
    dir: &std::path::Path,
) -> String {
    let snap = dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let repo = dir
        .ancestors()
        .find_map(|a| {
            let name = a.file_name()?.to_string_lossy().to_string();
            name.starts_with("models--").then_some(name)
        })
        .unwrap_or_else(|| "unknown-repo".into());
    format!("{repo}@{}", &snap[..snap.len().min(8)])
}

#[test]
#[ignore = "loads the ~22 GB Qwen3.6-35B NVFP4; set NV_QWEN36_CTX_TEST=1 -- decode ms/token vs KV depth (max_pos 262144 fits the full 256/8k/168k ladder), chunked-prefill primed, eager cuda path; timing only, the #95 free-running degeneracy is deliberately not judged; run ONE depth per process via NV_CTX_TOKENS (the #96 residual race)"]
fn qwen36_cuda_decode_ms_per_token_vs_context_depth_eager_path_chunked_prefill_primed() {
    if std::env::var("NV_QWEN36_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_QWEN36_CTX_TEST != 1");
        return;
    }
    let dir = qwen36_nvfp4_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);

    let depths = ctx_tokens_from_env_default_256_8k_168k();
    let max_depth = depths.iter().copied().max().unwrap();
    let max_pos = model.config().max_position_embeddings;
    assert!(
        max_depth + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN + 16 < max_pos,
        "requested depth {max_depth} exceeds max_position_embeddings {max_pos}"
    );

    for &depth in &depths {
        let mut cache = model
            .new_kv_cache(depth + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN + 16)
            .expect("kv cache");
        let chunk = PREFILL_CHUNK_512_QWEN3MOE_KV_IS_FLAT_WITH_NO_RING_CAP_ONLY_MAX_SEQ_BOUNDS_WRITES_512_MATCHES_THE_PROVEN_PPL_BLOCK;
        let prime_start = std::time::Instant::now();
        let mut pos = 0usize;
        while pos < depth {
            let n = chunk.min(depth - pos);
            let ids: Vec<u32> = (0..n).map(|i| 2000 + ((pos + i) as u32 % 30000)).collect();
            let tokens = Tensor::from_vec(ids, (1usize, n), &device).expect("tokens");
            let positions = Tensor::from_vec(
                (pos as i32..(pos + n) as i32).collect::<Vec<_>>(),
                n,
                &device,
            )
            .expect("positions");
            model
                .forward_with_cache(&tokens, &positions, &mut cache)
                .unwrap_or_else(|e| panic!("prefill chunk at pos {pos}: {e:#}"));
            pos += n;
        }
        let prime_s = prime_start.elapsed().as_secs_f64();

        let mut step_ms: Vec<f64> = Vec::new();
        for i in 0..WARMUP_DECODE_STEPS_8_SO_THE_FIRST_ALLOCS_DONT_COUNT
            + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
        {
            let p = depth + i;
            let tokens =
                Tensor::from_vec(vec![2000u32 + (p as u32 % 30000)], (1usize, 1usize), &device)
                    .expect("token");
            let positions = Tensor::from_vec(vec![p as i32], 1usize, &device).expect("position");
            let t0 = std::time::Instant::now();
            let logits = model
                .forward_with_cache(&tokens, &positions, &mut cache)
                .unwrap_or_else(|e| panic!("decode at depth {p}: {e:#}"));
            logits
                .to_dtype(DType::F32)
                .expect("f32")
                .flatten_all()
                .expect("flatten")
                .to_vec1::<f32>()
                .expect("sync to host");
            if i >= WARMUP_DECODE_STEPS_8_SO_THE_FIRST_ALLOCS_DONT_COUNT {
                step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = step_ms[step_ms.len() / 2];
        eprintln!(
            "CTX-SCALING qwen36-cuda-eager checkpoint={} depth={depth} median_ms_tok={median:.3} tok_s={:.1} prime_s={prime_s:.1} steps={}",
            checkpoint_label_repo_at_short_rev_so_runs_jsonl_rows_carry_their_basis(&dir),
            1000.0 / median,
            step_ms.len()
        );
        ctx_timing_common::emit_nv_measure_v1_alongside_the_legacy_line_which_stays_for_one_release(
            ctx_timing_common::NvMeasureCtxLine {
                instrument: "ctx-scaling",
                model_dir: &dir,
                backend: "cuda",
                device: ctx_timing_common::cuda_device_name_for_the_nv_measure_device_basis_field(),
                tokens_primed_kv_depth_of_this_run_never_a_capacity: depth,
                steps: step_ms.len(),
                warmup: WARMUP_DECODE_STEPS_8_SO_THE_FIRST_ALLOCS_DONT_COUNT,
                value: median,
                unit: "ms/tok",
                basis: "eager_cuda_decode_step_chunked_prefill_primed",
            },
            &[
                ("arm", ctx_timing_common::arm_label_from_the_resolved_value_of_every_gate_accessor_because_a_presence_label_lies_after_a_default_flip(
                    &ctx_timing_common::Q36_Q38_DECODE_NUMERICS_GATES_EVERY_ONE_APPEARS_RESOLVED_IN_THE_ARM_LABEL,
                )),
                ("prime_s", format!("{prime_s:.1}")),
            ],
        );
    }
}
