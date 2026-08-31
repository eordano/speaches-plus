#![cfg(feature = "cuda")]

mod ctx_timing_common;

use candle_core::{DType, Device, Tensor};
use nv_models::qwen3::{Qwen3, Qwen3Config};
use nv_weights::{QuantScheme, QuantizationConfig, WeightLoader};
use std::path::PathBuf;

const TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN: usize = 64;
const WARMUP_DECODE_STEPS_8_SO_THE_FIRST_ALLOCS_DONT_COUNT: usize = 8;
const PREFILL_CHUNK_512_MATCHES_THE_PPL_QWEN3_BLOCK: usize = 512;
const TOKEN_ID_BASE_2000_MOD_30000_STAYS_INSIDE_EVERY_QWEN_VOCAB: u32 = 2000;

fn ctx_tokens_from_env_default_256_4k_because_qwen3_embedding_max_pos_is_32768() -> Vec<usize> {
    match std::env::var("NV_CTX_TOKENS") {
        Ok(v) => v
            .split(',')
            .map(|s| {
                let s = s.trim();
                let (num, mult) = match s.strip_suffix('k') {
                    Some(n) => (n, 1024usize),
                    None => (s, 1usize),
                };
                num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
            })
            .collect(),
        Err(_) => vec![256, 4096],
    }
}

fn qwen3_dense_snapshot_dir_env_override_then_hub_caches() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN3_DIR") {
        return PathBuf::from(d);
    }
    let mut bases = Vec::new();
    if let Ok(hub) = std::env::var("HF_HUB_CACHE") {
        bases.push(PathBuf::from(hub).join("models--Qwen--Qwen3-Embedding-0.6B/snapshots"));
    }
    if let Ok(home) = std::env::var("HOME") {
        bases.push(
            PathBuf::from(home)
                .join(".cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots"),
        );
    }
    for base in &bases {
        if let Ok(entries) = std::fs::read_dir(base) {
            if let Some(snap) = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.join("config.json").is_file() && p.join("tokenizer.json").is_file())
            {
                return snap;
            }
        }
    }
    panic!("no complete Qwen3-Embedding-0.6B snapshot under {bases:?}; set NV_QWEN3_DIR")
}

fn checkpoint_label(dir: &std::path::Path) -> String {
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
#[ignore = "loads the real 0.6B; set NV_QWEN3_CTX_TEST=1; NV_QWEN3_WEIGHTS=bf16|nvfp4 selects the arm; NV_CTX_TOKENS overrides the 256,4096 depth ladder"]
fn qwen3_cuda_decode_ms_per_token_vs_context_depth_arm_from_nv_qwen3_weights() {
    if std::env::var("NV_QWEN3_CTX_TEST").ok().as_deref() != Some("1") {
        panic!("set NV_QWEN3_CTX_TEST=1 to run (this probe must never silently skip)");
    }
    let arm = match std::env::var("NV_QWEN3_WEIGHTS").ok().as_deref() {
        Some("bf16") => "bf16",
        None | Some("nvfp4") => "nvfp4",
        Some(other) => panic!("NV_QWEN3_WEIGHTS={other} is not an arm (bf16|nvfp4)"),
    };
    let dir = qwen3_dense_snapshot_dir_env_override_then_hub_caches();
    let device = Device::new_cuda(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3Config::from_hf_json_str(&raw_cfg).expect("parse config");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = if arm == "bf16" {
        Qwen3::from_loader(cfg, &weights, &device).expect("bf16 model")
    } else {
        let qcfg = QuantizationConfig {
            scheme: QuantScheme::Nvfp4,
            ignored_modules: Vec::new(),
            group_size: None,
        };
        Qwen3::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("nvfp4 model")
    };
    drop(weights);
    let label = checkpoint_label(&dir);

    let depths = ctx_tokens_from_env_default_256_4k_because_qwen3_embedding_max_pos_is_32768();
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
        let prime_start = std::time::Instant::now();
        let mut pos = 0usize;
        while pos < depth {
            let n = PREFILL_CHUNK_512_MATCHES_THE_PPL_QWEN3_BLOCK.min(depth - pos);
            let ids: Vec<u32> = (0..n)
                .map(|i| {
                    TOKEN_ID_BASE_2000_MOD_30000_STAYS_INSIDE_EVERY_QWEN_VOCAB
                        + ((pos + i) as u32 % 30000)
                })
                .collect();
            let tokens = Tensor::from_vec(ids, (1usize, n), &device).expect("tokens");
            let positions = Tensor::from_vec(
                (pos as i32..(pos + n) as i32).collect::<Vec<_>>(),
                n,
                &device,
            )
            .expect("positions");
            model
                .forward(&tokens, &positions, &mut cache)
                .unwrap_or_else(|e| panic!("prefill chunk at pos {pos}: {e:#}"));
            pos += n;
        }
        let prime_s = prime_start.elapsed().as_secs_f64();

        let mut step_ms: Vec<f64> = Vec::new();
        for i in 0..WARMUP_DECODE_STEPS_8_SO_THE_FIRST_ALLOCS_DONT_COUNT
            + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
        {
            let p = depth + i;
            let tokens = Tensor::from_vec(
                vec![TOKEN_ID_BASE_2000_MOD_30000_STAYS_INSIDE_EVERY_QWEN_VOCAB + (p as u32 % 30000)],
                (1usize, 1usize),
                &device,
            )
            .expect("token");
            let positions = Tensor::from_vec(vec![p as i32], 1usize, &device).expect("position");
            let t0 = std::time::Instant::now();
            let logits = model
                .forward(&tokens, &positions, &mut cache)
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
            "CTX-SCALING qwen3-cuda-{arm} checkpoint={label} depth={depth} median_ms_tok={median:.3} tok_s={:.1} prime_s={prime_s:.1} steps={}",
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
                ("weights", arm.to_string()),
                ("prime_s", format!("{prime_s:.1}")),
            ],
        );
    }
}
