#![cfg(feature = "wgpu")]

use nv_models::qwen3_5_dense_wgpu::{Qwen3_5DenseConfig, Qwen3_5DenseWgpu};
use nv_weights::WeightLoader;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

const DIR_ENV: &str = "NV_MLXQ_DIR";
const GATE_ENV: &str = "NV_Q3D_MLX_TEST";

fn mlx_dir() -> PathBuf {
    let d = std::env::var(DIR_ENV).unwrap_or_default();
    assert!(
        !d.is_empty(),
        "set {DIR_ENV} to an mlx-community Qwen3.8-27B-4bit snapshot; this gated suite \
         refuses to vacuously pass"
    );
    let p = PathBuf::from(&d);
    assert!(p.is_dir(), "{DIR_ENV}={d} is not a directory");
    p
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .map(|v| {
            v.parse::<usize>()
                .unwrap_or_else(|_| panic!("{key}={v} is not a usize"))
        })
        .unwrap_or(default)
}

#[test]
#[ignore = "boots the ~15 GB mlx-community Qwen3.8-27B-4bit affine checkpoint on wgpu, prefills a \
            chat prompt via the per-token step path and greedily decodes; set NV_Q3D_MLX_TEST=1 \
            and NV_MLXQ_DIR (NV_MLXQ_LAYERS/NV_MLXQ_NEW/NV_MLXQ_PROMPT optional)"]
fn mlx_affine_27b_boots_prefills_via_step_path_and_greedily_decodes() {
    assert!(
        std::env::var(GATE_ENV).as_deref() == Ok("1"),
        "set {GATE_ENV}=1 to run the real-weights mlx-affine smoke (it must never silently skip)"
    );
    let dir = mlx_dir();
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let mut cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw)
        .expect("mlx Qwen3.8-27B-4bit config.json must parse as a Qwen3_5 dense config");
    let full_layers = cfg.num_hidden_layers;
    let truncated = std::env::var("NV_MLXQ_LAYERS").ok().map(|v| {
        let l: usize = v
            .parse()
            .unwrap_or_else(|_| panic!("NV_MLXQ_LAYERS={v} is not a usize"));
        assert!(l >= 1 && l <= full_layers, "NV_MLXQ_LAYERS={l} out of 1..={full_layers}");
        cfg.layer_types.truncate(l);
        cfg.num_hidden_layers = l;
        l
    });
    let truncated = truncated.filter(|l| *l != full_layers);
    let vocab = cfg.vocab_size;

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let q = std::env::var("NV_MLXQ_PROMPT")
        .unwrap_or_else(|_| "Explain, in a few sentences, why the sky appears blue.".into());
    let text =
        format!("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let prompt = tok
        .encode(text.as_str(), false)
        .expect("encode chat prompt")
        .get_ids()
        .to_vec();
    let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tok.token_to_id(s))
        .collect();
    assert!(!stop.is_empty(), "tokenizer resolves no stop tokens");

    let new_tokens = env_usize("NV_MLXQ_NEW", 32);
    assert!(new_tokens >= 16, "NV_MLXQ_NEW={new_tokens} < 16 gates nothing");
    let max_seq = prompt.len() + new_tokens + 8;

    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let t0 = Instant::now();
    let loader = WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("open weights");
    let mut m = Qwen3_5DenseWgpu::from_loader(cfg, &loader, max_seq)
        .expect("Qwen3_5DenseWgpu::from_loader on the mlx affine checkpoint");
    drop(loader);
    eprintln!("boot: {:.1}s, {} decode passes", t0.elapsed().as_secs_f64(), m.pass_count());

    let chunked_opt_in = std::env::var("NV_Q3D_AFFINE_CHUNKED_PF_V2_MEASURES_26TOK_0_97S_1613TOK_27_8MS_VS_STEPPED_1_10S_38_5MS")
        .is_ok_and(|v| v == "1");
    assert_eq!(
        m.prefill_chunk_len() > 0,
        chunked_opt_in,
        "affine prefill DEFAULTS to the stepped path that every published number was measured \
         on; the faster v2 chunked graph must appear exactly when its opt-in env is set"
    );

    let t1 = Instant::now();
    let first = m.prefill(&prompt).expect("prefill the chat prompt via the step path");
    let prefill_s = t1.elapsed().as_secs_f64();
    let mut generated: Vec<u32> = vec![first];
    let mut hit_stop_at: Option<usize> = None;
    let t2 = Instant::now();
    let mut prev = first;
    for i in 1..new_tokens {
        if stop.contains(&prev) {
            hit_stop_at = Some(i);
            break;
        }
        let next = m.decode_step(prev).expect("greedy decode step");
        generated.push(next);
        prev = next;
    }
    let decode_s = t2.elapsed().as_secs_f64();
    let stepped = generated.len();
    for (i, t) in generated.iter().enumerate() {
        assert!(
            (*t as usize) < vocab,
            "generated token {t} at step {i} is outside the vocab width {vocab}"
        );
    }
    let out_text = tok.decode(&generated, true).unwrap_or_default();
    println!(
        "Q3D-MLX-AFFINE-DECODE checkpoint=mlx-community/Qwen3.8-27B-4bit backend=wgpu batch=1 \
         layers={}{} prompt_tokens={} prefill_s={prefill_s:.2} generated_tokens={stepped} \
         ms_per_token={:.2} stop_at={hit_stop_at:?}",
        m.config().num_hidden_layers,
        if truncated.is_some() { "(TRUNCATED)" } else { "" },
        prompt.len(),
        decode_s * 1e3 / stepped.max(1) as f64
    );
    println!("Q3D-MLX-AFFINE-TEXT :: {out_text:?}");

    if truncated.is_some() {
        eprintln!(
            "NV_MLXQ_LAYERS truncated the trunk: boot/iteration arm only; the >=16-token and \
             non-degenerate assertions are skipped"
        );
        assert!(stepped >= 2, "even a truncated trunk must emit more than the prefill token");
        return;
    }
    assert!(
        stepped >= 16,
        "full-depth greedy decode produced only {stepped} tokens before stop {hit_stop_at:?}; \
         the smoke gate demands at least 16"
    );
    assert!(
        !out_text.trim().is_empty(),
        "decoded text is empty; print above is the human coherence checkpoint"
    );
    let distinct: std::collections::BTreeSet<u32> = generated.iter().copied().collect();
    assert!(
        distinct.len() >= 8,
        "greedy decode is degenerate: only {} distinct tokens across {stepped} steps ({out_text:?})",
        distinct.len()
    );
}
