#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;

mod ctx_timing_common;
mod hub_snapshot;

const XPPL_N_CTX_512_THE_LLAMACPP_DEFAULT_PERPLEXITY_CHUNK: usize = 512;
const XPPL_FIRST_256_LLAMACPP_SCORES_ONLY_THE_SECOND_HALF_OF_EACH_CHUNK: usize = 256;
const XPPL_TARGETS_PER_CHUNK_255_IS_N_CTX_MINUS_FIRST_MINUS_1: usize =
    XPPL_N_CTX_512_THE_LLAMACPP_DEFAULT_PERPLEXITY_CHUNK
        - XPPL_FIRST_256_LLAMACPP_SCORES_ONLY_THE_SECOND_HALF_OF_EACH_CHUNK
        - 1;
const XPPL_DEFAULT_TEXT_FILE: &str = "/tmp/nv-corpus/wikitext-2-raw/wiki.test.raw";
const XPPL_SANE_PPL_CEILING_100_A_BROKEN_ARM_READS_IN_THE_THOUSANDS: f64 = 100.0;

fn require_gate() {
    if std::env::var("NV_Q38_XPPL").as_deref() != Ok("1") {
        panic!("set NV_Q38_XPPL=1 to run the cross-engine same-method wikitext-2 scorer");
    }
}

fn snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN38_DIR") {
        return PathBuf::from(d);
    }
    hub_snapshot::snapshot_of(
        "unsloth/Qwen3.8-27B-NVFP4",
        &["config.json", "*.safetensors"],
    )
    .expect("no hydrated unsloth/Qwen3.8-27B-NVFP4 snapshot; set NV_QWEN38_DIR")
}

fn text_file() -> PathBuf {
    PathBuf::from(
        std::env::var("NV_XPPL_FILE").unwrap_or_else(|_| XPPL_DEFAULT_TEXT_FILE.to_string()),
    )
}

fn chunk_cap() -> usize {
    std::env::var("NV_XPPL_CHUNKS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(usize::MAX)
}

#[test]
#[ignore = "loads the ~54 GB Qwen3.8-27B NVFP4 dense arm and teacher-forces wikitext-2-raw \
            wiki.test.raw under the exact llama-perplexity protocol (n_ctx=512 chunks, fresh \
            context per chunk, NLL over positions 256..511 only, no BOS because the qwen \
            tokenizer has add_bos=false, ppl=exp(mean NLL)); set NV_Q38_XPPL=1; NV_XPPL_FILE \
            overrides the text, NV_XPPL_CHUNKS caps the chunk count for smoke runs"]
fn qwen38_nvfp4_wikitext2_ppl_under_the_llamacpp_512_chunk_protocol() {
    require_gate();
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    std::env::set_var("NV_Q38_GDN_CHUNK_PREFILL", "1");

    let dir = snapshot_dir();
    let file = text_file();
    let text = std::fs::read_to_string(&file)
        .unwrap_or_else(|e| panic!("read {}: {e}", file.display()));

    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .expect("tokenizer.json ships with the checkpoint");
    let ids: Vec<u32> = tokenizer
        .encode(text.as_str(), false)
        .expect("encode wiki.test.raw")
        .get_ids()
        .to_vec();

    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("quant config");
    let tok_vocab_with_added = tokenizer.get_vocab_size(true);
    assert!(
        tok_vocab_with_added <= cfg.vocab_size,
        "tokenizer vocab {tok_vocab_with_added} exceeds model vocab {}; wrong tokenizer for \
         this checkpoint",
        cfg.vocab_size
    );
    let max_id = *ids.iter().max().expect("nonempty corpus") as usize;
    assert!(
        max_id < cfg.vocab_size,
        "token id {max_id} out of model vocab {}",
        cfg.vocab_size
    );

    let n_ctx = XPPL_N_CTX_512_THE_LLAMACPP_DEFAULT_PERPLEXITY_CHUNK;
    let first = XPPL_FIRST_256_LLAMACPP_SCORES_ONLY_THE_SECOND_HALF_OF_EACH_CHUNK;
    let targets = XPPL_TARGETS_PER_CHUNK_255_IS_N_CTX_MINUS_FIRST_MINUS_1;
    let n_chunk_full = ids.len() / n_ctx;
    let n_chunk = n_chunk_full.min(chunk_cap());
    assert!(
        n_chunk >= 1,
        "corpus tokenizes to {} tokens, need at least {n_ctx}",
        ids.len()
    );

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg.clone(), &weights, &qcfg, &device)
        .expect("build Qwen3.8-27B dense arm as shipped");
    drop(weights);
    let vocab = cfg.vocab_size;

    let mut nll = 0f64;
    let mut nll2 = 0f64;
    let mut count = 0usize;
    let positions_v: Vec<i32> = (0..n_ctx as i32).collect();
    for chunk in 0..n_chunk {
        let start = chunk * n_ctx;
        let slice = &ids[start..start + n_ctx];
        let mut cache = model.new_kv_cache(n_ctx).expect("kv cache");
        let tokens =
            Tensor::from_vec(slice.to_vec(), (1usize, n_ctx), &device).expect("tokens");
        let positions =
            Tensor::from_vec(positions_v.clone(), n_ctx, &device).expect("positions");
        let logits = model
            .forward_with_cache(&tokens, &positions, &mut cache)
            .unwrap_or_else(|e| panic!("chunk {chunk}: {e:#}"));
        let scored: Vec<f32> = logits
            .narrow(1, first, targets)
            .expect("narrow to the scored second half")
            .to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1()
            .expect("to host");
        for row in 0..targets {
            let r = &scored[row * vocab..(row + 1) * vocab];
            let target = slice[first + row + 1] as usize;
            let m = r.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            assert!(
                m.is_finite(),
                "non-finite logits in chunk {chunk} row {row}"
            );
            let lse = m as f64
                + r.iter()
                    .map(|&v| ((v - m) as f64).exp())
                    .sum::<f64>()
                    .ln();
            let v = lse - r[target] as f64;
            nll += v;
            nll2 += v * v;
            count += 1;
        }
        if (chunk + 1) % 32 == 0 || chunk + 1 == n_chunk {
            eprintln!(
                "[xppl] chunk {}/{} running_ppl={:.4}",
                chunk + 1,
                n_chunk,
                (nll / count as f64).exp()
            );
        }
    }

    let mean = nll / count as f64;
    let ppl = mean.exp();
    let var = (nll2 / count as f64 - mean * mean).max(0.0);
    let sem = (var / (count as f64 - 1.0)).sqrt();
    eprintln!(
        "[xppl] FINAL ppl={ppl:.4} +/- {:.5} basis=(engine=speaches-plus nvfp4 dense arm as \
         shipped, fp8-resident modules, NV_Q38_GDN_CHUNK_PREFILL=1, checkpoint={}, file={}, \
         tokens={}, n_ctx={n_ctx}, first={first}, targets_per_chunk={targets}, \
         n_chunk={n_chunk} of {n_chunk_full}, count={count}, add_bos=false, \
         tokenizer_vocab_with_added={tok_vocab_with_added}, model_vocab={vocab})",
        sem * ppl,
        dir.display(),
        file.display(),
        ids.len()
    );
    assert!(
        ppl.is_finite() && ppl > 1.0 && ppl < XPPL_SANE_PPL_CEILING_100_A_BROKEN_ARM_READS_IN_THE_THOUSANDS,
        "wikitext-2 ppl {ppl} outside the sane band; the arm is broken, not merely worse"
    );
}
