#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::io::Write;
use std::path::PathBuf;

mod ctx_timing_common;
mod hub_snapshot;

const Q38QUAL_BLOCK_512_IS_THE_PROVEN_QWEN35_DENSE_PREFILL_BLOCK: usize = 512;
const Q38QUAL_DECODE_HEADROOM_SLOTS_BEYOND_THE_SCORED_OR_GENERATED_TAIL: usize = 16;
const Q38QUAL_DEFAULT_NCTX_16384_THE_STANDING_PRIORITY_DEPTH: usize = 16384;
const Q38QUAL_SANE_PPL_CEILING_100_A_BROKEN_ARM_READS_IN_THE_THOUSANDS: f64 = 100.0;

fn require_gate() {
    if std::env::var("NV_Q38QUAL").as_deref() != Ok("1") {
        panic!("set NV_Q38QUAL=1 to run the cross-engine quality lane");
    }
}

fn snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN38_DIR") {
        return PathBuf::from(d);
    }
    hub_snapshot::snapshot_of("unsloth/Qwen3.8-27B-NVFP4", &["config.json", "*.safetensors"])
        .expect("no hydrated unsloth/Qwen3.8-27B-NVFP4 snapshot; set NV_QWEN38_DIR")
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

fn print_live_arm_stack() {
    let mut armed: Vec<String> = std::env::vars()
        .filter(|(k, _)| k.starts_with("NV_"))
        .map(|(k, v)| format!("{k}={v}"))
        .collect();
    armed.sort();
    eprintln!("[q38qual] live_env_stack {}", armed.join(" "));
}

fn build_model(dir: &PathBuf, device: &Device) -> (Qwen3Moe, Qwen3_5DenseConfig) {
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("quant config");
    let weights = WeightLoader::open_dir(dir, device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg.clone(), &weights, &qcfg, device)
        .expect("build Qwen3.8-27B dense arm as shipped");
    drop(weights);
    (model, cfg)
}

fn corpus_ids(dir: &PathBuf) -> Vec<u32> {
    let file = PathBuf::from(
        std::env::var("NV_Q38QUAL_FILE")
            .unwrap_or_else(|_| "/home/user/tmp/wf/q38/xppl/wikitext-2-raw/wiki.test.raw".into()),
    );
    let text =
        std::fs::read_to_string(&file).unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
        .expect("tokenizer.json ships with the checkpoint");
    let ids: Vec<u32> = tokenizer
        .encode(text.as_str(), false)
        .expect("encode corpus")
        .get_ids()
        .to_vec();
    eprintln!(
        "[q38qual] corpus file={} bytes={} tokens={}",
        file.display(),
        text.len(),
        ids.len()
    );
    ids
}

fn nll_and_top1_of_rows(rows: &[f32], vocab: usize, targets: &[u32]) -> Vec<(f64, bool)> {
    (0..targets.len())
        .map(|i| {
            let r = &rows[i * vocab..(i + 1) * vocab];
            let m = r.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            assert!(m.is_finite(), "non-finite logits row {i}");
            let argmax = r
                .iter()
                .position(|&v| v == m)
                .expect("the row maximum is an element of the row");
            let lse =
                m as f64 + r.iter().map(|&v| ((v - m) as f64).exp()).sum::<f64>().ln();
            let t = targets[i] as usize;
            (lse - r[t] as f64, argmax == t)
        })
        .collect()
}

#[test]
#[ignore = "teacher-forces the canonical wikitext-2 corpus through the Qwen3.8-27B NVFP4 dense \
            arm under llama-perplexity's own protocol at an arbitrary n_ctx (NV_Q38QUAL_NCTX): \
            contiguous n_ctx chunks, fresh cache per chunk, the first half prefilled and \
            discarded, NLL accumulated over the second half only, ppl=exp(mean NLL). The chunk \
            is walked in 512-token blocks so the logits transient is bounded no matter how deep \
            n_ctx is. Per-position NLL is dumped to NV_Q38QUAL_NLL_OUT as little-endian f64 so a \
            rival engine's dump can be paired position-by-position. Set NV_Q38QUAL=1"]
fn q38qual_wikitext_ppl_at_env_nctx_under_the_llamacpp_second_half_protocol() {
    require_gate();
    let _one_gpu = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    std::env::set_var("NV_Q38_GDN_CHUNK_PREFILL", "1");
    print_live_arm_stack();

    let dir = snapshot_dir();
    let ids = corpus_ids(&dir);
    let n_ctx = env_usize("NV_Q38QUAL_NCTX", Q38QUAL_DEFAULT_NCTX_16384_THE_STANDING_PRIORITY_DEPTH);
    let first = n_ctx / 2;
    let targets_per_chunk = n_ctx - first - 1;
    let n_chunk_full = ids.len() / n_ctx;
    let n_chunk = n_chunk_full.min(env_usize("NV_Q38QUAL_CHUNKS", usize::MAX));
    assert!(n_chunk >= 1, "corpus has {} tokens, n_ctx {n_ctx}", ids.len());

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let (model, cfg) = build_model(&dir, &device);
    let vocab = cfg.vocab_size;
    let block = env_usize(
        "NV_Q38QUAL_BLOCK_IS_THE_REASSOCIATION_NULL_KNOB_SAME_FUNCTION_DIFFERENT_PREFILL_TILING",
        Q38QUAL_BLOCK_512_IS_THE_PROVEN_QWEN35_DENSE_PREFILL_BLOCK,
    );
    assert!(
        n_ctx % block == 0 && block >= 1,
        "block {block} must divide n_ctx {n_ctx} or the last block is ragged and the null stops \
         being the same arithmetic on the same grid"
    );

    let mut per_position: Vec<(f64, bool)> = Vec::with_capacity(n_chunk * targets_per_chunk);
    let t0 = std::time::Instant::now();
    for chunk in 0..n_chunk {
        let base = chunk * n_ctx;
        let slice = &ids[base..base + n_ctx];
        let mut cache = model.new_kv_cache(n_ctx).expect("kv cache");
        let mut pos = 0usize;
        let mut scored_this_chunk = 0usize;
        while pos < n_ctx {
            let n = block.min(n_ctx - pos);
            let tokens = Tensor::from_vec(slice[pos..pos + n].to_vec(), (1usize, n), &device)
                .expect("tokens");
            let positions_v: Vec<i32> = (pos as i32..(pos + n) as i32).collect();
            let positions = Tensor::from_vec(positions_v, n, &device).expect("positions");
            let logits = model
                .forward_with_cache(&tokens, &positions, &mut cache)
                .unwrap_or_else(|e| panic!("chunk {chunk} block at {pos}: {e:#}"));
            if pos + n > first {
                let lo = first.max(pos);
                let hi = (pos + n).min(n_ctx - 1);
                if hi > lo {
                    let rows: Vec<f32> = logits
                        .narrow(1, lo - pos, hi - lo)
                        .expect("narrow to scored rows")
                        .to_dtype(DType::F32)
                        .expect("f32")
                        .flatten_all()
                        .expect("flatten")
                        .to_vec1()
                        .expect("to host");
                    let tgt: Vec<u32> = (lo..hi).map(|p| slice[p + 1]).collect();
                    per_position.extend(nll_and_top1_of_rows(&rows, vocab, &tgt));
                    scored_this_chunk += hi - lo;
                }
            }
            pos += n;
        }
        assert_eq!(
            scored_this_chunk, targets_per_chunk,
            "chunk {chunk} scored {scored_this_chunk} of {targets_per_chunk} positions; a chunk \
             that does not cover the whole second half is a different measurement"
        );
        let tail = &per_position[per_position.len() - targets_per_chunk..];
        let m = tail.iter().map(|x| x.0).sum::<f64>() / targets_per_chunk as f64;
        eprintln!(
            "[q38qual] chunk {}/{} nll={:.9} ppl={:.4} acc={:.4} elapsed_s={:.1}",
            chunk + 1,
            n_chunk,
            m,
            m.exp(),
            tail.iter().filter(|x| x.1).count() as f64 / targets_per_chunk as f64,
            t0.elapsed().as_secs_f64()
        );
    }

    let count = per_position.len();
    let mean = per_position.iter().map(|x| x.0).sum::<f64>() / count as f64;
    let ppl = mean.exp();
    let var = per_position
        .iter()
        .map(|x| (x.0 - mean) * (x.0 - mean))
        .sum::<f64>()
        / count as f64;
    let sem = (var / (count as f64 - 1.0)).sqrt();
    let acc = per_position.iter().filter(|x| x.1).count() as f64 / count as f64;
    if let Ok(out) = std::env::var("NV_Q38QUAL_NLL_OUT") {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&out).expect("nll out"));
        for (v, hit) in &per_position {
            f.write_all(&v.to_le_bytes()).expect("write nll");
            f.write_all(&[*hit as u8]).expect("write hit");
        }
        f.flush().expect("flush nll out");
        eprintln!("[q38qual] wrote {count} positions to {out}");
    }
    eprintln!(
        "[q38qual] FINAL_PPL n_ctx={n_ctx} ppl={ppl:.5} sem_ppl={:.5} nll={mean:.6} acc={acc:.5} \
         count={count} chunks={n_chunk}/{n_chunk_full} first={first} targets_per_chunk=\
         {targets_per_chunk} vocab={vocab} logits_dtype=bf16_scored_in_f64 checkpoint={} \
         elapsed_s={:.1}",
        sem * ppl,
        dir.display(),
        t0.elapsed().as_secs_f64()
    );
    assert!(
        ppl.is_finite() && ppl > 1.0 && ppl < Q38QUAL_SANE_PPL_CEILING_100_A_BROKEN_ARM_READS_IN_THE_THOUSANDS,
        "ppl {ppl} outside the sane band; the arm is broken, not merely worse"
    );
}

#[test]
#[ignore = "greedy-continues pre-tokenized chat jobs on the GRAPHED serving decode path -- the \
            same path every published ms/tok number is measured on -- so a retrieval failure this \
            finds is a failure of the shipped serving stack, not of an eval scaffold. Jobs are \
            read from NV_Q38QUAL_JOBS as one JSON object per line with `id`, `ids` (token ids \
            already rendered through the checkpoint's own chat template by the caller, so all \
            three engines consume identical ids) and `gen`. Generated ids go to \
            NV_Q38QUAL_GEN_OUT as one JSON object per line. Set NV_Q38QUAL=1"]
fn q38qual_greedy_continuation_of_pretokenized_chat_jobs_on_the_graphed_serving_path() {
    require_gate();
    let _one_gpu = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    std::env::set_var("NV_Q38_GDN_CHUNK_PREFILL", "1");
    print_live_arm_stack();

    let jobs_path = std::env::var("NV_Q38QUAL_JOBS").expect("set NV_Q38QUAL_JOBS to a jsonl file");
    let raw = std::fs::read_to_string(&jobs_path).expect("read jobs");
    let jobs: Vec<serde_json::Value> = raw
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("job line is json"))
        .collect();
    assert!(!jobs.is_empty(), "no jobs in {jobs_path}");
    let parsed: Vec<(String, Vec<u32>, usize)> = jobs
        .iter()
        .map(|j| {
            (
                j["id"].as_str().expect("job id").to_string(),
                j["ids"]
                    .as_array()
                    .expect("job ids")
                    .iter()
                    .map(|v| v.as_u64().expect("token id") as u32)
                    .collect(),
                j["gen"].as_u64().expect("job gen") as usize,
            )
        })
        .collect();
    let max_slots = parsed
        .iter()
        .map(|(_, ids, gen)| ids.len() + gen + Q38QUAL_DECODE_HEADROOM_SLOTS_BEYOND_THE_SCORED_OR_GENERATED_TAIL)
        .max()
        .expect("nonempty");

    let dir = snapshot_dir();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let (model, _cfg) = build_model(&dir, &device);
    let mut eng = GraphedQwen3Moe::new(model, &device, max_slots).expect("graphed engine");
    let block = Q38QUAL_BLOCK_512_IS_THE_PROVEN_QWEN35_DENSE_PREFILL_BLOCK;

    let mut out = std::io::BufWriter::new(
        std::fs::File::create(std::env::var("NV_Q38QUAL_GEN_OUT").expect("NV_Q38QUAL_GEN_OUT"))
            .expect("gen out"),
    );
    for (id, ids, gen) in &parsed {
        let t0 = std::time::Instant::now();
        eng.set_moe_dispatch(None);
        eng.reset().expect("engine reset");
        let mut pos = 0usize;
        let mut row: Vec<f32> = Vec::new();
        while pos < ids.len() {
            let n = block.min(ids.len() - pos);
            row = eng
                .prefill(&ids[pos..pos + n])
                .unwrap_or_else(|e| panic!("job {id} prefill at {pos}: {e:#}"));
            pos += n;
        }
        assert_eq!(eng.current_pos(), ids.len(), "job {id}: prefill landed short");
        eng.install_grouped_moe()
            .expect("dense branch of install_grouped_moe arms capture");
        let mut generated: Vec<u32> = Vec::with_capacity(*gen);
        for step in 0..*gen {
            let m = row.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            assert!(m.is_finite(), "job {id}: non-finite logits at step {step}");
            let tok = row
                .iter()
                .position(|&v| v == m)
                .expect("the row maximum is an element of the row") as u32;
            generated.push(tok);
            if step + 1 < *gen {
                row = eng
                    .forward_decode_logits_vec(tok)
                    .unwrap_or_else(|e| panic!("job {id} decode step {step}: {e:#}"));
            }
        }
        let captured = eng.capture_active();
        eprintln!(
            "[q38qual] job {id} prompt={} gen={} capture_active={captured} elapsed_s={:.1}",
            ids.len(),
            gen,
            t0.elapsed().as_secs_f64()
        );
        writeln!(
            out,
            "{}",
            serde_json::json!({"id": id, "prompt_tokens": ids.len(), "gen": generated, "capture_active": captured})
        )
        .expect("write gen");
        out.flush().expect("flush gen");
    }
}

#[test]
#[ignore = "prefills NV_Q38QUAL_DEPTH real corpus tokens and then teacher-forces \
            NV_Q38QUAL_STEPS decode steps on the GRAPHED serving path, so the NLL it reports is \
            produced by the very kernels the ms/tok record is set on. Run it once per arm stack \
            in the same session and pair the per-position dumps: that is what prices a speed arm \
            in quality. Set NV_Q38QUAL=1"]
fn q38qual_teacher_forced_nll_at_depth_on_the_graphed_serving_decode_path() {
    require_gate();
    let _one_gpu = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    std::env::set_var("NV_Q38_GDN_CHUNK_PREFILL", "1");
    print_live_arm_stack();

    let dir = snapshot_dir();
    let ids = corpus_ids(&dir);
    let depth = env_usize("NV_Q38QUAL_DEPTH", 16384);
    let steps = env_usize("NV_Q38QUAL_STEPS", 512);
    let skip = env_usize("NV_Q38QUAL_SKIP", 0);
    assert!(
        ids.len() > skip + depth + steps + 1,
        "corpus has {} tokens, need {}",
        ids.len(),
        skip + depth + steps + 1
    );
    let window = &ids[skip..skip + depth + steps + 1];

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let (model, cfg) = build_model(&dir, &device);
    let mut eng = GraphedQwen3Moe::new(
        model,
        &device,
        depth + steps + Q38QUAL_DECODE_HEADROOM_SLOTS_BEYOND_THE_SCORED_OR_GENERATED_TAIL,
    )
    .expect("graphed engine");
    let block = Q38QUAL_BLOCK_512_IS_THE_PROVEN_QWEN35_DENSE_PREFILL_BLOCK;

    eng.set_moe_dispatch(None);
    eng.reset().expect("engine reset");
    let t0 = std::time::Instant::now();
    let mut pos = 0usize;
    while pos < depth {
        let n = block.min(depth - pos);
        eng.prefill(&window[pos..pos + n])
            .unwrap_or_else(|e| panic!("prefill at {pos}: {e:#}"));
        pos += n;
    }
    assert_eq!(eng.current_pos(), depth, "chunked prefill must land at {depth}");
    eng.install_grouped_moe()
        .expect("dense branch of install_grouped_moe arms capture");
    eprintln!("[q38qual] prefilled {depth} in {:.1}s", t0.elapsed().as_secs_f64());

    let mut per_position: Vec<(f64, bool)> = Vec::with_capacity(steps);
    for i in 0..steps {
        let row = eng
            .forward_decode_logits_vec(window[depth + i])
            .unwrap_or_else(|e| panic!("teacher-forced decode step {i}: {e:#}"));
        let tgt = [window[depth + i + 1]];
        per_position.extend(nll_and_top1_of_rows(&row, cfg.vocab_size, &tgt));
    }
    assert!(
        eng.capture_active(),
        "the graphed serving basis fell back to uncaptured; diagnose the blocker printed above, \
         do not relabel the number"
    );

    let mean = per_position.iter().map(|x| x.0).sum::<f64>() / steps as f64;
    let acc = per_position.iter().filter(|x| x.1).count() as f64 / steps as f64;
    if let Ok(out) = std::env::var("NV_Q38QUAL_NLL_OUT") {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&out).expect("nll out"));
        for (v, hit) in &per_position {
            f.write_all(&v.to_le_bytes()).expect("write nll");
            f.write_all(&[*hit as u8]).expect("write hit");
        }
        f.flush().expect("flush");
    }
    eprintln!(
        "[q38qual] FINAL_DEPTH_NLL depth={depth} skip={skip} steps={steps} nll={mean:.6} \
         ppl={:.5} acc={acc:.5} capture_active=true path=graphed_serving_decode \
         logits_dtype=bf16_scored_in_f64 elapsed_s={:.1}",
        mean.exp(),
        t0.elapsed().as_secs_f64()
    );
}
