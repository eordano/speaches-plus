#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{
    Qwen3Moe, Qwen3_5DenseConfig, KVSHARE_ENGAGES_PAST_THE_MEASURED_M1_DECODE_CROSSOVER_TOTAL,
    M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM,
};
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
const XPPL_DEFAULT_TEXT_FILE: &str = "/home/user/tmp/wf/q38/xppl/wikitext-2-raw/wiki.test.raw";
const XPPL_SANE_PPL_CEILING_100_A_BROKEN_ARM_READS_IN_THE_THOUSANDS: f64 = 100.0;
const XPPL_SANE_TOP1_FLOOR_0_10_IS_WHAT_A_WORD_SHUFFLED_CONTROL_SCORES_WHILE_REAL_TEXT_READS_0_56:
    f64 = 0.10;
const XPPL_RUNNING_FIGURE_EVERY_32_CHUNKS_IS_A_PROGRESS_TRACE_NOT_THE_PAIRED_TEST_GRID: usize = 32;
const XPPL_SLICE_SUM_MUST_RECONSTRUCT_THE_AGGREGATE_NLL_OR_THE_SLICING_CHANGED_THE_NUMBER_IT_SLICES:
    f64 = 1e-9;

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
    let mut top1 = 0usize;
    let mut slice_sum = 0f64;
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
        let mut slice_nll = 0f64;
        let mut slice_top1 = 0usize;
        for row in 0..targets {
            let r = &scored[row * vocab..(row + 1) * vocab];
            let target = slice[first + row + 1] as usize;
            let m = r.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            assert!(
                m.is_finite(),
                "non-finite logits in chunk {chunk} row {row}"
            );
            let argmax = r
                .iter()
                .position(|&v| v == m)
                .expect("the row maximum is an element of the row");
            let lse = m as f64
                + r.iter()
                    .map(|&v| ((v - m) as f64).exp())
                    .sum::<f64>()
                    .ln();
            let v = lse - r[target] as f64;
            nll += v;
            nll2 += v * v;
            count += 1;
            slice_nll += v;
            if argmax == target {
                slice_top1 += 1;
            }
        }
        slice_sum += slice_nll;
        top1 += slice_top1;
        eprintln!(
            "[xppl] slice {}/{} nll={:.9} n={targets} top1={slice_top1}",
            chunk + 1,
            n_chunk,
            slice_nll / targets as f64
        );
        if (chunk + 1) % XPPL_RUNNING_FIGURE_EVERY_32_CHUNKS_IS_A_PROGRESS_TRACE_NOT_THE_PAIRED_TEST_GRID
            == 0
            || chunk + 1 == n_chunk
        {
            eprintln!(
                "[xppl] chunk {}/{} running_ppl={:.4} running_acc={:.4}",
                chunk + 1,
                n_chunk,
                (nll / count as f64).exp(),
                top1 as f64 / count as f64
            );
        }
    }

    let mean = nll / count as f64;
    let ppl = mean.exp();
    let var = (nll2 / count as f64 - mean * mean).max(0.0);
    let sem = (var / (count as f64 - 1.0)).sqrt();
    let acc = top1 as f64 / count as f64;
    eprintln!(
        "[xppl] FINAL ppl={ppl:.4} +/- {:.5} acc={acc:.4} top1={top1}/{count} slices={n_chunk} \
         basis=(engine=speaches-plus nvfp4 dense arm as \
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
        ((slice_sum - nll) / count as f64).abs()
            < XPPL_SLICE_SUM_MUST_RECONSTRUCT_THE_AGGREGATE_NLL_OR_THE_SLICING_CHANGED_THE_NUMBER_IT_SLICES,
        "the printed per-slice NLLs sum to {slice_sum} but the aggregate accumulated {nll}; a \
         slice grid that does not reconstruct the published number is a different measurement, \
         not a finer one"
    );
    assert!(
        ppl.is_finite() && ppl > 1.0 && ppl < XPPL_SANE_PPL_CEILING_100_A_BROKEN_ARM_READS_IN_THE_THOUSANDS,
        "wikitext-2 ppl {ppl} outside the sane band; the arm is broken, not merely worse"
    );
    assert!(
        acc > XPPL_SANE_TOP1_FLOOR_0_10_IS_WHAT_A_WORD_SHUFFLED_CONTROL_SCORES_WHILE_REAL_TEXT_READS_0_56,
        "wikitext-2 top-1 accuracy {acc} is at or below what a word-shuffled control scores over \
         {vocab} ids; accuracy informs a verdict and gates nothing, but a run this low scored \
         noise, not the corpus"
    );
}

const XPPL8K_PREFILL_DEPTH_8192_PUTS_EVERY_SCORED_DECODE_PAST_THE_DEPTHGATE_KNEE: usize = 8192;
const XPPL8K_SCORED_DECODE_STEPS_256: usize = 256;
const XPPL8K_PREFILL_CHUNK_512_THE_PROVEN_PREFILL_BLOCK: usize = 512;
const XPPL8K_MAX_ARM_DELTA_NATS_0_02_TC_REASSOCIATION_IS_CENTINAT_SCALE_A_DEFECT_IS_WHOLE_NATS:
    f64 = 0.02;

#[test]
#[ignore = "loads the ~54 GB Qwen3.8-27B NVFP4 dense arm GRAPHED and teacher-forces 256 \
            wikitext-2 tokens at KV depth 8192 twice in ONE process: once on the default \
            decode dispatch and once with NV_Q38_ATTN_DEPTHGATE=1 (every scored step is past \
            the knee, so the gated pass runs the TC kvshare arm); set NV_Q38_XPPL=1; the two \
            teacher-forced mean NLLs must agree to centinats because TC reassociation is \
            bf16-rounding-scale while a dispatch defect is whole-nat scale"]
fn qwen38_nvfp4_wikitext2_teacher_forced_nll_at_8k_depth_default_vs_depthgate_same_process() {
    require_gate();
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    std::env::set_var("NV_Q38_GDN_CHUNK_PREFILL", "1");
    std::env::set_var("NV_Q36_GRAPHED_DECODE_FIX", "1");
    std::env::set_var("NV_Q38_ATTN_DEPTHGATE", "1");

    let depth = XPPL8K_PREFILL_DEPTH_8192_PUTS_EVERY_SCORED_DECODE_PAST_THE_DEPTHGATE_KNEE;
    let steps = XPPL8K_SCORED_DECODE_STEPS_256;
    let knee = nv_kernels::cuda::flash_depthgate_knee_tokens();
    assert!(
        depth > knee,
        "prefill depth {depth} must exceed the knee {knee} or the gated pass never \
         engages the TC arm and this gate tests nothing"
    );

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
    assert!(
        ids.len() > depth + steps + 1,
        "corpus tokenizes to {} tokens, need {}",
        ids.len(),
        depth + steps + 1
    );

    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("quant config");
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("build Qwen3.8-27B dense arm");
    drop(weights);
    let mut eng = GraphedQwen3Moe::new(model, &device, depth + steps + 16)
        .expect("graphed engine with depthgate-sized splitk scratch");

    let chunk = XPPL8K_PREFILL_CHUNK_512_THE_PROVEN_PREFILL_BLOCK;
    let mut run_pass = |gate_on: bool| -> Vec<f64> {
        if gate_on {
            std::env::set_var("NV_Q38_ATTN_DEPTHGATE", "1");
        } else {
            std::env::remove_var("NV_Q38_ATTN_DEPTHGATE");
        }
        eng.set_moe_dispatch(None);
        eng.reset().expect("engine reset");
        let mut pos = 0usize;
        while pos < depth {
            let n = chunk.min(depth - pos);
            eng.prefill(&ids[pos..pos + n])
                .unwrap_or_else(|e| panic!("prefill chunk at pos {pos}: {e:#}"));
            pos += n;
        }
        assert_eq!(eng.current_pos(), depth, "chunked prefill must land at {depth}");
        eng.install_grouped_moe()
            .expect("dense branch of install_grouped_moe arms capture");
        let mut nlls = Vec::with_capacity(steps);
        for i in 0..steps {
            eng.forward_decode(ids[depth + i])
                .unwrap_or_else(|e| panic!("teacher-forced decode step {i}: {e:#}"));
            let r = eng.logits_host().expect("logits to host");
            let target = ids[depth + i + 1] as usize;
            let m = r.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            assert!(m.is_finite(), "non-finite logits at step {i} gate_on={gate_on}");
            let lse = m as f64
                + r.iter().map(|&v| ((v - m) as f64).exp()).sum::<f64>().ln();
            nlls.push(lse - r[target] as f64);
        }
        assert!(
            eng.capture_active(),
            "gate_on={gate_on}: the graphed serving basis fell back to uncaptured; \
             diagnose the blocker printed above, do not relabel the number"
        );
        nlls
    };

    let nll_off = run_pass(false);
    let nll_on = run_pass(true);
    std::env::remove_var("NV_Q38_ATTN_DEPTHGATE");

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let m_off = mean(&nll_off);
    let m_on = mean(&nll_on);
    let max_pos_delta = nll_off
        .iter()
        .zip(&nll_on)
        .map(|(a, b)| (a - b).abs())
        .fold(0f64, f64::max);
    eprintln!(
        "[xppl8k] teacher-forced wikitext-2 at depth {depth}: default mean_nll={m_off:.5} \
         ppl={:.4} | depthgate mean_nll={m_on:.5} ppl={:.4} | delta_nats={:+.5} \
         max_pos_delta={max_pos_delta:.5} steps={steps} knee={knee} basis=(graphed dense arm, \
         NV_Q36_GRAPHED_DECODE_FIX=1, NV_Q38_GDN_CHUNK_PREFILL=1, chunked real prefill, \
         teacher-forced decode, lse over full vocab in f64)",
        m_off.exp(),
        m_on.exp(),
        m_on - m_off
    );
    assert!(
        (m_on - m_off).abs()
            < XPPL8K_MAX_ARM_DELTA_NATS_0_02_TC_REASSOCIATION_IS_CENTINAT_SCALE_A_DEFECT_IS_WHOLE_NATS,
        "depthgate moved teacher-forced mean NLL by {:.5} nats at depth {depth}; TC \
         reassociation is centinat-scale, a whole-nat move is a dispatch defect",
        m_on - m_off
    );
}

const Q38NULL_SCORED_STEPS_512_EVERY_STEP_DECODES_AT_OR_PAST_THE_KVSHARE_KNEE: usize = 512;
const Q38NULL_WINDOW_COUNT_2_DISJOINT_CORPUS_SPANS: usize = 2;
const Q38NULL_RUNG_LADDER_SIX_SPLIT_COUNTS_OF_THE_ONE_FUSED_KERNEL_ARE_SIX_SUMMATION_ORDERS_OF_THE_SAME_COMPUTATION:
    [i32; 6] = [8, 12, 16, 24, 32, 47];
const A_REASSOCIATION_NULL_MAY_NOT_MOVE_TOP1_ACCURACY_BY_MORE_THAN_THIS_OR_THE_SPLIT_COUNT_IS_NOT_THE_SAME_ARITHMETIC:
    f64 = 0.05;
const Q38NULL_SHUFFLED_CONTROL_MUST_READ_AT_LEAST_THIS_MANY_TIMES_THE_REAL_PPL_OR_THE_HARNESS_SCORED_NOISE:
    f64 = 10.0;
const Q38NULL_KV_SLOT_HEADROOM_16_BEYOND_THE_SCORED_STEPS: usize = 16;
const Q38NULL_STATIC_CACHED_ENVS_THAT_WOULD_SILENTLY_OVERRIDE_OR_RECLASS_EVERY_PASS: [&str; 3] =
    ["NV_E4B_FLASH_SPLITS", "NV_Q38_ATTN_GQA_STAGE", "NV_Q38_ATTN_DEPTHGATE"];

struct Q38NullPass {
    label: &'static str,
    rung: Option<i32>,
    kvshare: bool,
    shuffled: bool,
}

fn q38null_shuffle_in_place_lcg_fisher_yates(v: &mut [u32]) {
    let mut state = 0x9e3779b97f4a7c15u64;
    for i in (1..v.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        v.swap(i, j);
    }
}

#[test]
#[ignore = "loads the ~54 GB Qwen3.8-27B NVFP4 dense arm GRAPHED and teacher-forces 512 \
            wikitext-2 tokens at KV depth 16384 (the kvshare m=1 decode knee, so the arm is \
            engaged at every scored step) over 2 disjoint corpus windows, once per arm in ONE \
            process with a fresh real chunked prefill and a fresh decode-graph capture per \
            pass: a control, a control repeat, the six-rung NV_FLASH_SPLITS_RUNG split-count \
            ladder of the same fused fp8 splitk kernel (the family's reassociation null), and \
            the NV_Q38_DECODE_KVSHARE=1 group-fold arm read against that null; one window adds \
            a shuffled-continuation control that gates the run; set NV_Q38_XPPL=1"]
fn qwen38_nvfp4_wikitext2_split_rung_ladder_is_the_reassociation_null_the_kvshare_decode_arm_reads_against_at_16k_depth(
) {
    require_gate();
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    for e in Q38NULL_STATIC_CACHED_ENVS_THAT_WOULD_SILENTLY_OVERRIDE_OR_RECLASS_EVERY_PASS {
        assert!(
            std::env::var(e).is_err(),
            "{e} is set; it is read once per process (or reclasses the decode route) and would \
             poison every rung of the ladder"
        );
    }
    std::env::set_var("NV_Q38_GDN_CHUNK_PREFILL", "1");
    std::env::set_var("NV_Q36_GRAPHED_DECODE_FIX", "1");
    std::env::remove_var("NV_FLASH_SM_RUNGS");
    std::env::remove_var("NV_FLASH_SPLITS_RUNG");
    std::env::remove_var("NV_FLASH_SPLITS_RUNG_WINDOWED");

    let depth = KVSHARE_ENGAGES_PAST_THE_MEASURED_M1_DECODE_CROSSOVER_TOTAL;
    let steps = Q38NULL_SCORED_STEPS_512_EVERY_STEP_DECODES_AT_OR_PAST_THE_KVSHARE_KNEE;
    let windows = Q38NULL_WINDOW_COUNT_2_DISJOINT_CORPUS_SPANS;
    let span = depth + steps + 1;

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
    assert!(
        ids.len() >= windows * span,
        "corpus tokenizes to {} tokens, need {} for {windows} disjoint windows of {span}",
        ids.len(),
        windows * span
    );

    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("quant config");
    let nh = cfg.num_attention_heads as i32;
    assert!(
        cfg.head_dim == 256 && cfg.num_attention_heads % cfg.num_key_value_heads == 0
            && matches!(cfg.num_attention_heads / cfg.num_key_value_heads, 6 | 8),
        "kvshare routes only hd256 group-6/8 shapes; this checkpoint reads nh={} nkv={} hd={} \
         and the arm pass would silently score the fused kernel",
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim
    );
    let stock_pick = nv_kernels::cuda::flash_decode_fp8_splits_pick(nh, 0);

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("build Qwen3.8-27B dense arm");
    drop(weights);
    std::env::set_var(
        M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM,
        "1",
    );
    let mut eng = GraphedQwen3Moe::new(
        model,
        &device,
        depth + steps + Q38NULL_KV_SLOT_HEADROOM_16_BEYOND_THE_SCORED_STEPS,
    )
    .expect("graphed engine whose cache saw the kvshare env and sized the 128-split scratch");
    std::env::set_var(
        M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM,
        "0",
    );

    let chunk = XPPL8K_PREFILL_CHUNK_512_THE_PROVEN_PREFILL_BLOCK;
    let mut run_pass = |off: usize,
                        rung: Option<i32>,
                        kvshare_on: bool,
                        scored: &[u32],
                        label: &str|
     -> (Vec<f64>, Vec<bool>, i32) {
        std::env::set_var(
            M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM,
            if kvshare_on { "1" } else { "0" },
        );
        match rung {
            Some(r) => {
                std::env::set_var("NV_FLASH_SM_RUNGS", "1");
                std::env::set_var("NV_FLASH_SPLITS_RUNG", r.to_string());
            }
            None => {
                std::env::remove_var("NV_FLASH_SM_RUNGS");
                std::env::remove_var("NV_FLASH_SPLITS_RUNG");
            }
        }
        let picked = nv_kernels::cuda::flash_decode_fp8_splits_pick(nh, 0);
        if let Some(r) = rung {
            assert_eq!(
                picked, r,
                "{label}: NV_FLASH_SPLITS_RUNG={r} did not reach the picker (picked {picked}); \
                 a rung that silently clamps poisons the null"
            );
        } else {
            assert_eq!(
                picked, stock_pick,
                "{label}: env hygiene failed, a rung pin leaked into a no-rung pass"
            );
        }
        eng.set_moe_dispatch(None);
        eng.reset().expect("engine reset invalidates the prior pass's captured graph");
        let mut pos = 0usize;
        while pos < depth {
            let n = chunk.min(depth - pos);
            eng.prefill(&ids[off + pos..off + pos + n])
                .unwrap_or_else(|e| panic!("{label}: prefill chunk at pos {pos}: {e:#}"));
            pos += n;
        }
        assert_eq!(
            eng.current_pos(),
            depth,
            "{label}: chunked prefill must land at {depth}"
        );
        eng.install_grouped_moe()
            .expect("dense branch of install_grouped_moe arms capture");
        let mut nlls = Vec::with_capacity(steps);
        let mut hits = Vec::with_capacity(steps);
        for i in 0..steps {
            eng.forward_decode(scored[i])
                .unwrap_or_else(|e| panic!("{label}: teacher-forced decode step {i}: {e:#}"));
            let r = eng.logits_host().expect("logits to host");
            let target = scored[i + 1] as usize;
            assert!(target < r.len(), "{label}: target id {target} outside vocab {}", r.len());
            let m = r.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
            assert!(m.is_finite(), "{label}: non-finite logits at step {i}");
            let argmax = r
                .iter()
                .position(|&v| v == m)
                .expect("the row maximum is an element of the row");
            let lse = m as f64
                + r.iter().map(|&v| ((v - m) as f64).exp()).sum::<f64>().ln();
            nlls.push(lse - r[target] as f64);
            hits.push(argmax == target);
        }
        assert!(
            eng.capture_active(),
            "{label}: the graphed serving basis fell back to uncaptured; diagnose the blocker \
             printed above, do not relabel the number"
        );
        (nlls, hits, picked)
    };

    let passes = [
        Q38NullPass { label: "control", rung: None, kvshare: false, shuffled: false },
        Q38NullPass { label: "control-repeat", rung: None, kvshare: false, shuffled: false },
        Q38NullPass { label: "rung8", rung: Some(8), kvshare: false, shuffled: false },
        Q38NullPass { label: "rung12", rung: Some(12), kvshare: false, shuffled: false },
        Q38NullPass { label: "rung16", rung: Some(16), kvshare: false, shuffled: false },
        Q38NullPass { label: "rung24", rung: Some(24), kvshare: false, shuffled: false },
        Q38NullPass { label: "rung32", rung: Some(32), kvshare: false, shuffled: false },
        Q38NullPass { label: "rung47", rung: Some(47), kvshare: false, shuffled: false },
        Q38NullPass { label: "kvshare", rung: None, kvshare: true, shuffled: false },
        Q38NullPass { label: "shuffled-control", rung: None, kvshare: false, shuffled: true },
    ];
    assert_eq!(
        Q38NULL_RUNG_LADDER_SIX_SPLIT_COUNTS_OF_THE_ONE_FUSED_KERNEL_ARE_SIX_SUMMATION_ORDERS_OF_THE_SAME_COMPUTATION
            .to_vec(),
        passes.iter().filter_map(|p| p.rung).collect::<Vec<_>>(),
        "the pass table must walk exactly the declared rung ladder"
    );

    let mut results: Vec<Vec<Option<(Vec<f64>, Vec<bool>, i32)>>> =
        (0..windows).map(|_| passes.iter().map(|_| None).collect()).collect();
    for w in 0..windows {
        let off = w * span;
        let real: Vec<u32> = ids[off + depth..off + depth + steps + 1].to_vec();
        let mut shuffled = real.clone();
        q38null_shuffle_in_place_lcg_fisher_yates(&mut shuffled);
        for (pi, p) in passes.iter().enumerate() {
            if p.shuffled && w != 0 {
                continue;
            }
            let scored = if p.shuffled { &shuffled } else { &real };
            let (nlls, hits, picked) = run_pass(off, p.rung, p.kvshare, scored, p.label);
            let sum_nll: f64 = nlls.iter().sum();
            let ppl = (sum_nll / steps as f64).exp();
            let top1 = hits.iter().filter(|&&h| h).count();
            eprintln!(
                "[q38null] arm={} window={w} ppl={ppl:.6} acc={:.6} sum_nll={sum_nll:.9} \
                 top1={top1}/{steps} picked_splits={picked}",
                p.label,
                top1 as f64 / steps as f64
            );
            results[w][pi] = Some((nlls, hits, picked));
        }
    }
    std::env::remove_var("NV_FLASH_SM_RUNGS");
    std::env::remove_var("NV_FLASH_SPLITS_RUNG");
    std::env::remove_var(
        M1_DECODE_KVSHARE_ENV_1_ALWAYS_GATE_AT_THE_MEASURED_KNEE_0_OR_UNSET_FOR_THE_FUSED_ARM,
    );

    eprintln!(
        "[q38null] BASIS engine=speaches-plus graphed nvfp4 dense arm, checkpoint={}, file={}, \
         depth={depth} steps={steps} windows={windows} span={span} chunk={chunk} \
         stock_pick={stock_pick} env=(NV_Q38_GDN_CHUNK_PREFILL=1, NV_Q36_GRAPHED_DECODE_FIX=1, \
         NV_Q38_DECODE_KVSHARE=0|1 per pass and =1 at cache construction, NV_FLASH_SM_RUNGS=1 \
         with NV_FLASH_SPLITS_RUNG=r on rung passes only), add_bos=false, lse over full vocab \
         in f64, fresh real chunked prefill and fresh decode-graph capture per pass",
        dir.display(),
        file.display()
    );

    for w in 0..windows {
        let (c_nlls, c_hits, _) = results[w][0].as_ref().expect("control ran");
        let c_sum: f64 = c_nlls.iter().sum();
        let c_ppl = (c_sum / steps as f64).exp();
        let c_acc = c_hits.iter().filter(|&&h| h).count() as f64 / steps as f64;
        let mut rung_dppls: Vec<(i32, f64, f64, f64)> = Vec::new();
        let mut kvshare_read: Option<(f64, f64, f64)> = None;
        for (pi, p) in passes.iter().enumerate() {
            if pi == 0 {
                continue;
            }
            let Some((nlls, hits, _)) = results[w][pi].as_ref() else { continue };
            let sum: f64 = nlls.iter().sum();
            let ppl = (sum / steps as f64).exp();
            let acc = hits.iter().filter(|&&h| h).count() as f64 / steps as f64;
            let diff_positions = nlls
                .iter()
                .zip(c_nlls.iter())
                .filter(|(a, b)| a != b)
                .count();
            let max_abs_pos_dnll = nlls
                .iter()
                .zip(c_nlls.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f64, f64::max);
            let dppl_pct = (ppl / c_ppl - 1.0) * 100.0;
            let dacc = acc - c_acc;
            let dsum = sum - c_sum;
            eprintln!(
                "[q38null] delta arm={} window={w} dppl_pct={dppl_pct:+.6} dacc={dacc:+.6} \
                 dsum_nll={dsum:+.9} diff_positions={diff_positions}/{steps} \
                 max_abs_pos_dnll={max_abs_pos_dnll:.9}",
                p.label
            );
            if p.shuffled {
                assert!(
                    ppl > Q38NULL_SHUFFLED_CONTROL_MUST_READ_AT_LEAST_THIS_MANY_TIMES_THE_REAL_PPL_OR_THE_HARNESS_SCORED_NOISE
                        * c_ppl,
                    "shuffled control ppl {ppl:.4} is not orders of magnitude above real \
                     {c_ppl:.4}; the harness did not score the real sequence through a live \
                     model and every row of this run is void"
                );
                continue;
            }
            assert!(
                ppl.is_finite() && ppl > 1.0
                    && ppl < XPPL_SANE_PPL_CEILING_100_A_BROKEN_ARM_READS_IN_THE_THOUSANDS,
                "{}: ppl {ppl} outside the sane band; the arm is broken, not merely worse",
                p.label
            );
            if let Some(r) = p.rung {
                assert!(
                    dacc.abs()
                        <= A_REASSOCIATION_NULL_MAY_NOT_MOVE_TOP1_ACCURACY_BY_MORE_THAN_THIS_OR_THE_SPLIT_COUNT_IS_NOT_THE_SAME_ARITHMETIC,
                    "rung {r} window {w} moved top-1 accuracy by {dacc:+.4}; a reduction-order \
                     change flips only near-ties, so this rung is a broken knob, not a null"
                );
                if r != stock_pick {
                    assert!(
                        diff_positions > 0,
                        "rung {r} window {w} is digit-identical to the control at every scored \
                         position while pinning a different split count than the stock pick \
                         {stock_pick}; the rung never took and the null is one arm scored twice"
                    );
                    rung_dppls.push((r, dppl_pct, dacc, dsum));
                }
            }
            if p.kvshare {
                assert!(
                    diff_positions > 0,
                    "kvshare window {w} is digit-identical to the fused control at every scored \
                     position; the group-fold kernel splits the scan differently by \
                     construction, so identity means the env never engaged"
                );
                assert!(
                    (dsum / steps as f64).abs()
                        < XPPL8K_MAX_ARM_DELTA_NATS_0_02_TC_REASSOCIATION_IS_CENTINAT_SCALE_A_DEFECT_IS_WHOLE_NATS,
                    "kvshare moved teacher-forced mean NLL by {:.5} nats at depth {depth}; \
                     fp-reorder is centinat-scale, a whole-nat move is a dispatch defect",
                    dsum / steps as f64
                );
                assert!(
                    dacc.abs()
                        <= A_REASSOCIATION_NULL_MAY_NOT_MOVE_TOP1_ACCURACY_BY_MORE_THAN_THIS_OR_THE_SPLIT_COUNT_IS_NOT_THE_SAME_ARITHMETIC,
                    "kvshare window {w} moved top-1 accuracy by {dacc:+.4}; an fp-reorder arm \
                     flips only near-ties, so this is an indexing defect, not reordering"
                );
                kvshare_read = Some((dppl_pct, dacc, dsum));
            }
        }
        let dppl_lo = rung_dppls.iter().map(|x| x.1).fold(f64::INFINITY, f64::min);
        let dppl_hi = rung_dppls.iter().map(|x| x.1).fold(f64::NEG_INFINITY, f64::max);
        let dacc_lo = rung_dppls.iter().map(|x| x.2).fold(f64::INFINITY, f64::min);
        let dacc_hi = rung_dppls.iter().map(|x| x.2).fold(f64::NEG_INFINITY, f64::max);
        let dppl_abs_max = rung_dppls.iter().map(|x| x.1.abs()).fold(0f64, f64::max);
        eprintln!(
            "[q38null] NULL window={w} rungs={:?} dppl_pct=[{dppl_lo:+.6},{dppl_hi:+.6}] \
             dacc=[{dacc_lo:+.6},{dacc_hi:+.6}] dppl_abs_max={dppl_abs_max:.6}",
            rung_dppls.iter().map(|x| x.0).collect::<Vec<_>>()
        );
        let (k_dppl, k_dacc, k_dsum) = kvshare_read.expect("kvshare pass ran");
        eprintln!(
            "[q38null] ARM kvshare window={w} dppl_pct={k_dppl:+.6} dacc={k_dacc:+.6} \
             dsum_nll={k_dsum:+.9} inside_reassociation_null={}",
            k_dppl.abs() <= dppl_abs_max
                && k_dacc >= dacc_lo.min(0.0)
                && k_dacc <= dacc_hi.max(0.0)
        );
    }
}
