#![cfg(feature = "cuda")]

use candle_core::Device;
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use tokenizers::Tokenizer;

const ROWS_SHAPES_K2_TO_K4_PLUS_MK_VERIFY_CEILING: [usize; 4] = [3, 4, 5, 8];
const CHAINS: usize = 4;
const MAX_SEQ: usize = 512;
const EAGER_GROUPED_PREFILL_LEN_UNDER_MAX_VERIFY_MOE_TOKENS: usize = 24;

fn snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN36_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("qwen3.6 snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file())
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no qwen3.6 snapshot; set NV_QWEN36_DIR")
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}

struct PassOut {
    label: String,
    argmax_per_chain: Vec<Vec<u32>>,
    nonfinite_total: usize,
    last_chain_logits: Vec<f32>,
    capture_active: bool,
    node_count: usize,
}

fn run_pass(
    engine: &mut GraphedQwen3Moe,
    prompt: &[u32],
    device_routing: bool,
    label: &str,
) -> PassOut {
    engine.reset().expect("reset");
    engine.set_moe_dispatch(None);
    let last = engine.prefill(prompt).expect("host-routed prefill");
    let anchor = argmax(&last);
    engine.install_grouped_moe().expect("install grouped moe");
    engine.set_device_routing(device_routing);
    for &rows in &ROWS_SHAPES_K2_TO_K4_PLUS_MK_VERIFY_CEILING {
        engine.ensure_verify_lane(rows).expect("verify lane");
    }
    let vocab = engine.vocab_size();
    let mut argmax_per_chain: Vec<Vec<u32>> = Vec::new();
    let mut nonfinite_total = 0usize;
    let mut last_chain_logits: Vec<f32> = Vec::new();
    for c in 0..CHAINS {
        for (s, &rows) in ROWS_SHAPES_K2_TO_K4_PLUS_MK_VERIFY_CEILING
            .iter()
            .enumerate()
        {
            let mut block: Vec<u32> = Vec::with_capacity(rows);
            block.push(anchor);
            for j in 1..rows {
                block.push(prompt[(c * 31 + s * 7 + j) % prompt.len()]);
            }
            engine
                .forward_verify_chain(&block)
                .unwrap_or_else(|e| panic!("{label}: chain c={c} rows={rows}: {e:#}"));
            let flat = engine.verify_logits_host().expect("verify logits");
            nonfinite_total += flat.iter().filter(|v| !v.is_finite()).count();
            let rows_argmax: Vec<u32> = (0..rows)
                .map(|r| argmax(&flat[r * vocab..(r + 1) * vocab]))
                .collect();
            argmax_per_chain.push(rows_argmax);
            last_chain_logits = flat;
            engine.commit_verify_consumed(rows).expect("commit");
        }
    }
    PassOut {
        label: label.to_string(),
        argmax_per_chain,
        nonfinite_total,
        last_chain_logits,
        capture_active: engine.capture_active(),
        node_count: engine.captured_graph_node_count(),
    }
}

fn report(truth: &PassOut, arm: &PassOut) -> (bool, f32) {
    let same = truth.argmax_per_chain == arm.argmax_per_chain;
    let max_abs = truth
        .last_chain_logits
        .iter()
        .zip(arm.last_chain_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[moecap] {} vs {}: argmax_rows_identical={} last_chain_logits_max_abs_diff={:.6} \
         nonfinite={} capture_active={} nodes={}",
        truth.label, arm.label, same, max_abs, arm.nonfinite_total, arm.capture_active,
        arm.node_count,
    );
    for (c, (t, a)) in truth
        .argmax_per_chain
        .iter()
        .zip(arm.argmax_per_chain.iter())
        .enumerate()
    {
        if t != a {
            eprintln!("[moecap]   block {c}: truth={t:?} arm={a:?}");
        }
    }
    (same, max_abs)
}

#[test]
#[ignore = "loads the ~22 GB Qwen3.6-35B checkpoint; set NV_QWEN36_MOECAP=1 -- proves the \
            grouped m>1 verify no longer needs its host stall (default-off vs NV_MOE_MT_STALL=1 \
            truth arm, rows 3/4/5/8), that all four rows-keyed verify graphs capture and replay \
            bit-identically in one process, and that eager grouped prefill (m=24 through the \
            verify dispatch) returns finite logits agreeing with the host-routed prefill"]
fn grouped_multi_token_verify_needs_no_stall_and_captures_across_rows_shapes() {
    if std::env::var("NV_QWEN36_MOECAP").as_deref() != Ok("1") {
        panic!("set NV_QWEN36_MOECAP=1 to run (it must never silently skip)");
    }
    let dir = snapshot_dir();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw).expect("config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let base = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let prompt_text = "<|im_start|>user\nWhat is the capital of France? Answer in one short \
                       sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let prompt: Vec<u32> = tok
        .encode(prompt_text, false)
        .expect("encode")
        .get_ids()
        .to_vec();
    let per_pass: usize = CHAINS * ROWS_SHAPES_K2_TO_K4_PLUS_MK_VERIFY_CEILING.iter().sum::<usize>();
    assert!(prompt.len() + per_pass + 8 < MAX_SEQ);

    let mut engine = GraphedQwen3Moe::new(base, &device, MAX_SEQ).expect("graphed engine");

    std::env::set_var("NV_MOE_MT_STALL", "1");
    let arm_a = run_pass(&mut engine, &prompt, false, "A_stall_forced_uncaptured");
    assert_eq!(
        arm_a.nonfinite_total, 0,
        "truth arm produced non-finite verify logits; nothing downstream is interpretable"
    );

    std::env::remove_var("NV_MOE_MT_STALL");
    let arm_b = run_pass(&mut engine, &prompt, false, "B_default_no_stall_uncaptured");
    let (b_same, b_diff) = report(&arm_a, &arm_b);

    let eager_grouped_prefix = &prompt[..EAGER_GROUPED_PREFILL_LEN_UNDER_MAX_VERIFY_MOE_TOKENS];
    engine.reset().expect("reset for host prefix");
    engine.set_moe_dispatch(None);
    let host_last = engine.prefill(eager_grouped_prefix).expect("host prefix");
    engine.reset().expect("reset for grouped prefix");
    engine.install_grouped_moe().expect("grouped for prefill");
    engine.set_device_routing(false);
    let grouped_last = engine
        .prefill(eager_grouped_prefix)
        .expect("eager grouped prefill");
    let grouped_nonfinite = grouped_last.iter().filter(|v| !v.is_finite()).count();
    let d_same = argmax(&host_last) == argmax(&grouped_last);
    eprintln!(
        "[moecap] D_eager_grouped_prefill m={}: nonfinite={grouped_nonfinite} \
         argmax_matches_host_routed={d_same}",
        eager_grouped_prefix.len()
    );

    let arm_c = run_pass(&mut engine, &prompt, true, "C_default_no_stall_captured");
    let (c_same, c_diff) = report(&arm_a, &arm_c);

    eprintln!(
        "[moecap] VERDICT basis=(RedHatAI/Qwen3.6-35B-A3B-NVFP4 @ {}, rows_shapes={:?}, \
         chains={CHAINS}, greedy argmax rows, perf envs: {}): B_same={b_same} \
         B_maxdiff={b_diff:.6} C_same={c_same} C_maxdiff={c_diff:.6} C_capture_active={} \
         C_nodes={} D_nonfinite={grouped_nonfinite} D_same={d_same}",
        dir.file_name().unwrap_or_default().to_string_lossy(),
        ROWS_SHAPES_K2_TO_K4_PLUS_MK_VERIFY_CEILING,
        std::env::var("NV_MOECAP_ENV_LABEL").unwrap_or_else(|_| "none".into()),
        arm_c.capture_active,
        arm_c.node_count,
    );

    assert_eq!(
        arm_b.nonfinite_total, 0,
        "default (no stall) uncaptured arm produced non-finite logits: the m>1 grouped race \
         is live again"
    );
    assert!(
        b_same,
        "default (no stall) uncaptured arm diverged from the forced-stall truth arm"
    );
    assert_eq!(grouped_nonfinite, 0, "eager grouped prefill returned non-finite logits");
    assert!(
        d_same,
        "eager grouped prefill disagrees with host-routed prefill at the last row"
    );
    assert_eq!(arm_c.nonfinite_total, 0, "captured arm produced non-finite logits");
    assert!(
        arm_c.capture_active,
        "the rows-keyed verify graphs did not capture with the stall off by default"
    );
    assert!(
        c_same,
        "captured verify replay diverged from the forced-stall uncaptured truth arm"
    );
}
