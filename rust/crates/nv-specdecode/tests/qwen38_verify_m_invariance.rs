#![cfg(feature = "cuda")]

mod hub_dirs;
mod sufspec_classes;

use candle_core::Device;
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";
const CHECKPOINT_LABEL: &str = "unsloth/Qwen3.8-27B-NVFP4";
const PROMPT_TOKENS_1024_THE_DEPTH_THE_SUFSPEC_CAMPAIGN_SAW_THE_M7_ROW0_FLIP_AT: usize = 1024;
const MAX_SEQ_2048_COVERS_PROMPT_CHAIN_BUILD_AND_THE_WIDEST_VERIFY: usize = 2048;
const ROWS_SWEEP_4_TO_8_BRACKETS_THE_MK_VERIFY_TEMPLATE_CAP_8: [usize; 5] = [4, 5, 6, 7, 8];
const ROW0_DRIFT_BAND_1_0_SITS_BETWEEN_CROSS_M_GEMM_PLAN_ROUNDING_AND_THE_3_5_LOGIT_COLLAPSE_THE_M7_DEFECT_SHOWS: f32 =
    1.0;

fn require_gate(gate: &str) {
    if std::env::var(gate).ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {gate}=1");
    }
}

fn top2(row: &[f32]) -> (usize, f32, usize, f32) {
    let (mut i1, mut v1, mut i2, mut v2) = (0usize, f32::NEG_INFINITY, 0usize, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > v1 {
            i2 = i1;
            v2 = v1;
            i1 = i;
            v1 = v;
        } else if v > v2 {
            i2 = i;
            v2 = v;
        }
    }
    (i1, v1, i2, v2)
}

fn fnv64_over_f32_bits(vals: &[f32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for v in vals {
        for b in v.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 once; set NV_QWEN38_SERVING_TEST=1 -- row 0 of \
            a verify is the anchor's next-token logits over the SAME committed kv whatever the \
            chain length, so its logits must agree across m=4..8 in both the eager and the \
            graphed verify; the sufspec campaign saw m=7 return top1=47193 where m=4 returns 62 \
            for the same anchor at depth 1024, so this sweep is the reproducing gate for that \
            defect and prints the full m x arm matrix before asserting"]
fn qwen38_verify_row0_logits_agree_across_m4_to_m8_eager_and_graphed_because_row0_reads_only_the_anchor_and_committed_kv(
) {
    require_gate("NV_QWEN38_SERVING_TEST");
    let dir = match std::env::var("NV_QWEN38_DIR") {
        Ok(d) if !d.is_empty() => std::path::PathBuf::from(d),
        _ => hub_dirs::snapshot(NVFP4_REPO, &["config.json", "tokenizer.json"]).unwrap_or_else(
            || {
                panic!(
                    "PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: no hydrated {NVFP4_REPO} \
                     snapshot; set NV_QWEN38_DIR"
                )
            },
        ),
    };
    let tok =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let depth = PROMPT_TOKENS_1024_THE_DEPTH_THE_SUFSPEC_CAMPAIGN_SAW_THE_M7_ROW0_FLIP_AT;
    let prompt = sufspec_classes::tokens_of(
        &tok,
        &sufspec_classes::agentic_transcript_text(4 + depth / 128),
        depth,
        "agentic",
    );

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = nv_weights::QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("weights");
    let base = Qwen3Moe::from_loader_dense_quantized(cfg, &weights, &qcfg, &device)
        .expect("track-2 dense NVFP4 loader");
    drop(weights);
    let mut engine =
        GraphedQwen3Moe::new(base, &device, MAX_SEQ_2048_COVERS_PROMPT_CHAIN_BUILD_AND_THE_WIDEST_VERIFY)
            .expect("graphed engine");
    engine.install_grouped_moe().expect("routing install");
    engine.set_device_routing(false);

    let max_rows = *ROWS_SWEEP_4_TO_8_BRACKETS_THE_MK_VERIFY_TEMPLATE_CAP_8
        .iter()
        .max()
        .unwrap();
    for rows in ROWS_SWEEP_4_TO_8_BRACKETS_THE_MK_VERIFY_TEMPLATE_CAP_8
        .iter()
        .rev()
    {
        engine.ensure_verify_lane(*rows).expect("verify lane");
    }
    engine.synchronize().expect("post-lane sync");

    let prefill = |engine: &mut GraphedQwen3Moe| -> Vec<f32> {
        let mut fed = 0usize;
        let mut last = Vec::new();
        while fed < prompt.len() {
            let n = 512.min(prompt.len() - fed);
            last = engine
                .prefill(&prompt[fed..fed + n])
                .unwrap_or_else(|e| panic!("prefill chunk at {fed}: {e:#}"));
            fed += n;
        }
        last
    };

    let last = prefill(&mut engine);
    let anchor = top2(&last).0 as u32;
    let mut chain: Vec<u32> = vec![anchor];
    let mut cur = anchor;
    for _ in 0..max_rows - 1 {
        engine.forward_decode(cur).expect("chain-build decode");
        cur = top2(&engine.logits_host().expect("chain-build logits")).0 as u32;
        chain.push(cur);
    }
    eprintln!(
        "[m-invariance] basis: checkpoint={CHECKPOINT_LABEL} backend=cuda depth={depth} \
         anchor={anchor} chain={chain:?} (greedy m=1 continuation; rows past 0 never feed row 0)"
    );

    engine.reset().expect("reset before decode oracle");
    let _ = prefill(&mut engine);
    engine
        .forward_decode(anchor)
        .expect("decode oracle for the anchor row");
    let decode_row = engine.logits_host().expect("decode oracle logits");

    let vocab = engine.vocab_size();
    let mut row0: Vec<(bool, usize, Vec<f32>)> = Vec::new();
    for graphed in [false, true] {
        for &m in &ROWS_SWEEP_4_TO_8_BRACKETS_THE_MK_VERIFY_TEMPLATE_CAP_8 {
            engine.reset().expect("reset between arms");
            engine.set_device_routing(graphed);
            let _ = prefill(&mut engine);
            engine
                .forward_verify_chain(&chain[..m])
                .unwrap_or_else(|e| panic!("verify graphed={graphed} m={m}: {e:#}"));
            if graphed {
                assert!(
                    engine.capture_active(),
                    "graphed arm m={m} fell back to uncaptured; the captured replay is half the \
                     subject -- diagnose the stderr blocker above"
                );
            }
            let flat = engine.verify_logits_host().expect("verify logits");
            let hid: Vec<f32> = engine
                .verify_hidden_rows_tensor_valid_until_next_forward()
                .expect("verify hidden rows")
                .to_dtype(candle_core::DType::F32)
                .expect("hidden f32")
                .flatten_all()
                .expect("hidden flatten")
                .to_vec1()
                .expect("hidden host");
            let hidden_cols = hid.len() / m;
            eprintln!(
                "[m-invariance] graphed={graphed} m={m} hidden_row0_fnv={:016x}",
                fnv64_over_f32_bits(&hid[..hidden_cols])
            );
            for r in 0..m {
                let (i1, v1, i2, v2) = top2(&flat[r * vocab..(r + 1) * vocab]);
                eprintln!(
                    "[m-invariance] graphed={graphed} m={m} row={r} tok_in={} top1={i1}:{v1:.4} \
                     top2={i2}:{v2:.4} margin={:.4}",
                    chain[r],
                    v1 - v2
                );
            }
            row0.push((graphed, m, flat[..vocab].to_vec()));
        }
    }

    let base_row0 = |graphed: bool| -> &Vec<f32> {
        &row0
            .iter()
            .find(|(g, m, _)| *g == graphed && *m == 4)
            .unwrap()
            .2
    };
    let mut failures = String::new();
    let (di, dv, ..) = top2(&decode_row);
    for (graphed, m, logits) in &row0 {
        let base = base_row0(*graphed);
        let (bi, bv, ..) = top2(base);
        let (i1, v1, ..) = top2(logits);
        let drift = max_abs_diff(logits, base);
        let drift_vs_decode = max_abs_diff(logits, &decode_row);
        eprintln!(
            "[m-invariance] graphed={graphed} m={m} row0 top1={i1}:{v1:.4} vs m=4 top1={bi}:{bv:.4} \
             max_abs_drift={drift:.4} vs_decode top1={di}:{dv:.4} drift={drift_vs_decode:.4}"
        );
        if i1 != bi
            || drift
                > ROW0_DRIFT_BAND_1_0_SITS_BETWEEN_CROSS_M_GEMM_PLAN_ROUNDING_AND_THE_3_5_LOGIT_COLLAPSE_THE_M7_DEFECT_SHOWS
        {
            failures.push_str(&format!(
                "graphed={graphed} m={m}: row0 top1={i1}:{v1:.4} but m=4 says {bi}:{bv:.4}, \
                 max_abs_drift={drift:.4}\n"
            ));
        }
    }
    let (e4, g4) = (base_row0(false), base_row0(true));
    let cross = max_abs_diff(e4, g4);
    let (ei, ..) = top2(e4);
    let (gi, ..) = top2(g4);
    eprintln!("[m-invariance] eager-vs-graphed m=4 row0: top1 {ei} vs {gi} max_abs_drift={cross:.4}");
    assert!(
        failures.is_empty(),
        "row 0 of the verify changed with the chain length m; the anchor's next-token logits \
         must not depend on how many draft rows ride behind it:\n{failures}"
    );
}
