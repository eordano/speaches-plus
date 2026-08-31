#![cfg(feature = "cuda")]

mod hub_dirs;
mod sufspec_classes;

use candle_core::Device;
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::verify_trunk_tap;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};

const NVFP4_REPO: &str = "models--unsloth--Qwen3.8-27B-NVFP4";
const CHECKPOINT_LABEL: &str = "unsloth/Qwen3.8-27B-NVFP4";
const PROMPT_TOKENS_1024_THE_DEPTH_THE_SUFSPEC_CAMPAIGN_SAW_THE_M7_ROW0_FLIP_AT: usize = 1024;
const MAX_SEQ_2048_COVERS_PROMPT_CHAIN_BUILD_AND_THE_WIDEST_VERIFY: usize = 2048;
const ROWS_CENSUS_4_6_7_8_BRACKET_THE_ONE_TIME_6_TO_7_TRUNK_TRANSITION: [usize; 4] = [4, 6, 7, 8];

fn require_gate(gate: &str) {
    if std::env::var(gate).ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set {gate}=1");
    }
}

fn top2(row: &[f32]) -> (usize, f32) {
    let (mut i1, mut v1) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > v1 {
            i1 = i;
            v1 = v;
        }
    }
    (i1, v1)
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 once; set NV_QWEN38_SERVING_TEST=1 -- row 0 of \
            an eager verify reads only the anchor and committed kv, so every per-layer row-0 \
            hash must agree across m; the m7triage residual is a one-time bitwise 6->7 trunk \
            transition, and this census taps the trunk after every mixer/norm/ffn/residual \
            stage of all 64 layers to name the first op whose row 0 differs between m=6 and m=7"]
fn qwen38_eager_verify_row0_per_layer_taps_agree_between_m6_and_m7_naming_the_first_divergent_op(
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
    let mut engine = GraphedQwen3Moe::new(
        base,
        &device,
        MAX_SEQ_2048_COVERS_PROMPT_CHAIN_BUILD_AND_THE_WIDEST_VERIFY,
    )
    .expect("graphed engine");
    engine.install_grouped_moe().expect("routing install");
    engine.set_device_routing(false);

    let max_rows = *ROWS_CENSUS_4_6_7_8_BRACKET_THE_ONE_TIME_6_TO_7_TRUNK_TRANSITION
        .iter()
        .max()
        .unwrap();
    for rows in ROWS_CENSUS_4_6_7_8_BRACKET_THE_ONE_TIME_6_TO_7_TRUNK_TRANSITION
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
        "[m67-census] basis: checkpoint={CHECKPOINT_LABEL} backend=cuda depth={depth} \
         anchor={anchor} chain={chain:?} (eager verify only; taps sync per stage)"
    );

    std::env::set_var("NV_Q38_VERIFY_TRUNK_TAP", "1");
    let _ = verify_trunk_tap::take();
    let mut per_m: Vec<(usize, Vec<verify_trunk_tap::TapRecord>)> = Vec::new();
    for &m in &ROWS_CENSUS_4_6_7_8_BRACKET_THE_ONE_TIME_6_TO_7_TRUNK_TRANSITION {
        engine.reset().expect("reset between arms");
        let _ = prefill(&mut engine);
        let _ = verify_trunk_tap::take();
        engine
            .forward_verify_chain(&chain[..m])
            .unwrap_or_else(|e| panic!("eager verify m={m}: {e:#}"));
        engine.synchronize().expect("post-verify sync");
        let recs = verify_trunk_tap::take();
        assert!(
            !recs.is_empty(),
            "m={m}: the trunk tap recorded nothing; the census executed no probed forward -- \
             check NV_Q38_VERIFY_TRUNK_TAP wiring before reading any of this as coverage"
        );
        per_m.push((m, recs));
    }
    std::env::remove_var("NV_Q38_VERIFY_TRUNK_TAP");

    let base_recs = &per_m[0].1;
    let mut failures = String::new();
    for (m, recs) in per_m.iter().skip(1) {
        assert_eq!(
            recs.len(),
            base_recs.len(),
            "m={m}: tap count {} != m={} count {} -- the two runs dispatched different stage \
             sequences and the per-stage alignment below would lie",
            recs.len(),
            per_m[0].0,
            base_recs.len()
        );
        let mut first: Option<String> = None;
        for (r, b) in recs.iter().zip(base_recs.iter()) {
            assert_eq!(
                (r.li, r.label),
                (b.li, b.label),
                "m={m}: stage misalignment against m={}",
                per_m[0].0
            );
            if r.fnv != b.fnv && first.is_none() {
                first = Some(format!(
                    "m={m} vs m={}: first divergent stage li={:02} {} (fnv {:016x} vs {:016x}, \
                     maxabs {} vs {})",
                    per_m[0].0, r.li, r.label, r.fnv, b.fnv, r.maxabs, b.maxabs
                ));
            }
        }
        match first {
            Some(f) => {
                eprintln!("[m67-census] {f}");
                failures.push_str(&f);
                failures.push('\n');
            }
            None => eprintln!(
                "[m67-census] m={m} vs m={}: all {} per-layer row-0 taps bitwise identical",
                per_m[0].0,
                base_recs.len()
            ),
        }
    }
    assert!(
        failures.is_empty(),
        "row 0 of the eager verify trunk diverged per layer with the chain length m; the first \
         divergent stage names the culprit op family:\n{failures}"
    );
}
