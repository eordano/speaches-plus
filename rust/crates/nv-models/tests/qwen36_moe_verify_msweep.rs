#![cfg(feature = "cuda")]

use candle_core::Device;
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use tokenizers::Tokenizer;

const DEPTH_256_MATCHES_THE_ECON_AND_RECORD_BASIS: usize = 256;
const M1_DECODE_STEPS_64_MATCH_THE_M1AB_MEDIAN_BASIS: usize = 64;
const STORY_SEED: &str = "The lighthouse keeper had counted nine hundred and twelve storms from \
    the gallery deck, and he wrote each one into the ledger with the same steel pen. On the \
    morning the sea turned the color of old brass, he found a rowboat lodged in the rocks below \
    the light, its oars shipped neatly, its hull dry on the inside. There was no name on the \
    stern. There was a brass key on the center thwart, and under the key a note in his own \
    handwriting, dated forty years ahead. ";

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

fn prompt_of_depth(tok: &Tokenizer, depth: usize) -> Vec<u32> {
    let enc = |s: &str| -> Vec<u32> { tok.encode(s, false).expect("encode").get_ids().to_vec() };
    let head = enc("<|im_start|>user\n");
    let tail = enc(
        "\n\nContinue this story in the same style.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    );
    let seed = enc(STORY_SEED);
    let ctx_needed = depth
        .checked_sub(head.len() + tail.len())
        .expect("depth must fit the template frame");
    let mut out = head;
    let mut ctx: Vec<u32> = Vec::with_capacity(ctx_needed + seed.len());
    while ctx.len() < ctx_needed {
        ctx.extend_from_slice(&seed);
    }
    ctx.truncate(ctx_needed);
    out.extend_from_slice(&ctx);
    out.extend_from_slice(&tail);
    assert_eq!(out.len(), depth);
    out
}

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i as u32;
        }
    }
    best
}

fn median_of(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "loads the ~22 GB Qwen3.6-35B NVFP4; set NV_QWEN36_MSWEEP=1 and run in --release \
            under the GPU lock -- times the captured grouped-MoE verify replay alone (no host \
            logits readback billed) at rows m swept per NV_QWEN36_MSWEEP_ROWS (default 2,3,5), \
            self-consistent greedy chains, one process, lanes ensured largest-first; \
            NV_GRAPH_OFF=1 turns this into the eager stage-profile run for NV_MOE_DECODE_PROF"]
fn qwen36_moe_verify_msweep_captured_replay_ms_by_rows() {
    if std::env::var("NV_QWEN36_MSWEEP").as_deref() != Ok("1") {
        panic!("set NV_QWEN36_MSWEEP=1 to run (it must never silently skip)");
    }
    let rows_sweep: Vec<usize> = std::env::var("NV_QWEN36_MSWEEP_ROWS")
        .unwrap_or_else(|_| "2,3,5".to_string())
        .split(',')
        .map(|s| s.trim().parse().expect("NV_QWEN36_MSWEEP_ROWS"))
        .collect();
    let rounds = env_usize("NV_QWEN36_MSWEEP_ROUNDS", 40);
    let with_m1 = std::env::var("NV_QWEN36_MSWEEP_M1").as_deref() == Ok("1");
    let graph_off = std::env::var("NV_GRAPH_OFF").as_deref() == Ok("1");

    let dir = snapshot_dir();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw).expect("config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");

    let depth = DEPTH_256_MATCHES_THE_ECON_AND_RECORD_BASIS;
    let max_rows = rows_sweep.iter().copied().max().unwrap();
    let slots = depth
        + 16
        + M1_DECODE_STEPS_64_MATCH_THE_M1AB_MEDIAN_BASIS
        + rounds * rows_sweep.len()
        + max_rows
        + 64;
    let mut engine = GraphedQwen3Moe::new(model, &device, slots).expect("graphed engine");

    let snap = dir
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    eprintln!(
        "[q36-msweep] basis: checkpoint=RedHatAI/Qwen3.6-35B-A3B-NVFP4@{snap} backend=cuda \
         engine=GraphedQwen3Moe grouped-moe device-routed verify, greedy self-consistent \
         chains, bs=1, depth={depth}, rounds={rounds} per rows shape, replay wall excludes \
         the host logits readback (timed separately)"
    );

    let prompt = prompt_of_depth(&tok, depth);
    let stashed = engine
        .take_moe_dispatch_for_host_routed_prefill_then_restore_reinstalling_would_drop_and_rebuild_the_folded_weights_a_torn_down_graph_baked();
    let mut fed = 0usize;
    let mut last = Vec::new();
    while fed < prompt.len() {
        let n = 512.min(prompt.len() - fed);
        last = engine
            .prefill(&prompt[fed..fed + n])
            .unwrap_or_else(|e| panic!("prefill chunk at {fed}: {e:#}"));
        fed += n;
    }
    match stashed {
        Some(hook) => engine.set_moe_dispatch(Some(hook)),
        None => engine.install_grouped_moe().expect("grouped install"),
    }
    engine.set_device_routing(true);
    let anchor0 = argmax(&last);

    for &rows in rows_sweep.iter().rev() {
        assert!(rows >= 2, "rows={rows}: a 1-row verify is a plain decode");
        engine.ensure_verify_lane(rows).expect("verify lane");
    }
    engine.synchronize().expect("post-lane sync");

    if with_m1 {
        let mut cur = anchor0;
        let mut step_ms = Vec::new();
        for i in 0..(16 + M1_DECODE_STEPS_64_MATCH_THE_M1AB_MEDIAN_BASIS) {
            let t0 = std::time::Instant::now();
            engine.forward_decode(cur).expect("decode");
            cur = argmax(&engine.logits_host().expect("logits"));
            if i >= 16 {
                step_ms.push(1000.0 * t0.elapsed().as_secs_f64());
            }
        }
        eprintln!(
            "MSWEEP depth={depth} m=1 decode_ms_med={:.3} capture_active={}",
            median_of(step_ms),
            engine.capture_active()
        );
    }

    let vocab = engine.vocab_size();
    for &rows in &rows_sweep {
        let mut anchor = anchor0;
        let mut chain: Vec<u32> = vec![anchor; rows];
        let mut replay_ms: Vec<f64> = Vec::new();
        let mut readback_ms: Vec<f64> = Vec::new();
        for round in 0..rounds {
            let t0 = std::time::Instant::now();
            engine.forward_verify_chain(&chain).expect("verify chain");
            engine.synchronize().expect("verify sync");
            let dt = 1000.0 * t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            let flat = engine.verify_logits_host().expect("verify logits");
            let rb = 1000.0 * t1.elapsed().as_secs_f64();
            if round > 0 {
                replay_ms.push(dt);
                readback_ms.push(rb);
            }
            let greedy: Vec<u32> = (0..rows)
                .map(|r| argmax(&flat[r * vocab..(r + 1) * vocab]))
                .collect();
            engine.commit_verify_consumed(1).expect("commit");
            anchor = greedy[0];
            chain.clear();
            chain.push(anchor);
            chain.extend_from_slice(&greedy[..rows - 1]);
        }
        if !graph_off {
            assert!(
                engine.capture_active(),
                "rows={rows}: the verify fell back to uncaptured; the captured replay is the \
                 subject of this sweep -- diagnose the stderr blocker above"
            );
        }
        if let Some(d) = engine.moe_dispatch_ref() {
            d.dump_phase_ladder_after_replay_when_nv_q36_moe_events_1();
        }
        engine
            .underlying()
            .dump_trunk_phase_ladder_after_replay_when_nv_q36_trunk_events_1();
        eprintln!(
            "MSWEEP depth={depth} m={rows} verify_replay_ms_med={:.3} readback_ms_med={:.3} \
             rounds={} capture_active={}",
            median_of(replay_ms),
            median_of(readback_ms),
            rounds - 1,
            engine.capture_active()
        );
    }
}
