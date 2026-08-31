#![cfg(feature = "cuda")]

mod ctx_timing_common;

use candle_core::{DType, Device, Tensor};
use nv_models::laguna::{Laguna, LagunaConfig};

#[path = "laguna_prompts.rs"]
mod prompts;

const GREEDY_CHAIN_STEPS_128_IS_THE_QUALITY_PROTOCOL_MINIMUM_FOR_A_QUANTIZATION_CLASS_ARM: usize =
    128;
const PREFILL_CHUNK_256_MATCHES_THE_LAGUNA_RING_SLIDING_CAP: usize = 256;

fn arm_label_from_the_resolved_value_of_every_numerics_changing_gate() -> String {
    ctx_timing_common::arm_label_from_the_resolved_value_of_every_gate_accessor_because_a_presence_label_lies_after_a_default_flip(
        &ctx_timing_common::LAGUNA_DECODE_NUMERICS_GATES_EVERY_ONE_APPEARS_RESOLVED_IN_THE_ARM_LABEL,
    )
}

fn chain_steps_from_env_default_128() -> usize {
    match std::env::var("NV_LAGUNA_CHAIN_STEPS") {
        Ok(v) => v.parse::<usize>().expect("NV_LAGUNA_CHAIN_STEPS"),
        Err(_) => GREEDY_CHAIN_STEPS_128_IS_THE_QUALITY_PROTOCOL_MINIMUM_FOR_A_QUANTIZATION_CLASS_ARM,
    }
}

fn decode_through_the_step_graph_instead_of_the_eager_path() -> bool {
    std::env::var("NV_LAGUNA_CHAIN_GRAPH")
        .map(|v| v != "0")
        .unwrap_or(false)
}

fn argmax_with_margin(row: &[f32]) -> (u32, f32) {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    let mut second_v = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_v {
            second_v = best_v;
            best_v = v;
            best = i;
        } else if v > second_v {
            second_v = v;
        }
    }
    (best as u32, best_v - second_v)
}

#[test]
#[ignore = "loads the Laguna-XS-2.1 NVFP4 checkpoint + prompt pack; set NV_LAGUNA_CTX_TEST=1 -- prints a 128-token greedy argmax chain (token + top1-top2 margin per step) over a real chat-templated prompt; NV_LAGUNA_CHAIN_STEPS overrides the length and NV_LAGUNA_CHAIN_GRAPH=1 decodes through LagunaStepGraph instead of the eager path, so the same probe covers arm-vs-basis and graph-vs-eager; run once per arm set and diff the CHAIN lines for the argmax-class parity bar"]
fn laguna_greedy_chain_over_a_real_templated_prompt_prints_tokens_and_margins_for_arm_diffing() {
    if std::env::var("NV_LAGUNA_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_LAGUNA_CTX_TEST != 1");
        return;
    }
    let ev = prompts::LagunaEval::open().expect("laguna snapshot + prompt pack");
    let dir = ev.dir.clone();
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = LagunaConfig::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = nv_weights::QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let device = Device::new_cuda(0).expect("cuda");
    let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = std::sync::Arc::new(
        Laguna::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model"),
    );
    drop(weights);

    let ids: Vec<u32> = ev
        .builder()
        .expect("pack scaffold")
        .ids("Summarize the tradeoffs between static and dynamic linking in three sentences.")
        .expect("templated prompt ids");
    let arm = arm_label_from_the_resolved_value_of_every_numerics_changing_gate();
    let graphed = decode_through_the_step_graph_instead_of_the_eager_path();
    let steps = chain_steps_from_env_default_128();
    eprintln!(
        "[fold_parity] arm {arm} decode={} prompt_ids={} steps={steps}",
        if graphed { "step_graph" } else { "eager" },
        ids.len()
    );
    let mut cache = model
        .new_kv_cache(ids.len() + steps + 8)
        .expect("kv cache");
    let mut pos = 0usize;
    let mut last_logits: Option<Vec<f32>> = None;
    while pos < ids.len() {
        let n = PREFILL_CHUNK_256_MATCHES_THE_LAGUNA_RING_SLIDING_CAP.min(ids.len() - pos);
        let chunk_ids: Vec<u32> = ids[pos..pos + n].to_vec();
        let tokens = Tensor::from_vec(chunk_ids, (1usize, n), &device).expect("tokens");
        let positions = Tensor::from_vec(
            (pos as i32..(pos + n) as i32).collect::<Vec<_>>(),
            n,
            &device,
        )
        .expect("positions");
        let logits = model
            .forward_with_cache(&tokens, &positions, &mut cache)
            .unwrap_or_else(|e| panic!("prefill chunk at pos {pos}: {e:#}"));
        let flat = logits
            .to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("host");
        let vocab = flat.len() / n;
        last_logits = Some(flat[(n - 1) * vocab..n * vocab].to_vec());
        pos += n;
    }
    let mut row = last_logits.expect("prefill produced logits");
    let mut chain: Vec<u32> = Vec::with_capacity(steps);
    let (mut graph, mut eager_cache) = if graphed {
        (
            Some(
                nv_models::laguna_step_graph::LagunaStepGraph::new(
                    std::sync::Arc::clone(&model),
                    cache,
                )
                .expect("step graph over the eagerly prefilled ring cache"),
            ),
            None,
        )
    } else {
        (None, Some(cache))
    };
    for step in 0..steps {
        let (tok, margin) = argmax_with_margin(&row);
        eprintln!("[fold_parity] step={step} token={tok} margin={margin:.6}");
        chain.push(tok);
        row = match graph.as_mut() {
            Some(sg) => {
                sg.step(tok)
                    .unwrap_or_else(|e| panic!("graphed decode at pos {pos}: {e:#}"));
                sg.logits_host()
                    .unwrap_or_else(|e| panic!("graphed logits at pos {pos}: {e:#}"))
            }
            None => {
                let tokens = Tensor::from_vec(vec![tok], (1usize, 1usize), &device).expect("token");
                let positions =
                    Tensor::from_vec(vec![pos as i32], 1usize, &device).expect("position");
                let logits = model
                    .forward_with_cache(
                        &tokens,
                        &positions,
                        eager_cache.as_mut().expect("eager cache"),
                    )
                    .unwrap_or_else(|e| panic!("decode at pos {pos}: {e:#}"));
                logits
                    .to_dtype(DType::F32)
                    .expect("f32")
                    .flatten_all()
                    .expect("flatten")
                    .to_vec1::<f32>()
                    .expect("host")
            }
        };
        pos += 1;
    }
    let rendered: Vec<String> = chain.iter().map(|t| t.to_string()).collect();
    eprintln!("[fold_parity] CHAIN arm={arm} {}", rendered.join(","));
}
