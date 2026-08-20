#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::{Gemma4, Gemma4Config};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[path = "../../../tests/common/chat_eval_core.rs"]
mod harness_self_test_no_server_code;
use harness_self_test_no_server_code::{compare, free_running_batch_prefilled, FreeRun, PromptPack, SuiteReport};

fn env_path(k: &str) -> Option<PathBuf> {
    std::env::var(k).ok().map(PathBuf::from)
}

fn last_position_logits(logits: &Tensor, vocab: usize) -> Vec<f32> {
    let flat: Vec<f32> = logits
        .to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flat")
        .to_vec1()
        .expect("vec");
    flat[flat.len() - vocab..].to_vec()
}

#[test]
#[ignore = "loads the real 31B; set NV_W4A4_ARM=<name> NV_W4A4_OUT=<json> (+ pack/weights envs)"]
fn w4a4_arm_emit() {
    let Ok(arm) = std::env::var("NV_W4A4_ARM") else {
        panic!("set NV_W4A4_ARM=<label> to run this arm (it must never silently skip)");
    };
    let out = env_path("NV_W4A4_OUT").expect("set NV_W4A4_OUT=<json path>");
    let pack_p = env_path("NV_CHAT_EVAL_PACK").expect("set NV_CHAT_EVAL_PACK");
    let dir = env_path("NV_CHAT_EVAL_WEIGHTS").expect("set NV_CHAT_EVAL_WEIGHTS");
    let pack = PromptPack::load_for_snapshot(&pack_p, &dir).expect("pack/snapshot mismatch");
    let stops = pack.stop_set();
    let steps = std::env::var("NV_CHAT_EVAL_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(harness_self_test_no_server_code::DEFAULT_MAX_STEPS);
    eprintln!(
        "[w4a4-arm] {arm}: NV_PREFILL_W4A4={:?}, {} prompts, max_steps {steps}",
        std::env::var("NV_PREFILL_W4A4").ok(),
        pack.prompts.len()
    );

    let device = Device::new_cuda(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("read config");
    let cfg = Gemma4Config::from_hf_json_str(&raw).expect("config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let weights = WeightLoader::open_dir(&dir, &device).expect("open weights");
    let model =
        Gemma4::from_loader_quantized(cfg.clone(), &weights, &qcfg, &device).expect("build");
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");

    let longest = pack.prompts.iter().map(|p| p.ids.len()).max().unwrap();
    let max_seq = longest + steps + 16;
    let mut runs: Vec<FreeRun> = Vec::new();
    for p in &pack.prompts {
        let cache = std::cell::RefCell::new(model.new_kv_cache(max_seq).expect("cache"));
        let mut r = free_running_batch_prefilled(
            &arm,
            p,
            &stops,
            steps,
            |ids| {
                let seq = ids.len();
                let tokens = Tensor::from_vec(ids.to_vec(), (1usize, seq), &device)?;
                let pos = Tensor::from_vec((0..seq as i32).collect::<Vec<_>>(), seq, &device)?;
                let logits = model.forward_with_cache(&tokens, &pos, &mut *cache.borrow_mut())?;
                Ok(last_position_logits(&logits, cfg.vocab_size))
            },
            |t| {
                let mut c = cache.borrow_mut();
                let pos_i = nv_models::gemma4::Gemma4Cache::current_len(&*c) as i32;
                let tt = Tensor::from_vec(vec![t], (1usize, 1usize), &device)?;
                let pp = Tensor::from_vec(vec![pos_i], 1usize, &device)?;
                let logits = model.forward_with_cache(&tt, &pp, &mut *c)?;
                Ok(last_position_logits(&logits, cfg.vocab_size))
            },
        )
        .unwrap();
        r.text = tok.decode(&r.tokens, false).unwrap_or_default();
        eprintln!(
            "[w4a4-arm] {} -> {} tokens ({})",
            p.label,
            r.tokens.len(),
            r.reason
        );
        runs.push(r);
    }
    std::fs::write(&out, serde_json::to_vec_pretty(&runs).unwrap()).expect("write dump");
    eprintln!("[w4a4-arm] wrote {} runs to {}", runs.len(), out.display());
}

#[test]
#[ignore = "compares two arm dumps; set NV_W4A4_REF/NV_W4A4_CAND (+ pack/weights envs)"]
fn w4a4_compare() {
    let ref_p = env_path("NV_W4A4_REF").expect("set NV_W4A4_REF");
    let cand_p = env_path("NV_W4A4_CAND").expect("set NV_W4A4_CAND");
    let pack_p = env_path("NV_CHAT_EVAL_PACK").expect("set NV_CHAT_EVAL_PACK");
    let dir = env_path("NV_CHAT_EVAL_WEIGHTS").expect("set NV_CHAT_EVAL_WEIGHTS");
    let pack = PromptPack::load_for_snapshot(&pack_p, &dir).expect("pack/snapshot mismatch");
    let refs: Vec<FreeRun> =
        serde_json::from_slice(&std::fs::read(&ref_p).expect("read ref")).expect("parse ref");
    let cands: Vec<FreeRun> =
        serde_json::from_slice(&std::fs::read(&cand_p).expect("read cand")).expect("parse cand");
    assert_eq!(refs.len(), pack.prompts.len(), "ref dump/pack mismatch");
    assert_eq!(cands.len(), pack.prompts.len(), "cand dump/pack mismatch");

    let mut suite = SuiteReport::new(
        "Gemma4 CUDA prefill: baseline vs NV_PREFILL_W4A4=1 (two-process, batch prefill)",
        &refs[0].arm.clone(),
        &cands[0].arm.clone(),
    );
    for i in 0..pack.prompts.len() {
        assert_eq!(refs[i].prompt_label, pack.prompts[i].label, "ref order");
        assert_eq!(cands[i].prompt_label, pack.prompts[i].label, "cand order");
        suite.push(compare(&pack.prompts[i], &refs[i], &cands[i]));
    }
    suite.validate().unwrap();
    eprintln!("{suite}");
    suite.assert_controls_exact().unwrap();
}
