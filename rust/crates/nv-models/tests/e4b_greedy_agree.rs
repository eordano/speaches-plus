#![cfg(feature = "cuda")]

mod hub_snapshot;
mod official_template;

use candle_core::Device;
use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_e4b::Gemma4E4b;
use nv_weights::WeightLoader;
use official_template::OfficialTemplate;
use std::path::Path;
use tokenizers::Tokenizer;

const PROJ: [&str; 7] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

const PROMPTS: [&str; 6] = [
    "What is the capital of France, and what river runs through it?",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "If a train travels 60 km in 45 minutes, what is its average speed in km/h? Think step by step.",
    "Summarize in one sentence why the sky is blue.",
    "Write a haiku about mountains.",
    "List three uses for a paperclip.",
];

const STEPS: usize = 64;
const EOS: [u32; 2] = [1, 106];

fn weight_basis(names: &[String]) -> (&'static str, usize, usize) {
    let mut dense = 0usize;
    let mut packed = 0usize;
    for n in names {
        if !n.starts_with("model.language_model.layers.") {
            continue;
        }
        let Some((module, leaf)) = n.rsplit_once('.') else {
            continue;
        };
        let Some((_, kind)) = module.rsplit_once('.') else {
            continue;
        };
        if !PROJ.contains(&kind) {
            continue;
        }
        match leaf {
            "weight" => dense += 1,
            "weight_packed" => packed += 1,
            _ => {}
        }
    }
    let label = match (dense, packed) {
        (0, 0) => "empty",
        (0, _) => "pack-quantized",
        (_, 0) => "dense-bf16",
        _ => "mixed",
    };
    (label, dense, packed)
}

fn templated(tpl: &OfficialTemplate, user: &str) -> String {
    let rendered = tpl.render_user(user);
    assert!(
        rendered.starts_with("<bos>"),
        "official template must emit BOS itself: {rendered:?}"
    );
    rendered
}

fn log_softmax_at(logits: &[f32], target: usize) -> f64 {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let s: f64 = logits.iter().map(|&v| (v as f64 - m).exp()).sum();
    logits[target] as f64 - m - s.ln()
}

fn top2(logits: &[f32]) -> ((usize, f32), (usize, f32)) {
    let mut b1 = (0usize, f32::NEG_INFINITY);
    let mut b2 = (0usize, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > b1.1 {
            b2 = b1;
            b1 = (i, v);
        } else if v > b2.1 {
            b2 = (i, v);
        }
    }
    (b1, b2)
}

fn degeneracy(ids: &[u32]) -> (f64, usize) {
    let distinct = ids.iter().collect::<std::collections::HashSet<_>>().len();
    let mut longest = 0usize;
    let mut run = 0usize;
    let mut prev = None;
    for &t in ids {
        run = if prev == Some(t) { run + 1 } else { 1 };
        longest = longest.max(run);
        prev = Some(t);
    }
    (distinct as f64 / ids.len().max(1) as f64, longest)
}

fn load(dir: &Path, device: &Device) -> (Gemma4E4b, &'static str) {
    let cfg = Gemma4Config::from_hf_json_file(&dir.join("config.json")).expect("config");
    assert!(
        cfg.has_per_layer_embeddings(),
        "not an E4B checkpoint: {dir:?}"
    );
    let weights = WeightLoader::open_dir(dir, device).expect("weights");
    let (basis, dense, packed) = weight_basis(&weights.names());
    eprintln!("[load] {dir:?} basis={basis} (dense={dense} packed={packed})");
    let model = Gemma4E4b::from_loader(cfg, &weights, device).expect("load Gemma4E4b");
    (model, basis)
}

#[test]
#[ignore]
fn e4b_cross_checkpoint_greedy_agreement() {
    let bf16_dir = hub_snapshot::dir_from_env_or_hub(
        "GEMMA4_E4B_BF16_DIR",
        "google/gemma-4-E4B-it",
        &["config.json", "*.safetensors"],
    );
    let qat_dir = hub_snapshot::dir_from_env_or_hub(
        "GEMMA4_E4B_QAT_DIR",
        "google/gemma-4-E4B-it-qat-w4a16-ct",
        &["config.json", "*.safetensors"],
    );
    let (Some(bf16_dir), Some(qat_dir)) = (bf16_dir, qat_dir) else {
        hub_snapshot::precondition_absent(
            "e4b_cross_checkpoint_greedy_agreement",
            "both E4B arms are required: the dense bf16 google/gemma-4-E4B-it AND the \
             pack-quantized google/gemma-4-E4B-it-qat-w4a16-ct (the latter is NOT cached \
             on this box despite CLAUDE.md listing it)",
            "cache google/gemma-4-E4B-it-qat-w4a16-ct into /tank (NOT zroot, 91% CAP), or \
             set GEMMA4_E4B_BF16_DIR and GEMMA4_E4B_QAT_DIR",
        );
        return;
    };
    if std::env::var("NV_E4B_AGREE_TEST").ok().as_deref() != Some("1") {
        hub_snapshot::precondition_absent(
            "e4b_cross_checkpoint_greedy_agreement",
            "NV_E4B_AGREE_TEST != 1",
            "set NV_E4B_AGREE_TEST=1",
        );
        return;
    }
    let device = Device::new_cuda(0).expect("cuda device required");

    let tok = Tokenizer::from_file(Path::new(&bf16_dir).join("tokenizer.json")).expect("tokenizer");
    let tpl = OfficialTemplate::load(Path::new(&bf16_dir));
    let prompt_ids: Vec<Vec<u32>> = PROMPTS
        .iter()
        .map(|p| {
            tok.encode(templated(&tpl, p).as_str(), false)
                .expect("encode")
                .get_ids()
                .to_vec()
        })
        .collect();

    let mut ref_streams: Vec<Vec<u32>> = Vec::new();
    {
        let (model, basis) = load(Path::new(&bf16_dir), &device);
        assert_eq!(
            basis, "dense-bf16",
            "GEMMA4_E4B_BF16_DIR is not the dense checkpoint"
        );
        for (pi, ids) in prompt_ids.iter().enumerate() {
            let cont = model.generate(ids, STEPS, &EOS).expect("bf16 generate");
            assert!(
                cont.len() >= 8,
                "prompt {pi} bf16 continuation too short ({}) to score",
                cont.len()
            );
            let (dr, run) = degeneracy(&cont);
            eprintln!(
                "[bf16 p{pi}] {} tok, distinct-ratio {dr:.2}, longest-run {run}: {:?}",
                cont.len(),
                tok.decode(&cont, false).unwrap_or_default()
            );
            ref_streams.push(cont);
        }
    }

    let (model, basis) = load(Path::new(&qat_dir), &device);
    assert_eq!(
        basis, "pack-quantized",
        "GEMMA4_E4B_QAT_DIR is not the w4a16 checkpoint"
    );
    let vocab = model.config().vocab_size;

    let mut total_agree = 0usize;
    let mut total_scored = 0usize;
    let mut sum_lp = 0.0f64;
    for (pi, (ids, cont)) in prompt_ids.iter().zip(&ref_streams).enumerate() {
        let own = model.generate(ids, STEPS, &EOS).expect("qat generate");
        let (dr, run) = degeneracy(&own);
        eprintln!(
            "[qat  p{pi}] {} tok, distinct-ratio {dr:.2}, longest-run {run}: {:?}",
            own.len(),
            tok.decode(&own, false).unwrap_or_default()
        );

        let mut full = ids.clone();
        full.extend_from_slice(cont);
        let mut agree = 0usize;
        let mut first_div: Option<usize> = None;
        let mut div_margins: Vec<f32> = Vec::new();
        let mut lp = 0.0f64;
        for i in ids.len()..full.len() {
            let tr = model.trace(&full[..i]).expect("trace");
            assert_eq!(tr.logits_last.len(), vocab);
            let want = full[i];
            lp += log_softmax_at(&tr.logits_last, want as usize);
            let ((got, top), (_, second)) = top2(&tr.logits_last);
            if got as u32 == want {
                agree += 1;
            } else {
                first_div.get_or_insert(i - ids.len());

                div_margins.push(top - tr.logits_last[want as usize].min(second));
            }
        }
        let scored = full.len() - ids.len();
        total_agree += agree;
        total_scored += scored;
        sum_lp += lp;
        eprintln!(
            "[agree p{pi}] {agree}/{scored} teacher-forced on bf16 stream, first divergence {first_div:?}, \
             mean logprob of bf16 tokens {:.4}, divergence margins {:?}",
            lp / scored as f64,
            div_margins.iter().map(|m| (m * 100.0).round() / 100.0).collect::<Vec<_>>()
        );
    }

    let rate = total_agree as f64 / total_scored as f64;
    eprintln!(
        "AGREE-SUMMARY {total_agree}/{total_scored} = {rate:.4} argmax agreement of w4a16 \
         teacher-forced on bf16 greedy streams over {} external chat prompts; mean logprob of \
         bf16 tokens under w4a16 {:.4}. No threshold asserted: this is calibration evidence, \
         not a pass/fail gate.",
        PROMPTS.len(),
        sum_lp / total_scored as f64
    );
    assert!(
        total_scored >= 48,
        "too few scored positions for evidence: {total_scored}"
    );
}
