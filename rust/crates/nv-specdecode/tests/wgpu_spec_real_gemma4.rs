#![cfg(feature = "wgpu")]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[path = "../../nv-models/tests/official_template/mod.rs"]
mod official_template;

use anyhow::{anyhow, Context, Result};
use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_e4b_wgpu::{e4b_host_weights_from_loader, Gemma4E4bWgpu};
use nv_models::gemma4_wgpu::{host_weights_from_loader, Gemma4Wgpu};
use nv_specdecode::wgpu_spec::{
    ChainDrafter, LockstepChainSpec, ModelDrafter, PromptLookupDrafter, RealSpecStats, StepDecoder,
};
use serde_json::{json, Value};

fn env_flag(k: &str) -> bool {
    std::env::var(k).ok().as_deref() == Some("1")
}

fn env_usize(k: &str, default: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn hub_snapshot(repo: &str) -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME")?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo)
        .join("snapshots");
    let mut cands: Vec<PathBuf> = std::fs::read_dir(&base)
        .with_context(|| format!("read {}", base.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").exists() && p.join("chat_template.jinja").exists())
        .collect();
    cands.sort();
    let refs_main = PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub")
        .join(repo)
        .join("refs/main");
    if let Ok(sha) = std::fs::read_to_string(&refs_main) {
        let want = base.join(sha.trim());
        if cands.contains(&want) {
            return Ok(want);
        }
    }
    cands
        .pop()
        .ok_or_else(|| anyhow!("no usable snapshot under {}", base.display()))
}

fn model_dir(env_key: &str, repo: &str) -> Result<PathBuf> {
    match std::env::var(env_key) {
        Ok(d) => Ok(PathBuf::from(d)),
        Err(_) => hub_snapshot(repo),
    }
}

use official_template::OfficialTemplate;

fn eos_ids(dir: &Path) -> Result<Vec<u32>> {
    let raw = std::fs::read_to_string(dir.join("generation_config.json"))
        .context("read generation_config.json")?;
    let v: Value = serde_json::from_str(&raw)?;
    let e = v
        .get("eos_token_id")
        .ok_or_else(|| anyhow!("generation_config.json has no eos_token_id"))?;
    Ok(match e {
        Value::Number(n) => vec![n.as_u64().unwrap() as u32],
        Value::Array(a) => a
            .iter()
            .filter_map(|x| x.as_u64())
            .map(|x| x as u32)
            .collect(),
        other => anyhow::bail!("unexpected eos_token_id {other}"),
    })
}

fn distinct_ratio(ids: &[u32]) -> f64 {
    if ids.is_empty() {
        return 0.0;
    }
    let mut set: HashMap<u32, ()> = HashMap::new();
    for &t in ids {
        set.insert(t, ());
    }
    set.len() as f64 / ids.len() as f64
}

fn max_consecutive_ngram_repeat(ids: &[u32], n: usize) -> usize {
    if ids.len() < 2 * n {
        return 1;
    }
    let mut best = 1usize;
    for start in 0..=ids.len() - n {
        let pat = &ids[start..start + n];
        let mut reps = 1usize;
        let mut at = start + n;
        while at + n <= ids.len() && &ids[at..at + n] == pat {
            reps += 1;
            at += n;
        }
        best = best.max(reps);
    }
    best
}

fn assert_not_degenerate(label: &str, ids: &[u32], text: &str) {
    assert!(!ids.is_empty(), "{label}: no tokens generated");
    assert!(
        !text.trim().is_empty(),
        "{label}: generated text is empty/whitespace"
    );
    let dr = distinct_ratio(ids);
    let r4 = max_consecutive_ngram_repeat(ids, 4);
    eprintln!("[{label}] distinct_ratio={dr:.3} max_consecutive_4gram_repeat={r4}");
    assert!(
        dr > 0.30,
        "{label}: degenerate output, distinct token ratio {dr:.3} <= 0.30"
    );
    assert!(
        r4 <= 4,
        "{label}: degenerate output, a 4-gram repeats {r4} times back to back"
    );
}

fn chat_prompt_ids(
    dir: &Path,
    tok: &tokenizers::Tokenizer,
    user: &str,
) -> Result<(Vec<u32>, String)> {
    let tmpl = OfficialTemplate::try_load(dir).map_err(|e| anyhow!(e))?;
    let msgs = json!([{ "role": "user", "content": user }]);
    let rendered = tmpl.try_render(&msgs, true).map_err(|e| anyhow!(e))?;
    let ids = tok
        .encode(rendered.as_str(), false)
        .map_err(|e| anyhow!("encode: {e}"))?
        .get_ids()
        .to_vec();
    Ok((ids, rendered))
}

fn report(
    label: &str,
    stats: &RealSpecStats,
    tok: &tokenizers::Tokenizer,
    drafter: &str,
) -> String {
    let text = tok.decode(&stats.emitted, true).unwrap_or_default();
    eprintln!("[{label}] drafter={drafter}");
    eprintln!("[{label}] {}", stats.summary());
    eprintln!("[{label}] ids: {:?}", stats.emitted);
    eprintln!("[{label}] text: {text:?}");
    text
}

fn load_e4b(dir: &Path, max_seq: usize) -> Result<Gemma4E4bWgpu> {
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json"))?;
    let loader = nv_weights::WeightLoader::open_dir(dir, &candle_core::Device::Cpu)?;
    let host = e4b_host_weights_from_loader(&config, &loader)?;
    Gemma4E4bWgpu::new(config, &host, max_seq)
}

fn load_g31(dir: &Path, max_seq: usize) -> Result<Gemma4Wgpu> {
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json"))?;
    let loader = nv_weights::WeightLoader::open_dir(dir, &candle_core::Device::Cpu)?;
    let host = host_weights_from_loader(&config, &loader)?;
    Gemma4Wgpu::new(config, &host, max_seq)
}

fn run_case<V: StepDecoder, D: ChainDrafter>(
    label: &str,
    spec: &mut LockstepChainSpec<V, D>,
    tok: &tokenizers::Tokenizer,
    dir: &Path,
    user: &str,
    max_new: usize,
) -> Result<RealSpecStats> {
    let (ids, rendered) = chat_prompt_ids(dir, tok, user)?;
    eprintln!(
        "[{label}] rendered prompt ({} tok): {rendered:?}",
        ids.len()
    );
    let drafter_label = spec.drafter.label();
    let t0 = std::time::Instant::now();
    let stats = spec.generate(&ids, max_new)?;
    let dt = t0.elapsed().as_secs_f64();
    let text = report(label, &stats, tok, &drafter_label);
    eprintln!(
        "[{label}] CONTAMINATED wall clock (co-tenanted GPU): {:.3}s for {} tokens over {} verifier steps",
        dt,
        stats.emitted.len(),
        stats.verifier_steps
    );
    assert_not_degenerate(label, &stats.emitted, &text);
    let greedy = spec.greedy(&ids, stats.emitted.len())?;
    assert_eq!(
        stats.emitted, greedy,
        "{label}: speculative stream must be bit-identical to the verifier's own greedy stream"
    );
    let r = stats.acceptance_rate();
    assert!(
        (0.0..=1.0).contains(&r),
        "{label}: acceptance rate {r} out of range"
    );
    Ok(stats)
}

fn require_model_dir(test: &str, env_key: &str, repo: &str) -> Option<PathBuf> {
    match model_dir(env_key, repo) {
        Ok(d) if d.join("chat_template.jinja").exists() => Some(d),
        Ok(d) => {
            loud(test, &format!("no chat_template.jinja in {}", d.display()));
            None
        }
        Err(e) => {
            loud(test, &format!("no {repo} snapshot: {e}"));
            None
        }
    }
}

fn loud(test: &str, msg: &str) {
    if std::env::var("NV_WGPU_SPEC_ALLOW_SKIP").as_deref() == Ok("1") {
        eprintln!("SKIP (NV_WGPU_SPEC_ALLOW_SKIP=1): {test}: {msg}. This is a SKIP, not a pass.");
        return;
    }
    panic!(
        "{test}: {msg}. Both gemma-4 snapshots this file names are cached on this box with \
         chat_template.jinja and tokenizer.json, so a miss means the hub moved. Set \
         NV_WGPU_SPEC_ALLOW_SKIP=1 to skip on purpose."
    );
}

fn require_flag(test: &str, var: &str) {
    if !env_flag(var) {
        panic!(
            "{test}: {var} != 1. This test is already #[ignore]d, so naming it IS the opt-in; the \
             old early return printed a pass in 0.00s having loaded no weights. Both checkpoints \
             are cached. Set {var}=1 to run it."
        );
    }
}

#[test]
#[ignore]
fn wgpu_spec_real_gemma4_prompt_lookup() {
    require_flag(
        "wgpu_spec_real_gemma4_prompt_lookup",
        "NV_WGPU_SPEC_REAL_TEST",
    );
    let which = std::env::var("NV_WGPU_SPEC_VERIFIER").unwrap_or_else(|_| "e4b".into());
    let dir = match which.as_str() {
        "g31" => model_dir("NV_GEMMA4_DIR", "models--nvidia--Gemma-4-31B-IT-NVFP4").unwrap(),
        _ => model_dir("NV_E4B_DIR", "models--google--gemma-4-E4B-it").unwrap(),
    };
    eprintln!("verifier snapshot: {}", dir.display());
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let eos = eos_ids(&dir).unwrap();
    eprintln!("eos ids: {eos:?}");
    let max_seq = env_usize("NV_WGPU_SPEC_MAXSEQ", 1024);
    let max_new = env_usize("NV_WGPU_SPEC_NEW", 96);
    let k = env_usize("NV_WGPU_SPEC_K", 4);
    let min_match = env_usize("NV_WGPU_SPEC_MINMATCH", 2);

    let t_load = std::time::Instant::now();
    let generic = std::env::var("NV_WGPU_SPEC_PROMPT").unwrap_or_else(|_| {
        "Name three primary colors and say why they are called primary.".into()
    });
    let copy_heavy = std::env::var("NV_WGPU_SPEC_PROMPT2").unwrap_or_else(|_| {
        "Here is a list: alpha, bravo, charlie, delta, echo. \
         Repeat the list back exactly as written, then repeat it once more."
            .into()
    });

    let mut summaries: Vec<String> = Vec::new();
    if which == "g31" {
        let v = load_g31(&dir, max_seq).unwrap();
        eprintln!(
            "loaded {} in {:.1}s",
            v.label(),
            t_load.elapsed().as_secs_f64()
        );
        let mut spec =
            LockstepChainSpec::new(v, PromptLookupDrafter::new(min_match), k, eos.clone()).unwrap();
        for (label, user) in [("g31-generic", &generic), ("g31-copy", &copy_heavy)] {
            let s = run_case(label, &mut spec, &tok, &dir, user, max_new).unwrap();
            summaries.push(format!("{label}: {}", s.summary()));
        }
    } else {
        let v = load_e4b(&dir, max_seq).unwrap();
        eprintln!(
            "loaded {} in {:.1}s",
            v.label(),
            t_load.elapsed().as_secs_f64()
        );
        let mut spec =
            LockstepChainSpec::new(v, PromptLookupDrafter::new(min_match), k, eos.clone()).unwrap();
        for (label, user) in [("e4b-generic", &generic), ("e4b-copy", &copy_heavy)] {
            let s = run_case(label, &mut spec, &tok, &dir, user, max_new).unwrap();
            summaries.push(format!("{label}: {}", s.summary()));
        }
    }
    for s in &summaries {
        eprintln!("SUMMARY {s}");
    }
}

#[test]
#[ignore]
fn wgpu_spec_real_gemma4_pair_e4b_drafts_31b() {
    require_flag(
        "wgpu_spec_real_gemma4_pair_e4b_drafts_31b",
        "NV_WGPU_SPEC_REAL_PAIR",
    );
    let vdir = model_dir("NV_GEMMA4_DIR", "models--nvidia--Gemma-4-31B-IT-NVFP4").unwrap();
    let ddir = model_dir("NV_E4B_DIR", "models--google--gemma-4-E4B-it").unwrap();
    eprintln!("verifier: {}", vdir.display());
    eprintln!("drafter:  {}", ddir.display());

    let vtok = tokenizers::Tokenizer::from_file(vdir.join("tokenizer.json")).unwrap();
    let dtok = tokenizers::Tokenizer::from_file(ddir.join("tokenizer.json")).unwrap();
    assert_eq!(
        vtok.get_vocab_size(true),
        dtok.get_vocab_size(true),
        "drafter and verifier must share a tokenizer"
    );

    let max_seq = env_usize("NV_WGPU_SPEC_MAXSEQ", 1024);
    let max_new = env_usize("NV_WGPU_SPEC_NEW", 96);
    let k = env_usize("NV_WGPU_SPEC_K", 4);
    let eos = eos_ids(&vdir).unwrap();

    let verifier = load_g31(&vdir, max_seq).unwrap();
    let drafter = ModelDrafter::new(load_e4b(&ddir, max_seq).unwrap());
    let mut spec = LockstepChainSpec::new(verifier, drafter, k, eos).unwrap();

    let user = std::env::var("NV_WGPU_SPEC_PROMPT").unwrap_or_else(|_| {
        "Name three primary colors and say why they are called primary.".into()
    });
    let s = run_case(
        "pair-e4b-drafts-g31",
        &mut spec,
        &vtok,
        &vdir,
        &user,
        max_new,
    )
    .unwrap();
    eprintln!("SUMMARY pair-e4b-drafts-g31: {}", s.summary());
}

#[test]
fn official_template_renders_gemma4_chat_without_hand_rolling() {
    let Some(dir) = require_model_dir(
        "official_template_renders_gemma4_chat_without_hand_rolling",
        "NV_E4B_DIR",
        "models--google--gemma-4-E4B-it",
    ) else {
        return;
    };
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let (ids, rendered) = chat_prompt_ids(&dir, &tok, "Name three primary colors.").unwrap();
    eprintln!("rendered: {rendered:?}");
    eprintln!("ids: {ids:?}");
    assert!(rendered.starts_with("<bos>"), "template must emit BOS");
    assert!(
        rendered.ends_with("<|turn>model\n"),
        "E4B template must end on the model turn opener, got {rendered:?}"
    );
    assert_eq!(ids.first().copied(), Some(2), "BOS must tokenize to id 2");
    assert_eq!(
        ids.iter().filter(|t| **t == 105).count(),
        2,
        "one <|turn> for user, one for model"
    );
    assert_eq!(eos_ids(&dir).unwrap(), vec![1, 106, 50]);
}

#[test]
fn official_template_31b_ends_with_preclosed_thought_channel() {
    let Some(dir) = require_model_dir(
        "official_template_31b_ends_with_preclosed_thought_channel",
        "NV_GEMMA4_DIR",
        "models--nvidia--Gemma-4-31B-IT-NVFP4",
    ) else {
        return;
    };
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let (ids, rendered) = chat_prompt_ids(&dir, &tok, "Name three primary colors.").unwrap();
    eprintln!("rendered: {rendered:?}");
    eprintln!("ids: {ids:?}");
    assert!(rendered.starts_with("<bos>"));
    assert!(
        rendered.ends_with("<|turn>model\n<|channel>thought\n<channel|>"),
        "served default must end on an already-closed empty thought block, got {rendered:?}"
    );
    assert_eq!(ids.last().copied(), Some(101), "must end on <channel|>");
    assert_eq!(eos_ids(&dir).unwrap(), vec![1, 106, 50]);
}
