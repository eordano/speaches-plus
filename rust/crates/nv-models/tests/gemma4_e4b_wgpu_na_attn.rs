#![cfg(feature = "wgpu")]

mod common;
use common::e4b_qat_w4a16_ct_snapshot_dir as e4b_snapshot_dir;
use nv_models::gemma4::Gemma4Config;
use nv_models::gemma4_e4b_wgpu::{e4b_host_weights_from_loader, Gemma4E4bWgpu};

fn ctx_or_skip() -> Option<&'static nv_kernels::wgpu_backend::WgpuContext> {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("skip: no wgpu adapter: {e}");
            None
        }
    }
}

struct ArmResult {
    seqs: Vec<Vec<u32>>,
    ms_per_tok: Vec<Vec<f64>>,
    profile: Vec<(String, u64, f64)>,
    chunk_m: usize,
    logits_bits: Vec<u32>,
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    attn: bool,
    prefill_m: usize,
    config: &Gemma4Config,
    host: &nv_models::gemma4_e4b_wgpu::E4bHostWeights,
    prompts: &[(String, Vec<u32>)],
    n_new: usize,
    max_seq: usize,
    reps: usize,
) -> ArmResult {
    std::env::set_var("NV_WGPU_NA", "1");
    std::env::set_var("NV_WGPU_NA_ATTN", if attn { "1" } else { "0" });
    std::env::set_var("NV_WGPU_PREFILL_M", prefill_m.to_string());
    let t0 = std::time::Instant::now();
    let mut m = Gemma4E4bWgpu::new(config.clone(), host, max_seq).unwrap();
    eprintln!(
        "[arm attn={attn} m={prefill_m}] model built in {:.1}s, prefill chunk {}, {} prefill passes",
        t0.elapsed().as_secs_f64(),
        m.prefill_chunk_len(),
        m.prefill_pass_count()
    );
    let cm = m.prefill_chunk_len();
    assert_eq!(cm, prefill_m, "NV_WGPU_PREFILL_M did not take effect");
    m.prefill_chunk(&vec![5u32; cm]).unwrap();
    m.sync().unwrap();
    m.reset();

    let mut seqs = Vec::new();
    let mut ms_per_tok = Vec::new();
    for (label, prompt) in prompts {
        let mut rep_seqs: Vec<Vec<u32>> = Vec::new();
        let mut rep_ms = Vec::new();
        for _ in 0..reps {
            m.reset();
            let t = std::time::Instant::now();
            let mut next = m.prefill_prompt(prompt).unwrap();
            m.sync().unwrap();
            let wall = t.elapsed().as_secs_f64();
            let mut seq = Vec::with_capacity(n_new);
            for _ in 0..n_new {
                seq.push(next);
                next = m.decode_step(next).unwrap();
            }
            rep_ms.push(1000.0 * wall / prompt.len() as f64);
            rep_seqs.push(seq);
        }
        assert_eq!(
            rep_seqs[0],
            rep_seqs[reps - 1],
            "[arm attn={attn} m={prefill_m}] {label}: reps diverged"
        );
        eprintln!(
            "[arm attn={attn} m={prefill_m}] {label} ({} tok): prefill {} ms/prompt-tok",
            prompt.len(),
            rep_ms
                .iter()
                .map(|v| format!("{v:.2}"))
                .collect::<Vec<_>>()
                .join(" / ")
        );
        seqs.push(rep_seqs.into_iter().next().unwrap());
        ms_per_tok.push(rep_ms);
    }

    m.reset();
    let next = m.prefill_prompt(&prompts[0].1).unwrap();
    let (_, logits) = m.decode_step_logits(next).unwrap();
    let logits_bits: Vec<u32> = logits.iter().map(|v| v.to_bits()).collect();

    m.reset();
    let prof_m = prefill_m.min(64);
    let per_pass = if prof_m == cm {
        m.prefill_chunk_profiled(&vec![5u32; cm])
    } else {
        std::env::set_var("NV_WGPU_PREFILL_M", prof_m.to_string());
        let mut pm = Gemma4E4bWgpu::new(config.clone(), host, max_seq).unwrap();
        std::env::set_var("NV_WGPU_PREFILL_M", prefill_m.to_string());
        pm.prefill_chunk_profiled(&vec![5u32; prof_m])
    };
    let per_pass = match per_pass {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[arm attn={attn} m={prefill_m}] profiled chunk skipped: {e}");
            Vec::new()
        }
    };
    let mut agg: std::collections::BTreeMap<String, (u64, f64)> = Default::default();
    for (label, ns) in per_pass {
        let key = label
            .split_whitespace()
            .next()
            .unwrap_or("other")
            .to_string();
        let e = agg.entry(key).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += ns;
    }
    let mut profile: Vec<(String, u64, f64)> =
        agg.into_iter().map(|(k, (c, ns))| (k, c, ns)).collect();
    profile.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    let total: f64 = profile.iter().map(|r| r.2).sum();
    eprintln!(
        "[arm attn={attn} m={prefill_m}] profiled chunk (m={prof_m}): {:.3} ms GPU total (pass-split inflated)",
        total / 1e6
    );
    for (label, count, ns) in &profile {
        eprintln!(
            "[arm attn={attn} m={prefill_m}]   {label:<24} n={count:<4} {:>9.3} ms  {:>5.1}%",
            ns / 1e6,
            ns / total * 100.0
        );
    }
    ArmResult {
        seqs,
        ms_per_tok,
        profile,
        chunk_m: cm,
        logits_bits,
    }
}

#[test]
#[ignore]
fn real_e4b_na_attn_parity_and_speed() {
    if std::env::var("NV_E4B_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping: set NV_E4B_WGPU_TEST=1 to run");
        return;
    }
    let Some(_ctx) = ctx_or_skip() else { return };
    let dir = e4b_snapshot_dir();
    eprintln!("E4B checkpoint: {}", dir.display());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let host = e4b_host_weights_from_loader(&config, &loader).unwrap();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let bos = tok.token_to_id("<bos>").expect("<bos>");
    let sentence = "The history of navigation at sea spans thousands of years, from Polynesian wayfinding by stars and swells to the magnetic compass, the marine chronometer, and satellite positioning. ";
    let mk_prompt = |target: usize| -> Vec<u32> {
        let mut ids = vec![bos];
        while ids.len() < target {
            ids.extend(tok.encode(sentence, false).unwrap().get_ids());
        }
        ids.truncate(target);
        ids
    };
    let prompts: Vec<(String, Vec<u32>)> = vec![
        ("pp128".to_string(), mk_prompt(128)),
        ("pp512".to_string(), mk_prompt(512)),
    ];
    let n_new = 24;
    let max_seq = 1024;
    let reps: usize = std::env::var("NV_NA_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let prefill_m: usize = std::env::var("NV_NA_ATTN_M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);

    let base = run_arm(
        false, prefill_m, &config, &host, &prompts, n_new, max_seq, reps,
    );
    let attn = run_arm(
        true, prefill_m, &config, &host, &prompts, n_new, max_seq, reps,
    );
    std::env::remove_var("NV_WGPU_NA_ATTN");
    std::env::remove_var("NV_WGPU_NA");
    std::env::remove_var("NV_WGPU_PREFILL_M");

    let attn_share: f64 = attn
        .profile
        .iter()
        .filter(|(l, _, _)| l.starts_with("na_attn"))
        .map(|r| r.2)
        .sum();
    let flash_gone = attn
        .profile
        .iter()
        .all(|(l, _, _)| !l.starts_with("flash_stage"));
    assert!(
        attn_share > 0.0,
        "NV_WGPU_NA_ATTN=1 arm shows no na_attn dispatches in the profiled chunk"
    );
    assert!(
        flash_gone,
        "NV_WGPU_NA_ATTN=1 arm still shows flash_stage passes in the prefill chunk"
    );
    assert_eq!(base.chunk_m, attn.chunk_m);

    let mut bits_exact = true;
    if base.logits_bits != attn.logits_bits {
        bits_exact = false;
        let n_diff = base
            .logits_bits
            .iter()
            .zip(&attn.logits_bits)
            .filter(|(a, b)| a != b)
            .count();
        eprintln!(
            "post-prefill logits bits differ in {n_diff}/{} lanes (expected: softmax reorder)",
            base.logits_bits.len()
        );
    }
    eprintln!(
        "post-prefill logits bit-exact: {}",
        if bits_exact { "YES" } else { "NO" }
    );

    for (i, (label, prompt)) in prompts.iter().enumerate() {
        eprintln!(
            "[compare] {label} ({} tok): attn-off {} vs attn-on {} ms/prompt-tok",
            prompt.len(),
            base.ms_per_tok[i]
                .iter()
                .map(|v| format!("{v:.2}"))
                .collect::<Vec<_>>()
                .join(" / "),
            attn.ms_per_tok[i]
                .iter()
                .map(|v| format!("{v:.2}"))
                .collect::<Vec<_>>()
                .join(" / ")
        );
        assert_eq!(
            base.seqs[i], attn.seqs[i],
            "[{label}] greedy continuation diverged between NV_WGPU_NA_ATTN=0 and =1"
        );
    }
}
