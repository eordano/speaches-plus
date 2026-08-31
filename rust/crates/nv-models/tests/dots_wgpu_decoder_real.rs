#![cfg(feature = "wgpu")]

mod common;
mod hub_snapshot;

use nv_models::dots_ocr::decoder_wgpu::dots_decoder_dense_config_from_hf_json_str;
use nv_models::dots_ocr::{ASSISTANT_TOKEN_ID, ENDOFUSER_TOKEN_ID, EOS_TOKEN_IDS, USER_TOKEN_ID};
use nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseWgpu;
use nv_weights::WeightLoader;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

const REPO: &str = "mlx-community/dots.ocr-4bit";
const DIR_ENV: &str = "NV_DOTS_MLX_DIR";
const GATE_ENV: &str = "NV_DOTS_WGPU_TEST";

fn dots_mlx_dir() -> PathBuf {
    if let Ok(d) = std::env::var(DIR_ENV) {
        if !d.is_empty() {
            let p = PathBuf::from(&d);
            assert!(p.is_dir(), "{DIR_ENV}={d} is not a directory");
            return p;
        }
    }
    hub_snapshot::snapshot_of(REPO, &["config.json", "tokenizer.json", "*.safetensors"])
        .unwrap_or_else(|| {
            panic!(
                "no complete {REPO} snapshot under the HF hub roots {:?}; set {DIR_ENV} \
                 (this gated suite refuses to vacuously pass)",
                hub_snapshot::hub_roots()
            )
        })
}

fn pld_draft(hist: &[u32], anchor: u32, max: usize) -> Vec<u32> {
    let mut full = Vec::with_capacity(hist.len() + 1);
    full.extend_from_slice(hist);
    full.push(anchor);
    for sl in (1..=3usize).rev() {
        if full.len() < sl + 2 {
            continue;
        }
        let suffix = &full[full.len() - sl..];
        let mut i = full.len() - sl - 1;
        loop {
            if full[i..i + sl] == *suffix {
                let cont = &full[i + sl..];
                if !cont.is_empty() {
                    return cont[..cont.len().min(max)].to_vec();
                }
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }
    }
    Vec::new()
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .map(|v| {
            v.parse::<usize>()
                .unwrap_or_else(|_| panic!("{key}={v} is not a usize"))
        })
        .unwrap_or(default)
}

#[test]
#[ignore = "boots the mlx-community dots.ocr-4bit text decoder on the q3d wgpu bias arm, \
            prefills a text-only user turn and greedily decodes with per-token timing; set \
            NV_DOTS_WGPU_TEST=1 (NV_DOTS_MLX_DIR/NV_DOTS_WGPU_NEW/NV_DOTS_WGPU_PROMPT optional)"]
fn dots_mlx_4bit_decoder_boots_prefills_and_times_greedy_decode() {
    assert!(
        std::env::var(GATE_ENV).as_deref() == Ok("1"),
        "set {GATE_ENV}=1 to run the real-weights dots.ocr wgpu smoke (it must never silently skip)"
    );
    let dir = dots_mlx_dir();
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let mut cfg = dots_decoder_dense_config_from_hf_json_str(&raw)
        .expect("dots.ocr config.json must parse into the qwen2-arch dense config");
    let full_layers = cfg.num_hidden_layers;
    let truncated = std::env::var("NV_DOTS_WGPU_LAYERS").ok().map(|v| {
        let l: usize = v
            .parse()
            .unwrap_or_else(|_| panic!("NV_DOTS_WGPU_LAYERS={v} is not a usize"));
        assert!(
            l >= 1 && l <= full_layers,
            "NV_DOTS_WGPU_LAYERS={l} out of 1..={full_layers}"
        );
        cfg.layer_types.truncate(l);
        cfg.num_hidden_layers = l;
        l
    });
    let truncated = truncated.filter(|l| *l != full_layers);
    let vocab = cfg.vocab_size;

    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let q = std::env::var("NV_DOTS_WGPU_PROMPT").unwrap_or_else(|_| {
        "Transcribe the following sentence exactly, then continue the count to twenty: \
         one two three four five six seven"
            .into()
    });
    let mut prompt: Vec<u32> = vec![USER_TOKEN_ID];
    prompt.extend(tok.encode(q.as_str(), false).expect("encode prompt").get_ids());
    prompt.push(ENDOFUSER_TOKEN_ID);
    prompt.push(ASSISTANT_TOKEN_ID);

    let new_tokens = env_usize("NV_DOTS_WGPU_NEW", 128);
    assert!(new_tokens >= 16, "NV_DOTS_WGPU_NEW={new_tokens} < 16 gates nothing");
    let spec_tokens = env_usize("NV_DOTS_WGPU_SPEC", 0);
    let max_seq = prompt.len() + new_tokens + 8 + spec_tokens + 64;

    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let t0 = Instant::now();
    let loader = WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("open weights");
    let mut m = Qwen3_5DenseWgpu::from_loader(cfg, &loader, max_seq)
        .expect("Qwen3_5DenseWgpu::from_loader on the mlx dots.ocr-4bit checkpoint");
    drop(loader);
    eprintln!(
        "boot: {:.1}s, {} decode passes",
        t0.elapsed().as_secs_f64(),
        m.pass_count()
    );
    let mut census: std::collections::BTreeMap<String, usize> = Default::default();
    for l in m.debug_step_pass_labels_for_test() {
        *census.entry(l).or_default() += 1;
    }
    let mut rows: Vec<(usize, String)> = census.into_iter().map(|(l, n)| (n, l)).collect();
    rows.sort_unstable_by(|a, b| b.cmp(a));
    for (n, l) in &rows {
        eprintln!("pass-census {n:4} {l}");
    }

    let forced_text = " The sentence is: one two three four five six seven. Continuing: eight \
                       nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen \
                       eighteen nineteen twenty.";
    let forced_base: Vec<u32> = tok
        .encode(forced_text, false)
        .expect("encode forced continuation")
        .get_ids()
        .to_vec();
    let forced: Vec<u32> = (0..new_tokens)
        .map(|i| forced_base[i % forced_base.len()])
        .collect();

    let t1 = Instant::now();
    let first = m.prefill(&prompt).expect("prefill the text-only user turn");
    let prefill_s = t1.elapsed().as_secs_f64();
    assert!(
        truncated.is_some() || EOS_TOKEN_IDS.contains(&first),
        "dots refuses text-only turns with an immediate <|endofassistant|> (mlx-vlm agrees \
         token-for-token on this prompt); first={first} breaking that expectation means the \
         prompt or the arch wiring changed"
    );
    let mut argmaxes: Vec<u32> = Vec::with_capacity(new_tokens);
    let t2 = Instant::now();
    for t in &forced {
        argmaxes.push(m.decode_step(*t).expect("teacher-forced decode step"));
    }
    let decode_s = t2.elapsed().as_secs_f64();
    let stepped = argmaxes.len();
    for (i, t) in argmaxes.iter().enumerate() {
        assert!(
            (*t as usize) < vocab,
            "argmax {t} at step {i} is outside the vocab width {vocab}"
        );
    }
    let echo_text = tok.decode(&argmaxes, true).unwrap_or_default();
    println!(
        "DOTS-WGPU-DECODE checkpoint={REPO} backend=wgpu batch=1 prompt_tokens={} \
         prefill_s={prefill_s:.2} forced_tokens={stepped} ms_per_token={:.2} decode_tok_s={:.1}",
        prompt.len(),
        decode_s * 1e3 / stepped.max(1) as f64,
        stepped.max(1) as f64 / decode_s.max(1e-9)
    );
    println!("DOTS-WGPU-ARGMAX-HEAD :: {:?}", &argmaxes[..stepped.min(32)]);
    println!("DOTS-WGPU-ECHO :: {echo_text:?}");
    common::nozi_prof_dump();

    if spec_tokens > 0 {
        let vr = m.verify_max_rows();
        eprintln!(
            "[spec] verify rows={vr} dedicated_graph={}",
            m.verify_chain_rides_dedicated_graph()
        );
        assert!(vr >= 2, "speculative phase needs a verify graph (set NV_Q3D_VERIFY_M)");
        let mut hist: Vec<u32> = prompt.clone();
        hist.extend(&forced);
        let mut anchor = *argmaxes.last().expect("teacher-forced phase ran");
        let mut emitted = 0usize;
        let mut rounds = 0usize;
        let mut drafted_total = 0usize;
        let mut fallback_steps = 0usize;
        let mut verify_s = 0f64;
        let mut advance_s = 0f64;
        let t4 = Instant::now();
        while emitted < spec_tokens && m.current_pos() + vr + 2 < max_seq {
            let draft = pld_draft(&hist, anchor, (vr - 1).min(env_usize("NV_DOTS_WGPU_DRAFT", vr - 1)));
            if draft.is_empty() {
                let nt = m.decode_step(anchor).expect("spec fallback step");
                hist.push(anchor);
                anchor = nt;
                emitted += 1;
                fallback_steps += 1;
                continue;
            }
            let mut chain = Vec::with_capacity(draft.len() + 1);
            chain.push(anchor);
            chain.extend(&draft);
            let tv = Instant::now();
            let preds = m.verify_chain(&chain).expect("spec verify_chain");
            verify_s += tv.elapsed().as_secs_f64();
            let mut acc = 1usize;
            while acc < chain.len() && preds[acc - 1] == chain[acc] {
                acc += 1;
            }
            let ta = Instant::now();
            m.advance(acc).expect("spec advance");
            advance_s += ta.elapsed().as_secs_f64();
            hist.extend(&chain[..acc]);
            anchor = preds[acc - 1];
            emitted += acc;
            drafted_total += draft.len();
            rounds += 1;
        }
        let spec_s = t4.elapsed().as_secs_f64();
        println!(
            "DOTS-WGPU-SPEC emitted={emitted} rounds={rounds} fallback_steps={fallback_steps} \
             drafted={drafted_total} accept_per_round={:.2} ms_per_token={:.2} \
             effective_tok_s={:.1} (echo-workload acceptance; page acceptance needs vision e2e)",
            (emitted.saturating_sub(fallback_steps)) as f64 / rounds.max(1) as f64,
            spec_s * 1e3 / emitted.max(1) as f64,
            emitted as f64 / spec_s.max(1e-9)
        );
        println!(
            "DOTS-WGPU-SPEC-PHASES verify_ms_per_round={:.2} advance_ms_per_round={:.2}",
            verify_s * 1e3 / rounds.max(1) as f64,
            advance_s * 1e3 / rounds.max(1) as f64
        );
    }

    assert!(
        stepped >= 16,
        "teacher-forced decode covered only {stepped} steps; the timing gate demands at least 16"
    );
    let distinct: std::collections::BTreeSet<u32> = argmaxes.iter().copied().collect();
    assert!(
        truncated.is_some() || distinct.len() >= 8,
        "teacher-forced argmaxes are degenerate: only {} distinct tokens across {stepped} steps \
         ({echo_text:?})",
        distinct.len()
    );
}
