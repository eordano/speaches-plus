
#![cfg(feature = "wgpu")]

mod common;
use common::env_usize;
use common::median;
use nv_models::gemma4_moe::Gemma4MoeConfig;
use nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu;
use nv_weights::GgufLoader;

fn spread(v: &[f64]) -> (f64, f64, f64) {
    let lo = v.iter().cloned().fold(f64::MAX, f64::min);
    let hi = v.iter().cloned().fold(0f64, f64::max);
    let mut c = v.to_vec();
    let m = median(&mut c);
    (lo, hi, (hi - lo) / m)
}

fn prompt_tokens() -> Vec<u32> {
    let text = std::env::var("NV_G4MOE_WIDE_PROMPT").unwrap_or_else(|_| {
        "Explain, in three short paragraphs, why a memory-bandwidth-bound decoder \
         cannot be made faster by adding more compute units."
            .to_string()
    });
    let path = std::env::var("NV_G4MOE_WIDE_TOKENIZER").unwrap_or_else(|_| {
        format!(
            "{}/.cache/nv-gguf-serve/gemma-4-26B-A4B-it-Q8_0/tokenizer.json",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    match tokenizers::Tokenizer::from_file(&path) {
        Ok(t) => {
            let enc = t.encode(text.as_str(), false).expect("encode");
            let mut ids = vec![2u32];
            ids.extend_from_slice(enc.get_ids());
            eprintln!("[wideab] prompt: {} tokens from {path}", ids.len());
            ids
        }
        Err(e) => {
            eprintln!(
                "[wideab] NO TOKENIZER at {path} ({e}); decoding from BOS alone. This \
                 checkpoint then falls into a repetition and the agreement rate below is a \
                 fact about that attractor, not about prose."
            );
            vec![2u32]
        }
    }
}

#[test]
#[ignore = "wires three full-depth graphs from a 26 GB GGUF; --ignored --release"]
fn the_wide_dense_load_is_worth_its_place_in_the_26b_token() {
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("this A/B needs a wgpu adapter; there is no skip path");
    eprintln!("[wideab] adapter: {}", ctx.info.name);
    assert!(
        std::env::var("NV_G4MOE_GEMV_WIDE").is_err(),
        "NV_G4MOE_GEMV_WIDE is pre-set on the runner; this test owns that knob and setting \
         it outside makes both arms the same graph"
    );

    let path =
        std::env::var("NV_GGUF_PATH").expect("set NV_GGUF_PATH to a gemma-4-26B-A4B-it-Q8_0 GGUF");
    assert!(
        std::path::Path::new(&path).exists(),
        "no GGUF at {path}; set NV_GGUF_PATH"
    );
    let gguf = GgufLoader::open(&path, &candle_core::Device::Cpu).expect("open gguf");
    let mut cfg: Gemma4MoeConfig =
        nv_models::gemma4_gguf::gemma4_moe_config_from_gguf(&gguf).expect("gguf config");
    if let Ok(n) = std::env::var("NV_G4MOE_WIDE_LAYERS") {
        let n: usize = n.parse().expect("NV_G4MOE_WIDE_LAYERS");
        assert!(n > 0 && n <= cfg.base.num_hidden_layers);
        cfg.base.num_hidden_layers = n;
        cfg.base.layer_types.truncate(n);
        eprintln!("[wideab] TRUNCATED to {n} layers -- smoke run, not the model");
    }
    let max_seq = env_usize("NV_G4MOE_WIDE_MAXSEQ", 256);
    let warm = env_usize("NV_G4MOE_WIDE_WARM", 8);
    let reps = env_usize("NV_G4MOE_WIDE_REPS", 15);
    let with_control = env_usize("NV_G4MOE_WIDE_CONTROL", 1) == 1;

    let classes = std::env::var("NV_G4MOE_WIDE_CLASSES").ok();
    if let Some(c) = &classes {
        eprintln!("[wideab] arm B restricted to dispatch labels containing {c:?}");
    }
    let build = |wide: bool| -> Gemma4MoeWgpu {
        if wide {
            let v = classes.clone().unwrap_or_else(|| "1".to_string());
            unsafe { std::env::set_var("NV_G4MOE_GEMV_WIDE", &v) }
        } else {
            unsafe { std::env::set_var("NV_G4MOE_GEMV_WIDE", "0") }
        }
        let m = Gemma4MoeWgpu::from_loader(cfg.clone(), &gguf, max_seq).expect("build");
        unsafe { std::env::remove_var("NV_G4MOE_GEMV_WIDE") };
        m
    };

    let mut narrow = build(false);
    let mut wide = build(true);
    let mut ctl = with_control.then(|| build(false));
    let (nw, nn) = narrow.dense_gemv_wide_counts();
    let (ww, wn) = wide.dense_gemv_wide_counts();
    assert_eq!(
        nw, 0,
        "NV_G4MOE_GEMV_WIDE=0 still built {nw} wide dense GEMVs; the knob did not reach the \
         builder and both arms are the same graph"
    );
    assert!(
        ww > 0 && (wn == 0 || classes.is_some()),
        "the default arm built {ww} wide / {wn} scalar dense GEMVs; every dense projection \
         in this checkpoint is vec4-coverable, so a scalar one means the guard is refusing a \
         shape it should take"
    );
    assert_eq!(
        nn,
        ww + wn,
        "the arms disagree on how many dense GEMVs exist ({nn} vs {})",
        ww + wn
    );
    assert_eq!(
        narrow.pass_count(),
        wide.pass_count(),
        "the two arms must differ only in one uniform, not in the dispatch list"
    );
    eprintln!(
        "[wideab] two graphs at {} dispatches/token, {} of them dense GEMVs, {} layers, \
         {:.3} GB of weight per token",
        wide.pass_count(),
        ww,
        cfg.base.num_hidden_layers,
        wide.weight_bytes_per_token() as f64 / 1e9
    );

    narrow.reset().expect("reset");
    wide.reset().expect("reset");
    let trace = env_usize("NV_G4MOE_WIDE_TRACE", 8);
    let prompt_ids = prompt_tokens();
    let mut fed = if prompt_ids.len() > 1 {
        let a = narrow.prefill(&prompt_ids).expect("narrow prefill");
        let b = wide.prefill(&prompt_ids).expect("wide prefill");
        eprintln!(
            "[wideab] first token after prefill: A {a}, B {b} ({}); the prompt's last {} \
             positions ran the decode graph, not pm_gemm_bf16",
            if a == b { "agree" } else { "DIFFER" },
            prompt_ids.len() % 8
        );
        a
    } else {
        2u32
    };
    let mut trace_agree = 0usize;
    for step in 0..trace {
        let (tn, ln) = narrow.decode_step_logits(fed).expect("narrow trace");
        let (tw, lw) = wide.decode_step_logits(fed).expect("wide trace");
        let mag = ln.iter().fold(0f32, |a, b| a.max(b.abs())) as f64;
        assert!(
            mag > 1e-3,
            "trace step {step}: the logit vector is degenerate (max |logit| {mag:.3e})"
        );
        let mut num = 0f64;
        let mut den = 0f64;
        let mut worst = 0f64;
        for (a, b) in ln.iter().zip(lw.iter()) {
            let d = (*a - *b) as f64;
            num += d * d;
            den += (*a as f64) * (*a as f64);
            worst = worst.max(d.abs() / mag);
        }
        if tn == tw {
            trace_agree += 1;
        }
        eprintln!(
            "[wideab] divergence step {step}: rel-RMS {:.3e}, worst {worst:.3e} of max logit, \
             argmax {}",
            (num / den.max(1e-30)).sqrt(),
            if tn == tw { "same" } else { "DIFFERENT" }
        );
        fed = tn;
    }
    eprintln!(
        "[wideab] step 0 above is the kernel's OWN numeric effect -- one forward pass, no \
         accumulated state. Everything after it is that error amplified by an \
         autoregressive MoE; {trace_agree}/{trace} argmaxes agreed along the trace."
    );

    narrow.reset().expect("reset");
    wide.reset().expect("reset");
    if let Some(c) = ctl.as_mut() {
        c.reset().expect("reset");
    }
    fed = if prompt_ids.len() > 1 {
        let a = narrow.prefill(&prompt_ids).expect("narrow prefill");
        wide.prefill(&prompt_ids).expect("wide prefill");
        if let Some(c) = ctl.as_mut() {
            let b = c.prefill(&prompt_ids).expect("ctl prefill");
            assert_eq!(
                a, b,
                "the two scalar graphs disagree straight out of prefill"
            );
        }
        a
    } else {
        2u32
    };
    let mut null_differing = 0usize;
    let mut null_words = 0usize;
    for _ in 0..warm {
        let (tn, ln) = narrow.decode_step_logits(fed).expect("narrow warm");
        wide.decode_step(fed).expect("wide warm");
        if let Some(c) = ctl.as_mut() {
            let (_, lc) = c.decode_step_logits(fed).expect("ctl warm");
            null_words += ln.len();
            null_differing += ln
                .iter()
                .zip(lc.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
        }
        fed = tn;
    }
    if ctl.is_some() {
        eprintln!(
            "[wideab] numeric null control: {null_differing}/{null_words} logit words differ \
             between two separately built SCALAR graphs"
        );
        assert_eq!(
            null_differing, 0,
            "two scalar graphs disagree on {null_differing} of {null_words} logit words; \
             this graph is not deterministic across instances and no numeric difference \
             below is attributable to the load width"
        );
    }

    let mut ms_n = Vec::with_capacity(reps);
    let mut ms_w = Vec::with_capacity(reps);
    let mut differing = 0usize;
    let mut total = 0usize;
    let mut worst_rel = 0f64;
    let mut sq_num = 0f64;
    let mut sq_den = 0f64;
    let mut over_1e3 = 0usize;
    let mut agree = 0usize;
    let mut disagree: Vec<(usize, u32, u32, f64)> = Vec::new();
    let mut prev: Option<Vec<f32>> = None;
    let mut sensitive = false;
    for step in 0..reps {
        let t0 = std::time::Instant::now();
        let (tn, ln) = narrow.decode_step_logits(fed).expect("narrow");
        ms_n.push(t0.elapsed().as_secs_f64() * 1e3);

        let t1 = std::time::Instant::now();
        let (tw, lw) = wide.decode_step_logits(fed).expect("wide");
        ms_w.push(t1.elapsed().as_secs_f64() * 1e3);

        assert_eq!(ln.len(), lw.len());
        let mag = ln.iter().fold(0f32, |a, b| a.max(b.abs())) as f64;
        assert!(
            mag > 1e-3,
            "step {step}: the logit vector is degenerate (max |logit| {mag:.3e}); this \
             comparison would pass on zeros"
        );
        for (a, b) in ln.iter().zip(lw.iter()) {
            total += 1;
            let d = (*a - *b) as f64;
            sq_num += d * d;
            sq_den += (*a as f64) * (*a as f64);
            if a.to_bits() != b.to_bits() {
                differing += 1;
                let rel = d.abs() / mag;
                worst_rel = worst_rel.max(rel);
                if rel > 1e-3 {
                    over_1e3 += 1;
                }
            }
        }
        if tn == tw {
            agree += 1;
        } else {
            let margin = (ln[tn as usize] - ln[tw as usize]).abs() as f64 / mag;
            disagree.push((step, tn, tw, margin));
        }
        if let Some(p) = &prev {
            if p.iter()
                .zip(ln.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                sensitive = true;
            }
        }
        prev = Some(ln);
        fed = tn;
        if let Some(c) = ctl.as_mut() {
            c.decode_step(fed).expect("ctl lockstep");
        }
    }
    assert!(
        sensitive,
        "positive control failed: consecutive decode steps produced bit-identical logits, so \
         this comparison cannot detect a kernel difference"
    );

    let mut ms_n2 = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        let tn = narrow.decode_step(fed).expect("narrow A-prime");
        ms_n2.push(t0.elapsed().as_secs_f64() * 1e3);
        wide.decode_step(fed).expect("wide lockstep");
        if let Some(c) = ctl.as_mut() {
            c.decode_step(fed).expect("ctl lockstep");
        }
        fed = tn;
    }

    let (n_lo, n_hi, n_sp) = spread(&ms_n);
    let (w_lo, w_hi, w_sp) = spread(&ms_w);
    let (_, _, n2_sp) = spread(&ms_n2);
    let n = median(&mut ms_n.clone());
    let w = median(&mut ms_w.clone());
    let n2 = median(&mut ms_n2.clone());
    let mut pairs: Vec<f64> = ms_n.iter().zip(ms_w.iter()).map(|(a, b)| a - b).collect();
    let d = median(&mut pairs);
    let aba = (n2 - n).abs() / n;

    eprintln!(
        "[wideab] scalar A  {n:.3} ms/tok  (reps {n_lo:.3}..{n_hi:.3}, within-arm spread \
         {:.1}%)",
        n_sp * 100.0
    );
    eprintln!(
        "[wideab] wide   B  {w:.3} ms/tok  (reps {w_lo:.3}..{w_hi:.3}, within-arm spread \
         {:.1}%)",
        w_sp * 100.0
    );
    eprintln!(
        "[wideab] scalar A' {n2:.3} ms/tok (within-arm spread {:.1}%) -- A/A' drift {:.2}%",
        n2_sp * 100.0,
        aba * 100.0
    );
    eprintln!(
        "[wideab] paired median delta {d:.3} ms/token = {:.4}x, {:.2} -> {:.2} tok/s, over \
         {reps} interleaved reps at pos {warm}..{}",
        n / w,
        1000.0 / n,
        1000.0 / w,
        warm + reps
    );
    eprintln!(
        "[wideab] logits: {differing}/{total} f32 words differ; rel-RMS of the whole vector \
         {:.3e}; {over_1e3} words ({:.4}%) differ by more than 1e-3 of the step's max logit; \
         worst {worst_rel:.3e}. The wide load reassociates a {}-term reduction and is NOT \
         bit-identical -- this is what that costs, and the tail is bigger than \
         reassociation alone because the router turns it into expert swaps.",
        (sq_num / sq_den.max(1e-30)).sqrt(),
        over_1e3 as f64 / total as f64 * 100.0,
        cfg.base.hidden_size
    );
    eprintln!(
        "[wideab] teacher-forced greedy agreement: {agree}/{reps} positions ({:.2}%)",
        agree as f64 / reps as f64 * 100.0
    );
    for (st, a, b, m) in &disagree {
        eprintln!(
            "[wideab]   disagreement at position {st}: A picked {a}, B picked {b}; A's own \
             margin between them was {m:.3e} of its max logit"
        );
    }
    if !disagree.is_empty() {
        let mut ms: Vec<f64> = disagree.iter().map(|d| d.3).collect();
        let worst_margin = ms.iter().cloned().fold(0f64, f64::max);
        eprintln!(
            "[wideab] {} disagreements, median margin {:.3e}, widest {worst_margin:.3e} of \
             the max logit. A flip at a near-tie is the model being undecided; a flip at a \
             wide margin is not.",
            disagree.len(),
            median(&mut ms)
        );
    }

    if let Some(ctl) = ctl.as_mut() {
        let (cw, _) = ctl.dense_gemv_wide_counts();
        assert_eq!(cw, 0, "the null control was built with the wide load on");
        let mut ms_c = Vec::with_capacity(reps);
        let mut ms_n3 = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = std::time::Instant::now();
            let tn = narrow.decode_step(fed).expect("narrow ctl-pair");
            ms_n3.push(t0.elapsed().as_secs_f64() * 1e3);
            let t1 = std::time::Instant::now();
            ctl.decode_step(fed).expect("ctl");
            ms_c.push(t1.elapsed().as_secs_f64() * 1e3);
            wide.decode_step(fed).expect("wide lockstep");
            fed = tn;
        }
        let mut cp: Vec<f64> = ms_n3.iter().zip(ms_c.iter()).map(|(a, b)| a / b).collect();
        let cd = (median(&mut cp) - 1.0).abs();
        eprintln!(
            "[wideab] null control: a second scalar graph, paired the same way, reads {:.2}% \
             off arm A. Anything the wide arm buys has to clear that.",
            cd * 100.0
        );
        assert!(
            cd < 0.05,
            "the null control ran {:.1}% off arm A on paired reps; this box moved under the \
             measurement and nothing here is attributable",
            cd * 100.0
        );
    }

    assert!(
        aba < 0.05,
        "A/A' drift is {:.1}%: the scalar arm measured {n:.3} then {n2:.3} ms/tok around the \
         wide arm, so the {:.3} ms paired delta is weather, not a kernel",
        aba * 100.0,
        d
    );
    assert!(
        d > 0.0,
        "the wide dense load is not faster (paired median {d:.3} ms/token). It reassociates \
         the reduction for nothing and must not be the default."
    );
    let bar = env_usize("NV_G4MOE_WIDE_AGREE_BAR_PCT", 70) as f64 / 100.0;
    assert!(
        agree as f64 / reps as f64 >= bar,
        "teacher-forced greedy agreement is {agree}/{reps} against a {:.0}% floor. That is \
         not the model being sensitive to a valid reassociation -- at that rate the wide \
         kernel is computing something else",
        bar * 100.0
    );
}
