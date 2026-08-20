#![cfg(feature = "wgpu")]

mod common;
use common::env_usize;
use common::median;
use nv_models::gemma4_moe::Gemma4MoeConfig;
use nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu;
use nv_weights::GgufLoader;

fn topk_entry(m: &Gemma4MoeWgpu) -> String {
    let mut it = m
        .pass_rows()
        .into_iter()
        .filter(|(l, _, _, _)| l == "g4m-moe-topk")
        .map(|(_, e, _, _)| e);
    let first = it
        .next()
        .expect("no g4m-moe-topk dispatch in the decode graph");
    assert!(
        it.all(|e| e == first),
        "the graph mixes router top-k entries across layers"
    );
    first
}

#[test]
#[ignore = "wires two full-depth graphs (~33 GiB) from a 26 GB GGUF; --ignored --release"]
fn parallel_router_topk_is_bit_identical_and_faster() {
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("this A/B needs a wgpu adapter; there is no skip path");
    eprintln!("[topk] adapter: {}", ctx.info.name);
    assert!(
        std::env::var("NV_G4MOE_ROUTER_TOPK").is_err(),
        "NV_G4MOE_ROUTER_TOPK is pre-set on the runner; this test owns that knob and \
         setting it outside makes both arms the same graph"
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
    if let Ok(n) = std::env::var("NV_G4MOE_TOPK_LAYERS") {
        let n: usize = n.parse().expect("NV_G4MOE_TOPK_LAYERS");
        assert!(n > 0 && n <= cfg.base.num_hidden_layers);
        cfg.base.num_hidden_layers = n;
        cfg.base.layer_types.truncate(n);
        eprintln!("[topk] TRUNCATED to {n} layers -- smoke run, not the model");
    }
    let max_seq = env_usize("NV_G4MOE_TOPK_MAXSEQ", 256);
    let warm = env_usize("NV_G4MOE_TOPK_WARM", 8);
    let reps = env_usize("NV_G4MOE_TOPK_REPS", 15);

    let build = |forced: Option<&str>| -> Gemma4MoeWgpu {
        match forced {
            Some(v) => unsafe { std::env::set_var("NV_G4MOE_ROUTER_TOPK", v) },
            None => unsafe { std::env::remove_var("NV_G4MOE_ROUTER_TOPK") },
        }
        let m = Gemma4MoeWgpu::from_loader(cfg.clone(), &gguf, max_seq).expect("build");
        unsafe { std::env::remove_var("NV_G4MOE_ROUTER_TOPK") };
        m
    };

    let mut serial = build(Some("serial"));
    let mut par = build(None);

    assert_eq!(topk_entry(&serial), "g4m_router_topk");
    assert_eq!(topk_entry(&par), "g4m_router_topk_par");
    assert_eq!(
        serial.pass_count(),
        par.pass_count(),
        "the two arms must differ only in which entry the top-k dispatch names"
    );
    eprintln!(
        "[topk] two graphs at {} dispatches/token, {} experts top-{}, {} layers",
        par.pass_count(),
        cfg.num_experts,
        cfg.top_k_experts,
        cfg.base.num_hidden_layers
    );

    serial.reset().expect("reset");
    par.reset().expect("reset");
    let mut ts = serial.decode_step(2).expect("serial warm");
    let mut tp = par.decode_step(2).expect("par warm");
    for _ in 1..warm {
        ts = serial.decode_step(ts).expect("serial warm");
        tp = par.decode_step(tp).expect("par warm");
    }
    assert_eq!(ts, tp, "the two arms diverged during warmup");

    let mut ms_s = Vec::with_capacity(reps);
    let mut ms_p = Vec::with_capacity(reps);
    let mut differing = 0usize;
    let mut worst = 0f32;
    let mut prev: Option<Vec<f32>> = None;
    let mut sensitive = false;
    for step in 0..reps {
        let t0 = std::time::Instant::now();
        ts = serial.decode_step(ts).expect("serial");
        ms_s.push(t0.elapsed().as_secs_f64() * 1e3);
        let ls = serial.read_logits().expect("serial logits");

        let t1 = std::time::Instant::now();
        tp = par.decode_step(tp).expect("par");
        ms_p.push(t1.elapsed().as_secs_f64() * 1e3);
        let lp = par.read_logits().expect("par logits");

        assert_eq!(ls.len(), lp.len());
        for (a, b) in ls.iter().zip(lp.iter()) {
            if a.to_bits() != b.to_bits() {
                differing += 1;
                worst = worst.max((a - b).abs());
            }
        }
        assert_eq!(ts, tp, "step {step}: argmax token diverged");

        if let Some(p) = &prev {
            if p.iter()
                .zip(ls.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                sensitive = true;
            }
        }
        prev = Some(ls);
    }
    assert!(
        sensitive,
        "positive control failed: consecutive decode steps produced bit-identical logits, \
         so this comparison cannot detect a router difference"
    );
    assert_eq!(
        differing, 0,
        "the parallel router top-k changed {differing} logits (worst |delta| {worst}); it is \
         not bit-identical and must not be the default"
    );

    let s_lo = ms_s.iter().cloned().fold(f64::MAX, f64::min);
    let s_hi = ms_s.iter().cloned().fold(0f64, f64::max);
    let p_lo = ms_p.iter().cloned().fold(f64::MAX, f64::min);
    let p_hi = ms_p.iter().cloned().fold(0f64, f64::max);
    let s = median(&mut ms_s.clone());
    let p = median(&mut ms_p.clone());
    let mut pairs: Vec<f64> = ms_s.iter().zip(ms_p.iter()).map(|(a, b)| a - b).collect();
    let d = median(&mut pairs);
    eprintln!(
        "[topk] serial   {s:.3} ms/tok  (reps {s_lo:.3}..{s_hi:.3}, within-arm spread {:.1}%)",
        (s_hi - s_lo) / s * 100.0
    );
    eprintln!(
        "[topk] parallel {p:.3} ms/tok  (reps {p_lo:.3}..{p_hi:.3}, within-arm spread {:.1}%)",
        (p_hi - p_lo) / p * 100.0
    );
    eprintln!(
        "[topk] paired median delta {d:.3} ms/token = {:.3}x, {:.2} -> {:.2} tok/s, \
         over {} interleaved reps at pos {warm}..{}; logits bit-identical",
        s / p,
        1000.0 / s,
        1000.0 / p,
        reps,
        warm + reps
    );
    assert!(
        d > 0.0,
        "the parallel router top-k is not faster (paired median {d:.3} ms); it buys nothing"
    );
}
