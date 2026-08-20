#![cfg(feature = "wgpu")]

mod common;
use common::env_usize;
use common::median;
use nv_models::gemma4_moe::Gemma4MoeConfig;
use nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu;
use nv_weights::GgufLoader;

fn entry_of(m: &Gemma4MoeWgpu, class: &str) -> String {
    let mut it = m
        .pass_rows()
        .into_iter()
        .filter(|(l, _, _, _)| l == class)
        .map(|(_, e, _, _)| e);
    let first = it.next().unwrap_or_else(|| panic!("no {class} dispatch"));
    assert!(
        it.all(|e| e == first),
        "{class} mixes entries across layers"
    );
    first
}

#[test]
#[ignore = "wires two full-depth graphs (~33 GiB) from a 26 GB GGUF; --ignored --release"]
fn narrow_row_w4_gemv_is_bit_identical_and_faster_on_moe_down() {
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("this A/B needs a wgpu adapter; there is no skip path");
    eprintln!("[w4] adapter: {}", ctx.info.name);
    assert!(
        std::env::var("NV_G4MOE_W4_GEMV").is_err(),
        "NV_G4MOE_W4_GEMV is pre-set on the runner; this test owns that knob and setting \
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
    if let Ok(n) = std::env::var("NV_G4MOE_W4_LAYERS") {
        let n: usize = n.parse().expect("NV_G4MOE_W4_LAYERS");
        assert!(n > 0 && n <= cfg.base.num_hidden_layers);
        cfg.base.num_hidden_layers = n;
        cfg.base.layer_types.truncate(n);
        eprintln!("[w4] TRUNCATED to {n} layers -- smoke run, not the model");
    }
    let max_seq = env_usize("NV_G4MOE_W4_MAXSEQ", 256);
    let warm = env_usize("NV_G4MOE_W4_WARM", 8);
    let reps = env_usize("NV_G4MOE_W4_REPS", 15);

    let build = |forced: Option<&str>| -> Gemma4MoeWgpu {
        match forced {
            Some(v) => unsafe { std::env::set_var("NV_G4MOE_W4_GEMV", v) },
            None => unsafe { std::env::remove_var("NV_G4MOE_W4_GEMV") },
        }
        let m = Gemma4MoeWgpu::from_loader(cfg.clone(), &gguf, max_seq).expect("build");
        unsafe { std::env::remove_var("NV_G4MOE_W4_GEMV") };
        m
    };

    let mut wide = build(Some("wide"));
    let mut narrow = build(None);
    for c in ["g4m-moe-gate", "g4m-moe-up", "g4m-moe-down"] {
        assert_eq!(entry_of(&wide, c), "g4m_gemv_w4", "{c} in the wide arm");
    }

    assert_eq!(entry_of(&narrow, "g4m-moe-gate"), "g4m_gemv_w4");
    assert_eq!(entry_of(&narrow, "g4m-moe-up"), "g4m_gemv_w4");
    assert_eq!(entry_of(&narrow, "g4m-moe-down"), "g4m_gemv_w4_r8");
    assert_eq!(
        wide.pass_count(),
        narrow.pass_count(),
        "the arms must differ only in which entry the down GEMV names"
    );
    eprintln!(
        "[w4] hidden {} / moe_inter {} -> gate/up {} groups, down {} groups; {} layers",
        cfg.base.hidden_size,
        cfg.moe_intermediate_size,
        cfg.base.hidden_size / 32,
        cfg.moe_intermediate_size / 32,
        cfg.base.num_hidden_layers
    );

    wide.reset().expect("reset");
    narrow.reset().expect("reset");
    let mut tw = wide.decode_step(2).expect("wide warm");
    let mut tn = narrow.decode_step(2).expect("narrow warm");
    for _ in 1..warm {
        tw = wide.decode_step(tw).expect("wide warm");
        tn = narrow.decode_step(tn).expect("narrow warm");
    }
    assert_eq!(tw, tn, "the two arms diverged during warmup");

    let mut ms_w = Vec::with_capacity(reps);
    let mut ms_n = Vec::with_capacity(reps);
    let mut differing = 0usize;
    let mut worst = 0f32;
    let mut prev: Option<Vec<f32>> = None;
    let mut sensitive = false;
    for step in 0..reps {
        let t0 = std::time::Instant::now();
        tw = wide.decode_step(tw).expect("wide");
        ms_w.push(t0.elapsed().as_secs_f64() * 1e3);
        let lw = wide.read_logits().expect("wide logits");

        let t1 = std::time::Instant::now();
        tn = narrow.decode_step(tn).expect("narrow");
        ms_n.push(t1.elapsed().as_secs_f64() * 1e3);
        let ln = narrow.read_logits().expect("narrow logits");

        for (a, b) in lw.iter().zip(ln.iter()) {
            if a.to_bits() != b.to_bits() {
                differing += 1;
                worst = worst.max((a - b).abs());
            }
        }
        assert_eq!(tw, tn, "step {step}: argmax token diverged");
        if let Some(p) = &prev {
            if p.iter()
                .zip(lw.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                sensitive = true;
            }
        }
        prev = Some(lw);
    }
    assert!(
        sensitive,
        "positive control failed: consecutive decode steps produced bit-identical logits, \
         so this comparison cannot detect a kernel difference"
    );
    assert_eq!(
        differing, 0,
        "the narrow-row W4 GEMV changed {differing} logits (worst |delta| {worst}); the \
         groups<=32 bit-identity argument is wrong and it must not be the default"
    );

    let w_lo = ms_w.iter().cloned().fold(f64::MAX, f64::min);
    let w_hi = ms_w.iter().cloned().fold(0f64, f64::max);
    let n_lo = ms_n.iter().cloned().fold(f64::MAX, f64::min);
    let n_hi = ms_n.iter().cloned().fold(0f64, f64::max);
    let w = median(&mut ms_w.clone());
    let n = median(&mut ms_n.clone());
    let mut pairs: Vec<f64> = ms_w.iter().zip(ms_n.iter()).map(|(a, b)| a - b).collect();
    let d = median(&mut pairs);
    eprintln!(
        "[w4] wide   {w:.3} ms/tok  (reps {w_lo:.3}..{w_hi:.3}, within-arm spread {:.1}%)",
        (w_hi - w_lo) / w * 100.0
    );
    eprintln!(
        "[w4] narrow {n:.3} ms/tok  (reps {n_lo:.3}..{n_hi:.3}, within-arm spread {:.1}%)",
        (n_hi - n_lo) / n * 100.0
    );
    eprintln!(
        "[w4] paired median delta {d:.3} ms/token = {:.3}x, {:.2} -> {:.2} tok/s, over {reps} \
         interleaved reps at pos {warm}..{}; logits bit-identical",
        w / n,
        1000.0 / w,
        1000.0 / n,
        warm + reps
    );
    assert!(
        d > 0.0,
        "the narrow-row W4 GEMV is not faster (paired median {d:.3} ms); it buys nothing"
    );
}
