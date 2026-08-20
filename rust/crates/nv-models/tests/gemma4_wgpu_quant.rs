#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use common::TINY_CONFIG;
use nv_models::gemma4::{Gemma4Config, LayerType};
use nv_models::gemma4_wgpu::{Gemma4Wgpu, HostBf16Lin, HostLayer, HostProj, HostWeights};
use std::sync::Mutex;
use common::LcgCentered0p1Shift32F64 as Lcg;

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn nvidia_smi_occupancy(tag: &str) {
    match std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ])
        .output()
    {
        Ok(o) => eprintln!(
            "[occupancy:{tag}] {}",
            String::from_utf8_lossy(&o.stdout).trim()
        ),
        Err(e) => eprintln!("[occupancy:{tag}] nvidia-smi unavailable: {e}"),
    }
}

fn external_gpu_util() -> Option<u32> {
    let o = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    String::from_utf8_lossy(&o.stdout).trim().parse().ok()
}

fn idle_gate(tag: &str) {
    let budget = std::time::Duration::from_secs(
        std::env::var("NV_GPU_IDLE_WAIT_S")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(90),
    );
    let t0 = std::time::Instant::now();
    let mut clean = 0usize;
    while t0.elapsed() < budget {
        match external_gpu_util() {
            Some(u) if u <= 3 => {
                clean += 1;
                if clean >= 3 {
                    eprintln!(
                        "[idle-gate:{tag}] external util <=3% for 3 samples after {:.0}s",
                        t0.elapsed().as_secs_f64()
                    );
                    return;
                }
            }
            Some(_) => clean = 0,
            None => return,
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    eprintln!(
        "[idle-gate:{tag}] TIMED OUT after {:.0}s with external util {:?}% - timings below are CONTAMINATED",
        t0.elapsed().as_secs_f64(),
        external_gpu_util()
    );
}

struct MsStats {
    mean: f64,
    median: f64,
    p10: f64,
    min: f64,
    max: f64,
}

impl MsStats {
    fn of(v: &[f64]) -> Self {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = s.len();
        Self {
            mean: v.iter().sum::<f64>() / n as f64,
            median: s[n / 2],
            p10: s[n / 10],
            min: s[0],
            max: s[n - 1],
        }
    }
}

impl std::fmt::Display for MsStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mean {:.2} median {:.2} p10 {:.2} min {:.2} max {:.2} ms",
            self.mean, self.median, self.p10, self.min, self.max
        )
    }
}

fn tiny_host_weights(config: &Gemma4Config, seed: u64) -> HostWeights {
    let mut rng = Lcg(seed);
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let n_q = config.num_attention_heads;
    let mut layers = Vec::new();
    for i in 0..config.num_hidden_layers {
        let kind = config.layer_kind(i);
        let hd = config.head_dim_for(kind);
        let nkv = config.num_kv_heads_for(kind);
        let q_dim = n_q * hd;
        let kv_dim = nkv * hd;
        let has_v = !matches!(
            (kind, config.attention_k_eq_v),
            (LayerType::FullAttention, true)
        );
        let qkv_rows = q_dim + kv_dim * if has_v { 2 } else { 1 };
        let mk = |rng: &mut Lcg, n: usize, k: usize| {
            HostProj::Bf16(HostBf16Lin {
                w: rng.bf16_vec(n * k),
                n,
                k,
            })
        };
        layers.push(HostLayer {
            kind,
            input_ln: rng.bf16_vec_around_one(hidden),
            post_attn_ln: rng.bf16_vec_around_one(hidden),
            pre_ff_ln: rng.bf16_vec_around_one(hidden),
            post_ff_ln: rng.bf16_vec_around_one(hidden),
            q_norm: rng.bf16_vec_around_one(hd),
            k_norm: rng.bf16_vec_around_one(hd),
            layer_scalar: 0.9,
            has_v,
            qkv: mk(&mut rng, qkv_rows, hidden),
            o: mk(&mut rng, hidden, q_dim),
            gate_up: mk(&mut rng, 2 * inter, hidden),
            down: mk(&mut rng, hidden, inter),
        });
    }
    HostWeights {
        embed: rng.bf16_vec(config.vocab_size * hidden),
        final_norm: rng.bf16_vec_around_one(hidden),
        layers,
    }
}

#[test]
fn synthetic_lmhead_int8_agreement_and_determinism() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap();
    let weights = tiny_host_weights(&config, 0x5eed);
    let steps: Vec<u32> = vec![7, 300, 42, 511, 0, 13, 99, 250, 8, 9, 10, 11];

    let run = |int8: bool| -> Vec<(u32, Vec<f32>)> {
        if int8 {
            std::env::set_var("NV_WGPU_LMHEAD_INT8", "1");
        } else {
            std::env::remove_var("NV_WGPU_LMHEAD_INT8");
        }
        let mut m = Gemma4Wgpu::new(
            Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap(),
            &weights,
            64,
        )
        .unwrap();
        std::env::remove_var("NV_WGPU_LMHEAD_INT8");
        steps
            .iter()
            .map(|t| m.decode_step_logits(*t).unwrap())
            .collect()
    };

    let base = run(false);
    let a = run(true);
    let b = run(true);

    let mut diff_bits = 0usize;
    for ((_, la), (_, lb)) in a.iter().zip(b.iter()) {
        for (x, y) in la.iter().zip(lb.iter()) {
            if x.to_bits() != y.to_bits() {
                diff_bits += 1;
            }
        }
    }
    assert_eq!(
        diff_bits, 0,
        "int8 lm_head path must be deterministic run-to-run"
    );

    let mut agree = 0usize;
    let mut worst_abs = 0f32;
    for (i, ((bt, bl), (at, al))) in base.iter().zip(a.iter()).enumerate() {
        assert!(
            al.iter().all(|v| v.is_finite()),
            "step {i}: non-finite int8 logits"
        );
        let mut max_abs = 0f32;
        for (x, y) in bl.iter().zip(al.iter()) {
            max_abs = max_abs.max((x - y).abs());
        }
        worst_abs = worst_abs.max(max_abs);
        if bt == at {
            agree += 1;
        }
        eprintln!("step {i}: bf16 argmax {bt} int8 argmax {at} max_abs {max_abs:.6e}");
    }
    eprintln!(
        "bf16-vs-int8 lm_head: argmax agreement {agree}/{} worst max_abs {worst_abs:.6e}",
        steps.len()
    );
    assert!(worst_abs < 0.5, "int8 lm_head logits drifted: {worst_abs}");
    assert!(
        agree * 12 >= steps.len() * 6,
        "argmax agreement below 6/12: {agree}. NOTE: this is a 6-layer model of uniform random \
         weights, so its logit margins are near zero and argmax agreement is NOT a quality \
         signal here - it only catches gross breakage. The meaningful gate is the real-model \
         free-running test in wgpu_fp8_freerun.rs. The old 9/12 threshold was calibrated against \
         a generator that emitted only negative values."
    );
}

#[test]
fn synthetic_o_proj_nvfp4_agreement_and_bytes() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap();
    let weights = tiny_host_weights(&config, 0x0cafe);
    let steps: Vec<u32> = vec![7, 300, 42, 511, 0, 13, 99, 250, 8, 9, 10, 11];

    let run = |quant: bool| -> (u64, usize, Vec<(u32, Vec<f32>)>) {
        nv_models::gemma4_wgpu::set_attn_variant(Some(nv_models::gemma4_wgpu::AttnVariant {
            on: false,
            ..nv_models::gemma4_wgpu::ATTN_VARIANT_DEFAULT
        }));
        if quant {
            std::env::set_var("NV_WGPU_O_NVFP4", "1");
        } else {
            std::env::remove_var("NV_WGPU_O_NVFP4");
        }
        let mut m = Gemma4Wgpu::new(
            Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap(),
            &weights,
            64,
        )
        .unwrap();
        std::env::remove_var("NV_WGPU_O_NVFP4");
        nv_models::gemma4_wgpu::set_attn_variant(None);
        let out = steps
            .iter()
            .map(|t| m.decode_step_logits(*t).unwrap())
            .collect();
        (m.weight_bytes_per_token(), m.pass_count(), out)
    };

    let (base_bytes, base_passes, base) = run(false);
    let (q_bytes, q_passes, a) = run(true);
    let (q_bytes2, _, b) = run(true);
    assert_eq!(q_bytes, q_bytes2);

    let mut diff_bits = 0usize;
    for ((_, la), (_, lb)) in a.iter().zip(b.iter()) {
        for (x, y) in la.iter().zip(lb.iter()) {
            if x.to_bits() != y.to_bits() {
                diff_bits += 1;
            }
        }
    }
    assert_eq!(
        diff_bits, 0,
        "nvfp4 o_proj path must be deterministic run-to-run"
    );

    let mut agree = 0usize;
    let mut worst_abs = 0f32;
    for (i, ((bt, bl), (at, al))) in base.iter().zip(a.iter()).enumerate() {
        assert!(
            al.iter().all(|v| v.is_finite()),
            "step {i}: non-finite logits"
        );
        let mut max_abs = 0f32;
        for (x, y) in bl.iter().zip(al.iter()) {
            max_abs = max_abs.max((x - y).abs());
        }
        worst_abs = worst_abs.max(max_abs);
        if bt == at {
            agree += 1;
        }
    }
    eprintln!(
        "bf16-vs-nvfp4 o_proj: argmax agreement {agree}/{} worst max_abs {worst_abs:.6e}",
        steps.len()
    );
    eprintln!(
        "weight bytes/token {base_bytes} -> {q_bytes} ({:+.2}%), passes {base_passes} -> {q_passes} ({:+})",
        100.0 * (q_bytes as f64 - base_bytes as f64) / base_bytes as f64,
        q_passes as i64 - base_passes as i64
    );
    assert!(
        q_bytes < base_bytes,
        "nvfp4 o_proj must reduce weight bytes"
    );
    assert_eq!(
        q_passes - base_passes,
        config.num_hidden_layers,
        "nvfp4 o_proj adds exactly one activation-quantize dispatch per layer"
    );
    assert!(
        agree * 12 >= steps.len() * 6,
        "argmax agreement below 6/12: {agree}. NOTE: this is a 6-layer model of uniform random \
         weights, so its logit margins are near zero and argmax agreement is NOT a quality \
         signal here - it only catches gross breakage. The meaningful gate is the real-model \
         free-running test in wgpu_fp8_freerun.rs. The old 9/12 threshold was calibrated against \
         a generator that emitted only negative values."
    );
}

#[test]
fn synthetic_attn_fp8_agreement_bytes_and_determinism() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if ctx_or_skip().is_none() {
        return;
    }
    let config = Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap();
    let weights = tiny_host_weights(&config, 0xf8f8);
    let steps: Vec<u32> = vec![7, 300, 42, 511, 0, 13, 99, 250, 8, 9, 10, 11];

    let run = |fp8: bool| -> (u64, usize, Vec<(u32, Vec<f32>)>) {
        nv_models::gemma4_wgpu::set_attn_variant(Some(nv_models::gemma4_wgpu::AttnVariant {
            on: fp8,
            ..nv_models::gemma4_wgpu::ATTN_VARIANT_DEFAULT
        }));
        let mut m = Gemma4Wgpu::new(
            Gemma4Config::from_hf_json_str(TINY_CONFIG).unwrap(),
            &weights,
            64,
        )
        .unwrap();
        nv_models::gemma4_wgpu::set_attn_variant(None);
        let out = steps
            .iter()
            .map(|t| m.decode_step_logits(*t).unwrap())
            .collect();
        (m.weight_bytes_per_token(), m.pass_count(), out)
    };

    let (base_bytes, base_passes, base) = run(false);
    let (fp8_bytes, fp8_passes, a) = run(true);
    let (fp8_bytes2, fp8_passes2, b) = run(true);
    assert_eq!((fp8_bytes, fp8_passes), (fp8_bytes2, fp8_passes2));

    let mut diff_bits = 0usize;
    for ((_, la), (_, lb)) in a.iter().zip(b.iter()) {
        for (x, y) in la.iter().zip(lb.iter()) {
            if x.to_bits() != y.to_bits() {
                diff_bits += 1;
            }
        }
    }
    assert_eq!(diff_bits, 0, "fp8 attention path must be deterministic");

    let mut agree = 0usize;
    let mut worst_abs = 0f32;
    for (i, ((bt, bl), (at, al))) in base.iter().zip(a.iter()).enumerate() {
        assert!(
            al.iter().all(|v| v.is_finite()),
            "step {i}: non-finite logits under fp8 attention"
        );
        for (x, y) in bl.iter().zip(al.iter()) {
            worst_abs = worst_abs.max((x - y).abs());
        }
        if bt == at {
            agree += 1;
        }
    }
    eprintln!(
        "bf16-vs-fp8 attention (q/k/v/o): argmax agreement {agree}/{} worst max_abs {worst_abs:.6e}",
        steps.len()
    );
    eprintln!(
        "weight bytes/token {base_bytes} -> {fp8_bytes} ({:+.2}%), passes {base_passes} -> {fp8_passes} ({:+})",
        100.0 * (fp8_bytes as f64 - base_bytes as f64) / base_bytes as f64,
        fp8_passes as i64 - base_passes as i64
    );
    assert!(
        fp8_bytes < base_bytes,
        "fp8 attention must reduce weight bytes"
    );
    assert_eq!(
        fp8_passes, base_passes,
        "fp8 attention must not change the dispatch count"
    );
    assert!(
        agree * 12 >= steps.len() * 6,
        "argmax agreement below 6/12: {agree}. NOTE: this is a 6-layer model of uniform random \
         weights, so its logit margins are near zero and argmax agreement is NOT a quality \
         signal here - it only catches gross breakage. The meaningful gate is the real-model \
         free-running test in wgpu_fp8_freerun.rs. The old 9/12 threshold was calibrated against \
         a generator that emitted only negative values."
    );
}

#[test]
fn fp8_row_quant_error_on_real_o_proj_shape() {
    let Some(ctx) = ctx_or_skip() else { return };
    let n = 5376usize;
    let k = 8192usize;
    let mut rng = Lcg(0x00f8_5376);
    let w = rng.bf16_vec(n * k);
    let x = rng.bf16_vec(k);
    let signs = w.iter().filter(|b| (*b & 0x8000) != 0).count();
    assert!(
        signs * 4 > w.len() && signs * 4 < w.len() * 3,
        "the generator must be zero-mean: {signs}/{} negative. An earlier revision of this test \
         used a generator whose values were ALL negative, which removed the sign cancellation \
         from the reference dot product and understated the fp8 output error by ~44x. That is \
         where the bogus \"fp8 error is one bf16 ulp\" claim came from.",
        w.len()
    );

    let f = |b: u16| f32::from_bits((b as u32) << 16);
    let mut want = vec![0f32; n];
    for (r, dst) in want.iter_mut().enumerate() {
        let row = &w[r * k..(r + 1) * k];
        *dst = row.iter().zip(x.iter()).map(|(a, b)| f(*a) * f(*b)).sum();
    }

    let (wq, scales) = nv_kernels::wgpu_backend::kernels::quant_gemv::quantize_rows_fp8(&w, n, k);
    assert_eq!(wq.len(), n * k / 4);
    assert_eq!(scales.len(), n);
    let mut got = vec![0u16; n];
    nv_kernels::wgpu_backend::kernels::quant_gemv::gemv_fp8_bf16(
        ctx, &wq, &scales, &x, &mut got, n, k,
    )
    .unwrap();

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut sum_sq_err = 0f64;
    let mut sum_sq_ref = 0f64;
    let scale_ref = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    for (g, wv) in got.iter().zip(want.iter()) {
        let d = (f(*g) - wv).abs();
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(d / scale_ref.max(1e-9));
        sum_sq_err += (d as f64) * (d as f64);
        sum_sq_ref += (*wv as f64) * (*wv as f64);
    }
    let rms_rel = (sum_sq_err / sum_sq_ref.max(1e-30)).sqrt();
    eprintln!(
        "fp8 e4m3 per-row-amax on real o_proj shape [{n} x {k}]: max_abs {max_abs:.6e}, \
         max_abs/|y|_inf {max_rel:.6e}, rms_rel {rms_rel:.6e} (|y|_inf {scale_ref:.6e})"
    );
    eprintln!(
        "bytes for this projection: bf16 {} -> fp8 {} ({:.2}x)",
        2 * n * k,
        n * k + 4 * n,
        (2 * n * k) as f64 / (n * k + 4 * n) as f64
    );

    let i8_rel;
    {
        let (wq8, sc8) = nv_kernels::wgpu_backend::kernels::quant_gemv::quantize_groups(
            &w,
            n,
            k,
            128,
            nv_kernels::wgpu_backend::kernels::quant_gemv::QFormat::Int8,
        );
        let mut got8 = vec![0u16; n];
        nv_kernels::wgpu_backend::kernels::quant_gemv::gemv_group_bf16(
            ctx,
            &wq8,
            &sc8,
            &x,
            &mut got8,
            n,
            k,
            128,
            nv_kernels::wgpu_backend::kernels::quant_gemv::QFormat::Int8,
        )
        .unwrap();
        let mut se = 0f64;
        let mut sr = 0f64;
        for (g, wv) in got8.iter().zip(want.iter()) {
            se += ((f(*g) - wv) as f64).powi(2);
            sr += (*wv as f64).powi(2);
        }
        i8_rel = (se / sr).sqrt();
        eprintln!("int8 group=128 on the same shape: rms_rel {i8_rel:.6e}");
    }

    assert!(
        rms_rel > 0.01,
        "with an honest zero-mean generator, e4m3 per-row output error is ~2.6e-2; a value \
         below 1e-2 ({rms_rel}) means the generator regressed to same-sign again"
    );
    assert!(
        rms_rel < 0.05,
        "fp8 row-scaled gemv rms relative error unexpectedly large: {rms_rel}"
    );
    assert!(
        i8_rel * 2.0 < rms_rel,
        "int8 group=128 must beat e4m3 per-row by >2x at the same byte cost: {i8_rel:.4e} vs {rms_rel:.4e}"
    );
}

fn snapshot_dir() -> std::path::PathBuf {
    match std::env::var("NV_GEMMA4_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => {
            let home = std::env::var("HOME").unwrap();
            let base = std::path::PathBuf::from(home)
                .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
            std::fs::read_dir(&base)
                .expect("hub snapshot dir")
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.join("config.json").exists())
                .expect("no snapshot with config.json")
        }
    }
}

#[test]
#[ignore]
fn real_gemma4_31b_lmhead_int8_ms_per_token() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if std::env::var("NV_GEMMA4_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping: set NV_GEMMA4_WGPU_TEST=1 to run");
        return;
    }
    let Some(_ctx) = ctx_or_skip() else { return };
    let dir = snapshot_dir();
    eprintln!("loading Gemma4 config from {}", dir.display());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let t_load = std::time::Instant::now();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader).unwrap();
    eprintln!(
        "host weight staging took {:.1}s",
        t_load.elapsed().as_secs_f64()
    );
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let enc = tokenizer.encode("The capital of France is", false).unwrap();
    let mut prompt: Vec<u32> = vec![2];
    prompt.extend(enc.get_ids());

    let warmup = 8usize;
    let timed = 32usize;

    std::env::remove_var("NV_WGPU_LMHEAD_INT8");
    let t_up = std::time::Instant::now();
    let mut base = Gemma4Wgpu::new(config.clone(), &host, 4096).unwrap();
    eprintln!(
        "[bf16] upload + pass build took {:.1}s ({} passes)",
        t_up.elapsed().as_secs_f64(),
        base.pass_count()
    );
    let mut last = 0u32;
    for t in &prompt {
        last = base.decode_step(*t).unwrap();
    }
    let mut base_ids = vec![last];
    for _ in 0..warmup - 1 {
        last = base.decode_step(last).unwrap();
        base_ids.push(last);
    }
    nvidia_smi_occupancy("bf16-timed");
    let t0 = std::time::Instant::now();
    for _ in 0..timed {
        last = base.decode_step(last).unwrap();
        base_ids.push(last);
    }
    let base_ms = t0.elapsed().as_secs_f64() * 1000.0 / timed as f64;
    eprintln!(
        "[bf16] {timed} tokens -> {base_ms:.2} ms/token ({:.2} tok/s)",
        1000.0 / base_ms
    );
    eprintln!(
        "[bf16] text: {:?}",
        tokenizer.decode(&base_ids, false).unwrap()
    );
    drop(base);

    std::env::set_var("NV_WGPU_LMHEAD_INT8", "1");
    let t_up = std::time::Instant::now();
    let mut i8m = Gemma4Wgpu::new(config.clone(), &host, 4096).unwrap();
    std::env::remove_var("NV_WGPU_LMHEAD_INT8");
    eprintln!(
        "[int8] upload + rowquant + pass build took {:.1}s ({} passes)",
        t_up.elapsed().as_secs_f64(),
        i8m.pass_count()
    );
    let mut agree = 0usize;
    let mut out = 0u32;
    for t in &prompt {
        out = i8m.decode_step(*t).unwrap();
    }
    if out == base_ids[0] {
        agree += 1;
    }
    let n_replay = base_ids.len();
    let mut replay_times = Vec::new();
    nvidia_smi_occupancy("int8-timed");
    for i in 0..n_replay - 1 {
        let t0 = std::time::Instant::now();
        out = i8m.decode_step(base_ids[i]).unwrap();
        replay_times.push(t0.elapsed().as_secs_f64());
        if out == base_ids[i + 1] {
            agree += 1;
        }
    }
    let tail: Vec<f64> = replay_times[replay_times.len() - timed..].to_vec();
    let i8_ms = tail.iter().sum::<f64>() * 1000.0 / tail.len() as f64;
    eprintln!(
        "[int8] {} forced tokens, last {timed} -> {i8_ms:.2} ms/token ({:.2} tok/s)",
        n_replay - 1,
        1000.0 / i8_ms
    );
    eprintln!("[int8] argmax agreement with bf16 under identical context: {agree}/{n_replay}");
    let mut free = Vec::new();
    let mut cur = *base_ids.last().unwrap();
    for _ in 0..16 {
        cur = i8m.decode_step(cur).unwrap();
        free.push(cur);
    }
    eprintln!(
        "[int8] free continuation: {:?}",
        tokenizer.decode(&free, false).unwrap()
    );
    eprintln!(
        "summary: bf16 {base_ms:.2} ms/tok -> int8 {i8_ms:.2} ms/tok (delta {:.2} ms, {:.1}%)",
        base_ms - i8_ms,
        100.0 * (base_ms - i8_ms) / base_ms
    );
    assert!(free.iter().all(|t| (*t as usize) < config.vocab_size));
    assert!(
        agree * 10 >= n_replay * 9,
        "int8 lm_head argmax agreement below 90%: {agree}/{n_replay}"
    );
}

#[test]
#[ignore]
fn nvfp4_checkpoint_ships_packed_mlp_and_bf16_attention() {
    if std::env::var("NV_GEMMA4_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping: set NV_GEMMA4_WGPU_TEST=1 to run");
        return;
    }
    let dir = snapshot_dir();
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    for i in 0..config.num_hidden_layers {
        let p = format!("model.language_model.layers.{i}");
        for m in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
            assert!(
                loader.has(&format!("{p}.{m}.weight_scale"))
                    && loader.has(&format!("{p}.{m}.weight_scale_2")),
                "layer {i} {m}: expected packed NVFP4 in checkpoint"
            );
        }
        for m in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ] {
            let name = format!("{p}.{m}.weight");
            if !loader.has(&name) {
                continue;
            }
            assert!(
                !loader.has(&format!("{p}.{m}.weight_scale")),
                "layer {i} {m}: checkpoint unexpectedly ships packed attention"
            );
        }
    }
    eprintln!(
        "all {} layers: mlp packed NVFP4 (loaded direct), attention bf16 (no packed form exists)",
        config.num_hidden_layers
    );

    let module = "model.language_model.layers.0.mlp.down_proj";
    let n = config.hidden_size;
    let k = config.intermediate_size;
    let packed = loader.raw_bytes(&format!("{module}.weight")).unwrap();
    let scales = loader.raw_bytes(&format!("{module}.weight_scale")).unwrap();
    let g2: Vec<f32> = loader
        .get(&format!("{module}.weight_scale_2"), candle_core::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let global = g2[0];
    let k_blocks = k / 16;
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
    for r in 0..n {
        let mut row = Vec::with_capacity(k);
        for b in 0..k_blocks {
            let sc = nv_quant::nvfp4::decode_ue4m3(scales[r * k_blocks + b]) * global;
            let blk = &packed[r * k / 2 + b * 8..r * k / 2 + b * 8 + 8];
            for byte in blk {
                let (lo, hi) = nv_quant::nvfp4::unpack_e2m1_pair(*byte);
                row.push(nv_quant::nvfp4::decode_e2m1(lo) * sc);
                row.push(nv_quant::nvfp4::decode_e2m1(hi) * sc);
            }
        }
        rows.push(row);
    }
    let bits: Vec<u16> = rows
        .iter()
        .flatten()
        .map(|v| half::bf16::from_f32(*v).to_bits())
        .collect();
    let requant = nv_models::gemma4_wgpu::quantize_nvfp4_host(&bits, n, k);
    let mut byte_mismatch = 0usize;
    for (a, b) in requant.packed.iter().zip(packed.iter()) {
        if a != b {
            byte_mismatch += 1;
        }
    }
    let ckpt_alpha = global;
    eprintln!(
        "requantize-vs-packed on {module}: {}/{} packed bytes differ ({:.2}%), checkpoint global {ckpt_alpha:.6e} vs requant alpha {:.6e}",
        byte_mismatch,
        packed.len(),
        100.0 * byte_mismatch as f64 / packed.len() as f64,
        requant.alpha
    );
    assert_eq!(requant.packed.len(), n * k / 2);
}

#[test]
#[ignore]
fn real_gemma4_31b_o_proj_nvfp4_bytes_vs_time() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if std::env::var("NV_GEMMA4_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping: set NV_GEMMA4_WGPU_TEST=1 to run");
        return;
    }
    let Some(_ctx) = ctx_or_skip() else { return };
    let dir = snapshot_dir();
    eprintln!("loading Gemma4 config from {}", dir.display());
    let mut config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let mut truncated = false;
    if let Some(n) = std::env::var("NV_GEMMA4_LAYERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        truncated = n < config.num_hidden_layers;
        let n = n.min(config.num_hidden_layers);
        eprintln!(
            "NV_GEMMA4_LAYERS={n}: truncating {} -> {n} layers to bound VRAM",
            config.num_hidden_layers
        );
        config.num_hidden_layers = n;
        config.layer_types.truncate(n);
    }
    let t_load = std::time::Instant::now();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader).unwrap();
    drop(loader);
    eprintln!(
        "host weight staging took {:.1}s",
        t_load.elapsed().as_secs_f64()
    );
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let enc = tokenizer.encode("The capital of France is", false).unwrap();
    let mut prompt: Vec<u32> = vec![2];
    prompt.extend(enc.get_ids());

    let warmup = 8usize;
    let timed = 24usize;

    std::env::remove_var("NV_WGPU_O_NVFP4");
    nv_models::gemma4_wgpu::set_attn_variant(Some(nv_models::gemma4_wgpu::AttnVariant {
        on: false,
        ..nv_models::gemma4_wgpu::ATTN_VARIANT_DEFAULT
    }));
    let t_up = std::time::Instant::now();
    let mut base = Gemma4Wgpu::new(config.clone(), &host, 4096).unwrap();
    let base_bytes = base.weight_bytes_per_token();
    let base_passes = base.pass_count();
    eprintln!(
        "[bf16-o] upload + pass build {:.1}s, {base_passes} passes, {base_bytes} weight bytes/token ({:.3} GB)",
        t_up.elapsed().as_secs_f64(),
        base_bytes as f64 / 1e9
    );
    let mut last = 0u32;
    for t in &prompt {
        last = base.decode_step(*t).unwrap();
    }
    let mut base_ids = vec![last];
    for _ in 0..warmup - 1 {
        last = base.decode_step(last).unwrap();
        base_ids.push(last);
    }
    idle_gate("bf16-o");
    nvidia_smi_occupancy("bf16-o-timed");
    let mut base_times = Vec::with_capacity(timed);
    for _ in 0..timed {
        let t0 = std::time::Instant::now();
        last = base.decode_step(last).unwrap();
        base_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        base_ids.push(last);
    }
    let base_st = MsStats::of(&base_times);
    let base_ms = base_st.median;
    eprintln!(
        "[bf16-o] {timed} tokens -> {base_st} ; median {base_ms:.2} ms/token ({:.2} tok/s), implied {:.1} GB/s at median",
        1000.0 / base_ms,
        base_bytes as f64 / 1e9 / (base_ms / 1000.0)
    );
    eprintln!(
        "[bf16-o] text: {:?}",
        tokenizer.decode(&base_ids, false).unwrap()
    );
    drop(base);

    std::env::set_var("NV_WGPU_O_NVFP4", "1");
    let t_up = std::time::Instant::now();
    let mut q = Gemma4Wgpu::new(config.clone(), &host, 4096).unwrap();
    std::env::remove_var("NV_WGPU_O_NVFP4");
    nv_models::gemma4_wgpu::set_attn_variant(None);
    let q_bytes = q.weight_bytes_per_token();
    let q_passes = q.pass_count();
    eprintln!(
        "[nvfp4-o] host quantize + upload + pass build {:.1}s, {q_passes} passes, {q_bytes} weight bytes/token ({:.3} GB)",
        t_up.elapsed().as_secs_f64(),
        q_bytes as f64 / 1e9
    );
    let mut agree = 0usize;
    let mut out = 0u32;
    for t in &prompt {
        out = q.decode_step(*t).unwrap();
    }
    if out == base_ids[0] {
        agree += 1;
    }
    let n_replay = base_ids.len();
    let mut replay_times = Vec::new();
    idle_gate("nvfp4-o");
    nvidia_smi_occupancy("nvfp4-o-timed");
    for i in 0..n_replay - 1 {
        let t0 = std::time::Instant::now();
        out = q.decode_step(base_ids[i]).unwrap();
        replay_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        if out == base_ids[i + 1] {
            agree += 1;
        }
    }
    let tail: Vec<f64> = replay_times[replay_times.len() - timed..].to_vec();
    let q_st = MsStats::of(&tail);
    let q_ms = q_st.median;
    eprintln!(
        "[nvfp4-o] {} forced tokens, last {timed} -> {q_st} ; median {q_ms:.2} ms/token ({:.2} tok/s), implied {:.1} GB/s at median",
        n_replay - 1,
        1000.0 / q_ms,
        q_bytes as f64 / 1e9 / (q_ms / 1000.0)
    );
    eprintln!("[nvfp4-o] argmax agreement with bf16 under identical context: {agree}/{n_replay}");
    let mut free = Vec::new();
    let mut cur = *base_ids.last().unwrap();
    for _ in 0..16 {
        cur = q.decode_step(cur).unwrap();
        free.push(cur);
    }
    eprintln!(
        "[nvfp4-o] free continuation: {:?}",
        tokenizer.decode(&free, false).unwrap()
    );

    let byte_cut = 100.0 * (base_bytes as f64 - q_bytes as f64) / base_bytes as f64;
    for (label, b_ms, q_msx) in [
        ("median", base_st.median, q_st.median),
        ("p10", base_st.p10, q_st.p10),
        ("min", base_st.min, q_st.min),
        ("mean", base_st.mean, q_st.mean),
    ] {
        let time_cut = 100.0 * (b_ms - q_msx) / b_ms;
        eprintln!(
            "PREMISE[{label}]: bytes/token {base_bytes} -> {q_bytes} ({byte_cut:.2}% cut); time {b_ms:.2} -> {q_msx:.2} ms/tok ({time_cut:+.2}% cut); conversion ratio {:+.3}",
            time_cut / byte_cut
        );
    }
    eprintln!(
        "PREMISE: passes {base_passes} -> {q_passes}; conversion ratio 1.0 = purely bandwidth-bound, 0.0 = byte reduction buys nothing, negative = byte reduction costs time"
    );
    eprintln!("PREMISE: CONTAMINATED unless both occupancy lines above read 0% external util");
    assert!(
        byte_cut > 5.0,
        "expected a material byte cut, got {byte_cut:.2}%"
    );
    assert!(free.iter().all(|t| (*t as usize) < config.vocab_size));
    if truncated {
        eprintln!(
            "[nvfp4-o] argmax agreement {agree}/{n_replay} NOT asserted: a truncated model emits degenerate near-tied logits"
        );
    } else {
        assert!(
            agree * 10 >= n_replay * 9,
            "nvfp4 o_proj argmax agreement below 90%: {agree}/{n_replay}"
        );
    }
}

#[test]
#[ignore]
fn real_gemma4_31b_attn_fp8_agreement_bytes_and_time() {
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if std::env::var("NV_GEMMA4_WGPU_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping: set NV_GEMMA4_WGPU_TEST=1 to run");
        return;
    }
    let Some(_ctx) = ctx_or_skip() else { return };
    let dir = snapshot_dir();
    eprintln!("loading Gemma4 config from {}", dir.display());
    let config = Gemma4Config::from_hf_json_file(&dir.join("config.json")).unwrap();
    let t_load = std::time::Instant::now();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).unwrap();
    let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader).unwrap();
    drop(loader);
    eprintln!(
        "host weight staging took {:.1}s",
        t_load.elapsed().as_secs_f64()
    );
    for (i, l) in host.layers.iter().enumerate().take(2) {
        eprintln!(
            "layer {i}: qkv n={} k={} (k%32={}), o n={} k={} (k%32={})",
            l.qkv.n(),
            l.qkv.k(),
            l.qkv.k() % 32,
            l.o.n(),
            l.o.k(),
            l.o.k() % 32
        );
    }
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let enc = tokenizer.encode("The capital of France is", false).unwrap();
    let mut prompt: Vec<u32> = vec![2];
    prompt.extend(enc.get_ids());

    let warmup = 8usize;
    let timed = 32usize;

    nv_models::gemma4_wgpu::set_attn_variant(Some(nv_models::gemma4_wgpu::AttnVariant {
        on: false,
        ..nv_models::gemma4_wgpu::ATTN_VARIANT_DEFAULT
    }));
    let t_up = std::time::Instant::now();
    let mut base = Gemma4Wgpu::new(config.clone(), &host, 4096).unwrap();
    nv_models::gemma4_wgpu::set_attn_variant(None);
    let base_bytes = base.weight_bytes_per_token();
    let base_passes = base.pass_count();
    eprintln!(
        "[bf16-attn] upload + pass build {:.1}s, {base_passes} passes, {base_bytes} weight bytes/token ({:.3} GB)",
        t_up.elapsed().as_secs_f64(),
        base_bytes as f64 / 1e9
    );
    let mut last = 0u32;
    for t in &prompt {
        last = base.decode_step(*t).unwrap();
    }
    let mut base_ids = vec![last];
    for _ in 0..warmup - 1 {
        last = base.decode_step(last).unwrap();
        base_ids.push(last);
    }
    nvidia_smi_occupancy("bf16-attn-timed");
    let t0 = std::time::Instant::now();
    for _ in 0..timed {
        last = base.decode_step(last).unwrap();
        base_ids.push(last);
    }
    let base_ms = t0.elapsed().as_secs_f64() * 1000.0 / timed as f64;
    eprintln!(
        "[bf16-attn] {timed} tokens -> {base_ms:.2} ms/token ({:.2} tok/s), implied {:.1} GB/s",
        1000.0 / base_ms,
        base_bytes as f64 / 1e9 / (base_ms / 1000.0)
    );
    eprintln!(
        "[bf16-attn] text: {:?}",
        tokenizer.decode(&base_ids, false).unwrap()
    );
    drop(base);

    nv_models::gemma4_wgpu::set_attn_variant(Some(nv_models::gemma4_wgpu::ATTN_VARIANT_DEFAULT));
    let t_up = std::time::Instant::now();
    let mut q = Gemma4Wgpu::new(config.clone(), &host, 4096).unwrap();
    nv_models::gemma4_wgpu::set_attn_variant(None);
    let q_bytes = q.weight_bytes_per_token();
    let q_passes = q.pass_count();
    eprintln!(
        "[fp8-attn] host quantize + upload + pass build {:.1}s, {q_passes} passes, {q_bytes} weight bytes/token ({:.3} GB)",
        t_up.elapsed().as_secs_f64(),
        q_bytes as f64 / 1e9
    );
    let mut agree = 0usize;
    let mut first_div: Option<usize> = None;
    let mut out = 0u32;
    for t in &prompt {
        out = q.decode_step(*t).unwrap();
    }
    if out == base_ids[0] {
        agree += 1;
    } else {
        first_div = Some(0);
    }
    let n_replay = base_ids.len();
    let mut replay_times = Vec::new();
    nvidia_smi_occupancy("fp8-attn-timed");
    for i in 0..n_replay - 1 {
        let t0 = std::time::Instant::now();
        out = q.decode_step(base_ids[i]).unwrap();
        replay_times.push(t0.elapsed().as_secs_f64());
        if out == base_ids[i + 1] {
            agree += 1;
        } else if first_div.is_none() {
            first_div = Some(i + 1);
            eprintln!(
                "[fp8-attn] first argmax divergence at forced position {}: bf16 {} vs fp8 {}",
                i + 1,
                base_ids[i + 1],
                out
            );
        }
    }
    let tail: Vec<f64> = replay_times[replay_times.len() - timed..].to_vec();
    let q_ms = tail.iter().sum::<f64>() * 1000.0 / tail.len() as f64;
    eprintln!(
        "[fp8-attn] {} forced tokens, last {timed} -> {q_ms:.2} ms/token ({:.2} tok/s), implied {:.1} GB/s",
        n_replay - 1,
        1000.0 / q_ms,
        q_bytes as f64 / 1e9 / (q_ms / 1000.0)
    );
    eprintln!(
        "[fp8-attn] argmax agreement with bf16 under identical context: {agree}/{n_replay} (first divergence {first_div:?})"
    );
    let mut free = Vec::new();
    let mut cur = *base_ids.last().unwrap();
    for _ in 0..16 {
        cur = q.decode_step(cur).unwrap();
        free.push(cur);
    }
    eprintln!(
        "[fp8-attn] free continuation: {:?}",
        tokenizer.decode(&free, false).unwrap()
    );

    let byte_cut = 100.0 * (base_bytes as f64 - q_bytes as f64) / base_bytes as f64;
    let time_cut = 100.0 * (base_ms - q_ms) / base_ms;
    eprintln!(
        "SUMMARY(CONTAMINATED): bytes/token {base_bytes} -> {q_bytes} ({byte_cut:.2}% cut); time {base_ms:.2} -> {q_ms:.2} ms/tok ({time_cut:.2}% cut); passes {base_passes} -> {q_passes}"
    );
    eprintln!(
        "SUMMARY(CONTAMINATED): conversion ratio time_cut/byte_cut = {:.3} (1.0 = purely bandwidth-bound, 0.0 = byte reduction buys nothing)",
        time_cut / byte_cut
    );
    assert_eq!(
        q_passes, base_passes,
        "fp8 attention must keep the dispatch count identical"
    );
    assert!(
        byte_cut > 5.0,
        "expected a material byte cut, got {byte_cut:.2}%"
    );
    assert!(free.iter().all(|t| (*t as usize) < config.vocab_size));
    assert!(
        agree * 10 >= n_replay * 9,
        "fp8 attention argmax agreement below 90%: {agree}/{n_replay}"
    );
}

#[test]
fn wgpu_weight_format_support_matrix() {
    use nv_kernels::wgpu_backend::kernels as wk;
    struct Row {
        fmt: &'static str,
        ckpt: bool,
        requant: bool,
        kernel: bool,
        kernels: &'static str,
        test: &'static str,
    }
    let rows = vec![
        Row {
            fmt: "nvfp4 (e2m1 + ue4m3 block scale, gs=16)",
            ckpt: true,
            requant: true,
            kernel: !wk::gemv_nvfp4::gemv_source().is_empty(),
            kernels: "gemv_nvfp4 {Tree,Sg} + gemm_nvfp4 + quantize_nvfp4_bf16",
            test: "real_gemma4_31b_wgpu_decode_ms_per_token; synthetic_o_proj_nvfp4_agreement_and_bytes",
        },
        Row {
            fmt: "bf16",
            ckpt: true,
            requant: false,
            kernel: !wk::gemv_bf16::WGSL.is_empty(),
            kernels: "gemv_bf16 vec8 + local pk/pk3 split epilogues",
            test: "real_gemma4_bf16_checkpoint_truncated_decode; cuda_vs_wgpu_synthetic_decode_logits",
        },
        Row {
            fmt: "fp8 (e4m3 per-row scale)",
            ckpt: false,
            requant: true,
            kernel: !wk::quant_gemv::source().is_empty(),
            kernels: "quant_gemv gemv_fp8_rowscale/_sg + local fp8 pk/pk3 epilogues; kv_fp8/kv_fp8_paged for KV",
            test: "synthetic_attn_fp8_agreement_bytes_and_determinism; fp8_row_quant_error_on_real_o_proj_shape",
        },
        Row {
            fmt: "int8 (per-row scale)",
            ckpt: false,
            requant: true,
            kernel: !wk::quant_gemv::source().is_empty() && !wk::gemv_bf16::WGSL.is_empty(),
            kernels: "quant_gemv gemv_int8_rowscale/_sg + gemv_bf16 ROWQUANT/I8_NORMED",
            test: "synthetic_lmhead_int8_agreement_and_determinism",
        },
        Row {
            fmt: "mxfp4 (e2m1 + e8m0 block scale, gs=32)",
            ckpt: false,
            requant: false,
            kernel: !wk::quant_gemv::source().is_empty(),
            kernels: "quant_gemv gemv_mxfp4/_sg",
            test: "nv-kernels only; no gemma4_wgpu caller",
        },
        Row {
            fmt: "w4a16 (int4 group-quant)",
            ckpt: false,
            requant: false,
            kernel: !wk::gemv_w4a16::WGSL.is_empty(),
            kernels: "gemv_w4a16 block/row/gelu_pli + gemv_w4a16_m1_proto",
            test: "nv-kernels only; no gemma4_wgpu caller",
        },
    ];
    let yn = |b: bool| if b { "YES" } else { "no " };
    eprintln!("format | ckpt-loader | load-time-requant | kernel | kernels | model-layer test");
    for r in &rows {
        eprintln!(
            "{} | {} | {} | {} | {} | {}",
            r.fmt,
            yn(r.ckpt),
            yn(r.requant),
            yn(r.kernel),
            r.kernels,
            r.test
        );
        assert!(r.kernel, "{}: expected a wgpu kernel to exist", r.fmt);
    }
    let ckpt: Vec<&str> = rows.iter().filter(|r| r.ckpt).map(|r| r.fmt).collect();
    let usable: Vec<&str> = rows
        .iter()
        .filter(|r| r.ckpt || r.requant)
        .map(|r| r.fmt)
        .collect();
    assert_eq!(
        ckpt.len(),
        2,
        "gemma4_wgpu reads exactly bf16 + nvfp4 from a checkpoint, got {ckpt:?}"
    );
    assert_eq!(
        usable.len(),
        4,
        "gemma4_wgpu can run bf16/nvfp4/fp8/int8 weights, got {usable:?}"
    );
    eprintln!("read from checkpoint: {ckpt:?}");
    eprintln!("runnable in the model layer (checkpoint or load-time requantize): {usable:?}");
    eprintln!(
        "kernels exist for all {} formats; the gap is loader/model-layer wiring, not shaders",
        rows.len()
    );
}
