#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use nv_kernels::cuda;

fn run_sampler(
    ctx: &std::sync::Arc<CudaContext>,
    logits: &[f32],
    seeds: &[u64],
    batch: usize,
    vocab: usize,
    temperature: f32,
    top_k: u32,
    top_p: f32,
) -> (Vec<f32>, Vec<u32>) {
    let stream = ctx.default_stream();
    #[allow(deprecated)]
    let dl: CudaSlice<f32> = stream.memcpy_stod(logits).unwrap();
    #[allow(deprecated)]
    let ds: CudaSlice<u64> = stream.memcpy_stod(seeds).unwrap();
    let mut dp: CudaSlice<f32> = stream.alloc_zeros::<f32>(batch * vocab).unwrap();
    let mut dt: CudaSlice<u32> = stream.alloc_zeros::<u32>(batch).unwrap();

    let rc = {
        let (pl, _g1) = dl.device_ptr(&stream);
        let (ps, _g2) = ds.device_ptr(&stream);
        let (pp, _g3) = dp.device_ptr_mut(&stream);
        let (pt, _g4) = dt.device_ptr_mut(&stream);
        unsafe {
            cuda::sampler_topk_topp(
                stream.cu_stream() as *mut _,
                pl as *const f32,
                ps as *const u64,
                pp as *mut f32,
                pt as *mut u32,
                batch,
                vocab,
                temperature,
                top_k,
                top_p,
            )
        }
    };
    assert_eq!(rc, 0, "kernel launch returned {rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let probs = stream.memcpy_dtov(&dp).unwrap();
    #[allow(deprecated)]
    let toks = stream.memcpy_dtov(&dt).unwrap();
    (probs, toks)
}

#[test]
fn sampler_softmax_probs_sum_to_one() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "sampler: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("sampler: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };

    let batch = 3usize;
    let vocab = 4096usize;
    let mut logits = Vec::with_capacity(batch * vocab);
    for b in 0..batch {
        for i in 0..vocab {
            logits.push(((b as f32 * 0.1 + i as f32) * 0.0013).sin() * 3.0);
        }
    }
    let seeds: Vec<u64> = (0..batch).map(|i| 0xdeadbeefu64 + i as u64).collect();

    let (probs, _toks) = run_sampler(&ctx, &logits, &seeds, batch, vocab, 1.0, 0, 1.0);
    for b in 0..batch {
        let s: f32 = probs[b * vocab..(b + 1) * vocab].iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "row {b} sum {s} not close to 1.0");
    }
}

#[test]
fn sampler_argmax_via_top_k_one() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "sampler: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("sampler: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };

    let batch = 2usize;
    let vocab = 1024usize;
    let mut logits = vec![0f32; batch * vocab];
    let truth = [573usize, 42usize];
    for b in 0..batch {
        for i in 0..vocab {
            logits[b * vocab + i] = ((i as f32) * 0.001).sin();
        }
        logits[b * vocab + truth[b]] = 100.0;
    }
    let seeds: Vec<u64> = (0..batch).map(|i| 7u64 + i as u64).collect();

    let (_probs, toks) = run_sampler(&ctx, &logits, &seeds, batch, vocab, 1.0, 1, 1.0);
    for b in 0..batch {
        assert_eq!(
            toks[b] as usize, truth[b],
            "row {b} expected {}, got {}",
            truth[b], toks[b]
        );
    }
}

#[test]
fn sampler_argmax_via_low_temperature() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "sampler: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("sampler: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };

    let batch = 2usize;
    let vocab = 1024usize;
    let mut logits = vec![0f32; batch * vocab];
    let truth = [573usize, 42usize];
    for b in 0..batch {
        for i in 0..vocab {
            logits[b * vocab + i] = ((i as f32) * 0.001).sin();
        }
        logits[b * vocab + truth[b]] = 100.0;
    }
    let seeds: Vec<u64> = (0..batch).map(|i| 99u64 + i as u64).collect();

    let (_probs, toks) = run_sampler(&ctx, &logits, &seeds, batch, vocab, 1e-4, 0, 1.0);
    for b in 0..batch {
        assert_eq!(
            toks[b] as usize, truth[b],
            "low-temp row {b} expected {}, got {}",
            truth[b], toks[b]
        );
    }
}

#[test]
fn sampler_same_seed_deterministic() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "sampler: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("sampler: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };

    let batch = 4usize;
    let vocab = 2048usize;
    let mut logits = Vec::with_capacity(batch * vocab);
    for b in 0..batch {
        for i in 0..vocab {
            logits.push(((b as f32 * 0.7 + i as f32) * 0.001).sin() * 4.0);
        }
    }
    let seeds = vec![1234u64; batch];

    let (_, toks_a) = run_sampler(&ctx, &logits, &seeds, batch, vocab, 1.0, 0, 1.0);
    let (_, toks_b) = run_sampler(&ctx, &logits, &seeds, batch, vocab, 1.0, 0, 1.0);
    assert_eq!(toks_a, toks_b, "same seed should give deterministic tokens");
}

#[test]
fn sampler_different_seeds_differ() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "sampler: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("sampler: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };

    let batch = 1usize;
    let vocab = 4096usize;
    let mut logits = Vec::with_capacity(vocab);
    for i in 0..vocab {
        logits.push(((i as f32) * 0.003).sin() * 2.0);
    }

    let n = 64;
    let mut picks = std::collections::HashSet::new();
    for s in 0..n {
        let seeds = vec![s as u64 * 7919u64 + 1u64];
        let (_, toks) = run_sampler(&ctx, &logits, &seeds, batch, vocab, 1.0, 0, 1.0);
        picks.insert(toks[0]);
    }
    assert!(
        picks.len() > 1,
        "expected diverse samples across {n} seeds, got {}",
        picks.len()
    );
}

#[test]
fn sampler_top_k_restricts_support() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "sampler: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("sampler: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };

    let batch = 1usize;
    let vocab = 4096usize;
    let mut logits = Vec::with_capacity(vocab);
    for i in 0..vocab {
        logits.push(((i as f32) * 0.003).sin() * 2.0);
    }
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_k = 16usize;
    let allowed: std::collections::HashSet<u32> =
        indexed.iter().take(top_k).map(|(i, _)| *i as u32).collect();

    let trials = 256;
    let mut all_in = true;
    for s in 0..trials {
        let seeds = vec![s as u64 * 31u64 + 11u64];
        let (_, toks) = run_sampler(&ctx, &logits, &seeds, batch, vocab, 1.0, top_k as u32, 1.0);
        if !allowed.contains(&toks[0]) {
            all_in = false;
            eprintln!("seed {} produced out-of-top-k token {}", s, toks[0]);
            break;
        }
    }
    assert!(
        all_in,
        "top-k mask leaked tokens outside the top-{top_k} set"
    );
}

#[test]
fn sampler_top_p_concentrates_mass() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "sampler: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("sampler: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };

    let batch = 1usize;
    let vocab = 4096usize;
    let mut logits = Vec::with_capacity(vocab);
    for i in 0..vocab {
        logits.push(((i as f32) * 0.003).sin() * 4.0);
    }

    let top_p = 0.5f32;
    let mut base = logits.clone();
    let m = base.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = base.iter().map(|v| (*v - m).exp()).collect();
    let sum: f32 = exps.iter().sum();
    base = exps.iter().map(|e| e / sum).collect();
    let mut indexed: Vec<(usize, f32)> = base.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut cum = 0f32;
    let mut nucleus_set = std::collections::HashSet::new();
    for (idx, p) in indexed.iter() {
        cum += p;
        nucleus_set.insert(*idx as u32);
        if cum >= top_p {
            break;
        }
    }
    let extra_slack: f32 = 0.05;
    cum = 0f32;
    let mut nucleus_with_slack = std::collections::HashSet::new();
    for (idx, p) in indexed.iter() {
        cum += p;
        nucleus_with_slack.insert(*idx as u32);
        if cum >= top_p + extra_slack {
            break;
        }
    }

    let trials = 256;
    let mut leaked = 0;
    for s in 0..trials {
        let seeds = vec![s as u64 * 13u64 + 3u64];
        let (_, toks) = run_sampler(&ctx, &logits, &seeds, batch, vocab, 1.0, 0, top_p);
        if !nucleus_with_slack.contains(&toks[0]) {
            leaked += 1;
        }
    }
    assert!(
        leaked == 0,
        "top-p produced {leaked}/{trials} samples outside nucleus+slack"
    );
}

#[test]
fn sampler_probs_sum_after_filtering() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "sampler: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("sampler: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };

    let batch = 1usize;
    let vocab = 2048usize;
    let mut logits = Vec::with_capacity(vocab);
    for i in 0..vocab {
        logits.push(((i as f32) * 0.005).sin() * 3.0);
    }
    let seeds = vec![42u64];

    let (probs, _) = run_sampler(&ctx, &logits, &seeds, batch, vocab, 1.0, 32, 0.9);
    let s: f32 = probs.iter().sum();
    assert!(
        (s - 1.0).abs() < 1e-4,
        "filtered prob sum {s} not close to 1.0"
    );
    let nonzero = probs.iter().filter(|p| **p > 0.0).count();
    assert!(nonzero <= 32, "top-k mask kept {nonzero} > 32 tokens");
}
