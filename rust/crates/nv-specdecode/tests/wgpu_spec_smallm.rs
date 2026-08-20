#![cfg(feature = "wgpu")]

use nv_kernels::wgpu_backend::WgpuContext;
use nv_specdecode::wgpu_spec::{SpecDims, SpecWeights, WgpuChainSpec, WgpuSpecModel};
mod common;
use common::spec_dims_tiny as dims;

fn ctx(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(c) => {
            eprintln!("{test}: {}", c.summary());
            Some(c)
        }
        Err(e) => {
            if std::env::var("NV_KERNELS_WGPU_ALLOW_SKIP").as_deref() == Ok("1") {
                eprintln!("SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1): {test}: no wgpu adapter: {e}. Not a pass.");
                return None;
            }
            panic!(
                "{test}: no wgpu adapter: {e}. The NV_KERNELS_WGPU_REQUIRE hatch this file used \
                 to carry was never armed -- nvk.sh exports only NV_KERNELS_PARITY_REQUIRE -- so \
                 the loud path was dead code. Set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
            );
        }
    }
}

fn prompt() -> Vec<u32> {
    vec![1, 17, 42, 5, 90, 33, 7]
}

fn wgpu_greedy(c: &'static WgpuContext, d: SpecDims, w: &SpecWeights, n: usize) -> Vec<u32> {
    let mut m = WgpuSpecModel::new_with_smallm(c, d, w, 1, true).unwrap();
    let mut out = Vec::with_capacity(n);
    let mut cur = m.prefill(&prompt()).unwrap();
    out.push(cur);
    while out.len() < n {
        cur = m.decode1(cur).unwrap();
        out.push(cur);
    }
    out
}

#[test]
fn smallm_verify_chain_matches_stepped_decode1() {
    let Some(c) = ctx("smallm_verify_chain_matches_stepped_decode1") else {
        return;
    };
    let d = dims();
    let w = SpecWeights::synthetic(&d, 3);
    let k = 4usize;
    let mut m = WgpuSpecModel::new_with_smallm(c, d, &w, k, true).unwrap();
    assert!(m.smallm(), "k=4 hd=8 must run the small-m attention path");
    m.prefill(&prompt()).unwrap();

    let mut seed = 11u32;
    for round in 0..8 {
        let batch: Vec<u32> = (0..k)
            .map(|i| (seed + (round * 13 + i * 7) as u32) % d.vocab as u32)
            .collect();
        let pos0 = m.committed();
        let va = m.verify_chain(&batch).unwrap();
        assert_eq!(m.committed(), pos0, "verify_chain must not advance");
        let sa: Vec<u32> = batch.iter().map(|&t| m.decode1(t).unwrap()).collect();
        m.rollback_to(pos0).unwrap();
        assert_eq!(va, sa, "round {round}: small-m verify != stepped decode1");
        m.advance(1 + round % k).unwrap();
        seed = va[0];
    }
}

#[test]
fn smallm_spec_loop_stays_lossless() {
    let Some(c) = ctx("smallm_spec_loop_stays_lossless") else {
        return;
    };
    let d = dims();
    let wv = SpecWeights::synthetic(&d, 3);
    let mut wd = SpecWeights::synthetic(&d, 3);
    for (i, v) in wd.wlm.iter_mut().enumerate() {
        *v += 0.015 * ((i as f32) * 0.911).sin();
    }
    let n = 48;
    let mut spec = WgpuChainSpec::new_with_smallm(c, d, &wv, &wd, 4, true).unwrap();
    assert!(spec.verifier.smallm());
    let stats = spec.generate(&prompt(), n).unwrap();
    assert!(stats.emitted.len() >= n);
    let greedy = wgpu_greedy(c, d, &wv, stats.emitted.len());
    assert_eq!(
        stats.emitted, greedy,
        "small-m spec stream must equal the small-m greedy stream"
    );
    let rate = stats.acceptance_rate();
    println!(
        "small-m wgpu chain spec: rounds={} drafted={} accepted={} acceptance_rate={rate:.3}",
        stats.rounds, stats.drafted, stats.accepted_drafts
    );
    assert!(rate > 0.0 && rate <= 1.0);
}

#[test]
fn smallm_falls_back_when_k_exceeds_max_m() {
    let Some(c) = ctx("smallm_falls_back_when_k_exceeds_max_m") else {
        return;
    };
    let d = dims();
    let w = SpecWeights::synthetic(&d, 3);
    let mut m = WgpuSpecModel::new_with_smallm(c, d, &w, 12, true).unwrap();
    assert!(!m.smallm(), "k_max 12 > MAX_M must fall back to sp_attn");
    let t = m.prefill(&prompt()).unwrap();
    assert!((t as usize) < d.vocab);
}

fn gpu_idle_pct() -> Option<u32> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let util = text.lines().next()?.trim().parse::<u32>().ok()?;
    Some(100u32.saturating_sub(util))
}

fn wait_for_idle(min_pct: u32, timeout: std::time::Duration) -> bool {
    let start = std::time::Instant::now();
    loop {
        match gpu_idle_pct() {
            Some(p) if p >= min_pct => {
                eprintln!("idle gate: {p}% idle, proceeding");
                return true;
            }
            Some(p) => eprintln!("idle gate: {p}% idle, waiting"),
            None => {
                eprintln!("idle gate: nvidia-smi unavailable, proceeding without gate");
                return true;
            }
        }
        if start.elapsed() >= timeout {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_secs(20));
    }
}

#[test]
#[ignore]
fn bench_verify_chain_sp_attn_vs_smallm() {
    if std::env::var("NV_SMALLM_BENCH").as_deref() != Ok("1") {
        eprintln!("bench_verify_chain_sp_attn_vs_smallm: SKIP set NV_SMALLM_BENCH=1 to run");
        return;
    }
    let Some(c) = ctx("bench_verify_chain_sp_attn_vs_smallm") else {
        return;
    };
    let quiet = wait_for_idle(85, std::time::Duration::from_secs(10 * 60));

    let d = SpecDims {
        h: 128,
        nh: 8,
        nkv: 4,
        hd: 64,
        inter: 256,
        vocab: 256,
        max_seq: 1200,
        eps: 1e-5,
        rope_theta: 10000.0,
    };
    let w = SpecWeights::synthetic(&d, 5);
    let k = 8usize;
    let committed = 1100usize;
    let batch: Vec<u32> = (0..k as u32)
        .map(|i| (i * 31 + 3) % d.vocab as u32)
        .collect();
    let warmup = 5usize;
    let iters = 40usize;

    let time_one = |smallm: bool| -> f64 {
        let mut m = WgpuSpecModel::new_with_smallm(c, d, &w, k, smallm).unwrap();
        assert_eq!(m.smallm(), smallm);
        m.advance(committed).unwrap();
        for _ in 0..warmup {
            m.verify_chain(&batch).unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            m.verify_chain(&batch).unwrap();
        }
        t0.elapsed().as_secs_f64() * 1e6 / iters as f64
    };

    let legacy_us = time_one(false);
    let smallm_us = time_one(true);
    println!(
        "wgpu_spec verify_chain bench (h=128 nh=8 nkv=4 hd=64, committed={committed}, k={k}, {}):",
        if quiet {
            "quiet window"
        } else {
            "PROVISIONAL: contended"
        }
    );
    println!("  sp_attn legacy : {legacy_us:>10.1} us/call");
    println!("  small-m f32    : {smallm_us:>10.1} us/call");
    println!("  ratio legacy/smallm: {:.2}x", legacy_us / smallm_us);
    println!("  PROVISIONAL: co-tenant GPU, whole verify-graph timing on a synthetic model");
}
