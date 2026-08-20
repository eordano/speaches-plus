#![cfg(feature = "wgpu")]

mod common;
use common::flash_gqa_fold::{inputs, Inputs, Shape, SPLITS};
use common::ctx;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::flash_decode as fd;
use nv_kernels::wgpu_backend::{compose, dispatch};
use nv_models::gemma4_e4b_wgpu::{flash1_e4b_entry, flash1_e4b_source, flash1_sg_supported};
use std::time::Instant;

const BATCH: usize = 64;

fn time_stage1(
    c: &WgpuContext,
    s: &Shape,
    inp: &Inputs,
    label: &str,
    source: &str,
    entry: &str,
    grid_x: u32,
) -> f64 {
    let scratch = dispatch::storage_zeroed(c, "fb-scratch", (s.scratch_elems() * 4) as u64);
    let pipeline = dispatch::cached_compute_pipeline(c, label, source, entry)
        .unwrap_or_else(|e| panic!("{label}: pipeline: {e}"));
    let bg = dispatch::bind_group(
        c,
        &pipeline,
        &[
            (0, &inp.q),
            (4, &inp.p),
            (5, &inp.k),
            (6, &inp.v),
            (7, &scratch),
            (8, &inp.ks),
            (9, &inp.vs),
        ],
    );
    let passes: Vec<dispatch::PassRef<'_>> =
        (0..BATCH).map(|_| (&*pipeline, &bg, (grid_x, SPLITS, 1))).collect();
    let labels: Vec<&str> = vec![label; BATCH];

    let mut go = || {
        dispatch::submit_passes(c, &passes, &labels)
            .unwrap_or_else(|e| panic!("{label}: submit: {e}"));
        c.device.poll(wgpu::PollType::wait_indefinitely()).ok();
    };
    for _ in 0..2 {
        go();
    }
    let reps = 5;
    let t0 = Instant::now();
    for _ in 0..reps {
        go();
    }
    t0.elapsed().as_secs_f64() * 1e6 / (reps * BATCH) as f64
}

#[test]
#[ignore]
fn what_the_gqa_fold_is_worth() {
    if std::env::var("NV_WGPU_FOLD_BENCH").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_WGPU_FOLD_BENCH=1");
    }
    let c = ctx();
    let mut reduce_arms = vec![false];
    if flash1_sg_supported(c) {
        reduce_arms.push(true);
    }
    for sg in reduce_arms {
        for total in [256u32, 1024, 4096, 16384, 65536] {
        let s = Shape {
            n_q: 16,
            n_kv: 2,
            hd: 256,
            total,
            start: 0,
        };
        let inp = inputs(c, &s, 0xC0FFEE);
        let base = time_stage1(
            c,
            &s,
            &inp,
            "fb-plain",
            &format!("{}\n{}", compose(fd::WGSL), flash1_e4b_source(s.hd, sg)),
            &flash1_e4b_entry(s.hd, sg),
            s.n_q,
        );

        let bytes = (s.total as f64) * (s.n_kv as f64) * (s.hd as f64) * 2.0;
        let mut line = format!(
            "[fold-bench] ctx {total:6} sg={sg}: fold 1 {base:9.2} us ({:6.1} GB/s unique)",
            bytes / base / 1e3
        );
        for fold in [2u32, 4, 8] {
            assert_eq!(
                (s.n_q / s.n_kv) % fold,
                0,
                "fold {fold} must divide the group"
            );
            let us = time_stage1(
                c,
                &s,
                &inp,
                "fb-folded",
                &format!(
                    "{}\n{}",
                    compose(fd::WGSL),
                    fd::fold_stage1_source(s.hd, sg, fold)
                ),
                &fd::fold_stage1_entry(s.hd, sg, fold),
                s.n_q / fold,
            );
            line.push_str(&format!(" | fold {fold} {us:9.2} us {:.3}x", us / base));
        }
        eprintln!("{line}");
        }
    }
}
