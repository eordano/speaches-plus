#![cfg(feature = "wgpu")]

mod common;
use common::LcgShift32TwoSided as Lcg;
use common::wgpu_allow_skip;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::{gemv_bf16, gemv_nvfp4, quantize_nvfp4_bf16 as qz};
use common::gpu_util;

fn ctx_or_skip(test: &str) -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("{test}: {}", ctx.summary());
            Some(ctx)
        }
        Err(e) => {
            if !wgpu_allow_skip() {
                panic!(
                    "{test}: no wgpu adapter: {e}. This gate refuses to report success \
                     without running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
                );
            }
            eprintln!("{test}: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) no wgpu adapter: {e}");
            None
        }
    }
}

fn wait_idle(tag: &str) {
    let t0 = std::time::Instant::now();
    let mut streak = 0;
    loop {
        match gpu_util() {
            Some(u) if u <= 2 => streak += 1,
            Some(_) => streak = 0,
            None => return,
        }
        if streak >= 2 {
            eprintln!("{tag}: gpu idle after {:.0}s", t0.elapsed().as_secs_f64());
            return;
        }
        if t0.elapsed().as_secs_f64() > 600.0 {
            eprintln!("{tag}: WARNING gpu never idle within 600s");
            return;
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
}

#[test]
fn adaptive_width_selection_covers_8_16_32_64_and_variable() {
    assert_eq!(gemv_bf16::adaptive_width(true, 8, 8, None), Some(8));
    assert_eq!(gemv_bf16::adaptive_width(true, 16, 16, None), Some(16));
    assert_eq!(gemv_bf16::adaptive_width(true, 32, 32, None), Some(32));
    assert_eq!(gemv_bf16::adaptive_width(true, 64, 64, None), Some(32));
    assert_eq!(gemv_bf16::adaptive_width(true, 8, 32, None), Some(8));
    assert_eq!(gemv_bf16::adaptive_width(true, 16, 64, None), Some(16));
    assert_eq!(gemv_bf16::adaptive_width(true, 4, 4, None), None);
    assert_eq!(gemv_bf16::adaptive_width(true, 12, 12, None), None);
    assert_eq!(gemv_bf16::adaptive_width(true, 32, 16, None), None);
    assert_eq!(gemv_bf16::adaptive_width(false, 32, 32, None), None);
    assert_eq!(gemv_bf16::adaptive_width(true, 4, 64, Some(16)), Some(16));
    assert_eq!(gemv_bf16::adaptive_width(true, 4, 64, Some(64)), Some(32));
    assert_eq!(gemv_bf16::adaptive_width(true, 4, 64, Some(4)), None);
}

#[test]
fn stride_sequences_always_reassemble_the_cuda_warp_tree() {
    for x in [8u32, 16, 32] {
        let (fold, shuffle) = gemv_bf16::reduction_strides(x);
        let mut all = fold.clone();
        all.extend(&shuffle);
        assert_eq!(all, vec![16, 8, 4, 2, 1], "x={x}");
        for s in &fold {
            assert!(*s >= x);
        }
        for s in &shuffle {
            assert!(*s < x);
        }
    }
    assert_eq!(gemv_bf16::reduction_strides(32).0, Vec::<u32>::new());
    assert_eq!(gemv_bf16::reduction_strides(32).1, vec![16, 8, 4, 2, 1]);
    assert_eq!(gemv_bf16::reduction_strides(16).0, vec![16]);
    assert_eq!(gemv_bf16::reduction_strides(16).1, vec![8, 4, 2, 1]);
    assert_eq!(gemv_bf16::reduction_strides(8).0, vec![16, 8]);
    assert_eq!(gemv_bf16::reduction_strides(8).1, vec![4, 2, 1]);
}

#[test]
fn kernel_selection_by_caps_width() {
    use gemv_bf16::{select_kernel_from, GemvKernel};
    assert_eq!(
        select_kernel_from(true, 32, 32, None, 4096),
        GemvKernel::SgV4 {
            wg: gemv_bf16::SG_DEFAULT_WG
        }
    );
    assert_eq!(
        select_kernel_from(true, 32, 32, None, 4098),
        GemvKernel::SgScalar
    );
    assert_eq!(
        select_kernel_from(true, 64, 64, None, 4096),
        GemvKernel::SgV4Adaptive {
            wg: gemv_bf16::SG_DEFAULT_WG,
            x: 32
        }
    );
    assert_eq!(
        select_kernel_from(true, 16, 16, None, 4096),
        GemvKernel::SgV4Adaptive {
            wg: gemv_bf16::SG_DEFAULT_WG,
            x: 16
        }
    );
    assert_eq!(
        select_kernel_from(true, 8, 32, None, 4096),
        GemvKernel::SgV4Adaptive {
            wg: gemv_bf16::SG_DEFAULT_WG,
            x: 8
        }
    );
    assert_eq!(
        select_kernel_from(true, 64, 64, None, 4098),
        GemvKernel::TreeScalar
    );
    assert_eq!(
        select_kernel_from(false, 0, 0, None, 4096),
        GemvKernel::TreeVec8
    );
    assert_eq!(
        select_kernel_from(false, 0, 0, None, 4098),
        GemvKernel::TreeScalar
    );
    assert_eq!(
        select_kernel_from(true, 4, 4, None, 4096),
        GemvKernel::TreeVec8
    );
    assert_eq!(
        select_kernel_from(true, 4, 64, Some(32), 4096),
        GemvKernel::SgV4 {
            wg: gemv_bf16::SG_DEFAULT_WG
        }
    );
    assert_eq!(
        select_kernel_from(true, 4, 64, None, 4096),
        GemvKernel::TreeVec8
    );
    assert_eq!(
        select_kernel_from(true, 4, 64, Some(16), 4096),
        GemvKernel::SgV4Adaptive {
            wg: gemv_bf16::SG_DEFAULT_WG,
            x: 16
        }
    );
    assert_eq!(
        GemvKernel::SgV4Adaptive { wg: 128, x: 8 }.rows_per_group(),
        16
    );
    assert_eq!(
        GemvKernel::SgV4Adaptive { wg: 128, x: 32 }.rows_per_group(),
        4
    );
}

#[test]
fn adaptive_source_encodes_the_expected_reduction() {
    let s32 = gemv_bf16::adaptive_source(32, 128);
    assert!(s32.contains("subgroupShuffleXor(a0, 16u)"));
    assert!(s32.contains("subgroupShuffleXor(a0, 1u)"));
    assert!(!s32.contains("acc[j ^"));
    let s16 = gemv_bf16::adaptive_source(16, 128);
    assert!(s16.contains("nxt[j] = acc[j] + acc[j ^ 1u]"));
    assert!(s16.contains("subgroupShuffleXor(a0, 8u)"));
    assert!(!s16.contains("subgroupShuffleXor(a0, 16u)"));
    let s8 = gemv_bf16::adaptive_source(8, 128);
    assert!(s8.contains("acc[j ^ 2u]"));
    assert!(s8.contains("acc[j ^ 1u]"));
    assert!(s8.contains("subgroupShuffleXor(a0, 4u)"));
    assert!(!s8.contains("subgroupShuffleXor(a0, 8u)"));
    for s in [&s32, &s16, &s8] {
        assert!(s.contains(gemv_bf16::ADAPTIVE_ENTRY));
        assert!(s.contains("subgroup_size"));
        assert!(s.contains("fn bf16_encode("));
    }
}

#[test]
fn act_grid_source_declares_grid_entries_and_grouping() {
    let src = qz::act_grid_source();
    assert!(src.contains(qz::ACT_GRID_ENTRY));
    assert!(src.contains(qz::ACT_GRID_ENTRY_WG64));
    assert!(src.contains(gemv_nvfp4::QUANTIZE_ENTRY));
    assert!(src.contains("fn q_div_small("));
    assert_eq!(qz::act_grid_groups(1344, 256), 6);
    assert_eq!(qz::act_grid_groups(1344, 64), 21);
    assert_eq!(qz::act_grid_groups(1, 256), 1);
    assert_eq!(qz::act_grid_groups(0, 256), 1);
}

#[test]
fn adaptive_variants_compile_on_this_adapter() {
    let Some(ctx) = ctx_or_skip("adaptive_variants_compile") else {
        return;
    };
    if !ctx.caps.subgroup {
        eprintln!("adaptive_variants_compile: SKIP no subgroup support");
        return;
    }
    for x in [8u32, 16, 32] {
        for wg in [64u32, 128, 256] {
            let src = gemv_bf16::adaptive_source(x, wg);
            dispatch::compute_pipeline(ctx, "portability-compile", &src, gemv_bf16::ADAPTIVE_ENTRY)
                .unwrap_or_else(|e| panic!("x={x} wg={wg}: {e}"));
        }
    }
    dispatch::compute_pipeline(
        ctx,
        "act-grid-compile",
        &qz::act_grid_source(),
        qz::ACT_GRID_ENTRY,
    )
    .expect("act grid wg256");
    dispatch::compute_pipeline(
        ctx,
        "act-grid-compile-64",
        &qz::act_grid_source(),
        qz::ACT_GRID_ENTRY_WG64,
    )
    .expect("act grid wg64");
}

#[test]
fn adaptive_widths_are_bit_exact_against_the_warp32_kernel() {
    let Some(ctx) = ctx_or_skip("adaptive_bit_exact") else {
        return;
    };
    if !gemv_bf16::sg32_ok(ctx) {
        if !wgpu_allow_skip() {
            panic!(
                "adaptive_bit_exact: probed subgroup width {:?} is not 32, so the SgV4 reference \
                 arm cannot run and this gate would report success having compared nothing. Set \
                 NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose.",
                ctx.subgroup_width()
            );
        }
        eprintln!("adaptive_bit_exact: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) probed subgroup width is not 32");
        return;
    }
    let shapes = [
        (33usize, 4096usize),
        (256, 2048),
        (8, 64),
        (129, 21504),
        (5, 8),
    ];
    for (n, k) in shapes {
        let mut rng = Lcg(0x5eed ^ (n as u64) ^ ((k as u64) << 20));
        let w = rng.bf16_vec(n * k, 0.25);
        let x = rng.bf16_vec(k, 1.0);
        let (y_ref, _) = gemv_bf16::gemv_bf16_probe(
            ctx,
            &w,
            &x,
            n,
            k,
            1,
            1,
            gemv_bf16::GemvKernel::SgV4 { wg: 128 },
        )
        .expect("reference sg kernel");
        for kern in [
            gemv_bf16::GemvKernel::SgV4Adaptive { wg: 128, x: 32 },
            gemv_bf16::GemvKernel::SgV4Adaptive { wg: 128, x: 16 },
            gemv_bf16::GemvKernel::SgV4Adaptive { wg: 128, x: 8 },
            gemv_bf16::GemvKernel::SgV4Adaptive { wg: 256, x: 8 },
            gemv_bf16::GemvKernel::TreeVec8V4,
        ] {
            let (y, _) = gemv_bf16::gemv_bf16_probe(ctx, &w, &x, n, k, 1, 1, kern)
                .unwrap_or_else(|e| panic!("{kern:?} n={n} k={k}: {e}"));
            assert_eq!(y, y_ref, "{kern:?} diverges at n={n} k={k}");
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct QuantParams {
    global_scale: f32,
    k_blocks: u32,
    pad0: u32,
    pad1: u32,
}

fn pack_u16_words(src: &[u16]) -> Vec<u32> {
    src.chunks_exact(2)
        .map(|c| (c[0] as u32) | ((c[1] as u32) << 16))
        .collect()
}

fn run_act_quant(
    ctx: &WgpuContext,
    source: &str,
    entry: &str,
    x_words: &[u32],
    global_scale: f32,
    k_blocks: usize,
    groups: (u32, u32, u32),
) -> (Vec<u32>, Vec<u32>) {
    let xb = dispatch::storage_from_slice(ctx, "aq-x", x_words);
    let pb = dispatch::storage_zeroed(ctx, "aq-packed", (k_blocks * 8) as u64);
    let sb = dispatch::storage_zeroed(ctx, "aq-scales", (k_blocks * 4) as u64);
    let params = QuantParams {
        global_scale,
        k_blocks: k_blocks as u32,
        pad0: 0,
        pad1: 0,
    };
    let ub = dispatch::uniform_from(ctx, "aq-params", &params);
    dispatch::run(
        ctx,
        "aq-run",
        source,
        entry,
        &[(0, &xb), (1, &ub), (2, &pb), (3, &sb)],
        groups,
    )
    .expect("act quant dispatch");
    let packed = dispatch::read_back::<u32>(ctx, &pb, k_blocks * 2).expect("packed");
    let scales = dispatch::read_back::<u32>(ctx, &sb, k_blocks).expect("scales");
    (packed, scales)
}

#[test]
fn grid_act_quant_is_bit_exact_against_the_single_group_kernel() {
    let Some(ctx) = ctx_or_skip("grid_act_quant_exact") else {
        return;
    };
    let base = gemv_nvfp4::quantize_source();
    let grid = qz::act_grid_source();
    for k in [16usize, 208, 4096, 21504] {
        let k_blocks = k / 16;
        let mut rng = Lcg(0xacce55 ^ (k as u64));
        let mut inputs: Vec<(String, Vec<u16>)> = vec![
            ("random".into(), rng.bf16_vec(k, 4.0)),
            ("zeros".into(), vec![0u16; k]),
        ];
        let mut edge = rng.bf16_vec(k, 0.01);
        edge[0] = 0x7f80;
        edge[1 % k] = 0xff80;
        edge[2 % k] = 0x7fc0;
        edge[3 % k] = 0x0001;
        edge[4 % k] = 0x8001;
        edge[5 % k] = 0x7f7f;
        inputs.push(("edge".into(), edge));
        for (tag, x) in inputs {
            let x_words = pack_u16_words(&x);
            for gs in [1.0f32, 0.007326, 448.0, 0.0] {
                let (p_ref, s_ref) = run_act_quant(
                    ctx,
                    &base,
                    gemv_nvfp4::QUANTIZE_ENTRY,
                    &x_words,
                    gs,
                    k_blocks,
                    (1, 1, 1),
                );
                for (entry, wg) in [
                    (qz::ACT_GRID_ENTRY, qz::ACT_GRID_WG),
                    (qz::ACT_GRID_ENTRY_WG64, qz::ACT_GRID_WG64),
                ] {
                    let groups = (qz::act_grid_groups(k_blocks as u32, wg), 1, 1);
                    let (p, s) = run_act_quant(ctx, &grid, entry, &x_words, gs, k_blocks, groups);
                    assert_eq!(p, p_ref, "{entry} packed k={k} gs={gs} {tag}");
                    assert_eq!(s, s_ref, "{entry} scales k={k} gs={gs} {tag}");
                }
            }
        }
    }
}

struct UtilSampler {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: std::thread::JoinHandle<Vec<u32>>,
}

impl UtilSampler {
    fn start() -> Self {
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = stop.clone();
        let handle = std::thread::spawn(move || {
            let mut samples = Vec::new();
            while !flag.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(u) = gpu_util() {
                    samples.push(u);
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            samples
        });
        Self { stop, handle }
    }
    fn finish(self) -> Vec<u32> {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        self.handle.join().unwrap_or_default()
    }
}

#[test]
#[ignore]
fn perf_gemv_subgroup_widths() {
    let Some(ctx) = ctx_or_skip("perf_gemv_subgroup_widths") else {
        return;
    };
    if !gemv_bf16::sg32_ok(ctx) {
        if !wgpu_allow_skip() {
            panic!(
                "perf_gemv_subgroup_widths: probed subgroup width {:?} is not 32, so the sg32 and \
                 adaptive x=32 arms of this sweep cannot run and the numbers would compare fewer \
                 variants than they name. Set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose.",
                ctx.subgroup_width()
            );
        }
        eprintln!(
            "perf_gemv_subgroup_widths: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) probed subgroup width is not 32"
        );
        return;
    }
    let (n, k) = (24576usize, 8192usize);
    let mut rng = Lcg(0xbeef);
    let w = rng.bf16_vec(n * k, 0.05);
    let x = rng.bf16_vec(k, 0.5);
    let iters = 30usize;
    wait_idle("perf_gemv_subgroup_widths");
    eprintln!("util before window: {:?}", gpu_util());
    let variants = [
        (
            "tree_v4 (pre-portability non-32 path)",
            gemv_bf16::GemvKernel::TreeVec8V4,
        ),
        ("sg32 legacy wg128", gemv_bf16::GemvKernel::SgV4 { wg: 128 }),
        (
            "adaptive x=32 wg128",
            gemv_bf16::GemvKernel::SgV4Adaptive { wg: 128, x: 32 },
        ),
        (
            "adaptive x=16 wg128",
            gemv_bf16::GemvKernel::SgV4Adaptive { wg: 128, x: 16 },
        ),
        (
            "adaptive x=8 wg128",
            gemv_bf16::GemvKernel::SgV4Adaptive { wg: 128, x: 8 },
        ),
    ];
    let bytes = (n as f64) * (k as f64) * 2.0;
    for (name, kern) in variants {
        let (_, secs) = gemv_bf16::gemv_bf16_probe(ctx, &w, &x, n, k, 10, iters, kern)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let per = secs / iters as f64;
        eprintln!(
            "{name}: {:.3} ms/iter, {:.1} GB/s (wall clock over {iters} batched dispatches)",
            per * 1e3,
            bytes / per / 1e9
        );
    }
    eprintln!("util after window: {:?}", gpu_util());
}

#[test]
#[ignore]
fn perf_act_quant_grid() {
    let Some(ctx) = ctx_or_skip("perf_act_quant_grid") else {
        return;
    };
    let base = gemv_nvfp4::quantize_source();
    let grid = qz::act_grid_source();
    wait_idle("perf_act_quant_grid");
    for k in [21504usize, 5376] {
        let k_blocks = k / 16;
        let mut rng = Lcg(0xac7_9a17 ^ (k as u64));
        let x = rng.bf16_vec(k, 4.0);
        let x_words = pack_u16_words(&x);
        let iters = 400usize;
        let variants: Vec<(&str, &str, &str, (u32, u32, u32))> = vec![
            (
                "before: quantize_row_nvfp4_bf16 grid(1,1,1)",
                &base,
                gemv_nvfp4::QUANTIZE_ENTRY,
                (1, 1, 1),
            ),
            (
                "after: grid wg256",
                &grid,
                qz::ACT_GRID_ENTRY,
                (qz::act_grid_groups(k_blocks as u32, 256), 1, 1),
            ),
            (
                "after: grid wg64",
                &grid,
                qz::ACT_GRID_ENTRY_WG64,
                (qz::act_grid_groups(k_blocks as u32, 64), 1, 1),
            ),
        ];
        for (name, source, entry, groups) in variants {
            let xb = dispatch::storage_from_slice(ctx, "aqp-x", &x_words);
            let pb = dispatch::storage_zeroed(ctx, "aqp-packed", (k_blocks * 8) as u64);
            let sb = dispatch::storage_zeroed(ctx, "aqp-scales", (k_blocks * 4) as u64);
            let params = QuantParams {
                global_scale: 0.007326,
                k_blocks: k_blocks as u32,
                pad0: 0,
                pad1: 0,
            };
            let ub = dispatch::uniform_from(ctx, "aqp-params", &params);
            let pipeline =
                dispatch::cached_compute_pipeline(ctx, "aqp", source, entry).expect("pipeline");
            let group =
                dispatch::bind_group(ctx, &pipeline, &[(0, &xb), (1, &ub), (2, &pb), (3, &sb)]);
            let submit = |count: usize| {
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &group, &[]);
                    for _ in 0..count {
                        pass.dispatch_workgroups(groups.0, groups.1, groups.2);
                    }
                }
                ctx.queue.submit([enc.finish()]);
            };
            let before = gpu_util();
            submit(20);
            ctx.poll_blocking().expect("warmup");
            let sampler = UtilSampler::start();
            let t0 = std::time::Instant::now();
            submit(iters);
            ctx.poll_blocking().expect("timed");
            let secs = t0.elapsed().as_secs_f64();
            let samples = sampler.finish();
            let after = gpu_util();
            eprintln!(
            "k={k} {name}: {:.4} ms/iter over {iters} batched dispatches (wall clock), grid=({},{},{}), util before={before:?} during={samples:?} after={after:?}",
            secs / iters as f64 * 1e3,
            groups.0,
            groups.1,
            groups.2
        );
        }
    }
}
