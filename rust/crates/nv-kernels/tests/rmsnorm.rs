#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
mod common;
use common::cpu_rmsnorm;

#[test]
fn rmsnorm_f32_matches_cpu_reference() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "rmsnorm: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("rmsnorm: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    let batch = 4usize;
    let hidden = 1024usize;
    let eps = 1e-5f32;

    let mut x = Vec::with_capacity(batch * hidden);
    let mut weight = Vec::with_capacity(hidden);
    for i in 0..(batch * hidden) {
        x.push(((i as f32) * 0.0001).sin());
    }
    for i in 0..hidden {
        weight.push(1.0 + ((i as f32) * 0.001).cos());
    }

    #[allow(deprecated)]
    let dx: CudaSlice<f32> = stream.memcpy_stod(&x).unwrap();
    #[allow(deprecated)]
    let dw: CudaSlice<f32> = stream.memcpy_stod(&weight).unwrap();
    let mut dy: CudaSlice<f32> = stream.alloc_zeros::<f32>(batch * hidden).unwrap();

    let rc = {
        let (px, _g1) = dx.device_ptr(&stream);
        let (pw, _g2) = dw.device_ptr(&stream);
        let (py, _g3) = dy.device_ptr_mut(&stream);
        unsafe {
            cuda::rmsnorm_f32(
                stream.cu_stream() as *mut _,
                px as *const f32,
                pw as *const f32,
                py as *mut f32,
                batch,
                hidden,
                eps,
            )
        }
    };
    assert_eq!(rc, 0, "kernel launch returned {rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let got = stream.memcpy_dtov(&dy).unwrap();
    let expect = cpu_rmsnorm(&x, &weight, hidden, eps);
    let mut max_err = 0f32;
    for (g, e) in got.iter().zip(expect.iter()) {
        max_err = max_err.max((g - e).abs());
    }
    assert!(max_err < 1e-4, "max abs error {max_err} exceeds 1e-4");
}

#[test]
fn rmsnorm_bf16_matches_cpu_reference() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "rmsnorm: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("rmsnorm: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    let batch = 4usize;
    let hidden = 1024usize;
    let eps = 1e-5f32;

    let mut x = Vec::with_capacity(batch * hidden);
    let mut weight = Vec::with_capacity(hidden);
    for i in 0..(batch * hidden) {
        x.push(bf16::from_f32(((i as f32) * 0.0001).sin()));
    }
    for i in 0..hidden {
        weight.push(bf16::from_f32(1.0 + ((i as f32) * 0.001).cos()));
    }

    let x_f32: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let w_f32: Vec<f32> = weight.iter().map(|v| v.to_f32()).collect();
    let expect_f32 = cpu_rmsnorm(&x_f32, &w_f32, hidden, eps);

    let x_u16: Vec<u16> = x.iter().map(|v| v.to_bits()).collect();
    let w_u16: Vec<u16> = weight.iter().map(|v| v.to_bits()).collect();
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.memcpy_stod(&x_u16).unwrap();
    #[allow(deprecated)]
    let dw: CudaSlice<u16> = stream.memcpy_stod(&w_u16).unwrap();
    let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(batch * hidden).unwrap();

    let rc = {
        let (px, _g1) = dx.device_ptr(&stream);
        let (pw, _g2) = dw.device_ptr(&stream);
        let (py, _g3) = dy.device_ptr_mut(&stream);
        unsafe {
            cuda::rmsnorm_bf16(
                stream.cu_stream() as *mut _,
                px as *const u16,
                pw as *const u16,
                py as *mut u16,
                batch,
                hidden,
                eps,
            )
        }
    };
    assert_eq!(rc, 0, "kernel launch returned {rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let got_u16 = stream.memcpy_dtov(&dy).unwrap();
    let got: Vec<f32> = got_u16
        .iter()
        .map(|b| bf16::from_bits(*b).to_f32())
        .collect();

    let mut max_err = 0f32;
    for (g, e) in got.iter().zip(expect_f32.iter()) {
        max_err = max_err.max((g - e).abs());
    }
    assert!(max_err < 0.05, "bf16 rmsnorm drift {max_err}");
}

const NORM_M1_WIDE_LADDER_256_IS_THE_STOCK_WIDTH_AND_MUST_STAY_BITWISE_512_AND_1024_ARE_THE_FP_REORDER_RUNGS:
    [i32; 3] = [256, 512, 1024];

fn norm_m1_wide_shapes_are_the_measured_decode_rows_gemma4_31b_5376_qwen35_9b_4096_the_head_dim_norms_and_the_wide_row_counts_a_single_row_cannot_expose_a_reassociated_reduction_with(
) -> [(usize, usize); 8] {
    [
        (1, 5376),
        (1, 4096),
        (32, 256),
        (16, 512),
        (256, 4096),
        (256, 5376),
        (1, 5375),
        (33, 1023),
    ]
}

fn run_rmsnorm_bf16(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    x: &[u16],
    w: &[u16],
    batch: usize,
    hidden: usize,
) -> Vec<u16> {
    let dx = common::htod_u16(stream, x);
    let dw = common::htod_u16(stream, w);
    let mut dy: CudaSlice<u16> = stream.alloc_zeros(batch * hidden).unwrap();
    let rc = {
        let (px, _g1) = dx.device_ptr(stream);
        let (pw, _g2) = dw.device_ptr(stream);
        let (py, _g3) = dy.device_ptr_mut(stream);
        unsafe {
            cuda::rmsnorm_bf16(
                stream.cu_stream() as *mut _,
                px as *const u16,
                pw as *const u16,
                py as *mut u16,
                batch,
                hidden,
                1e-6,
            )
        }
    };
    assert_eq!(rc, 0, "rmsnorm_bf16 launch rc={rc}");
    stream.synchronize().unwrap();
    common::dtoh_u16(stream, &dy)
}

fn run_rmsnorm_residual_bf16(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    x: &[u16],
    res: &[u16],
    w: &[u16],
    batch: usize,
    hidden: usize,
) -> (Vec<u16>, Vec<u16>) {
    let dx = common::htod_u16(stream, x);
    let dw = common::htod_u16(stream, w);
    let mut dres = common::htod_u16(stream, res);
    let mut dout: CudaSlice<u16> = stream.alloc_zeros(batch * hidden).unwrap();
    let rc = {
        let (px, _g1) = dx.device_ptr(stream);
        let (pr, _g2) = dres.device_ptr_mut(stream);
        let (pw, _g3) = dw.device_ptr(stream);
        let (po, _g4) = dout.device_ptr_mut(stream);
        unsafe {
            cuda::rmsnorm_residual_bf16(
                stream.cu_stream() as *mut _,
                px as *const u16,
                pr as *mut u16,
                pw as *const u16,
                po as *mut u16,
                batch,
                hidden,
                1e-6,
            )
        }
    };
    assert_eq!(rc, 0, "rmsnorm_residual_bf16 launch rc={rc}");
    stream.synchronize().unwrap();
    (
        common::dtoh_u16(stream, &dout),
        common::dtoh_u16(stream, &dres),
    )
}

const NORM_M1_SPLIT_LADDER_1_IS_THE_STOCK_GRID_SPLIT_ACROSS_TWO_LAUNCHES_AND_MUST_STAY_BITWISE_2_TO_32_ARE_THE_FP_REORDER_RUNGS:
    [i32; 6] = [1, 2, 4, 8, 16, 32];

const NORM_M1_ARMS_SERVE_BATCH_ONE_ONLY_A_MULTI_ROW_NORM_ALREADY_HAS_ONE_CTA_PER_ROW: usize = 1;

fn norm_m1_wide_effective_block_the_cap_is_clamped_down_per_row_so_a_head_dim_norm_never_runs_wider_than_its_row(
    cap: i32,
    batch: usize,
    hidden: usize,
) -> i32 {
    if batch > NORM_M1_ARMS_SERVE_BATCH_ONE_ONLY_A_MULTI_ROW_NORM_ALREADY_HAS_ONE_CTA_PER_ROW {
        return 0;
    }
    let vpt = |b: i32| -> usize {
        match b {
            256 => 24,
            512 => 12,
            1024 => 8,
            _ => 0,
        }
    };
    if vpt(cap) == 0 {
        return 0;
    }
    let mut chosen = 0i32;
    for c in [256i32, 512, 1024] {
        if c <= cap && (c as usize) <= hidden {
            chosen = c;
        }
    }
    if chosen == 0 || hidden > chosen as usize * vpt(chosen) {
        return 0;
    }
    chosen
}

fn max_rel_bf16(a: &[u16], b: &[u16]) -> f32 {
    let mut m = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let xf = bf16::from_bits(*x).to_f32();
        let yf = bf16::from_bits(*y).to_f32();
        let den = yf.abs().max(1e-3);
        m = m.max((xf - yf).abs() / den);
    }
    m
}

#[test]
fn norm_m1_arms_are_both_default_off_because_every_published_gemma4_31b_qwen35_9b_and_qwen38_27b_decode_number_was_measured_on_the_stock_single_cta_norm_kernels(
) {
    assert_eq!(
        unsafe { cuda::norm_m1_wide_block() },
        0,
        "NV_NORM_M1_WIDE must be OFF by default: every published gemma4-31B, qwen3.5-9B and \
         qwen3.8-27B decode number was measured on the stock <<<batch,256>>> norm kernels"
    );
    assert_eq!(
        unsafe { cuda::norm_m1_split_ctas() },
        0,
        "NV_NORM_M1_SPLIT must be OFF by default: the split arm turns one norm launch into two \
         and every published decode number was measured on the single-launch stock kernels"
    );
}

#[test]
fn norm_m1_arms_are_inert_at_every_batch_above_one_asked_of_the_launcher_itself_not_of_a_mirror_of_its_own_ladder(
) {
    if CudaContext::new(0).is_err() {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "norm_m1 batch guard: no CUDA device 0. This gate refuses to report success \
                 without running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("norm_m1 batch guard: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    }
    for (batch, hidden) in norm_m1_wide_shapes_are_the_measured_decode_rows_gemma4_31b_5376_qwen35_9b_4096_the_head_dim_norms_and_the_wide_row_counts_a_single_row_cannot_expose_a_reassociated_reduction_with()
    {
        for block in NORM_M1_WIDE_LADDER_256_IS_THE_STOCK_WIDTH_AND_MUST_STAY_BITWISE_512_AND_1024_ARE_THE_FP_REORDER_RUNGS
        {
            unsafe { cuda::norm_m1_split_force(0) };
            unsafe { cuda::norm_m1_wide_force(block) };
            let picked = unsafe { cuda::norm_m1_wide_block_for_launch(batch, hidden) };
            if batch > NORM_M1_ARMS_SERVE_BATCH_ONE_ONLY_A_MULTI_ROW_NORM_ALREADY_HAS_ONE_CTA_PER_ROW
            {
                assert_eq!(
                    picked, 0,
                    "NV_NORM_M1_WIDE={block} served a {batch}-row norm at hidden {hidden}. \
                     A multi-row norm already dispatches one CTA per row, so widening buys it \
                     nothing and the register staging costs it: measured +12% on gemma4-31B's \
                     head-dim norms (grids 4/16/32) and 2.6x on qwen3.5-9B's 4096-row prefill \
                     norm. The arm serves batch 1 only."
                );
            } else {
                assert_ne!(
                    picked, 0,
                    "NV_NORM_M1_WIDE={block} refused the M=1 row hidden {hidden} the arm exists \
                     to serve"
                );
            }
        }
        for ctas in NORM_M1_SPLIT_LADDER_1_IS_THE_STOCK_GRID_SPLIT_ACROSS_TWO_LAUNCHES_AND_MUST_STAY_BITWISE_2_TO_32_ARE_THE_FP_REORDER_RUNGS
        {
            unsafe { cuda::norm_m1_wide_force(0) };
            unsafe { cuda::norm_m1_split_force(ctas) };
            let picked = unsafe { cuda::norm_m1_split_ctas_for_launch(batch, hidden) };
            if batch > NORM_M1_ARMS_SERVE_BATCH_ONE_ONLY_A_MULTI_ROW_NORM_ALREADY_HAS_ONE_CTA_PER_ROW
            {
                assert_eq!(
                    picked, 0,
                    "NV_NORM_M1_SPLIT={ctas} served a {batch}-row norm at hidden {hidden}. \
                     The split arm exists to give the hidden-dim reduction of a SINGLE row to \
                     several CTAs; a multi-row norm already has that parallelism across rows, \
                     and its partial-sum slots are indexed by blockIdx.x alone."
                );
            } else {
                assert!(
                    picked >= 1
                        && picked <= ctas
                        && (picked & (picked - 1)) == 0
                        && (picked as usize) * 256 <= hidden + 255,
                    "NV_NORM_M1_SPLIT={ctas} picked {picked} CTAs for the M=1 row hidden {hidden}"
                );
            }
        }
    }
    unsafe { cuda::norm_m1_wide_force(0) };
    unsafe { cuda::norm_m1_split_force(0) };
}

#[test]
fn norm_m1_split_ladder_1_cta_is_bitwise_against_the_stock_single_cta_kernel_and_the_wider_grids_stay_inside_the_bf16_rounding_band(
) {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "norm_m1_split: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("norm_m1_split: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    for (batch, hidden) in norm_m1_wide_shapes_are_the_measured_decode_rows_gemma4_31b_5376_qwen35_9b_4096_the_head_dim_norms_and_the_wide_row_counts_a_single_row_cannot_expose_a_reassociated_reduction_with()
    {
        let mut seed = 0x5eed_a11e_0bad_c0deu64 ^ ((batch * hidden) as u64);
        let x = common::rand_bf16(&mut seed, batch * hidden, -2.0, 2.0);
        let res = common::rand_bf16(&mut seed, batch * hidden, -2.0, 2.0);
        let w = common::rand_bf16(&mut seed, hidden, 0.5, 1.5);

        assert_eq!(unsafe { cuda::norm_m1_wide_force(0) }, 0);
        assert_eq!(unsafe { cuda::norm_m1_split_force(0) }, 0);
        let base_norm = run_rmsnorm_bf16(&stream, &x, &w, batch, hidden);
        let (base_res_out, base_res_inplace) =
            run_rmsnorm_residual_bf16(&stream, &x, &res, &w, batch, hidden);

        for ctas in NORM_M1_SPLIT_LADDER_1_IS_THE_STOCK_GRID_SPLIT_ACROSS_TWO_LAUNCHES_AND_MUST_STAY_BITWISE_2_TO_32_ARE_THE_FP_REORDER_RUNGS
        {
            assert_eq!(unsafe { cuda::norm_m1_split_force(ctas) }, ctas);
            let picked = unsafe { cuda::norm_m1_split_ctas_for_launch(batch, hidden) };
            let got_norm = run_rmsnorm_bf16(&stream, &x, &w, batch, hidden);
            let (got_res_out, got_res_inplace) =
                run_rmsnorm_residual_bf16(&stream, &x, &res, &w, batch, hidden);
            let tag = format!("b{batch}xh{hidden} split{ctas} picked={picked}");
            if picked <= 1 {
                common::assert_u16_bits(&format!("rmsnorm {tag}"), &got_norm, &base_norm);
                common::assert_u16_bits(
                    &format!("rmsnorm_residual out {tag}"),
                    &got_res_out,
                    &base_res_out,
                );
                common::assert_u16_bits(
                    &format!("rmsnorm_residual inplace {tag}"),
                    &got_res_inplace,
                    &base_res_inplace,
                );
            } else {
                common::assert_u16_bits(
                    &format!("rmsnorm_residual inplace {tag}"),
                    &got_res_inplace,
                    &base_res_inplace,
                );
                let r1 = max_rel_bf16(&got_norm, &base_norm);
                let r2 = max_rel_bf16(&got_res_out, &base_res_out);
                assert!(
                    r1 < 0.02 && r2 < 0.02,
                    "fp-reorder rung {tag} drifted beyond the bf16 rounding band: \
                     rmsnorm {r1}, rmsnorm_residual {r2}"
                );
            }
        }
        assert_eq!(unsafe { cuda::norm_m1_split_force(0) }, 0);
    }
}

#[test]
fn norm_m1_wide_ladder_256_rung_is_bitwise_against_the_stock_single_cta_kernel() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "norm_m1_wide: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("norm_m1_wide: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    assert_eq!(unsafe { cuda::norm_m1_split_force(0) }, 0);

    for (batch, hidden) in norm_m1_wide_shapes_are_the_measured_decode_rows_gemma4_31b_5376_qwen35_9b_4096_the_head_dim_norms_and_the_wide_row_counts_a_single_row_cannot_expose_a_reassociated_reduction_with()
    {
        let mut seed = 0xfeed_1234_5678_9abcu64 ^ ((batch * hidden) as u64);
        let x = common::rand_bf16(&mut seed, batch * hidden, -2.0, 2.0);
        let res = common::rand_bf16(&mut seed, batch * hidden, -2.0, 2.0);
        let w = common::rand_bf16(&mut seed, hidden, 0.5, 1.5);

        assert_eq!(unsafe { cuda::norm_m1_wide_force(0) }, 0);
        let base_norm = run_rmsnorm_bf16(&stream, &x, &w, batch, hidden);
        let (base_res_out, base_res_inplace) =
            run_rmsnorm_residual_bf16(&stream, &x, &res, &w, batch, hidden);

        for block in NORM_M1_WIDE_LADDER_256_IS_THE_STOCK_WIDTH_AND_MUST_STAY_BITWISE_512_AND_1024_ARE_THE_FP_REORDER_RUNGS
        {
            let armed = unsafe { cuda::norm_m1_wide_force(block) };
            assert_eq!(armed, block, "force({block}) did not stick");
            let eff = norm_m1_wide_effective_block_the_cap_is_clamped_down_per_row_so_a_head_dim_norm_never_runs_wider_than_its_row(
                block, batch, hidden,
            );
            assert_eq!(
                eff,
                unsafe { cuda::norm_m1_wide_block_for_launch(batch, hidden) },
                "the test's mirror of the wide ladder disagrees with the launcher for \
                 b{batch}xh{hidden} cap {block}"
            );
            let serves = eff > 256;
            let got_norm = run_rmsnorm_bf16(&stream, &x, &w, batch, hidden);
            let (got_res_out, got_res_inplace) =
                run_rmsnorm_residual_bf16(&stream, &x, &res, &w, batch, hidden);
            let tag = format!("b{batch}xh{hidden} block{block} serves={serves}");
            if block == 256 || !serves {
                common::assert_u16_bits(&format!("rmsnorm {tag}"), &got_norm, &base_norm);
                common::assert_u16_bits(
                    &format!("rmsnorm_residual out {tag}"),
                    &got_res_out,
                    &base_res_out,
                );
                common::assert_u16_bits(
                    &format!("rmsnorm_residual inplace {tag}"),
                    &got_res_inplace,
                    &base_res_inplace,
                );
            } else {
                let r1 = max_rel_bf16(&got_norm, &base_norm);
                let r2 = max_rel_bf16(&got_res_out, &base_res_out);
                common::assert_u16_bits(
                    &format!("rmsnorm_residual inplace {tag}"),
                    &got_res_inplace,
                    &base_res_inplace,
                );
                assert!(
                    r1 < 0.02 && r2 < 0.02,
                    "fp-reorder rung {tag} drifted beyond the bf16 rounding band: \
                     rmsnorm {r1}, rmsnorm_residual {r2}"
                );
            }
        }
        assert_eq!(unsafe { cuda::norm_m1_wide_force(0) }, 0);
    }
}

const NVFP4_GEMV_PRENORM_FUSE_SHAPES_ARE_THE_MEASURED_GEMMA4_31B_DECODE_GEMVS_QKV_16384X5376_O_AND_DOWN_5376X5376_GATE_UP_43008X5376_TRIMMED_TO_ROW_COUNTS_A_TEST_CAN_ALLOCATE:
    [(usize, usize); 4] = [(1024, 5376), (2048, 5376), (1024, 4096), (137, 5376)];

const NVFP4_GEMV_PRENORM_ROWS_PER_WARP_LADDER_IS_BITWISE_AT_EVERY_RUNG_BECAUSE_A_ROW_KEEPS_ITS_OWN_ACCUMULATOR_AND_ITS_OWN_KB_ORDER:
    [i32; 4] = [1, 2, 4, 8];

fn nvfp4_scale_bytes_for_the_swizzled_layout_the_gemv_indexes_with(n: usize, k: usize) -> usize {
    let k_blocks = k / 16;
    let k_tiles = k_blocks.div_ceil(4);
    let m_tiles = n.div_ceil(128);
    m_tiles * k_tiles * 32 * 16
}

fn run_nvfp4_gemv_prenorm(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    w_packed: &[u8],
    w_scales: &[u8],
    x_raw: &[u16],
    norm_w: &[u16],
    n: usize,
    k: usize,
    eps: f32,
) -> Vec<u16> {
    #[allow(deprecated)]
    let dw: CudaSlice<u8> = stream.memcpy_stod(w_packed).unwrap();
    #[allow(deprecated)]
    let ds: CudaSlice<u8> = stream.memcpy_stod(w_scales).unwrap();
    let dx = common::htod_u16(stream, x_raw);
    let dnw = common::htod_u16(stream, norm_w);
    let mut dscratch: CudaSlice<u16> = stream.alloc_zeros::<u16>(k).unwrap();
    let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
    let rc = {
        let (pw, _g1) = dw.device_ptr(stream);
        let (ps, _g2) = ds.device_ptr(stream);
        let (px, _g3) = dx.device_ptr(stream);
        let (pnw, _g4) = dnw.device_ptr(stream);
        let (pscr, _g5) = dscratch.device_ptr_mut(stream);
        let (py, _g6) = dy.device_ptr_mut(stream);
        unsafe {
            cuda::nvfp4_gemv_bf16act_prenorm(
                stream.cu_stream() as *mut _,
                pw as *const u8,
                ps as *const u8,
                px as *const u16,
                pnw as *const u16,
                pscr as *mut u16,
                py as *mut u16,
                1.0f32,
                n as i32,
                k as i32,
                eps,
            )
        }
    };
    assert_eq!(rc, 0, "nvfp4_gemv_bf16act_prenorm rc={rc} n={n} k={k}");
    stream.synchronize().unwrap();
    common::dtoh_u16(stream, &dy)
}

#[test]
fn nvfp4_gemv_prenorm_fuse_is_default_off_because_every_published_gemma4_31b_decode_number_was_measured_on_the_input_layernorm_as_its_own_graph_node(
) {
    assert_eq!(
        unsafe { cuda::nvfp4_gemv_prenorm_fused() },
        0,
        "NV_NORM_GEMV_FUSED must be OFF by default: folding the input layernorm into the qkv \
         gemv removes a graph node from the captured gemma4-31B decode, and every published \
         number for that family was measured with the norm as its own node"
    );
}

#[test]
fn nvfp4_gemv_prenorm_fused_entry_is_bitwise_against_the_stock_rmsnorm_then_gemv_pair_because_it_runs_the_same_reduction_in_the_same_order(
) {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "nvfp4_gemv prenorm fuse: no CUDA device 0. This gate refuses to report success \
                 without running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("nvfp4_gemv prenorm fuse: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let eps = 1e-6f32;

    for (n, k) in NVFP4_GEMV_PRENORM_FUSE_SHAPES_ARE_THE_MEASURED_GEMMA4_31B_DECODE_GEMVS_QKV_16384X5376_O_AND_DOWN_5376X5376_GATE_UP_43008X5376_TRIMMED_TO_ROW_COUNTS_A_TEST_CAN_ALLOCATE
    {
        let mut seed = 0x51ed_2718_2818_2845u64 ^ ((n * k) as u64);
        let x_raw = common::rand_bf16(&mut seed, k, -3.0, 3.0);
        let norm_w = common::rand_bf16(&mut seed, k, 0.25, 1.75);
        let w_packed: Vec<u8> = (0..(n * k / 2))
            .map(|_| (common::lcg_unit_f32(&mut seed) * 255.0) as u8)
            .collect();
        let w_scales: Vec<u8> = (0..nvfp4_scale_bytes_for_the_swizzled_layout_the_gemv_indexes_with(n, k))
            .map(|_| 0x38u8 | ((common::lcg_unit_f32(&mut seed) * 7.0) as u8))
            .collect();

        assert_eq!(unsafe { cuda::nvfp4_gemv_prenorm_fused_force(0) }, 0);
        assert_eq!(
            unsafe { cuda::nvfp4_gemv_prenorm_fuses_this_row(k as i32) },
            0,
            "the unfused arm must not fuse row k={k}"
        );
        let base = run_nvfp4_gemv_prenorm(&stream, &w_packed, &w_scales, &x_raw, &norm_w, n, k, eps);

        for rpw in NVFP4_GEMV_PRENORM_ROWS_PER_WARP_LADDER_IS_BITWISE_AT_EVERY_RUNG_BECAUSE_A_ROW_KEEPS_ITS_OWN_ACCUMULATOR_AND_ITS_OWN_KB_ORDER
        {
            assert_eq!(unsafe { cuda::nvfp4_gemv_prenorm_fused_force(rpw) }, rpw);
            assert_eq!(
                unsafe { cuda::nvfp4_gemv_prenorm_fuses_this_row(k as i32) },
                1,
                "the armed fuse must serve the gemma4-31B decode row width k={k}"
            );
            let fused =
                run_nvfp4_gemv_prenorm(&stream, &w_packed, &w_scales, &x_raw, &norm_w, n, k, eps);
            common::assert_u16_bits(
                &format!("nvfp4_gemv prenorm fuse n{n}xk{k} rows_per_warp{rpw}"),
                &fused,
                &base,
            );
        }
        assert_eq!(unsafe { cuda::nvfp4_gemv_prenorm_fused_force(0) }, 0);
    }
}

#[test]
fn nvfp4_gemv_prenorm_fuse_declines_any_row_whose_normed_copy_does_not_fit_the_48_kib_dynamic_shared_limit(
) {
    assert_eq!(unsafe { cuda::nvfp4_gemv_prenorm_fused_force(1) }, 1);
    assert_eq!(
        unsafe { cuda::nvfp4_gemv_prenorm_fuses_this_row(24576) },
        1,
        "24576 bf16 is exactly 48 KiB and must still fuse"
    );
    assert_eq!(
        unsafe { cuda::nvfp4_gemv_prenorm_fuses_this_row(24592) },
        0,
        "a row past 48 KiB must fall back to the two-launch pair, not launch and fail"
    );
    assert_eq!(
        unsafe { cuda::nvfp4_gemv_prenorm_fuses_this_row(5370) },
        0,
        "a row that is not a multiple of 16 is not an NVFP4 gemv row at all"
    );
    assert_eq!(unsafe { cuda::nvfp4_gemv_prenorm_fused_force(0) }, 0);
}
