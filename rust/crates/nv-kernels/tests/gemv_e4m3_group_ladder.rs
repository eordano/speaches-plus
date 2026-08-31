#![cfg(feature = "cuda")]

mod common;
use common::LcgMask23TwoSided as Lcg;
use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;

pub const THE_FOUR_PROJECTIONS_THE_CONCURRENCY_GAP_LIVES_IN_ARE_FP8_E4M3_PER_OUTPUT_ROW_SCALED_AND_THEIR_SHAPES_COME_FROM_QWEN38_27B_CONFIG_JSON: &str =
    "hidden 5120, linear_num_key_heads 16 x 128, linear_num_value_heads 48 x 128, \
     num_attention_heads 24 x head_dim 256 with attn_output_gate, num_key_value_heads 4; \
     48 gdn layers and 16 full-attention layers over 64";

pub const QWEN38_FP8_PROJECTION_SHAPES_N_K_LAYERS: [(&str, usize, usize, usize); 7] = [
    ("gdn.in_proj_qkv", 10240, 5120, 48),
    ("gdn.in_proj_z", 6144, 5120, 48),
    ("gdn.out_proj", 5120, 6144, 48),
    ("attn.q_proj", 12288, 5120, 16),
    ("attn.k_proj", 1024, 5120, 16),
    ("attn.v_proj", 1024, 5120, 16),
    ("attn.o_proj", 5120, 6144, 16),
];

pub const A_WEIGHT_THAT_FITS_IN_L2_MAKES_EVERY_ROUTE_LOOK_BANDWIDTH_RICH_AND_HIDES_EXACTLY_THE_TERM_SERVING_PAYS_SO_THE_LADDER_ROTATES_OVER_A_WORKING_SET_LARGER_THAN_L2: usize =
    768 * 1024 * 1024;

fn bench_us(stream: &Arc<CudaStream>, iters: usize, mut launch: impl FnMut(usize) -> i32) -> (f64, i32) {
    let mut rc = 0;
    for i in 0..3 {
        rc = launch(i);
    }
    if rc != 0 {
        return (f64::NAN, rc);
    }
    stream.synchronize().unwrap();
    let t0 = Instant::now();
    for i in 0..iters {
        rc = launch(i);
    }
    stream.synchronize().unwrap();
    (t0.elapsed().as_secs_f64() * 1e6 / iters as f64, rc)
}

#[test]
#[ignore = "measurement ladder, GPU-time only; set NV_E4M3_GROUP_LADDER=1 -- prices the m-row fp8 route on the five qwen3.8 fp8 projection shapes at m=1..16 and prints the device dynamic-smem optin cap that decides which m the staged route can reach at all"]
fn the_m_row_fp8_route_is_priced_at_every_m_on_the_qwen38_projection_shapes_with_its_refusals_visible()
{
    if std::env::var("NV_E4M3_GROUP_LADDER").as_deref() != Ok("1") {
        panic!(
            "NV_E4M3_GROUP_LADDER is not 1, so this ladder executed nothing. It is already \
             #[ignore]d, which makes --ignored the opt-in; a second env gate behind that can only \
             make an --ignored run print `1 passed` for a ladder that measured nothing. A suite \
             whose precondition is absent must fail or be ignored, never return green."
        );
    }
    let ctx = CudaContext::new(0).expect("cuda device 0");
    let stream = ctx.default_stream();
    let optin = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
        .unwrap();
    eprintln!("E4M3-LADDER device_max_dynamic_smem_optin_bytes={optin}");
    let iters: usize = std::env::var("NV_E4M3_GROUP_LADDER_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let rows_per_warp: i32 = std::env::var("NV_E4M3_GROUP_R")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    eprintln!(
        "E4M3-LADDER arm rows_per_warp={rows_per_warp} (0 = the shipped gemv_e4m3_mk route, \
         -1 = the rowquant + cublasLt e4m3 + scale_rowcol fallback the shipped route takes \
         whenever the staged route refuses)"
    );
    let mut runner = if rows_per_warp < 0 {
        Some(nv_quant::fp8::Fp8GemmRunner::new(stream.clone()).expect("cublasLt fp8 runner"))
    } else {
        None
    };

    let mut rng = Lcg(0x9e3779b97f4a7c15);
    for (name, n, k, layers) in QWEN38_FP8_PROJECTION_SHAPES_N_K_LAYERS {
        let wq: Vec<u8> = (0..n * k).map(|_| (rng.next_u32() & 0xff) as u8).collect();
        let row_scale: Vec<f32> = (0..n).map(|_| 0.01 + rng.next_f32().abs() * 0.01).collect();
        let rot = (A_WEIGHT_THAT_FITS_IN_L2_MAKES_EVERY_ROUTE_LOOK_BANDWIDTH_RICH_AND_HIDES_EXACTLY_THE_TERM_SERVING_PAYS_SO_THE_LADDER_ROTATES_OVER_A_WORKING_SET_LARGER_THAN_L2
            / (n * k))
            .max(1);
        #[allow(deprecated)]
        let dws: Vec<CudaSlice<u8>> = (0..rot).map(|_| stream.clone_htod(&wq).unwrap()).collect();
        #[allow(deprecated)]
        let ds: CudaSlice<f32> = stream.clone_htod(&row_scale).unwrap();
        eprintln!("E4M3-LADDER shape={name} n={n} k={k} weight_copies_rotated={rot}");
        let mut base_us = f64::NAN;
        for m in 1usize..=16 {
            let x = rng.bf16_words(m * k, 1.0);
            #[allow(deprecated)]
            let dx: CudaSlice<u16> = stream.clone_htod(&x).unwrap();
            let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();
            let mut xq: CudaSlice<u8> = stream.alloc_zeros::<u8>(m * k).unwrap();
            let mut xrs: CudaSlice<f32> = stream.alloc_zeros::<f32>(m).unwrap();
            let mut dbf: CudaSlice<half::bf16> = stream.alloc_zeros::<half::bf16>(m * n).unwrap();
            let (us, rc) = bench_us(&stream, iters, |it| {
                let dw = &dws[it % rot];
                if let Some(r) = runner.as_mut() {
                    let rc_q = {
                        let (px, _c) = dx.device_ptr(&stream);
                        let (pq, _d) = xq.device_ptr_mut(&stream);
                        let (pr, _e) = xrs.device_ptr_mut(&stream);
                        unsafe {
                            cuda::rowquant_e4m3(
                                stream.cu_stream() as *mut c_void,
                                px as *const u16,
                                pq as *mut u8,
                                pr as *mut f32,
                                m as i32,
                                k as i32,
                            )
                        }
                    };
                    if rc_q != 0 {
                        return rc_q;
                    }
                    if rows_per_warp == -2 {
                        r.matmul_e4m3_row_scaled(
                            &xq, dw, &mut dbf, m as u64, n as u64, k as u64, &xrs, &ds,
                        )
                        .expect("cublasLt e4m3 row-scaled matmul");
                        return 0;
                    }
                    r.matmul_e4m3_weight_row(
                        &xq, dw, &mut dbf, m as u64, n as u64, k as u64, &xrs, &xrs,
                    )
                    .expect("cublasLt e4m3 matmul");
                    let (pd, _f) = dbf.device_ptr_mut(&stream);
                    let (pr, _g) = xrs.device_ptr(&stream);
                    let (pc, _h) = ds.device_ptr(&stream);
                    return unsafe {
                        cuda::scale_rowcol_bf16(
                            stream.cu_stream() as *mut c_void,
                            pd as *mut u16,
                            pr as *const f32,
                            pc as *const f32,
                            m as i32,
                            n as i32,
                        )
                    };
                }
                let (pw, _a) = dw.device_ptr(&stream);
                let (ps, _b) = ds.device_ptr(&stream);
                let (px, _c) = dx.device_ptr(&stream);
                let (py, _d) = dy.device_ptr_mut(&stream);
                unsafe {
                    if rows_per_warp == 0 {
                        cuda::gemv_e4m3_mk(
                            stream.cu_stream() as *mut c_void,
                            pw as *const u8,
                            ps as *const f32,
                            px as *const u16,
                            py as *mut u16,
                            n as i32,
                            k as i32,
                            m as i32,
                        )
                    } else {
                        cuda::gemv_e4m3_group(
                            stream.cu_stream() as *mut c_void,
                            pw as *const u8,
                            ps as *const f32,
                            px as *const u16,
                            py as *mut u16,
                            n as i32,
                            k as i32,
                            m as i32,
                            rows_per_warp,
                        )
                    }
                }
            });
            if m == 1 {
                base_us = us;
            }
            if rc != 0 {
                eprintln!(
                    "E4M3-LADDER shape={name} n={n} k={k} m={m} REFUSED rc={rc} \
                     staged_smem_bytes={}",
                    m * (k >> 4) * 9 * 4
                );
                continue;
            }
            let gbs = (n * k) as f64 / us / 1e3;
            let ms_per_step = us * layers as f64 / 1e3;
            let tile_v = unsafe { cuda::gemv_e4m3_group_tile_v(m as i32, k as i32) };
            eprintln!(
                "E4M3-LADDER shape={name} n={n} k={k} m={m} us={us:.2} weight_gbs={gbs:.0} \
                 growth_vs_m1={:.3} layers={layers} ms_per_step={ms_per_step:.3} tile_v={tile_v}",
                us / base_us
            );
        }
    }
}
