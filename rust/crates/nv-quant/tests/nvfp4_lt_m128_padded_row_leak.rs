#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use nv_quant::nvfp4::Nvfp4GemmRunner;
use std::sync::Arc;

const N_5120_K_17408_THE_QWEN38_DOWN_PROJ_THE_ONLY_TRUNK_SHAPE_WHOSE_LT_HEURISTIC_PICKS_SPLITK: (
    usize,
    usize,
) = (5120, 17408);
const M_PADDED_128_THE_STAGED_VERIFY_LAUNCH_EVERY_LOGICAL_M_2_TO_8_SHARES: usize = 128;
const LOGICAL_ROWS_LADDER_4_6_7_8_THE_CENSUS_ARMS_THAT_FLIP_AT_7: [usize; 4] = [4, 6, 7, 8];
const SEEDS_SWEPT_8_BECAUSE_THE_MODEL_FLIP_WAS_DATA_DEPENDENT_SURFACING_AT_LAYER_56: usize = 8;

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn rnd_bf16_unit(state: &mut u64) -> u16 {
    half::bf16::from_f32(((lcg(state) >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0).to_bits()
}

fn swizzled_scale_bytes(rows: usize, k: usize) -> usize {
    let k_blocks = k / 16;
    rows.div_ceil(128) * k_blocks.div_ceil(4) * 512
}

fn quantize_rows(
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    packed: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<u8>,
    m_rows: usize,
    k: usize,
) {
    let (xp, _gx) = x.device_ptr(stream);
    let (pp, _gp) = packed.device_ptr_mut(stream);
    let (sp, _gs) = scales.device_ptr_mut(stream);
    let rc = unsafe {
        nv_kernels::cuda::quantize_nvfp4_bf16_rows(
            stream.cu_stream() as *mut _,
            xp as *const u16,
            pp as *mut u8,
            sp as *mut u8,
            1.0f32,
            m_rows as i32,
            k as i32,
        )
    };
    assert_eq!(rc, 0, "quantize_nvfp4_bf16_rows rc={rc}");
}

#[test]
fn nvfp4_lt_m128_staged_gemm_rows_0_to_5_are_bitwise_invariant_to_how_many_logical_rows_were_quantized(
) {
    let stream: Arc<CudaStream> = match CudaContext::new(0) {
        Ok(ctx) => ctx.default_stream(),
        Err(e) => panic!(
            "no CUDA device 0: this is the staged-m128 Lt padded-row leak probe and must not \
             report success having executed nothing: {e}"
        ),
    };
    std::env::set_var("NV_NVFP4_LT_ALGO_LOG", "1");
    let (n, k) = N_5120_K_17408_THE_QWEN38_DOWN_PROJ_THE_ONLY_TRUNK_SHAPE_WHOSE_LT_HEURISTIC_PICKS_SPLITK;
    let m_padded = M_PADDED_128_THE_STAGED_VERIFY_LAUNCH_EVERY_LOGICAL_M_2_TO_8_SHARES;
    let mut runner = Nvfp4GemmRunner::new(stream.clone()).expect("nvfp4 runner");
    let alpha_dev: CudaSlice<f32> = stream.clone_htod(&[1.0f32]).unwrap();

    let mut any_leak = String::new();
    for seed in 0..SEEDS_SWEPT_8_BECAUSE_THE_MODEL_FLIP_WAS_DATA_DEPENDENT_SURFACING_AT_LAYER_56 {
        let mut st = 0x9e37_79b9_7f4a_7c15u64 ^ ((seed as u64) << 32);
        let wq: Vec<u8> = (0..n * k / 2).map(|_| (lcg(&mut st) >> 40) as u8).collect();
        let wsc: Vec<u8> = (0..swizzled_scale_bytes(n, k))
            .map(|_| 0x30 + ((lcg(&mut st) >> 40) as u8 & 0x0f))
            .collect();
        let d_wq: CudaSlice<u8> = stream.clone_htod(&wq).unwrap();
        let d_wsc: CudaSlice<u8> = stream.clone_htod(&wsc).unwrap();

        let max_m = *LOGICAL_ROWS_LADDER_4_6_7_8_THE_CENSUS_ARMS_THAT_FLIP_AT_7
            .iter()
            .max()
            .unwrap();
        let hx: Vec<u16> = (0..max_m * k).map(|_| rnd_bf16_unit(&mut st)).collect();
        let d_x: CudaSlice<u16> = stream.clone_htod(&hx).unwrap();

        let mut a_packed: CudaSlice<u8> = stream.alloc_zeros::<u8>(m_padded * k / 2).unwrap();
        let mut a_scales: CudaSlice<u8> =
            stream.alloc_zeros::<u8>(swizzled_scale_bytes(m_padded, k)).unwrap();

        let mut rows_by_m: Vec<(usize, Vec<Vec<u16>>)> = Vec::new();
        for &m in &LOGICAL_ROWS_LADDER_4_6_7_8_THE_CENSUS_ARMS_THAT_FLIP_AT_7 {
            quantize_rows(&stream, &d_x, &mut a_packed, &mut a_scales, m, k);
            let mut d: CudaSlice<half::bf16> =
                stream.alloc_zeros::<half::bf16>(m_padded * n).unwrap();
            runner
                .matmul_scaled_alpha_dev(
                    &a_packed,
                    &a_scales,
                    &d_wq,
                    &d_wsc,
                    &mut d,
                    m_padded as u64,
                    n as u64,
                    k as u64,
                    &alpha_dev,
                    1.0,
                )
                .unwrap_or_else(|e| panic!("nvfp4 gemm m_logical={m}: {e:#}"));
            stream.synchronize().unwrap();
            let out: Vec<half::bf16> = stream.clone_dtoh(&d).unwrap();
            let rows: Vec<Vec<u16>> = (0..6)
                .map(|r| out[r * n..(r + 1) * n].iter().map(|b| b.to_bits()).collect())
                .collect();
            rows_by_m.push((m, rows));
        }
        let (m_base, base_rows) = &rows_by_m[0];
        for (m, rows) in &rows_by_m[1..] {
            for (r, (row, base)) in rows.iter().zip(base_rows.iter()).enumerate() {
                if *m <= *m_base && r >= *m {
                    continue;
                }
                let cmp_rows = (*m).min(*m_base);
                if r >= cmp_rows {
                    continue;
                }
                let ndiff = row.iter().zip(base.iter()).filter(|(a, b)| a != b).count();
                if ndiff > 0 {
                    any_leak.push_str(&format!(
                        "seed={seed} m_logical={m} vs {m_base}: shared row {r} differs in \
                         {ndiff}/{n} outputs\n"
                    ));
                }
            }
        }
    }
    eprintln!(
        "[nvfp4-m128-leak] staged m128 Lt gemm, logical-rows ladder \
         {LOGICAL_ROWS_LADDER_4_6_7_8_THE_CENSUS_ARMS_THAT_FLIP_AT_7:?}:\n{}",
        if any_leak.is_empty() {
            "all shared rows bitwise invariant across the ladder".to_string()
        } else {
            any_leak.clone()
        }
    );
    assert!(
        any_leak.is_empty(),
        "a row of the staged m128 nvfp4 Lt gemm changed when LATER logical rows were quantized \
         into the shared staged buffer; a padded launch must not let one row's content reach \
         another row's output:\n{any_leak}"
    );
}
