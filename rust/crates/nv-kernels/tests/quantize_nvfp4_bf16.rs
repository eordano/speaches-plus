#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_quant::nvfp4::{swizzle_scales, Nvfp4Tensor, BLOCK_SIZE};
use std::ffi::c_void;

fn run_case(
    m: usize,
    k: usize,
    stored_global: f32,
    seed_off: f32,
) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let ctx = CudaContext::new(0).expect("cuda ctx");
    let stream = ctx.default_stream();

    let host_bf: Vec<bf16> = (0..m * k)
        .map(|n| {
            let i = n / k;
            let j = n % k;
            let v = (((i * k + j) as f32 * 0.013) + seed_off).sin() * 2.5;
            bf16::from_f32(v)
        })
        .collect();
    let rows: Vec<Vec<f32>> = (0..m)
        .map(|i| {
            host_bf[i * k..(i + 1) * k]
                .iter()
                .map(|v| v.to_f32())
                .collect()
        })
        .collect();

    let q_cpu = Nvfp4Tensor::quantize_rows_with_global(&rows, stored_global);
    let scales_sw_cpu = swizzle_scales(&q_cpu.scales, m, k / BLOCK_SIZE);

    #[allow(deprecated)]
    let x_dev = stream
        .memcpy_stod(unsafe {
            std::slice::from_raw_parts(host_bf.as_ptr() as *const u16, host_bf.len())
        })
        .unwrap();
    let packed_bytes = m * k / 2;
    let scales_bytes = ((m + 127) / 128) * 128 * ((k / BLOCK_SIZE + 3) / 4) * 4;
    let mut packed_dev = stream.alloc_zeros::<u8>(packed_bytes).unwrap();
    let mut scales_dev = stream.alloc_zeros::<u8>(scales_bytes).unwrap();

    let rc = {
        let (x_ptr, _gx) = x_dev.device_ptr(&stream);
        let (p_ptr, _gp) = packed_dev.device_ptr_mut(&stream);
        let (s_ptr, _gs) = scales_dev.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::quantize_nvfp4_bf16(
                stream.cu_stream() as *mut c_void,
                x_ptr as *const u16,
                p_ptr as *mut u8,
                s_ptr as *mut u8,
                stored_global,
                m as i32,
                m as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "quantize_nvfp4_bf16 rc={rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let packed_got = stream.memcpy_dtov(&packed_dev).unwrap();
    #[allow(deprecated)]
    let scales_got = stream.memcpy_dtov(&scales_dev).unwrap();

    (q_cpu.data, scales_sw_cpu, packed_got, scales_got)
}

#[test]
fn gpu_nvfp4_quantize_matches_cpu_basic_global_1() {
    let (packed_cpu, scales_cpu, packed_got, scales_got) = run_case(128, 128, 1.0, 0.0);
    assert_eq!(packed_cpu.len(), packed_got.len());
    let mismatches: usize = packed_cpu
        .iter()
        .zip(packed_got.iter())
        .filter(|(a, b)| a != b)
        .count();
    let near_match: usize = packed_cpu
        .iter()
        .zip(packed_got.iter())
        .filter(|(a, b)| {
            let (lo_a, hi_a) = ((**a & 0x0F), ((**a >> 4) & 0x0F));
            let (lo_b, hi_b) = ((**b & 0x0F), ((**b >> 4) & 0x0F));
            let dlo = (lo_a as i32 - lo_b as i32).abs();
            let dhi = (hi_a as i32 - hi_b as i32).abs();
            (dlo <= 1) && (dhi <= 1)
        })
        .count();
    let n = packed_cpu.len();
    eprintln!("packed: total={n} mismatches={mismatches} all-near-by-1={near_match}",);
    assert_eq!(
        mismatches, 0,
        "GPU vs CPU packed FP4 differs at {mismatches}/{n} bytes"
    );
    assert_eq!(scales_cpu.len(), scales_got.len());
    let sc_mismatches: usize = scales_cpu
        .iter()
        .zip(scales_got.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        sc_mismatches, 0,
        "swizzled scales differ at {sc_mismatches} bytes"
    );
}

#[test]
fn gpu_nvfp4_quantize_matches_cpu_large_global() {
    let (packed_cpu, scales_cpu, packed_got, scales_got) = run_case(128, 4096, 0.3, 1.7);
    assert_eq!(packed_cpu.len(), packed_got.len());
    let mismatches: usize = packed_cpu
        .iter()
        .zip(packed_got.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        mismatches, 0,
        "GPU vs CPU packed FP4 differs at {mismatches} bytes"
    );
    assert_eq!(scales_cpu.len(), scales_got.len());
    let sc_mismatches: usize = scales_cpu
        .iter()
        .zip(scales_got.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        sc_mismatches, 0,
        "swizzled scales differ at {sc_mismatches} bytes"
    );
}
