#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;

fn exact_floor_log2(x: f32) -> i32 {
    assert!(x.is_finite() && x > 0.0);
    let e = ((x.to_bits() >> 23) & 0xff) as i32;
    assert!(e != 0, "probe scales must be f32-normal, got {x:e}");
    e - 127
}

const UE4M3_MIN_NORMAL: f32 = 1.0 / 64.0;
const UE4M3_SUBNORMAL_STEP: f32 = 1.0 / 512.0;

fn ref_encode_ue4m3_exact_floor(scale: f32) -> u8 {
    if !scale.is_finite() || scale <= 0.0 {
        return 0;
    }
    let clamped = scale.min(448.0);
    if clamped < UE4M3_MIN_NORMAL {
        let m = (clamped / UE4M3_SUBNORMAL_STEP).round() as i32;
        if m <= 0 {
            return 0;
        }
        if m <= 7 {
            return m as u8;
        }
        return 0x08;
    }
    let mut exp_v = exact_floor_log2(clamped);
    let mant_f = clamped / (2f32).powi(exp_v) - 1.0;
    let mut mant = (mant_f * 8.0).round() as i32;
    if mant < 0 {
        mant = 0;
    }
    if mant >= 8 {
        mant = 0;
        exp_v += 1;
    }
    let biased = (exp_v + 7).clamp(1, 15);
    let byte = ((biased as u8) << 3) | (mant as u8 & 0x07);
    if byte == 0x7F {
        0x7E
    } else {
        byte
    }
}

fn cuda_scale_byte(stream: &std::sync::Arc<cudarc::driver::CudaStream>, stored: f32) -> u8 {
    let k = 16usize;
    let row: Vec<u16> = (0..k)
        .map(|i| {
            if i == 0 {
                bf16::from_f32(6.0).to_bits()
            } else {
                0u16
            }
        })
        .collect();
    #[allow(deprecated)]
    let d_x = stream.memcpy_stod(&row).unwrap();
    let mut d_packed = stream.alloc_zeros::<u8>(k / 2).unwrap();
    let mut d_scales = stream.alloc_zeros::<u8>(512).unwrap();
    let rc = {
        let (px, _a) = d_x.device_ptr(stream);
        let (pp, _b) = d_packed.device_ptr_mut(stream);
        let (ps, _c) = d_scales.device_ptr_mut(stream);
        unsafe {
            nv_kernels::cuda::quantize_nvfp4_bf16(
                stream.cu_stream() as *mut c_void,
                px as *const u16,
                pp as *mut u8,
                ps as *mut u8,
                stored,
                1,
                1,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "quantize_nvfp4_bf16 rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let scales = stream.memcpy_dtov(&d_scales).unwrap();
    scales[0]
}

fn step_bits(x: f32, steps: i32) -> f32 {
    let b = x.to_bits() as i64 + steps as i64;
    f32::from_bits(b as u32)
}

#[test]
fn ue4m3_encode_power_of_two_boundaries_match_exact_floor() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "ue4m3_log2f_boundary: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("ue4m3_log2f_boundary: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    let mut probes: Vec<f32> = Vec::new();
    for e in -9..=8i32 {
        let base = (2f32).powi(e);
        for j in -4..=4i32 {
            probes.push(step_bits(base, j));
        }
    }
    for e in [-3i32, 0, 5] {
        let base = (2f32).powi(e);
        for m in 0..8u32 {
            let half_pt = (1.0 + (2 * m + 1) as f32 / 16.0) * base;
            for j in -2..=2i32 {
                probes.push(step_bits(half_pt, j));
            }
        }
    }
    for x in [
        448.0f32,
        step_bits(448.0, -1),
        step_bits(448.0, 1),
        447.5,
        449.0,
        512.0,
    ] {
        probes.push(x);
    }

    let mut mismatches = 0usize;
    for &s in &probes {
        let got = cuda_scale_byte(&stream, s);
        let want = ref_encode_ue4m3_exact_floor(s);
        if got != want {
            mismatches += 1;
            eprintln!(
                "MISMATCH scale={s:.9e} bits=0x{:08x}: cuda=0x{got:02x} exact_floor=0x{want:02x}",
                s.to_bits()
            );
        }
    }
    eprintln!(
        "ue4m3 boundary probes: {} scales, {mismatches} mismatch(es)",
        probes.len()
    );
    assert_eq!(
        mismatches, 0,
        "device floorf(log2f(x)) diverges from the exact floor(log2 x) at {mismatches} boundary scales"
    );
}
