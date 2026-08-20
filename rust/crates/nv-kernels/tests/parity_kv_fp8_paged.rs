#![cfg(all(feature = "cuda", feature = "wgpu"))]

mod common;
use common::backends;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::kv_fp8_paged;
use std::ffi::c_void;
use std::sync::Arc;
use common::sample_bf16;

fn tie_heavy_bf16(n_tokens: usize, n_kv: usize, head_dim: usize) -> Vec<u16> {
    let ties: [f32; 16] = [
        1.0625,
        1.1875,
        1.3125,
        1.4375,
        1.5625,
        1.6875,
        1.8125,
        1.9375,
        0.000_976_562_5,
        0.002_929_687_5,
        0.004_882_812_5,
        0.006_835_937_5,
        0.008_789_062_5,
        0.010_742_187_5,
        0.012_695_312_5,
        0.014_648_437_5,
    ];
    let mut x = vec![0u16; n_tokens * n_kv * head_dim];
    for token in 0..n_tokens {
        for kv_head in 0..n_kv {
            let base = (token * n_kv + kv_head) * head_dim;
            x[base] = bf16::from_f32(448.0).to_bits();
            for d in 1..head_dim {
                let mut v = ties[(d + kv_head) % ties.len()];
                if (d + token) % 2 == 0 {
                    v = -v;
                }
                x[base + d] = bf16::from_f32(v).to_bits();
            }
        }
    }
    x
}

fn extreme_bf16(n_tokens: usize, n_kv: usize, head_dim: usize) -> Vec<u16> {
    let mut x = vec![0u16; n_tokens * n_kv * head_dim];
    for token in 0..n_tokens {
        for kv_head in 0..n_kv {
            let base = (token * n_kv + kv_head) * head_dim;
            match token % 4 {
                0 => {}
                1 => {
                    for d in 0..head_dim {
                        x[base + d] = bf16::from_f32(1e-8 * (d as f32 + 1.0)).to_bits();
                    }
                }
                2 => {
                    for d in 0..head_dim {
                        let v = if d % 7 == 0 { 60000.0 } else { 1e-4 * d as f32 };
                        x[base + d] = bf16::from_f32(v).to_bits();
                    }
                }
                _ => {
                    for d in 0..head_dim {
                        let v = if d % 2 == 0 { -0.0 } else { 3.5 };
                        x[base + d] = bf16::from_f32(v).to_bits();
                    }
                }
            }
        }
    }
    x
}

const MAGNITUDE_MULTS: [f32; 12] = [
    1.0, -0.5, 0.75, -0.25, 0.9375, -0.875, 0.125, -0.0625, 0.5, -1.0, 0.375, -0.75,
];

fn pow2(e: i32) -> f32 {
    if (-126..=127).contains(&e) {
        f32::from_bits(((e + 127) as u32) << 23)
    } else if (-149..-126).contains(&e) {
        f32::from_bits(1u32 << (e + 149) as u32)
    } else {
        0.0
    }
}

fn magnitude_bf16(n_tokens: usize, n_kv: usize, head_dim: usize, exp: i32) -> Vec<u16> {
    let amp = pow2(exp);
    let mut x = vec![0u16; n_tokens * n_kv * head_dim];
    for token in 0..n_tokens {
        for kv_head in 0..n_kv {
            let p = token * n_kv + kv_head;
            let base = p * head_dim;
            x[base] = bf16::from_f32(amp).to_bits();
            for d in 1..head_dim {
                x[base + d] =
                    bf16::from_f32(amp * MAGNITUDE_MULTS[(d + p) % MAGNITUDE_MULTS.len()])
                        .to_bits();
            }
        }
    }
    x
}

fn hashed_bytes(n: usize, seed: u32) -> Vec<u8> {
    (0..n)
        .map(|j| {
            let mut s = (j as u32).wrapping_mul(2_654_435_761) ^ seed.wrapping_mul(0x9e37_79b9);
            s ^= s >> 15;
            s = s.wrapping_mul(0x85eb_ca6b);
            s ^= s >> 13;
            s = s.wrapping_mul(0xc2b2_ae35);
            s ^= s >> 16;
            (s & 0xff) as u8
        })
        .collect()
}

struct QuantResult {
    fp8: Vec<u8>,
    scales: Vec<f32>,
}

fn cuda_quantize_paged(
    stream: &Arc<CudaStream>,
    x: &[u16],
    base_fp8: &[u8],
    base_scales: &[f32],
    block_table: &[i32],
    start: i32,
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
    block_size: usize,
) -> QuantResult {
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.clone_htod(x).unwrap();
    #[allow(deprecated)]
    let mut dfp8: CudaSlice<u8> = stream.clone_htod(base_fp8).unwrap();
    #[allow(deprecated)]
    let mut dscales: CudaSlice<f32> = stream.clone_htod(base_scales).unwrap();
    #[allow(deprecated)]
    let dstart: CudaSlice<i32> = stream.clone_htod(&[start]).unwrap();
    #[allow(deprecated)]
    let dtable: CudaSlice<i32> = stream.clone_htod(block_table).unwrap();
    let rc = {
        let (px, _a) = dx.device_ptr(stream);
        let (pstart, _d) = dstart.device_ptr(stream);
        let (ptable, _e) = dtable.device_ptr(stream);
        let (pfp8, _b) = dfp8.device_ptr_mut(stream);
        let (psc, _c) = dscales.device_ptr_mut(stream);
        unsafe {
            cuda::quantize_kv_fp8_paged(
                stream.cu_stream() as *mut c_void,
                px as *const u16,
                pfp8 as *mut u8,
                psc as *mut f32,
                pstart as *const i32,
                ptable as *const i32,
                block_size as i32,
                n_tokens as i32,
                n_kv as i32,
                head_dim as i32,
            )
        }
    };
    assert_eq!(rc, 0, "cuda quantize_kv_fp8_paged rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let fp8 = stream.memcpy_dtov(&dfp8).unwrap();
    #[allow(deprecated)]
    let scales = stream.memcpy_dtov(&dscales).unwrap();
    QuantResult { fp8, scales }
}

fn cuda_dequantize_paged(
    stream: &Arc<CudaStream>,
    src: &[u8],
    scales: &[f32],
    block_table: &[i32],
    len: usize,
    n_kv: usize,
    head_dim: usize,
    block_size: usize,
) -> Vec<u16> {
    #[allow(deprecated)]
    let dsrc: CudaSlice<u8> = stream.clone_htod(src).unwrap();
    #[allow(deprecated)]
    let dsc: CudaSlice<f32> = stream.clone_htod(scales).unwrap();
    #[allow(deprecated)]
    let dtable: CudaSlice<i32> = stream.clone_htod(block_table).unwrap();
    let mut dout: CudaSlice<u16> = stream.alloc_zeros::<u16>(len * n_kv * head_dim).unwrap();
    let rc = {
        let (psrc, _a) = dsrc.device_ptr(stream);
        let (psc, _b) = dsc.device_ptr(stream);
        let (ptable, _d) = dtable.device_ptr(stream);
        let (pout, _c) = dout.device_ptr_mut(stream);
        unsafe {
            cuda::dequantize_kv_fp8_paged(
                stream.cu_stream() as *mut c_void,
                psrc as *const u8,
                psc as *const f32,
                pout as *mut u16,
                ptable as *const i32,
                block_size as i32,
                len as i32,
                n_kv as i32,
                head_dim as i32,
            )
        }
    };
    assert_eq!(rc, 0, "cuda dequantize_kv_fp8_paged rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let out = stream.memcpy_dtov(&dout).unwrap();
    out
}

fn cuda_copy_block_inplace(
    stream: &Arc<CudaStream>,
    fp8: &[u8],
    scales: &[f32],
    src_block: usize,
    dst_block: usize,
    block_size: usize,
    n_kv: usize,
    head_dim: usize,
) -> QuantResult {
    #[allow(deprecated)]
    let mut dfp8: CudaSlice<u8> = stream.clone_htod(fp8).unwrap();
    #[allow(deprecated)]
    let mut dsc: CudaSlice<f32> = stream.clone_htod(scales).unwrap();
    let rc = {
        let (pfp8, _a) = dfp8.device_ptr_mut(stream);
        let (psc, _b) = dsc.device_ptr_mut(stream);
        unsafe {
            cuda::copy_kv_block_fp8(
                stream.cu_stream() as *mut c_void,
                pfp8 as *const u8,
                psc as *const f32,
                pfp8 as *mut u8,
                psc as *mut f32,
                src_block as i32,
                dst_block as i32,
                block_size as i32,
                n_kv as i32,
                head_dim as i32,
            )
        }
    };
    assert_eq!(rc, 0, "cuda copy_kv_block_fp8 rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let out_fp8 = stream.memcpy_dtov(&dfp8).unwrap();
    #[allow(deprecated)]
    let out_sc = stream.memcpy_dtov(&dsc).unwrap();
    QuantResult {
        fp8: out_fp8,
        scales: out_sc,
    }
}

fn cuda_copy_block_cross(
    stream: &Arc<CudaStream>,
    src_fp8: &[u8],
    src_scales: &[f32],
    dst_fp8: &[u8],
    dst_scales: &[f32],
    src_block: usize,
    dst_block: usize,
    block_size: usize,
    n_kv: usize,
    head_dim: usize,
) -> QuantResult {
    #[allow(deprecated)]
    let dsrc: CudaSlice<u8> = stream.clone_htod(src_fp8).unwrap();
    #[allow(deprecated)]
    let dsrc_sc: CudaSlice<f32> = stream.clone_htod(src_scales).unwrap();
    #[allow(deprecated)]
    let mut ddst: CudaSlice<u8> = stream.clone_htod(dst_fp8).unwrap();
    #[allow(deprecated)]
    let mut ddst_sc: CudaSlice<f32> = stream.clone_htod(dst_scales).unwrap();
    let rc = {
        let (psrc, _a) = dsrc.device_ptr(stream);
        let (psrc_sc, _b) = dsrc_sc.device_ptr(stream);
        let (pdst, _c) = ddst.device_ptr_mut(stream);
        let (pdst_sc, _d) = ddst_sc.device_ptr_mut(stream);
        unsafe {
            cuda::copy_kv_block_fp8(
                stream.cu_stream() as *mut c_void,
                psrc as *const u8,
                psrc_sc as *const f32,
                pdst as *mut u8,
                pdst_sc as *mut f32,
                src_block as i32,
                dst_block as i32,
                block_size as i32,
                n_kv as i32,
                head_dim as i32,
            )
        }
    };
    assert_eq!(rc, 0, "cuda copy_kv_block_fp8 (cross) rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let out_fp8 = stream.memcpy_dtov(&ddst).unwrap();
    #[allow(deprecated)]
    let out_sc = stream.memcpy_dtov(&ddst_sc).unwrap();
    QuantResult {
        fp8: out_fp8,
        scales: out_sc,
    }
}

fn compare_quant(name: &str, cu: &QuantResult, wg: &QuantResult) {
    assert_eq!(cu.fp8.len(), wg.fp8.len());
    assert_eq!(cu.scales.len(), wg.scales.len());
    let byte_diff = cu
        .fp8
        .iter()
        .zip(wg.fp8.iter())
        .filter(|(a, b)| a != b)
        .count();
    let mut first = (0u8, 0u8, usize::MAX);
    for (i, (a, b)) in cu.fp8.iter().zip(wg.fp8.iter()).enumerate() {
        if a != b {
            first = (*a, *b, i);
            break;
        }
    }
    let scale_diff = cu
        .scales
        .iter()
        .zip(wg.scales.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let max_scale_err = cu
        .scales
        .iter()
        .zip(wg.scales.iter())
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!(
        "{name}: fp8 bytes differing {byte_diff}/{}, scales differing (bitwise) {scale_diff}/{}, max scale abs err {max_scale_err:e}",
        cu.fp8.len(),
        cu.scales.len()
    );
    assert_eq!(
        byte_diff, 0,
        "{name}: fp8 must be byte-exact; first mismatch at {} cuda {:#04x} wgpu {:#04x}",
        first.2, first.0, first.1
    );
    assert_eq!(scale_diff, 0, "{name}: scales must be bit-exact");
}

struct Case {
    n_tokens: usize,
    n_kv: usize,
    head_dim: usize,
    block_size: usize,
    table: Vec<i32>,
    slots: usize,
    start: i32,
}

fn quantize_cases() -> Vec<Case> {
    vec![
        Case {
            n_tokens: 5,
            n_kv: 3,
            head_dim: 128,
            block_size: 4,
            table: vec![3, 1, 0],
            slots: 16,
            start: 2,
        },
        Case {
            n_tokens: 4,
            n_kv: 2,
            head_dim: 64,
            block_size: 1,
            table: vec![5, 0, 3, 1],
            slots: 6,
            start: 0,
        },
        Case {
            n_tokens: 7,
            n_kv: 4,
            head_dim: 256,
            block_size: 3,
            table: vec![2, 0, 1],
            slots: 9,
            start: 1,
        },
        Case {
            n_tokens: 2,
            n_kv: 8,
            head_dim: 128,
            block_size: 4,
            table: vec![1, 0],
            slots: 8,
            start: 3,
        },
        Case {
            n_tokens: 9,
            n_kv: 1,
            head_dim: 512,
            block_size: 5,
            table: vec![2, 0, 3, 1],
            slots: 20,
            start: 4,
        },
        Case {
            n_tokens: 5,
            n_kv: 5,
            head_dim: 32,
            block_size: 2,
            table: vec![3, 0, 2],
            slots: 8,
            start: 0,
        },
        Case {
            n_tokens: 6,
            n_kv: 2,
            head_dim: 64,
            block_size: 4,
            table: vec![6, 0, 5, 1, 3],
            slots: 32,
            start: 9,
        },
        Case {
            n_tokens: 5,
            n_kv: 3,
            head_dim: 16,
            block_size: 2,
            table: vec![4, 1, 3],
            slots: 10,
            start: 1,
        },
        Case {
            n_tokens: 3,
            n_kv: 2,
            head_dim: 4,
            block_size: 1,
            table: vec![2, 0, 1],
            slots: 3,
            start: 0,
        },
    ]
}

fn run_quantize_case(stream: &Arc<CudaStream>, wg: &WgpuContext, label: &str, c: &Case, x: &[u16]) {
    let base_fp8: Vec<u8> = (0..c.slots * c.n_kv * c.head_dim)
        .map(|i| (i % 251) as u8)
        .collect();
    let base_scales: Vec<f32> = (0..c.slots * c.n_kv).map(|i| -1.0 - i as f32).collect();

    let cu = cuda_quantize_paged(
        stream,
        x,
        &base_fp8,
        &base_scales,
        &c.table,
        c.start,
        c.n_tokens,
        c.n_kv,
        c.head_dim,
        c.block_size,
    );

    let mut fp8 = base_fp8.clone();
    let mut scales = base_scales.clone();
    kv_fp8_paged::quantize_kv_fp8_paged(
        wg,
        x,
        &mut fp8,
        &mut scales,
        &c.table,
        &[c.start],
        c.n_tokens,
        c.n_kv,
        c.head_dim,
        c.block_size,
    )
    .expect("wgpu quantize_kv_fp8_paged");

    let touched = cu
        .fp8
        .iter()
        .zip(base_fp8.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        touched > 0,
        "{label}: cuda quantize wrote nothing - the case is degenerate"
    );

    let mut cpu_fp8 = base_fp8.clone();
    let mut cpu_scales = base_scales.clone();
    kv_fp8_paged::cpu_quantize_kv_fp8_paged(
        x,
        &mut cpu_fp8,
        &mut cpu_scales,
        &c.table,
        c.start as usize,
        c.n_tokens,
        c.n_kv,
        c.head_dim,
        c.block_size,
    );
    compare_quant(
        &format!("{label} cpu-oracle"),
        &cu,
        &QuantResult {
            fp8: cpu_fp8,
            scales: cpu_scales,
        },
    );

    compare_quant(label, &cu, &QuantResult { fp8, scales });
}

#[test]
fn quantize_kv_fp8_paged_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("quantize_kv_fp8_paged_cuda_vs_wgpu") else {
        return;
    };
    for (i, c) in quantize_cases().iter().enumerate() {
        let x = sample_bf16(c.n_tokens * c.n_kv * c.head_dim, 7 + i as u32);
        run_quantize_case(
            &stream,
            wg,
            &format!(
                "quantize case{i} t{} kv{} d{} bs{} start{} table{:?}",
                c.n_tokens, c.n_kv, c.head_dim, c.block_size, c.start, c.table
            ),
            c,
            &x,
        );
    }
}

#[test]
fn quantize_kv_fp8_paged_ties_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("quantize_kv_fp8_paged_ties_cuda_vs_wgpu") else {
        return;
    };
    let c = Case {
        n_tokens: 5,
        n_kv: 3,
        head_dim: 128,
        block_size: 4,
        table: vec![3, 1, 0],
        slots: 16,
        start: 2,
    };
    let x = tie_heavy_bf16(c.n_tokens, c.n_kv, c.head_dim);
    run_quantize_case(&stream, wg, "quantize paged ties", &c, &x);
}

#[test]
fn quantize_kv_fp8_paged_extremes_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("quantize_kv_fp8_paged_extremes_cuda_vs_wgpu") else {
        return;
    };
    let c = Case {
        n_tokens: 7,
        n_kv: 4,
        head_dim: 256,
        block_size: 3,
        table: vec![2, 0, 1],
        slots: 9,
        start: 1,
    };
    let x = extreme_bf16(c.n_tokens, c.n_kv, c.head_dim);
    run_quantize_case(&stream, wg, "quantize paged extremes", &c, &x);
}

fn magnitude_diffs(
    stream: &Arc<CudaStream>,
    wg: &WgpuContext,
    c: &Case,
    exp: i32,
) -> (usize, usize, usize, usize) {
    let x = magnitude_bf16(c.n_tokens, c.n_kv, c.head_dim, exp);
    let base_fp8: Vec<u8> = (0..c.slots * c.n_kv * c.head_dim)
        .map(|i| (i % 251) as u8)
        .collect();
    let base_scales: Vec<f32> = (0..c.slots * c.n_kv).map(|i| -1.0 - i as f32).collect();

    let cu = cuda_quantize_paged(
        stream,
        &x,
        &base_fp8,
        &base_scales,
        &c.table,
        c.start,
        c.n_tokens,
        c.n_kv,
        c.head_dim,
        c.block_size,
    );

    let mut fp8 = base_fp8.clone();
    let mut scales = base_scales.clone();
    kv_fp8_paged::quantize_kv_fp8_paged(
        wg,
        &x,
        &mut fp8,
        &mut scales,
        &c.table,
        &[c.start],
        c.n_tokens,
        c.n_kv,
        c.head_dim,
        c.block_size,
    )
    .expect("wgpu quantize_kv_fp8_paged");

    let mut cpu_fp8 = base_fp8.clone();
    let mut cpu_scales = base_scales.clone();
    kv_fp8_paged::cpu_quantize_kv_fp8_paged(
        &x,
        &mut cpu_fp8,
        &mut cpu_scales,
        &c.table,
        c.start as usize,
        c.n_tokens,
        c.n_kv,
        c.head_dim,
        c.block_size,
    );

    let wb = cu
        .fp8
        .iter()
        .zip(fp8.iter())
        .filter(|(a, b)| a != b)
        .count();
    let ws = cu
        .scales
        .iter()
        .zip(scales.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let cb = cu
        .fp8
        .iter()
        .zip(cpu_fp8.iter())
        .filter(|(a, b)| a != b)
        .count();
    let cs = cu
        .scales
        .iter()
        .zip(cpu_scales.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    (wb, ws, cb, cs)
}

fn magnitude_case() -> Case {
    Case {
        n_tokens: 18,
        n_kv: 4,
        head_dim: 64,
        block_size: 5,
        table: vec![3, 0, 2, 1],
        slots: 20,
        start: 1,
    }
}

const NORMAL_SCALE_MIN_EXP: i32 = -122;
const NORMAL_SCALE_MAX_EXP: i32 = 127;

#[test]
fn quantize_kv_fp8_paged_magnitudes_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("quantize_kv_fp8_paged_magnitudes_cuda_vs_wgpu") else {
        return;
    };
    let c = magnitude_case();
    let mut worst = (0usize, 0usize, 0i32);
    for exp in NORMAL_SCALE_MIN_EXP..=NORMAL_SCALE_MAX_EXP {
        let (wb, ws, cb, cs) = magnitude_diffs(&stream, wg, &c, exp);
        assert_eq!(
            cb, 0,
            "magnitude exp={exp} (amax=2^{exp}): cpu oracle must be byte-exact vs cuda"
        );
        assert_eq!(
            cs, 0,
            "magnitude exp={exp} (amax=2^{exp}): cpu oracle scales must be bit-exact vs cuda"
        );
        assert_eq!(
            wb, 0,
            "magnitude exp={exp} (amax=2^{exp}): wgpu fp8 must be byte-exact vs cuda"
        );
        assert_eq!(
            ws, 0,
            "magnitude exp={exp} (amax=2^{exp}): wgpu scales must be bit-exact vs cuda"
        );
        if wb + ws > worst.0 + worst.1 {
            worst = (wb, ws, exp);
        }
    }
    let spans = NORMAL_SCALE_MAX_EXP - NORMAL_SCALE_MIN_EXP + 1;
    eprintln!(
        "magnitude sweep: {spans} exponents 2^{NORMAL_SCALE_MIN_EXP}..2^{NORMAL_SCALE_MAX_EXP}, \
         worst fp8 bytes {} scales {} at exp {}",
        worst.0, worst.1, worst.2
    );
}

#[test]
fn quantize_kv_fp8_paged_subnormal_input_band_is_the_only_gap() {
    let Some((stream, wg)) = backends("quantize_kv_fp8_paged_subnormal_input_band_is_the_only_gap")
    else {
        return;
    };
    let c = magnitude_case();
    let (lo_b, lo_s, ..) = magnitude_diffs(&stream, wg, &c, NORMAL_SCALE_MIN_EXP - 1);
    assert!(
        lo_b + lo_s > 0,
        "exp {} should still be outside the bit-exact band; if this now passes, widen \
         NORMAL_SCALE_MIN_EXP rather than deleting this test",
        NORMAL_SCALE_MIN_EXP - 1
    );
    assert_eq!(
        pow2(NORMAL_SCALE_MAX_EXP) * 2.0,
        f32::INFINITY,
        "the sweep must reach the top finite binade of f32"
    );
    assert_eq!(
        pow2(NORMAL_SCALE_MAX_EXP + 1),
        0.0,
        "there is no representable binade above NORMAL_SCALE_MAX_EXP, so no high-side gap can \
         exist; if this ever changes, extend the sweep rather than deleting this test"
    );
    let (hi_b, hi_s, ..) = magnitude_diffs(&stream, wg, &c, NORMAL_SCALE_MAX_EXP);
    assert_eq!(
        hi_b + hi_s,
        0,
        "the top binade 2^{NORMAL_SCALE_MAX_EXP} must be inside the bit-exact band"
    );
    eprintln!(
        "band pinned: exp {} (subnormal inputs) differs by {lo_b} bytes/{lo_s} scales; \
         top binade exp {NORMAL_SCALE_MAX_EXP} is exact ({hi_b} bytes/{hi_s} scales)",
        NORMAL_SCALE_MIN_EXP - 1
    );
}

#[test]
fn dequantize_kv_fp8_paged_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("dequantize_kv_fp8_paged_cuda_vs_wgpu") else {
        return;
    };
    for (i, c) in quantize_cases().iter().enumerate() {
        let len = c.n_tokens;
        let src: Vec<u8> = (0..c.slots * c.n_kv * c.head_dim)
            .map(|j| {
                let b = ((j * 7 + i) % 256) as u8;
                if b & 0x7f == 0x7f {
                    0x12
                } else {
                    b
                }
            })
            .collect();
        let scales: Vec<f32> = (0..c.slots * c.n_kv)
            .map(|j| 0.002 + (j as f32) * 0.0037)
            .collect();

        let cu = cuda_dequantize_paged(
            &stream,
            &src,
            &scales,
            &c.table,
            len,
            c.n_kv,
            c.head_dim,
            c.block_size,
        );

        let mut out = vec![0u16; len * c.n_kv * c.head_dim];
        kv_fp8_paged::dequantize_kv_fp8_paged(
            wg,
            &src,
            &scales,
            &c.table,
            &mut out,
            len,
            c.n_kv,
            c.head_dim,
            c.block_size,
        )
        .expect("wgpu dequantize_kv_fp8_paged");

        let mut cpu = vec![0u16; len * c.n_kv * c.head_dim];
        kv_fp8_paged::cpu_dequantize_kv_fp8_paged(
            &src,
            &scales,
            &c.table,
            &mut cpu,
            len,
            c.n_kv,
            c.head_dim,
            c.block_size,
        );

        let diff = cu.iter().zip(out.iter()).filter(|(a, b)| a != b).count();
        let cpu_diff = cu.iter().zip(cpu.iter()).filter(|(a, b)| a != b).count();
        let max_ulp = cu
            .iter()
            .zip(out.iter())
            .fold(0i32, |m, (a, b)| m.max((*a as i32 - *b as i32).abs()));
        eprintln!(
            "dequantize case{i} len{len} kv{} d{} bs{} table{:?}: {diff}/{} bf16 words differ (cpu oracle {cpu_diff}), max_ulp={max_ulp}",
            c.n_kv,
            c.head_dim,
            c.block_size,
            c.table,
            cu.len()
        );
        assert_eq!(
            cpu_diff, 0,
            "dequantize case{i}: cpu oracle must be bit-exact"
        );
        assert_eq!(diff, 0, "dequantize case{i}: must be bit-exact");
    }
}

#[test]
fn copy_kv_block_fp8_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("copy_kv_block_fp8_cuda_vs_wgpu") else {
        return;
    };
    for (i, (n_kv, head_dim, block_size, blocks, src_block, dst_block)) in [
        (3usize, 128usize, 4usize, 5usize, 0usize, 3usize),
        (3, 128, 4, 5, 3, 0),
        (2, 64, 1, 6, 5, 1),
        (4, 256, 3, 4, 1, 2),
        (1, 512, 5, 3, 2, 0),
        (2, 128, 4, 4, 2, 2),
        (3, 16, 7, 4, 0, 2),
        (2, 4, 1, 3, 2, 0),
    ]
    .iter()
    .copied()
    .enumerate()
    {
        let slots = blocks * block_size;
        let n = slots * n_kv * head_dim;
        let k_fp8 = hashed_bytes(n, 0x51ed + i as u32);
        let v_fp8 = hashed_bytes(n, 0xa37c + i as u32);
        let k_scales: Vec<f32> = (0..slots * n_kv).map(|j| 0.5 + j as f32 * 0.125).collect();
        let v_scales: Vec<f32> = (0..slots * n_kv)
            .map(|j| -0.25 - j as f32 * 0.0625)
            .collect();

        let cu_k = cuda_copy_block_inplace(
            &stream, &k_fp8, &k_scales, src_block, dst_block, block_size, n_kv, head_dim,
        );
        let cu_v = cuda_copy_block_inplace(
            &stream, &v_fp8, &v_scales, src_block, dst_block, block_size, n_kv, head_dim,
        );

        let mut wk = k_fp8.clone();
        let mut wv = v_fp8.clone();
        let mut wks = k_scales.clone();
        let mut wvs = v_scales.clone();
        kv_fp8_paged::copy_kv_block_fp8(
            wg, &mut wk, &mut wv, &mut wks, &mut wvs, src_block, dst_block, block_size, n_kv,
            head_dim,
        )
        .expect("wgpu copy_kv_block_fp8");

        let mut ck = k_fp8.clone();
        let mut cks = k_scales.clone();
        kv_fp8_paged::cpu_copy_kv_block_fp8(
            &mut ck, &mut cks, src_block, dst_block, block_size, n_kv, head_dim,
        );

        compare_quant(
            &format!("copy case{i} K kv{n_kv} d{head_dim} bs{block_size} {src_block}->{dst_block}"),
            &cu_k,
            &QuantResult {
                fp8: wk,
                scales: wks,
            },
        );
        compare_quant(
            &format!("copy case{i} V kv{n_kv} d{head_dim} bs{block_size} {src_block}->{dst_block}"),
            &cu_v,
            &QuantResult {
                fp8: wv,
                scales: wvs,
            },
        );
        compare_quant(
            &format!("copy case{i} K cpu-oracle"),
            &cu_k,
            &QuantResult {
                fp8: ck,
                scales: cks,
            },
        );

        if src_block == dst_block {
            assert_eq!(cu_k.fp8, k_fp8, "copy case{i}: src==dst must be a no-op");
            assert_eq!(
                cu_k.scales, k_scales,
                "copy case{i}: src==dst must be a no-op"
            );
        } else {
            let moved = cu_k
                .fp8
                .iter()
                .zip(k_fp8.iter())
                .filter(|(a, b)| a != b)
                .count();
            let moved_scales = cu_k
                .scales
                .iter()
                .zip(k_scales.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!(
                "copy case{i}: cuda moved {moved}/{} fp8 bytes and {moved_scales}/{} scales",
                n,
                slots * n_kv
            );
            assert!(moved > 0, "copy case{i}: cuda copy wrote no fp8 bytes");
            assert_eq!(
                moved_scales,
                block_size * n_kv,
                "copy case{i}: cuda copy must rewrite exactly one block of scales"
            );
            let untouched = block_size * n_kv * head_dim;
            assert!(
                moved >= untouched * 3 / 4,
                "copy case{i}: only {moved} of {untouched} destination bytes changed"
            );
        }
    }
}

#[test]
fn paged_round_trip_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("paged_round_trip_cuda_vs_wgpu") else {
        return;
    };
    let (n_tokens, n_kv, head_dim, block_size) = (7usize, 3usize, 128usize, 3usize);
    let table: Vec<i32> = vec![2, 0, 1];
    let slots = 9usize;
    let x = sample_bf16(n_tokens * n_kv * head_dim, 11);
    let base_fp8 = vec![0u8; slots * n_kv * head_dim];
    let base_scales = vec![0f32; slots * n_kv];

    let cu = cuda_quantize_paged(
        &stream,
        &x,
        &base_fp8,
        &base_scales,
        &table,
        0,
        n_tokens,
        n_kv,
        head_dim,
        block_size,
    );
    let cu_back = cuda_dequantize_paged(
        &stream, &cu.fp8, &cu.scales, &table, n_tokens, n_kv, head_dim, block_size,
    );

    let mut fp8 = base_fp8.clone();
    let mut scales = base_scales.clone();
    kv_fp8_paged::quantize_kv_fp8_paged(
        wg,
        &x,
        &mut fp8,
        &mut scales,
        &table,
        &[0],
        n_tokens,
        n_kv,
        head_dim,
        block_size,
    )
    .expect("wgpu quantize");
    let mut wg_back = vec![0u16; x.len()];
    kv_fp8_paged::dequantize_kv_fp8_paged(
        wg,
        &fp8,
        &scales,
        &table,
        &mut wg_back,
        n_tokens,
        n_kv,
        head_dim,
        block_size,
    )
    .expect("wgpu dequantize");

    let mut max_abs = 0f32;
    for (a, b) in cu_back.iter().zip(wg_back.iter()) {
        let av = f32::from_bits((*a as u32) << 16);
        let bv = f32::from_bits((*b as u32) << 16);
        max_abs = max_abs.max((av - bv).abs());
    }
    let diff = cu_back
        .iter()
        .zip(wg_back.iter())
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "paged round trip: {diff}/{} bf16 words differ, max abs err {max_abs:e}",
        cu_back.len()
    );
    assert_eq!(cu_back, wg_back, "paged round trip must be bit-exact");
}

#[test]
fn copy_then_dequantize_reads_the_moved_page() {
    let Some((stream, wg)) = backends("copy_then_dequantize_reads_the_moved_page") else {
        return;
    };
    let (n_tokens, n_kv, head_dim, block_size) = (4usize, 2usize, 128usize, 4usize);
    let slots = 12usize;
    let x = sample_bf16(n_tokens * n_kv * head_dim, 3);
    let base_fp8 = vec![0u8; slots * n_kv * head_dim];
    let base_scales = vec![0f32; slots * n_kv];
    let table_write: Vec<i32> = vec![2];

    let mut fp8 = base_fp8.clone();
    let mut scales = base_scales.clone();
    kv_fp8_paged::quantize_kv_fp8_paged(
        wg,
        &x,
        &mut fp8,
        &mut scales,
        &table_write,
        &[0],
        n_tokens,
        n_kv,
        head_dim,
        block_size,
    )
    .expect("wgpu quantize");

    let mut dummy_v = fp8.clone();
    let mut dummy_vs = scales.clone();
    let mut moved = fp8.clone();
    let mut moved_scales = scales.clone();
    kv_fp8_paged::copy_kv_block_fp8(
        wg,
        &mut moved,
        &mut dummy_v,
        &mut moved_scales,
        &mut dummy_vs,
        2,
        0,
        block_size,
        n_kv,
        head_dim,
    )
    .expect("wgpu copy");

    let cu_moved =
        cuda_copy_block_inplace(&stream, &fp8, &scales, 2, 0, block_size, n_kv, head_dim);
    compare_quant(
        "copy page 2 -> page 0",
        &cu_moved,
        &QuantResult {
            fp8: moved.clone(),
            scales: moved_scales.clone(),
        },
    );

    let table_read: Vec<i32> = vec![0];
    let cu_back = cuda_dequantize_paged(
        &stream,
        &moved,
        &moved_scales,
        &table_read,
        n_tokens,
        n_kv,
        head_dim,
        block_size,
    );
    let mut wg_back = vec![0u16; x.len()];
    kv_fp8_paged::dequantize_kv_fp8_paged(
        wg,
        &moved,
        &moved_scales,
        &table_read,
        &mut wg_back,
        n_tokens,
        n_kv,
        head_dim,
        block_size,
    )
    .expect("wgpu dequantize");

    let diff = cu_back
        .iter()
        .zip(wg_back.iter())
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "copy+dequantize: {diff}/{} bf16 words differ",
        cu_back.len()
    );
    assert_eq!(cu_back, wg_back, "copy then dequantize must be bit-exact");

    let mut direct = vec![0u16; x.len()];
    kv_fp8_paged::dequantize_kv_fp8_paged(
        wg,
        &fp8,
        &scales,
        &table_write,
        &mut direct,
        n_tokens,
        n_kv,
        head_dim,
        block_size,
    )
    .expect("wgpu dequantize direct");
    assert_eq!(
        direct, wg_back,
        "the moved page must dequantize to the same values as the original page"
    );
}

#[test]
fn dequantize_kv_fp8_paged_shared_pages_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("dequantize_kv_fp8_paged_shared_pages_cuda_vs_wgpu") else {
        return;
    };
    for (i, (n_kv, head_dim, block_size, slots, len, table)) in [
        (3usize, 128usize, 4usize, 12usize, 10usize, vec![2i32, 0, 2]),
        (2, 64, 2, 8, 7, vec![3i32, 3, 1, 3]),
        (5, 32, 2, 8, 5, vec![3i32, 0, 3]),
        (1, 256, 3, 9, 9, vec![1i32, 1, 1]),
    ]
    .iter()
    .cloned()
    .enumerate()
    {
        let src: Vec<u8> = hashed_bytes(slots * n_kv * head_dim, 0x2ba1 + i as u32)
            .into_iter()
            .map(|b| if b & 0x7f == 0x7f { 0x31 } else { b })
            .collect();
        let scales: Vec<f32> = (0..slots * n_kv)
            .map(|j| 0.0011 + (j as f32) * 0.0041)
            .collect();

        let cu = cuda_dequantize_paged(
            &stream, &src, &scales, &table, len, n_kv, head_dim, block_size,
        );
        let mut out = vec![0u16; len * n_kv * head_dim];
        kv_fp8_paged::dequantize_kv_fp8_paged(
            wg, &src, &scales, &table, &mut out, len, n_kv, head_dim, block_size,
        )
        .expect("wgpu dequantize_kv_fp8_paged shared pages");
        let mut cpu = vec![0u16; len * n_kv * head_dim];
        kv_fp8_paged::cpu_dequantize_kv_fp8_paged(
            &src, &scales, &table, &mut cpu, len, n_kv, head_dim, block_size,
        );

        let diff = cu.iter().zip(out.iter()).filter(|(a, b)| a != b).count();
        let cpu_diff = cu.iter().zip(cpu.iter()).filter(|(a, b)| a != b).count();
        eprintln!(
            "shared-page dequantize case{i} len{len} kv{n_kv} d{head_dim} bs{block_size} table{table:?}: {diff}/{} bf16 words differ (cpu oracle {cpu_diff})",
            cu.len()
        );
        let repeated = table.iter().collect::<std::collections::HashSet<_>>().len() < table.len();
        assert!(repeated, "shared-page case{i}: table must repeat a page");
        assert_eq!(
            cpu_diff, 0,
            "shared-page case{i}: cpu oracle must be bit-exact"
        );
        assert_eq!(diff, 0, "shared-page case{i}: must be bit-exact");
    }
}

#[test]
fn copy_kv_block_fp8_cross_buffer_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("copy_kv_block_fp8_cross_buffer_cuda_vs_wgpu") else {
        return;
    };
    for (i, (n_kv, head_dim, block_size, src_blocks, dst_blocks, src_block, dst_block)) in [
        (3usize, 128usize, 4usize, 5usize, 3usize, 4usize, 0usize),
        (2, 64, 1, 6, 6, 5, 1),
        (4, 256, 3, 4, 7, 1, 6),
        (5, 32, 2, 3, 3, 2, 0),
        (1, 512, 5, 3, 4, 2, 2),
        (2, 16, 3, 4, 5, 3, 4),
        (3, 4, 1, 4, 4, 3, 0),
    ]
    .iter()
    .copied()
    .enumerate()
    {
        let src_slots = src_blocks * block_size;
        let dst_slots = dst_blocks * block_size;
        let src_fp8 = hashed_bytes(src_slots * n_kv * head_dim, 0x77a1 + i as u32);
        let dst_fp8 = hashed_bytes(dst_slots * n_kv * head_dim, 0xc41d + i as u32);
        let src_scales: Vec<f32> = (0..src_slots * n_kv)
            .map(|j| 1.5 + j as f32 * 0.25)
            .collect();
        let dst_scales: Vec<f32> = (0..dst_slots * n_kv)
            .map(|j| -3.0 - j as f32 * 0.125)
            .collect();

        let cu = cuda_copy_block_cross(
            &stream,
            &src_fp8,
            &src_scales,
            &dst_fp8,
            &dst_scales,
            src_block,
            dst_block,
            block_size,
            n_kv,
            head_dim,
        );

        let mut wg_fp8 = dst_fp8.clone();
        let mut wg_scales = dst_scales.clone();
        kv_fp8_paged::copy_kv_block_fp8_into(
            wg,
            &src_fp8,
            &src_scales,
            &mut wg_fp8,
            &mut wg_scales,
            src_block,
            dst_block,
            block_size,
            n_kv,
            head_dim,
        )
        .expect("wgpu copy_kv_block_fp8_into");

        let mut cpu_fp8 = dst_fp8.clone();
        let mut cpu_scales = dst_scales.clone();
        kv_fp8_paged::cpu_copy_kv_block_fp8_into(
            &src_fp8,
            &src_scales,
            &mut cpu_fp8,
            &mut cpu_scales,
            src_block,
            dst_block,
            block_size,
            n_kv,
            head_dim,
        );

        let label = format!(
            "cross copy case{i} kv{n_kv} d{head_dim} bs{block_size} {src_block}->{dst_block}"
        );
        compare_quant(
            &label,
            &cu,
            &QuantResult {
                fp8: wg_fp8,
                scales: wg_scales,
            },
        );
        compare_quant(
            &format!("{label} cpu-oracle"),
            &cu,
            &QuantResult {
                fp8: cpu_fp8,
                scales: cpu_scales,
            },
        );

        if src_block == dst_block {
            assert_eq!(
                cu.fp8, dst_fp8,
                "cross copy case{i}: cuda treats src_block == dst_block as a no-op even across buffers"
            );
            assert_eq!(
                cu.scales, dst_scales,
                "cross copy case{i}: scales untouched"
            );
        } else {
            let moved = cu
                .fp8
                .iter()
                .zip(dst_fp8.iter())
                .filter(|(a, b)| a != b)
                .count();
            let moved_scales = cu
                .scales
                .iter()
                .zip(dst_scales.iter())
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!(
                "cross copy case{i}: cuda moved {moved}/{} fp8 bytes and {moved_scales}/{} scales",
                dst_slots * n_kv * head_dim,
                dst_slots * n_kv
            );
            assert!(moved > 0, "cross copy case{i}: cuda wrote no fp8 bytes");
            assert_eq!(
                moved_scales,
                block_size * n_kv,
                "cross copy case{i}: exactly one block of scales must be rewritten"
            );
            let block_bytes = block_size * n_kv * head_dim;
            assert!(
                moved >= block_bytes * 3 / 4,
                "cross copy case{i}: only {moved} of {block_bytes} destination bytes changed"
            );
            for slot in 0..dst_slots {
                if slot / block_size == dst_block {
                    continue;
                }
                let base = slot * n_kv * head_dim;
                assert_eq!(
                    &cu.fp8[base..base + n_kv * head_dim],
                    &dst_fp8[base..base + n_kv * head_dim],
                    "cross copy case{i}: slot {slot} outside the destination block was modified"
                );
            }
        }
    }
}

#[test]
fn paged_wide_head_dim_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("paged_wide_head_dim_cuda_vs_wgpu") else {
        return;
    };
    for (i, (n_tokens, n_kv, head_dim, block_size, table, slots, start)) in [
        (1usize, 2usize, 96usize, 4usize, vec![2i32], 12usize, 0i32),
        (1, 3, 128, 4, vec![1i32, 0], 8, 3),
        (5, 2, 320, 3, vec![2i32, 0, 1], 9, 1),
        (6, 1, 640, 4, vec![3i32, 0, 2], 16, 2),
        (3, 2, 1024, 2, vec![2i32, 0], 6, 1),
        (7, 1, 96, 5, vec![1i32, 3, 0], 20, 4),
    ]
    .iter()
    .cloned()
    .enumerate()
    {
        let c = Case {
            n_tokens,
            n_kv,
            head_dim,
            block_size,
            table,
            slots,
            start,
        };
        let x = sample_bf16(n_tokens * n_kv * head_dim, 41 + i as u32);
        run_quantize_case(
            &stream,
            wg,
            &format!(
                "wide quantize case{i} t{n_tokens} kv{n_kv} d{head_dim} bs{block_size} start{start}"
            ),
            &c,
            &x,
        );

        let src: Vec<u8> = hashed_bytes(slots * n_kv * head_dim, 0x9e11 + i as u32)
            .into_iter()
            .map(|b| if b & 0x7f == 0x7f { 0x44 } else { b })
            .collect();
        let scales: Vec<f32> = (0..slots * n_kv)
            .map(|j| 0.0007 + (j as f32) * 0.0029)
            .collect();
        let cu = cuda_dequantize_paged(
            &stream, &src, &scales, &c.table, n_tokens, n_kv, head_dim, block_size,
        );
        let mut out = vec![0u16; n_tokens * n_kv * head_dim];
        kv_fp8_paged::dequantize_kv_fp8_paged(
            wg, &src, &scales, &c.table, &mut out, n_tokens, n_kv, head_dim, block_size,
        )
        .expect("wgpu dequantize_kv_fp8_paged wide");
        let mut cpu = vec![0u16; n_tokens * n_kv * head_dim];
        kv_fp8_paged::cpu_dequantize_kv_fp8_paged(
            &src, &scales, &c.table, &mut cpu, n_tokens, n_kv, head_dim, block_size,
        );
        let diff = cu.iter().zip(out.iter()).filter(|(a, b)| a != b).count();
        let cpu_diff = cu.iter().zip(cpu.iter()).filter(|(a, b)| a != b).count();
        eprintln!(
            "wide dequantize case{i} t{n_tokens} kv{n_kv} d{head_dim} bs{block_size}: {diff}/{} bf16 words differ (cpu oracle {cpu_diff})",
            cu.len()
        );
        assert_eq!(
            cpu_diff, 0,
            "wide dequantize case{i}: cpu oracle must be bit-exact"
        );
        assert_eq!(diff, 0, "wide dequantize case{i}: must be bit-exact");

        let mut k = src.clone();
        let mut v = src.clone();
        let mut ks = scales.clone();
        let mut vs = scales.clone();
        let blocks = slots / block_size;
        if blocks >= 2 {
            let (sb, db) = (blocks - 1, 0usize);
            let cu_copy =
                cuda_copy_block_inplace(&stream, &src, &scales, sb, db, block_size, n_kv, head_dim);
            kv_fp8_paged::copy_kv_block_fp8(
                wg, &mut k, &mut v, &mut ks, &mut vs, sb, db, block_size, n_kv, head_dim,
            )
            .expect("wgpu copy_kv_block_fp8 wide");
            compare_quant(
                &format!("wide copy case{i} d{head_dim} bs{block_size} {sb}->{db}"),
                &cu_copy,
                &QuantResult { fp8: k, scales: ks },
            );
        }
    }
}

#[test]
fn paged_host_guards_reject_unmappable_tables() {
    let Some((_stream, wg)) = backends("paged_host_guards_reject_unmappable_tables") else {
        return;
    };
    let (n_tokens, n_kv, head_dim, block_size, slots) = (4usize, 2usize, 64usize, 2usize, 8usize);
    let x = sample_bf16(n_tokens * n_kv * head_dim, 5);
    let mut fp8 = vec![0u8; slots * n_kv * head_dim];
    let mut scales = vec![0f32; slots * n_kv];

    assert!(
        kv_fp8_paged::quantize_kv_fp8_paged(
            wg,
            &x,
            &mut fp8,
            &mut scales,
            &[0i32],
            &[0],
            n_tokens,
            n_kv,
            head_dim,
            block_size,
        )
        .is_err(),
        "a block table shorter than the sequence must be rejected"
    );
    assert!(
        kv_fp8_paged::quantize_kv_fp8_paged(
            wg,
            &x,
            &mut fp8,
            &mut scales,
            &[-1i32, 0],
            &[0],
            n_tokens,
            n_kv,
            head_dim,
            block_size,
        )
        .is_err(),
        "a negative page index must be rejected"
    );
    assert!(
        kv_fp8_paged::quantize_kv_fp8_paged(
            wg,
            &x,
            &mut fp8,
            &mut scales,
            &[0i32, 7],
            &[0],
            n_tokens,
            n_kv,
            head_dim,
            block_size,
        )
        .is_err(),
        "a page index past the end of the cache must be rejected"
    );
    assert!(
        kv_fp8_paged::quantize_kv_fp8_paged(
            wg,
            &x,
            &mut fp8,
            &mut scales,
            &[0i32, 1],
            &[-3],
            n_tokens,
            n_kv,
            head_dim,
            block_size,
        )
        .is_err(),
        "a negative start must be rejected"
    );

    let mut out = vec![0u16; n_tokens * n_kv * head_dim];
    assert!(
        kv_fp8_paged::dequantize_kv_fp8_paged(
            wg,
            &fp8,
            &scales,
            &[0i32],
            &mut out,
            n_tokens,
            n_kv,
            head_dim,
            block_size,
        )
        .is_err(),
        "dequantize must reject a block table shorter than the sequence"
    );

    let mut v_fp8 = fp8.clone();
    let mut v_scales = scales.clone();
    assert!(
        kv_fp8_paged::copy_kv_block_fp8(
            wg,
            &mut fp8,
            &mut v_fp8,
            &mut scales,
            &mut v_scales,
            0,
            9,
            block_size,
            n_kv,
            head_dim,
        )
        .is_err(),
        "a destination block past the end of the cache must be rejected"
    );

    assert!(
        kv_fp8_paged::quantize_kv_fp8_paged(
            wg,
            &x,
            &mut fp8,
            &mut scales,
            &[0i32, 1],
            &[0],
            n_tokens,
            n_kv,
            head_dim,
            block_size,
        )
        .is_ok(),
        "the same call with a well-formed table must succeed, so the guards above are not vacuous"
    );
}

const BINADE_SHIFTS: [u32; 6] = [0, 1, 2, 3, 1, 2];

fn bf16_binade_down(bits: u16, down: u32) -> u16 {
    let sign = bits & 0x8000;
    let e = u32::from((bits >> 7) & 0xff);
    let m = bits & 0x7f;
    assert!(e > down, "bf16 exponent {e} cannot be lowered by {down}");
    sign | (((e - down) as u16) << 7) | m
}

fn amax_targets_bf16(n_tokens: usize, n_kv: usize, head_dim: usize, targets: &[u16]) -> Vec<u16> {
    let mut x = vec![0u16; n_tokens * n_kv * head_dim];
    for p in 0..n_tokens * n_kv {
        let t = targets[p % targets.len()];
        let base = p * head_dim;
        for d in 0..head_dim {
            let down = if d == 0 {
                0
            } else {
                BINADE_SHIFTS[d % BINADE_SHIFTS.len()]
            };
            let mut v = bf16_binade_down(t, down);
            if d % 2 == 1 {
                v |= 0x8000;
            }
            x[base + d] = v;
        }
    }
    x
}

fn count_subnormal_scales(c: &Case, x: &[u16]) -> (usize, usize) {
    let mut fp8 = vec![0u8; c.slots * c.n_kv * c.head_dim];
    let mut scales = vec![0f32; c.slots * c.n_kv];
    kv_fp8_paged::cpu_quantize_kv_fp8_paged(
        x,
        &mut fp8,
        &mut scales,
        &c.table,
        c.start as usize,
        c.n_tokens,
        c.n_kv,
        c.head_dim,
        c.block_size,
    );
    let sub = scales
        .iter()
        .filter(|s| **s != 0.0 && s.abs() < f32::MIN_POSITIVE)
        .count();
    let norm = scales
        .iter()
        .filter(|s| s.abs() >= f32::MIN_POSITIVE)
        .count();
    (sub, norm)
}

#[test]
fn quantize_kv_fp8_paged_scale_mantissas_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("quantize_kv_fp8_paged_scale_mantissas_cuda_vs_wgpu") else {
        return;
    };
    let c = magnitude_case();
    let pairs = c.n_tokens * c.n_kv;
    let mut targets: Vec<u16> = Vec::new();
    for e in 5u16..=20 {
        for m in 0u16..128 {
            targets.push((e << 7) | m);
        }
    }
    let mut subnormal = 0usize;
    let mut normal = 0usize;
    let chunks: Vec<Vec<u16>> = targets
        .chunks(pairs)
        .map(|chunk| {
            let mut t = chunk.to_vec();
            while t.len() < pairs {
                t.push(chunk[chunk.len() - 1]);
            }
            t
        })
        .collect();
    for (i, t) in chunks.iter().enumerate() {
        let x = amax_targets_bf16(c.n_tokens, c.n_kv, c.head_dim, t);
        let (sub, norm) = count_subnormal_scales(&c, &x);
        subnormal += sub;
        normal += norm;
        run_quantize_case(
            &stream,
            wg,
            &format!(
                "scale mantissa chunk {i} amax bf16 {:#06x}..{:#06x}",
                t[0],
                t[t.len() - 1]
            ),
            &c,
            &x,
        );
    }
    assert!(
        subnormal >= 128,
        "the sweep must actually land in the subnormal-scale region; saw only {subnormal} \
         subnormal scales"
    );
    assert!(
        normal >= 128,
        "the sweep must also cover normal scales; saw only {normal}"
    );
    eprintln!(
        "scale mantissa sweep: {} amax values over {} launches, {subnormal} subnormal + {normal} \
         normal scales, all bit-exact",
        targets.len(),
        chunks.len()
    );
}

#[test]
fn paged_host_guards_reject_unpackable_head_dims() {
    let Some((_stream, wg)) = backends("paged_host_guards_reject_unpackable_head_dims") else {
        return;
    };
    let (n_tokens, n_kv, block_size, slots) = (2usize, 2usize, 2usize, 4usize);
    let table = [0i32, 1];

    for head_dim in [6usize, 7, 9] {
        let x = sample_bf16(n_tokens * n_kv * head_dim, 11);
        let mut fp8 = vec![0u8; slots * n_kv * head_dim];
        let mut scales = vec![0f32; slots * n_kv];
        assert!(
            kv_fp8_paged::quantize_kv_fp8_paged(
                wg,
                &x,
                &mut fp8,
                &mut scales,
                &table,
                &[0],
                n_tokens,
                n_kv,
                head_dim,
                block_size,
            )
            .is_err(),
            "quantize must refuse head_dim {head_dim} rather than write partial u32 words"
        );
        assert!(
            fp8.iter().all(|b| *b == 0) && scales.iter().all(|s| *s == 0.0),
            "a refused quantize must not have touched the destination"
        );

        let mut k = fp8.clone();
        let mut v = fp8.clone();
        let mut ks = scales.clone();
        let mut vs = scales.clone();
        assert!(
            kv_fp8_paged::copy_kv_block_fp8(
                wg, &mut k, &mut v, &mut ks, &mut vs, 0, 1, block_size, n_kv, head_dim,
            )
            .is_err(),
            "copy must refuse head_dim {head_dim}"
        );
        assert!(
            kv_fp8_paged::copy_kv_block_fp8_into(
                wg, &fp8, &scales, &mut k, &mut ks, 0, 1, block_size, n_kv, head_dim,
            )
            .is_err(),
            "cross-buffer copy must refuse head_dim {head_dim}"
        );
    }

    for head_dim in [7usize, 9] {
        let fp8 = vec![0u8; slots * n_kv * head_dim];
        let scales = vec![1f32; slots * n_kv];
        let mut out = vec![0u16; n_tokens * n_kv * head_dim];
        assert!(
            kv_fp8_paged::dequantize_kv_fp8_paged(
                wg, &fp8, &scales, &table, &mut out, n_tokens, n_kv, head_dim, block_size,
            )
            .is_err(),
            "dequantize must refuse odd head_dim {head_dim}"
        );
    }

    let head_dim = 6usize;
    let fp8 = vec![0u8; slots * n_kv * head_dim];
    let scales = vec![1f32; slots * n_kv];
    let mut out = vec![0u16; n_tokens * n_kv * head_dim];
    assert!(
        kv_fp8_paged::dequantize_kv_fp8_paged(
            wg, &fp8, &scales, &table, &mut out, n_tokens, n_kv, head_dim, block_size,
        )
        .is_ok(),
        "dequantize accepts any even head_dim, so the guard above is not vacuous"
    );
    let x = sample_bf16(n_tokens * n_kv * 8, 11);
    let mut ok_fp8 = vec![0u8; slots * n_kv * 8];
    let mut ok_scales = vec![0f32; slots * n_kv];
    assert!(
        kv_fp8_paged::quantize_kv_fp8_paged(
            wg,
            &x,
            &mut ok_fp8,
            &mut ok_scales,
            &table,
            &[0],
            n_tokens,
            n_kv,
            8,
            block_size,
        )
        .is_ok(),
        "quantize accepts head_dim 8, so the guards above are not vacuous"
    );
}

fn extreme_scale_set(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let e = if n > 1 {
                -140 + (i as i32 * 260) / (n as i32 - 1)
            } else {
                -140
            };
            let m = (i * 3) % 8;
            pow2(e) * (1.0 + (m as f32) / 8.0)
        })
        .collect()
}

#[test]
fn dequantize_kv_fp8_paged_extreme_scales_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("dequantize_kv_fp8_paged_extreme_scales_cuda_vs_wgpu") else {
        return;
    };
    let mut all_subnormal = 0usize;
    let mut all_infinite = 0usize;
    for (i, c) in quantize_cases().iter().enumerate() {
        let len = c.n_tokens;
        let src: Vec<u8> = (0..c.slots * c.n_kv * c.head_dim)
            .map(|j| {
                let b = ((j * 11 + i * 5) % 256) as u8;
                if b & 0x7f == 0x7f {
                    0x7e
                } else {
                    b
                }
            })
            .collect();
        let scales = extreme_scale_set(c.slots * c.n_kv);
        let subnormal = scales
            .iter()
            .filter(|s| **s != 0.0 && s.abs() < f32::MIN_POSITIVE)
            .count();

        let cu = cuda_dequantize_paged(
            &stream,
            &src,
            &scales,
            &c.table,
            len,
            c.n_kv,
            c.head_dim,
            c.block_size,
        );
        let mut out = vec![0u16; len * c.n_kv * c.head_dim];
        kv_fp8_paged::dequantize_kv_fp8_paged(
            wg,
            &src,
            &scales,
            &c.table,
            &mut out,
            len,
            c.n_kv,
            c.head_dim,
            c.block_size,
        )
        .expect("wgpu dequantize_kv_fp8_paged");
        let mut cpu = vec![0u16; len * c.n_kv * c.head_dim];
        kv_fp8_paged::cpu_dequantize_kv_fp8_paged(
            &src,
            &scales,
            &c.table,
            &mut cpu,
            len,
            c.n_kv,
            c.head_dim,
            c.block_size,
        );
        let diff = cu.iter().zip(out.iter()).filter(|(a, b)| a != b).count();
        let cpu_diff = cu.iter().zip(cpu.iter()).filter(|(a, b)| a != b).count();
        let nonzero = cu.iter().filter(|w| **w & 0x7fff != 0).count();
        let infinite = cu.iter().filter(|w| **w & 0x7fff == 0x7f80).count();
        eprintln!(
            "extreme-scale dequantize case{i} len{len} kv{} d{} bs{}: {diff}/{} bf16 words differ \
             (cpu oracle {cpu_diff}), {subnormal} subnormal scales, {nonzero} nonzero and \
             {infinite} infinite outputs",
            c.n_kv,
            c.head_dim,
            c.block_size,
            cu.len()
        );
        assert!(
            subnormal > 0,
            "case{i}: the scale set must include subnormals"
        );
        assert!(nonzero > 0, "case{i}: cuda produced an all-zero output");
        assert_eq!(
            cpu_diff, 0,
            "extreme-scale dequantize case{i}: cpu oracle must be bit-exact"
        );
        assert_eq!(
            diff, 0,
            "extreme-scale dequantize case{i}: must be bit-exact"
        );
        all_subnormal += subnormal;
        all_infinite += infinite;
    }
    assert!(
        all_subnormal > 0,
        "the scale set must exercise subnormal scales"
    );
    assert!(
        all_infinite > 0,
        "the scale set must also drive the product past f32 range"
    );
    eprintln!(
        "extreme-scale dequantize: {all_subnormal} subnormal scales and {all_infinite} overflowed \
         outputs across all cases, all bit-exact"
    );
}

#[test]
fn paged_round_trip_through_subnormal_scales_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("paged_round_trip_through_subnormal_scales_cuda_vs_wgpu")
    else {
        return;
    };
    let (n_tokens, n_kv, head_dim, block_size) = (7usize, 3usize, 128usize, 3usize);
    let table: Vec<i32> = vec![2, 0, 1];
    let slots = 9usize;
    let base_fp8 = vec![0u8; slots * n_kv * head_dim];
    let base_scales = vec![0f32; slots * n_kv];

    for exp in [NORMAL_SCALE_MIN_EXP, -120, -119, -118] {
        let x = magnitude_bf16(n_tokens, n_kv, head_dim, exp);
        let cu = cuda_quantize_paged(
            &stream,
            &x,
            &base_fp8,
            &base_scales,
            &table,
            0,
            n_tokens,
            n_kv,
            head_dim,
            block_size,
        );
        let subnormal = cu
            .scales
            .iter()
            .filter(|s| **s != 0.0 && s.abs() < f32::MIN_POSITIVE)
            .count();
        assert!(
            subnormal > 0,
            "exp {exp}: cuda must actually produce subnormal scales here"
        );
        let cu_back = cuda_dequantize_paged(
            &stream, &cu.fp8, &cu.scales, &table, n_tokens, n_kv, head_dim, block_size,
        );

        let mut fp8 = base_fp8.clone();
        let mut scales = base_scales.clone();
        kv_fp8_paged::quantize_kv_fp8_paged(
            wg,
            &x,
            &mut fp8,
            &mut scales,
            &table,
            &[0],
            n_tokens,
            n_kv,
            head_dim,
            block_size,
        )
        .expect("wgpu quantize");
        let mut wg_back = vec![0u16; x.len()];
        kv_fp8_paged::dequantize_kv_fp8_paged(
            wg,
            &fp8,
            &scales,
            &table,
            &mut wg_back,
            n_tokens,
            n_kv,
            head_dim,
            block_size,
        )
        .expect("wgpu dequantize");

        let scale_diff = cu
            .scales
            .iter()
            .zip(scales.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let diff = cu_back
            .iter()
            .zip(wg_back.iter())
            .filter(|(a, b)| a != b)
            .count();
        let nonzero = cu_back.iter().filter(|w| **w & 0x7fff != 0).count();
        eprintln!(
            "subnormal-scale round trip exp {exp}: {subnormal} subnormal scales, \
             {scale_diff}/{} scales differ, {diff}/{} bf16 words differ, {nonzero} nonzero",
            cu.scales.len(),
            cu_back.len()
        );
        assert!(
            nonzero > 0,
            "exp {exp}: cuda round trip produced only zeros"
        );
        assert_eq!(scale_diff, 0, "exp {exp}: scales must be bit-exact");
        assert_eq!(diff, 0, "exp {exp}: round trip must be bit-exact");
    }
}

fn wide_dynamic_range_bf16(n_tokens: usize, n_kv: usize, head_dim: usize, top: i32) -> Vec<u16> {
    let steps: [i32; 8] = [0, -1, -60, -120, -200, -240, -250, -253];
    let mut x = vec![0u16; n_tokens * n_kv * head_dim];
    for p in 0..n_tokens * n_kv {
        let base = p * head_dim;
        for d in 0..head_dim {
            let e = if d == 0 {
                top
            } else {
                (top + steps[(d + p) % steps.len()]).max(-126)
            };
            let v = pow2(e) * if d % 3 == 1 { -1.0 } else { 1.0 };
            x[base + d] = bf16::from_f32(v).to_bits();
        }
    }
    x
}

#[test]
fn quantize_kv_fp8_paged_wide_dynamic_range_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("quantize_kv_fp8_paged_wide_dynamic_range_cuda_vs_wgpu")
    else {
        return;
    };
    let c = magnitude_case();
    for top in [127i32, 100, 0, -60, NORMAL_SCALE_MIN_EXP] {
        let x = wide_dynamic_range_bf16(c.n_tokens, c.n_kv, c.head_dim, top);
        let mut cpu_fp8 = vec![0u8; c.slots * c.n_kv * c.head_dim];
        let mut cpu_scales = vec![0f32; c.slots * c.n_kv];
        kv_fp8_paged::cpu_quantize_kv_fp8_paged(
            &x,
            &mut cpu_fp8,
            &mut cpu_scales,
            &c.table,
            c.start as usize,
            c.n_tokens,
            c.n_kv,
            c.head_dim,
            c.block_size,
        );
        let flushed = cpu_fp8.iter().filter(|b| **b & 0x7f == 0).count();
        assert!(
            flushed > 0,
            "top {top}: the pattern must contain values that underflow to fp8 zero"
        );
        run_quantize_case(
            &stream,
            wg,
            &format!("wide dynamic range top 2^{top} ({flushed} fp8 zeros)"),
            &c,
            &x,
        );
    }
}
