#![cfg(feature = "cuda")]

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_quant::nvfp4::{supports_nvfp4, swizzle_scales, Nvfp4Tensor, BLOCK_SIZE};
use std::ffi::c_void;
use std::sync::Arc;
mod common;
use common::rel_rms_bf16 as rel_rms;

fn swizzled_scale_bytes(rows: usize, cols: usize) -> usize {
    let k_blocks = cols / BLOCK_SIZE;
    let m_tiles = (rows + 127) / 128;
    let k_tiles = (k_blocks + 3) / 4;
    m_tiles * 128 * k_tiles * 4
}

struct Inputs {
    a_packed: CudaSlice<u8>,
    a_scales: CudaSlice<u8>,
    b_packed: CudaSlice<u8>,
    b_scales: CudaSlice<u8>,
    alphas: CudaSlice<f32>,
    expert_offsets: CudaSlice<i32>,
    sf_offsets: CudaSlice<i32>,
    problem_sizes: CudaSlice<i32>,
    active_ids: CudaSlice<i32>,
    e_total: usize,
    a: usize,
    m_per_expert: usize,
    n: usize,
    k: usize,
}

fn build(
    stream: &Arc<CudaStream>,
    e_total: usize,
    active_ids_host: &[i32],
    m_per_expert: usize,
    n: usize,
    k: usize,
) -> Inputs {
    let a = active_ids_host.len();

    let mut a_rows: Vec<Vec<f32>> = Vec::with_capacity(a * m_per_expert);
    for i in 0..a {
        for r in 0..m_per_expert {
            let row: Vec<f32> = (0..k)
                .map(|j| (((i * 137 + r * 31 + j) as f32 * 0.07).sin()))
                .collect();
            a_rows.push(row);
        }
    }
    let a_q = Nvfp4Tensor::quantize_rows(&a_rows);
    let a_scales_sw = swizzle_scales(&a_q.scales, a * m_per_expert, k / BLOCK_SIZE);

    let mut b_packed_concat = Vec::with_capacity(e_total * n * k / 2);
    let mut b_scales_concat = Vec::with_capacity(e_total * swizzled_scale_bytes(n, k));
    let mut alphas_host = Vec::with_capacity(e_total);
    for e in 0..e_total {
        let b_rows: Vec<Vec<f32>> = (0..n)
            .map(|j| {
                (0..k)
                    .map(|p| (((e * 211 + j * 13 + p) as f32 * 0.05).cos()))
                    .collect()
            })
            .collect();
        let b_q = Nvfp4Tensor::quantize_rows(&b_rows);
        let b_sf = swizzle_scales(&b_q.scales, n, k / BLOCK_SIZE);
        b_packed_concat.extend_from_slice(&b_q.data);
        b_scales_concat.extend_from_slice(&b_sf);
        alphas_host.push(1.0f32);
    }

    #[allow(deprecated)]
    let a_packed = stream.memcpy_stod(&a_q.data).unwrap();
    #[allow(deprecated)]
    let a_scales = stream.memcpy_stod(&a_scales_sw).unwrap();
    #[allow(deprecated)]
    let b_packed = stream.memcpy_stod(&b_packed_concat).unwrap();
    #[allow(deprecated)]
    let b_scales = stream.memcpy_stod(&b_scales_concat).unwrap();
    #[allow(deprecated)]
    let alphas = stream.memcpy_stod(&alphas_host).unwrap();

    let expert_offsets_host: Vec<i32> = (0..a).map(|i| (i * m_per_expert) as i32).collect();
    let sf_offsets_host = expert_offsets_host.clone();
    let mut problem_sizes_host = Vec::with_capacity(a * 3);
    for _ in 0..a {
        problem_sizes_host.extend_from_slice(&[m_per_expert as i32, n as i32, k as i32]);
    }
    #[allow(deprecated)]
    let expert_offsets = stream.memcpy_stod(&expert_offsets_host).unwrap();
    #[allow(deprecated)]
    let sf_offsets = stream.memcpy_stod(&sf_offsets_host).unwrap();
    #[allow(deprecated)]
    let problem_sizes = stream.memcpy_stod(&problem_sizes_host).unwrap();
    #[allow(deprecated)]
    let active_ids = stream.memcpy_stod(active_ids_host).unwrap();

    Inputs {
        a_packed,
        a_scales,
        b_packed,
        b_scales,
        alphas,
        expert_offsets,
        sf_offsets,
        problem_sizes,
        active_ids,
        e_total,
        a,
        m_per_expert,
        n,
        k,
    }
}

fn run_grouped(stream: &Arc<CudaStream>, inp: &Inputs) -> Vec<bf16> {
    let mut d = stream
        .alloc_zeros::<bf16>(inp.a * inp.m_per_expert * inp.n)
        .unwrap();
    let mut meta_scratch = stream.alloc_zeros::<u8>(128 * 1024).unwrap();
    let mut gemm_ws = stream.alloc_zeros::<u8>(64 * 1024 * 1024).unwrap();
    let _needed = {
        let (ap, _g1) = inp.a_packed.device_ptr(stream);
        let (asp, _g2) = inp.a_scales.device_ptr(stream);
        let (bp, _g3) = inp.b_packed.device_ptr(stream);
        let (bsp, _g4) = inp.b_scales.device_ptr(stream);
        let (alp, _g5) = inp.alphas.device_ptr(stream);
        let (dp, _g6) = d.device_ptr_mut(stream);
        let (eo, _g7) = inp.expert_offsets.device_ptr(stream);
        let (sfo, _g8) = inp.sf_offsets.device_ptr(stream);
        let (ps, _g9) = inp.problem_sizes.device_ptr(stream);
        let (aei, _ga) = inp.active_ids.device_ptr(stream);
        let (ms, _gb) = meta_scratch.device_ptr_mut(stream);
        let (ws, _gc) = gemm_ws.device_ptr_mut(stream);
        unsafe {
            nv_kernels::cuda::cutlass_moe_grouped_fp4_gemm_sm120_bf16(
                stream.cu_stream() as *mut c_void,
                ap as *const c_void,
                asp as *const c_void,
                bp as *const c_void,
                bsp as *const c_void,
                alp as *const f32,
                dp as *mut c_void,
                eo as *const i32,
                sfo as *const i32,
                ps as *const i32,
                aei as *const i32,
                inp.n as i32,
                inp.k as i32,
                inp.a as i32,
                inp.k as i64,
                inp.k as i64,
                inp.n as i64,
                ms as *mut c_void,
                128 * 1024,
                ws as *mut c_void,
                64 * 1024 * 1024,
            )
            .expect("grouped FP4 GEMM launch")
        }
    };
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    stream.memcpy_dtov(&d).unwrap()
}

fn run_single_expert(
    stream: &Arc<CudaStream>,
    inp: &Inputs,
    active_slot: usize,
    expert_id: usize,
) -> Vec<bf16> {
    let m = inp.m_per_expert;
    let n = inp.n;
    let k = inp.k;

    let a_row_bytes = k / 2;
    let a_offset_bytes = active_slot * m * a_row_bytes;
    let a_len_bytes = m * a_row_bytes;
    let a_sf_per_tile = swizzled_scale_bytes(m, k);
    let a_sf_offset = active_slot * a_sf_per_tile;
    let b_per_expert = n * a_row_bytes;
    let b_sf_per_expert = swizzled_scale_bytes(n, k);

    #[allow(deprecated)]
    let a_full = stream.memcpy_dtov(&inp.a_packed).unwrap();
    #[allow(deprecated)]
    let a_sf_full = stream.memcpy_dtov(&inp.a_scales).unwrap();
    #[allow(deprecated)]
    let b_full = stream.memcpy_dtov(&inp.b_packed).unwrap();
    #[allow(deprecated)]
    let b_sf_full = stream.memcpy_dtov(&inp.b_scales).unwrap();

    let a_slice = &a_full[a_offset_bytes..a_offset_bytes + a_len_bytes];
    let a_sf_slice = &a_sf_full[a_sf_offset..a_sf_offset + a_sf_per_tile];
    let b_slice = &b_full[expert_id * b_per_expert..(expert_id + 1) * b_per_expert];
    let b_sf_slice = &b_sf_full[expert_id * b_sf_per_expert..(expert_id + 1) * b_sf_per_expert];

    #[allow(deprecated)]
    let a_dev = stream.memcpy_stod(a_slice).unwrap();
    #[allow(deprecated)]
    let a_sf_dev = stream.memcpy_stod(a_sf_slice).unwrap();
    #[allow(deprecated)]
    let b_dev = stream.memcpy_stod(b_slice).unwrap();
    #[allow(deprecated)]
    let b_sf_dev = stream.memcpy_stod(b_sf_slice).unwrap();
    let gsf = stream.memcpy_stod(&[1.0f32]).unwrap();
    let mut d = stream.alloc_zeros::<bf16>(m * n).unwrap();
    let mut ws = stream.alloc_zeros::<u8>(64 * 1024 * 1024).unwrap();

    {
        let (ap, _g1) = a_dev.device_ptr(stream);
        let (asp, _g2) = a_sf_dev.device_ptr(stream);
        let (bp, _g3) = b_dev.device_ptr(stream);
        let (bsp, _g4) = b_sf_dev.device_ptr(stream);
        let (gp, _g5) = gsf.device_ptr(stream);
        let (dp, _g6) = d.device_ptr_mut(stream);
        let (wp, _g7) = ws.device_ptr_mut(stream);
        let _ = unsafe {
            nv_kernels::cuda::cutlass_fp4_gemm_sm120_bf16(
                stream.cu_stream() as *mut c_void,
                ap as *const c_void,
                asp as *const c_void,
                bp as *const c_void,
                bsp as *const c_void,
                gp as *const f32,
                dp as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                wp as *mut c_void,
                64 * 1024 * 1024,
            )
            .expect("single-tile launch")
        };
    }
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    stream.memcpy_dtov(&d).unwrap()
}

fn case(
    stream: &Arc<CudaStream>,
    label: &str,
    e_total: usize,
    active_ids: Vec<i32>,
    n: usize,
    k: usize,
) {
    eprintln!(
        "--- {label}: E_total={e_total} A={} N={n} K={k}",
        active_ids.len()
    );
    let inp = build(stream, e_total, &active_ids, 128, n, k);
    let grouped = run_grouped(stream, &inp);

    let nz = grouped.iter().filter(|x| x.to_f32() != 0.0).count();
    eprintln!(
        "  grouped output non-zero = {nz}/{}",
        inp.a * inp.m_per_expert * inp.n
    );
    assert!(
        nz > (inp.a * inp.m_per_expert * inp.n) / 4,
        "grouped output mostly zero -- kernel did not run"
    );

    let m = inp.m_per_expert;
    let n = inp.n;
    for (i, &eid) in active_ids.iter().enumerate() {
        let ref_out = run_single_expert(stream, &inp, i, eid as usize);
        let lo = i * m * n;
        let got_slice = &grouped[lo..lo + m * n];
        let r = rel_rms(got_slice, &ref_out);
        eprintln!(
            "  active slot {i} -> expert {eid}: rel_rms vs single = {r:.6} (first 4 grouped {:?} single {:?})",
            &got_slice[..4].iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
            &ref_out[..4].iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        );
        assert!(
            r < 1e-3,
            "{label}: slot {i} expert {eid} diverges: rel_rms={r}"
        );
    }
}

#[test]
fn moe_grouped_fp4_multi_expert_identity_A8() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "moe_grouped_fp4_multi_expert: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!(
            "moe_grouped_fp4_multi_expert: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0"
        );
        return;
    };
    let major = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0);
    let minor = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap_or(0);
    if !supports_nvfp4(major) || (major, minor) != (12, 0) {
        eprintln!("skip: requires SM 12.0");
        return;
    }
    let stream = ctx.default_stream();
    case(&stream, "identity A=8", 8, (0..8).collect(), 768, 2048);
}

#[test]
fn moe_grouped_fp4_multi_expert_remapped_A4_of_8() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "moe_grouped_fp4_multi_expert: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!(
            "moe_grouped_fp4_multi_expert: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0"
        );
        return;
    };
    let major = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0);
    let minor = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap_or(0);
    if !supports_nvfp4(major) || (major, minor) != (12, 0) {
        eprintln!("skip: requires SM 12.0");
        return;
    }
    let stream = ctx.default_stream();
    case(&stream, "remapped A=4/8", 8, vec![3, 1, 6, 0], 768, 2048);
}

#[test]
fn moe_grouped_fp4_multi_expert_down_shape() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "moe_grouped_fp4_multi_expert: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!(
            "moe_grouped_fp4_multi_expert: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0"
        );
        return;
    };
    let major = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0);
    let minor = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap_or(0);
    if !supports_nvfp4(major) || (major, minor) != (12, 0) {
        eprintln!("skip: requires SM 12.0");
        return;
    }
    let stream = ctx.default_stream();
    case(&stream, "down N=2048 K=768", 8, vec![2, 5, 7, 3], 2048, 768);
}

#[test]
fn moe_grouped_fp4_qwen3_5_moe_down_exact_dims() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "moe_grouped_fp4_multi_expert: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!(
            "moe_grouped_fp4_multi_expert: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0"
        );
        return;
    };
    let major = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0);
    let minor = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap_or(0);
    if !supports_nvfp4(major) || (major, minor) != (12, 0) {
        eprintln!("skip: requires SM 12.0");
        return;
    }
    let stream = ctx.default_stream();
    case(
        &stream,
        "qwen down E_total=256 A=8 K=512",
        256,
        vec![14, 73, 110, 195, 22, 167, 88, 244],
        2048,
        512,
    );
}

#[test]
fn moe_grouped_fp4_qwen3_5_moe_down_a16() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "moe_grouped_fp4_multi_expert: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!(
            "moe_grouped_fp4_multi_expert: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0"
        );
        return;
    };
    let major = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0);
    let minor = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap_or(0);
    if !supports_nvfp4(major) || (major, minor) != (12, 0) {
        eprintln!("skip: requires SM 12.0");
        return;
    }
    let stream = ctx.default_stream();
    let active_ids: Vec<i32> = (0..16).map(|i| (i * 7 + 3) % 256).collect();
    case(
        &stream,
        "qwen down E_total=256 A=16 K=512",
        256,
        active_ids,
        2048,
        512,
    );
}

#[test]
fn moe_grouped_fp4_qwen3_5_moe_down_a32() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "moe_grouped_fp4_multi_expert: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!(
            "moe_grouped_fp4_multi_expert: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0"
        );
        return;
    };
    let major = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0);
    let minor = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap_or(0);
    if !supports_nvfp4(major) || (major, minor) != (12, 0) {
        eprintln!("skip: requires SM 12.0");
        return;
    }
    let stream = ctx.default_stream();
    let active_ids: Vec<i32> = (0..32).map(|i| (i * 7 + 3) % 256).collect();
    case(
        &stream,
        "qwen down E_total=256 A=32 K=512",
        256,
        active_ids,
        2048,
        512,
    );
}

#[test]
fn moe_grouped_fp4_qwen3_5_moe_down_a40() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "moe_grouped_fp4_multi_expert: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!(
            "moe_grouped_fp4_multi_expert: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0"
        );
        return;
    };
    let major = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0);
    let minor = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap_or(0);
    if !supports_nvfp4(major) || (major, minor) != (12, 0) {
        eprintln!("skip: requires SM 12.0");
        return;
    }
    let stream = ctx.default_stream();
    let active_ids: Vec<i32> = (0..40).map(|i| (i * 7 + 3) % 256).collect();
    case(
        &stream,
        "qwen down E_total=256 A=40 K=512",
        256,
        active_ids,
        2048,
        512,
    );
}

#[test]
fn moe_grouped_fp4_qwen3_5_moe_gate_exact_dims() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "moe_grouped_fp4_multi_expert: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!(
            "moe_grouped_fp4_multi_expert: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0"
        );
        return;
    };
    let major = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0);
    let minor = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .unwrap_or(0);
    if !supports_nvfp4(major) || (major, minor) != (12, 0) {
        eprintln!("skip: requires SM 12.0");
        return;
    }
    let stream = ctx.default_stream();
    case(
        &stream,
        "qwen gate E_total=256 A=8 K=2048",
        256,
        vec![14, 73, 110, 195, 22, 167, 88, 244],
        512,
        2048,
    );
}
