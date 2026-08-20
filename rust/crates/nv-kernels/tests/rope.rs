#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use nv_kernels::cuda;

fn build_rope_tables(max_pos: usize, half_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0f32; max_pos * half_dim];
    let mut sin = vec![0f32; max_pos * half_dim];
    for p in 0..max_pos {
        for i in 0..half_dim {
            let theta = (p as f32) / base.powf((i as f32 * 2.0) / (half_dim as f32 * 2.0));
            cos[p * half_dim + i] = theta.cos();
            sin[p * half_dim + i] = theta.sin();
        }
    }
    (cos, sin)
}

fn cpu_rope_apply(
    q: &mut [f32],
    positions: &[i32],
    cos_tbl: &[f32],
    sin_tbl: &[f32],
    n_heads: usize,
    head_dim: usize,
) {
    let half = head_dim / 2;
    for (t, pos) in positions.iter().enumerate() {
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            for i in 0..half {
                let c = cos_tbl[(*pos as usize) * half + i];
                let s = sin_tbl[(*pos as usize) * half + i];
                let a = q[base + i];
                let b = q[base + i + half];
                q[base + i] = a * c - b * s;
                q[base + i + half] = a * s + b * c;
            }
        }
    }
}

#[test]
fn rope_f32_matches_cpu_reference() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "rope: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("rope: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    let batch = 8usize;
    let n_heads = 16usize;
    let n_kv_heads = 4usize;
    let head_dim = 64usize;
    let half = head_dim / 2;
    let max_pos = 32usize;

    let (cos_tbl, sin_tbl) = build_rope_tables(max_pos, half, 10_000.0);

    let mut q_host = Vec::with_capacity(batch * n_heads * head_dim);
    let mut k_host = Vec::with_capacity(batch * n_kv_heads * head_dim);
    for i in 0..(batch * n_heads * head_dim) {
        q_host.push((i as f32 * 0.0013).sin());
    }
    for i in 0..(batch * n_kv_heads * head_dim) {
        k_host.push((i as f32 * 0.0017).cos());
    }
    let positions: Vec<i32> = (0..batch).map(|i| (i as i32) % (max_pos as i32)).collect();

    let mut q_expect = q_host.clone();
    let mut k_expect = k_host.clone();
    cpu_rope_apply(
        &mut q_expect,
        &positions,
        &cos_tbl,
        &sin_tbl,
        n_heads,
        head_dim,
    );
    cpu_rope_apply(
        &mut k_expect,
        &positions,
        &cos_tbl,
        &sin_tbl,
        n_kv_heads,
        head_dim,
    );

    #[allow(deprecated)]
    let mut q_dev: CudaSlice<f32> = stream.memcpy_stod(&q_host).unwrap();
    #[allow(deprecated)]
    let mut k_dev: CudaSlice<f32> = stream.memcpy_stod(&k_host).unwrap();
    #[allow(deprecated)]
    let cos_dev: CudaSlice<f32> = stream.memcpy_stod(&cos_tbl).unwrap();
    #[allow(deprecated)]
    let sin_dev: CudaSlice<f32> = stream.memcpy_stod(&sin_tbl).unwrap();
    #[allow(deprecated)]
    let pos_dev: CudaSlice<i32> = stream.memcpy_stod(&positions).unwrap();

    let rc = {
        let (pq, _g1) = q_dev.device_ptr_mut(&stream);
        let (pk, _g2) = k_dev.device_ptr_mut(&stream);
        let (pc, _g3) = cos_dev.device_ptr(&stream);
        let (ps, _g4) = sin_dev.device_ptr(&stream);
        let (pp, _g5) = pos_dev.device_ptr(&stream);
        unsafe {
            cuda::rope_f32(
                stream.cu_stream() as *mut _,
                pq as *mut f32,
                pk as *mut f32,
                pc as *const f32,
                ps as *const f32,
                pp as *const i32,
                batch,
                n_heads,
                n_kv_heads,
                head_dim,
            )
        }
    };
    assert_eq!(rc, 0, "kernel launch returned {rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let q_got = stream.memcpy_dtov(&q_dev).unwrap();
    #[allow(deprecated)]
    let k_got = stream.memcpy_dtov(&k_dev).unwrap();

    let mut max_q = 0f32;
    let mut max_k = 0f32;
    for (g, e) in q_got.iter().zip(q_expect.iter()) {
        max_q = max_q.max((g - e).abs());
    }
    for (g, e) in k_got.iter().zip(k_expect.iter()) {
        max_k = max_k.max((g - e).abs());
    }
    assert!(max_q < 1e-5, "rope q drift {max_q}");
    assert!(max_k < 1e-5, "rope k drift {max_k}");
}
