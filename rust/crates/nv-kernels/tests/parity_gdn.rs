#![cfg(all(feature = "cuda", feature = "wgpu"))]

mod common;
use common::backends;
use common::lcg_unit_f32 as lcg;
use common::to_bf16;
mod gdn_host_ref;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::{gdn_gating, gdn_recurrent};
use std::ffi::c_void;
use std::sync::Arc;
use common::lcg_hi32_u32 as rand_bits;

fn ordered(x: f32) -> i64 {
    let b = x.to_bits() as i32;
    if b < 0 {
        (i32::MIN as i64) - (b as i64)
    } else {
        b as i64
    }
}

fn assert_f32_ulp(name: &str, cu: &[f32], wg: &[f32], max_allowed: i64) {
    assert_eq!(cu.len(), wg.len(), "{name}: length");
    let mut diff = 0usize;
    let mut max_ulp = 0i64;
    let mut first = None;
    for (i, (a, b)) in cu.iter().zip(wg.iter()).enumerate() {
        if a.to_bits() == b.to_bits() {
            continue;
        }
        if a.is_nan() && b.is_nan() {
            continue;
        }
        diff += 1;
        let u = (ordered(*a) - ordered(*b)).abs();
        if u > max_ulp {
            max_ulp = u;
        }
        if first.is_none() {
            first = Some((i, *a, *b, u));
        }
    }
    eprintln!(
        "{name}: {diff}/{} f32 lanes differ, max_ulp={max_ulp}",
        cu.len()
    );
    assert!(
        max_ulp <= max_allowed,
        "{name}: max_ulp {max_ulp} > {max_allowed}, first {first:?}"
    );
    if max_allowed == 0 {
        assert_eq!(diff, 0, "{name}: not bit-exact, first {first:?}");
    }
}

fn assert_exercises_subnormals(name: &str, v: &[f32]) {
    let sub = v.iter().filter(|x| x.is_subnormal()).count();
    let zero = v.iter().filter(|x| **x == 0.0).count();
    eprintln!("{name}: {sub}/{} subnormal, {zero} zero", v.len());
    assert!(
        sub > 0,
        "{name}: fixture no longer reaches the gradual-underflow path"
    );
}

fn assert_bf16_exact(name: &str, cu: &[u16], wg: &[u16]) {
    assert_eq!(cu.len(), wg.len(), "{name}: length");
    let mut diff = 0usize;
    let mut max_ulp = 0i32;
    let mut first = None;
    for (i, (a, b)) in cu.iter().zip(wg.iter()).enumerate() {
        if a != b {
            diff += 1;
            max_ulp = max_ulp.max((*a as i32 - *b as i32).abs());
            if first.is_none() {
                first = Some((i, *a, *b));
            }
        }
    }
    eprintln!(
        "{name}: {diff}/{} bf16 words differ, max_ulp={max_ulp}",
        cu.len()
    );
    assert_eq!(diff, 0, "{name}: first mismatch {first:?}");
}

fn gating_inputs(tokens: usize, num_heads: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let total = tokens * num_heads;
    let mut seed = 0x51ed_2701_u64 ^ (total as u64);
    let mut a = Vec::with_capacity(total);
    let mut b = Vec::with_capacity(total);
    for i in 0..total {
        let u = lcg(&mut seed);
        let w = lcg(&mut seed);
        a.push(u * 90.0 - 45.0);
        b.push(w * 60.0 - 30.0);
        if i % 37 == 0 {
            let last = a.len() - 1;
            a[last] = 20.0;
        }
        if i % 53 == 0 {
            let last = a.len() - 1;
            a[last] = -20.0;
        }
        if i % 71 == 0 {
            let last = a.len() - 1;
            a[last] = 0.0;
        }
    }
    let a_log: Vec<f32> = (0..num_heads).map(|h| -3.5 + (h as f32) * 0.37).collect();
    let dt_bias: Vec<f32> = (0..num_heads)
        .map(|h| ((h as f32) * 0.913).sin() * 2.0 - 0.25)
        .collect();
    (a, b, a_log, dt_bias)
}

const GATING_SHAPES: [(usize, usize); 5] = [(1, 1), (1, 8), (3, 5), (257, 8), (1024, 16)];

#[test]
fn gdn_gating_f32_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gdn_gating_f32") else {
        return;
    };
    for (tokens, num_heads) in GATING_SHAPES {
        let total = tokens * num_heads;
        let (a, b, a_log, dt_bias) = gating_inputs(tokens, num_heads);

        #[allow(deprecated)]
        let da: CudaSlice<f32> = stream.clone_htod(&a).unwrap();
        #[allow(deprecated)]
        let db: CudaSlice<f32> = stream.clone_htod(&b).unwrap();
        #[allow(deprecated)]
        let dl: CudaSlice<f32> = stream.clone_htod(&a_log).unwrap();
        #[allow(deprecated)]
        let dt: CudaSlice<f32> = stream.clone_htod(&dt_bias).unwrap();
        let mut dg: CudaSlice<f32> = stream.alloc_zeros::<f32>(total).unwrap();
        let mut dbeta: CudaSlice<f32> = stream.alloc_zeros::<f32>(total).unwrap();
        let rc = {
            let (pa, _1) = da.device_ptr(&stream);
            let (pb, _2) = db.device_ptr(&stream);
            let (pl, _3) = dl.device_ptr(&stream);
            let (pt, _4) = dt.device_ptr(&stream);
            let (pg, _5) = dg.device_ptr_mut(&stream);
            let (pbe, _6) = dbeta.device_ptr_mut(&stream);
            unsafe {
                cuda::gdn_gating_f32(
                    stream.cu_stream() as *mut c_void,
                    pa as *const f32,
                    pb as *const f32,
                    pl as *const f32,
                    pt as *const f32,
                    pg as *mut f32,
                    pbe as *mut f32,
                    tokens,
                    num_heads,
                )
            }
        };
        assert_eq!(rc, 0, "cuda gdn_gating_f32 rc={rc}");
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        let cu_g = stream.memcpy_dtov(&dg).unwrap();
        #[allow(deprecated)]
        let cu_beta = stream.memcpy_dtov(&dbeta).unwrap();

        let mut wg_g = vec![0f32; total];
        let mut wg_beta = vec![0f32; total];
        gdn_gating::gdn_gating_f32(
            wg,
            &a,
            &b,
            &a_log,
            &dt_bias,
            &mut wg_g,
            &mut wg_beta,
            tokens,
            num_heads,
        )
        .unwrap();

        assert_f32_ulp(
            &format!("gating_f32 g t={tokens} h={num_heads}"),
            &cu_g,
            &wg_g,
            0,
        );
        assert_f32_ulp(
            &format!("gating_f32 beta t={tokens} h={num_heads}"),
            &cu_beta,
            &wg_beta,
            0,
        );
    }
}

#[test]
fn gdn_gating_bf16_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gdn_gating_bf16") else {
        return;
    };
    for (tokens, num_heads) in GATING_SHAPES {
        let total = tokens * num_heads;
        let (af, bf, lf, tf) = gating_inputs(tokens, num_heads);
        let a = to_bf16(&af);
        let b = to_bf16(&bf);
        let a_log = to_bf16(&lf);
        let dt_bias = to_bf16(&tf);

        #[allow(deprecated)]
        let da: CudaSlice<u16> = stream.clone_htod(&a).unwrap();
        #[allow(deprecated)]
        let db: CudaSlice<u16> = stream.clone_htod(&b).unwrap();
        #[allow(deprecated)]
        let dl: CudaSlice<u16> = stream.clone_htod(&a_log).unwrap();
        #[allow(deprecated)]
        let dt: CudaSlice<u16> = stream.clone_htod(&dt_bias).unwrap();
        let mut dg: CudaSlice<f32> = stream.alloc_zeros::<f32>(total).unwrap();
        let mut dbeta: CudaSlice<u16> = stream.alloc_zeros::<u16>(total).unwrap();
        let rc = {
            let (pa, _1) = da.device_ptr(&stream);
            let (pb, _2) = db.device_ptr(&stream);
            let (pl, _3) = dl.device_ptr(&stream);
            let (pt, _4) = dt.device_ptr(&stream);
            let (pg, _5) = dg.device_ptr_mut(&stream);
            let (pbe, _6) = dbeta.device_ptr_mut(&stream);
            unsafe {
                cuda::gdn_gating_bf16(
                    stream.cu_stream() as *mut c_void,
                    pa as *const u16,
                    pb as *const u16,
                    pl as *const u16,
                    pt as *const u16,
                    pg as *mut f32,
                    pbe as *mut u16,
                    tokens,
                    num_heads,
                )
            }
        };
        assert_eq!(rc, 0, "cuda gdn_gating_bf16 rc={rc}");
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        let cu_g = stream.memcpy_dtov(&dg).unwrap();
        #[allow(deprecated)]
        let cu_beta = stream.memcpy_dtov(&dbeta).unwrap();

        let mut wg_g = vec![0f32; total];
        let mut wg_beta = vec![0u16; total];
        gdn_gating::gdn_gating_bf16(
            wg,
            &a,
            &b,
            &a_log,
            &dt_bias,
            &mut wg_g,
            &mut wg_beta,
            tokens,
            num_heads,
        )
        .unwrap();

        assert_f32_ulp(
            &format!("gating_bf16 g t={tokens} h={num_heads}"),
            &cu_g,
            &wg_g,
            0,
        );
        assert_bf16_exact(
            &format!("gating_bf16 beta t={tokens} h={num_heads}"),
            &cu_beta,
            &wg_beta,
        );
    }
}

const DIM: usize = 128;

const F32_KERNEL_VS_F64_HOST_ORACLE_REL_TOL_T_LE16_DIM128_4X_HEADROOM: f64 = 5.5e-4;

fn assert_f64_host_oracle_rel_tol(name: &str, cand: &[f32], oracle: &[f32]) {
    assert_eq!(cand.len(), oracle.len(), "{name}: length");
    let mut max_rel = 0f64;
    let mut first = None;
    for (i, (a, r)) in cand.iter().zip(oracle.iter()).enumerate() {
        let rel = ((*a as f64) - (*r as f64)).abs() / (a.abs() as f64).max(r.abs() as f64).max(1e-3);
        if rel > max_rel {
            max_rel = rel;
            first = Some((i, *a, *r, rel));
        }
    }
    assert!(
        max_rel <= F32_KERNEL_VS_F64_HOST_ORACLE_REL_TOL_T_LE16_DIM128_4X_HEADROOM,
        "{name}: max_rel {max_rel:.3e} > {F32_KERNEL_VS_F64_HOST_ORACLE_REL_TOL_T_LE16_DIM128_4X_HEADROOM:.3e}, first {first:?}"
    );
}

fn recurrent_inputs(
    b: usize,
    t: usize,
    h: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let rows = b * t * h;
    let vecs = rows * DIM;
    let mut seed = 0x9e37_79b9_u64 ^ (vecs as u64);
    let mut q = Vec::with_capacity(vecs);
    let mut k = Vec::with_capacity(vecs);
    let mut v = Vec::with_capacity(vecs);
    for _ in 0..vecs {
        q.push(lcg(&mut seed) * 2.0 - 1.0);
        k.push((lcg(&mut seed) * 2.0 - 1.0) * 0.125);
        v.push(lcg(&mut seed) * 2.0 - 1.0);
    }
    let mut g = Vec::with_capacity(rows);
    let mut beta = Vec::with_capacity(rows);
    for _ in 0..rows {
        g.push(0.75 + lcg(&mut seed) * 0.25);
        beta.push(lcg(&mut seed));
    }
    (q, k, v, g, beta)
}

#[test]
fn gdn_recurrent_f32_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gdn_recurrent_f32") else {
        return;
    };
    for (b, t, h) in [(1usize, 1usize, 1usize), (1, 7, 2), (2, 5, 3), (2, 16, 4)] {
        let rows = b * t * h;
        let vecs = rows * DIM;
        let (q, k, v, g, beta) = recurrent_inputs(b, t, h);

        #[allow(deprecated)]
        let dq: CudaSlice<f32> = stream.clone_htod(&q).unwrap();
        #[allow(deprecated)]
        let dk: CudaSlice<f32> = stream.clone_htod(&k).unwrap();
        #[allow(deprecated)]
        let dv: CudaSlice<f32> = stream.clone_htod(&v).unwrap();
        #[allow(deprecated)]
        let dg: CudaSlice<f32> = stream.clone_htod(&g).unwrap();
        #[allow(deprecated)]
        let dbeta: CudaSlice<f32> = stream.clone_htod(&beta).unwrap();
        let mut dout: CudaSlice<f32> = stream.alloc_zeros::<f32>(vecs).unwrap();
        let rc = {
            let (pq, _1) = dq.device_ptr(&stream);
            let (pk, _2) = dk.device_ptr(&stream);
            let (pv, _3) = dv.device_ptr(&stream);
            let (pg, _4) = dg.device_ptr(&stream);
            let (pb, _5) = dbeta.device_ptr(&stream);
            let (po, _6) = dout.device_ptr_mut(&stream);
            unsafe {
                cuda::gdn_recurrent_f32(
                    stream.cu_stream() as *mut c_void,
                    pq as *const f32,
                    pk as *const f32,
                    pv as *const f32,
                    pg as *const f32,
                    pb as *const f32,
                    po as *mut f32,
                    b as i32,
                    t as i32,
                    h as i32,
                    DIM as i32,
                    DIM as i32,
                )
            }
        };
        assert_eq!(rc, 0, "cuda gdn_recurrent_f32 rc={rc}");
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        let cu_out = stream.memcpy_dtov(&dout).unwrap();

        let mut wg_out = vec![0f32; vecs];
        let mut wg_state = vec![0f32; b * h * DIM * DIM];
        gdn_recurrent::gdn_recurrent_f32(
            wg,
            &q,
            &k,
            &v,
            &g,
            &beta,
            &mut wg_out,
            &mut wg_state,
            b,
            h,
            t,
        )
        .unwrap();

        assert_f32_ulp(&format!("recurrent b={b} t={t} h={h}"), &cu_out, &wg_out, 0);
        assert!(
            wg_state.iter().any(|x| *x != 0.0),
            "recurrent b={b} t={t} h={h}: final state must not be all zero"
        );

        let f64_oracle_out =
            gdn_host_ref::ref_rank1_algebra_f64(&q, &k, &v, &g, &beta, b, t, h, gdn_host_ref::PlantedBug::None);
        assert_f64_host_oracle_rel_tol(
            &format!("recurrent cuda vs f64 host oracle b={b} t={t} h={h}"),
            &cu_out,
            &f64_oracle_out,
        );
        assert_f64_host_oracle_rel_tol(
            &format!("recurrent wgpu vs f64 host oracle b={b} t={t} h={h}"),
            &wg_out,
            &f64_oracle_out,
        );
    }
}

fn drive_gating_f32(
    stream: &Arc<CudaStream>,
    wg: &'static WgpuContext,
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    tokens: usize,
    num_heads: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let total = tokens * num_heads;
    #[allow(deprecated)]
    let da: CudaSlice<f32> = stream.clone_htod(&a.to_vec()).unwrap();
    #[allow(deprecated)]
    let db: CudaSlice<f32> = stream.clone_htod(&b.to_vec()).unwrap();
    #[allow(deprecated)]
    let dl: CudaSlice<f32> = stream.clone_htod(&a_log.to_vec()).unwrap();
    #[allow(deprecated)]
    let dt: CudaSlice<f32> = stream.clone_htod(&dt_bias.to_vec()).unwrap();
    let mut dg: CudaSlice<f32> = stream.alloc_zeros::<f32>(total).unwrap();
    let mut dbeta: CudaSlice<f32> = stream.alloc_zeros::<f32>(total).unwrap();
    let rc = {
        let (pa, _1) = da.device_ptr(stream);
        let (pb, _2) = db.device_ptr(stream);
        let (pl, _3) = dl.device_ptr(stream);
        let (pt, _4) = dt.device_ptr(stream);
        let (pg, _5) = dg.device_ptr_mut(stream);
        let (pbe, _6) = dbeta.device_ptr_mut(stream);
        unsafe {
            cuda::gdn_gating_f32(
                stream.cu_stream() as *mut c_void,
                pa as *const f32,
                pb as *const f32,
                pl as *const f32,
                pt as *const f32,
                pg as *mut f32,
                pbe as *mut f32,
                tokens,
                num_heads,
            )
        }
    };
    assert_eq!(rc, 0, "cuda gdn_gating_f32 rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let cu_g = stream.memcpy_dtov(&dg).unwrap();
    #[allow(deprecated)]
    let cu_beta = stream.memcpy_dtov(&dbeta).unwrap();

    let mut wg_g = vec![0f32; total];
    let mut wg_beta = vec![0f32; total];
    gdn_gating::gdn_gating_f32(
        wg,
        a,
        b,
        a_log,
        dt_bias,
        &mut wg_g,
        &mut wg_beta,
        tokens,
        num_heads,
    )
    .unwrap();
    (cu_g, cu_beta, wg_g, wg_beta)
}

fn drive_gating_bf16(
    stream: &Arc<CudaStream>,
    wg: &'static WgpuContext,
    a: &[u16],
    b: &[u16],
    a_log: &[u16],
    dt_bias: &[u16],
    tokens: usize,
    num_heads: usize,
) -> (Vec<f32>, Vec<u16>, Vec<f32>, Vec<u16>) {
    let total = tokens * num_heads;
    #[allow(deprecated)]
    let da: CudaSlice<u16> = stream.clone_htod(&a.to_vec()).unwrap();
    #[allow(deprecated)]
    let db: CudaSlice<u16> = stream.clone_htod(&b.to_vec()).unwrap();
    #[allow(deprecated)]
    let dl: CudaSlice<u16> = stream.clone_htod(&a_log.to_vec()).unwrap();
    #[allow(deprecated)]
    let dt: CudaSlice<u16> = stream.clone_htod(&dt_bias.to_vec()).unwrap();
    let mut dg: CudaSlice<f32> = stream.alloc_zeros::<f32>(total).unwrap();
    let mut dbeta: CudaSlice<u16> = stream.alloc_zeros::<u16>(total).unwrap();
    let rc = {
        let (pa, _1) = da.device_ptr(stream);
        let (pb, _2) = db.device_ptr(stream);
        let (pl, _3) = dl.device_ptr(stream);
        let (pt, _4) = dt.device_ptr(stream);
        let (pg, _5) = dg.device_ptr_mut(stream);
        let (pbe, _6) = dbeta.device_ptr_mut(stream);
        unsafe {
            cuda::gdn_gating_bf16(
                stream.cu_stream() as *mut c_void,
                pa as *const u16,
                pb as *const u16,
                pl as *const u16,
                pt as *const u16,
                pg as *mut f32,
                pbe as *mut u16,
                tokens,
                num_heads,
            )
        }
    };
    assert_eq!(rc, 0, "cuda gdn_gating_bf16 rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let cu_g = stream.memcpy_dtov(&dg).unwrap();
    #[allow(deprecated)]
    let cu_beta = stream.memcpy_dtov(&dbeta).unwrap();

    let mut wg_g = vec![0f32; total];
    let mut wg_beta = vec![0u16; total];
    gdn_gating::gdn_gating_bf16(
        wg,
        a,
        b,
        a_log,
        dt_bias,
        &mut wg_g,
        &mut wg_beta,
        tokens,
        num_heads,
    )
    .unwrap();
    (cu_g, cu_beta, wg_g, wg_beta)
}

const EXTREME_TOKENS: usize = 4096;
const EXTREME_HEADS: usize = 8;

fn extreme_gating_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let total = EXTREME_TOKENS * EXTREME_HEADS;
    let denom = (total - 1) as f32;
    let mut a = Vec::with_capacity(total);
    let mut b = Vec::with_capacity(total);
    for i in 0..total {
        let u = (i as f32) / denom;
        a.push(u * 240.0 - 120.0);
        b.push(120.0 - u * 240.0);
    }
    let anchors: [f32; 16] = [
        -120.0, -104.0, -95.0, -88.7283, -88.7, -88.0, -87.4, -30.0, -20.0, 0.0, 20.0, 20.0001,
        87.4, 88.0, 88.7283, 120.0,
    ];
    for (j, v) in anchors.iter().enumerate() {
        a[j] = *v;
        b[j] = *v;
        a[total - 1 - j] = -*v;
        b[total - 1 - j] = -*v;
    }
    let hd = (EXTREME_HEADS - 1) as f32;
    let a_log: Vec<f32> = (0..EXTREME_HEADS)
        .map(|h| -40.0 + (h as f32) * (80.0 / hd))
        .collect();
    let dt_bias: Vec<f32> = (0..EXTREME_HEADS)
        .map(|h| ((h as f32) * 1.7).sin() * 3.0)
        .collect();
    (a, b, a_log, dt_bias)
}

#[test]
fn gdn_gating_f32_extremes_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gdn_gating_f32_extremes") else {
        return;
    };
    let (a, b, a_log, dt_bias) = extreme_gating_inputs();
    let (cu_g, cu_beta, wg_g, wg_beta) = drive_gating_f32(
        &stream,
        wg,
        &a,
        &b,
        &a_log,
        &dt_bias,
        EXTREME_TOKENS,
        EXTREME_HEADS,
    );
    assert_exercises_subnormals("gating_f32_extremes g", &cu_g);
    assert_exercises_subnormals("gating_f32_extremes beta", &cu_beta);
    assert!(
        cu_beta.iter().any(|x| *x == 0.0),
        "gating_f32_extremes: fixture no longer reaches the saturating-sigmoid path"
    );
    assert_f32_ulp("gating_f32_extremes g", &cu_g, &wg_g, 0);
    assert_f32_ulp("gating_f32_extremes beta", &cu_beta, &wg_beta, 0);
}

#[test]
fn gdn_gating_bf16_extremes_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gdn_gating_bf16_extremes") else {
        return;
    };
    let (af, bf, lf, tf) = extreme_gating_inputs();
    let a = to_bf16(&af);
    let b = to_bf16(&bf);
    let a_log = to_bf16(&lf);
    let dt_bias = to_bf16(&tf);
    let (cu_g, cu_beta, wg_g, wg_beta) = drive_gating_bf16(
        &stream,
        wg,
        &a,
        &b,
        &a_log,
        &dt_bias,
        EXTREME_TOKENS,
        EXTREME_HEADS,
    );
    assert_exercises_subnormals("gating_bf16_extremes g", &cu_g);
    assert_f32_ulp("gating_bf16_extremes g", &cu_g, &wg_g, 0);
    assert_bf16_exact("gating_bf16_extremes beta", &cu_beta, &wg_beta);
}

fn recurrent_oracle(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    b: usize,
    t: usize,
    h: usize,
) -> (Vec<f32>, Vec<f32>) {
    let rows = b * t * h;
    let mut out = vec![0f32; rows * DIM];
    let mut state = vec![0f32; b * h * DIM * DIM];
    for bi in 0..b {
        for hi in 0..h {
            let sbase = (bi * h + hi) * DIM * DIM;
            for ti in 0..t {
                let kv_base = (bi * t + ti) * h + hi;
                let vec_base = kv_base * DIM;
                let ge = g[kv_base];
                let bt = beta[kv_base];
                for my_v in 0..DIM {
                    let v_t = v[vec_base + my_v];
                    let mut kv_mem = 0f32;
                    for kk in 0..DIM {
                        let slot = sbase + kk * DIM + my_v;
                        let s = state[slot] * ge;
                        state[slot] = s;
                        kv_mem = s.mul_add(k[vec_base + kk], kv_mem);
                    }
                    let delta = (v_t - kv_mem) * bt;
                    let mut out_v = 0f32;
                    for kk in 0..DIM {
                        let slot = sbase + kk * DIM + my_v;
                        let s = k[vec_base + kk].mul_add(delta, state[slot]);
                        state[slot] = s;
                        out_v = s.mul_add(q[vec_base + kk], out_v);
                    }
                    out[vec_base + my_v] = out_v;
                }
            }
        }
    }
    (out, state)
}

#[test]
fn gdn_recurrent_f32_state_matches_cpu_oracle() {
    let Some((_stream, wg)) = backends("gdn_recurrent_state") else {
        return;
    };
    for (b, t, h) in [(1usize, 1usize, 1usize), (1, 64, 2), (3, 9, 2)] {
        let rows = b * t * h;
        let vecs = rows * DIM;
        let (q, k, v, g, beta) = recurrent_inputs(b, t, h);
        let (ref_out, ref_state) = recurrent_oracle(&q, &k, &v, &g, &beta, b, t, h);

        let mut wg_out = vec![0f32; vecs];
        let mut wg_state = vec![0f32; b * h * DIM * DIM];
        gdn_recurrent::gdn_recurrent_f32(
            wg,
            &q,
            &k,
            &v,
            &g,
            &beta,
            &mut wg_out,
            &mut wg_state,
            b,
            h,
            t,
        )
        .unwrap();

        assert_f32_ulp(
            &format!("oracle out b={b} t={t} h={h}"),
            &ref_out,
            &wg_out,
            0,
        );
        assert_f32_ulp(
            &format!("oracle state b={b} t={t} h={h}"),
            &ref_state,
            &wg_state,
            0,
        );
    }
}

fn recurrent_stress_inputs(
    b: usize,
    t: usize,
    h: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let rows = b * t * h;
    let vecs = rows * DIM;
    let mut seed = 0xc0ff_ee11_u64 ^ (vecs as u64);
    let mut q = Vec::with_capacity(vecs);
    let mut k = Vec::with_capacity(vecs);
    let mut v = Vec::with_capacity(vecs);
    for i in 0..vecs {
        let scale = if i % 5 == 0 { 8.0 } else { 0.03125 };
        q.push((lcg(&mut seed) * 2.0 - 1.0) * scale);
        k.push((lcg(&mut seed) * 2.0 - 1.0) * 0.25);
        v.push((lcg(&mut seed) * 2.0 - 1.0) * scale);
        if i % 257 == 0 {
            let last = k.len() - 1;
            k[last] = 0.0;
        }
    }
    let g_choices = [0.0f32, 1.0, 0.5, 1.25, 0.9999999];
    let beta_choices = [0.0f32, 1.0, -0.5, 2.0, 0.75];
    let mut g = Vec::with_capacity(rows);
    let mut beta = Vec::with_capacity(rows);
    for i in 0..rows {
        g.push(g_choices[i % g_choices.len()]);
        beta.push(beta_choices[(i / 3) % beta_choices.len()]);
    }
    (q, k, v, g, beta)
}

#[test]
fn gdn_recurrent_f32_stress_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gdn_recurrent_stress") else {
        return;
    };
    for (b, t, h) in [(1usize, 11usize, 1usize), (2, 24, 3)] {
        let rows = b * t * h;
        let vecs = rows * DIM;
        let (q, k, v, g, beta) = recurrent_stress_inputs(b, t, h);

        #[allow(deprecated)]
        let dq: CudaSlice<f32> = stream.clone_htod(&q).unwrap();
        #[allow(deprecated)]
        let dk: CudaSlice<f32> = stream.clone_htod(&k).unwrap();
        #[allow(deprecated)]
        let dv: CudaSlice<f32> = stream.clone_htod(&v).unwrap();
        #[allow(deprecated)]
        let dg: CudaSlice<f32> = stream.clone_htod(&g).unwrap();
        #[allow(deprecated)]
        let dbeta: CudaSlice<f32> = stream.clone_htod(&beta).unwrap();
        let mut dout: CudaSlice<f32> = stream.alloc_zeros::<f32>(vecs).unwrap();
        let rc = {
            let (pq, _1) = dq.device_ptr(&stream);
            let (pk, _2) = dk.device_ptr(&stream);
            let (pv, _3) = dv.device_ptr(&stream);
            let (pg, _4) = dg.device_ptr(&stream);
            let (pb, _5) = dbeta.device_ptr(&stream);
            let (po, _6) = dout.device_ptr_mut(&stream);
            unsafe {
                cuda::gdn_recurrent_f32(
                    stream.cu_stream() as *mut c_void,
                    pq as *const f32,
                    pk as *const f32,
                    pv as *const f32,
                    pg as *const f32,
                    pb as *const f32,
                    po as *mut f32,
                    b as i32,
                    t as i32,
                    h as i32,
                    DIM as i32,
                    DIM as i32,
                )
            }
        };
        assert_eq!(rc, 0, "cuda gdn_recurrent_f32 rc={rc}");
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        let cu_out = stream.memcpy_dtov(&dout).unwrap();

        let mut wg_out = vec![0f32; vecs];
        let mut wg_state = vec![0f32; b * h * DIM * DIM];
        gdn_recurrent::gdn_recurrent_f32(
            wg,
            &q,
            &k,
            &v,
            &g,
            &beta,
            &mut wg_out,
            &mut wg_state,
            b,
            h,
            t,
        )
        .unwrap();

        let (ref_out, ref_state) = recurrent_oracle(&q, &k, &v, &g, &beta, b, t, h);
        assert_f32_ulp(
            &format!("stress out b={b} t={t} h={h}"),
            &cu_out,
            &wg_out,
            0,
        );
        assert_f32_ulp(
            &format!("stress oracle out b={b} t={t} h={h}"),
            &ref_out,
            &wg_out,
            0,
        );
        assert_f32_ulp(
            &format!("stress oracle state b={b} t={t} h={h}"),
            &ref_state,
            &wg_state,
            0,
        );
    }
}

#[test]
fn gdn_recurrent_f32_long_seq_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gdn_recurrent_long") else {
        return;
    };
    for (b, t, h) in [(1usize, 128usize, 1usize), (2, 33, 5)] {
        let rows = b * t * h;
        let vecs = rows * DIM;
        let (q, k, v, g, beta) = recurrent_inputs(b, t, h);

        #[allow(deprecated)]
        let dq: CudaSlice<f32> = stream.clone_htod(&q).unwrap();
        #[allow(deprecated)]
        let dk: CudaSlice<f32> = stream.clone_htod(&k).unwrap();
        #[allow(deprecated)]
        let dv: CudaSlice<f32> = stream.clone_htod(&v).unwrap();
        #[allow(deprecated)]
        let dg: CudaSlice<f32> = stream.clone_htod(&g).unwrap();
        #[allow(deprecated)]
        let dbeta: CudaSlice<f32> = stream.clone_htod(&beta).unwrap();
        let mut dout: CudaSlice<f32> = stream.alloc_zeros::<f32>(vecs).unwrap();
        let rc = {
            let (pq, _1) = dq.device_ptr(&stream);
            let (pk, _2) = dk.device_ptr(&stream);
            let (pv, _3) = dv.device_ptr(&stream);
            let (pg, _4) = dg.device_ptr(&stream);
            let (pb, _5) = dbeta.device_ptr(&stream);
            let (po, _6) = dout.device_ptr_mut(&stream);
            unsafe {
                cuda::gdn_recurrent_f32(
                    stream.cu_stream() as *mut c_void,
                    pq as *const f32,
                    pk as *const f32,
                    pv as *const f32,
                    pg as *const f32,
                    pb as *const f32,
                    po as *mut f32,
                    b as i32,
                    t as i32,
                    h as i32,
                    DIM as i32,
                    DIM as i32,
                )
            }
        };
        assert_eq!(rc, 0, "cuda gdn_recurrent_f32 rc={rc}");
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        let cu_out = stream.memcpy_dtov(&dout).unwrap();

        let mut wg_out = vec![0f32; vecs];
        let mut wg_state = vec![0f32; b * h * DIM * DIM];
        gdn_recurrent::gdn_recurrent_f32(
            wg,
            &q,
            &k,
            &v,
            &g,
            &beta,
            &mut wg_out,
            &mut wg_state,
            b,
            h,
            t,
        )
        .unwrap();

        assert_f32_ulp(
            &format!("recurrent_long b={b} t={t} h={h}"),
            &cu_out,
            &wg_out,
            0,
        );
    }
}

const SWEEP_TOKENS: usize = 65536;
const SWEEP_HEADS: usize = 8;

fn sweep_value(seed: &mut u64) -> f32 {
    let r = rand_bits(seed);
    let mant = r & 0x007f_ffff;
    let sign = r >> 31;
    let e = -34i32 + (rand_bits(seed) % 44) as i32;
    f32::from_bits((sign << 31) | (((e + 127) as u32) << 23) | mant)
}

fn sweep_gating_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let total = SWEEP_TOKENS * SWEEP_HEADS;
    let mut seed = 0x2b1f_a97c_u64;
    let mut a = Vec::with_capacity(total);
    let mut b = Vec::with_capacity(total);
    for _ in 0..total {
        a.push(sweep_value(&mut seed));
        b.push(sweep_value(&mut seed));
    }
    let anchors: [f32; 14] = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        20.0,
        -20.0,
        88.72283,
        -87.336545,
    ];
    for (j, v) in anchors.iter().enumerate() {
        a[j * SWEEP_HEADS] = *v;
        b[j * SWEEP_HEADS] = *v;
        a[total - 1 - j * SWEEP_HEADS] = -*v;
        b[total - 1 - j * SWEEP_HEADS] = -*v;
    }
    let hd = (SWEEP_HEADS - 1) as f32;
    let a_log: Vec<f32> = (0..SWEEP_HEADS)
        .map(|h| -45.0 + (h as f32) * (90.0 / hd))
        .collect();
    let mut dt_bias: Vec<f32> = (0..SWEEP_HEADS)
        .map(|h| ((h as f32) * 2.31).cos() * 4.0 - 0.5)
        .collect();
    dt_bias[0] = 0.0;
    (a, b, a_log, dt_bias)
}

fn assert_sweep_reaches_every_branch(a: &[f32], dt_bias: &[f32], g: &[f32], beta: &[f32]) {
    let heads = dt_bias.len();
    let mut upper = 0usize;
    let mut lower = 0usize;
    let mut middle = 0usize;
    for (i, x) in a.iter().enumerate() {
        let s = x + dt_bias[i % heads];
        if s > 20.0 {
            upper += 1;
        } else if s < -20.0 {
            lower += 1;
        } else {
            middle += 1;
        }
    }
    let sub = g.iter().filter(|x| x.is_subnormal()).count();
    let inf = g.iter().filter(|x| x.is_infinite()).count();
    let bsub = beta.iter().filter(|x| x.is_subnormal()).count();
    let bzero = beta.iter().filter(|x| **x == 0.0).count();
    let bone = beta.iter().filter(|x| **x == 1.0).count();
    eprintln!(
        "sweep coverage: softplus upper={upper} lower={lower} middle={middle}; \
         g subnormal={sub} inf={inf}; beta subnormal={bsub} zero={bzero} one={bone}"
    );
    assert!(
        upper > 0 && lower > 0 && middle > 0,
        "sweep misses a softplus branch"
    );
    assert!(sub > 0, "sweep no longer produces subnormal g");
    assert!(inf > 0, "sweep no longer produces overflowing g");
    assert!(bsub > 0, "sweep no longer produces subnormal beta");
    assert!(
        bzero > 0 && bone > 0,
        "sweep no longer saturates the sigmoid"
    );
}

#[test]
fn gdn_gating_f32_random_sweep_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gdn_gating_f32_sweep") else {
        return;
    };
    let (a, b, a_log, dt_bias) = sweep_gating_inputs();
    let (cu_g, cu_beta, wg_g, wg_beta) = drive_gating_f32(
        &stream,
        wg,
        &a,
        &b,
        &a_log,
        &dt_bias,
        SWEEP_TOKENS,
        SWEEP_HEADS,
    );
    assert_sweep_reaches_every_branch(&a, &dt_bias, &cu_g, &cu_beta);
    assert_f32_ulp("gating_f32_sweep g", &cu_g, &wg_g, 0);
    assert_f32_ulp("gating_f32_sweep beta", &cu_beta, &wg_beta, 0);
}

#[test]
fn gdn_gating_bf16_random_sweep_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gdn_gating_bf16_sweep") else {
        return;
    };
    let (af, bf, lf, tf) = sweep_gating_inputs();
    let a = to_bf16(&af);
    let b = to_bf16(&bf);
    let a_log = to_bf16(&lf);
    let dt_bias = to_bf16(&tf);
    let (cu_g, cu_beta, wg_g, wg_beta) = drive_gating_bf16(
        &stream,
        wg,
        &a,
        &b,
        &a_log,
        &dt_bias,
        SWEEP_TOKENS,
        SWEEP_HEADS,
    );
    let sub = cu_g.iter().filter(|x| x.is_subnormal()).count();
    let inf = cu_g.iter().filter(|x| x.is_infinite()).count();
    let bzero = cu_beta.iter().filter(|w| **w == 0).count();
    eprintln!(
        "bf16 sweep coverage: g subnormal={sub} inf={inf}; beta zero words={bzero}/{}",
        cu_beta.len()
    );
    assert!(
        sub > 0 && inf > 0,
        "bf16 sweep no longer spans the f32 range of g"
    );
    assert!(bzero > 0, "bf16 sweep no longer saturates the sigmoid");
    assert_f32_ulp("gating_bf16_sweep g", &cu_g, &wg_g, 0);
    assert_bf16_exact("gating_bf16_sweep beta", &cu_beta, &wg_beta);
}

#[test]
fn gdn_recurrent_f32_many_pairs_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gdn_recurrent_many_pairs") else {
        return;
    };
    let (b, t, h) = (4usize, 6usize, 8usize);
    let rows = b * t * h;
    let vecs = rows * DIM;
    let (q, k, v, g, beta) = recurrent_stress_inputs(b, t, h);

    #[allow(deprecated)]
    let dq: CudaSlice<f32> = stream.clone_htod(&q).unwrap();
    #[allow(deprecated)]
    let dk: CudaSlice<f32> = stream.clone_htod(&k).unwrap();
    #[allow(deprecated)]
    let dv: CudaSlice<f32> = stream.clone_htod(&v).unwrap();
    #[allow(deprecated)]
    let dg: CudaSlice<f32> = stream.clone_htod(&g).unwrap();
    #[allow(deprecated)]
    let dbeta: CudaSlice<f32> = stream.clone_htod(&beta).unwrap();
    let mut dout: CudaSlice<f32> = stream.alloc_zeros::<f32>(vecs).unwrap();
    let rc = {
        let (pq, _1) = dq.device_ptr(&stream);
        let (pk, _2) = dk.device_ptr(&stream);
        let (pv, _3) = dv.device_ptr(&stream);
        let (pg, _4) = dg.device_ptr(&stream);
        let (pb, _5) = dbeta.device_ptr(&stream);
        let (po, _6) = dout.device_ptr_mut(&stream);
        unsafe {
            cuda::gdn_recurrent_f32(
                stream.cu_stream() as *mut c_void,
                pq as *const f32,
                pk as *const f32,
                pv as *const f32,
                pg as *const f32,
                pb as *const f32,
                po as *mut f32,
                b as i32,
                t as i32,
                h as i32,
                DIM as i32,
                DIM as i32,
            )
        }
    };
    assert_eq!(rc, 0, "cuda gdn_recurrent_f32 rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let cu_out = stream.memcpy_dtov(&dout).unwrap();

    let mut wg_out = vec![0f32; vecs];
    let mut wg_state = vec![0f32; b * h * DIM * DIM];
    gdn_recurrent::gdn_recurrent_f32(
        wg,
        &q,
        &k,
        &v,
        &g,
        &beta,
        &mut wg_out,
        &mut wg_state,
        b,
        h,
        t,
    )
    .unwrap();

    let (ref_out, ref_state) = recurrent_oracle(&q, &k, &v, &g, &beta, b, t, h);
    let nonzero = cu_out.iter().filter(|x| **x != 0.0).count();
    eprintln!("many_pairs: {nonzero}/{vecs} cuda outputs nonzero");
    assert!(nonzero * 2 > vecs, "many_pairs: cuda output is mostly zero");
    assert_f32_ulp("many_pairs out", &cu_out, &wg_out, 0);
    assert_f32_ulp("many_pairs oracle out", &ref_out, &wg_out, 0);
    assert_f32_ulp("many_pairs oracle state", &ref_state, &wg_state, 0);
}

#[test]
fn gdn_recurrent_f32_rejects_bad_shapes() {
    let Some((_stream, wg)) = backends("gdn_recurrent_shapes") else {
        return;
    };
    let (b, t, h) = (1usize, 2usize, 1usize);
    let (q, k, v, g, beta) = recurrent_inputs(b, t, h);
    let mut out = vec![0f32; b * t * h * DIM];
    let mut state = vec![0f32; b * h * DIM * DIM];
    let e = gdn_recurrent::gdn_recurrent_f32(
        wg,
        &q[..q.len() - 1],
        &k,
        &v,
        &g,
        &beta,
        &mut out,
        &mut state,
        b,
        h,
        t,
    )
    .unwrap_err();
    eprintln!("short q rejected: {e}");
    let mut short_state = vec![0f32; b * h * DIM * DIM - 1];
    let e = gdn_recurrent::gdn_recurrent_f32(
        wg,
        &q,
        &k,
        &v,
        &g,
        &beta,
        &mut out,
        &mut short_state,
        b,
        h,
        t,
    )
    .unwrap_err();
    eprintln!("short state rejected: {e}");
}

fn assert_nan_pattern_and_bits(name: &str, cu: &[f32], wg: &[f32]) -> usize {
    assert_eq!(cu.len(), wg.len(), "{name}: length");
    let mut nans = 0usize;
    let mut mism = 0usize;
    let mut first = None;
    for (i, (a, b)) in cu.iter().zip(wg.iter()).enumerate() {
        if a.is_nan() {
            nans += 1;
        }
        let bad = if a.is_nan() || b.is_nan() {
            a.is_nan() != b.is_nan()
        } else {
            a.to_bits() != b.to_bits()
        };
        if bad {
            mism += 1;
            if first.is_none() {
                first = Some((i, *a, a.to_bits(), *b, b.to_bits()));
            }
        }
    }
    eprintln!(
        "{name}: {nans}/{} cuda NaN, {mism} lanes disagree",
        cu.len()
    );
    assert_eq!(mism, 0, "{name}: first mismatch {first:?}");
    nans
}

fn bf16_is_nan(w: u16) -> bool {
    (w & 0x7f80) == 0x7f80 && (w & 0x007f) != 0
}

fn assert_bf16_nan_pattern_and_bits(name: &str, cu: &[u16], wg: &[u16]) -> usize {
    assert_eq!(cu.len(), wg.len(), "{name}: length");
    let mut nans = 0usize;
    let mut mism = 0usize;
    let mut first = None;
    for (i, (a, b)) in cu.iter().zip(wg.iter()).enumerate() {
        let an = bf16_is_nan(*a);
        let bn = bf16_is_nan(*b);
        if an {
            nans += 1;
        }
        let bad = if an || bn { an != bn } else { a != b };
        if bad {
            mism += 1;
            if first.is_none() {
                first = Some((i, *a, *b));
            }
        }
    }
    eprintln!(
        "{name}: {nans}/{} cuda NaN words, {mism} words disagree",
        cu.len()
    );
    assert_eq!(mism, 0, "{name}: first mismatch {first:?}");
    nans
}

const NAN_TOKENS: usize = 96;
const NAN_HEADS: usize = 8;

fn nan_gating_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let total = NAN_TOKENS * NAN_HEADS;
    let payloads: [u32; 6] = [
        0x7fc0_0000,
        0xffc0_0000,
        0x7fff_ffff,
        0xffbf_ffff,
        0x7f80_0001,
        0xff80_0001,
    ];
    let mut seed = 0x7c1d_a55e_u64;
    let mut a = Vec::with_capacity(total);
    let mut b = Vec::with_capacity(total);
    for i in 0..total {
        let u = lcg(&mut seed);
        let w = lcg(&mut seed);
        if i % 5 == 0 {
            a.push(f32::from_bits(payloads[(i / 5) % payloads.len()]));
        } else {
            a.push(u * 90.0 - 45.0);
        }
        if i % 7 == 0 {
            b.push(f32::from_bits(payloads[(i / 7) % payloads.len()]));
        } else {
            b.push(w * 60.0 - 30.0);
        }
    }
    let mut a_log: Vec<f32> = (0..NAN_HEADS).map(|h| -6.0 + (h as f32) * 1.5).collect();
    let mut dt_bias: Vec<f32> = (0..NAN_HEADS)
        .map(|h| ((h as f32) * 0.913).sin() * 2.0 - 0.25)
        .collect();
    a_log[3] = f32::from_bits(0x7fc0_0000);
    dt_bias[6] = f32::from_bits(0xffc0_0000);
    (a, b, a_log, dt_bias)
}

#[test]
fn gdn_gating_f32_nan_propagation_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gdn_gating_f32_nan") else {
        return;
    };
    let (a, b, a_log, dt_bias) = nan_gating_inputs();
    let (cu_g, cu_beta, wg_g, wg_beta) =
        drive_gating_f32(&stream, wg, &a, &b, &a_log, &dt_bias, NAN_TOKENS, NAN_HEADS);
    let gn = assert_nan_pattern_and_bits("gating_f32_nan g", &cu_g, &wg_g);
    let bn = assert_nan_pattern_and_bits("gating_f32_nan beta", &cu_beta, &wg_beta);
    assert!(gn > 0, "fixture no longer drives NaN into g");
    assert!(bn > 0, "fixture no longer drives NaN into beta");
    let finite_g = cu_g.iter().filter(|x| x.is_finite() && **x != 0.0).count();
    assert!(
        finite_g > cu_g.len() / 4,
        "fixture degenerated: only {finite_g} finite nonzero g lanes"
    );
}

#[test]
fn gdn_gating_bf16_nan_propagation_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gdn_gating_bf16_nan") else {
        return;
    };
    let (a, b, a_log, dt_bias) = nan_gating_inputs();
    let (cu_g, cu_beta, wg_g, wg_beta) = drive_gating_bf16(
        &stream,
        wg,
        &to_bf16(&a),
        &to_bf16(&b),
        &to_bf16(&a_log),
        &to_bf16(&dt_bias),
        NAN_TOKENS,
        NAN_HEADS,
    );
    let gn = assert_nan_pattern_and_bits("gating_bf16_nan g", &cu_g, &wg_g);
    let bn = assert_bf16_nan_pattern_and_bits("gating_bf16_nan beta", &cu_beta, &wg_beta);
    assert!(gn > 0, "bf16 fixture no longer drives NaN into g");
    assert!(bn > 0, "bf16 fixture no longer drives NaN into beta");
}

#[test]
fn gdn_gating_f32_two_dim_grid_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gdn_gating_f32_two_dim_grid") else {
        return;
    };
    let limit = wg.caps.max_compute_workgroups_per_dimension as usize;
    let block = gdn_gating::WORKGROUP_SIZE as usize;
    let num_heads = 8usize;
    let total = (limit + 1) * block;
    assert_eq!(total % num_heads, 0, "fixture must divide evenly by heads");
    assert!(
        total.div_ceil(block) > limit,
        "fixture stays inside one grid row: {} groups <= {limit}",
        total.div_ceil(block)
    );
    let bytes = (total * 4) as u64;
    assert!(
        bytes <= wg.caps.max_storage_buffer_binding_size,
        "device storage binding limit {} < {bytes}",
        wg.caps.max_storage_buffer_binding_size
    );
    let tokens = total / num_heads;
    eprintln!(
        "two_dim_grid: total={total} groups={} limit={limit}",
        total.div_ceil(block)
    );

    let mut a = Vec::with_capacity(total);
    let mut b = Vec::with_capacity(total);
    for i in 0..total {
        let u = ((i as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / (1u32 << 24) as f32;
        let v = ((i as u32).wrapping_mul(40_503) >> 8) as f32 / (1u32 << 24) as f32;
        a.push(u * 90.0 - 45.0);
        b.push(v * 60.0 - 30.0);
    }
    let a_log: Vec<f32> = (0..num_heads).map(|h| -3.5 + (h as f32) * 0.37).collect();
    let dt_bias: Vec<f32> = (0..num_heads)
        .map(|h| ((h as f32) * 0.913).sin() * 2.0 - 0.25)
        .collect();

    let (cu_g, cu_beta, wg_g, wg_beta) =
        drive_gating_f32(&stream, wg, &a, &b, &a_log, &dt_bias, tokens, num_heads);

    let tail = limit * block;
    let tail_nz = cu_g[tail..].iter().filter(|x| **x != 0.0).count();
    let tail_beta_nz = cu_beta[tail..].iter().filter(|x| **x != 0.0).count();
    eprintln!(
        "two_dim_grid second grid row: {tail_nz}/{} g nonzero, {tail_beta_nz} beta nonzero",
        total - tail
    );
    assert_eq!(
        total - tail,
        block,
        "second grid row should hold exactly one workgroup"
    );
    assert!(
        tail_nz == block && tail_beta_nz == block,
        "second grid row is not exercised by the reference"
    );
    assert_f32_ulp("two_dim_grid g", &cu_g, &wg_g, 0);
    assert_f32_ulp("two_dim_grid beta", &cu_beta, &wg_beta, 0);
}

fn drive_recurrent(
    stream: &Arc<CudaStream>,
    wg: &'static WgpuContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    b: usize,
    t: usize,
    h: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let vecs = b * t * h * DIM;
    #[allow(deprecated)]
    let dq: CudaSlice<f32> = stream.clone_htod(&q.to_vec()).unwrap();
    #[allow(deprecated)]
    let dk: CudaSlice<f32> = stream.clone_htod(&k.to_vec()).unwrap();
    #[allow(deprecated)]
    let dv: CudaSlice<f32> = stream.clone_htod(&v.to_vec()).unwrap();
    #[allow(deprecated)]
    let dg: CudaSlice<f32> = stream.clone_htod(&g.to_vec()).unwrap();
    #[allow(deprecated)]
    let dbeta: CudaSlice<f32> = stream.clone_htod(&beta.to_vec()).unwrap();
    let mut dout: CudaSlice<f32> = stream.alloc_zeros::<f32>(vecs).unwrap();
    let rc = {
        let (pq, _1) = dq.device_ptr(stream);
        let (pk, _2) = dk.device_ptr(stream);
        let (pv, _3) = dv.device_ptr(stream);
        let (pg, _4) = dg.device_ptr(stream);
        let (pb, _5) = dbeta.device_ptr(stream);
        let (po, _6) = dout.device_ptr_mut(stream);
        unsafe {
            cuda::gdn_recurrent_f32(
                stream.cu_stream() as *mut c_void,
                pq as *const f32,
                pk as *const f32,
                pv as *const f32,
                pg as *const f32,
                pb as *const f32,
                po as *mut f32,
                b as i32,
                t as i32,
                h as i32,
                DIM as i32,
                DIM as i32,
            )
        }
    };
    assert_eq!(rc, 0, "cuda gdn_recurrent_f32 rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let cu_out = stream.memcpy_dtov(&dout).unwrap();

    let mut wg_out = vec![0f32; vecs];
    let mut wg_state = vec![0f32; b * h * DIM * DIM];
    gdn_recurrent::gdn_recurrent_f32(wg, q, k, v, g, beta, &mut wg_out, &mut wg_state, b, h, t)
        .unwrap();
    (cu_out, wg_out, wg_state)
}

fn recurrent_nonfinite_inputs(
    b: usize,
    t: usize,
    h: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut q, mut k, mut v, mut g, mut beta) = recurrent_inputs(b, t, h);
    assert!(b >= 2 && t >= 4, "nonfinite fixture needs b>=2 and t>=4");
    let inf_batch = b - 2;
    let nan_batch = b - 1;
    for hi in 0..h {
        for ti in 0..t {
            let irow = (inf_batch * t + ti) * h + hi;
            let ibase = irow * DIM;
            if ti == t - 4 {
                v[ibase + 3] = f32::INFINITY;
                v[ibase + 61] = f32::NEG_INFINITY;
            }
            if ti == t - 1 {
                q[ibase + 17] = f32::INFINITY;
            }
            let nrow = (nan_batch * t + ti) * h + hi;
            let nbase = nrow * DIM;
            if ti == t - 2 && hi == 0 {
                g[nrow] = f32::INFINITY;
            }
            if ti == t - 1 {
                k[nbase + 5] = f32::from_bits(0x7fc0_0000);
                if hi + 1 == h {
                    beta[nrow] = f32::from_bits(0xffc0_0000);
                }
            }
        }
    }
    (q, k, v, g, beta)
}

#[test]
fn gdn_recurrent_f32_nonfinite_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gdn_recurrent_nonfinite") else {
        return;
    };
    for (b, t, h) in [(2usize, 9usize, 2usize), (3, 6, 1)] {
        let (q, k, v, g, beta) = recurrent_nonfinite_inputs(b, t, h);
        let (cu_out, wg_out, wg_state) =
            drive_recurrent(&stream, wg, &q, &k, &v, &g, &beta, b, t, h);
        let (ref_out, ref_state) = recurrent_oracle(&q, &k, &v, &g, &beta, b, t, h);

        let nan_lanes = assert_nan_pattern_and_bits(
            &format!("nonfinite out b={b} t={t} h={h}"),
            &cu_out,
            &wg_out,
        );
        assert_nan_pattern_and_bits(
            &format!("nonfinite oracle out b={b} t={t} h={h}"),
            &ref_out,
            &wg_out,
        );
        assert_nan_pattern_and_bits(
            &format!("nonfinite oracle state b={b} t={t} h={h}"),
            &ref_state,
            &wg_state,
        );
        let inf_lanes = cu_out.iter().filter(|x| x.is_infinite()).count();
        let finite_lanes = cu_out
            .iter()
            .filter(|x| x.is_finite() && **x != 0.0)
            .count();
        eprintln!(
            "nonfinite b={b} t={t} h={h}: {nan_lanes} nan, {inf_lanes} inf, {finite_lanes} finite-nonzero"
        );
        assert!(nan_lanes > 0, "fixture no longer produces NaN outputs");
        assert!(inf_lanes > 0, "fixture no longer produces infinite outputs");
        assert!(
            finite_lanes > cu_out.len() / 4,
            "fixture degenerated: only {finite_lanes} finite-nonzero lanes"
        );
    }
}
