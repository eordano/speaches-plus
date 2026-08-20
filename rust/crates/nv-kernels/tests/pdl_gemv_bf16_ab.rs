#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;
mod common;
use common::LcgShift40Top24TwoSided as Lcg;

fn bench_chain(stream: &Arc<CudaStream>, d: usize, chain: usize, reps: usize) -> f64 {
    let (n, k) = (d, d);
    let mut rng = Lcg(0x243f6a8885a308d3);
    let gain = 1.0 / (k as f32).sqrt();
    let w: Vec<u16> = (0..n * k)
        .map(|_| bf16::from_f32(rng.next_f32() * gain).to_bits())
        .collect();
    let x: Vec<u16> = (0..k)
        .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
        .collect();

    #[allow(deprecated)]
    let dw: CudaSlice<u16> = stream.clone_htod(&w).unwrap();
    #[allow(deprecated)]
    let mut da: CudaSlice<u16> = stream.clone_htod(&x).unwrap();
    let mut db: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();

    let pw = {
        let (p, _g) = dw.device_ptr(stream);
        p
    };
    let pa = {
        let (p, _g) = da.device_ptr_mut(stream);
        p
    };
    let pb = {
        let (p, _g) = db.device_ptr_mut(stream);
        p
    };

    let run = || {
        let (mut src, mut dst) = (pa, pb);
        for _ in 0..chain {
            let rc = unsafe {
                cuda::gemv_bf16(
                    stream.cu_stream() as *mut c_void,
                    pw as *const u16,
                    src as *const u16,
                    dst as *mut u16,
                    n as i32,
                    k as i32,
                )
            };
            assert_eq!(rc, 0, "gemv_bf16 rc={rc} n={n} k={k}");
            std::mem::swap(&mut src, &mut dst);
        }
        stream.synchronize().unwrap();
    };

    run();

    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = std::time::Instant::now();
        run();
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[reps / 2]
}

fn bench_norm_gemv_chain(stream: &Arc<CudaStream>, d: usize, chain: usize, reps: usize) -> f64 {
    let mut rng = Lcg(0x9e3779b97f4a7c15);
    let gain = 1.0 / (d as f32).sqrt();
    let w: Vec<u16> = (0..d * d)
        .map(|_| bf16::from_f32(rng.next_f32() * gain).to_bits())
        .collect();
    let nw: Vec<u16> = (0..d).map(|_| bf16::from_f32(1.0).to_bits()).collect();
    let x: Vec<u16> = (0..d)
        .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
        .collect();

    #[allow(deprecated)]
    let dw: CudaSlice<u16> = stream.clone_htod(&w).unwrap();
    #[allow(deprecated)]
    let dnw: CudaSlice<u16> = stream.clone_htod(&nw).unwrap();
    #[allow(deprecated)]
    let mut da: CudaSlice<u16> = stream.clone_htod(&x).unwrap();
    let mut db: CudaSlice<u16> = stream.alloc_zeros::<u16>(d).unwrap();

    let pw = {
        let (p, _g) = dw.device_ptr(stream);
        p
    };
    let pnw = {
        let (p, _g) = dnw.device_ptr(stream);
        p
    };
    let pa = {
        let (p, _g) = da.device_ptr_mut(stream);
        p
    };
    let pb = {
        let (p, _g) = db.device_ptr_mut(stream);
        p
    };

    let run = || {
        let sp = stream.cu_stream() as *mut c_void;
        for _ in 0..chain {
            let rc = unsafe {
                cuda::rmsnorm_bf16(
                    sp,
                    pa as *const u16,
                    pnw as *const u16,
                    pb as *mut u16,
                    1,
                    d,
                    1e-6,
                )
            };
            assert_eq!(rc, 0, "rmsnorm_bf16 rc={rc} d={d}");
            let rc = unsafe {
                cuda::gemv_bf16(
                    sp,
                    pw as *const u16,
                    pb as *const u16,
                    pa as *mut u16,
                    d as i32,
                    d as i32,
                )
            };
            assert_eq!(rc, 0, "gemv_bf16 rc={rc} d={d}");
        }
        stream.synchronize().unwrap();
    };

    run();

    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = std::time::Instant::now();
        run();
        samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[reps / 2]
}

#[test]
fn pdl_norm_gemv_chain_matches_cpu() {
    let ctx = CudaContext::new(0).expect("no CUDA device 0");
    let stream = ctx.default_stream();
    let d = 4096usize;
    let chain = 2usize;
    let eps = 1e-6f32;

    let mut rng = Lcg(0xdeadbeefcafef00d);
    let gain = 1.0 / (d as f32).sqrt();
    let w: Vec<u16> = (0..d * d)
        .map(|_| bf16::from_f32(rng.next_f32() * gain).to_bits())
        .collect();
    let nw: Vec<u16> = (0..d)
        .map(|_| bf16::from_f32(0.5 + rng.next_f32() * 0.25).to_bits())
        .collect();
    let x: Vec<u16> = (0..d)
        .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
        .collect();

    #[allow(deprecated)]
    let dw: CudaSlice<u16> = stream.clone_htod(&w).unwrap();
    #[allow(deprecated)]
    let dnw: CudaSlice<u16> = stream.clone_htod(&nw).unwrap();
    #[allow(deprecated)]
    let mut da: CudaSlice<u16> = stream.clone_htod(&x).unwrap();
    let mut db: CudaSlice<u16> = stream.alloc_zeros::<u16>(d).unwrap();

    let pw = {
        let (p, _g) = dw.device_ptr(&stream);
        p
    };
    let pnw = {
        let (p, _g) = dnw.device_ptr(&stream);
        p
    };
    let pa = {
        let (p, _g) = da.device_ptr_mut(&stream);
        p
    };
    let pb = {
        let (p, _g) = db.device_ptr_mut(&stream);
        p
    };

    let sp = stream.cu_stream() as *mut c_void;
    for _ in 0..chain {
        let rc = unsafe {
            cuda::rmsnorm_bf16(
                sp,
                pa as *const u16,
                pnw as *const u16,
                pb as *mut u16,
                1,
                d,
                eps,
            )
        };
        assert_eq!(rc, 0);
        let rc = unsafe {
            cuda::gemv_bf16(
                sp,
                pw as *const u16,
                pb as *const u16,
                pa as *mut u16,
                d as i32,
                d as i32,
            )
        };
        assert_eq!(rc, 0);
    }
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let got = stream.memcpy_dtov(&da).unwrap();

    let mut cur: Vec<f32> = x.iter().map(|b| bf16::from_bits(*b).to_f32()).collect();
    for _ in 0..chain {
        let ms: f32 = cur.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let r = 1.0 / (ms + eps).sqrt();
        let normed: Vec<f32> = (0..d)
            .map(|i| bf16::from_f32(cur[i] * r * bf16::from_bits(nw[i]).to_f32()).to_f32())
            .collect();
        cur = (0..d)
            .map(|row| {
                let acc: f32 = (0..d)
                    .map(|j| bf16::from_bits(w[row * d + j]).to_f32() * normed[j])
                    .sum();
                bf16::from_f32(acc).to_f32()
            })
            .collect();
    }

    let scale = (cur.iter().map(|v| v * v).sum::<f32>() / d as f32)
        .sqrt()
        .max(1e-6);
    let mut worst = 0.0f32;
    for i in 0..d {
        let g = bf16::from_bits(got[i]).to_f32();
        let e = (g - cur[i]).abs() / scale;
        worst = worst.max(e);
        assert!(
            e <= 0.05,
            "pdl chain diverged at {i}: gpu {g} vs cpu {} ({e} of rms {scale})",
            cur[i]
        );
    }
    eprintln!(
        "pdl_norm_gemv_chain_matches_cpu OK (NV_PDL={}, worst rel {worst:.4})",
        std::env::var("NV_PDL").unwrap_or_else(|_| "unset".to_string())
    );
}

#[test]
#[ignore]
fn pdl_norm_gemv_chain_ab() {
    let ctx = CudaContext::new(0).expect("no CUDA device 0");
    let stream = ctx.default_stream();
    let pdl = std::env::var("NV_PDL").unwrap_or_else(|_| "unset".to_string());
    let chain = 200usize;
    let reps = 9usize;

    for &d in &[2048usize, 5376usize] {
        let ms = bench_norm_gemv_chain(&stream, d, chain, reps);
        println!(
            "NORMGEMV NV_PDL={pdl} d={d} chain={chain} reps={reps} \
             median_ms={ms:.4} per_pair_us={:.3}",
            ms * 1e3 / chain as f64
        );
    }
}

#[test]
#[ignore]
fn pdl_gemv_bf16_chain_ab() {
    let ctx = CudaContext::new(0).expect("no CUDA device 0");
    let stream = ctx.default_stream();
    let pdl = std::env::var("NV_PDL").unwrap_or_else(|_| "unset".to_string());
    let chain = 200usize;
    let reps = 9usize;

    for &d in &[2048usize, 5376usize, 2050usize] {
        let branch = if d % 8 != 0 {
            "scalar"
        } else if d <= 4096 {
            "shared"
        } else {
            "nonshared"
        };
        let ms = bench_chain(&stream, d, chain, reps);
        println!(
            "NV_PDL={pdl} d={d} branch={branch} chain={chain} reps={reps} \
             median_ms={ms:.4} per_launch_us={:.3}",
            ms * 1e3 / chain as f64
        );
    }
}

fn time_body(
    stream: &Arc<CudaStream>,
    use_graph: bool,
    reps: usize,
    body: &dyn Fn(&Arc<CudaStream>),
) -> f64 {
    let mut samples = Vec::with_capacity(reps);
    if use_graph {
        let mut runner = nv_kernels::graph::CudaGraphRunner::new(stream.clone());
        runner
            .run(1u64, |s| {
                body(s);
                Ok(())
            })
            .expect("graph capture");
        stream.synchronize().unwrap();
        runner.run(1u64, |_| Ok(())).expect("graph replay warmup");
        stream.synchronize().unwrap();
        for _ in 0..reps {
            let t = std::time::Instant::now();
            runner.run(1u64, |_| Ok(())).expect("graph replay");
            stream.synchronize().unwrap();
            samples.push(t.elapsed().as_secs_f64() * 1e3);
        }
    } else {
        body(stream);
        stream.synchronize().unwrap();
        for _ in 0..reps {
            let t = std::time::Instant::now();
            body(stream);
            stream.synchronize().unwrap();
            samples.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[reps / 2]
}

fn bench_norm_residual_chain(
    stream: &Arc<CudaStream>,
    d: usize,
    chain: usize,
    reps: usize,
    use_graph: bool,
) -> f64 {
    let mut rng = Lcg(0x5851f42d4c957f2d);
    let nw: Vec<u16> = (0..d).map(|_| bf16::from_f32(1.0).to_bits()).collect();
    let x: Vec<u16> = (0..d)
        .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
        .collect();
    let res: Vec<u16> = (0..d)
        .map(|_| bf16::from_f32(rng.next_f32() * 0.1).to_bits())
        .collect();

    #[allow(deprecated)]
    let dnw: CudaSlice<u16> = stream.clone_htod(&nw).unwrap();
    #[allow(deprecated)]
    let mut dx: CudaSlice<u16> = stream.clone_htod(&x).unwrap();
    #[allow(deprecated)]
    let dres: CudaSlice<u16> = stream.clone_htod(&res).unwrap();
    let mut dtmp: CudaSlice<u16> = stream.alloc_zeros::<u16>(d).unwrap();

    let pnw = {
        let (p, _g) = dnw.device_ptr(stream);
        p
    };
    let pres = {
        let (p, _g) = dres.device_ptr(stream);
        p
    };
    let px = {
        let (p, _g) = dx.device_ptr_mut(stream);
        p
    };
    let ptmp = {
        let (p, _g) = dtmp.device_ptr_mut(stream);
        p
    };

    let body = |s: &Arc<CudaStream>| {
        let sp = s.cu_stream() as *mut c_void;
        for _ in 0..chain {
            let rc = unsafe {
                cuda::rmsnorm_bf16(
                    sp,
                    px as *const u16,
                    pnw as *const u16,
                    ptmp as *mut u16,
                    1,
                    d,
                    1e-6,
                )
            };
            assert_eq!(rc, 0, "rmsnorm_bf16 rc={rc}");
            let rc = unsafe {
                cuda::residual_add_scale_bf16(
                    sp,
                    pres as *const u16,
                    ptmp as *const u16,
                    px as *mut u16,
                    1.0,
                    d,
                )
            };
            assert_eq!(rc, 0, "residual_add_scale_bf16 rc={rc}");
        }
    };

    time_body(stream, use_graph, reps, &body)
}

fn bench_scale_out_chain(
    stream: &Arc<CudaStream>,
    d: usize,
    chain: usize,
    reps: usize,
    use_graph: bool,
) -> f64 {
    let mut rng = Lcg(0x14057b7ef767814f);
    let x: Vec<u16> = (0..d)
        .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
        .collect();
    #[allow(deprecated)]
    let mut da: CudaSlice<u16> = stream.clone_htod(&x).unwrap();
    let mut db: CudaSlice<u16> = stream.alloc_zeros::<u16>(d).unwrap();
    let pa = {
        let (p, _g) = da.device_ptr_mut(stream);
        p
    };
    let pb = {
        let (p, _g) = db.device_ptr_mut(stream);
        p
    };

    let body = |s: &Arc<CudaStream>| {
        let sp = s.cu_stream() as *mut c_void;
        let (mut src, mut dst) = (pa, pb);
        for _ in 0..chain {
            let rc =
                unsafe { cuda::scale_out_bf16(sp, src as *const u16, dst as *mut u16, 1.0, d) };
            assert_eq!(rc, 0, "scale_out_bf16 rc={rc}");
            std::mem::swap(&mut src, &mut dst);
        }
    };

    time_body(stream, use_graph, reps, &body)
}

#[allow(clippy::too_many_arguments)]
fn bench_attn_tail_chain(
    stream: &Arc<CudaStream>,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    hidden: usize,
    ctx: usize,
    chain: usize,
    reps: usize,
    use_graph: bool,
) -> f64 {
    let qdim = n_q * head_dim;
    let kdim = n_kv * head_dim;
    let half = head_dim / 2;
    let ring = ctx;
    let splits_max = 32usize;

    let mut rng = Lcg(0x2545f4914f6cdd1d);
    let q_in: Vec<u16> = (0..qdim)
        .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
        .collect();
    let v_in: Vec<u16> = (0..kdim)
        .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
        .collect();
    let vnw: Vec<u16> = (0..head_dim)
        .map(|_| bf16::from_f32(1.0).to_bits())
        .collect();
    let cos_t: Vec<f32> = (0..ctx * half)
        .map(|i| ((i % 97) as f32 * 0.01).cos())
        .collect();
    let sin_t: Vec<f32> = (0..ctx * half)
        .map(|i| ((i % 97) as f32 * 0.01).sin())
        .collect();
    let gain = 1.0 / (qdim as f32).sqrt();
    let wo: Vec<u16> = (0..hidden * qdim)
        .map(|_| bf16::from_f32(rng.next_f32() * gain).to_bits())
        .collect();

    #[allow(deprecated)]
    let dq_in: CudaSlice<u16> = stream.clone_htod(&q_in).unwrap();
    #[allow(deprecated)]
    let dv_in: CudaSlice<u16> = stream.clone_htod(&v_in).unwrap();
    #[allow(deprecated)]
    let dvnw: CudaSlice<u16> = stream.clone_htod(&vnw).unwrap();
    #[allow(deprecated)]
    let dcos: CudaSlice<f32> = stream.clone_htod(&cos_t).unwrap();
    #[allow(deprecated)]
    let dsin: CudaSlice<f32> = stream.clone_htod(&sin_t).unwrap();
    #[allow(deprecated)]
    let dpos: CudaSlice<i32> = stream.clone_htod(&vec![0i32]).unwrap();
    #[allow(deprecated)]
    let dstart: CudaSlice<i32> = stream.clone_htod(&vec![0i32]).unwrap();
    #[allow(deprecated)]
    let dntot: CudaSlice<i32> = stream.clone_htod(&vec![ctx as i32]).unwrap();
    #[allow(deprecated)]
    let dwo: CudaSlice<u16> = stream.clone_htod(&wo).unwrap();

    let mut dv_normed: CudaSlice<u16> = stream.alloc_zeros::<u16>(kdim).unwrap();
    let mut dq_rot: CudaSlice<u16> = stream.alloc_zeros::<u16>(qdim).unwrap();
    let mut dk_rot: CudaSlice<u16> = stream.alloc_zeros::<u16>(kdim).unwrap();
    let mut dk_fp8: CudaSlice<u8> = stream.alloc_zeros::<u8>(ring * kdim).unwrap();
    let mut dv_fp8: CudaSlice<u8> = stream.alloc_zeros::<u8>(ring * kdim).unwrap();
    let mut dk_sc: CudaSlice<f32> = stream.alloc_zeros::<f32>(ring * n_kv).unwrap();
    let mut dv_sc: CudaSlice<f32> = stream.alloc_zeros::<f32>(ring * n_kv).unwrap();
    let mut dscratch: CudaSlice<f32> = stream
        .alloc_zeros::<f32>(n_q * splits_max * (head_dim + 2))
        .unwrap();
    let mut dfan: CudaSlice<u32> = stream.alloc_zeros::<u32>(n_q).unwrap();
    let mut dattn: CudaSlice<u16> = stream.alloc_zeros::<u16>(qdim).unwrap();
    let mut dhid: CudaSlice<u16> = stream.alloc_zeros::<u16>(hidden).unwrap();

    macro_rules! pc {
        ($b:expr) => {{
            let (p, _g) = $b.device_ptr(stream);
            p
        }};
    }
    macro_rules! pm {
        ($b:expr) => {{
            let (p, _g) = $b.device_ptr_mut(stream);
            p
        }};
    }

    let pq_in = pc!(dq_in);
    let pv_in = pc!(dv_in);
    let pvnw = pc!(dvnw);
    let pcos = pc!(dcos);
    let psin = pc!(dsin);
    let ppos = pc!(dpos);
    let pstart = pc!(dstart);
    let pntot = pc!(dntot);
    let pwo = pc!(dwo);
    let pv_normed = pm!(dv_normed);
    let pq_rot = pm!(dq_rot);
    let pk_rot = pm!(dk_rot);
    let pk_fp8 = pm!(dk_fp8);
    let pv_fp8 = pm!(dv_fp8);
    let pk_sc = pm!(dk_sc);
    let pv_sc = pm!(dv_sc);
    let pscratch = pm!(dscratch);
    let pfan = pm!(dfan);
    let pattn = pm!(dattn);
    let phid = pm!(dhid);

    let scaling = 1.0f32 / (head_dim as f32).sqrt();

    let body = |s: &Arc<CudaStream>| {
        let sp = s.cu_stream() as *mut c_void;
        for _ in 0..chain {
            let rc = unsafe {
                cuda::rmsnorm_bf16(
                    sp,
                    pv_in as *const u16,
                    pvnw as *const u16,
                    pv_normed as *mut u16,
                    n_kv,
                    head_dim,
                    1e-6,
                )
            };
            assert_eq!(rc, 0, "rmsnorm_bf16(v_norm) rc={rc}");
            let rc = unsafe {
                cuda::rope_bf16_oop(
                    sp,
                    pq_in as *const u16,
                    pv_normed as *const u16,
                    pq_rot as *mut u16,
                    pk_rot as *mut u16,
                    pcos as *const f32,
                    psin as *const f32,
                    ppos as *const i32,
                    1,
                    n_q,
                    n_kv,
                    head_dim,
                )
            };
            assert_eq!(rc, 0, "rope_bf16_oop rc={rc}");
            let rc = unsafe {
                cuda::quantize_kv_fp8(
                    sp,
                    pk_rot as *const u16,
                    pk_fp8 as *mut u8,
                    pk_sc as *mut f32,
                    pstart as *const i32,
                    1,
                    n_kv as i32,
                    head_dim as i32,
                    ring as i32,
                )
            };
            assert_eq!(rc, 0, "quantize_kv_fp8(k) rc={rc}");
            let rc = unsafe {
                cuda::quantize_kv_fp8(
                    sp,
                    pv_normed as *const u16,
                    pv_fp8 as *mut u8,
                    pv_sc as *mut f32,
                    pstart as *const i32,
                    1,
                    n_kv as i32,
                    head_dim as i32,
                    ring as i32,
                )
            };
            assert_eq!(rc, 0, "quantize_kv_fp8(v) rc={rc}");
            let rc = unsafe {
                cuda::flash_decode_fused_fp8kv(
                    sp,
                    pq_rot as *const u16,
                    pk_fp8 as *const u8,
                    pv_fp8 as *const u8,
                    pk_sc as *const f32,
                    pv_sc as *const f32,
                    pattn as *mut u16,
                    pntot as *const i32,
                    pscratch as *mut f32,
                    pfan as *mut u32,
                    n_q as i32,
                    n_kv as i32,
                    head_dim as i32,
                    ctx as i32,
                    ring as i32,
                    scaling,
                )
            };
            assert_eq!(rc, 0, "flash_decode_fused_fp8kv rc={rc}");
            let rc = unsafe {
                cuda::gemv_bf16(
                    sp,
                    pwo as *const u16,
                    pattn as *const u16,
                    phid as *mut u16,
                    hidden as i32,
                    qdim as i32,
                )
            };
            assert_eq!(rc, 0, "gemv_bf16(o_proj) rc={rc}");
        }
    };

    time_body(stream, use_graph, reps, &body)
}

#[test]
#[ignore]
fn pdl_norm_residual_chain_ab() {
    let ctx = CudaContext::new(0).expect("no CUDA device 0");
    let stream = ctx.new_stream().expect("new_stream");
    let pdl = std::env::var("NV_PDL").unwrap_or_else(|_| "unset".to_string());
    let chain = 200usize;
    let reps = 9usize;

    for &d in &[5376usize] {
        for &(mode, g) in &[("stream", false), ("graph", true)] {
            let ms = bench_norm_residual_chain(&stream, d, chain, reps, g);
            println!(
                "NORMRESID mode={mode} NV_PDL={pdl} d={d} chain={chain} reps={reps} \
                 median_ms={ms:.4} per_iter_us={:.3}",
                ms * 1e3 / chain as f64
            );
            let ctl = bench_scale_out_chain(&stream, d, chain, reps, g);
            println!(
                "CONTROL_SCALEOUT mode={mode} NV_PDL={pdl} d={d} chain={chain} reps={reps} \
                 median_ms={ctl:.4} per_launch_us={:.3}",
                ctl * 1e3 / chain as f64
            );
        }
    }
}

#[test]
#[ignore]
fn pdl_attn_tail_chain_ab() {
    let cx = CudaContext::new(0).expect("no CUDA device 0");
    let stream = cx.new_stream().expect("new_stream");
    let pdl = std::env::var("NV_PDL").unwrap_or_else(|_| "unset".to_string());
    let chain = 50usize;
    let reps = 9usize;

    for &(mode, g) in &[("stream", false), ("graph", true)] {
        let ms = bench_attn_tail_chain(&stream, 32, 16, 256, 5376, 1024, chain, reps, g);
        println!(
            "ATTNTAIL mode={mode} NV_PDL={pdl} n_q=32 n_kv=16 hd=256 hidden=5376 ctx=1024 \
             chain={chain} reps={reps} median_ms={ms:.4} per_iter_us={:.3}",
            ms * 1e3 / chain as f64
        );
    }
}
