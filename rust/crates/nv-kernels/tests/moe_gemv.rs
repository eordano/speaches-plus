#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn rand_bf16(rng: &mut SplitMix64, n: usize, scale: f32) -> Vec<bf16> {
    (0..n)
        .map(|_| bf16::from_f32(rng.next_f32() * scale))
        .collect()
}

fn dot_ref(w: &[bf16], x: &[f32]) -> f32 {
    w.iter().zip(x.iter()).map(|(a, b)| a.to_f32() * b).sum()
}

fn silu_bf16_ref(g: bf16) -> f32 {
    let gf = g.to_f32();
    gf / (1.0 + (-gf).exp())
}

fn swiglu_ref(
    gate: &[bf16],
    up: &[bf16],
    ids: &[i32],
    x: &[bf16],
    e: usize,
    inter: usize,
    hidden: usize,
) -> Vec<f32> {
    let xf: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let mut h = vec![0.0f32; ids.len() * inter];
    for (slot, &id) in ids.iter().enumerate() {
        if id < 0 || id as usize >= e {
            continue;
        }
        let base = id as usize * inter * hidden;
        for n in 0..inter {
            let row = &gate[base + n * hidden..base + (n + 1) * hidden];
            let urow = &up[base + n * hidden..base + (n + 1) * hidden];
            let g = bf16::from_f32(dot_ref(row, &xf));
            let u = bf16::from_f32(dot_ref(urow, &xf));
            let s = bf16::from_f32(silu_bf16_ref(g));
            h[slot * inter + n] = (s * u).to_f32();
        }
    }
    h
}

fn down_tail_ref(
    down: &[bf16],
    ids: &[i32],
    weights: &[f32],
    h: &[bf16],
    shared: &[f32],
    resid: &[bf16],
    e: usize,
    hidden: usize,
    inter: usize,
) -> Vec<f32> {
    let hf: Vec<f32> = h.iter().map(|v| v.to_f32()).collect();
    let mut out = vec![0.0f32; hidden];
    for n in 0..hidden {
        let mut acc = 0.0f32;
        for (slot, &id) in ids.iter().enumerate() {
            if id < 0 || id as usize >= e {
                continue;
            }
            let base = id as usize * hidden * inter + n * inter;
            let row = &down[base..base + inter];
            let a: f32 = row
                .iter()
                .zip(hf[slot * inter..(slot + 1) * inter].iter())
                .map(|(w, v)| w.to_f32() * v)
                .sum();
            acc += weights[slot] * bf16::from_f32(a).to_f32();
        }
        let t = acc + shared[n];
        let fb = bf16::from_f32(t);
        out[n] = bf16::from_f32(resid[n].to_f32() + fb.to_f32()).to_f32();
    }
    out
}

fn to_u16(v: &[bf16]) -> &[u16] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u16, v.len()) }
}

fn assert_close(got: f32, want: f32, tol: f32, ctx: &str) {
    let err = (got - want).abs();
    let denom = want.abs().max(1.0);
    assert!(
        err / denom < tol,
        "{ctx}: got {got} want {want} (rel err {})",
        err / denom
    );
}

fn run_case(e: usize, k: usize, inter: usize, hidden: usize, ids: Vec<i32>, seed: u64) {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "moe_gemv: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("moe_gemv: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let mut rng = SplitMix64(seed);
    let scale = 1.0 / (hidden as f32).sqrt();

    let gate = rand_bf16(&mut rng, e * inter * hidden, scale);
    let up = rand_bf16(&mut rng, e * inter * hidden, scale);
    let down = rand_bf16(&mut rng, e * hidden * inter, scale);
    let x = rand_bf16(&mut rng, hidden, 1.0);
    let resid = rand_bf16(&mut rng, hidden, 1.0);
    let shared: Vec<f32> = (0..hidden).map(|_| rng.next_f32()).collect();
    let weights: Vec<f32> = (0..k).map(|_| rng.next_f32().abs()).collect();
    assert_eq!(ids.len(), k);

    #[allow(deprecated)]
    let gate_d = stream.memcpy_stod(to_u16(&gate)).unwrap();
    #[allow(deprecated)]
    let up_d = stream.memcpy_stod(to_u16(&up)).unwrap();
    #[allow(deprecated)]
    let down_d = stream.memcpy_stod(to_u16(&down)).unwrap();
    #[allow(deprecated)]
    let x_d = stream.memcpy_stod(to_u16(&x)).unwrap();
    #[allow(deprecated)]
    let resid_d = stream.memcpy_stod(to_u16(&resid)).unwrap();
    #[allow(deprecated)]
    let shared_d = stream.memcpy_stod(&shared).unwrap();
    #[allow(deprecated)]
    let weights_d = stream.memcpy_stod(&weights).unwrap();
    #[allow(deprecated)]
    let ids_d = stream.memcpy_stod(&ids).unwrap();
    let mut h_d = stream.alloc_zeros::<u16>(k * inter).unwrap();
    let mut out_d = stream.alloc_zeros::<u16>(hidden).unwrap();

    let rc = {
        let (gp, _g1) = gate_d.device_ptr(&stream);
        let (upp, _g2) = up_d.device_ptr(&stream);
        let (ip, _g3) = ids_d.device_ptr(&stream);
        let (xp, _g4) = x_d.device_ptr(&stream);
        let (hp, _g5) = h_d.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::moe_gemv_swiglu_bf16_m1(
                stream.cu_stream() as *mut c_void,
                gp as *const u16,
                upp as *const u16,
                ip as *const i32,
                xp as *const u16,
                hp as *mut u16,
                k as i32,
                e as i32,
                inter as i32,
                hidden as i32,
            )
        }
    };
    assert_eq!(rc, 0, "moe_gemv_swiglu rc={rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let h_raw = stream.memcpy_dtov(&h_d).unwrap();
    let h_gpu: Vec<bf16> = h_raw.iter().map(|&v| bf16::from_bits(v)).collect();
    let h_ref = swiglu_ref(&gate, &up, &ids, &x, e, inter, hidden);
    for slot in 0..k {
        for n in 0..inter {
            let i = slot * inter + n;
            assert_close(
                h_gpu[i].to_f32(),
                bf16::from_f32(h_ref[i]).to_f32(),
                3e-2,
                &format!("h slot {slot} n {n}"),
            );
        }
    }

    let rc = {
        let (dp, _g1) = down_d.device_ptr(&stream);
        let (ip, _g2) = ids_d.device_ptr(&stream);
        let (wp, _g3) = weights_d.device_ptr(&stream);
        let (hp, _g4) = h_d.device_ptr(&stream);
        let (sp, _g5) = shared_d.device_ptr(&stream);
        let (rp, _g6) = resid_d.device_ptr(&stream);
        let (op, _g7) = out_d.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::moe_gemv_down_tail_bf16_m1(
                stream.cu_stream() as *mut c_void,
                dp as *const u16,
                ip as *const i32,
                wp as *const f32,
                hp as *const u16,
                sp as *const f32,
                rp as *const u16,
                op as *mut u16,
                k as i32,
                e as i32,
                hidden as i32,
                inter as i32,
            )
        }
    };
    assert_eq!(rc, 0, "moe_gemv_down_tail rc={rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let out_raw = stream.memcpy_dtov(&out_d).unwrap();
    let out_gpu: Vec<f32> = out_raw
        .iter()
        .map(|&v| bf16::from_bits(v).to_f32())
        .collect();
    let out_ref = down_tail_ref(
        &down, &ids, &weights, &h_gpu, &shared, &resid, e, hidden, inter,
    );
    for n in 0..hidden {
        assert_close(out_gpu[n], out_ref[n], 3e-2, &format!("out n {n}"));
    }
}

#[test]
#[ignore = "GPU bandwidth bench; run explicitly with --ignored"]
fn moe_gemv_dsocr_shape_bandwidth() {
    let ctx = CudaContext::new(0).unwrap_or_else(|e| {
        panic!(
            "moe_gemv_dsocr_shape_bandwidth: no CUDA device 0: {e}. This bench measures \
             nothing without one and must not report success instead."
        )
    });
    let stream = ctx.default_stream();
    let (e, k, inter, hidden) = (64usize, 6usize, 896usize, 1280usize);
    let mut rng = SplitMix64(0xbead);
    let scale = 1.0 / (hidden as f32).sqrt();
    let gate = rand_bf16(&mut rng, e * inter * hidden, scale);
    let up = rand_bf16(&mut rng, e * inter * hidden, scale);
    let down = rand_bf16(&mut rng, e * hidden * inter, scale);
    let x = rand_bf16(&mut rng, hidden, 1.0);
    let resid = rand_bf16(&mut rng, hidden, 1.0);
    let shared: Vec<f32> = (0..hidden).map(|_| rng.next_f32()).collect();
    let weights: Vec<f32> = (0..k).map(|_| rng.next_f32().abs()).collect();
    let ids: Vec<i32> = vec![3, 12, 25, 38, 51, 63];

    #[allow(deprecated)]
    let gate_d = stream.memcpy_stod(to_u16(&gate)).unwrap();
    #[allow(deprecated)]
    let up_d = stream.memcpy_stod(to_u16(&up)).unwrap();
    #[allow(deprecated)]
    let down_d = stream.memcpy_stod(to_u16(&down)).unwrap();
    #[allow(deprecated)]
    let x_d = stream.memcpy_stod(to_u16(&x)).unwrap();
    #[allow(deprecated)]
    let resid_d = stream.memcpy_stod(to_u16(&resid)).unwrap();
    #[allow(deprecated)]
    let shared_d = stream.memcpy_stod(&shared).unwrap();
    #[allow(deprecated)]
    let weights_d = stream.memcpy_stod(&weights).unwrap();
    #[allow(deprecated)]
    let ids_d = stream.memcpy_stod(&ids).unwrap();
    let mut h_d = stream.alloc_zeros::<u16>(k * inter).unwrap();
    let mut out_d = stream.alloc_zeros::<u16>(hidden).unwrap();

    let iters = 2000usize;
    let mut launch = || {
        let (gp, _g1) = gate_d.device_ptr(&stream);
        let (upp, _g2) = up_d.device_ptr(&stream);
        let (ip, _g3) = ids_d.device_ptr(&stream);
        let (xp, _g4) = x_d.device_ptr(&stream);
        let rc = {
            let (hp, _g5) = h_d.device_ptr_mut(&stream);
            unsafe {
                nv_kernels::cuda::moe_gemv_swiglu_bf16_m1(
                    stream.cu_stream() as *mut c_void,
                    gp as *const u16,
                    upp as *const u16,
                    ip as *const i32,
                    xp as *const u16,
                    hp as *mut u16,
                    k as i32,
                    e as i32,
                    inter as i32,
                    hidden as i32,
                )
            }
        };
        assert_eq!(rc, 0);
        let (dp, _g6) = down_d.device_ptr(&stream);
        let (wp, _g7) = weights_d.device_ptr(&stream);
        let (hp2, _g8) = h_d.device_ptr(&stream);
        let (sp, _g9) = shared_d.device_ptr(&stream);
        let (rp, _g10) = resid_d.device_ptr(&stream);
        let (op, _g11) = out_d.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::moe_gemv_down_tail_bf16_m1(
                stream.cu_stream() as *mut c_void,
                dp as *const u16,
                ip as *const i32,
                wp as *const f32,
                hp2 as *const u16,
                sp as *const f32,
                rp as *const u16,
                op as *mut u16,
                k as i32,
                e as i32,
                hidden as i32,
                inter as i32,
            )
        };
        assert_eq!(rc, 0);
    };
    for _ in 0..50 {
        launch();
    }
    stream.synchronize().unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        launch();
    }
    stream.synchronize().unwrap();
    let dt = t0.elapsed().as_secs_f64();
    let bytes_per_iter = (2 * k * inter * hidden + k * hidden * inter) as f64 * 2.0;
    let gbps = bytes_per_iter * iters as f64 / dt / 1e9;
    eprintln!(
        "[moe_gemv bench] {iters} iters in {:.3}s -> {:.1} us/layer-step, {:.0} GB/s (weights {:.2} MB/iter)",
        dt,
        dt / iters as f64 * 1e6,
        gbps,
        bytes_per_iter / 1e6
    );
}

#[test]
fn moe_gemv_small_edge_ids() {
    run_case(8, 4, 64, 96, vec![0, 7, 3, 7], 0x1234);
}

#[test]
fn moe_gemv_invalid_ids_zero() {
    run_case(8, 4, 64, 96, vec![0, -1, 8, 5], 0x5678);
}

#[test]
fn moe_gemv_dsocr_shape() {
    run_case(64, 6, 896, 1280, vec![0, 63, 17, 5, 42, 31], 0x9abc);
}
