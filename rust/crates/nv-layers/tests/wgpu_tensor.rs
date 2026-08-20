#![cfg(feature = "wgpu")]

use candle_core::{DType, Device, Tensor};
use half::{bf16, f16};
use nv_kernels::wgpu_backend::compose;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::dispatch::{self, Chain, GpuBind, GpuUniform};
use nv_kernels::wgpu_backend::kernels::{rmsnorm, silu};
use nv_layers::backend::wgpu_tensor::{ResidencyCache, WgpuTensor};
use nv_layers::backend::{Backend, BackendError, BackendKind};

#[path = "wgpu_common.rs"]
mod wgpu_common;

use wgpu_common::wgpu_allow_skip;
use wgpu_common::wgpu_ctx_or_skip as ctx_or_skip;

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

fn vals(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.0007 + phase).sin() * 3.0)
        .collect()
}

#[test]
fn candle_round_trip_is_bitwise() {
    let Some(ctx) = ctx_or_skip("candle_round_trip_is_bitwise") else {
        return;
    };
    let dev = Device::Cpu;

    let mut state = 0xDEAD_BEEFu32;
    let mut f32_bits: Vec<u32> = vec![
        0x7FC0_0001,
        0x7F80_0001,
        0x7F80_0000,
        0xFF80_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x0000_0000,
    ];
    while f32_bits.len() < 37 * 113 {
        f32_bits.push(lcg(&mut state));
    }
    let f32_vals: Vec<f32> = f32_bits.iter().map(|b| f32::from_bits(*b)).collect();
    let t = Tensor::from_vec(f32_vals, (37, 113), &dev).unwrap();
    let g = WgpuTensor::from_candle(ctx, "rt-f32", &t).unwrap();
    assert_eq!(g.dtype(), DType::F32);
    assert_eq!(g.shape().dims(), &[37, 113]);
    let back = g.to_candle(ctx, &dev).unwrap();
    assert_eq!(back.dtype(), DType::F32);
    assert_eq!(back.shape().dims(), &[37, 113]);
    let got: Vec<f32> = back.flatten_all().unwrap().to_vec1().unwrap();
    for (i, (b, v)) in f32_bits.iter().zip(got.iter()).enumerate() {
        assert_eq!(*b, v.to_bits(), "f32 idx {i}");
    }

    let all_bf16: Vec<bf16> = (0u16..=u16::MAX).map(bf16::from_bits).collect();
    let t = Tensor::from_vec(all_bf16, (256, 256), &dev).unwrap();
    let g = WgpuTensor::from_candle(ctx, "rt-bf16", &t).unwrap();
    let back = g.to_candle(ctx, &dev).unwrap();
    assert_eq!(back.dtype(), DType::BF16);
    let got: Vec<bf16> = back.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(got.len(), 65536);
    for (i, v) in got.iter().enumerate() {
        assert_eq!(v.to_bits(), i as u16, "bf16 bit pattern {i:#06x}");
    }

    let odd_f16: Vec<f16> = (0..4097u32)
        .map(|_| f16::from_bits(lcg(&mut state) as u16))
        .collect();
    let t = Tensor::from_vec(odd_f16.clone(), 4097, &dev).unwrap();
    let g = WgpuTensor::from_candle(ctx, "rt-f16", &t).unwrap();
    assert_eq!(g.len(), 4097);
    assert_eq!(g.word_len(), 4097usize.div_ceil(2));
    let got: Vec<f16> = g
        .to_candle(ctx, &dev)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for (i, (a, b)) in odd_f16.iter().zip(got.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "f16 idx {i}");
    }

    let u8_vals: Vec<u8> = (0..4097u32).map(|_| lcg(&mut state) as u8).collect();
    let t = Tensor::from_vec(u8_vals.clone(), (17, 241), &dev).unwrap();
    let g = WgpuTensor::from_candle(ctx, "rt-u8", &t).unwrap();
    assert_eq!(g.word_len(), 4097usize.div_ceil(4));
    let got: Vec<u8> = g
        .to_candle(ctx, &dev)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(got, u8_vals);

    let u32_vals: Vec<u32> = (0..513u32).map(|_| lcg(&mut state)).collect();
    let t = Tensor::from_vec(u32_vals.clone(), 513, &dev).unwrap();
    let g = WgpuTensor::from_candle(ctx, "rt-u32", &t).unwrap();
    let got: Vec<u32> = g
        .to_candle(ctx, &dev)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(got, u32_vals);

    let base = Tensor::from_vec(vals(24, 0.4), (4, 6), &dev).unwrap();
    let tr = base.t().unwrap();
    assert!(!tr.is_contiguous());
    let g = WgpuTensor::from_candle(ctx, "rt-noncontig", &tr).unwrap();
    let expect: Vec<f32> = tr.flatten_all().unwrap().to_vec1().unwrap();
    let got: Vec<f32> = g
        .to_candle(ctx, &dev)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for (i, (a, b)) in expect.iter().zip(got.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "noncontig idx {i}");
    }
    assert_eq!(g.shape().dims(), &[6, 4]);

    println!(
        "candle_round_trip_is_bitwise: f32 {} elems (NaN/inf/-0/subnormal + random bits), bf16 all 65536 patterns, f16 4097 odd-len, u8 4097 padded, u32 513, non-contiguous transpose - all bitwise"
    , 37 * 113);
}

#[test]
fn unsupported_dtype_is_rejected() {
    let Some(ctx) = ctx_or_skip("unsupported_dtype_is_rejected") else {
        return;
    };
    let dev = Device::Cpu;
    let t = Tensor::from_vec(vec![1f64, 2.0, 3.0], 3, &dev).unwrap();
    match WgpuTensor::from_candle(ctx, "rt-f64", &t) {
        Err(BackendError::Bridge(m)) => {
            assert!(m.contains("F64"), "{m}");
            println!("unsupported_dtype_is_rejected: {m}");
        }
        Err(e) => panic!("wrong error kind: {e}"),
        Ok(_) => panic!("F64 must not bridge"),
    }
    let i = Tensor::from_vec(vec![1i64, 2, 3], 3, &dev).unwrap();
    assert!(WgpuTensor::from_candle(ctx, "rt-i64", &i).is_err());
}

#[test]
fn residency_cache_uploads_each_weight_once() {
    let Some(_ctx) = ctx_or_skip("residency_cache_uploads_each_weight_once") else {
        return;
    };
    let backend = match Backend::open(BackendKind::Wgpu) {
        Ok(b) => b,
        Err(e) => {
            if !wgpu_allow_skip() {
                panic!("wgpu backend unavailable: {e}");
            }
            eprintln!(
                "SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1), NOT PASSED: \
                 residency_cache_uploads_each_weight_once: {e}"
            );
            return;
        }
    };
    let w = backend.wgpu().unwrap();
    let dev = Device::Cpu;
    let w1 = Tensor::from_vec(vals(1024, 0.7), 1024, &dev).unwrap();
    let w2 = Tensor::from_vec(vals(2048, 1.1), (2, 1024), &dev).unwrap();

    let a = w.upload_weight("w1", &w1).unwrap();
    let b = w.upload_weight("w1", &w1).unwrap();
    assert!(std::sync::Arc::ptr_eq(&a, &b));
    let c = w.upload_weight("w2", &w2).unwrap();
    assert!(!std::sync::Arc::ptr_eq(&a, &c));

    let s = w.residency.stats();
    assert_eq!(s.misses, 2);
    assert_eq!(s.hits, 1);
    assert_eq!(s.entries, 2);
    assert_eq!(s.resident_bytes, (1024 + 2048) * 4);

    assert!(w.residency.contains(&w1));
    assert!(w.residency.evict(&w1));
    assert!(!w.residency.contains(&w1));
    assert!(!w.residency.evict(&w1));
    let s = w.residency.stats();
    assert_eq!(s.entries, 1);
    assert_eq!(s.resident_bytes, 2048 * 4);
    w.residency.clear();
    assert_eq!(w.residency.stats().entries, 0);
    assert_eq!(w.residency.stats().resident_bytes, 0);

    println!(
        "residency_cache_uploads_each_weight_once: 2 misses, 1 hit, ptr-equal Arc on re-request, evict and clear tracked"
    );
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SiluParams {
    n: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[allow(clippy::too_many_arguments)]
fn host_step(
    ctx: &WgpuContext,
    x: &[f32],
    w1: &[f32],
    gate: &[f32],
    w2: &[f32],
    batch: usize,
    hidden: usize,
    eps: f32,
    layers: usize,
) -> Vec<f32> {
    let n = batch * hidden;
    let mut cur = x.to_vec();
    let mut h1 = vec![0f32; n];
    let mut h2 = vec![0f32; n];
    let mut nxt = vec![0f32; n];
    for _ in 0..layers {
        rmsnorm::rmsnorm_f32(ctx, &cur, w1, &mut h1, batch, hidden, eps).unwrap();
        silu::silu_mul_f32(ctx, &h1, gate, &mut h2, n).unwrap();
        rmsnorm::rmsnorm_f32(ctx, &h2, w2, &mut nxt, batch, hidden, eps).unwrap();
        std::mem::swap(&mut cur, &mut nxt);
    }
    cur
}

#[test]
fn bridged_weights_run_multi_kernel_chain_without_intermediate_readback() {
    let Some(ctx) = ctx_or_skip("bridged_weights_chain") else {
        return;
    };
    let dev = Device::Cpu;
    let batch = 8usize;
    let hidden = 4096usize;
    let n = batch * hidden;
    let eps = 1e-5f32;
    let layers = 8usize;
    let kernels_per_step = layers * 3;

    let w1_host = vals(hidden, 0.7);
    let gate_host = vals(n, 1.3);
    let w2_host = vals(hidden, 2.9);
    let w1_t = Tensor::from_vec(w1_host.clone(), hidden, &dev).unwrap();
    let gate_t = Tensor::from_vec(gate_host.clone(), (batch, hidden), &dev).unwrap();
    let w2_t = Tensor::from_vec(w2_host.clone(), hidden, &dev).unwrap();

    let cache = ResidencyCache::new();
    let rms_src = compose(rmsnorm::WGSL);
    let silu_src = compose(silu::WGSL);
    let rms_params = GpuUniform::new(
        ctx,
        "bridge-rms-p",
        &RmsParams {
            hidden: hidden as u32,
            batch: batch as u32,
            eps,
            words_per_row: hidden as u32,
        },
    );
    let silu_params = GpuUniform::new(
        ctx,
        "bridge-silu-p",
        &SiluParams {
            n: n as u32,
            ..Default::default()
        },
    );
    let rms_groups = dispatch::workgroup_count_1d(ctx, batch as u64, 1);
    let silu_groups = dispatch::workgroup_count_1d(ctx, n as u64, silu::WORKGROUP_SIZE);

    let xa = WgpuTensor::zeroed(ctx, "bridge-xa", DType::F32, (batch, hidden)).unwrap();
    let xb = WgpuTensor::zeroed(ctx, "bridge-xb", DType::F32, (batch, hidden)).unwrap();
    let h1 = WgpuTensor::zeroed(ctx, "bridge-h1", DType::F32, (batch, hidden)).unwrap();
    let h2 = WgpuTensor::zeroed(ctx, "bridge-h2", DType::F32, (batch, hidden)).unwrap();

    let resident_step = |x_t: &Tensor| -> Vec<f32> {
        let gw1 = cache.get_or_upload(ctx, "bridge-w1", &w1_t).unwrap();
        let ggate = cache.get_or_upload(ctx, "bridge-gate", &gate_t).unwrap();
        let gw2 = cache.get_or_upload(ctx, "bridge-w2", &w2_t).unwrap();
        xa.write_candle(ctx, x_t).unwrap();
        let mut cur = &xa;
        let mut nxt = &xb;
        let mut chain = Chain::new(ctx);
        for _ in 0..layers {
            chain
                .push(
                    "nv_kernels_rmsnorm_f32",
                    &rms_src,
                    "rmsnorm_f32",
                    &[
                        (0, cur.gpu() as &dyn GpuBind),
                        (1, gw1.gpu()),
                        (2, h1.gpu()),
                        (3, &rms_params),
                    ],
                    rms_groups,
                )
                .unwrap();
            chain
                .push(
                    "silu-mul-f32",
                    &silu_src,
                    "silu_mul_f32",
                    &[
                        (0, h1.gpu() as &dyn GpuBind),
                        (1, ggate.gpu()),
                        (2, h2.gpu()),
                        (3, &silu_params),
                    ],
                    silu_groups,
                )
                .unwrap();
            chain
                .push(
                    "nv_kernels_rmsnorm_f32",
                    &rms_src,
                    "rmsnorm_f32",
                    &[
                        (0, h2.gpu() as &dyn GpuBind),
                        (1, gw2.gpu()),
                        (2, nxt.gpu()),
                        (3, &rms_params),
                    ],
                    rms_groups,
                )
                .unwrap();
            std::mem::swap(&mut cur, &mut nxt);
        }
        assert_eq!(chain.len(), kernels_per_step);
        chain.submit().unwrap();
        let out = cur.to_candle(ctx, &dev).unwrap();
        assert_eq!(out.dims(), &[batch, hidden]);
        out.flatten_all().unwrap().to_vec1().unwrap()
    };

    let x1_host = vals(n, 0.1);
    let x1_t = Tensor::from_vec(x1_host.clone(), (batch, hidden), &dev).unwrap();
    let expect1 = host_step(
        ctx, &x1_host, &w1_host, &gate_host, &w2_host, batch, hidden, eps, layers,
    );
    let got1 = resident_step(&x1_t);
    for (i, (g, e)) in got1.iter().zip(expect1.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "step1 idx {i}: {g} vs {e}");
    }

    let x2_host = vals(n, 4.2);
    let x2_t = Tensor::from_vec(x2_host.clone(), (batch, hidden), &dev).unwrap();
    let expect2 = host_step(
        ctx, &x2_host, &w1_host, &gate_host, &w2_host, batch, hidden, eps, layers,
    );
    let got2 = resident_step(&x2_t);
    for (i, (g, e)) in got2.iter().zip(expect2.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "step2 idx {i}: {g} vs {e}");
    }

    let s = cache.stats();
    assert_eq!(s.misses, 3);
    assert_eq!(s.hits, 3);
    assert_eq!(s.entries, 3);

    let steps = 20usize;
    let time_it = |f: &mut dyn FnMut()| -> f64 {
        let t = std::time::Instant::now();
        for _ in 0..steps {
            f();
        }
        t.elapsed().as_secs_f64()
    };
    let mut host_arm = || {
        std::hint::black_box(host_step(
            ctx, &x1_host, &w1_host, &gate_host, &w2_host, batch, hidden, eps, layers,
        ));
    };
    let host_a1 = time_it(&mut host_arm);
    let resident_s = time_it(&mut || {
        std::hint::black_box(resident_step(&x1_t));
    });
    let host_a2 = time_it(&mut host_arm);

    let s = cache.stats();
    assert_eq!(s.misses, 3, "weights must upload exactly once ever");
    assert_eq!(s.hits, 3 * (2 + steps as u64 - 1));

    let host_s = host_a1.min(host_a2);
    let null = (host_a1 - host_a2).abs() / host_a1.min(host_a2);
    let speedup = host_s / resident_s;
    println!(
        "bridged chain: {} layers x 3 kernels = {} kernels/step on {} f32 elems, bitwise vs host-slice API on both steps; A/B/A {} steps/arm: host_A1 {:.3} ms/step, bridged-resident {:.3} ms/step (1 upload + {} dispatches + 1 readback), host_A2 {:.3} ms/step; same-arm null {:.1}%, speedup {:.1}x; weights resident: {} tensors, {} bytes, uploaded once",
        layers,
        kernels_per_step,
        n,
        steps,
        host_a1 * 1e3 / steps as f64,
        resident_s * 1e3 / steps as f64,
        kernels_per_step,
        host_a2 * 1e3 / steps as f64,
        null * 100.0,
        speedup,
        s.entries,
        s.resident_bytes
    );
    if null > 0.25 {
        println!(
            "TIMING INCONCLUSIVE, NOT ASSERTED: the host arm moved {:.1}% between its own two \
             measurements, which is more than this test can resolve. Reporting {speedup:.1}x \
             without gating on it. Re-run on an idle box for a verdict.",
            null * 100.0
        );
        return;
    }
    assert!(
        speedup > 1.0 + null,
        "bridged-resident ({:.3} ms/step) is not faster than host round-trips ({:.3} ms/step) by \
         more than the same-arm null ({:.1}%): speedup {speedup:.2}x",
        resident_s * 1e3 / steps as f64,
        host_s * 1e3 / steps as f64,
        null * 100.0
    );
}
