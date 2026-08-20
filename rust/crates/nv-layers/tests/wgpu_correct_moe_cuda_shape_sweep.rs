#![cfg(all(feature = "cuda", feature = "wgpu"))]

mod common;
use common::HostExperts;
use common::HostMat;
use common::routing;
use common::sources;
use common::splat;
use candle_core::{Device, Tensor};
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_layers::moe_grouped::{self, MoeGroupedWeights};
use nv_layers::moe_wgpu::{self, MoeWgpuExpertSource, MoeWgpuWeights};
use nv_quant::nvfp4::{swizzle_scales, Nvfp4GemmRunner, Nvfp4Tensor, BLOCK_SIZE};
use std::sync::{Arc, Mutex};
use common::expert_mats;
use common::expert_mats_live;

fn backend(test: &str) -> Option<&'static WgpuContext> {
    let allow_skip = std::env::var("NV_KERNELS_WGPU_ALLOW_SKIP").as_deref() == Ok("1");
    match WgpuContext::shared() {
        Ok(ctx) if ctx.qualify().qualified => Some(ctx),
        Ok(ctx) => {
            if allow_skip {
                eprintln!(
                    "SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1): {test}: adapter not qualified: {:?}. \
                     Not a pass.",
                    ctx.qualify().reason
                );
                return None;
            }
            panic!(
                "{test}: wgpu adapter not qualified: {:?}. Set NV_KERNELS_WGPU_ALLOW_SKIP=1 to \
                 skip on purpose.",
                ctx.qualify().reason
            );
        }
        Err(e) => {
            if allow_skip {
                eprintln!(
                    "SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1): {test}: no wgpu adapter: {e}. Not a pass."
                );
                return None;
            }
            panic!(
                "{test}: no wgpu adapter: {e}. Set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
            );
        }
    }
}

fn cuda_device(test: &str) -> Option<Device> {
    match Device::new_cuda(0) {
        Ok(d) => Some(d),
        Err(e) => {
            if std::env::var("NV_KERNELS_PARITY_ALLOW_SKIP").as_deref() == Ok("1") {
                eprintln!(
                    "SKIP (NV_KERNELS_PARITY_ALLOW_SKIP=1): {test}: no CUDA device 0: {e}. Not a \
                     pass."
                );
                return None;
            }
            panic!(
                "{test}: no CUDA device 0: {e}. This is a cuda-vs-wgpu divergence sweep; set \
                 NV_KERNELS_PARITY_ALLOW_SKIP=1 to skip on purpose."
            );
        }
    }
}

fn host_experts(e_total: usize, hidden: usize, inter: usize) -> HostExperts {
    host_experts_live(e_total, hidden, inter, inter)
}

fn host_experts_live(
    e_total: usize,
    hidden: usize,
    inter: usize,
    live_inter: usize,
) -> HostExperts {
    let globals_gu: Vec<f32> = (0..e_total).map(|e| 1.5 + 0.01 * e as f32).collect();
    let globals_dn: Vec<f32> = (0..e_total).map(|e| 2.0 + 0.02 * e as f32).collect();
    HostExperts {
        gate: expert_mats_live(e_total, inter, hidden, live_inter, hidden, 0xa11ce),
        up: expert_mats_live(e_total, inter, hidden, live_inter, hidden, 0xb0b),
        down: expert_mats_live(e_total, hidden, inter, hidden, live_inter, 0xcafe),
        gate_alphas: globals_gu.iter().map(|g| 1.0 / g).collect(),
        up_alphas: globals_gu.iter().map(|g| 0.5 / g).collect(),
        down_alphas: globals_dn.iter().map(|g| 1.0 / g).collect(),
        globals_gu,
        globals_dn,
    }
}

fn cuda_weights(
    device: &Device,
    h: &HostExperts,
    e_total: usize,
    hidden: usize,
    inter: usize,
) -> MoeGroupedWeights {
    let dev = match device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let stream = dev.cuda_stream();
    let runner = Arc::new(Mutex::new(
        Nvfp4GemmRunner::new(stream.clone()).expect("nvfp4 runner"),
    ));
    let concat = |mats: &HostMat| -> (Vec<u8>, Vec<u8>) {
        let mut p = Vec::new();
        let mut s = Vec::new();
        for e in 0..e_total {
            p.extend_from_slice(&mats.packed[e]);
            s.extend_from_slice(&mats.scales_swizzled[e]);
        }
        (p, s)
    };
    let (gate_p, gate_s) = concat(&h.gate);
    let (up_p, up_s) = concat(&h.up);
    let (down_p, down_s) = concat(&h.down);
    #[allow(deprecated)]
    let htod_u8 = |v: &[u8]| stream.clone_htod(v).expect("htod");
    #[allow(deprecated)]
    let htod_f32 = |v: &[f32]| stream.clone_htod(v).expect("htod");
    MoeGroupedWeights {
        num_experts: e_total,
        hidden_size: hidden,
        intermediate_size: inter,
        gate_w: htod_u8(&gate_p),
        gate_w_scales: htod_u8(&gate_s),
        gate_alphas: htod_f32(&h.gate_alphas),
        gate_a_stride_elems: hidden as i64,
        gate_b_stride_elems: hidden as i64,
        gate_c_stride_elems: inter as i64,
        up_w: htod_u8(&up_p),
        up_w_scales: htod_u8(&up_s),
        up_alphas: htod_f32(&h.up_alphas),
        down_w: htod_u8(&down_p),
        down_w_scales: htod_u8(&down_s),
        down_alphas: htod_f32(&h.down_alphas),
        down_a_stride_elems: inter as i64,
        down_b_stride_elems: inter as i64,
        down_c_stride_elems: hidden as i64,
        runner,
        input_globals_gate_up: htod_f32(&h.globals_gu),
        input_globals_down: htod_f32(&h.globals_dn),
        input_globals_gate_up_host: h.globals_gu.clone(),
        input_globals_down_host: h.globals_dn.clone(),
    }
}

fn x_bf16(n_tokens: usize, hidden: usize) -> Vec<u16> {
    (0..n_tokens * hidden)
        .map(|i| bf16::from_f32(splat(0xf00d, i / hidden, i % hidden) * 0.5).to_bits())
        .collect()
}

fn f32_ord(v: f32) -> i64 {
    let b = v.to_bits() as i32;
    if b < 0 {
        (i32::MIN - b) as i64
    } else {
        b as i64
    }
}

struct Outcome {
    differ: usize,
    total: usize,
    max_ulp: i64,
    max_rel: f64,
}

fn run_case(
    ctx: &'static WgpuContext,
    device: &Device,
    e_total: usize,
    hidden: usize,
    inter: usize,
    n_tokens: usize,
    k: usize,
) -> Outcome {
    let h = host_experts(e_total, hidden, inter);
    let (ids, wts) = routing(n_tokens, k, e_total);
    let x = x_bf16(n_tokens, hidden);

    let cw = cuda_weights(device, &h, e_total, hidden, inter);
    let x_vals: Vec<bf16> = x.iter().map(|b| bf16::from_bits(*b)).collect();
    let x_t = Tensor::from_vec(x_vals, (n_tokens, hidden), device).expect("x tensor");
    let cuda_t = moe_grouped::forward_grouped(&cw, &cw, &x_t, &ids, &wts, n_tokens, k, device)
        .expect("cuda forward_grouped");
    let cuda: Vec<f32> = cuda_t.flatten_all().unwrap().to_vec1().unwrap();

    let ww = MoeWgpuWeights::from_expert_sources(ctx, hidden, inter, &sources(&h))
        .expect("wgpu weights");
    let wg = moe_wgpu::try_forward(&ww, ctx, &x, &ids, &wts, n_tokens, k)
        .expect("wgpu forward")
        .expect("wgpu forward should not decline");

    assert_eq!(cuda.len(), wg.len());
    let nz = cuda.iter().filter(|v| **v != 0.0).count();
    assert!(
        nz > cuda.len() / 4,
        "cuda reference mostly zero ({nz}/{}) at hidden={hidden} inter={inter}",
        cuda.len()
    );

    let mut differ = 0usize;
    let mut max_ulp = 0i64;
    let mut max_rel = 0f64;
    for (g, w) in wg.iter().zip(cuda.iter()) {
        if g.to_bits() != w.to_bits() {
            differ += 1;
            max_ulp = max_ulp.max((f32_ord(*g) - f32_ord(*w)).abs());
            let den = (g.abs() as f64).max(w.abs() as f64).max(1e-30);
            max_rel = max_rel.max((*g as f64 - *w as f64).abs() / den);
        }
    }
    println!(
        "moe cuda-vs-wgpu E={e_total} hidden={hidden} inter={inter} (inter%128={}) tokens={n_tokens} k={k}: {differ}/{} differ max_ulp={max_ulp} max_rel={max_rel:.3e}",
        inter % 128,
        cuda.len()
    );
    Outcome {
        differ,
        total: cuda.len(),
        max_ulp,
        max_rel,
    }
}

#[test]
fn moe_cuda_wgpu_divergence_tracks_intermediate_size_mod_128() {
    let name = "moe_cuda_wgpu_divergence_tracks_intermediate_size_mod_128";
    let Some(ctx) = backend(name) else { return };
    let Some(device) = cuda_device(name) else {
        return;
    };

    let mut aligned_bad = Vec::new();
    let mut misaligned_bad = Vec::new();
    for inter in [512usize, 640, 704, 768, 896] {
        let o = run_case(ctx, &device, 16, 2816, inter, 13, 8);
        if o.differ > 0 {
            if inter % 128 == 0 {
                aligned_bad.push((inter, o.differ, o.total, o.max_ulp, o.max_rel));
            } else {
                misaligned_bad.push((inter, o.differ, o.total, o.max_ulp, o.max_rel));
            }
        }
    }
    println!("aligned (inter%128==0) failures: {aligned_bad:?}");
    println!("misaligned (inter%128!=0) failures: {misaligned_bad:?}");
    assert!(
        aligned_bad.is_empty() && misaligned_bad.is_empty(),
        "wgpu MoE forward must be bit-exact against the CUDA grouped path; \
         aligned failures {aligned_bad:?}, misaligned failures {misaligned_bad:?}"
    );
}

#[test]
fn moe_cuda_wgpu_divergence_is_independent_of_hidden_size() {
    let name = "moe_cuda_wgpu_divergence_is_independent_of_hidden_size";
    let Some(ctx) = backend(name) else { return };
    let Some(device) = cuda_device(name) else {
        return;
    };

    let mut bad = Vec::new();
    for (hidden, inter) in [(2048usize, 704usize), (2816, 512), (2048, 512), (2816, 704)] {
        let o = run_case(ctx, &device, 16, hidden, inter, 13, 8);
        if o.differ > 0 {
            bad.push((hidden, inter, o.differ, o.total, o.max_ulp));
        }
    }
    assert!(
        bad.is_empty(),
        "wgpu MoE forward diverges from CUDA at (hidden, inter, differ, total, max_ulp) = {bad:?}"
    );
}

fn cuda_only(
    device: &Device,
    e_total: usize,
    hidden: usize,
    inter: usize,
    live_inter: usize,
    n_tokens: usize,
    k: usize,
) -> Vec<f32> {
    let h = host_experts_live(e_total, hidden, inter, live_inter);
    let (ids, wts) = routing(n_tokens, k, e_total);
    let x = x_bf16(n_tokens, hidden);
    let cw = cuda_weights(device, &h, e_total, hidden, inter);
    let x_vals: Vec<bf16> = x.iter().map(|b| bf16::from_bits(*b)).collect();
    let x_t = Tensor::from_vec(x_vals, (n_tokens, hidden), device).expect("x tensor");
    let t = moe_grouped::forward_grouped(&cw, &cw, &x_t, &ids, &wts, n_tokens, k, device)
        .expect("cuda forward_grouped");
    t.flatten_all().unwrap().to_vec1().unwrap()
}

#[test]
fn cuda_moe_forward_is_invariant_to_zero_padding_intermediate_rows_to_128() {
    let name = "cuda_moe_forward_is_invariant_to_zero_padding_intermediate_rows_to_128";
    let Some(device) = cuda_device(name) else {
        return;
    };
    let (e_total, hidden, n_tokens, k) = (16usize, 2816usize, 13usize, 8usize);
    let mut bad = Vec::new();
    for inter in [512usize, 704] {
        let padded = inter.div_ceil(128) * 128;
        let a = cuda_only(&device, e_total, hidden, inter, inter, n_tokens, k);
        let b = cuda_only(&device, e_total, hidden, padded, inter, n_tokens, k);
        assert_eq!(a.len(), b.len());
        let nz = a.iter().filter(|v| **v != 0.0).count();
        assert!(nz > a.len() / 4, "degenerate cuda output at inter={inter}");
        let mut max_rel = 0f64;
        let mut differ = 0usize;
        for (x, y) in a.iter().zip(b.iter()) {
            if x.to_bits() != y.to_bits() {
                differ += 1;
                let den = (x.abs() as f64).max(y.abs() as f64).max(1e-30);
                max_rel = max_rel.max((*x as f64 - *y as f64).abs() / den);
            }
        }
        println!(
            "cuda zero-pad hidden={hidden} inter={inter} -> {padded} (inter%128={}): {differ}/{} differ max_rel={max_rel:.3e}",
            inter % 128,
            a.len()
        );
        if max_rel > 1e-2 {
            bad.push((inter, differ, a.len(), max_rel));
        }
    }
    assert!(
        bad.is_empty(),
        "the CUDA grouped MoE path is NOT invariant to zero-padding the intermediate dim to a \
         multiple of 128 (inter, differ, total, max_rel) = {bad:?}; padded gate/up rows are zero \
         and padded down columns are multiplied by zero, so the result must not change"
    );
}
