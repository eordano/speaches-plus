#![allow(dead_code)]
#![allow(unused_imports)]

use candle_core::{DType, Device, Tensor, Var};
#[cfg(feature = "wgpu")]
use nv_quant::nvfp4::{swizzle_scales, Nvfp4Tensor, BLOCK_SIZE};
#[cfg(feature = "cuda")]
use nv_layers::linear::Linear;
#[cfg(feature = "cuda")]
use nv_layers::linear_attn::{LinearAttention, LinearAttentionConfig};
#[cfg(feature = "cuda")]
use cudarc::driver::sys::CUdevice_attribute;
#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
#[cfg(feature = "wgpu")]
use nv_layers::moe_wgpu::{self, MoeWgpuExpertSource, MoeWgpuWeights, MIN_TILE};

#[cfg(feature = "cuda")]
pub fn build_layer(cfg: LinearAttentionConfig, dev: &Device) -> LinearAttention {
    let h = cfg.hidden_size;
    let conv_dim = cfg.conv_dim();
    let value_dim = cfg.value_dim();
    let n_v = cfg.linear_num_value_heads;
    let mk = |o: usize, i: usize| Linear::new(randn(&[o, i], dev), None).unwrap();
    LinearAttention::new(
        cfg,
        mk(conv_dim, h),
        mk(value_dim, h),
        mk(n_v, h),
        mk(n_v, h),
        randn(&[conv_dim, 1, cfg.linear_conv_kernel_dim], dev),
        randn(&[n_v], dev),
        randn(&[n_v], dev),
        randn(&[cfg.linear_value_head_dim], dev),
        mk(h, value_dim),
    )
    .unwrap()
}

#[cfg(feature = "cuda")]
pub fn cuda() -> Option<Device> {
    Device::new_cuda(0).ok()
}

#[cfg(feature = "cuda")]
pub fn detect_major(ctx: &CudaContext) -> i32 {
    ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .unwrap_or(0)
}

#[cfg(feature = "cuda")]
pub fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

#[cfg(feature = "wgpu")]
pub struct HostExperts {
    pub gate: HostMat,
    pub up: HostMat,
    pub down: HostMat,
    pub gate_alphas: Vec<f32>,
    pub up_alphas: Vec<f32>,
    pub down_alphas: Vec<f32>,
    pub globals_gu: Vec<f32>,
    pub globals_dn: Vec<f32>,
}

#[cfg(feature = "wgpu")]
pub struct HostMat {
    pub packed: Vec<Vec<u8>>,
    pub scales_swizzled: Vec<Vec<u8>>,
}

#[cfg(feature = "cuda")]
pub fn randn(shape: &[usize], dev: &Device) -> Tensor {
    Tensor::randn(0f32, 0.05, shape, &Device::Cpu)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap()
        .to_device(dev)
        .unwrap()
}

#[cfg(feature = "cuda")]
pub fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (num / den.max(1e-12)).sqrt() as f32
}

#[cfg(feature = "wgpu")]
pub fn routing(n_tokens: usize, k: usize, e_total: usize) -> (Vec<u32>, Vec<f32>) {
    let mut ids = Vec::with_capacity(n_tokens * k);
    let mut wts = Vec::with_capacity(n_tokens * k);
    for n in 0..n_tokens {
        let mut chosen: Vec<u32> = Vec::with_capacity(k);
        let mut step = 0usize;
        while chosen.len() < k {
            let e = ((n * 31 + step * 17 + 5) % e_total) as u32;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
            step += 1;
        }
        let raw: Vec<f32> = (0..k).map(|j| 0.2 + 0.1 * ((n + j) % 7) as f32).collect();
        let z: f32 = raw.iter().sum();
        ids.extend_from_slice(&chosen);
        wts.extend(raw.iter().map(|w| w / z));
    }
    (ids, wts)
}

#[cfg(feature = "wgpu")]
pub fn sources(h: &HostExperts) -> Vec<MoeWgpuExpertSource<'_>> {
    (0..h.gate.packed.len())
        .map(|e| MoeWgpuExpertSource {
            gate_packed: &h.gate.packed[e],
            gate_scales_swizzled: &h.gate.scales_swizzled[e],
            gate_alpha: h.gate_alphas[e],
            up_packed: &h.up.packed[e],
            up_scales_swizzled: &h.up.scales_swizzled[e],
            up_alpha: h.up_alphas[e],
            down_packed: &h.down.packed[e],
            down_scales_swizzled: &h.down.scales_swizzled[e],
            down_alpha: h.down_alphas[e],
            input_global_gate_up: h.globals_gu[e],
            input_global_down: h.globals_dn[e],
        })
        .collect()
}

#[cfg(feature = "wgpu")]
pub fn splat(seed: u64, r: usize, c: usize) -> f32 {
    let mut h = seed ^ (r as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^= (c as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    ((h >> 33) as u32 as f32 / 2147483647.0) * 2.0 - 1.0
}

#[cfg(feature = "wgpu")]
pub fn expert_mats(e_total: usize, n: usize, k: usize, seed: u64) -> HostMat {
    expert_mats_live(e_total, n, k, n, k, seed)
}

#[cfg(feature = "wgpu")]
pub fn expert_mats_live(
    e_total: usize,
    n: usize,
    k: usize,
    live_n: usize,
    live_k: usize,
    seed: u64,
) -> HostMat {
    let mut packed = Vec::with_capacity(e_total);
    let mut scales_swizzled = Vec::with_capacity(e_total);
    for e in 0..e_total {
        let s = seed ^ (e as u64).wrapping_mul(0x1234_5678_9abc_def1);
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|r| {
                (0..k)
                    .map(|c| {
                        if r < live_n && c < live_k {
                            splat(s, r, c)
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();
        let t = Nvfp4Tensor::quantize_rows(&rows);
        scales_swizzled.push(swizzle_scales(&t.scales, n, k / BLOCK_SIZE));
        packed.push(t.data);
    }
    HostMat {
        packed,
        scales_swizzled,
    }
}
