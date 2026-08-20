#![cfg(feature = "wgpu")]

mod common;
use common::HostMat;
use common::routing;
use common::splat;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_layers::moe_wgpu::{self, MoeWgpuExpertSource, MoeWgpuWeights};
use nv_quant::nvfp4::{swizzle_scales, Nvfp4Tensor, BLOCK_SIZE};
use common::expert_mats_live as expert_mats;

fn backend(test: &str) -> Option<&'static WgpuContext> {
    let allow_skip = std::env::var("NV_KERNELS_WGPU_ALLOW_SKIP").as_deref() == Ok("1");
    match WgpuContext::shared() {
        Ok(ctx) if ctx.qualify().qualified => {
            eprintln!("{test}: {}", ctx.summary());
            Some(ctx)
        }
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
                "{test}: no wgpu adapter: {e}. nvk.sh wires VK_ICD_FILENAMES and the store's \
                 vulkan-loader, so a miss means that wiring regressed. Set \
                 NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
            );
        }
    }
}

struct HostExperts {
    gate: HostMat,
    up: HostMat,
    down: HostMat,
    globals_gu: Vec<f32>,
    globals_dn: Vec<f32>,
}

fn host_experts(e_total: usize, hidden: usize, inter: usize, live_inter: usize) -> HostExperts {
    HostExperts {
        gate: expert_mats(e_total, inter, hidden, live_inter, hidden, 0xa11ce),
        up: expert_mats(e_total, inter, hidden, live_inter, hidden, 0xb0b),
        down: expert_mats(e_total, hidden, inter, hidden, live_inter, 0xcafe),
        globals_gu: (0..e_total).map(|e| 1.5 + 0.01 * e as f32).collect(),
        globals_dn: (0..e_total).map(|e| 2.0 + 0.02 * e as f32).collect(),
    }
}

fn sources(h: &HostExperts) -> Vec<MoeWgpuExpertSource<'_>> {
    (0..h.gate.packed.len())
        .map(|e| MoeWgpuExpertSource {
            gate_packed: &h.gate.packed[e],
            gate_scales_swizzled: &h.gate.scales_swizzled[e],
            gate_alpha: 1.0 / h.globals_gu[e],
            up_packed: &h.up.packed[e],
            up_scales_swizzled: &h.up.scales_swizzled[e],
            up_alpha: 0.5 / h.globals_gu[e],
            down_packed: &h.down.packed[e],
            down_scales_swizzled: &h.down.scales_swizzled[e],
            down_alpha: 1.0 / h.globals_dn[e],
            input_global_gate_up: h.globals_gu[e],
            input_global_down: h.globals_dn[e],
        })
        .collect()
}

fn x_bf16(n_tokens: usize, hidden: usize) -> Vec<u16> {
    (0..n_tokens * hidden)
        .map(|i| half::bf16::from_f32(splat(0xf00d, i / hidden, i % hidden) * 0.5).to_bits())
        .collect()
}

fn forward(
    ctx: &'static WgpuContext,
    hidden: usize,
    inter: usize,
    live_inter: usize,
    e_total: usize,
    n_tokens: usize,
    k: usize,
) -> Vec<f32> {
    let h = host_experts(e_total, hidden, inter, live_inter);
    let (ids, wts) = routing(n_tokens, k, e_total);
    let x = x_bf16(n_tokens, hidden);
    let w = MoeWgpuWeights::from_expert_sources(ctx, hidden, inter, &sources(&h))
        .expect("wgpu weights");
    moe_wgpu::try_forward(&w, ctx, &x, &ids, &wts, n_tokens, k)
        .expect("wgpu forward")
        .expect("wgpu forward should not decline")
}

fn compare(a: &[f32], b: &[f32]) -> (usize, i64, f64) {
    let mut differ = 0usize;
    let mut max_ulp = 0i64;
    let mut max_rel = 0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.to_bits() != y.to_bits() {
            differ += 1;
            max_ulp = max_ulp.max((x.to_bits() as i64 - y.to_bits() as i64).abs());
            let d = (*x as f64 - *y as f64).abs();
            let den = (x.abs() as f64).max(y.abs() as f64).max(1e-30);
            max_rel = max_rel.max(d / den);
        }
    }
    (differ, max_ulp, max_rel)
}

fn case(
    ctx: &'static WgpuContext,
    hidden: usize,
    inter: usize,
    e_total: usize,
    n_tokens: usize,
    k: usize,
) -> usize {
    let padded = inter.div_ceil(128) * 128;
    let a = forward(ctx, hidden, inter, inter, e_total, n_tokens, k);
    let b = forward(ctx, hidden, padded, inter, e_total, n_tokens, k);
    assert_eq!(a.len(), b.len());
    let nz = a.iter().filter(|v| **v != 0.0).count();
    assert!(
        nz > a.len() / 4,
        "degenerate: only {nz}/{} outputs nonzero at inter={inter}",
        a.len()
    );
    let (differ, max_ulp, max_rel) = compare(&a, &b);
    println!(
        "moe zero-pad hidden={hidden} inter={inter} -> padded={padded} (inter%128={}) E={e_total} tokens={n_tokens} k={k}: {differ}/{} differ max_ulp={max_ulp} max_rel={max_rel:.3e}",
        inter % 128,
        a.len()
    );
    differ
}

#[test]
fn moe_forward_is_invariant_to_zero_padding_intermediate_rows_to_128() {
    let Some(ctx) = backend("moe_forward_is_invariant_to_zero_padding_intermediate_rows_to_128")
    else {
        return;
    };

    let mut bad: Vec<usize> = Vec::new();
    for inter in [128usize, 192, 256, 320, 512, 704, 768] {
        if case(ctx, 256, inter, 8, 5, 2) != 0 {
            bad.push(inter);
        }
    }
    assert!(
        bad.is_empty(),
        "zero-padding the intermediate dim to a multiple of 128 changed the wgpu MoE result for inter={bad:?}; \
         a padded row contributes gate=0,up=0 -> gelu(0)*0 = 0 and a padded down column is multiplied by 0, \
         so the result must be unchanged"
    );
}

#[test]
fn moe_forward_is_invariant_to_zero_padding_at_gemma4_26b_shapes() {
    let Some(ctx) = backend("moe_forward_is_invariant_to_zero_padding_at_gemma4_26b_shapes") else {
        return;
    };
    let differ = case(ctx, 2816, 704, 16, 13, 8);
    assert_eq!(
        differ, 0,
        "zero-padding inter 704 -> 768 changed the wgpu MoE result at Gemma4-26B-A4B shapes"
    );
}
