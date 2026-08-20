#![cfg(feature = "laguna-wip")]

mod common;
use common::rel_err;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use nv_models::laguna::LagunaConfig;
use nv_models::laguna_wgpu::gpu::{Builder, Pass, Sources};
use nv_models::laguna_wgpu::moe;
use nv_models::laguna_wgpu::weights::{
    bf16_bits, bf16_val, pack_pairs, quantize_nvfp4_host, stack_bf16_host, stack_nvfp4_host,
    HostBf16Lin, HostExperts, HostLin, HostMoe, HostNvfp4Lin, Lcg,
};
use nv_models::laguna_wgpu::{LagunaShapes, LayerShape};

const TINY_CONFIG: &str = r#"{
    "architectures": ["LagunaForCausalLM"],
    "model_type": "laguna",
    "vocab_size": 64,
    "hidden_size": 128,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "num_experts": 6,
    "num_experts_per_tok": 3,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
    "norm_topk_prob": true,
    "mlp_only_layers": [],
    "decoder_sparse_step": 1,
    "tie_word_embeddings": false,
    "gating": "per-head",
    "sliding_window": 8,
    "moe_routed_scaling_factor": 2.5,
    "moe_router_logit_softcapping": 4.0,
    "rope_parameters": {
        "full_attention": {
            "rope_theta": 500000.0,
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 64,
            "beta_slow": 1.0,
            "beta_fast": 64.0,
            "attention_factor": 1.3465735902799727,
            "partial_rotary_factor": 0.5
        },
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0
        }
    },
    "layer_types": ["full_attention", "sliding_attention"],
    "mlp_layer_types": ["dense", "sparse"],
    "num_attention_heads_per_layer": [4, 4]
}"#;

fn tiny_shapes() -> LagunaShapes {
    let cfg = LagunaConfig::from_hf_json_str(TINY_CONFIG).unwrap();
    LagunaShapes::derive(&cfg, 32).unwrap()
}

fn moe_layer(shapes: &LagunaShapes) -> LayerShape {
    let idx = *shapes
        .moe_layer_indices()
        .first()
        .expect("tiny config must have a moe layer");
    *shapes.layer(idx)
}

fn bf16_lin(r: &mut Lcg, n: usize, k: usize, scale: f32) -> HostBf16Lin {
    HostBf16Lin {
        w: r.fill_bf16(n * k, scale),
        n,
        k,
    }
}

fn nvfp4_lin(r: &mut Lcg, n: usize, k: usize, scale: f32) -> HostNvfp4Lin {
    let w = r.fill_bf16(n * k, scale);
    quantize_nvfp4_host(&w, n, k)
}

fn expert_stack(r: &mut Lcg, e: usize, n: usize, k: usize, scale: f32, q: bool) -> HostExperts {
    if q {
        let mats: Vec<HostNvfp4Lin> = (0..e).map(|_| nvfp4_lin(r, n, k, scale)).collect();
        HostExperts::Nvfp4(stack_nvfp4_host(&mats))
    } else {
        let mats: Vec<HostBf16Lin> = (0..e).map(|_| bf16_lin(r, n, k, scale)).collect();
        HostExperts::Bf16(stack_bf16_host(&mats))
    }
}

fn lin(r: &mut Lcg, n: usize, k: usize, scale: f32, q: bool) -> HostLin {
    if q {
        HostLin::Nvfp4(nvfp4_lin(r, n, k, scale))
    } else {
        HostLin::Bf16(bf16_lin(r, n, k, scale))
    }
}

fn tiny_moe(shapes: &LagunaShapes, layer: &LayerShape, seed: u64, q: bool) -> HostMoe {
    let mut r = Lcg::new(seed);
    let hidden = shapes.hidden_size;
    let inter = layer.ffn_intermediate;
    let sinter = shapes.shared_expert_intermediate_size;
    let e = shapes.num_experts;
    HostMoe {
        router: bf16_lin(&mut r, e, hidden, 0.3),
        selection_bias: (0..e).map(|_| r.next_f32() * 0.25).collect(),
        experts_gate: expert_stack(&mut r, e, inter, hidden, 0.15, q),
        experts_up: expert_stack(&mut r, e, inter, hidden, 0.15, q),
        experts_down: expert_stack(&mut r, e, hidden, inter, 0.15, q),
        shared_gate: lin(&mut r, sinter, hidden, 0.15, q),
        shared_up: lin(&mut r, sinter, hidden, 0.15, q),
        shared_down: lin(&mut r, hidden, sinter, 0.15, q),
    }
}

fn tiny_x(seed: u64, hidden: usize) -> Vec<f32> {
    let mut r = Lcg::new(seed);
    (0..hidden)
        .map(|_| bf16_val(bf16_bits(r.next_f32())))
        .collect()
}

fn have_gpu() -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(ctx) => {
            eprintln!("[wgpu] adapter: {}", ctx.info.name);
            Some(ctx)
        }
        Err(e) => {
            eprintln!("[skip] no wgpu adapter: {e}");
            None
        }
    }
}

fn run_passes(ctx: &WgpuContext, passes: &[Pass]) {
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut cp = enc.begin_compute_pass(&Default::default());
        for p in passes {
            cp.set_pipeline(&p.pipeline);
            cp.set_bind_group(0, &p.bind, &[]);
            cp.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
        }
    }
    ctx.queue.submit([enc.finish()]);
    if let Some(e) = pollster::block_on(scope.pop()) {
        panic!("moe graph validation: {e}");
    }
}

fn gpu_moe(shapes: &LagunaShapes, layer: &LayerShape, m: &HostMoe, x: &[f32]) -> Vec<f32> {
    let ctx = have_gpu().expect("gpu checked by caller");
    let s = Sources::new();
    let hidden = shapes.hidden_size;
    let words = shapes.hidden_words();
    let mut b = Builder::new(ctx);
    let bits: Vec<u16> = x.iter().map(|v| bf16_bits(*v)).collect();
    let xb = b.upload_u32("lgw-test-x", &pack_pairs(&bits));
    let out = b.zeros("lgw-test-out", (words * 4) as u64);
    moe::build_moe_layer(&mut b, &s, shapes, layer, m, &xb, &out).expect("build moe layer");
    b.flush_staging();
    run_passes(ctx, &b.passes);
    let got: Vec<u32> = dispatch::read_back(ctx, &out, words).expect("read back");
    let mut y = vec![0f32; hidden];
    for (i, w) in got.iter().enumerate() {
        y[2 * i] = bf16_val((*w & 0xffff) as u16);
        y[2 * i + 1] = bf16_val((*w >> 16) as u16);
    }
    y
}

#[test]
fn router_takes_topk_on_selection_but_weights_from_scores() {
    let shapes = tiny_shapes();
    let layer = moe_layer(&shapes);
    let hidden = shapes.hidden_size;
    let n_e = shapes.num_experts;

    let mut m = tiny_moe(&shapes, &layer, 0xa11ce, false);
    let logit_of = [2.0f32, -1.0, 0.5, 3.0, -2.0, 1.0];
    let mut rw = vec![0u16; n_e * hidden];
    for (e, v) in logit_of.iter().enumerate() {
        rw[e * hidden] = bf16_bits(*v);
    }
    m.router = HostBf16Lin {
        w: rw,
        n: n_e,
        k: hidden,
    };
    m.selection_bias = vec![0f32; n_e];

    let mut x = vec![0f32; hidden];
    x[0] = 1.0;

    let softcap = shapes.router_softcap;
    assert!(softcap > 0.0, "tiny config must exercise the softcap");
    let scores: Vec<f32> = logit_of
        .iter()
        .map(|l| {
            let c = softcap * (*l / softcap).tanh();
            1.0 / (1.0 + (-c).exp())
        })
        .collect();

    let (ids, wts) = moe::ref_router_topk(&shapes, &m, &x).unwrap();
    assert_eq!(ids, vec![3u32, 0, 5], "top-k must follow descending logits");
    let sum: f32 = ids.iter().map(|e| scores[*e as usize]).sum();
    for (j, e) in ids.iter().enumerate() {
        let want = scores[*e as usize] / sum;
        assert!(
            (wts[j] - want).abs() < 1e-6,
            "weight {j} = {} want {want}",
            wts[j]
        );
    }

    let mut biased = m.clone();
    biased.selection_bias = vec![0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
    let (ids2, wts2) = moe::ref_router_topk(&shapes, &biased, &x).unwrap();
    assert_eq!(
        ids2,
        vec![0u32, 3, 5],
        "selection bias must reorder the top-k"
    );
    let sum2: f32 = ids2.iter().map(|e| scores[*e as usize]).sum();
    for (j, e) in ids2.iter().enumerate() {
        let want = scores[*e as usize] / sum2;
        assert!(
            (wts2[j] - want).abs() < 1e-6,
            "biased weight {j} = {} want {want}: bias must not leak into the weights",
            wts2[j]
        );
    }
}

#[test]
fn router_softcap_off_is_plain_sigmoid_and_norm_is_optional() {
    let mut shapes = tiny_shapes();
    let layer = moe_layer(&shapes);
    let hidden = shapes.hidden_size;
    let n_e = shapes.num_experts;

    let mut m = tiny_moe(&shapes, &layer, 0xb22dd, false);
    let logit_of = [2.0f32, -1.0, 0.5, 3.0, -2.0, 1.0];
    let mut rw = vec![0u16; n_e * hidden];
    for (e, v) in logit_of.iter().enumerate() {
        rw[e * hidden] = bf16_bits(*v);
    }
    m.router = HostBf16Lin {
        w: rw,
        n: n_e,
        k: hidden,
    };
    m.selection_bias = vec![0f32; n_e];
    let mut x = vec![0f32; hidden];
    x[0] = 1.0;

    shapes.router_softcap = 0.0;
    shapes.norm_topk_prob = false;
    let (ids, wts) = moe::ref_router_topk(&shapes, &m, &x).unwrap();
    assert_eq!(ids, vec![3u32, 0, 5]);
    for (j, e) in ids.iter().enumerate() {
        let want = 1.0 / (1.0 + (-logit_of[*e as usize]).exp());
        assert!(
            (wts[j] - want).abs() < 1e-6,
            "unnormalised weight {j} = {} want {want}",
            wts[j]
        );
    }

    shapes.norm_topk_prob = true;
    let (_, wn) = moe::ref_router_topk(&shapes, &m, &x).unwrap();
    let s: f32 = wn.iter().sum();
    assert!((s - 1.0).abs() < 1e-6, "normalised weights must sum to 1");
}

#[test]
fn shared_expert_is_added_unscaled() {
    let shapes = tiny_shapes();
    let layer = moe_layer(&shapes);
    let hidden = shapes.hidden_size;
    let mut m = tiny_moe(&shapes, &layer, 0xc33ee, false);
    let zero = HostExperts::Bf16(nv_models::laguna_wgpu::weights::HostBf16ExpertStack {
        w: vec![0u16; shapes.num_experts * hidden * layer.ffn_intermediate],
        e: shapes.num_experts,
        n: hidden,
        k: layer.ffn_intermediate,
    });
    m.experts_down = zero;

    let x = tiny_x(0xd44, hidden);
    let got = moe::ref_moe(&shapes, &m, &x).unwrap();
    let shared = moe::ref_shared_expert(&shapes, &m, &x).unwrap();
    assert!(
        shapes.routed_scaling != 1.0,
        "tiny config must exercise a non-unit routed scaling"
    );
    for i in 0..hidden {
        assert_eq!(
            got[i], shared[i],
            "routed_scaling must not touch the shared expert at {i}"
        );
    }
}

#[test]
fn moe_layer_bf16_matches_cpu_oracle() {
    if have_gpu().is_none() {
        return;
    }
    let shapes = tiny_shapes();
    let layer = moe_layer(&shapes);
    for s in 0..8u64 {
        let m = tiny_moe(&shapes, &layer, 0x5a1a_d000_0001 + s * 0x9e37, false);
        let x = tiny_x(0x9e11 + s * 0x1f13, shapes.hidden_size);
        let want = moe::ref_moe(&shapes, &m, &x).unwrap();
        let got = gpu_moe(&shapes, &layer, &m, &x);
        let (abs, rel) = rel_err(&got, &want);
        let (ids, _) = moe::ref_router_topk(&shapes, &m, &x).unwrap();
        eprintln!("[moe bf16] seed {s} experts {ids:?} max_abs={abs:.6} rel={rel:.6}");
        assert_eq!(
            got, want,
            "seed {s}: the bf16 moe graph must reproduce the order-replicating oracle \
             exactly (max_abs {abs}, rel {rel})"
        );
    }
}

#[test]
fn moe_layer_nvfp4_matches_cpu_oracle() {
    if have_gpu().is_none() {
        return;
    }
    let shapes = tiny_shapes();
    let layer = moe_layer(&shapes);
    let mut worst = 0f32;
    for s in 0..8u64 {
        let m = tiny_moe(&shapes, &layer, 0x5a1a_d000_0002 + s * 0x9e37, true);
        let x = tiny_x(0x9e12 + s * 0x1f13, shapes.hidden_size);
        let want = moe::ref_moe(&shapes, &m, &x).unwrap();
        let got = gpu_moe(&shapes, &layer, &m, &x);
        let (abs, rel) = rel_err(&got, &want);
        let (ids, _) = moe::ref_router_topk(&shapes, &m, &x).unwrap();
        eprintln!("[moe nvfp4] seed {s} experts {ids:?} max_abs={abs:.6} rel={rel:.6}");
        worst = worst.max(rel);
        assert!(
            rel < 6e-2,
            "seed {s}: nvfp4 moe rel err {rel} (abs {abs}) exceeds the chained substrate \
             budget; see nvfp4_gemv_substrate_noise_floor for the single-gemv floor"
        );
        let shared = moe::ref_shared_expert(&shapes, &m, &x).unwrap();
        let (_, rel_shared_only) = rel_err(&want, &shared);
        assert!(
            rel_shared_only > 10.0 * rel.max(1e-3),
            "seed {s}: gpu/oracle disagreement {rel} is not small against {rel_shared_only}, \
             the error of omitting the routed experts entirely -- the routed path is not \
             carrying its weight"
        );
    }
    eprintln!("[moe nvfp4] worst rel over 8 seeds = {worst:.6}");
}

#[test]
fn moe_layer_is_deterministic_across_replays() {
    if have_gpu().is_none() {
        return;
    }
    let shapes = tiny_shapes();
    let layer = moe_layer(&shapes);
    let m = tiny_moe(&shapes, &layer, 0x5a1a_d000_0003, false);
    let x = tiny_x(0x9e13, shapes.hidden_size);
    let a = gpu_moe(&shapes, &layer, &m, &x);
    let b = gpu_moe(&shapes, &layer, &m, &x);
    assert_eq!(a, b, "moe graph replay must be bit-identical");
}

#[test]
fn nvfp4_gemv_substrate_noise_floor() {
    let Some(ctx) = have_gpu() else {
        return;
    };
    let s = Sources::new();
    let mut r = Lcg::new(0x777);
    let n = 128usize;
    let k = 128usize;
    let l = HostLin::Nvfp4(nvfp4_lin(&mut r, n, k, 0.15));
    let x = tiny_x(0x778, k);
    let want = nv_models::laguna_wgpu::ref_gemv_lin(&l, &x);

    let mut b = Builder::new(ctx);
    let bits: Vec<u16> = x.iter().map(|v| bf16_bits(*v)).collect();
    let xb = b.upload_u32("d-x", &pack_pairs(&bits));
    let out = b.zeros("d-out", (n / 2 * 4) as u64);
    let g = nv_models::laguna_wgpu::gpu::upload_lin(
        &mut b,
        "d-l",
        &l,
        nv_models::laguna_wgpu::gpu::W8Scope::Ffn,
    );
    let sc = nv_models::laguna_wgpu::gpu::alloc_lin_scratch(&mut b, "d-l", &g);
    nv_models::laguna_wgpu::gpu::push_lin_gemv(&mut b, &s, "d-gemv", &g, &sc, &xb, &out).unwrap();
    b.flush_staging();
    run_passes(ctx, &b.passes);
    let got_w: Vec<u32> = dispatch::read_back(ctx, &out, n / 2).unwrap();
    let mut got = vec![0f32; n];
    for (i, w) in got_w.iter().enumerate() {
        got[2 * i] = bf16_val((*w & 0xffff) as u16);
        got[2 * i + 1] = bf16_val((*w >> 16) as u16);
    }
    let mut worst = 0f32;
    for i in 0..n {
        worst = worst.max((got[i] - want[i]).abs() / want[i].abs().max(1e-6));
    }
    eprintln!("[nvfp4 substrate] one gemv, worst per-elem rel={worst:.6}");
    assert!(
        worst < 1e-2,
        "gpu::push_lin_gemv vs mod::ref_gemv_lin disagree by {worst} on a single nvfp4 \
         projection; no moe code is involved, so any moe nvfp4 budget starts here"
    );
}
