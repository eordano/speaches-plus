#![cfg(feature = "laguna-wip")]

mod common;
use common::CFG;
use common::env_lock;
use common::worst_rel;
use std::sync::{Mutex, MutexGuard, OnceLock};

use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use nv_models::laguna::LagunaConfig;
use nv_models::laguna_wgpu::config::{LagunaShapes, LayerShape};
use nv_models::laguna_wgpu::dense::{build_dense_mlp, ref_dense_mlp};
use nv_models::laguna_wgpu::gpu::{Builder, Pass, Sources};
use nv_models::laguna_wgpu::moe::{build_moe_layer, ref_moe};
use nv_models::laguna_wgpu::weights::{
    bf16_bits, bf16_val, dequantize_nvfp4_host, expert_slice, pack_pairs, quantize_nvfp4_host,
    random_host_weights, stack_bf16_host, stack_nvfp4_host, HostAttention, HostBf16Lin,
    HostDenseMlp, HostExperts, HostFfn, HostLayer, HostLin, HostMoe, HostNvfp4Lin, HostWeights,
};
use nv_models::laguna_wgpu::{ref_argmax, reference_step, LagunaWgpu, RefState};

const MAX_SEQ: usize = 24;
const TOKENS: [u32; 8] = [3, 17, 5, 42, 8, 61, 12, 30];
const W8_GROUP: usize = 32;

struct W8Env;

impl W8Env {
    fn set(mode: &str, group: usize) -> Self {
        std::env::set_var("NV_LAGUNA_WGPU_W8", mode);
        std::env::set_var("NV_LAGUNA_WGPU_W8_GROUP", group.to_string());
        Self
    }
}

impl Drop for W8Env {
    fn drop(&mut self) {
        std::env::remove_var("NV_LAGUNA_WGPU_W8");
        std::env::remove_var("NV_LAGUNA_WGPU_W8_GROUP");
    }
}

fn have_gpu() -> Option<&'static WgpuContext> {
    match WgpuContext::shared() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP: no wgpu adapter: {e}");
            None
        }
    }
}

fn subgroups_ok(ctx: &WgpuContext) -> bool {
    let ok = nv_kernels::wgpu_backend::kernels::gemv_nvfp4_v2::subgroup32_ok(ctx);
    if !ok {
        eprintln!("SKIP: adapter has no 32-wide subgroups; the int8 gemv needs them");
    }
    ok
}

fn shapes_of(cfg: &LagunaConfig) -> LagunaShapes {
    LagunaShapes::derive(cfg, MAX_SEQ).unwrap()
}

fn to_nvfp4_lin(l: &HostLin) -> HostLin {
    match l {
        HostLin::Bf16(b) => HostLin::Nvfp4(quantize_nvfp4_host(&b.w, b.n, b.k)),
        HostLin::Nvfp4(_) => l.clone(),
    }
}

fn to_nvfp4_experts(e: &HostExperts) -> HostExperts {
    match e {
        HostExperts::Nvfp4(_) => e.clone(),
        HostExperts::Bf16(s) => {
            let mats: Vec<HostNvfp4Lin> = (0..s.e)
                .map(|i| {
                    let w = &s.w[i * s.n * s.k..(i + 1) * s.n * s.k];
                    quantize_nvfp4_host(w, s.n, s.k)
                })
                .collect();
            HostExperts::Nvfp4(stack_nvfp4_host(&mats))
        }
    }
}

fn to_nvfp4_mlp(m: &HostDenseMlp) -> HostDenseMlp {
    HostDenseMlp {
        gate: to_nvfp4_lin(&m.gate),
        up: to_nvfp4_lin(&m.up),
        down: to_nvfp4_lin(&m.down),
    }
}

fn to_nvfp4_moe(m: &HostMoe) -> HostMoe {
    HostMoe {
        router: m.router.clone(),
        selection_bias: m.selection_bias.clone(),
        experts_gate: to_nvfp4_experts(&m.experts_gate),
        experts_up: to_nvfp4_experts(&m.experts_up),
        experts_down: to_nvfp4_experts(&m.experts_down),
        shared_gate: to_nvfp4_lin(&m.shared_gate),
        shared_up: to_nvfp4_lin(&m.shared_up),
        shared_down: to_nvfp4_lin(&m.shared_down),
    }
}

fn to_nvfp4(hw: &HostWeights) -> HostWeights {
    HostWeights {
        embed: hw.embed.clone(),
        final_norm: hw.final_norm.clone(),
        lm_head: hw.lm_head.clone(),
        layers: hw
            .layers
            .iter()
            .map(|l| HostLayer {
                kind: l.kind,
                input_ln: l.input_ln.clone(),
                post_attn_ln: l.post_attn_ln.clone(),
                attn: HostAttention {
                    q: to_nvfp4_lin(&l.attn.q),
                    k: to_nvfp4_lin(&l.attn.k),
                    v: to_nvfp4_lin(&l.attn.v),
                    o: to_nvfp4_lin(&l.attn.o),
                    g: l.attn.g.as_ref().map(to_nvfp4_lin),
                    q_norm: l.attn.q_norm.clone(),
                    k_norm: l.attn.k_norm.clone(),
                },
                ffn: match &l.ffn {
                    HostFfn::Dense(d) => HostFfn::Dense(Box::new(to_nvfp4_mlp(d))),
                    HostFfn::Moe(m) => HostFfn::Moe(Box::new(to_nvfp4_moe(m))),
                },
            })
            .collect(),
    }
}

fn dequant_bf16_lin(l: &HostNvfp4Lin) -> HostBf16Lin {
    let gi = if l.input_global == 0.0 || !l.input_global.is_finite() {
        1.0
    } else {
        l.input_global
    };
    let w = dequantize_nvfp4_host(l);
    HostBf16Lin {
        w: w.iter().map(|v| bf16_bits(v * gi)).collect(),
        n: l.n,
        k: l.k,
    }
}

fn twin_lin(l: &HostLin) -> HostLin {
    match l {
        HostLin::Nvfp4(x) => HostLin::Bf16(dequant_bf16_lin(x)),
        HostLin::Bf16(_) => l.clone(),
    }
}

fn twin_experts(e: &HostExperts) -> HostExperts {
    match e {
        HostExperts::Bf16(_) => e.clone(),
        HostExperts::Nvfp4(s) => {
            let mats: Vec<HostBf16Lin> = (0..s.e)
                .map(|i| dequant_bf16_lin(&expert_slice(s, i)))
                .collect();
            HostExperts::Bf16(stack_bf16_host(&mats))
        }
    }
}

fn twin_mlp(m: &HostDenseMlp) -> HostDenseMlp {
    HostDenseMlp {
        gate: twin_lin(&m.gate),
        up: twin_lin(&m.up),
        down: twin_lin(&m.down),
    }
}

fn twin_moe(m: &HostMoe) -> HostMoe {
    HostMoe {
        router: m.router.clone(),
        selection_bias: m.selection_bias.clone(),
        experts_gate: twin_experts(&m.experts_gate),
        experts_up: twin_experts(&m.experts_up),
        experts_down: twin_experts(&m.experts_down),
        shared_gate: twin_lin(&m.shared_gate),
        shared_up: twin_lin(&m.shared_up),
        shared_down: twin_lin(&m.shared_down),
    }
}

fn twin_weights(hw: &HostWeights) -> HostWeights {
    HostWeights {
        embed: hw.embed.clone(),
        final_norm: hw.final_norm.clone(),
        lm_head: hw.lm_head.clone(),
        layers: hw
            .layers
            .iter()
            .map(|l| HostLayer {
                kind: l.kind,
                input_ln: l.input_ln.clone(),
                post_attn_ln: l.post_attn_ln.clone(),
                attn: HostAttention {
                    q: twin_lin(&l.attn.q),
                    k: twin_lin(&l.attn.k),
                    v: twin_lin(&l.attn.v),
                    o: twin_lin(&l.attn.o),
                    g: l.attn.g.as_ref().map(twin_lin),
                    q_norm: l.attn.q_norm.clone(),
                    k_norm: l.attn.k_norm.clone(),
                },
                ffn: match &l.ffn {
                    HostFfn::Dense(d) => HostFfn::Dense(Box::new(twin_mlp(d))),
                    HostFfn::Moe(m) => HostFfn::Moe(Box::new(twin_moe(m))),
                },
            })
            .collect(),
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
        panic!("graph validation: {e}");
    }
}

fn quant_dispatches(passes: &[Pass]) -> usize {
    passes
        .iter()
        .filter(|p| p.entry == "lgw_quant_rows")
        .count()
}

fn int8_dispatches(passes: &[Pass]) -> usize {
    passes
        .iter()
        .filter(|p| p.entry.starts_with("lgw_gemv_i8"))
        .count()
}

fn tiny_x(seed: u64, hidden: usize) -> Vec<f32> {
    let mut st = seed | 1;
    (0..hidden)
        .map(|_| {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((st >> 33) as u32 as f32 / (1u64 << 31) as f32) - 1.0;
            bf16_val(bf16_bits(u))
        })
        .collect()
}

fn read_packed(ctx: &WgpuContext, buf: &wgpu::Buffer, n: usize) -> Vec<f32> {
    let words: Vec<u32> = dispatch::read_back(ctx, buf, n / 2).unwrap();
    let mut y = vec![0f32; n];
    for (i, w) in words.iter().enumerate() {
        y[2 * i] = bf16_val((*w & 0xffff) as u16);
        y[2 * i + 1] = bf16_val((*w >> 16) as u16);
    }
    y
}

struct Built {
    y: Vec<f32>,
    quant: usize,
    i8: usize,
    total: usize,
}

fn build_and_run<F>(ctx: &'static WgpuContext, hidden: usize, f: F) -> Built
where
    F: FnOnce(&mut Builder, &Sources, &wgpu::Buffer, &wgpu::Buffer),
{
    let s = Sources::new();
    let mut b = Builder::new(ctx);
    let x = tiny_x(0xbeef_0007, hidden);
    let bits: Vec<u16> = x.iter().map(|v| bf16_bits(*v)).collect();
    let xb = b.upload_u32("w8-x", &pack_pairs(&bits));
    let out = b.zeros("w8-out", (hidden * 2) as u64);
    f(&mut b, &s, &xb, &out);
    b.flush_staging();
    run_passes(ctx, &b.passes);
    Built {
        y: read_packed(ctx, &out, hidden),
        quant: quant_dispatches(&b.passes),
        i8: int8_dispatches(&b.passes),
        total: b.passes.len(),
    }
}

fn dense_layer(s: &LagunaShapes) -> LayerShape {
    let idx = *s
        .dense_layer_indices()
        .first()
        .expect("config must have a dense layer");
    *s.layer(idx)
}

fn moe_layer(s: &LagunaShapes) -> LayerShape {
    let idx = *s
        .moe_layer_indices()
        .first()
        .expect("config must have a moe layer");
    *s.layer(idx)
}

#[test]
fn dense_mlp_int8_tracks_nvfp4_and_deletes_its_activation_quant_dispatches() {
    let _g = env_lock();
    let Some(ctx) = have_gpu() else { return };
    if !subgroups_ok(ctx) {
        return;
    }
    let cfg = LagunaConfig::from_hf_json_str(CFG).unwrap();
    let shapes = shapes_of(&cfg);
    let layer = dense_layer(&shapes);
    let hw = to_nvfp4(&random_host_weights(&shapes, 0x1234_5678_9abc_def0));
    let HostFfn::Dense(mlp) = &hw.layers[layer.idx].ffn else {
        panic!("layer {} must be dense", layer.idx);
    };
    let x = tiny_x(0xbeef_0007, shapes.hidden_size);
    let want_nvfp4 = ref_dense_mlp(&shapes, &layer, mlp, &x).unwrap();
    let want = ref_dense_mlp(&shapes, &layer, &twin_mlp(mlp), &x).unwrap();

    let base = build_and_run(ctx, shapes.hidden_size, |b, s, xb, out| {
        build_dense_mlp(b, s, &shapes, &layer, mlp, xb, out).unwrap();
    });
    let w8 = {
        let _e = W8Env::set("ffn", W8_GROUP);
        build_and_run(ctx, shapes.hidden_size, |b, s, xb, out| {
            build_dense_mlp(b, s, &shapes, &layer, mlp, xb, out).unwrap();
        })
    };

    eprintln!(
        "[laguna-w8] dense mlp: nvfp4 {} passes ({} quant), int8 {} passes ({} quant, {} i8 gemv)",
        base.total, base.quant, w8.total, w8.quant, w8.i8
    );
    assert_eq!(
        base.quant, 3,
        "nvfp4 dense mlp should quantize x three times"
    );
    assert_eq!(w8.quant, 0, "W8A16 must consume bf16 activations directly");
    assert_eq!(
        w8.i8, 3,
        "all three dense projections should route to the int8 gemv"
    );
    assert_eq!(base.total - w8.total, 3);

    let r_base = worst_rel(&base.y, &want_nvfp4);
    let r_w8 = worst_rel(&w8.y, &want);
    let r_nvfp4_w16 = worst_rel(&base.y, &want);
    eprintln!(
        "[laguna-w8] dense mlp: nvfp4 vs its own oracle {r_base:.3e}; \
         vs the W16 twin: nvfp4 {r_nvfp4_w16:.3e}, int8-g{W8_GROUP} {r_w8:.3e}"
    );
    assert!(w8.y.iter().all(|v| v.is_finite()));
    assert!(
        r_w8 < 2e-2,
        "int8 dense mlp drifts {r_w8:.3e} from the bf16 twin of the same weights"
    );
    assert!(
        r_w8 < r_nvfp4_w16,
        "int8-g{W8_GROUP} ({r_w8:.3e}) should be closer to the true weights than the nvfp4 \
         source it was re-encoded from ({r_nvfp4_w16:.3e}) -- finer-than-source is the whole \
         premise of this conversion"
    );
}

#[test]
fn moe_layer_int8_tracks_nvfp4_and_deletes_its_activation_quant_dispatches() {
    let _g = env_lock();
    let Some(ctx) = have_gpu() else { return };
    if !subgroups_ok(ctx) {
        return;
    }
    let cfg = LagunaConfig::from_hf_json_str(CFG).unwrap();
    let shapes = shapes_of(&cfg);
    let layer = moe_layer(&shapes);
    let hw = to_nvfp4(&random_host_weights(&shapes, 0x0f0f_1234_5678_0001));
    let HostFfn::Moe(m) = &hw.layers[layer.idx].ffn else {
        panic!("layer {} must be moe", layer.idx);
    };
    let x = tiny_x(0xbeef_0007, shapes.hidden_size);
    let tw = twin_moe(m);
    let want_nvfp4 = ref_moe(&shapes, m, &x).unwrap();
    let want = ref_moe(&shapes, &tw, &x).unwrap();

    let base = build_and_run(ctx, shapes.hidden_size, |b, s, xb, out| {
        build_moe_layer(b, s, &shapes, &layer, m, xb, out).unwrap();
    });
    let w8 = {
        let _e = W8Env::set("ffn", W8_GROUP);
        build_and_run(ctx, shapes.hidden_size, |b, s, xb, out| {
            build_moe_layer(b, s, &shapes, &layer, m, xb, out).unwrap();
        })
    };

    eprintln!(
        "[laguna-w8] moe layer: nvfp4 {} passes ({} quant), int8 {} passes ({} quant, {} i8 gemv)",
        base.total, base.quant, w8.total, w8.quant, w8.i8
    );

    assert_eq!(base.quant, 5, "nvfp4 moe layer should quantize five times");
    assert_eq!(w8.quant, 0, "W8A16 must consume bf16 activations directly");
    assert_eq!(
        w8.i8, 6,
        "3 routed + 3 shared projections should route to int8"
    );
    assert_eq!(base.total - w8.total, 5);

    let r_base = worst_rel(&base.y, &want_nvfp4);
    let r_w8 = worst_rel(&w8.y, &want);
    let r_nvfp4_w16 = worst_rel(&base.y, &want);
    eprintln!(
        "[laguna-w8] moe: nvfp4 vs its own oracle {r_base:.3e}; \
         vs the W16 twin: nvfp4 {r_nvfp4_w16:.3e}, int8-g{W8_GROUP} {r_w8:.3e}"
    );
    assert!(w8.y.iter().all(|v| v.is_finite()));
    assert!(
        r_w8 < 5e-2,
        "int8 moe drifts {r_w8:.3e} from the bf16 twin of the same weights"
    );
    assert!(
        r_w8 < r_nvfp4_w16,
        "int8-g{W8_GROUP} ({r_w8:.3e}) should be closer to the true weights than the nvfp4 \
         source it was re-encoded from ({r_nvfp4_w16:.3e})"
    );
}

#[test]
fn full_graph_int8_matches_argmax_and_cuts_the_per_token_dispatch_count() {
    let _g = env_lock();
    let Some(ctx) = have_gpu() else { return };
    if !subgroups_ok(ctx) {
        return;
    }
    let cfg = LagunaConfig::from_hf_json_str(CFG).unwrap();
    let shapes = shapes_of(&cfg);
    let hw = to_nvfp4(&random_host_weights(&shapes, 0x51ee_d0d0_1234_beef));

    let mut base = LagunaWgpu::new(cfg.clone(), &hw, MAX_SEQ).unwrap();
    let base_passes = base.pass_count();
    let mut base_logits = Vec::new();
    for i in 0..6 {
        let (_, l) = base.decode_step_logits(TOKENS[i % TOKENS.len()]).unwrap();
        base_logits.push(l);
    }

    let (w8_passes, w8_logits) = {
        let _e = W8Env::set("all", W8_GROUP);
        let mut m = LagunaWgpu::new(cfg.clone(), &hw, MAX_SEQ).unwrap();
        let p = m.pass_count();
        let mut out = Vec::new();
        for i in 0..6 {
            let (_, l) = m.decode_step_logits(TOKENS[i % TOKENS.len()]).unwrap();
            out.push(l);
        }
        (p, out)
    };

    eprintln!(
        "[laguna-w8] full graph: nvfp4 {base_passes} passes/token, int8 {w8_passes} passes/token, \
         delta {}",
        base_passes as i64 - w8_passes as i64
    );
    assert!(
        w8_passes < base_passes,
        "W8A16 should delete the lgw_quant_rows dispatches ({base_passes} -> {w8_passes})"
    );

    let tw = twin_weights(&hw);
    let mut st = RefState::new(&shapes);
    let mut w8_worst = 0f32;
    let mut nvfp4_worst = 0f32;
    let mut agree = 0usize;
    for i in 0..6 {
        let want = reference_step(&shapes, &tw, &mut st, TOKENS[i % TOKENS.len()]).unwrap();
        w8_worst = w8_worst.max(worst_rel(&w8_logits[i], &want));
        nvfp4_worst = nvfp4_worst.max(worst_rel(&base_logits[i], &want));
        if ref_argmax(&w8_logits[i]) == ref_argmax(&want) {
            agree += 1;
        }
    }
    eprintln!(
        "[laguna-w8] full graph over 6 steps vs the W16 twin: int8 worst rel {w8_worst:.3e}, \
         nvfp4 worst rel {nvfp4_worst:.3e}, int8 argmax agrees {agree}/6"
    );
    assert!(
        w8_logits.iter().all(|l| l.iter().all(|v| v.is_finite())),
        "int8 graph produced a non-finite logit"
    );
    assert!(
        w8_worst < nvfp4_worst,
        "the int8 graph ({w8_worst:.3e}) should track the true weights more closely than the \
         nvfp4 graph it was re-encoded from ({nvfp4_worst:.3e})"
    );
    assert_eq!(
        agree, 6,
        "int8 graph argmax disagrees with the W16 twin oracle"
    );
}

#[test]
fn a_group_that_does_not_divide_k_is_a_hard_error_not_a_silent_per_row_fallback() {
    let _g = env_lock();
    let Some(ctx) = have_gpu() else { return };
    let _e = W8Env::set("ffn", 128);
    let l = {
        let n = 32usize;
        let k = 64usize;
        let w: Vec<u16> = (0..n * k)
            .map(|i| bf16_bits(((i % 17) as f32 - 8.0) * 0.01))
            .collect();
        quantize_nvfp4_host(&w, n, k)
    };
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut b = Builder::new(ctx);
        nv_models::laguna_wgpu::gpu::upload_nvfp4_i8(&mut b, "w8-bad", &l);
    }));
    assert!(
        caught.is_err(),
        "group 128 with k=64 must be a hard error; a silent per-row fallback would ship the \
         configuration the quality battery rejected at 2.899e-1 mean KL"
    );
}

#[test]
fn bf16_projections_are_left_alone_even_with_the_flag_on() {
    let _g = env_lock();
    let Some(ctx) = have_gpu() else { return };
    if !subgroups_ok(ctx) {
        return;
    }
    let cfg = LagunaConfig::from_hf_json_str(CFG).unwrap();
    let shapes = shapes_of(&cfg);
    let layer = dense_layer(&shapes);
    let hw = random_host_weights(&shapes, 0x2222_3333_4444_5555);
    let HostFfn::Dense(mlp) = &hw.layers[layer.idx].ffn else {
        panic!("layer {} must be dense", layer.idx);
    };
    assert!(
        matches!(mlp.gate, HostLin::Bf16(_)),
        "random_host_weights must hand us a bf16 checkpoint for this test"
    );
    let _e = W8Env::set("all", W8_GROUP);
    let got = build_and_run(ctx, shapes.hidden_size, |b, s, xb, out| {
        build_dense_mlp(b, s, &shapes, &layer, mlp, xb, out).unwrap();
    });
    assert_eq!(
        got.i8, 0,
        "bf16 -> int8 is quantizing from native precision and measured REJECT; it must not fire"
    );
    let x = tiny_x(0xbeef_0007, shapes.hidden_size);
    let want = ref_dense_mlp(&shapes, &layer, mlp, &x).unwrap();
    assert!(worst_rel(&got.y, &want) < 2e-2);
}
