#![cfg(feature = "laguna-wip")]

mod common;
use common::worst_rel;
use nv_models::laguna::LagunaConfig;
use nv_models::laguna_wgpu::config::{FfnKind, GateKind, LagunaShapes, LayerShape};
use nv_models::laguna_wgpu::dense::{build_dense_mlp, ref_dense_mlp};
use nv_models::laguna_wgpu::gpu::{Builder, Pass, Sources};
use nv_models::laguna_wgpu::weights::{
    bf16_bits, bf16_val, pack_pairs, quantize_nvfp4_host, random_host_weights, HostDenseMlp,
    HostFfn, HostLin,
};

const CFG: &str = r#"{
    "architectures": ["LagunaForCausalLM"],
    "model_type": "laguna",
    "vocab_size": 64,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
    "norm_topk_prob": true,
    "mlp_only_layers": [0],
    "decoder_sparse_step": 1,
    "tie_word_embeddings": false,
    "gating": "per-head",
    "sliding_window": 8,
    "moe_routed_scaling_factor": 2.5,
    "moe_router_logit_softcapping": 5.0,
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
    "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
    "mlp_layer_types": ["dense", "sparse", "sparse", "dense"],
    "num_attention_heads_per_layer": [4, 8, 8, 4]
}"#;

fn shapes() -> LagunaShapes {
    let cfg = LagunaConfig::from_hf_json_str(CFG).unwrap();
    LagunaShapes::derive(&cfg, 32).unwrap()
}

fn dense_layer(s: &LagunaShapes) -> LayerShape {
    *s.layer(0)
}

fn host_dense(s: &LagunaShapes) -> HostDenseMlp {
    let hw = random_host_weights(s, 0x1234_5678_9abc_def0);
    match &hw.layers[0].ffn {
        HostFfn::Dense(d) => (**d).clone(),
        HostFfn::Moe(_) => panic!("layer 0 must be dense"),
    }
}

fn quantized(d: &HostDenseMlp) -> HostDenseMlp {
    let q = |l: &HostLin| match l {
        HostLin::Bf16(b) => HostLin::Nvfp4(quantize_nvfp4_host(&b.w, b.n, b.k)),
        HostLin::Nvfp4(_) => l.clone(),
    };
    HostDenseMlp {
        gate: q(&d.gate),
        up: q(&d.up),
        down: q(&d.down),
    }
}

fn input_vec(hidden: usize, seed: u64) -> Vec<f32> {
    let mut st = seed | 1;
    (0..hidden)
        .map(|_| {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((st >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0;
            bf16_val(bf16_bits(u))
        })
        .collect()
}

fn run(ctx: &nv_kernels::wgpu_backend::WgpuContext, passes: &[Pass]) {
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
}

fn gpu_dense(w: &HostDenseMlp, x: &[f32]) -> Option<Vec<f32>> {
    let ctx = match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no wgpu adapter: {e}");
            return None;
        }
    };
    let s = shapes();
    let layer = dense_layer(&s);
    let hidden = s.hidden_size;
    let src = Sources::new();
    let mut b = Builder::new(ctx);
    let xbits: Vec<u16> = x.iter().map(|v| bf16_bits(*v)).collect();
    let xbuf = b.upload_u32("lgw-test-x", &pack_pairs(&xbits));
    let out = b.zeros("lgw-test-out", (hidden * 2) as u64);
    build_dense_mlp(&mut b, &src, &s, &layer, w, &xbuf, &out).unwrap();
    b.flush_staging();
    run(ctx, &b.passes);
    let words: Vec<u32> =
        nv_kernels::wgpu_backend::dispatch::read_back(ctx, &out, hidden / 2).unwrap();
    let mut y = vec![0f32; hidden];
    for (i, w) in words.iter().enumerate() {
        y[2 * i] = bf16_val((*w & 0xffff) as u16);
        y[2 * i + 1] = bf16_val((*w >> 16) as u16);
    }
    Some(y)
}

#[test]
fn config_marks_layer_zero_dense_at_intermediate_size() {
    let s = shapes();
    assert_eq!(s.layer(0).ffn_kind, FfnKind::Dense);
    assert_eq!(s.layer(3).ffn_kind, FfnKind::Dense);
    assert_eq!(s.layer(1).ffn_kind, FfnKind::Moe);
    assert_eq!(s.layer(0).ffn_intermediate, 128);
    assert_eq!(s.layer(1).ffn_intermediate, 64);
    assert_eq!(s.dense_intermediate_size, 128);
    assert_eq!(s.dense_layer_indices(), vec![0, 3]);
    assert!((s.router_softcap - 5.0).abs() < 1e-6);
    assert!((s.routed_scaling - 2.5).abs() < 1e-6);
}

#[test]
fn random_host_weights_match_derived_geometry() {
    let s = shapes();
    let hw = random_host_weights(&s, 7);
    assert_eq!(hw.embed.len(), s.vocab_size * s.hidden_size);
    assert_eq!(hw.lm_head.len(), s.vocab_size * s.hidden_size);
    assert_eq!(hw.final_norm.len(), s.hidden_size);
    assert_eq!(hw.layers.len(), s.num_layers);
    for li in 0..s.num_layers {
        let ls = s.layer(li);
        let l = &hw.layers[li];
        assert_eq!(l.kind, ls.attn_kind);
        assert_eq!(l.input_ln.len(), s.hidden_size);
        assert_eq!(l.post_attn_ln.len(), s.hidden_size);
        assert_eq!(l.attn.q.n(), ls.q_rows);
        assert_eq!(l.attn.q.k(), s.hidden_size);
        assert_eq!(l.attn.k.n(), ls.kv_rows);
        assert_eq!(l.attn.v.n(), ls.kv_rows);
        assert_eq!(l.attn.o.n(), s.hidden_size);
        assert_eq!(l.attn.o.k(), ls.q_rows);
        assert_eq!(l.attn.q_norm.len(), ls.head_dim);
        assert_eq!(l.attn.k_norm.len(), ls.head_dim);
        assert_eq!(s.gate_kind, GateKind::PerHead);
        let g = l.attn.g.as_ref().expect("per-head gating needs g_proj");
        assert_eq!(g.n(), ls.gate_rows);
        assert_eq!(g.n(), ls.num_q_heads);
        match (&l.ffn, ls.ffn_kind) {
            (HostFfn::Dense(d), FfnKind::Dense) => {
                assert_eq!(d.gate.n(), ls.ffn_intermediate);
                assert_eq!(d.up.n(), ls.ffn_intermediate);
                assert_eq!(d.down.n(), s.hidden_size);
                assert_eq!(d.down.k(), ls.ffn_intermediate);
            }
            (HostFfn::Moe(m), FfnKind::Moe) => {
                assert_eq!(m.router.n, s.num_experts);
                assert_eq!(m.router.k, s.hidden_size);
                assert_eq!(m.selection_bias.len(), s.num_experts);
                assert_eq!(m.experts_gate.num_experts(), s.num_experts);
                assert_eq!(m.experts_gate.n(), s.moe_intermediate_size);
                assert_eq!(m.experts_down.n(), s.hidden_size);
                assert_eq!(m.experts_down.k(), s.moe_intermediate_size);
                assert_eq!(m.shared_gate.n(), s.shared_expert_intermediate_size);
                assert_eq!(m.shared_down.k(), s.shared_expert_intermediate_size);
            }
            _ => panic!("layer {li} ffn kind mismatch"),
        }
    }
}

#[test]
fn tiny_weight_rng_spans_both_signs() {
    let mut rng = nv_models::laguna_wgpu::weights::Lcg::new(3);
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    let mut pos = 0usize;
    for _ in 0..20000 {
        let v = rng.next_f32();
        lo = lo.min(v);
        hi = hi.max(v);
        if v > 0.0 {
            pos += 1;
        }
    }
    assert!(
        lo < -0.9 && hi > 0.9,
        "rng range [{lo}, {hi}] is not symmetric"
    );
    assert!(
        (4000..16000).contains(&pos),
        "rng produced {pos}/20000 positive samples"
    );

    let s = shapes();
    let hw = random_host_weights(&s, 5);
    let neg = hw.embed.iter().filter(|b| bf16_val(**b) < 0.0).count();
    assert!(
        neg > hw.embed.len() / 4 && neg < 3 * hw.embed.len() / 4,
        "embed sign balance {neg}/{}",
        hw.embed.len()
    );
}

#[test]
fn random_host_weights_are_deterministic_per_seed() {
    let s = shapes();
    let a = random_host_weights(&s, 99);
    let b = random_host_weights(&s, 99);
    let c = random_host_weights(&s, 100);
    assert_eq!(a.embed, b.embed);
    assert_eq!(a.lm_head, b.lm_head);
    assert_eq!(a.layers[0].input_ln, b.layers[0].input_ln);
    assert_ne!(a.embed, c.embed);
}

#[test]
fn ref_dense_mlp_agrees_with_independent_swiglu() {
    let s = shapes();
    let layer = dense_layer(&s);
    let w = host_dense(&s);
    let x = input_vec(s.hidden_size, 0xfeed);
    let got = ref_dense_mlp(&s, &layer, &w, &x).unwrap();

    let inter = layer.ffn_intermediate;
    let hidden = s.hidden_size;
    let lin = |l: &HostLin, v: &[f32], n: usize, k: usize| -> Vec<f32> {
        let bits = match l {
            HostLin::Bf16(b) => &b.w,
            HostLin::Nvfp4(_) => panic!("bf16 only"),
        };
        (0..n)
            .map(|r| {
                let mut acc = 0f64;
                for c in 0..k {
                    acc += bf16_val(bits[r * k + c]) as f64 * v[c] as f64;
                }
                acc as f32
            })
            .collect()
    };
    let g = lin(&w.gate, &x, inter, hidden);
    let u = lin(&w.up, &x, inter, hidden);
    let act: Vec<f32> = (0..inter)
        .map(|i| {
            let gv = g[i] as f64;
            (gv / (1.0 + (-gv).exp())) as f32 * u[i]
        })
        .collect();
    let want = lin(&w.down, &act, hidden, inter);

    let e = worst_rel(&got, &want);
    assert!(e < 2e-2, "dense oracle vs f64 swiglu worst rel {e}");
    assert!(got.iter().all(|v| v.is_finite()));
}

#[test]
fn ref_dense_mlp_rejects_shape_mismatch() {
    let s = shapes();
    let moe_layer = *s.layer(1);
    let w = host_dense(&s);
    let x = input_vec(s.hidden_size, 1);
    assert!(ref_dense_mlp(&s, &moe_layer, &w, &x).is_err());
    assert!(ref_dense_mlp(&s, &dense_layer(&s), &w, &x[..4]).is_err());
}

#[test]
fn gpu_dense_mlp_bf16_matches_cpu_oracle() {
    let s = shapes();
    let layer = dense_layer(&s);
    let w = host_dense(&s);
    let x = input_vec(s.hidden_size, 0xabcdef);
    let want = ref_dense_mlp(&s, &layer, &w, &x).unwrap();
    let Some(got) = gpu_dense(&w, &x) else {
        return;
    };
    let e = worst_rel(&got, &want);
    let mag = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    let nz = got.iter().filter(|v| **v != 0.0).count();
    eprintln!(
        "laguna dense bf16 worst rel {e:.3e} max|y| {mag:.4} nonzero {nz}/{}",
        got.len()
    );
    assert!(mag > 1e-3 && nz > got.len() / 2, "degenerate dense output");
    assert!(e < 5e-3, "gpu dense bf16 worst rel {e}");
}

#[test]
fn gpu_dense_comparison_detects_a_perturbed_weight() {
    let s = shapes();
    let layer = dense_layer(&s);
    let w = host_dense(&s);
    let x = input_vec(s.hidden_size, 0xabcdef);
    let Some(got) = gpu_dense(&w, &x) else {
        return;
    };
    let mut bad = w.clone();
    match &mut bad.down {
        HostLin::Bf16(b) => {
            let old = bf16_val(b.w[0]);
            b.w[0] = bf16_bits(old + 0.5);
        }
        HostLin::Nvfp4(_) => unreachable!(),
    }
    let want_bad = ref_dense_mlp(&s, &layer, &bad, &x).unwrap();
    let e = worst_rel(&got, &want_bad);
    eprintln!("laguna dense perturbed worst rel {e:.3e}");
    assert!(
        e > 5e-3,
        "a perturbed down_proj must break the oracle comparison, got {e}"
    );
}

#[test]
fn gpu_dense_mlp_nvfp4_matches_cpu_oracle() {
    let s = shapes();
    let layer = dense_layer(&s);
    let w = quantized(&host_dense(&s));
    assert!(w.gate.is_nvfp4() && w.up.is_nvfp4() && w.down.is_nvfp4());
    let x = input_vec(s.hidden_size, 0x13579);
    let want = ref_dense_mlp(&s, &layer, &w, &x).unwrap();
    let Some(got) = gpu_dense(&w, &x) else {
        return;
    };
    let e = worst_rel(&got, &want);
    let mag = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    let nz = got.iter().filter(|v| **v != 0.0).count();
    eprintln!(
        "laguna dense nvfp4 worst rel {e:.3e} max|y| {mag:.4} nonzero {nz}/{}",
        got.len()
    );
    assert!(mag > 1e-3 && nz > got.len() / 2, "degenerate dense output");
    assert!(e < 2e-2, "gpu dense nvfp4 worst rel {e}");
}
