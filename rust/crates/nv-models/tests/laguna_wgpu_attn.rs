#![cfg(feature = "laguna-wip")]

mod common;
use common::worst_rel;
use nv_kernels::wgpu_backend::{dispatch, WgpuContext};
use nv_models::laguna::LagunaConfig;
use nv_models::laguna_wgpu::attn::{build_attn_layer, ref_attn, RopeTablesGpu};
use nv_models::laguna_wgpu::config::{rope_tables_from_inv_freq, LagunaShapes, LayerShape};
use nv_models::laguna_wgpu::gpu::{Builder, Pass, Sources, StepBuffers};
use nv_models::laguna_wgpu::weights::{
    bf16_bits, bf16_val, pack_pairs, quantize_nvfp4_host, HostAttention, HostBf16Lin, HostLin,
};
use nv_models::laguna_wgpu::{window_start, RefState};

const MAX_SEQ: usize = 32;
const STEPS: usize = 6;

fn cfg_json(gating: &str) -> String {
    format!(
        r#"{{
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
    "gating": {gating},
    "sliding_window": 3,
    "moe_routed_scaling_factor": 2.5,
    "rope_parameters": {{
        "full_attention": {{
            "rope_theta": 500000.0,
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 64,
            "beta_slow": 1.0,
            "beta_fast": 64.0,
            "attention_factor": 1.3465735902799727,
            "partial_rotary_factor": 0.5
        }},
        "sliding_attention": {{
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0
        }}
    }},
    "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
    "mlp_layer_types": ["dense", "sparse", "sparse", "dense"],
    "num_attention_heads_per_layer": [4, 8, 8, 4]
}}"#
    )
}

fn shapes(gating: &str) -> LagunaShapes {
    shapes_seq(gating, MAX_SEQ)
}

fn shapes_seq(gating: &str, max_seq: usize) -> LagunaShapes {
    let cfg = LagunaConfig::from_hf_json_str(&cfg_json(gating)).unwrap();
    LagunaShapes::derive(&cfg, max_seq).unwrap()
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    fn bits(&mut self, n: usize, scale: f32) -> Vec<u16> {
        (0..n).map(|_| bf16_bits(self.next() * scale)).collect()
    }

    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| bf16_val(bf16_bits(self.next() * scale)))
            .collect()
    }
}

fn lin(rng: &mut Rng, n: usize, k: usize, scale: f32) -> HostLin {
    HostLin::Bf16(HostBf16Lin {
        w: rng.bits(n * k, scale),
        n,
        k,
    })
}

fn host_attn(s: &LagunaShapes, l: &LayerShape, seed: u64) -> HostAttention {
    let mut rng = Rng::new(seed);
    let hidden = s.hidden_size;
    let g = if l.gate_rows > 0 {
        Some(lin(&mut rng, l.gate_rows, hidden, 0.4))
    } else {
        None
    };
    HostAttention {
        q: lin(&mut rng, l.q_rows, hidden, 0.3),
        k: lin(&mut rng, l.kv_rows, hidden, 0.3),
        v: lin(&mut rng, l.kv_rows, hidden, 0.3),
        o: lin(&mut rng, hidden, l.q_rows, 0.3),
        g,
        q_norm: rng.bits(l.head_dim, 0.8),
        k_norm: rng.bits(l.head_dim, 0.8),
    }
}

fn run(ctx: &WgpuContext, passes: &[Pass]) {
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

fn words_of(bits: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bits.len().div_ceil(2) * 4);
    for w in pack_pairs(bits) {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

fn unpack(words: &[u32], n: usize) -> Vec<f32> {
    let mut y = vec![0f32; n];
    for i in 0..n {
        let w = words[i / 2];
        y[i] = if i % 2 == 0 {
            bf16_val((w & 0xffff) as u16)
        } else {
            bf16_val((w >> 16) as u16)
        };
    }
    y
}

fn cpu_steps(
    s: &LagunaShapes,
    l: &LayerShape,
    w: &HostAttention,
    xs: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let mut st = RefState::new(s);
    let inv = if l.is_sliding() {
        &s.rope_inv_freq_sliding
    } else {
        &s.rope_inv_freq_full
    };
    let (cos, sin) = rope_tables_from_inv_freq(inv, s.max_seq_tokens);
    let half = inv.len();
    xs.iter()
        .enumerate()
        .map(|(pos, x)| {
            ref_attn(
                s,
                l,
                w,
                x,
                &cos[pos * half..(pos + 1) * half],
                &sin[pos * half..(pos + 1) * half],
                &mut st,
                pos,
            )
            .unwrap()
        })
        .collect()
}

struct GpuRun {
    outs: Vec<Vec<f32>>,
    replay0: Vec<f32>,
    passes: usize,
}

fn gpu_steps(
    s: &LagunaShapes,
    l: &LayerShape,
    w: &HostAttention,
    xs: &[Vec<f32>],
) -> Option<GpuRun> {
    let ctx = match WgpuContext::shared() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no wgpu adapter: {e}");
            return None;
        }
    };
    let hidden = s.hidden_size;
    let src = Sources::new();
    let mut b = Builder::new(ctx);
    let step = StepBuffers::alloc(&mut b);
    let xbuf = b.upload_u32("lgw-test-x", &pack_pairs(&vec![0u16; hidden]));
    let out = b.zeros("lgw-test-out", (hidden * 2) as u64);

    let inv = if l.is_sliding() {
        &s.rope_inv_freq_sliding
    } else {
        &s.rope_inv_freq_full
    };
    let (cos, sin) = rope_tables_from_inv_freq(inv, s.max_seq_tokens);
    let rope = RopeTablesGpu {
        cos: b.upload_f32("lgw-test-cos", &cos),
        sin: b.upload_f32("lgw-test-sin", &sin),
        half: inv.len(),
    };

    build_attn_layer(&mut b, &src, s, l, w, &xbuf, &out, &step, &rope).unwrap();
    b.flush_staging();

    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut outs = Vec::new();
    for (pos, x) in xs.iter().enumerate() {
        let bits: Vec<u16> = x.iter().map(|v| bf16_bits(*v)).collect();
        ctx.queue.write_buffer(&xbuf, 0, &words_of(&bits));
        let total = pos + 1;
        step.write(
            ctx,
            0,
            pos as u32,
            total as u32,
            window_start(total, Some(s.sliding_window)) as u32,
        );
        run(ctx, &b.passes);
        let words: Vec<u32> = dispatch::read_back(ctx, &out, hidden / 2).unwrap();
        outs.push(unpack(&words, hidden));
    }
    if let Some(e) = pollster::block_on(scope.pop()) {
        panic!("attn graph validation: {e}");
    }

    for (buf, bytes) in &b.state_buffers {
        ctx.queue.write_buffer(buf, 0, &vec![0u8; *bytes as usize]);
    }
    let bits: Vec<u16> = xs[0].iter().map(|v| bf16_bits(*v)).collect();
    ctx.queue.write_buffer(&xbuf, 0, &words_of(&bits));
    step.write(ctx, 0, 0, 1, 0);
    run(ctx, &b.passes);
    let words: Vec<u32> = dispatch::read_back(ctx, &out, hidden / 2).unwrap();

    Some(GpuRun {
        outs,
        replay0: unpack(&words, hidden),
        passes: b.passes.len(),
    })
}

fn check_layer(gating: &str, li: usize, seed: u64) {
    check_layer_seq(gating, li, seed, MAX_SEQ, STEPS);
}

fn check_layer_seq(gating: &str, li: usize, seed: u64, max_seq: usize, steps: usize) {
    check_layer_full(gating, li, seed, max_seq, steps, false, 3e-2);
}

fn quantized_attn(w: &HostAttention) -> HostAttention {
    let q = |l: &HostLin| match l {
        HostLin::Bf16(b) => HostLin::Nvfp4(quantize_nvfp4_host(&b.w, b.n, b.k)),
        HostLin::Nvfp4(_) => l.clone(),
    };
    HostAttention {
        q: q(&w.q),
        k: q(&w.k),
        v: q(&w.v),
        o: q(&w.o),
        g: w.g.as_ref().map(q),
        q_norm: w.q_norm.clone(),
        k_norm: w.k_norm.clone(),
    }
}

fn check_layer_full(
    gating: &str,
    li: usize,
    seed: u64,
    max_seq: usize,
    steps: usize,
    nvfp4: bool,
    tol: f32,
) {
    let s = shapes_seq(gating, max_seq);
    let l = *s.layer(li);
    let w = host_attn(&s, &l, seed);
    let w = if nvfp4 { quantized_attn(&w) } else { w };
    let mut rng = Rng::new(seed ^ 0xa5a5_5a5a);
    let xs: Vec<Vec<f32>> = (0..steps).map(|_| rng.vec(s.hidden_size, 0.7)).collect();

    let want = cpu_steps(&s, &l, &w, &xs);
    let energy = want[steps - 1].iter().fold(0f32, |a, v| a + v.abs());
    assert!(
        energy > 1e-2,
        "gating={gating} layer={li}: degenerate oracle output"
    );
    let Some(got) = gpu_steps(&s, &l, &w, &xs) else {
        return;
    };
    assert!(
        got.passes >= 8,
        "expected a real pass list, got {}",
        got.passes
    );
    assert_ne!(
        got.outs[0],
        got.outs[steps - 1],
        "gating={gating} layer={li}: output did not move across steps"
    );

    let mut worst = 0f32;
    for (i, (g, c)) in got.outs.iter().zip(&want).enumerate() {
        let r = worst_rel(g, c);
        assert!(
            r < tol,
            "gating={gating} layer={li} nvfp4={nvfp4} step={i}: worst rel {r:.3e}"
        );
        worst = worst.max(r);
    }
    assert_eq!(
        got.replay0, got.outs[0],
        "gating={gating} layer={li}: state reset did not reproduce step 0"
    );
    eprintln!(
        "laguna_wgpu attn gating={gating} layer={li} heads={} sliding={} steps={steps} cap={max_seq} nvfp4={nvfp4} worst rel {worst:.3e}",
        l.num_q_heads,
        l.is_sliding()
    );
}

fn cfg_json_hd128() -> String {
    r#"{
    "architectures": ["LagunaForCausalLM"],
    "model_type": "laguna",
    "vocab_size": 64,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 128,
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
    "sliding_window": 512,
    "moe_routed_scaling_factor": 2.5,
    "rope_parameters": {
        "full_attention": {
            "rope_theta": 500000.0,
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 8192,
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
    "num_attention_heads_per_layer": [4, 8]
}"#
    .to_string()
}

#[test]
fn real_dims_hd128_full_and_sliding_match_oracle() {
    let cfg = LagunaConfig::from_hf_json_str(&cfg_json_hd128()).unwrap();
    let s = LagunaShapes::derive(&cfg, 32).unwrap();
    assert_eq!(s.layer(0).head_dim, 128);
    assert_eq!(s.layer(0).rotary_dim, 64);
    assert!(!s.layer(0).is_sliding());
    assert_eq!(s.layer(1).rotary_dim, 128);
    assert!(s.layer(1).is_sliding());
    for li in [0usize, 1] {
        let l = *s.layer(li);
        let w = host_attn(&s, &l, 0x00c0_ffee_1234_5678 ^ li as u64);
        let mut rng = Rng::new(0x9e37_79b9_7f4a_7c15 ^ li as u64);
        let xs: Vec<Vec<f32>> = (0..6).map(|_| rng.vec(s.hidden_size, 0.7)).collect();
        let want = cpu_steps(&s, &l, &w, &xs);
        let Some(got) = gpu_steps(&s, &l, &w, &xs) else {
            return;
        };
        let mut worst = 0f32;
        for (i, (g, c)) in got.outs.iter().zip(&want).enumerate() {
            let r = worst_rel(g, c);
            eprintln!(
                "[hd128] layer {li} (sliding={}) step {i}: worst_rel {r:.3e}",
                l.is_sliding()
            );
            worst = worst.max(r);
        }
        assert!(
            worst < 3e-2,
            "hd128 layer {li}: worst rel {worst:.3e} over 6 steps exceeds 3e-2"
        );
    }
}

#[test]
fn per_layer_head_counts_and_windows_are_derived() {
    let s = shapes("\"per-head\"");
    assert_eq!(s.layer(0).num_q_heads, 4);
    assert_eq!(s.layer(1).num_q_heads, 8);
    assert_eq!(s.max_q_heads, 8);
    assert_eq!(s.layer(0).q_rows, 64);
    assert_eq!(s.layer(1).q_rows, 128);
    assert_eq!(s.layer(0).gqa_group, 2);
    assert_eq!(s.layer(1).gqa_group, 4);
    assert_eq!(s.layer(0).window_tokens, None);
    assert_eq!(s.layer(1).window_tokens, Some(3));
    assert_eq!(s.layer(0).rotary_dim, 8);
    assert_eq!(s.layer(1).rotary_dim, 16);
    assert!((s.layer(0).rope_out_scale - 1.3465736).abs() < 1e-6);
    assert!((s.layer(1).rope_out_scale - 1.0).abs() < 1e-6);
    assert_eq!(window_start(6, Some(3)), 3);
    assert_eq!(window_start(6, None), 0);
}

#[test]
fn full_attention_layer_matches_cpu_oracle() {
    check_layer("\"per-head\"", 0, 0x0f1e_2d3c_4b5a_6978);
}

#[test]
fn sliding_layer_with_wider_head_count_matches_cpu_oracle() {
    check_layer("\"per-head\"", 1, 0x1234_5678_9abc_def0);
}

#[test]
fn per_element_gating_matches_cpu_oracle() {
    check_layer("\"per-element\"", 1, 0x2222_3333_4444_5555);
}

#[test]
fn ungated_attention_matches_cpu_oracle() {
    check_layer("false", 0, 0x7777_8888_9999_aaaa);
}

#[test]
fn long_context_crosses_decode_tiles() {
    check_layer_seq("\"per-head\"", 0, 0x5150_4030_2010_0f0e, 520, 300);
    check_layer_seq("\"per-head\"", 1, 0x6160_5040_3020_1f1e, 520, 300);
}

#[test]
fn nvfp4_projection_seam_matches_cpu_oracle() {
    check_layer_full(
        "\"per-head\"",
        1,
        0x3141_5926_5358_9793,
        MAX_SEQ,
        STEPS,
        true,
        6e-2,
    );
    check_layer_full(
        "\"per-head\"",
        0,
        0x2718_2818_2845_9045,
        MAX_SEQ,
        STEPS,
        true,
        6e-2,
    );
}

#[test]
fn oracle_is_sensitive_to_the_sliding_window() {
    let s = shapes("\"per-head\"");
    let l = *s.layer(1);
    assert!(l.is_sliding());
    let mut unwindowed = l;
    unwindowed.window_tokens = None;
    let w = host_attn(&s, &l, 0x1234_5678_9abc_def0);
    let mut rng = Rng::new(0x1234_5678_9abc_def0 ^ 0xa5a5_5a5a);
    let xs: Vec<Vec<f32>> = (0..STEPS).map(|_| rng.vec(s.hidden_size, 0.7)).collect();
    let windowed = cpu_steps(&s, &l, &w, &xs);
    let full = cpu_steps(&s, &unwindowed, &w, &xs);
    assert_eq!(windowed[0], full[0], "step 0 has nothing to mask");
    assert_ne!(
        windowed[STEPS - 1],
        full[STEPS - 1],
        "sliding_window {} never masked a key over {STEPS} steps",
        s.sliding_window
    );
}

#[test]
fn oracle_is_sensitive_to_the_rope_output_scale() {
    let s = shapes("\"per-head\"");
    let l = *s.layer(0);
    assert!((l.rope_out_scale - 1.0).abs() > 1e-3);
    let mut unscaled = l;
    unscaled.rope_out_scale = 1.0;
    let w = host_attn(&s, &l, 0x0f1e_2d3c_4b5a_6978);
    let mut rng = Rng::new(0x0f1e_2d3c_4b5a_6978 ^ 0xa5a5_5a5a);
    let xs: Vec<Vec<f32>> = (0..STEPS).map(|_| rng.vec(s.hidden_size, 0.7)).collect();
    assert_ne!(
        cpu_steps(&s, &l, &w, &xs)[STEPS - 1],
        cpu_steps(&s, &unscaled, &w, &xs)[STEPS - 1],
        "attention_factor never reached the rotated lanes"
    );
}

#[test]
fn distinct_head_counts_share_one_pipeline_set() {
    let s = shapes("\"per-head\"");
    let l0 = *s.layer(0);
    let l1 = *s.layer(1);
    assert_ne!(l0.num_q_heads, l1.num_q_heads);
    let w0 = host_attn(&s, &l0, 11);
    let w1 = host_attn(&s, &l1, 22);
    let mut rng = Rng::new(0xdead_beef_cafe_f00d);
    let xs: Vec<Vec<f32>> = (0..STEPS).map(|_| rng.vec(s.hidden_size, 0.7)).collect();
    let want0 = cpu_steps(&s, &l0, &w0, &xs);
    let want1 = cpu_steps(&s, &l1, &w1, &xs);

    let ctx = match WgpuContext::shared() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no wgpu adapter: {e}");
            return;
        }
    };
    let hidden = s.hidden_size;
    let src = Sources::new();
    let mut b = Builder::new(ctx);
    let step = StepBuffers::alloc(&mut b);
    let xbuf = b.upload_u32("lgw-test2-x", &pack_pairs(&vec![0u16; hidden]));
    let out0 = b.zeros("lgw-test2-out0", (hidden * 2) as u64);
    let out1 = b.zeros("lgw-test2-out1", (hidden * 2) as u64);
    let (fcos, fsin) = rope_tables_from_inv_freq(&s.rope_inv_freq_full, MAX_SEQ);
    let rope0 = RopeTablesGpu {
        cos: b.upload_f32("lgw-test2-fcos", &fcos),
        sin: b.upload_f32("lgw-test2-fsin", &fsin),
        half: s.rope_inv_freq_full.len(),
    };
    let (scos, ssin) = rope_tables_from_inv_freq(&s.rope_inv_freq_sliding, MAX_SEQ);
    let rope1 = RopeTablesGpu {
        cos: b.upload_f32("lgw-test2-scos", &scos),
        sin: b.upload_f32("lgw-test2-ssin", &ssin),
        half: s.rope_inv_freq_sliding.len(),
    };
    build_attn_layer(&mut b, &src, &s, &l0, &w0, &xbuf, &out0, &step, &rope0).unwrap();
    let split = b.passes.len();
    build_attn_layer(&mut b, &src, &s, &l1, &w1, &xbuf, &out1, &step, &rope1).unwrap();
    b.flush_staging();

    let e0: Vec<&str> = b.passes[..split].iter().map(|p| p.entry.as_str()).collect();
    let e1: Vec<&str> = b.passes[split..].iter().map(|p| p.entry.as_str()).collect();
    assert_eq!(
        e0, e1,
        "layers with {} and {} q heads emitted different kernel sequences",
        l0.num_q_heads, l1.num_q_heads
    );
    let mut distinct: Vec<&str> = b.passes.iter().map(|p| p.entry.as_str()).collect();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(
        distinct.len() <= 5,
        "per-layer head counts specialised {} kernels across 2 layers: {distinct:?}",
        distinct.len()
    );
    assert!(
        distinct.contains(&"lgw_attn_decode") && distinct.contains(&"lgw_norm_rope"),
        "expected the attention kernels in {distinct:?}"
    );

    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    for (pos, x) in xs.iter().enumerate() {
        let bits: Vec<u16> = x.iter().map(|v| bf16_bits(*v)).collect();
        ctx.queue.write_buffer(&xbuf, 0, &words_of(&bits));
        let total = pos + 1;
        step.write(
            ctx,
            0,
            pos as u32,
            total as u32,
            window_start(total, Some(s.sliding_window)) as u32,
        );
        run(ctx, &b.passes);
        let g0: Vec<u32> = dispatch::read_back(ctx, &out0, hidden / 2).unwrap();
        let g1: Vec<u32> = dispatch::read_back(ctx, &out1, hidden / 2).unwrap();
        let r0 = worst_rel(&unpack(&g0, hidden), &want0[pos]);
        let r1 = worst_rel(&unpack(&g1, hidden), &want1[pos]);
        assert!(r0 < 3e-2, "step {pos} layer 0 worst rel {r0:.3e}");
        assert!(r1 < 3e-2, "step {pos} layer 1 worst rel {r1:.3e}");
    }
    if let Some(e) = pollster::block_on(scope.pop()) {
        panic!("two-layer attn graph validation: {e}");
    }
}
