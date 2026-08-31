#![cfg(feature = "wgpu")]

mod common;
use common::bf16_lin;
use common::have_gpu;
use common::norm_vec;
use common::rel_err;
use common::LcgOddSeedShift33SignedUnit as Lcg;
use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseConfig;
use nv_models::qwen3_5_moe::LayerType;

fn tiny_qwen2_config() -> Qwen3_5DenseConfig {
    Qwen3_5DenseConfig {
        hidden_size: 128,
        num_hidden_layers: 3,
        num_attention_heads: 12,
        num_key_value_heads: 2,
        head_dim: 128,
        intermediate_size: 96,
        vocab_size: 64,
        max_position_embeddings: 64,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        partial_rotary_factor: 1.0,
        bos_token_id: None,
        eos_token_id: 1,
        layer_types: vec![LayerType::FullAttention; 3],
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        attn_output_gate: false,
        tie_word_embeddings: false,
    }
}

fn tiny_qwen2_weights(cfg: &Qwen3_5DenseConfig, seed: u64) -> q3d::HostDenseWeights {
    let mut r = Lcg::new(seed);
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let hd = cfg.head_dim;
    let q_out = cfg.num_attention_heads * hd;
    let kv_out = cfg.num_key_value_heads * hd;
    let mut layers = Vec::new();
    for _ in 0..cfg.num_hidden_layers {
        layers.push(q3d::HostDenseLayer {
            input_ln: norm_vec(&mut r, hidden),
            post_attn_ln: norm_vec(&mut r, hidden),
            mixer: q3d::HostDenseMixer::Attn(Box::new(q3d::HostDenseAttention {
                q: bf16_lin(&mut r, q_out, hidden, 0.12).into(),
                k: bf16_lin(&mut r, kv_out, hidden, 0.12).into(),
                v: bf16_lin(&mut r, kv_out, hidden, 0.12).into(),
                o: bf16_lin(&mut r, hidden, q_out, 0.12).into(),
                q_norm: vec![],
                k_norm: vec![],
                qkv_bias_bf16: r.bf16_vec(q_out + 2 * kv_out, 0.3),
            })),
            delta_fp8: Default::default(),
            mlp: q3d::HostDenseMlp {
                gate: bf16_lin(&mut r, inter, hidden, 0.15).into(),
                up: bf16_lin(&mut r, inter, hidden, 0.15).into(),
                down: bf16_lin(&mut r, hidden, inter, 0.15).into(),
            },
        });
    }
    q3d::HostDenseWeights {
        embed: r.bf16_vec(cfg.vocab_size * hidden, 0.6),
        final_norm: norm_vec(&mut r, hidden),
        lm_head: r.bf16_vec(cfg.vocab_size * hidden, 0.2),
        layers,
    }
}

fn bf16_bits_of(v: f32) -> u16 {
    ((v.to_bits().wrapping_add(0x8000)) >> 16) as u16
}

fn affine_lin(r: &mut Lcg, n: usize, k: usize) -> q3d::HostDenseLin {
    let gs = 32usize;
    let u16s = r.bf16_vec(n * k / 4, 1.0);
    let words: Vec<u32> = u16s
        .chunks_exact(2)
        .map(|c| (c[0] as u32) | ((c[1] as u32) << 16))
        .collect();
    let scales_bf16: Vec<u16> = r
        .f32_vec(n * k / gs, 0.05)
        .iter()
        .map(|v| bf16_bits_of(v.abs().max(1e-3)))
        .collect();
    let biases_bf16: Vec<u16> = r
        .f32_vec(n * k / gs, 0.2)
        .iter()
        .map(|v| bf16_bits_of(*v))
        .collect();
    q3d::HostDenseLin::AffineW4(q3d::HostAffineW4Lin {
        words,
        scales_bf16,
        biases_bf16,
        n,
        k,
        gs,
    })
}

fn tiny_qwen2_affine_weights(cfg: &Qwen3_5DenseConfig, seed: u64) -> q3d::HostDenseWeights {
    let mut r = Lcg::new(seed);
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let hd = cfg.head_dim;
    let q_out = cfg.num_attention_heads * hd;
    let kv_out = cfg.num_key_value_heads * hd;
    let mut layers = Vec::new();
    for _ in 0..cfg.num_hidden_layers {
        layers.push(q3d::HostDenseLayer {
            input_ln: norm_vec(&mut r, hidden),
            post_attn_ln: norm_vec(&mut r, hidden),
            mixer: q3d::HostDenseMixer::Attn(Box::new(q3d::HostDenseAttention {
                q: affine_lin(&mut r, q_out, hidden),
                k: affine_lin(&mut r, kv_out, hidden),
                v: affine_lin(&mut r, kv_out, hidden),
                o: affine_lin(&mut r, hidden, q_out),
                q_norm: vec![],
                k_norm: vec![],
                qkv_bias_bf16: r.bf16_vec(q_out + 2 * kv_out, 0.3),
            })),
            delta_fp8: Default::default(),
            mlp: q3d::HostDenseMlp {
                gate: affine_lin(&mut r, inter, hidden),
                up: affine_lin(&mut r, inter, hidden),
                down: affine_lin(&mut r, hidden, inter),
            },
        });
    }
    q3d::HostDenseWeights {
        embed: r.bf16_vec(cfg.vocab_size * hidden, 0.6),
        final_norm: norm_vec(&mut r, hidden),
        lm_head: r.bf16_vec(cfg.vocab_size * hidden, 0.2),
        layers,
    }
}

#[test]
fn qwen2_bias_affine_chunked_prefill_matches_cpu_reference() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_qwen2_config();
    let hw = tiny_qwen2_affine_weights(&cfg, 0xd075_0c40_0005);
    let mut gpu = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32).expect("build wgpu model");
    eprintln!(
        "[affine-parity] prefill_chunk_len={} (>16 means the coop prefill lane engaged)",
        gpu.prefill_chunk_len()
    );
    let tokens: [u32; 6] = [3, 11, 5, 40, 2, 19];
    let split = 4usize;
    let done = gpu.prefill_tokens(&tokens[..split]).expect("prefill");
    assert_eq!(done, split, "chunked prefill must cover the whole prefix");
    let mut st = q3d::RefState::new(&cfg);
    let mut want_tail: Vec<Vec<f32>> = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        let want = q3d::reference_step(&cfg, &hw, &mut st, *t).expect("reference step");
        if i >= split {
            want_tail.push(want);
        }
    }
    for (j, t) in tokens[split..].iter().enumerate() {
        let (arg, logits) = gpu.decode_step_logits(*t).expect("decode step");
        let want = &want_tail[j];
        let (_, rel) = rel_err(&logits, want);
        let ref_arg = want
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        eprintln!("affine post-prefill step {j}: gpu_argmax={arg} ref_argmax={ref_arg} rel={rel:.6}");
        assert!(
            rel < 0.05,
            "affine chunked prefill diverged from CPU reference at step {j} (rel {rel})"
        );
        assert_eq!(arg, ref_arg, "affine post-prefill argmax diverged at step {j}");
    }
}

#[test]
fn qwen2_bias_partial_verify_accept_matches_stepped_state() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_qwen2_config();
    let hw = tiny_qwen2_weights(&cfg, 0xd075_0c40_0006);
    let mut spec = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 64).expect("build wgpu model");
    if spec.verify_max_rows() < 4 {
        eprintln!(
            "verify graph too small (rows {}); set NV_Q3D_VERIFY_M for the env'd run",
            spec.verify_max_rows()
        );
        return;
    }
    let prompt: [u32; 5] = [3, 11, 5, 40, 2];
    spec.prefill_tokens(&prompt).expect("prefill");
    let chain: [u32; 4] = [19, 7, 63, 9];
    let accepted = 2usize;
    spec.verify_chain(&chain).expect("verify");
    spec.advance(accepted).expect("partial accept");
    let follow: [u32; 3] = [30, 12, 44];
    let spec_out: Vec<u32> = follow
        .iter()
        .map(|&t| spec.decode_step(t).expect("post-accept step"))
        .collect();

    let mut stepped = q3d::Qwen3_5DenseWgpu::new(cfg, &hw, 64).expect("build wgpu model");
    stepped.prefill_tokens(&prompt).expect("prefill");
    for &t in &chain[..accepted] {
        stepped.prefill_step(t).expect("stepped prefix");
    }
    let want: Vec<u32> = follow
        .iter()
        .map(|&t| stepped.decode_step(t).expect("stepped step"))
        .collect();
    assert_eq!(
        spec_out, want,
        "decode after a partial verify accept diverged from the stepped prefix state; the \
         attention-only fast commit (pos += n, KV rows are position-local) is wrong"
    );
}

#[test]
fn qwen2_bias_decode_matches_cpu_reference() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_qwen2_config();
    let hw = tiny_qwen2_weights(&cfg, 0xd075_0c40_0001);
    let mut gpu = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32).expect("build wgpu model");
    let mut st = q3d::RefState::new(&cfg);
    let tokens: [u32; 6] = [3, 11, 5, 40, 2, 19];
    let mut top1_hits = 0usize;
    for (i, t) in tokens.iter().enumerate() {
        let (arg, logits) = gpu.decode_step_logits(*t).expect("decode step");
        let want = q3d::reference_step(&cfg, &hw, &mut st, *t).expect("reference step");
        let (_, rel) = rel_err(&logits, &want);
        let ref_arg = want
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        if arg == ref_arg {
            top1_hits += 1;
        }
        eprintln!("step {i}: tok={t} gpu_argmax={arg} ref_argmax={ref_arg} rel={rel:.6}");
        assert!(
            rel < 0.05,
            "step {i}: qwen2-bias logits diverged from CPU reference (rel {rel})"
        );
    }
    assert_eq!(
        top1_hits,
        tokens.len(),
        "argmax disagreed with the CPU reference on {} of {} steps",
        tokens.len() - top1_hits,
        tokens.len()
    );
}

#[test]
fn qwen2_bias_zero_bias_matches_biasless_norm_free_math() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_qwen2_config();
    let mut hw = tiny_qwen2_weights(&cfg, 0xd075_0c40_0003);
    for l in hw.layers.iter_mut() {
        let q3d::HostDenseMixer::Attn(a) = &mut l.mixer else {
            unreachable!("tiny_qwen2_config builds attention layers only")
        };
        for b in a.qkv_bias_bf16.iter_mut() {
            *b = 0;
        }
    }
    let mut gpu = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32).expect("build wgpu model");
    let mut st = q3d::RefState::new(&cfg);
    for t in [9u32, 30, 7] {
        let (arg, logits) = gpu.decode_step_logits(t).expect("decode step");
        let want = q3d::reference_step(&cfg, &hw, &mut st, t).expect("reference step");
        let (_, rel) = rel_err(&logits, &want);
        let ref_arg = want
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        assert!(rel < 0.05, "zero-bias decode diverged (rel {rel})");
        assert_eq!(arg, ref_arg, "zero-bias argmax diverged");
    }
}

#[test]
fn qwen2_bias_prefill_then_decode_matches_cpu_reference() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_qwen2_config();
    let hw = tiny_qwen2_weights(&cfg, 0xd075_0c40_0002);
    let mut gpu = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32).expect("build wgpu model");
    let tokens: [u32; 6] = [3, 11, 5, 40, 2, 19];
    let split = 4usize;
    gpu.prefill_tokens(&tokens[..split]).expect("prefill");
    let mut st = q3d::RefState::new(&cfg);
    let mut want_tail: Vec<Vec<f32>> = Vec::new();
    for (i, t) in tokens.iter().enumerate() {
        let want = q3d::reference_step(&cfg, &hw, &mut st, *t).expect("reference step");
        if i >= split {
            want_tail.push(want);
        }
    }
    for (j, t) in tokens[split..].iter().enumerate() {
        let (arg, logits) = gpu.decode_step_logits(*t).expect("decode step");
        let want = &want_tail[j];
        let (_, rel) = rel_err(&logits, want);
        let ref_arg = want
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        eprintln!("post-prefill step {j}: tok={t} gpu_argmax={arg} ref_argmax={ref_arg} rel={rel:.6}");
        assert!(
            rel < 0.05,
            "post-prefill step {j}: logits diverged from CPU reference (rel {rel}); the \
             q3w_attn_bias_rope_m prefill arm disagrees with the fused decode arm"
        );
        assert_eq!(arg, ref_arg, "post-prefill argmax diverged at step {j}");
    }
}

#[test]
fn qwen2_bias_image_row_splice_equals_the_tokens_it_replaces() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_qwen2_config();
    let hw = tiny_qwen2_weights(&cfg, 0xd075_0c40_0007);
    let hidden = cfg.hidden_size;
    let tokens: [u32; 7] = [3, 11, 5, 40, 2, 19, 9];
    let follow: [u32; 3] = [30, 12, 44];

    let mut plain = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32).expect("build wgpu model");
    plain.prefill_tokens(&tokens).expect("plain prefill");
    let want: Vec<u32> = follow
        .iter()
        .map(|&t| plain.decode_step(t).expect("plain step"))
        .collect();

    let mut spliced = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32).expect("build wgpu model");
    let mut rows_bf16: Vec<u16> = Vec::new();
    for &t in &tokens[2..4] {
        rows_bf16.extend_from_slice(&hw.embed[t as usize * hidden..(t as usize + 1) * hidden]);
    }
    let mut masked = tokens;
    masked[2] = 0;
    masked[3] = 0;
    let done = spliced
        .prefill_tokens_with_image_rows(
            &masked,
            &[q3d::ImageRowSplice {
                position: 2,
                rows_bf16,
            }],
        )
        .expect("spliced prefill");
    assert_eq!(done, tokens.len(), "spliced prefill must cover the whole prompt");
    let got: Vec<u32> = follow
        .iter()
        .map(|&t| spliced.decode_step(t).expect("spliced step"))
        .collect();
    assert_eq!(
        want, got,
        "splicing the exact embedding rows of tokens 2..4 must be indistinguishable from \
         prefilling those tokens; a mismatch means the image-row injection is broken on the \
         qwen2 bias chunk graph"
    );
}
