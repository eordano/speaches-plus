#![cfg(feature = "wgpu")]

mod common;
use common::bf16_lin_q3w as bf16_lin;
use common::LcgOddSeedShift33SignedUnit as Lcg;
use common::nvfp4;
use nv_models::qwen3_5_moe::{LayerType, Qwen3MoeConfig};
use nv_models::qwen3_5_moe_wgpu as q3w;

fn route_config() -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        hidden_size: 1024,
        num_hidden_layers: 2,
        num_attention_heads: 16,
        num_key_value_heads: 2,
        head_dim: 128,
        moe_intermediate_size: 512,
        shared_expert_intermediate_size: 512,
        num_experts: 8,
        num_experts_per_tok: 2,
        vocab_size: 64,
        max_position_embeddings: 64,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        partial_rotary_factor: 0.5,
        bos_token_id: 0,
        eos_token_id: 1,
        layer_types: vec![LayerType::LinearAttention, LayerType::FullAttention],
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        attn_output_gate: true,
        tie_word_embeddings: false,
    }
}

fn norm_vec(r: &mut Lcg, n: usize) -> Vec<u16> {
    (0..n)
        .map(|_| half::bf16::from_f32(1.0 + r.next_f32() * 0.05).to_bits())
        .collect()
}

fn weights(cfg: &Qwen3MoeConfig, seed: u64) -> q3w::HostWeights {
    let mut r = Lcg::new(seed);
    let hidden = cfg.hidden_size;
    let inter = cfg.moe_intermediate_size;
    let sinter = cfg.shared_expert_intermediate_size;
    let hd = cfg.head_dim;
    let n_k = cfg.linear_num_key_heads;
    let n_v = cfg.linear_num_value_heads;
    let d_k = cfg.linear_key_head_dim;
    let d_v = cfg.linear_value_head_dim;
    let key_dim = n_k * d_k;
    let value_dim = n_v * d_v;
    let conv_dim = 2 * key_dim + value_dim;
    let ks = cfg.linear_conv_kernel_dim;

    let mut layers = Vec::new();
    for li in 0..cfg.num_hidden_layers {
        let mixer = match cfg.layer_types[li] {
            LayerType::LinearAttention => q3w::HostMixer::Delta(Box::new(q3w::HostDeltaNet {
                in_proj_qkv: bf16_lin(&mut r, conv_dim, hidden, 0.12),
                in_proj_z: bf16_lin(&mut r, value_dim, hidden, 0.12),
                in_proj_ab: bf16_lin(&mut r, 2 * n_v, hidden, 0.12),
                conv1d: r.f32_vec(conv_dim * ks, 0.4),
                a_log: r.f32_vec(n_v, 0.5),
                dt_bias: r.f32_vec(n_v, 0.5),
                norm_w: norm_vec(&mut r, d_v),
                out_proj: bf16_lin(&mut r, hidden, value_dim, 0.12),
            })),
            LayerType::FullAttention => {
                let q_out = cfg.num_attention_heads * hd * 2;
                let kv_out = cfg.num_key_value_heads * hd;
                q3w::HostMixer::Attn(Box::new(q3w::HostAttention {
                    q: nvfp4(&mut r, q_out, hidden, 0.12),
                    k: nvfp4(&mut r, kv_out, hidden, 0.12),
                    v: nvfp4(&mut r, kv_out, hidden, 0.12),
                    o: nvfp4(&mut r, hidden, cfg.num_attention_heads * hd, 0.12),
                    q_norm: norm_vec(&mut r, hd),
                    k_norm: norm_vec(&mut r, hd),
                }))
            }
        };
        let gates: Vec<_> = (0..cfg.num_experts)
            .map(|_| nvfp4(&mut r, inter, hidden, 0.15))
            .collect();
        let ups: Vec<_> = (0..cfg.num_experts)
            .map(|_| nvfp4(&mut r, inter, hidden, 0.15))
            .collect();
        let downs: Vec<_> = (0..cfg.num_experts)
            .map(|_| nvfp4(&mut r, hidden, inter, 0.15))
            .collect();
        layers.push(q3w::HostLayer {
            input_ln: norm_vec(&mut r, hidden),
            post_attn_ln: norm_vec(&mut r, hidden),
            mixer,
            moe: q3w::HostMoe {
                router: bf16_lin(&mut r, cfg.num_experts, hidden, 0.3),
                experts_gate: q3w::stack_nvfp4_host(&gates),
                experts_up: q3w::stack_nvfp4_host(&ups),
                experts_down: q3w::stack_nvfp4_host(&downs),
                shared_gate: nvfp4(&mut r, sinter, hidden, 0.15),
                shared_up: nvfp4(&mut r, sinter, hidden, 0.15),
                shared_down: nvfp4(&mut r, hidden, sinter, 0.15),
                shared_expert_gate: bf16_lin(&mut r, 1, hidden, 0.3),
            },
        });
    }

    q3w::HostWeights {
        embed: r.bf16_vec(cfg.vocab_size * hidden, 0.6),
        final_norm: norm_vec(&mut r, hidden),
        lm_head: r.bf16_vec(cfg.vocab_size * hidden, 0.2),
        layers,
    }
}

fn trace(cfg: &Qwen3MoeConfig, hw: &q3w::HostWeights) -> (Vec<u32>, Vec<u32>, (usize, usize)) {
    let mut gpu = q3w::Qwen3MoeWgpu::new(cfg.clone(), hw, 32).expect("build wgpu model");
    let mut toks = Vec::new();
    let mut bits = Vec::new();
    for t in [3u32, 17, 5, 1, 11, 2] {
        let (arg, logits) = gpu
            .decode_step_logits(t % cfg.vocab_size as u32)
            .expect("decode step");
        toks.push(arg);
        bits.extend(logits.iter().map(|v| v.to_bits()));
    }
    (toks, bits, gpu.nvfp4_v2_gemvs())
}

#[test]
fn boot_line_reports_engagement_not_just_the_knob() {
    assert_eq!(q3w::nvfp4_v2_boot_line(false, 0, 96), None);
    assert_eq!(q3w::nvfp4_v2_boot_line(false, 96, 96), None);
    assert!(q3w::nvfp4_v2_boot_line(true, 0, 96)
        .unwrap()
        .contains("requested but 0 of 96"));
    assert!(q3w::nvfp4_v2_boot_line(true, 96, 96)
        .unwrap()
        .contains("engaged on 96 of 96"));
}

#[test]
fn qwen35_moe_nvfp4_v2_routes_match_the_shipping_kernel_bit_for_bit() {
    let ctx = match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[skip] no wgpu adapter: {e}");
            return;
        }
    };
    if !nv_kernels::wgpu_backend::kernels::gemv_nvfp4_v2::subgroup32_ok(ctx) {
        eprintln!("[skip] adapter subgroup width is not 32");
        return;
    }

    assert!(
        q3w::nvfp4_v2_enabled(ctx),
        "the nvfp4 v2 route is not reachable on this adapter, so the shape table below pins \
         kernels the graph never emits"
    );

    std::env::set_var("NV_Q3_WGPU_NVFP4_V2", "0");
    let off = !q3w::nvfp4_v2_enabled(ctx);
    std::env::remove_var("NV_Q3_WGPU_NVFP4_V2");
    assert!(off, "NV_Q3_WGPU_NVFP4_V2=0 does not revert the default");

    use nv_kernels::wgpu_backend::kernels::gemv_nvfp4_v2 as v2;
    let want = [
        (4096usize, 1024usize, v2::MROW_PK_ENTRY),
        (256, 1024, v2::FDEC_PK_ENTRY),
        (1024, 2048, v2::FDEC_PK_ENTRY),
        (512, 1024, v2::FDEC_PK_ENTRY),
        (1024, 512, v2::WARP_PK_ENTRY),
    ];
    for (n, k, e) in want {
        assert_eq!(
            v2::select_pk(n, k).map(|(_, _, x)| x).unwrap_or("none"),
            e,
            "shape {n}x{k}"
        );
    }

    let cfg = route_config();
    let hw = weights(&cfg, 0x9c31);

    std::env::set_var("NV_Q3_WGPU_NVFP4_V2", "0");
    let (old_toks, old_bits, old_v2) = trace(&cfg, &hw);
    std::env::set_var("NV_Q3_WGPU_NVFP4_V2", "1");
    let (new_toks, new_bits, new_v2) = trace(&cfg, &hw);
    std::env::remove_var("NV_Q3_WGPU_NVFP4_V2");

    assert_eq!(
        old_v2.0, 0,
        "{} GEMVs routed to v2 with NV_Q3_WGPU_NVFP4_V2=0",
        old_v2.0
    );
    assert!(
        new_v2.0 > 0,
        "NV_Q3_WGPU_NVFP4_V2=1 routed 0 of {} nvfp4 GEMVs to v2, so this arm measures the \
         shipping path under a v2 label",
        new_v2.1
    );

    let nonzero = old_bits.iter().filter(|w| **w != 0).count();
    assert!(
        nonzero * 4 >= old_bits.len() * 3,
        "degenerate trace, only {nonzero}/{} words nonzero",
        old_bits.len()
    );
    let diff = old_bits
        .iter()
        .zip(new_bits.iter())
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "qwen35-moe-nvfp4-v2 hidden={} inter={} experts={} logit-words={} differing={diff}",
        cfg.hidden_size,
        cfg.moe_intermediate_size,
        cfg.num_experts,
        old_bits.len()
    );
    assert_eq!(
        new_toks, old_toks,
        "v2 route and shipping kernel disagree on token ids"
    );
    assert_eq!(diff, 0, "v2 route differs in {diff} logit words");
}
