#![cfg(feature = "wgpu")]

mod common;
use common::greedy_after_prefill;
use common::bf16_lin;
use common::LcgOddSeedShift33SignedUnit as Lcg;
use common::norm_vec;
use common::nvfp4_dense_lin as nvfp4;
use common::tiny_config_qwen35_dense as tiny_config;
use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseConfig;
use nv_models::qwen3_5_moe::LayerType;

const CONTINUATION_TOKENS: usize = 12;

const PROMPT_LENGTHS_SPANNING_CHUNK_BOUNDARY: [usize; 3] = [33, 23, 5];

fn tiny_nvfp4_weights(cfg: &Qwen3_5DenseConfig, seed: u64) -> q3d::HostDenseWeights {
    let mut r = Lcg::new(seed);
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
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
            LayerType::LinearAttention => {
                q3d::HostDenseMixer::Delta(Box::new(nv_models::qwen3_5_moe_wgpu::HostDeltaNet {
                    in_proj_qkv: bf16_lin(&mut r, conv_dim, hidden, 0.12),
                    in_proj_z: bf16_lin(&mut r, value_dim, hidden, 0.12),
                    in_proj_ab: bf16_lin(&mut r, 2 * n_v, hidden, 0.12),
                    conv1d: r.f32_vec(conv_dim * ks, 0.4),
                    a_log: r.f32_vec(n_v, 0.5),
                    dt_bias: r.f32_vec(n_v, 0.5),
                    norm_w: norm_vec(&mut r, d_v),
                    out_proj: bf16_lin(&mut r, hidden, value_dim, 0.12),
                }))
            }
            LayerType::FullAttention => {
                let q_out = cfg.num_attention_heads * hd * 2;
                let kv_out = cfg.num_key_value_heads * hd;
                q3d::HostDenseMixer::Attn(Box::new(q3d::HostDenseAttention {
                    q: nvfp4(bf16_lin(&mut r, q_out, hidden, 0.12)),
                    k: nvfp4(bf16_lin(&mut r, kv_out, hidden, 0.12)),
                    v: nvfp4(bf16_lin(&mut r, kv_out, hidden, 0.12)),
                    o: nvfp4(bf16_lin(&mut r, hidden, cfg.num_attention_heads * hd, 0.12)),
                    q_norm: norm_vec(&mut r, hd),
                    k_norm: norm_vec(&mut r, hd),
                }))
            }
        };
        layers.push(q3d::HostDenseLayer {
            input_ln: norm_vec(&mut r, hidden),
            post_attn_ln: norm_vec(&mut r, hidden),
            mixer,
            delta_fp8: Default::default(),
            mlp: q3d::HostDenseMlp {
                gate: nvfp4(bf16_lin(&mut r, inter, hidden, 0.15)),
                up: nvfp4(bf16_lin(&mut r, inter, hidden, 0.15)),
                down: nvfp4(bf16_lin(&mut r, hidden, inter, 0.15)),
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

fn gpu_or_refuse() {
    match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => eprintln!("[q3d-pf4] adapter: {}", ctx.info.name),
        Err(e) => {
            if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
                panic!(
                    "no wgpu adapter ({e}). This gate covers the nvfp4 M-row prefill arm and \
                     refuses to report success without running; set NV_MODELS_ALLOW_SKIP=1 to \
                     skip it on purpose."
                );
            }
            eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): no wgpu adapter: {e}");
        }
    }
}

fn have_gpu() -> bool {
    nv_kernels::wgpu_backend::WgpuContext::shared().is_ok()
}

const BIT_IDENTITY_HOLDS_ONLY_ON_THE_LEGACY_ARMS_COOP_TILED_WY_ALL_REASSOCIATE: [&str; 4] = [
    "NV_Q3D_KV_FP8",
    "NV_Q3D_PF_COOP",
    "NV_Q3D_PF_ATTN_TILED",
    "NV_Q3D_PF_SCAN_WY",
];

#[test]
fn nvfp4_chunked_prefill_reproduces_the_per_token_replay_token_for_token() {
    gpu_or_refuse();
    if !have_gpu() {
        return;
    }
    for e in BIT_IDENTITY_HOLDS_ONLY_ON_THE_LEGACY_ARMS_COOP_TILED_WY_ALL_REASSOCIATE {
        std::env::set_var(e, "0");
    }
    let cfg = tiny_config();
    let hw = tiny_nvfp4_weights(&cfg, 0xd15e_9b00_0028);
    let mut gpu = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 96).expect("build wgpu model");
    for e in BIT_IDENTITY_HOLDS_ONLY_ON_THE_LEGACY_ARMS_COOP_TILED_WY_ALL_REASSOCIATE {
        std::env::remove_var(e);
    }

    let m = gpu.prefill_chunk_len();
    assert!(
        m >= 2,
        "an all-nvfp4 Qwen3.5 dense graph reported prefill_chunk_len()={m}: the M-row arm for \
         nvfp4 projections is absent, so the engine will replay this prompt one token per \
         command buffer. This is the defect task #28 exists to remove, not a reason to skip."
    );
    eprintln!(
        "[q3d-pf4] m={m}, prefill passes/chunk={}, decode passes/token={}",
        gpu.prefill_pass_count(),
        gpu.pass_count()
    );

    for len in PROMPT_LENGTHS_SPANNING_CHUNK_BOUNDARY {
        let tokens: Vec<u32> = (0..len as u32).map(|i| (i * 7 + 3) % 64).collect();
        let (ids_chunked, logits_chunked) = greedy_after_prefill(&mut gpu, &tokens, true, CONTINUATION_TOKENS);
        let (ids_replay, logits_replay) = greedy_after_prefill(&mut gpu, &tokens, false, CONTINUATION_TOKENS);
        let bit_diff = logits_chunked
            .iter()
            .zip(logits_replay.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let worst = logits_chunked
            .iter()
            .zip(logits_replay.iter())
            .fold(0f32, |a, (c, r)| a.max((c - r).abs()));
        eprintln!(
            "[q3d-pf4] prompt {len:>3} tokens ({} prefilled): {bit_diff} of {} logit lanes differ, \
             max_abs {worst:.6}; chunked {ids_chunked:?} replay {ids_replay:?}",
            tokens.len() - 1,
            logits_chunked.len()
        );
        assert_eq!(
            ids_chunked, ids_replay,
            "chunked prefill and per-token replay produced different greedy continuations at a \
             {len}-token prompt"
        );
        assert_eq!(
            bit_diff, 0,
            "chunked prefill and per-token replay must leave a BIT-IDENTICAL KV cache on this \
             config, because gemv_nvfp4_v2::select_slots picks the same kernel at 1 slot and at \
             {m} slots for every projection in tiny_config (k_blocks=8 is below the deep \
             threshold, so both land on Warp). {bit_diff} of {} lanes differ at a {len}-token \
             prompt, max_abs {worst}. Do not relax this to argmax: a slot-stride defect that \
             makes every chunk row read token 0 leaves the argmax and the CPU-reference check \
             of this tiny model untouched and moves only these bits.",
            logits_chunked.len()
        );

        let mut st = q3d::RefState::new(&cfg);
        let mut want = Vec::new();
        for t in &tokens {
            want = q3d::reference_step(&cfg, &hw, &mut st, *t).expect("cpu reference step");
        }
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-6);
        let rel = logits_chunked
            .iter()
            .zip(want.iter())
            .fold(0f32, |a, (g, w)| a.max((g - w).abs() / scale));
        eprintln!("[q3d-pf4] prompt {len:>3} tokens vs CPU reference: rel {rel:.4}");
        assert!(
            rel < 0.05,
            "chunked nvfp4 prefill diverged from the CPU reference at {len} tokens (rel {rel})"
        );
    }
}
