#![cfg(feature = "wgpu")]

mod common;
use common::argmax_partial_cmp as argmax;
use common::bf16_lin;
use common::env_lock;
use common::have_gpu;
use common::LcgOddSeedShift33SignedUnit as Lcg;

fn pin_fusion_off_process_wide_because_these_gates_pin_unfused_quant_staging_dispatch_counts() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        for e in ["NV_Q3D_FUSE_DN", "NV_Q3D_FUSE_ATTN", "NV_Q3D_FUSE_DN_GEMV", "NV_Q3D_FUSE_MLP"] {
            std::env::set_var(e, "0");
        }
    });
}
use common::norm_vec;
use common::nvfp4_dense_lin as nvfp4;
use common::tiny_config_qwen35_dense as tiny_config;
use common::worst_rel;
use std::sync::{Mutex, MutexGuard, OnceLock};

use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseConfig;
use nv_models::qwen3_5_moe::LayerType;
use nv_models::qwen3_5_moe_wgpu::{quantize_nvfp4_host, HostBf16Lin};

const TOKENS: [u32; 6] = [3, 11, 5, 40, 2, 19];
const W8_GROUP: usize = 32;

struct W8Env;

impl W8Env {
    fn set(mode: &str, group: usize) -> Self {
        std::env::set_var("NV_Q3D_WGPU_W8", mode);
        std::env::set_var("NV_Q3D_WGPU_W8_GROUP", group.to_string());
        Self
    }
}

impl Drop for W8Env {
    fn drop(&mut self) {
        std::env::remove_var("NV_Q3D_WGPU_W8");
        std::env::remove_var("NV_Q3D_WGPU_W8_GROUP");
    }
}

fn tiny_weights(cfg: &Qwen3_5DenseConfig, seed: u64, quantized: bool) -> q3d::HostDenseWeights {
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
    let lin = |l: HostBf16Lin| -> q3d::HostDenseLin {
        if quantized {
            nvfp4(l)
        } else {
            l.into()
        }
    };

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
                    q: lin(bf16_lin(&mut r, q_out, hidden, 0.12)),
                    k: lin(bf16_lin(&mut r, kv_out, hidden, 0.12)),
                    v: lin(bf16_lin(&mut r, kv_out, hidden, 0.12)),
                    o: lin(bf16_lin(&mut r, hidden, cfg.num_attention_heads * hd, 0.12)),
                    q_norm: norm_vec(&mut r, hd),
                    k_norm: norm_vec(&mut r, hd),
                }))
            }
        };
        layers.push(q3d::HostDenseLayer {
            input_ln: norm_vec(&mut r, hidden),
            post_attn_ln: norm_vec(&mut r, hidden),
            mixer,
            mlp: q3d::HostDenseMlp {
                gate: lin(bf16_lin(&mut r, inter, hidden, 0.15)),
                up: lin(bf16_lin(&mut r, inter, hidden, 0.15)),
                down: lin(bf16_lin(&mut r, hidden, inter, 0.15)),
            },
            delta_fp8: q3d::DeltaFp8::default(),
        });
    }

    q3d::HostDenseWeights {
        embed: r.bf16_vec(cfg.vocab_size * hidden, 0.6),
        final_norm: norm_vec(&mut r, hidden),
        lm_head: r.bf16_vec(cfg.vocab_size * hidden, 0.2),
        layers,
    }
}

fn subgroups_ok() -> bool {
    let Ok(ctx) = nv_kernels::wgpu_backend::WgpuContext::shared() else {
        return false;
    };
    let ok = nv_kernels::wgpu_backend::kernels::gemv_nvfp4_v2::subgroup32_ok(ctx);
    if !ok {
        eprintln!("[skip] adapter has no 32-wide subgroups; the int8 gemv needs them");
    }
    ok
}

struct Run {
    passes: usize,
    logits: Vec<Vec<f32>>,
}

fn drive(cfg: &Qwen3_5DenseConfig, hw: &q3d::HostDenseWeights) -> Run {
    let mut m = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), hw, 32).expect("build wgpu model");
    let passes = m.pass_count();
    let logits = TOKENS
        .iter()
        .map(|t| m.decode_step_logits(*t).expect("decode step").1)
        .collect();
    Run { passes, logits }
}

#[test]
fn int8_tracks_the_cpu_reference_more_closely_than_the_nvfp4_graph_it_came_from() {
    let _g = env_lock();
    if !have_gpu() || !subgroups_ok() {
        return;
    }
    pin_fusion_off_process_wide_because_these_gates_pin_unfused_quant_staging_dispatch_counts();
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0xd15e_9b00_0001, true);

    let base = drive(&cfg, &hw);
    let w8 = {
        let _e = W8Env::set("all", W8_GROUP);
        drive(&cfg, &hw)
    };

    eprintln!(
        "[q3d-w8] nvfp4 {} passes/token, int8 {} passes/token, delta {}",
        base.passes,
        w8.passes,
        base.passes as i64 - w8.passes as i64
    );

    assert_eq!(
        base.passes - w8.passes,
        10,
        "W8A16 should delete every q3w_quant_rows dispatch on this config"
    );

    let mut st = q3d::RefState::new(&cfg);
    let mut w8_worst = 0f32;
    let mut nvfp4_worst = 0f32;
    let mut agree = 0usize;
    for (i, t) in TOKENS.iter().enumerate() {
        let want = q3d::reference_step(&cfg, &hw, &mut st, *t).expect("reference step");
        w8_worst = w8_worst.max(worst_rel(&w8.logits[i], &want));
        nvfp4_worst = nvfp4_worst.max(worst_rel(&base.logits[i], &want));
        if argmax(&w8.logits[i]) == argmax(&want) {
            agree += 1;
        }
    }
    eprintln!(
        "[q3d-w8] over {} steps: int8-g{W8_GROUP} worst rel {w8_worst:.3e}, nvfp4 worst rel \
         {nvfp4_worst:.3e}, int8 argmax agrees {agree}/{}",
        TOKENS.len(),
        TOKENS.len()
    );
    assert!(
        w8.logits.iter().all(|l| l.iter().all(|v| v.is_finite())),
        "int8 graph produced a non-finite logit"
    );
    assert!(
        w8_worst < nvfp4_worst,
        "int8-g{W8_GROUP} ({w8_worst:.3e}) should be closer to the true weights than the nvfp4 \
         source it was re-encoded from ({nvfp4_worst:.3e}) -- finer-than-source is the whole \
         premise of this conversion"
    );
    assert_eq!(
        agree,
        TOKENS.len(),
        "int8 argmax disagrees with the reference"
    );
}

#[test]
fn the_ffn_and_attn_halves_of_the_flag_are_independently_selectable() {
    let _g = env_lock();
    if !have_gpu() || !subgroups_ok() {
        return;
    }
    pin_fusion_off_process_wide_because_these_gates_pin_unfused_quant_staging_dispatch_counts();
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0xd15e_9b00_0002, true);

    let base = drive(&cfg, &hw).passes;
    let ffn = {
        let _e = W8Env::set("ffn", W8_GROUP);
        drive(&cfg, &hw).passes
    };
    let attn = {
        let _e = W8Env::set("attn", W8_GROUP);
        drive(&cfg, &hw).passes
    };
    let all = {
        let _e = W8Env::set("all", W8_GROUP);
        drive(&cfg, &hw).passes
    };
    eprintln!("[q3d-w8] passes/token: off {base}, ffn {ffn}, attn {attn}, all {all}");
    assert_eq!(base - ffn, 8, "4 layers x 2 mlp quant dispatches");
    assert_eq!(base - attn, 2, "one full-attention layer: qkv + o");
    assert_eq!(base - all, (base - ffn) + (base - attn));
}

#[test]
fn a_group_that_does_not_divide_k_is_a_hard_error_not_a_silent_per_row_fallback() {
    let _g = env_lock();
    if !have_gpu() || !subgroups_ok() {
        return;
    }
    pin_fusion_off_process_wide_because_these_gates_pin_unfused_quant_staging_dispatch_counts();
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0xd15e_9b00_0003, true);

    let _e = W8Env::set("ffn", 128);
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32);
    }));
    assert!(
        caught.is_err(),
        "group 128 with k=192 must be a hard error; a silent per-row fallback would ship the \
         configuration the quality battery rejected at 2.899e-1 mean KL"
    );
}

#[test]
fn a_bf16_checkpoint_is_left_alone_even_with_the_flag_on() {
    let _g = env_lock();
    if !have_gpu() || !subgroups_ok() {
        return;
    }
    pin_fusion_off_process_wide_because_these_gates_pin_unfused_quant_staging_dispatch_counts();
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0xd15e_9b00_0004, false);
    let base = drive(&cfg, &hw);
    let w8 = {
        let _e = W8Env::set("all", W8_GROUP);
        drive(&cfg, &hw)
    };
    assert_eq!(
        base.passes, w8.passes,
        "bf16 -> int8 quantizes from the model's native precision and measured REJECT; \
         the flag must not touch a bf16 checkpoint"
    );
    for (a, b) in w8.logits.iter().zip(&base.logits) {
        assert!(
            a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "the flag perturbed a bf16 checkpoint"
        );
    }
}
