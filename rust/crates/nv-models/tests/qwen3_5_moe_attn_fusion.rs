#![cfg(feature = "wgpu")]

use nv_models::qwen3_5_moe::{LayerType, Qwen3MoeConfig};
use nv_models::qwen3_5_moe_wgpu as q3w;
mod common;
use common::LcgAdd1Shift33SignedUnit as Lcg;

const KV: &str = "NV_Q3_WGPU_FUSE_AT_KV";
const QK: &str = "NV_Q3_WGPU_FUSE_AT_QKNORM";
const V2: &str = "NV_Q3_WGPU_NVFP4_V2";

fn cfg() -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        hidden_size: 256,
        num_hidden_layers: 4,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        head_dim: 32,
        moe_intermediate_size: 128,
        shared_expert_intermediate_size: 128,
        num_experts: 8,
        num_experts_per_tok: 2,
        vocab_size: 64,
        max_position_embeddings: 64,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        partial_rotary_factor: 0.5,
        bos_token_id: 0,
        eos_token_id: 1,
        layer_types: vec![
            LayerType::LinearAttention,
            LayerType::FullAttention,
            LayerType::LinearAttention,
            LayerType::FullAttention,
        ],
        linear_num_key_heads: 4,
        linear_num_value_heads: 8,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_conv_kernel_dim: 4,
        attn_output_gate: true,
        tie_word_embeddings: false,
    }
}

fn weights(c: &Qwen3MoeConfig, shared_kv: bool) -> q3w::HostWeights {
    let mut r = Lcg(0x51ee_d100_0011);
    let h = c.hidden_size;
    let inter = c.moe_intermediate_size;
    let sinter = c.shared_expert_intermediate_size;
    let key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
    let value_dim = c.linear_num_value_heads * c.linear_value_head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let bf = |r: &mut Lcg, n: usize, k: usize, s: f32| q3w::HostBf16Lin {
        w: r.bf16_vec(n * k, s),
        n,
        k,
    };
    let nv = |r: &mut Lcg, n: usize, k: usize, s: f32| {
        q3w::quantize_nvfp4_host(&r.bf16_vec(n * k, s), n, k)
    };
    let pair = |r: &mut Lcg, n: usize, k: usize, s: f32| {
        let l = q3w::quantize_nvfp4_host(&r.bf16_vec(2 * n * k, s), 2 * n, k);
        assert_eq!(
            n % 128,
            0,
            "row split is only exact on the 128-row swizzle tile"
        );
        let (wh, sh) = (l.packed.len() / 2, l.scales_swizzled.len() / 2);
        let half = |w: &[u8], sf: &[u8]| q3w::HostNvfp4Lin {
            packed: w.to_vec(),
            scales_swizzled: sf.to_vec(),
            alpha: l.alpha,
            input_global: l.input_global,
            n,
            k,
        };
        (
            half(&l.packed[..wh], &l.scales_swizzled[..sh]),
            half(&l.packed[wh..], &l.scales_swizzled[sh..]),
        )
    };
    let mut layers = Vec::new();
    for li in 0..c.num_hidden_layers {
        let mixer = match c.layer_types[li] {
            LayerType::LinearAttention => q3w::HostMixer::Delta(Box::new(q3w::HostDeltaNet {
                in_proj_qkv: bf(&mut r, conv_dim, h, 0.12),
                in_proj_z: bf(&mut r, value_dim, h, 0.12),
                in_proj_ab: bf(&mut r, 2 * c.linear_num_value_heads, h, 0.12),
                conv1d: r.f32_vec(conv_dim * c.linear_conv_kernel_dim, 0.4),
                a_log: r.f32_vec(c.linear_num_value_heads, 0.5),
                dt_bias: r.f32_vec(c.linear_num_value_heads, 0.5),
                norm_w: r.norm_vec(c.linear_value_head_dim),
                out_proj: bf(&mut r, h, value_dim, 0.12),
            })),
            LayerType::FullAttention => {
                let kv_out = c.num_key_value_heads * c.head_dim;
                let (v, k) = if shared_kv {
                    pair(&mut r, kv_out, h, 0.12)
                } else {
                    (nv(&mut r, kv_out, h, 0.12), nv(&mut r, kv_out, h, 0.13))
                };
                q3w::HostMixer::Attn(Box::new(q3w::HostAttention {
                    q: nv(&mut r, c.num_attention_heads * c.head_dim * 2, h, 0.12),
                    k,
                    v,
                    o: nv(&mut r, h, c.num_attention_heads * c.head_dim, 0.12),
                    q_norm: r.norm_vec(c.head_dim),
                    k_norm: r.norm_vec(c.head_dim),
                }))
            }
        };
        let gu: Vec<_> = (0..c.num_experts)
            .map(|_| pair(&mut r, inter, h, 0.15))
            .collect();
        let g: Vec<_> = gu.iter().map(|p| p.0.clone()).collect();
        let u: Vec<_> = gu.iter().map(|p| p.1.clone()).collect();
        let d: Vec<_> = (0..c.num_experts)
            .map(|_| nv(&mut r, h, inter, 0.15))
            .collect();
        let shared = pair(&mut r, sinter, h, 0.15);
        layers.push(q3w::HostLayer {
            input_ln: r.norm_vec(h),
            post_attn_ln: r.norm_vec(h),
            mixer,
            moe: q3w::HostMoe {
                router: bf(&mut r, c.num_experts, h, 0.3),
                experts_gate: q3w::stack_nvfp4_host(&g),
                experts_up: q3w::stack_nvfp4_host(&u),
                experts_down: q3w::stack_nvfp4_host(&d),
                shared_gate: shared.0,
                shared_up: shared.1,
                shared_down: nv(&mut r, h, sinter, 0.15),
                shared_expert_gate: bf(&mut r, 1, h, 0.3),
            },
        });
    }
    q3w::HostWeights {
        embed: r.bf16_vec(c.vocab_size * h, 0.6),
        final_norm: r.norm_vec(h),
        lm_head: r.bf16_vec(c.vocab_size * h, 0.2),
        layers,
    }
}

fn set(var: &str, on: bool) {
    std::env::set_var(var, if on { "1" } else { "0" });
}

struct Arm {
    label: &'static str,
    kv: bool,
    qk: bool,
    v2: bool,
}

const ARMS: [Arm; 8] = [
    Arm {
        label: "split      ",
        kv: false,
        qk: false,
        v2: false,
    },
    Arm {
        label: "kv         ",
        kv: true,
        qk: false,
        v2: false,
    },
    Arm {
        label: "qknorm     ",
        kv: false,
        qk: true,
        v2: false,
    },
    Arm {
        label: "kv+qknorm  ",
        kv: true,
        qk: true,
        v2: false,
    },
    Arm {
        label: "split   v2 ",
        kv: false,
        qk: false,
        v2: true,
    },
    Arm {
        label: "kv      v2 ",
        kv: true,
        qk: false,
        v2: true,
    },
    Arm {
        label: "qknorm  v2 ",
        kv: false,
        qk: true,
        v2: true,
    },
    Arm {
        label: "kv+qknorm v2",
        kv: true,
        qk: true,
        v2: true,
    },
];

#[test]
fn whole_graph_decode_is_bit_identical_across_attn_fusion_arms() {
    let _env_guard = one_env_mutating_test_at_a_time();
    let c = cfg();
    let hw = weights(&c, true);
    let tokens: [u32; 8] = [3, 11, 5, 40, 2, 19, 7, 33];
    let full = c
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::FullAttention))
        .count();

    let mut reference: Option<(Vec<Vec<f32>>, usize)> = None;
    let mut failures = Vec::new();
    for a in &ARMS {
        set(KV, a.kv);
        set(QK, a.qk);
        set(V2, a.v2);
        let mut m = q3w::Qwen3MoeWgpu::new(c.clone(), &hw, 32).expect("build the decode graph");
        let (kvf, kvt) = m.attn_kv_fused_layers();
        let (qkf, qkt) = m.attn_qknorm_fused_layers();
        assert_eq!(
            (kvt, qkt),
            (full, full),
            "{}: the fusion counters saw {kvt}/{qkt} full-attention layers, not {full}",
            a.label
        );
        assert_eq!(
            (kvf > 0, qkf > 0),
            (a.kv, a.qk),
            "{}: asked for kv={} qknorm={}, graph built {kvf}/{qkf} -- the knob did not reach \
             the builder, so this arm measures nothing",
            a.label,
            a.kv,
            a.qk
        );
        let passes = m.pass_count();
        let mut got = Vec::new();
        for t in tokens {
            let (_, logits) = m.decode_step_logits(t).expect("decode step");
            got.push(logits);
        }
        let nonzero = got[0].iter().filter(|v| **v != 0.0).count();
        match &reference {
            None => {
                assert!(
                    nonzero > 0,
                    "the baseline graph produced all-zero logits, so this test measures nothing"
                );
                eprintln!(
                    "arm {} baseline, {passes} passes, {nonzero}/{} logit words nonzero",
                    a.label,
                    got[0].len()
                );
                reference = Some((got, passes));
            }
            Some((refv, refp)) => {
                let want = refp - usize::from(a.kv) * full - usize::from(a.qk) * full;
                if passes != want {
                    failures.push(format!(
                        "{}: {passes} passes, want {want} ({refp} minus one per fused pair per \
                         full-attention layer)",
                        a.label
                    ));
                }
                let bad = refv
                    .iter()
                    .zip(got.iter())
                    .enumerate()
                    .find_map(|(s, (x, y))| {
                        x.iter()
                            .zip(y.iter())
                            .position(|(p, q)| p.to_bits() != q.to_bits())
                            .map(|i| (s, i, x[i], y[i]))
                    });
                eprintln!(
                    "arm {} {passes} passes, {}",
                    a.label,
                    if bad.is_none() {
                        "bit-identical"
                    } else {
                        "DIFFER"
                    }
                );
                if let Some((s, i, x, y)) = bad {
                    failures.push(format!("{}: step {s} logit[{i}] {x} vs {y}", a.label));
                }
            }
        }
    }
    for v in [KV, QK, V2] {
        std::env::remove_var(v);
    }
    assert!(
        failures.is_empty(),
        "graph-level attention-fusion failures: {failures:#?}"
    );
}

fn one_env_mutating_test_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    static THESE_TESTS_TOGGLE_PROCESS_ENV_SO_CONCURRENT_BUILDS_READ_EACH_OTHERS_ARMS:
        std::sync::Mutex<()> = std::sync::Mutex::new(());
    THESE_TESTS_TOGGLE_PROCESS_ENV_SO_CONCURRENT_BUILDS_READ_EACH_OTHERS_ARMS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

#[test]
fn mismatched_kv_global_scales_decline_the_concat() {
    let _env_guard = one_env_mutating_test_at_a_time();
    let c = cfg();
    set(KV, true);
    set(QK, true);
    std::env::remove_var(V2);
    let shared = q3w::Qwen3MoeWgpu::new(c.clone(), &weights(&c, true), 32).expect("shared build");
    let split = q3w::Qwen3MoeWgpu::new(c.clone(), &weights(&c, false), 32).expect("split build");
    let full = c
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::FullAttention))
        .count();
    eprintln!(
        "shared k/v scales: {:?} fused, independent: {:?} fused",
        shared.attn_kv_fused_layers(),
        split.attn_kv_fused_layers()
    );
    assert_eq!(shared.attn_kv_fused_layers(), (full, full));
    assert_eq!(split.attn_kv_fused_layers(), (0, full));
    assert_eq!(
        split.pass_count(),
        shared.pass_count() + full,
        "declining the concat must put the second GEMV back"
    );
    for v in [KV, QK] {
        std::env::remove_var(v);
    }
}

#[test]
#[ignore = "loads ~20 GB of NVFP4 weights twice; set NV_QWEN36_ATTN_FUSION_TEST=1"]
fn real_checkpoint_decode_is_bit_identical_across_attn_fusion_arms() {
    if std::env::var("NV_QWEN36_ATTN_FUSION_TEST").is_err() {
        panic!("set NV_QWEN36_ATTN_FUSION_TEST=1; a silent skip here would report a pass");
    }
    let _env_guard = one_env_mutating_test_at_a_time();
    let dir = std::env::var("NV_QWEN36_DIR").unwrap_or_else(|_| {
        format!(
            "{}/.cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots/965bfb0e24d08e295cd641a15c7f231554078d0d",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let dir = std::path::PathBuf::from(dir);
    let cfg = Qwen3MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
    let full = cfg
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::FullAttention))
        .count();
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let tokens: [u32; 6] = [785, 6722, 315, 9625, 374, 279];

    let mut reference: [Option<(Vec<Vec<f32>>, usize)>; 2] = [None, None];
    let mut failures = Vec::new();
    for a in &ARMS {
        set(KV, a.kv);
        set(QK, a.qk);
        set(V2, a.v2);
        let t0 = std::time::Instant::now();
        let mut m = q3w::Qwen3MoeWgpu::from_loader(cfg.clone(), &loader, 64)
            .expect("build the decode graph from the loader");
        let (kvf, _) = m.attn_kv_fused_layers();
        let (qkf, _) = m.attn_qknorm_fused_layers();
        assert_eq!(
            (kvf > 0, qkf > 0),
            (a.kv, a.qk),
            "{}: asked for kv={} qknorm={}, graph built {kvf}/{qkf}",
            a.label,
            a.kv,
            a.qk
        );
        let passes = m.pass_count();
        let mut got = Vec::new();
        for t in tokens {
            let (_, logits) = m.decode_step_logits(t).expect("decode step");
            got.push(logits);
        }
        eprintln!(
            "arm {} built in {:.1}s, {passes} passes/token, kv {kvf}/{full} qknorm {qkf}/{full}, \
             nvfp4 v2 routed {:?}",
            a.label,
            t0.elapsed().as_secs_f64(),
            m.nvfp4_v2_gemvs()
        );
        let slot = &mut reference[usize::from(a.v2)];
        match slot {
            None => {
                assert!(
                    got[0].iter().any(|v| *v != 0.0),
                    "the baseline graph produced all-zero logits"
                );
                *slot = Some((got, passes));
            }
            Some((refv, refp)) => {
                let refp = *refp;
                let want = refp - usize::from(a.kv) * full - usize::from(a.qk) * full;
                if passes != want {
                    failures.push(format!("{}: {passes} passes, want {want}", a.label));
                }
                let bad = refv
                    .iter()
                    .zip(got.iter())
                    .enumerate()
                    .find_map(|(s, (x, y))| {
                        x.iter()
                            .zip(y.iter())
                            .position(|(p, q)| p.to_bits() != q.to_bits())
                            .map(|i| (s, i, x[i], y[i]))
                    });
                if let Some((s, i, x, y)) = bad {
                    failures.push(format!("{}: step {s} logit[{i}] {x} vs {y}", a.label));
                }
            }
        }
        drop(m);
    }
    for v in [KV, QK, V2] {
        std::env::remove_var(v);
    }
    assert!(
        failures.is_empty(),
        "real-checkpoint failures: {failures:#?}"
    );
}
