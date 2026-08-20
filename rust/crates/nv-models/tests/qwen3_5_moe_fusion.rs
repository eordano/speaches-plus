#![cfg(feature = "wgpu")]

use nv_models::qwen3_5_moe::{LayerType, Qwen3MoeConfig};
use nv_models::qwen3_5_moe_wgpu as q3w;
mod common;
use common::LcgAdd1Shift33SignedUnit as Lcg;

fn cfg() -> Qwen3MoeConfig {
    Qwen3MoeConfig {
        hidden_size: 256,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
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

fn weights(c: &Qwen3MoeConfig) -> q3w::HostWeights {
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
            LayerType::FullAttention => q3w::HostMixer::Attn(Box::new(q3w::HostAttention {
                q: nv(&mut r, c.num_attention_heads * c.head_dim * 2, h, 0.12),
                k: nv(&mut r, c.num_key_value_heads * c.head_dim, h, 0.12),
                v: nv(&mut r, c.num_key_value_heads * c.head_dim, h, 0.12),
                o: nv(&mut r, h, c.num_attention_heads * c.head_dim, 0.12),
                q_norm: r.norm_vec(c.head_dim),
                k_norm: r.norm_vec(c.head_dim),
            })),
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

const DN: &str = "NV_Q3_WGPU_FUSE_DN_INPROJ";
const DG: &str = "NV_Q3_WGPU_FUSE_DN_GATE";
const QC: &str = "NV_Q3_WGPU_FUSE_AT_QCAST";
const RG: &str = "NV_Q3_WGPU_FUSE_ROUTER_GATE";
const SE: &str = "NV_Q3_WGPU_FUSE_SHARED_EXPERT";
const GU: &str = "NV_Q3_WGPU_FUSE_GATEUP";
const AK: &str = "NV_Q3_WGPU_FUSE_AT_KV";
const AQ: &str = "NV_Q3_WGPU_FUSE_AT_QKNORM";

struct Arm {
    label: &'static str,
    dn: bool,
    dg: bool,
    qc: bool,
    rg: bool,
    se: bool,
    gu: bool,
}

const ARMS: [Arm; 8] = [
    Arm {
        label: "none",
        dn: false,
        dg: false,
        qc: false,
        rg: false,
        se: false,
        gu: false,
    },
    Arm {
        label: "dn_inproj",
        dn: true,
        dg: false,
        qc: false,
        rg: false,
        se: false,
        gu: false,
    },
    Arm {
        label: "dn_gate",
        dn: false,
        dg: true,
        qc: false,
        rg: false,
        se: false,
        gu: false,
    },
    Arm {
        label: "at_qcast",
        dn: false,
        dg: false,
        qc: true,
        rg: false,
        se: false,
        gu: false,
    },
    Arm {
        label: "router_gate",
        dn: false,
        dg: false,
        qc: false,
        rg: true,
        se: false,
        gu: false,
    },
    Arm {
        label: "shared_fold",
        dn: false,
        dg: false,
        qc: false,
        rg: false,
        se: true,
        gu: false,
    },
    Arm {
        label: "gate_up",
        dn: false,
        dg: false,
        qc: false,
        rg: false,
        se: false,
        gu: true,
    },
    Arm {
        label: "all (default)",
        dn: true,
        dg: true,
        qc: true,
        rg: true,
        se: true,
        gu: true,
    },
];

#[test]
fn fused_graphs_are_bit_identical_to_the_unfused_graph() {
    let _env_guard = one_env_mutating_test_at_a_time();
    let c = cfg();
    let hw = weights(&c);
    let tokens: [u32; 8] = [3, 11, 5, 40, 2, 19, 7, 33];

    let linear = c
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::LinearAttention))
        .count();
    let full = c.num_hidden_layers - linear;
    let moe_layers = c.num_hidden_layers;

    let mut reference: Option<(Vec<Vec<f32>>, usize)> = None;
    let mut failures: Vec<String> = Vec::new();
    for a in &ARMS {
        std::env::set_var(DN, if a.dn { "1" } else { "0" });
        std::env::set_var(DG, if a.dg { "1" } else { "0" });
        std::env::set_var(QC, if a.qc { "1" } else { "0" });
        std::env::set_var(RG, if a.rg { "1" } else { "0" });
        std::env::set_var(SE, if a.se { "1" } else { "0" });
        std::env::set_var(GU, if a.gu { "1" } else { "0" });

        std::env::set_var(AK, "0");
        std::env::set_var(AQ, "0");

        let mut m = q3w::Qwen3MoeWgpu::new(c.clone(), &hw, 32).expect("build the decode graph");
        let dn_on = m.delta_in_proj_fused_layers();
        let dg_on = m.delta_gate_fused_layers();
        let qc_on = m.attn_qcast_fused_layers();
        let se_on = m.shared_expert_folded_layers();
        let gu_on = m.moe_gate_up_fused_layers();
        let passes = m.pass_count();

        assert_eq!(
            dn_on,
            (if a.dn { linear } else { 0 }, linear),
            "arm {}: delta in_proj fusion did not reach the builder",
            a.label
        );
        assert_eq!(
            dg_on,
            (if a.dg { linear } else { 0 }, linear),
            "arm {}: delta gating fusion did not reach the builder",
            a.label
        );
        assert_eq!(
            qc_on,
            (if a.qc { full } else { 0 }, full),
            "arm {}: attention qcast fusion did not reach the builder",
            a.label
        );
        assert_eq!(
            se_on,
            (if a.se { moe_layers } else { 0 }, moe_layers),
            "arm {}: shared-expert fold did not reach the builder",
            a.label
        );
        assert_eq!(
            gu_on,
            (if a.gu { moe_layers } else { 0 }, moe_layers),
            "arm {}: the MoE gate+up concat did not reach the builder -- a checkpoint whose \
             gate and up disagree on their global scales declines it silently, and this fixture \
             must not be one",
            a.label
        );

        let mut got = Vec::new();
        for t in tokens {
            let (_, logits) = m.decode_step_logits(t).expect("decode step");
            got.push(logits);
        }

        match &reference {
            None => {
                let nonzero = got[0].iter().filter(|v| **v != 0.0).count();
                assert!(
                    nonzero > 0,
                    "the unfused baseline produced all-zero logits, so this test measures nothing"
                );
                assert!(
                    got.windows(2).any(|w| w[0] != w[1]),
                    "every step produced the same logits, so carried state is not exercised"
                );
                eprintln!(
                    "arm {:<14} baseline  passes={passes}  {nonzero}/{} logit words nonzero",
                    a.label,
                    got[0].len()
                );
                reference = Some((got, passes));
            }
            Some((refv, base_passes)) => {
                let bad = refv
                    .iter()
                    .zip(got.iter())
                    .enumerate()
                    .find_map(|(s, (p, q))| {
                        p.iter()
                            .zip(q.iter())
                            .position(|(x, y)| x.to_bits() != y.to_bits())
                            .map(|i| (s, i, p[i], q[i]))
                    });
                let saved = *base_passes as i64 - passes as i64;
                eprintln!(
                    "arm {:<14} {}  passes={passes} ({saved:+} vs unfused)",
                    a.label,
                    if bad.is_none() {
                        "bit-identical"
                    } else {
                        "DIFFER      "
                    },
                );
                if let Some((s, i, x, y)) = bad {
                    failures.push(format!("{}: step {s} logit[{i}] {x} vs {y}", a.label));
                }
                let want = (if a.dn { 2 * linear } else { 0 })
                    + (if a.dg { linear } else { 0 })
                    + (if a.qc { full } else { 0 })
                    + (if a.rg { moe_layers } else { 0 })
                    + (if a.se { 5 * moe_layers } else { 0 })
                    + (if a.gu { moe_layers } else { 0 });
                if saved != want as i64 {
                    failures.push(format!(
                        "{}: dropped {saved} dispatches, expected {want}",
                        a.label
                    ));
                }
            }
        }
    }
    for v in [DN, DG, QC, RG, SE, GU, AK, AQ] {
        std::env::remove_var(v);
    }
    assert!(failures.is_empty(), "fusion parity failures: {failures:#?}");
}

const QL: &str = "NV_Q3_WGPU_QUANT_LANE";

fn one_env_mutating_test_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    static BOTH_TESTS_TOGGLE_PROCESS_ENV_SO_CONCURRENT_BUILDS_READ_EACH_OTHERS_ARMS:
        std::sync::Mutex<()> = std::sync::Mutex::new(());
    BOTH_TESTS_TOGGLE_PROCESS_ENV_SO_CONCURRENT_BUILDS_READ_EACH_OTHERS_ARMS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

#[test]
fn lane_split_quantize_is_bit_identical_to_the_shipped_quantize() {
    let _env_guard = one_env_mutating_test_at_a_time();
    std::env::set_var("NV_Q3_WGPU_PF_MROW", "0");
    assert!(
        !nv_models::qwen3_5_moe_wgpu::pf_mrow_default_on_since_real_weights_first_token_parity_and_1_84x_at_8k(),
        "this gate derives want_passes from the DECODE graph alone (2 per layer + 2 per \
         full-attention layer); the default-on M-row prefill list emits its own quantize \
         passes into the same counter, so it must be pinned off here"
    );
    let c = cfg();
    let hw = weights(&c);
    let tokens: [u32; 8] = [3, 11, 5, 40, 2, 19, 7, 33];
    let full = c
        .layer_types
        .iter()
        .filter(|t| matches!(t, LayerType::FullAttention))
        .count();
    let want_passes = 2 * c.num_hidden_layers + 2 * full;
    let sg32 = match nv_kernels::wgpu_backend::WgpuContext::shared() {
        Ok(ctx) => nv_kernels::wgpu_backend::kernels::gemv_nvfp4_v2::subgroup32_ok(ctx),
        Err(e) => {
            eprintln!("[skip] no wgpu adapter: {e}");
            return;
        }
    };

    let mut reference: Option<(Vec<Vec<f32>>, usize)> = None;
    let mut failures: Vec<String> = Vec::new();
    for arm in ["0", "w32", "w256"] {
        std::env::set_var(QL, arm);
        let mut m = q3w::Qwen3MoeWgpu::new(c.clone(), &hw, 32).expect("build the decode graph");
        let (on, total) = m.quant_lane_passes();
        let passes = m.pass_count();
        assert_eq!(
            total, want_passes,
            "arm {arm}: the graph emitted {total} nvfp4 quantize passes, expected {want_passes} \
             -- the counter is not watching the passes this test thinks it is"
        );
        let want_on = if arm == "0" || !sg32 { 0 } else { want_passes };
        assert_eq!(
            on, want_on,
            "arm {arm}: {on}/{total} quantize passes are lane-split, expected {want_on} \
             (probed 32-lane subgroup: {sg32}) -- an arm that never reached the builder is \
             bit-identical for the wrong reason"
        );

        let mut got = Vec::new();
        for t in tokens {
            let (_, logits) = m.decode_step_logits(t).expect("decode step");
            got.push(logits);
        }
        match &reference {
            None => {
                assert!(
                    got[0].iter().filter(|v| **v != 0.0).count() > 0,
                    "the shipped-quantize baseline produced all-zero logits, so this test \
                     compares zeros to zeros"
                );
                assert!(
                    got.windows(2).any(|w| w[0] != w[1]),
                    "every step produced the same logits, so carried state is not exercised"
                );
                eprintln!("arm {arm:<5} baseline  passes={passes}  lane-split {on}/{total}");
                reference = Some((got, passes));
            }
            Some((refv, base_passes)) => {
                let bad = refv
                    .iter()
                    .zip(got.iter())
                    .enumerate()
                    .find_map(|(s, (p, q))| {
                        p.iter()
                            .zip(q.iter())
                            .position(|(x, y)| x.to_bits() != y.to_bits())
                            .map(|i| (s, i, p[i], q[i]))
                    });
                eprintln!(
                    "arm {arm:<5} {}  passes={passes}  lane-split {on}/{total}",
                    if bad.is_none() {
                        "bit-identical"
                    } else {
                        "DIFFER      "
                    }
                );
                if let Some((s, i, x, y)) = bad {
                    failures.push(format!("{arm}: step {s} logit[{i}] {x} vs {y}"));
                }
                if passes != *base_passes {
                    failures.push(format!(
                        "{arm}: pass count moved {base_passes} -> {passes}; the lane split is a \
                         kernel swap and must not add or drop a dispatch"
                    ));
                }
            }
        }
    }
    std::env::remove_var(QL);
    std::env::remove_var("NV_Q3_WGPU_PF_MROW");
    assert!(
        failures.is_empty(),
        "lane-split quantize parity failures: {failures:#?}"
    );
}
