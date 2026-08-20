#![cfg(feature = "wgpu")]

mod common;
use common::env_lock;
use common::have_gpu;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, MutexGuard, OnceLock};

use nv_models::gemma4::{Gemma4Config, LayerType as G4LayerType};
use nv_models::gemma4_wgpu::{
    quantize_nvfp4_host as g4_quantize_nvfp4_host, set_attn_variant, AttnVariant, Gemma4Wgpu,
    HostBf16Lin as G4HostBf16Lin, HostLayer as G4HostLayer, HostProj as G4HostProj,
    HostWeights as G4HostWeights, ATTN_VARIANT_DEFAULT,
};
use nv_models::gpt_oss_wgpu as gow;
use nv_models::gpt_oss_wgpu::{GptOssConfig, GptOssLayerType};
use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseConfig;
use nv_models::qwen3_5_moe::LayerType as QLayerType;
use nv_models::qwen3_5_moe_wgpu::HostBf16Lin as QHostBf16Lin;
use nv_quant::mxfp4::Mxfp4Tensor;
use common::LcgOddSeedShift33SignedUnitRows as Lcg;
use common::tiny_config_gpt_oss as gow_tiny_config;

const CENSUS_WHY: &str = "Runtime census of the shader ENTRIES the three single-owner graphs \
     build. The nv-kernels mutation sweep covered nv-kernels' own WGSL; these three graphs carry \
     their own embedded WGSL (g4w_*, q3d_*/q3w_*, gow_* compiled from string constants inside the \
     model files) and none of it was in that sweep. A sweep needs a worklist first, and the two \
     cheap ways to get one are both wrong here: grepping the source finds entries no graph ever \
     names (dead template expansions, format arms this checkpoint cannot reach), and the pipeline \
     LABEL is not the entry -- dispatch::run(ctx, \"g4w-gemv8-pk\", src, INT8_PK_ENTRY, ..) logs \
     one name and builds another. So this drives the real constructors under \
     NV_WGPU_PIPELINE_LOG and reads the ENTRY field back. It PANICS if a graph contributes \
     nothing, because a census that quietly covers two of three graphs is the failure it exists \
     to prevent.";

const LEDGER: &str = "COVERAGE LEDGER -- which shipped entries of the three single-owner graphs \
     still have no effective gate. A sweep's deliverable is the LIST, not the count: the original \
     \"23 shipped entries with no effective gate\" survived only as a number, the next lane could \
     not recover the names, and the list had to be reconstructed. It lives here, in the file whose \
     whole job is the worklist, and it lives in a CONST rather than a doc comment because \
     scripts/strip-comments.py -- the canonical formatter -- deletes doc comments, so the previous \
     ledger was one format run away from vanishing. Shorten it; do not replace it with a total. \
     Every line below was checked entry by entry against the tree, by grepping for the entry name \
     across every tests/ directory in the workspace and reading what the hit actually asserts.\n\
     \n\
     Qwen3.5-dense (qwen3_5_dense_wgpu.rs) -- NONE OPEN, reached by closing six and not by \
     recounting them. The six were the decode-side originals of the M-row prefill twins, plus the \
     argmax pair, which never had a twin at all:\n\
     \x20  q3w_delta_conv  q3w_delta_gating  q3w_delta_out  -- graph_q3d_delta_decode_oracle\n\
     \x20  q3w_gather_embed  q3w_argmax_stage1  q3w_argmax_stage2  -- graph_q3d_decode_head_oracle\n\
     COUNT A GATE ON A TWIN AS A GATE ON THE TWIN ONLY, and check the independence at BODY \
     granularity rather than by grep: an entry name is a prefix of its _m twin's, so a search for \
     the decode entry finds the twin's oracle and reads as covered, while a shared LINE proves \
     nothing in the other direction either -- `let silu = acc / (1.0 + exp(-acc));` is \
     character-identical in q3w_delta_conv and q3w_delta_conv_m. What holds is that the whole \
     function body of each decode entry is absent from shipped_prefill_source(), and both new \
     suites assert exactly that alongside their mutants.\n\
     \n\
     GPT-OSS (gpt_oss_wgpu.rs) -- 2 OPEN, was 3.\n\
     \x20  gow_attn_decode  gow_gemv_mx\n\
     gow_router_topk is CLOSED by graph_gow_router_topk_oracle: f64 top-k and softmax, four \
     mutants, and no checkpoint required. The two that remain are still driven only by \
     tiny_wgpu_decode_matches_cpu_reference. That fixture's reference, gow::reference_step, IS an \
     independent host implementation rather than the shader, so they are gated -- weakly, at \
     rel < 0.05 on the logits of a 2-layer synthetic model. NOTHING IN THIS PARAGRAPH NEEDS A \
     GPT-OSS CHECKPOINT: the fixture builds a tiny model from synthetic weights. The suites that \
     do need one -- real_snapshot_config_is_supported_by_the_wgpu_module, \
     real_snapshot_expert_tensors_have_the_expected_mxfp4_layout, gptoss_wgpu_real_weights_decode \
     -- report a precondition and return on a box without openai/gpt-oss-20b cached, and must \
     never be counted toward this ledger on such a box.\n\
     \n\
     Gemma-4-31B dense (gemma4_wgpu.rs) -- NONE OPEN. Every entry these arms build is answered by \
     a suite that carries its own f64 host oracle and its own mutants:\n\
     \x20  g4w_gemv_fp8_pk  g4w_gemv_fp8_pk3  g4w_gemm_fp8_mk_pk  g4w_gemm_fp8_mk_pk3 \
     g4w_gemm_int8_mk_pk  g4w_gemm_int8_mk_pk3  g4w_gemv_legacy_pk  g4w_gemv_legacy_pk3 -- \
     wgpu_fp8_epilogue\n\
     \x20  g4w_gemv_int8_pk  g4w_gemv_int8_pk3 -- graph_g4w_int8_epilogue_oracle, which also \
     proves its own attribution: dropping the sign from int8_decode moves the int8 arm and leaves \
     the fp8 one bit-identical\n\
     \x20  g4w_gemm_bf16_mk_pk  g4w_gemm_bf16_mk_pk3 -- graph_g4w_mk_bf16_oracle\n\
     \x20  g4w_norm_res_norm  g4w_norm_add_norm -- graph_g4w_norm_chain_oracle\n\
     \x20  g4w_quant_row_pk  g4w_qz_block -- graph_g4w_quant_row_oracle\n\
     \x20  gather2_bf16  gather2_bf16_mk -- graph_g4w_gather2_oracle\n\
     \x20  g4w_head_prep -- graph_g4w_head_prep_oracle. wgpu_nozi_graph_census names it too, but \
     that suite audits zero-init policy and asserts nothing numeric; naming is not gating.\n\
     \x20  g4w_gemv_nvfp4_pk -- wgpu_correct_gemma4_pk_variants, tree-vs-sg bit identity over a \
     whole decode trace. NV_WGPU_NVFP4_TREE is still read by nvfp4_variant(), so the two arms are \
     genuinely different text and the comparison is not vacuous.\n\
     WHAT A TEST CAN COMPILE IS WHAT AN ACCESSOR RETURNS. The bf16 mk pair was NOT-REACHED rather \
     than ungated -- strictly worse, because no mutation of their text could turn anything red -- \
     until mk_bf16_shader_source exposed what mk_bf16_source() builds in Rust; build_mk_pipelines \
     composes through that same function, so the gate cannot drift onto a copy. Moving static \
     WGSL into nv-kernels/wgsl does NOT by itself make an entry reachable: g4w_glue.wgsl needed \
     glue_shader_source() for exactly the same reason.";

struct EnvPins(Vec<(&'static str, Option<String>)>);

impl EnvPins {
    fn pin(vars: &[(&'static str, Option<&str>)]) -> Self {
        let mut saved = Vec::new();
        for (k, v) in vars {
            saved.push((*k, std::env::var(k).ok()));
            match v {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
        EnvPins(saved)
    }
}

impl Drop for EnvPins {
    fn drop(&mut self) {
        for (k, v) in self.0.drain(..) {
            match v {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
    }
}

fn capture(
    tag: &str,
    pins: &[(&'static str, Option<&str>)],
    f: impl FnOnce(),
) -> Vec<(String, String)> {
    let log = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "graph-entry-census-{}-{tag}.txt",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&log);
    let mut all: Vec<(&'static str, Option<&str>)> =
        vec![("NV_WGPU_PIPELINE_LOG", Some(log.to_str().unwrap()))];
    all.extend_from_slice(pins);
    let pinned = EnvPins::pin(&all);
    f();
    drop(pinned);
    let text = std::fs::read_to_string(&log).unwrap_or_default();
    let _ = std::fs::remove_file(&log);
    let mut out = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("[pipeline] ") else {
            continue;
        };
        let Some((label, entry)) = rest.rsplit_once(':') else {
            continue;
        };
        out.push((label.to_string(), entry.to_string()));
    }
    assert!(
        !out.is_empty(),
        "arm {tag} logged no pipeline requests at all -- NV_WGPU_PIPELINE_LOG did not reach \
         dispatch::cached_compute_pipeline, so this census is measuring nothing"
    );
    out
}

pub const G4_TINY_CONFIG: &str = r#"{
  "text_config": {
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "head_dim": 64,
    "global_head_dim": 128,
    "vocab_size": 512,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "sliding_window": 8,
    "final_logit_softcapping": 30.0,
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "sliding_attention", "full_attention"],
    "attention_k_eq_v": true,
    "hidden_activation": "gelu_pytorch_tanh",
    "num_kv_shared_layers": 0,
    "rope_parameters": {
      "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
      "sliding_attention": {"rope_theta": 10000.0}
    }
  },
  "tie_word_embeddings": true
}"#;

pub fn g4_tiny_weights(config: &Gemma4Config, nvfp4_mlp: bool, seed: u64) -> G4HostWeights {
    let mut rng = Lcg::new(seed);
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let n_q = config.num_attention_heads;
    let mut layers = Vec::new();
    for i in 0..config.num_hidden_layers {
        let kind = config.layer_kind(i);
        let hd = config.head_dim_for(kind);
        let nkv = config.num_kv_heads_for(kind);
        let q_dim = n_q * hd;
        let kv_dim = nkv * hd;
        let has_v = !matches!(
            (kind, config.attention_k_eq_v),
            (G4LayerType::FullAttention, true)
        );
        let qkv_rows = q_dim + kv_dim * if has_v { 2 } else { 1 };
        let mk_proj = |rng: &mut Lcg, n: usize, k: usize, quant: bool| {
            let w = rng.bf16_vec(n * k, 0.1);
            if quant {
                G4HostProj::Nvfp4(g4_quantize_nvfp4_host(&w, n, k))
            } else {
                G4HostProj::Bf16(G4HostBf16Lin { w, n, k })
            }
        };
        layers.push(G4HostLayer {
            kind,
            input_ln: rng.norm_vec(hidden),
            post_attn_ln: rng.norm_vec(hidden),
            pre_ff_ln: rng.norm_vec(hidden),
            post_ff_ln: rng.norm_vec(hidden),
            q_norm: rng.norm_vec(hd),
            k_norm: rng.norm_vec(hd),
            layer_scalar: 0.9,
            has_v,
            qkv: mk_proj(&mut rng, qkv_rows, hidden, false),
            o: mk_proj(&mut rng, hidden, q_dim, false),
            gate_up: mk_proj(&mut rng, 2 * inter, hidden, nvfp4_mlp),
            down: mk_proj(&mut rng, hidden, inter, nvfp4_mlp),
        });
    }
    G4HostWeights {
        embed: rng.bf16_vec(config.vocab_size * hidden, 0.1),
        final_norm: rng.norm_vec(hidden),
        layers,
    }
}

fn g4_exercise(nvfp4: bool) {
    let config = Gemma4Config::from_hf_json_str(G4_TINY_CONFIG).unwrap();
    let weights = g4_tiny_weights(&config, nvfp4, 0x1dea);
    let m = Gemma4Wgpu::new(config, &weights, 64);
    assert!(
        m.is_ok(),
        "the gemma4-dense census arm must construct; an arm that returns early contributes \
         no entries and silently narrows the sweep: {:?}",
        m.err()
    );
    let mut m = m.unwrap();
    let _ = m.prefill(&[
        7, 9, 11, 13, 3, 5, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39,
    ]);
    for t in [3u32, 5, 7] {
        let _ = m.decode_step(t);
    }
    let _ = m.decode_step_logits(9);
}

const LEGACY_ARM_WHY: &str = "The legacy row-scale fp8 epilogue, driven the only way it can be \
     driven. This arm used to pin NV_WGPU_ATTN_FP8_LEGACY=1, and NOTHING IN THE TREE READS THAT \
     VARIABLE -- grep -rn NV_WGPU_ATTN_FP8 finds it only in this file. The selector is the \
     process-global set_attn_variant, so the arm built exactly what g4w-bf16-default built and \
     g4w_gemv_legacy_pk / g4w_gemv_legacy_pk3 were NOT-REACHED: no suite in the tree ever built a \
     pipeline for them, so no mutation of their WGSL could turn anything red. That is strictly \
     worse than ungated, and it is the second of the two NOT-REACHED entries the mutation sweep \
     reported (the first, q3d_gemv_i8, was resolved in ae4bcd79f). The same dead-knob reading \
     applies to NV_WGPU_ATTN_FP8 and NV_WGPU_ATTN_FP8_FMT: fp8 attention has been ON by default \
     since 2026-08-10 and the format does not change which entries are compiled, so those pins \
     cost nothing -- but they are not what makes their arms distinct either. Reaching an entry is \
     not gating it: the numeric gate on both legacy entries lives in wgpu_fp8_epilogue.rs, and \
     until it asserted on them they were dispatched, reported and never checked.";

fn g4_exercise_legacy_epilogue() {
    set_attn_variant(Some(AttnVariant {
        on: true,
        legacy_epilogue: 1,
        ..ATTN_VARIANT_DEFAULT
    }));
    g4_exercise(false);
    set_attn_variant(None);
}

fn q3d_tiny_config() -> Qwen3_5DenseConfig {
    Qwen3_5DenseConfig {
        hidden_size: 128,
        num_hidden_layers: 4,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 32,
        intermediate_size: 192,
        vocab_size: 64,
        max_position_embeddings: 64,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-6,
        partial_rotary_factor: 0.25,
        bos_token_id: None,
        eos_token_id: 1,
        layer_types: vec![
            QLayerType::LinearAttention,
            QLayerType::LinearAttention,
            QLayerType::LinearAttention,
            QLayerType::FullAttention,
        ],
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_key_head_dim: 16,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        attn_output_gate: true,
        tie_word_embeddings: false,
    }
}

fn q3d_bf16_lin(r: &mut Lcg, n: usize, k: usize, scale: f32) -> QHostBf16Lin {
    QHostBf16Lin {
        w: r.bf16_vec(n * k, scale),
        n,
        k,
    }
}

fn q3d_tiny_weights(cfg: &Qwen3_5DenseConfig, seed: u64, nvfp4: bool) -> q3d::HostDenseWeights {
    let mut r = Lcg::new(seed);
    let lin = |l: QHostBf16Lin| -> q3d::HostDenseLin {
        if nvfp4 {
            q3d::HostDenseLin::Nvfp4(nv_models::qwen3_5_moe_wgpu::quantize_nvfp4_host(
                &l.w, l.n, l.k,
            ))
        } else {
            l.into()
        }
    };
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
            QLayerType::LinearAttention => {
                q3d::HostDenseMixer::Delta(Box::new(nv_models::qwen3_5_moe_wgpu::HostDeltaNet {
                    in_proj_qkv: q3d_bf16_lin(&mut r, conv_dim, hidden, 0.12),
                    in_proj_z: q3d_bf16_lin(&mut r, value_dim, hidden, 0.12),
                    in_proj_ab: q3d_bf16_lin(&mut r, 2 * n_v, hidden, 0.12),
                    conv1d: r.f32_vec(conv_dim * ks, 0.4),
                    a_log: r.f32_vec(n_v, 0.5),
                    dt_bias: r.f32_vec(n_v, 0.5),
                    norm_w: r.norm_vec(d_v),
                    out_proj: q3d_bf16_lin(&mut r, hidden, value_dim, 0.12),
                }))
            }
            QLayerType::FullAttention => {
                let q_out = cfg.num_attention_heads * hd * 2;
                let kv_out = cfg.num_key_value_heads * hd;
                q3d::HostDenseMixer::Attn(Box::new(q3d::HostDenseAttention {
                    q: lin(q3d_bf16_lin(&mut r, q_out, hidden, 0.12)),
                    k: lin(q3d_bf16_lin(&mut r, kv_out, hidden, 0.12)),
                    v: lin(q3d_bf16_lin(&mut r, kv_out, hidden, 0.12)),
                    o: lin(q3d_bf16_lin(
                        &mut r,
                        hidden,
                        cfg.num_attention_heads * hd,
                        0.12,
                    )),
                    q_norm: r.norm_vec(hd),
                    k_norm: r.norm_vec(hd),
                }))
            }
        };
        layers.push(q3d::HostDenseLayer {
            input_ln: r.norm_vec(hidden),
            post_attn_ln: r.norm_vec(hidden),
            mixer,
            delta_fp8: Default::default(),
            mlp: q3d::HostDenseMlp {
                gate: lin(q3d_bf16_lin(&mut r, inter, hidden, 0.15)),
                up: lin(q3d_bf16_lin(&mut r, inter, hidden, 0.15)),
                down: lin(q3d_bf16_lin(&mut r, hidden, inter, 0.15)),
            },
        });
    }
    q3d::HostDenseWeights {
        embed: r.bf16_vec(cfg.vocab_size * hidden, 0.6),
        final_norm: r.norm_vec(hidden),
        lm_head: r.bf16_vec(cfg.vocab_size * hidden, 0.2),
        layers,
    }
}

fn q3d_exercise_fmt(nvfp4: bool) {
    let cfg = q3d_tiny_config();
    let hw = q3d_tiny_weights(&cfg, 0xbeef, nvfp4);
    let m = q3d::Qwen3_5DenseWgpu::new(cfg, &hw, 48);
    assert!(
        m.is_ok(),
        "the qwen3.5-dense census arm must construct; an arm that returns early contributes \
         no entries and silently narrows the sweep: {:?}",
        m.err()
    );
    let mut m = m.unwrap();
    let _ = m.prefill(&[
        3, 11, 5, 40, 2, 19, 7, 23, 31, 9, 13, 17, 21, 29, 33, 37, 41, 43,
    ]);
    for t in [3u32, 5, 7] {
        let _ = m.decode_step(t);
    }
    let _ = m.decode_step_logits(9);
}

fn gow_bf16_lin(r: &mut Lcg, n: usize, k: usize, scale: f32, bias: bool) -> gow::HostBf16Lin {
    gow::HostBf16Lin {
        w: r.bf16_vec(n * k, scale),
        bias: if bias {
            r.bf16_vec(n, scale)
        } else {
            Vec::new()
        },
        n,
        k,
    }
}

fn gow_mx_stack(r: &mut Lcg, e: usize, n: usize, k: usize, scale: f32) -> gow::HostMxStack {
    let mats: Vec<Mxfp4Tensor> = (0..e)
        .map(|_| Mxfp4Tensor::quantize_rows(&r.f32_rows(n, k, scale)))
        .collect();
    let biases: Vec<Vec<u16>> = (0..e).map(|_| r.bf16_vec(n, scale)).collect();
    gow::stack_mx_host(&mats, &biases)
}

fn gow_tiny_weights(cfg: &GptOssConfig, seed: u64) -> gow::HostWeights {
    let mut r = Lcg::new(seed);
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let hd = cfg.head_dim;
    let q_out = cfg.num_attention_heads * hd;
    let kv_out = cfg.num_key_value_heads * hd;
    let mut layers = Vec::new();
    for _ in 0..cfg.num_hidden_layers {
        layers.push(gow::HostLayer {
            input_ln: r.norm_vec(hidden),
            post_attn_ln: r.norm_vec(hidden),
            attn: gow::HostAttn {
                q: gow_bf16_lin(&mut r, q_out, hidden, 0.12, true),
                k: gow_bf16_lin(&mut r, kv_out, hidden, 0.12, true),
                v: gow_bf16_lin(&mut r, kv_out, hidden, 0.12, true),
                o: gow_bf16_lin(&mut r, hidden, q_out, 0.12, true),
                sinks: (0..cfg.num_attention_heads)
                    .map(|_| r.next_f32() * 0.5)
                    .collect(),
            },
            moe: gow::HostMoe {
                router: gow_bf16_lin(&mut r, cfg.num_local_experts, hidden, 0.3, true),
                gate_up: gow_mx_stack(&mut r, cfg.num_local_experts, 2 * inter, hidden, 0.15),
                down: gow_mx_stack(&mut r, cfg.num_local_experts, hidden, inter, 0.15),
            },
        });
    }
    gow::HostWeights {
        embed: r.bf16_vec(cfg.vocab_size * hidden, 0.6),
        final_norm: r.norm_vec(hidden),
        lm_head: r.bf16_vec(cfg.vocab_size * hidden, 0.2),
        layers,
    }
}

fn gow_exercise() {
    let cfg = gow_tiny_config();
    let hw = gow_tiny_weights(&cfg, 0xf00d);
    let m = gow::GptOssWgpu::new(cfg, &hw, 32);
    assert!(
        m.is_ok(),
        "the gpt-oss census arm must construct; an arm that returns early contributes no \
         entries and silently narrows the sweep: {:?}",
        m.err()
    );
    let mut m = m.unwrap();
    let _ = m.prefill(&[3, 11, 5, 7]);
    for t in [3u32, 5, 7] {
        let _ = m.decode_step(t);
    }
    let _ = m.decode_step_logits(9);
}

#[test]
fn runtime_entry_census_over_the_three_single_owner_graphs() {
    let _g = env_lock();
    eprintln!("{CENSUS_WHY}\n\n{LEDGER}\n");
    assert!(
        have_gpu(),
        "the entry census must run on a real adapter; a skipped census reads as a covered one"
    );

    type Arm = (
        &'static str,
        Vec<(&'static str, Option<&'static str>)>,
        fn(),
    );
    let arms: Vec<Arm> = vec![
        (
            "g4w-bf16-default",
            vec![("NV_G4_WGPU_W8_FFN", None), ("NV_WGPU_ATTN_FP8", None)],
            (|| g4_exercise(false)) as fn(),
        ),
        (
            "g4w-bf16-attn-off",
            vec![
                ("NV_WGPU_ATTN_FP8", Some("0")),
                ("NV_G4_WGPU_W8_FFN", Some("off")),
            ],
            (|| g4_exercise(false)) as fn(),
        ),
        (
            "g4w-bf16-e4m3",
            vec![
                ("NV_WGPU_ATTN_FP8", Some("1")),
                ("NV_WGPU_ATTN_FP8_FMT", Some("e4m3")),
            ],
            (|| g4_exercise(false)) as fn(),
        ),
        (
            "g4w-bf16-legacy-epilogue",
            vec![],
            (g4_exercise_legacy_epilogue) as fn(),
        ),
        (
            "g4w-nvfp4-default",
            vec![("NV_G4_WGPU_W8_FFN", None), ("NV_WGPU_ATTN_FP8", None)],
            (|| g4_exercise(true)) as fn(),
        ),
        (
            "g4w-nvfp4-w8off-tree",
            vec![
                ("NV_G4_WGPU_W8_FFN", Some("off")),
                ("NV_WGPU_NVFP4_TREE", Some("1")),
            ],
            (|| g4_exercise(true)) as fn(),
        ),
        (
            "g4w-nvfp4-w8off-v2",
            vec![
                ("NV_G4_WGPU_W8_FFN", Some("off")),
                ("NV_WGPU_NVFP4_V2", Some("1")),
                ("NV_WGPU_NVFP4_TREE", None),
            ],
            (|| g4_exercise(true)) as fn(),
        ),
        (
            "g4w-bf16-nofuse",
            vec![
                ("NV_WGPU_FUSE", Some("0")),
                ("NV_G4_WGPU_W8_FFN", Some("off")),
            ],
            (|| g4_exercise(false)) as fn(),
        ),
        (
            "g4w-bf16-lmhead-int8",
            vec![
                ("NV_WGPU_LMHEAD_INT8", Some("1")),
                ("NV_G4_WGPU_W8_FFN", Some("off")),
            ],
            (|| g4_exercise(false)) as fn(),
        ),
        (
            "q3d-bf16-default",
            vec![("NV_Q3D_WGPU_W8", None)],
            (|| q3d_exercise_fmt(false)) as fn(),
        ),
        (
            "q3d-nvfp4-default",
            vec![("NV_Q3D_WGPU_W8", None), ("NV_Q3D_WGPU_W8_GROUP", None)],
            (|| q3d_exercise_fmt(true)) as fn(),
        ),
        (
            "q3d-nvfp4-w8-all-g32",
            vec![
                ("NV_Q3D_WGPU_W8", Some("all")),
                ("NV_Q3D_WGPU_W8_GROUP", Some("32")),
            ],
            (|| q3d_exercise_fmt(true)) as fn(),
        ),
        (
            "q3d-nvfp4-w8-all-rowscale",
            vec![
                ("NV_Q3D_WGPU_W8", Some("all")),
                ("NV_Q3D_WGPU_W8_GROUP", Some("0")),
            ],
            (|| q3d_exercise_fmt(true)) as fn(),
        ),
        (
            "q3d-na-off",
            vec![("NV_WGPU_NA", Some("0")), ("NV_Q3D_WGPU_W8", None)],
            (|| q3d_exercise_fmt(false)) as fn(),
        ),
        ("gow-default", vec![], (gow_exercise) as fn()),
    ];

    let mut by_graph: BTreeMap<&'static str, BTreeMap<String, BTreeSet<String>>> = BTreeMap::new();
    let mut per_arm: Vec<(String, usize)> = Vec::new();
    for (tag, pins, drive) in arms {
        let logged = capture(tag, &pins, drive);
        let n = logged.len();
        for (label, entry) in logged {
            let graph = if tag.starts_with("g4w") {
                "gemma4_wgpu.rs"
            } else if tag.starts_with("q3d") {
                "qwen3_5_dense_wgpu.rs"
            } else {
                "gpt_oss_wgpu.rs"
            };
            by_graph
                .entry(graph)
                .or_default()
                .entry(entry)
                .or_default()
                .insert(label);
        }
        per_arm.push((tag.to_string(), n));
    }

    eprintln!("=== runtime entry census (entry <- labels that requested it) ===");
    let mut total = 0usize;
    for (graph, entries) in &by_graph {
        eprintln!("--- {graph}: {} distinct entries", entries.len());
        total += entries.len();
        for (entry, labels) in entries {
            let labels: Vec<&str> = labels.iter().map(|s| s.as_str()).collect();
            eprintln!("  {entry}   <- {labels:?}");
        }
    }
    eprintln!("--- per-arm pipeline requests: {per_arm:?}");
    eprintln!(
        "--- {total} distinct entries across {} graphs",
        by_graph.len()
    );

    assert_eq!(
        by_graph.len(),
        3,
        "all three single-owner graphs must contribute entries; a graph that built nothing \
         means its arm silently failed to construct and the census under-reports: {:?}",
        by_graph.keys().collect::<Vec<_>>()
    );
    for (graph, entries) in &by_graph {
        assert!(
            entries.len() >= 5,
            "{graph} contributed only {} entries -- that is fewer than a decode graph can \
             possibly need, so the arm did not run a forward pass",
            entries.len()
        );
    }
}

#[test]
fn the_legacy_fp8_epilogue_arm_actually_selects_the_legacy_entries() {
    let _g = env_lock();
    eprintln!("{LEGACY_ARM_WHY}");
    assert!(
        have_gpu(),
        "must run on a real adapter; a skipped reachability proof reads as a passed one"
    );
    const LEGACY: [&str; 2] = ["g4w_gemv_legacy_pk", "g4w_gemv_legacy_pk3"];

    let default_arm = capture("legacy-control", &[], || g4_exercise(false));
    let default_entries: BTreeSet<&str> = default_arm.iter().map(|(_, e)| e.as_str()).collect();
    for e in LEGACY {
        assert!(
            !default_entries.contains(e),
            "{e} is built by the DEFAULT arm, so the legacy arm below proves nothing about \
             reachability -- either the default flipped to the legacy epilogue or this control is \
             no longer a control"
        );
    }

    let legacy_arm = capture("legacy-arm", &[], g4_exercise_legacy_epilogue);
    let legacy_entries: BTreeSet<&str> = legacy_arm.iter().map(|(_, e)| e.as_str()).collect();
    for e in LEGACY {
        assert!(
            legacy_entries.contains(e),
            "{e} is still NOT-REACHED: no pipeline was built for it even under \
             set_attn_variant(legacy_epilogue = 1). An entry no suite builds is worse than an \
             ungated one, because there is nothing a mutation of its text could turn red. Built \
             instead: {:?}",
            legacy_entries
        );
    }
    eprintln!(
        "[census] legacy epilogue arm reaches {LEGACY:?}; the default arm builds {} entries and \
         neither of them",
        default_entries.len()
    );

    assert!(
        std::env::var("NV_WGPU_ATTN_FP8_LEGACY").is_err(),
        "NV_WGPU_ATTN_FP8_LEGACY is set in this environment; it is read by nothing in the tree and \
         its presence here would only disguise that"
    );
}

#[test]
fn pipeline_labels_and_entries_are_different_namespaces() {
    let _g = env_lock();
    assert!(
        have_gpu(),
        "must run on a real adapter; a skipped proof reads as a passed one"
    );
    let logged = capture(
        "label-vs-entry",
        &[("NV_G4_WGPU_W8_FFN", None), ("NV_WGPU_ATTN_FP8", None)],
        || g4_exercise(true),
    );
    let mut mismatched: Vec<String> = Vec::new();
    for (label, entry) in &logged {
        if label != entry {
            let s = format!("{label} -> {entry}");
            if !mismatched.contains(&s) {
                mismatched.push(s);
            }
        }
    }
    eprintln!(
        "label != entry on {} of {} distinct requests:\n{}",
        mismatched.len(),
        logged.len(),
        mismatched.join("\n")
    );
    assert!(
        !mismatched.is_empty(),
        "every label equalled its entry -- if that is now true the label-derived worklist trap \
         is gone and this test should be retired, but it is far more likely the log format \
         changed and this parsed nothing"
    );
}
