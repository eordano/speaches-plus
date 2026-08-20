#![allow(dead_code)]

mod common;
use common::argmax_partial_cmp as argmax;
use common::worst_rel;
use nv_models::qwen3_5_moe::{LayerType, Qwen3_5DenseConfig};

#[cfg(feature = "cuda")]
mod hub_snapshot;
use common::LcgOddSeedShift33SignedUnitRows as Lcg;

const QWEN38_27B_CONFIG_JSON: &str = include_str!("qwen3_8_27b_config.json");

const TINY_LAYERS_8_KEEPS_TWO_FULL_ATTENTION_SLOTS_OF_THE_INTERVAL_4_PATTERN: usize = 8;
const TINY_HIDDEN_128_SMALLEST_MULTIPLE_OF_2X_NVFP4_BLOCK_THE_WGPU_BUILD_ACCEPTS: usize = 128;
const TINY_HEAD_DIM_64_SMALLEST_CUDA_FP8_DECODE_TEMPLATE_ARM_ROTARY_QUARTER_16_STILL_NONTRIVIAL: usize = 64;
const TINY_Q_HEADS_12_KV_2_KEEP_THE_RELEASE_GQA_RATIO_24_OVER_4: (usize, usize) = (12, 2);
const TINY_GDN_V6_K2_KEEP_THE_RELEASE_V_OVER_K_RATIO_48_OVER_16_WHICH_IS_3_NOT_THE_935_RATIO_2:
    (usize, usize) = (6, 2);
const TINY_GDN_HEAD_DIM_16: usize = 16;
const TINY_INTER_192_VOCAB_64_MAX_POS_64: (usize, usize, usize) = (192, 64, 64);

fn real_q38_config_reasserting_the_structural_facts_the_tiny_geometry_must_preserve(
) -> Qwen3_5DenseConfig {
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(QWEN38_27B_CONFIG_JSON)
        .expect("real unsloth/Qwen3.8-27B-NVFP4 config.json must parse as a Qwen3_5 dense config");
    assert_eq!(cfg.num_hidden_layers, 64);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.num_attention_heads, 24);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.linear_num_value_heads, 48);
    assert_eq!(cfg.linear_num_key_heads, 16);
    assert_eq!(cfg.linear_conv_kernel_dim, 4);
    assert!(cfg.attn_output_gate);
    assert_eq!(cfg.partial_rotary_factor, 0.25);
    assert_eq!(cfg.rope_theta, 10_000_000.0);
    assert_eq!(cfg.rotary_dim(), 64);
    cfg
}

fn tiny_q38_config_shrunk_from_the_real_release_config_keeping_every_structural_fact(
) -> Qwen3_5DenseConfig {
    let mut cfg = real_q38_config_reasserting_the_structural_facts_the_tiny_geometry_must_preserve();
    cfg.layer_types
        .truncate(TINY_LAYERS_8_KEEPS_TWO_FULL_ATTENTION_SLOTS_OF_THE_INTERVAL_4_PATTERN);
    cfg.num_hidden_layers = TINY_LAYERS_8_KEEPS_TWO_FULL_ATTENTION_SLOTS_OF_THE_INTERVAL_4_PATTERN;
    for (i, t) in cfg.layer_types.iter().enumerate() {
        let expected = if (i + 1) % 4 == 0 {
            LayerType::FullAttention
        } else {
            LayerType::LinearAttention
        };
        assert_eq!(
            *t, expected,
            "layer {i}: truncation must keep the full_attention_interval=4 pattern"
        );
    }
    cfg.hidden_size = TINY_HIDDEN_128_SMALLEST_MULTIPLE_OF_2X_NVFP4_BLOCK_THE_WGPU_BUILD_ACCEPTS;
    cfg.head_dim = TINY_HEAD_DIM_64_SMALLEST_CUDA_FP8_DECODE_TEMPLATE_ARM_ROTARY_QUARTER_16_STILL_NONTRIVIAL;
    let (n_q, n_kv) = TINY_Q_HEADS_12_KV_2_KEEP_THE_RELEASE_GQA_RATIO_24_OVER_4;
    cfg.num_attention_heads = n_q;
    cfg.num_key_value_heads = n_kv;
    let (gdn_v, gdn_k) =
        TINY_GDN_V6_K2_KEEP_THE_RELEASE_V_OVER_K_RATIO_48_OVER_16_WHICH_IS_3_NOT_THE_935_RATIO_2;
    cfg.linear_num_value_heads = gdn_v;
    cfg.linear_num_key_heads = gdn_k;
    cfg.linear_key_head_dim = TINY_GDN_HEAD_DIM_16;
    cfg.linear_value_head_dim = TINY_GDN_HEAD_DIM_16;
    let (inter, vocab, max_pos) = TINY_INTER_192_VOCAB_64_MAX_POS_64;
    cfg.intermediate_size = inter;
    cfg.vocab_size = vocab;
    cfg.max_position_embeddings = max_pos;
    cfg.bos_token_id = None;
    cfg.eos_token_id = 1;
    assert_eq!(cfg.rotary_dim(), 16, "64 * 0.25 rotated dims");
    cfg
}

const SMOKE_TOKENS: [u32; 12] = [3, 11, 5, 40, 2, 19, 7, 33, 21, 8, 50, 14];
const SMOKE_PREFILL_SPLIT_6_EXERCISES_MULTI_TOKEN_PREFILL_THEN_STEPWISE_DECODE: usize = 6;

enum TinyMixerValues {
    Delta {
        in_proj_qkv: Vec<f32>,
        in_proj_z: Vec<f32>,
        in_proj_a: Vec<f32>,
        in_proj_b: Vec<f32>,
        conv1d: Vec<f32>,
        a_log: Vec<f32>,
        dt_bias: Vec<f32>,
        norm_effective: Vec<f32>,
        out_proj: Vec<f32>,
    },
    Attn {
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        o: Vec<f32>,
        q_norm_effective: Vec<f32>,
        k_norm_effective: Vec<f32>,
    },
}

struct TinyLayerValues {
    input_ln_effective: Vec<f32>,
    post_attn_ln_effective: Vec<f32>,
    mixer: TinyMixerValues,
    mlp_gate: Vec<f32>,
    mlp_up: Vec<f32>,
    mlp_down: Vec<f32>,
}

struct TinyWeightValues {
    embed: Vec<f32>,
    final_norm_effective: Vec<f32>,
    lm_head: Vec<f32>,
    layers: Vec<TinyLayerValues>,
}

fn tiny_weight_values_shared_by_every_backend_so_parity_compares_identical_models(
    cfg: &Qwen3_5DenseConfig,
    seed: u64,
) -> TinyWeightValues {
    let mut r = Lcg::new(seed);
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let hd = cfg.head_dim;
    let n_v = cfg.linear_num_value_heads;
    let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let value_dim = n_v * cfg.linear_value_head_dim;
    let conv_dim = 2 * key_dim + value_dim;
    let ks = cfg.linear_conv_kernel_dim;
    let mut layers = Vec::new();
    for li in 0..cfg.num_hidden_layers {
        let mixer = match cfg.layer_types[li] {
            LayerType::LinearAttention => TinyMixerValues::Delta {
                in_proj_qkv: r.bf16_rounded_f32_vec(conv_dim * hidden, 0.12),
                in_proj_z: r.bf16_rounded_f32_vec(value_dim * hidden, 0.12),
                in_proj_a: r.bf16_rounded_f32_vec(n_v * hidden, 0.12),
                in_proj_b: r.bf16_rounded_f32_vec(n_v * hidden, 0.12),
                conv1d: r.bf16_rounded_f32_vec(conv_dim * ks, 0.4),
                a_log: r.bf16_rounded_f32_vec(n_v, 0.5),
                dt_bias: r.bf16_rounded_f32_vec(n_v, 0.5),
                norm_effective: r.norm_effective_vec_near_one(cfg.linear_value_head_dim),
                out_proj: r.bf16_rounded_f32_vec(hidden * value_dim, 0.12),
            },
            LayerType::FullAttention => {
                let q_out = cfg.num_attention_heads * hd * 2;
                assert!(
                    cfg.attn_output_gate,
                    "q_out doubling encodes attn_output_gate=true from the release config"
                );
                let kv_out = cfg.num_key_value_heads * hd;
                TinyMixerValues::Attn {
                    q: r.bf16_rounded_f32_vec(q_out * hidden, 0.12),
                    k: r.bf16_rounded_f32_vec(kv_out * hidden, 0.12),
                    v: r.bf16_rounded_f32_vec(kv_out * hidden, 0.12),
                    o: r.bf16_rounded_f32_vec(hidden * cfg.num_attention_heads * hd, 0.12),
                    q_norm_effective: r.norm_effective_vec_near_one(hd),
                    k_norm_effective: r.norm_effective_vec_near_one(hd),
                }
            }
        };
        layers.push(TinyLayerValues {
            input_ln_effective: r.norm_effective_vec_near_one(hidden),
            post_attn_ln_effective: r.norm_effective_vec_near_one(hidden),
            mixer,
            mlp_gate: r.bf16_rounded_f32_vec(inter * hidden, 0.15),
            mlp_up: r.bf16_rounded_f32_vec(inter * hidden, 0.15),
            mlp_down: r.bf16_rounded_f32_vec(hidden * inter, 0.15),
        });
    }
    TinyWeightValues {
        embed: r.bf16_rounded_f32_vec(cfg.vocab_size * hidden, 0.6),
        final_norm_effective: r.norm_effective_vec_near_one(hidden),
        lm_head: r.bf16_rounded_f32_vec(cfg.vocab_size * hidden, 0.2),
        layers,
    }
}

#[cfg(feature = "wgpu")]
mod mrope_degeneracy {
    use super::*;
    use nv_models::qwen3_5_dense_wgpu::rope_tables;

    fn mrope_section_from_the_release_fixture() -> Vec<usize> {
        let v: serde_json::Value = serde_json::from_str(QWEN38_27B_CONFIG_JSON).unwrap();
        v["text_config"]["rope_parameters"]["mrope_section"]
            .as_array()
            .expect("mrope_section in rope_parameters")
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect()
    }

    fn section_of_interleaved_half_index_matching_apply_interleaved_mrope(
        j: usize,
        section: &[usize],
    ) -> usize {
        let h_cap = 3 * section[1];
        let w_cap = 3 * section[2];
        match j % 3 {
            1 if j < h_cap => 1,
            2 if j < w_cap => 2,
            _ => 0,
        }
    }

    #[test]
    fn interleaved_mrope_with_equal_text_positions_is_bitwise_the_partial_rotary_table_matching_transformers_modeling_qwen3_5_qwen3_5textrotaryembedding_apply_interleaved_mrope(
    ) {
        let cfg = real_q38_config_reasserting_the_structural_facts_the_tiny_geometry_must_preserve();
        let rot = cfg.rotary_dim();
        assert_eq!(rot, 64, "head_dim 256 * partial_rotary_factor 0.25");
        let half = rot / 2;
        let section = mrope_section_from_the_release_fixture();
        assert_eq!(section, vec![11, 11, 10]);
        assert_eq!(
            section.iter().sum::<usize>(),
            half,
            "mrope sections must tile exactly the {half} rotary half-frequencies"
        );
        let mut consumed = [0usize; 3];
        for j in 0..half {
            consumed[section_of_interleaved_half_index_matching_apply_interleaved_mrope(
                j, &section,
            )] += 1;
        }
        assert_eq!(
            consumed.to_vec(),
            section,
            "the THWTHW...TT interleave of apply_interleaved_mrope consumes exactly [11,11,10]"
        );

        let rows = 512usize;
        let (cos, sin) = rope_tables(rot, cfg.rope_theta, rows);
        for p in 0..rows {
            let positions_thw_equal_for_text_only_serving = [p, p, p];
            for j in 0..half {
                let s =
                    section_of_interleaved_half_index_matching_apply_interleaved_mrope(j, &section);
                let inv = 1.0f32 / cfg.rope_theta.powf((j as f32 * 2.0) / rot as f32);
                let th = positions_thw_equal_for_text_only_serving[s] as f32 * inv;
                assert_eq!(
                    cos[p * half + j].to_bits(),
                    th.cos().to_bits(),
                    "cos row {p} freq {j}: with t=h=w the section gather must be bitwise the 1D partial-rotary table"
                );
                assert_eq!(
                    sin[p * half + j].to_bits(),
                    th.sin().to_bits(),
                    "sin row {p} freq {j}"
                );
            }
        }

        let p = 7usize;
        let positions_thw_diverge_as_they_would_with_a_real_image_grid = [p, p + 3, p + 5];
        let mut any_bit_differs = false;
        for j in 0..half {
            let s = section_of_interleaved_half_index_matching_apply_interleaved_mrope(j, &section);
            let inv = 1.0f32 / cfg.rope_theta.powf((j as f32 * 2.0) / rot as f32);
            let th = positions_thw_diverge_as_they_would_with_a_real_image_grid[s] as f32 * inv;
            if cos[p * half + j].to_bits() != th.cos().to_bits() {
                any_bit_differs = true;
            }
        }
        assert!(
            any_bit_differs,
            "diverging grid positions must NOT reduce to the 1D table, or this test proves nothing"
        );
    }
}

#[cfg(feature = "wgpu")]
mod tiny_host_weights {
    use super::*;
    use nv_models::qwen3_5_dense_wgpu as q3d;
    use nv_models::qwen3_5_moe_wgpu::{HostBf16Lin, HostDeltaNet};

    fn bf16_bits(v: &[f32]) -> Vec<u16> {
        v.iter().map(|x| half::bf16::from_f32(*x).to_bits()).collect()
    }

    pub fn host_dense_weights(
        cfg: &Qwen3_5DenseConfig,
        vals: &TinyWeightValues,
    ) -> q3d::HostDenseWeights {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let conv_dim = 2 * key_dim + value_dim;
        let layers = vals
            .layers
            .iter()
            .map(|l| {
                let mixer = match &l.mixer {
                    TinyMixerValues::Delta {
                        in_proj_qkv,
                        in_proj_z,
                        in_proj_a,
                        in_proj_b,
                        conv1d,
                        a_log,
                        dt_bias,
                        norm_effective,
                        out_proj,
                    } => {
                        let mut ab = bf16_bits(in_proj_a);
                        ab.extend_from_slice(&bf16_bits(in_proj_b));
                        q3d::HostDenseMixer::Delta(Box::new(HostDeltaNet {
                            in_proj_qkv: HostBf16Lin {
                                w: bf16_bits(in_proj_qkv),
                                n: conv_dim,
                                k: hidden,
                            },
                            in_proj_z: HostBf16Lin {
                                w: bf16_bits(in_proj_z),
                                n: value_dim,
                                k: hidden,
                            },
                            in_proj_ab: HostBf16Lin {
                                w: ab,
                                n: 2 * cfg.linear_num_value_heads,
                                k: hidden,
                            },
                            conv1d: conv1d.clone(),
                            a_log: a_log.clone(),
                            dt_bias: dt_bias.clone(),
                            norm_w: bf16_bits(norm_effective),
                            out_proj: HostBf16Lin {
                                w: bf16_bits(out_proj),
                                n: hidden,
                                k: value_dim,
                            },
                        }))
                    }
                    TinyMixerValues::Attn {
                        q,
                        k,
                        v,
                        o,
                        q_norm_effective,
                        k_norm_effective,
                    } => q3d::HostDenseMixer::Attn(Box::new(q3d::HostDenseAttention {
                        q: HostBf16Lin {
                            w: bf16_bits(q),
                            n: cfg.num_attention_heads * cfg.head_dim * 2,
                            k: hidden,
                        }
                        .into(),
                        k: HostBf16Lin {
                            w: bf16_bits(k),
                            n: cfg.num_key_value_heads * cfg.head_dim,
                            k: hidden,
                        }
                        .into(),
                        v: HostBf16Lin {
                            w: bf16_bits(v),
                            n: cfg.num_key_value_heads * cfg.head_dim,
                            k: hidden,
                        }
                        .into(),
                        o: HostBf16Lin {
                            w: bf16_bits(o),
                            n: hidden,
                            k: cfg.num_attention_heads * cfg.head_dim,
                        }
                        .into(),
                        q_norm: bf16_bits(q_norm_effective),
                        k_norm: bf16_bits(k_norm_effective),
                    })),
                };
                q3d::HostDenseLayer {
                    input_ln: bf16_bits(&l.input_ln_effective),
                    post_attn_ln: bf16_bits(&l.post_attn_ln_effective),
                    mixer,
                    mlp: q3d::HostDenseMlp {
                        gate: HostBf16Lin {
                            w: bf16_bits(&l.mlp_gate),
                            n: inter,
                            k: hidden,
                        }
                        .into(),
                        up: HostBf16Lin {
                            w: bf16_bits(&l.mlp_up),
                            n: inter,
                            k: hidden,
                        }
                        .into(),
                        down: HostBf16Lin {
                            w: bf16_bits(&l.mlp_down),
                            n: hidden,
                            k: inter,
                        }
                        .into(),
                    },
                    delta_fp8: q3d::DeltaFp8::default(),
                }
            })
            .collect();
        q3d::HostDenseWeights {
            embed: bf16_bits(&vals.embed),
            final_norm: bf16_bits(&vals.final_norm_effective),
            lm_head: bf16_bits(&vals.lm_head),
            layers,
        }
    }
}

#[cfg(feature = "wgpu")]
mod wgpu_tiny_geometry {
    use super::tiny_host_weights::host_dense_weights;
    use super::*;
    use nv_models::qwen3_5_dense_wgpu as q3d;

    fn have_gpu() -> bool {
        match nv_kernels::wgpu_backend::WgpuContext::shared() {
            Ok(ctx) => {
                eprintln!("[wgpu] adapter: {}", ctx.info.name);
                true
            }
            Err(e) => {
                eprintln!("[skip] no wgpu adapter: {e}");
                false
            }
        }
    }

    #[test]
    fn cpu_reference_forward_on_tiny_q38_geometry_is_finite_and_deterministic_without_any_gpu() {
        let cfg = tiny_q38_config_shrunk_from_the_real_release_config_keeping_every_structural_fact();
        let vals = tiny_weight_values_shared_by_every_backend_so_parity_compares_identical_models(
            &cfg, 0x9380_27b0_0001,
        );
        let hw = host_dense_weights(&cfg, &vals);
        let mut st_a = q3d::RefState::new(&cfg);
        let mut st_b = q3d::RefState::new(&cfg);
        for t in SMOKE_TOKENS {
            let a = q3d::reference_step(&cfg, &hw, &mut st_a, t).expect("reference step");
            let b = q3d::reference_step(&cfg, &hw, &mut st_b, t).expect("reference step twin");
            assert_eq!(a.len(), cfg.vocab_size);
            assert!(
                a.iter().all(|v| v.is_finite()),
                "reference logits must be finite on the q38 tiny geometry (gdn v/k ratio 3, gqa 6, rotary 0.25, output gate)"
            );
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "two identical reference runs diverged"
            );
        }
    }

    const WGPU_VS_F32_REF_WORST_REL_TOL_2_5E2_BF16_ROUNDS_EVERY_GEMV_OVER_8_LAYERS: f32 = 2.5e-2;

    #[test]
    fn wgpu_forward_matches_the_cpu_reference_on_tiny_q38_geometry_decode_and_prefill() {
        if !have_gpu() {
            return;
        }
        let cfg = tiny_q38_config_shrunk_from_the_real_release_config_keeping_every_structural_fact();
        let vals = tiny_weight_values_shared_by_every_backend_so_parity_compares_identical_models(
            &cfg, 0x9380_27b0_0001,
        );
        let hw = host_dense_weights(&cfg, &vals);

        let mut model = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32).expect("build wgpu model");
        let mut st = q3d::RefState::new(&cfg);
        let mut worst = 0f32;
        let mut agree = 0usize;
        for t in SMOKE_TOKENS {
            let (_, got) = model.decode_step_logits(t).expect("wgpu decode step");
            let want = q3d::reference_step(&cfg, &hw, &mut st, t).expect("reference step");
            worst = worst.max(worst_rel(&got, &want));
            assert!(
                got.iter().all(|v| v.is_finite()),
                "wgpu logits must be finite at pos {}",
                model.current_pos()
            );
            if argmax(&got) == argmax(&want) {
                agree += 1;
            }
        }
        eprintln!(
            "[q38-wgpu-smoke] basis: synthetic tiny geometry seed=0x938027b00001 backend=wgpu steps={} worst_rel={worst:.3e} argmax_agree={agree}/{}",
            SMOKE_TOKENS.len(),
            SMOKE_TOKENS.len()
        );
        assert!(
            worst < WGPU_VS_F32_REF_WORST_REL_TOL_2_5E2_BF16_ROUNDS_EVERY_GEMV_OVER_8_LAYERS,
            "wgpu drifted {worst:.3e} from the cpu reference"
        );
        assert_eq!(agree, SMOKE_TOKENS.len(), "wgpu argmax disagrees with the reference");

        let split = SMOKE_PREFILL_SPLIT_6_EXERCISES_MULTI_TOKEN_PREFILL_THEN_STEPWISE_DECODE;
        let mut pf_model =
            q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, 32).expect("build wgpu prefill model");
        pf_model
            .prefill_tokens(&SMOKE_TOKENS[..split])
            .expect("prefill tokens");
        assert_eq!(pf_model.current_pos(), split);
        let mut st2 = q3d::RefState::new(&cfg);
        for t in &SMOKE_TOKENS[..split] {
            q3d::reference_step(&cfg, &hw, &mut st2, *t).expect("reference prefix step");
        }
        let (_, got_after_prefill) = pf_model
            .decode_step_logits(SMOKE_TOKENS[split])
            .expect("decode after prefill");
        let want_after_prefill =
            q3d::reference_step(&cfg, &hw, &mut st2, SMOKE_TOKENS[split]).expect("reference step");
        let pf_rel = worst_rel(&got_after_prefill, &want_after_prefill);
        eprintln!("[q38-wgpu-smoke] prefill({split})+decode worst_rel={pf_rel:.3e}");
        assert!(
            pf_rel < WGPU_VS_F32_REF_WORST_REL_TOL_2_5E2_BF16_ROUNDS_EVERY_GEMV_OVER_8_LAYERS,
            "multi-token prefill drifted {pf_rel:.3e} from the stepwise reference"
        );
        assert_eq!(
            argmax(&got_after_prefill),
            argmax(&want_after_prefill),
            "prefill-then-decode argmax disagrees with the stepwise reference"
        );
    }
}

#[cfg(all(feature = "cuda", feature = "wgpu"))]
mod cuda_tiny_geometry {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use nv_models::qwen3_5_dense_wgpu as q3d;
    use nv_models::qwen3_5_moe::Qwen3Moe;
    use std::collections::HashMap;
    use std::path::PathBuf;

    fn bf16_tensor(vals: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(vals.to_vec(), shape, &Device::Cpu)
            .expect("cpu tensor")
            .to_dtype(DType::BF16)
            .expect("bf16 cast")
    }

    fn minus_one_because_both_loaders_add_one_to_zero_centered_norm_weights(v: &[f32]) -> Vec<f32> {
        v.iter().map(|x| x - 1.0).collect()
    }

    fn write_tiny_safetensors_dir(cfg: &Qwen3_5DenseConfig, vals: &TinyWeightValues) -> PathBuf {
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let hd = cfg.head_dim;
        let n_v = cfg.linear_num_value_heads;
        let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let value_dim = n_v * cfg.linear_value_head_dim;
        let conv_dim = 2 * key_dim + value_dim;
        let ks = cfg.linear_conv_kernel_dim;
        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert(
            "model.language_model.embed_tokens.weight".into(),
            bf16_tensor(&vals.embed, &[cfg.vocab_size, hidden]),
        );
        t.insert(
            "model.language_model.norm.weight".into(),
            bf16_tensor(
                &minus_one_because_both_loaders_add_one_to_zero_centered_norm_weights(
                    &vals.final_norm_effective,
                ),
                &[hidden],
            ),
        );
        t.insert(
            "lm_head.weight".into(),
            bf16_tensor(&vals.lm_head, &[cfg.vocab_size, hidden]),
        );
        for (i, l) in vals.layers.iter().enumerate() {
            let p = format!("model.language_model.layers.{i}");
            t.insert(
                format!("{p}.input_layernorm.weight"),
                bf16_tensor(
                    &minus_one_because_both_loaders_add_one_to_zero_centered_norm_weights(
                        &l.input_ln_effective,
                    ),
                    &[hidden],
                ),
            );
            t.insert(
                format!("{p}.post_attention_layernorm.weight"),
                bf16_tensor(
                    &minus_one_because_both_loaders_add_one_to_zero_centered_norm_weights(
                        &l.post_attn_ln_effective,
                    ),
                    &[hidden],
                ),
            );
            match &l.mixer {
                TinyMixerValues::Delta {
                    in_proj_qkv,
                    in_proj_z,
                    in_proj_a,
                    in_proj_b,
                    conv1d,
                    a_log,
                    dt_bias,
                    norm_effective,
                    out_proj,
                } => {
                    let q = format!("{p}.linear_attn");
                    t.insert(
                        format!("{q}.in_proj_qkv.weight"),
                        bf16_tensor(in_proj_qkv, &[conv_dim, hidden]),
                    );
                    t.insert(
                        format!("{q}.in_proj_z.weight"),
                        bf16_tensor(in_proj_z, &[value_dim, hidden]),
                    );
                    t.insert(
                        format!("{q}.in_proj_a.weight"),
                        bf16_tensor(in_proj_a, &[n_v, hidden]),
                    );
                    t.insert(
                        format!("{q}.in_proj_b.weight"),
                        bf16_tensor(in_proj_b, &[n_v, hidden]),
                    );
                    t.insert(
                        format!("{q}.conv1d.weight"),
                        bf16_tensor(conv1d, &[conv_dim, 1, ks]),
                    );
                    t.insert(format!("{q}.A_log"), bf16_tensor(a_log, &[n_v]));
                    t.insert(format!("{q}.dt_bias"), bf16_tensor(dt_bias, &[n_v]));
                    t.insert(
                        format!("{q}.norm.weight"),
                        bf16_tensor(norm_effective, &[cfg.linear_value_head_dim]),
                    );
                    t.insert(
                        format!("{q}.out_proj.weight"),
                        bf16_tensor(out_proj, &[hidden, value_dim]),
                    );
                }
                TinyMixerValues::Attn {
                    q,
                    k,
                    v,
                    o,
                    q_norm_effective,
                    k_norm_effective,
                } => {
                    let a = format!("{p}.self_attn");
                    t.insert(
                        format!("{a}.q_proj.weight"),
                        bf16_tensor(q, &[cfg.num_attention_heads * hd * 2, hidden]),
                    );
                    t.insert(
                        format!("{a}.k_proj.weight"),
                        bf16_tensor(k, &[cfg.num_key_value_heads * hd, hidden]),
                    );
                    t.insert(
                        format!("{a}.v_proj.weight"),
                        bf16_tensor(v, &[cfg.num_key_value_heads * hd, hidden]),
                    );
                    t.insert(
                        format!("{a}.o_proj.weight"),
                        bf16_tensor(o, &[hidden, cfg.num_attention_heads * hd]),
                    );
                    t.insert(
                        format!("{a}.q_norm.weight"),
                        bf16_tensor(
                            &minus_one_because_both_loaders_add_one_to_zero_centered_norm_weights(
                                q_norm_effective,
                            ),
                            &[hd],
                        ),
                    );
                    t.insert(
                        format!("{a}.k_norm.weight"),
                        bf16_tensor(
                            &minus_one_because_both_loaders_add_one_to_zero_centered_norm_weights(
                                k_norm_effective,
                            ),
                            &[hd],
                        ),
                    );
                }
            }
            t.insert(
                format!("{p}.mlp.gate_proj.weight"),
                bf16_tensor(&l.mlp_gate, &[inter, hidden]),
            );
            t.insert(
                format!("{p}.mlp.up_proj.weight"),
                bf16_tensor(&l.mlp_up, &[inter, hidden]),
            );
            t.insert(
                format!("{p}.mlp.down_proj.weight"),
                bf16_tensor(&l.mlp_down, &[hidden, inter]),
            );
        }
        let dir = std::env::temp_dir().join(format!("q38-tiny-geometry-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mk temp safetensors dir");
        candle_core::safetensors::save(&t, dir.join("model.safetensors")).expect("save tiny model");
        dir
    }

    const CUDA_BF16_VS_F32_REF_WORST_REL_TOL_2_5E2_BF16_ROUNDS_EVERY_GEMV_OVER_8_LAYERS: f32 =
        2.5e-2;

    fn logits_rows(logits: &Tensor, vocab: usize) -> Vec<Vec<f32>> {
        let flat = logits
            .to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to host");
        assert_eq!(flat.len() % vocab, 0, "logits not a multiple of vocab");
        flat.chunks(vocab).map(|c| c.to_vec()).collect()
    }

    #[test]
    fn cuda_forward_matches_the_cpu_reference_on_tiny_q38_geometry_chunk_prefill_then_stepwise() {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("[skip] no cuda device; the q38 cuda tiny-geometry smoke needs the card");
            return;
        };
        let cfg = tiny_q38_config_shrunk_from_the_real_release_config_keeping_every_structural_fact();
        let vals = tiny_weight_values_shared_by_every_backend_so_parity_compares_identical_models(
            &cfg, 0x9380_27b0_0001,
        );
        let dir = write_tiny_safetensors_dir(&cfg, &vals);
        let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("open tiny dir");
        let model =
            Qwen3Moe::from_loader_dense(cfg.clone(), &weights, &device).expect("build cuda model");
        drop(weights);
        let mut cache = model.new_kv_cache(32).expect("kv cache");

        let hw = super::tiny_host_weights::host_dense_weights(&cfg, &vals);
        let mut st = q3d::RefState::new(&cfg);

        let split = SMOKE_PREFILL_SPLIT_6_EXERCISES_MULTI_TOKEN_PREFILL_THEN_STEPWISE_DECODE;
        let prefix: Vec<u32> = SMOKE_TOKENS[..split].to_vec();
        let tokens = Tensor::from_vec(prefix.clone(), (1usize, split), &device).expect("tokens");
        let positions =
            Tensor::from_vec((0..split as i32).collect::<Vec<_>>(), split, &device).expect("pos");
        let logits = model
            .forward_with_cache(&tokens, &positions, &mut cache)
            .expect("chunk prefill forward");
        let rows = logits_rows(&logits, cfg.vocab_size);
        let mut want_last: Vec<f32> = Vec::new();
        for tk in &prefix {
            want_last = q3d::reference_step(&cfg, &hw, &mut st, *tk).expect("reference step");
        }
        let got_last = rows.last().expect("chunk prefill emitted no rows").clone();
        let chunk_rel = worst_rel(&got_last, &want_last);
        eprintln!(
            "[q38-cuda-smoke] basis: synthetic tiny geometry seed=0x938027b00001 backend=cuda chunk_prefill({split}) worst_rel={chunk_rel:.3e}"
        );
        assert!(
            got_last.iter().all(|v| v.is_finite()),
            "cuda chunk-prefill logits must be finite"
        );
        assert!(
            chunk_rel < CUDA_BF16_VS_F32_REF_WORST_REL_TOL_2_5E2_BF16_ROUNDS_EVERY_GEMV_OVER_8_LAYERS,
            "cuda chunk prefill drifted {chunk_rel:.3e} from the stepwise cpu reference"
        );
        assert_eq!(argmax(&got_last), argmax(&want_last), "chunk prefill argmax");

        let mut worst = chunk_rel;
        let mut agree = 0usize;
        let tail = &SMOKE_TOKENS[split..];
        for (i, tk) in tail.iter().enumerate() {
            let pos = split + i;
            let tokens = Tensor::from_vec(vec![*tk], (1usize, 1usize), &device).expect("token");
            let positions = Tensor::from_vec(vec![pos as i32], 1usize, &device).expect("pos");
            let logits = model
                .forward_with_cache(&tokens, &positions, &mut cache)
                .expect("cuda decode step");
            let got = logits_rows(&logits, cfg.vocab_size).pop().expect("row");
            let want = q3d::reference_step(&cfg, &hw, &mut st, *tk).expect("reference step");
            worst = worst.max(worst_rel(&got, &want));
            assert!(got.iter().all(|v| v.is_finite()), "cuda logits finite at pos {pos}");
            if argmax(&got) == argmax(&want) {
                agree += 1;
            }
        }
        eprintln!(
            "[q38-cuda-smoke] stepwise decode steps={} worst_rel={worst:.3e} argmax_agree={agree}/{}",
            tail.len(),
            tail.len()
        );
        assert!(
            worst < CUDA_BF16_VS_F32_REF_WORST_REL_TOL_2_5E2_BF16_ROUNDS_EVERY_GEMV_OVER_8_LAYERS,
            "cuda decode drifted {worst:.3e} from the cpu reference"
        );
        assert_eq!(agree, tail.len(), "cuda argmax disagrees with the reference");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(feature = "cuda")]
mod cuda_quantized_arm_tiny_fixture {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    type StEntry = (String, &'static str, Vec<usize>, Vec<u8>);

    fn write_safetensors_by_hand_because_candle_save_has_no_f8e4m3(
        path: &Path,
        entries: &[StEntry],
    ) {
        let mut header = String::from("{");
        let mut off = 0usize;
        for (i, (name, dt, shape, bytes)) in entries.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            let end = off + bytes.len();
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dt}\",\"shape\":{shape:?},\"data_offsets\":[{off},{end}]}}"
            ));
            off = end;
        }
        header.push('}');
        let mut hb = header.into_bytes();
        while hb.len() % 8 != 0 {
            hb.push(b' ');
        }
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(hb.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&hb).unwrap();
        for (_, _, _, bytes) in entries {
            f.write_all(bytes).unwrap();
        }
    }

    fn quantized_fixture_dir(tag: &str, entries: &[StEntry]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "q38-tiny-quant-{tag}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("mk quant fixture dir");
        write_safetensors_by_hand_because_candle_save_has_no_f8e4m3(
            &dir.join("model.safetensors"),
            entries,
        );
        dir
    }

    fn host_f32(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to host")
    }

    #[test]
    fn fp8_rowscale_arm_on_a_tiny_fixture_with_real_dtypes_is_bitwise_the_host_dequant() {
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("[skip] no cuda device; the fp8 tiny-fixture arm needs the card");
            return;
        };
        let cfg = tiny_q38_config_shrunk_from_the_real_release_config_keeping_every_structural_fact();
        let n = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
            + cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let k = cfg.hidden_size;
        let mut rng = Lcg::new(0x9380_f8f8_0001);
        let vals: Vec<half::bf16> = (0..n * k)
            .map(|_| half::bf16::from_f32(rng.next_f32() * 0.5))
            .collect();
        let (bytes, scales_f32) =
            nv_quant::fp8::quantize_e4m3_per_row(&vals, n, k).expect("quantize fp8 rows");
        let scale_bf16_stored_like_the_real_checkpoint: Vec<half::bf16> =
            scales_f32.iter().map(|s| half::bf16::from_f32(*s)).collect();
        let scale_bytes: Vec<u8> = scale_bf16_stored_like_the_real_checkpoint
            .iter()
            .flat_map(|s| s.to_bits().to_le_bytes())
            .collect();
        let module = "model.language_model.layers.0.linear_attn.in_proj_qkv";
        let entries: Vec<StEntry> = vec![
            (format!("{module}.weight"), "F8_E4M3", vec![n, k], bytes.clone()),
            (format!("{module}.weight_scale"), "BF16", vec![n, 1], scale_bytes),
        ];
        let dir = quantized_fixture_dir("fp8", &entries);
        let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("open fixture");
        let lin = nv_layers::linear::fp8_e4m3_rowscale_checkpoint_dequant_linear(
            &weights,
            module,
            n,
            k,
            DType::BF16,
        )
        .expect("fp8 dequant arm on the tiny fixture");
        let got = host_f32(lin.weight().expect("bf16 storage"));
        let rounded_scales: Vec<f32> = scale_bf16_stored_like_the_real_checkpoint
            .iter()
            .map(|s| s.to_f32())
            .collect();
        let reference =
            nv_quant::fp8::dequantize_e4m3_per_row(&bytes, n, k, &rounded_scales).expect("ref");
        let mismatches = got
            .iter()
            .zip(&reference)
            .filter(|(g, r)| **g != half::bf16::from_f32(**r).to_f32())
            .count();
        eprintln!(
            "[q38-tiny-quant] fp8 arm n={n} k={k} mismatches={mismatches} basis=synthetic seed=0x9380f8f80001"
        );
        assert_eq!(
            mismatches, 0,
            "fp8 rowscale loader arm deviates from host dequantize_e4m3_per_row on the tiny fixture"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn nvfp4_native_gemm_arm_on_a_tiny_fixture_with_real_dtypes_matches_host_dequant_forward() {
        let Ok(device) = Device::new_cuda_with_stream(0) else {
            eprintln!("[skip] no cuda device; the nvfp4 tiny-fixture arm needs the card");
            return;
        };
        let cfg = tiny_q38_config_shrunk_from_the_real_release_config_keeping_every_structural_fact();
        let (n, k) = (cfg.intermediate_size, cfg.hidden_size);
        assert!(
            n >= nv_quant::nvfp4::MIN_TILE && k >= nv_quant::nvfp4::MIN_TILE,
            "tiny mlp [{n}, {k}] fell below the nvfp4 MIN_TILE {}; the fixture no longer \
             exercises the native gemm arm",
            nv_quant::nvfp4::MIN_TILE
        );
        let stored_global_2_so_a_dropped_division_doubles_every_weight = 2.0f32;
        let mut rng = Lcg::new(0x9380_4bed_0001);
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|_| (0..k).map(|_| rng.next_f32() * 0.5).collect())
            .collect();
        let q = nv_quant::nvfp4::Nvfp4Tensor::quantize_rows_with_global(
            &rows,
            stored_global_2_so_a_dropped_division_doubles_every_weight,
        );
        let module = "model.language_model.layers.0.mlp.gate_proj";
        let entries: Vec<StEntry> = vec![
            (format!("{module}.weight_packed"), "U8", vec![n, k / 2], q.data.clone()),
            (
                format!("{module}.weight_scale"),
                "F8_E4M3",
                vec![n, k / 16],
                q.scales.clone(),
            ),
            (
                format!("{module}.weight_global_scale"),
                "F32",
                vec![1],
                stored_global_2_so_a_dropped_division_doubles_every_weight
                    .to_le_bytes()
                    .to_vec(),
            ),
            (
                format!("{module}.input_global_scale"),
                "F32",
                vec![1],
                1.0f32.to_le_bytes().to_vec(),
            ),
        ];
        let dir = quantized_fixture_dir("nvfp4", &entries);
        let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("open fixture");
        let dev = match &device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let runner = Arc::new(Mutex::new(
            nv_quant::nvfp4::Nvfp4GemmRunner::new(dev.cuda_stream()).expect("nvfp4 runner"),
        ));
        let native =
            nv_layers::moe::nvfp4_linear_from_disk_pub(&weights, module, n, k, runner, &device)
                .expect("native nvfp4 arm on the tiny fixture");
        let host_w = nv_quant::nvfp4::dequantize_packed_linear(
            &q.data,
            &q.scales,
            n,
            k,
            1.0 / stored_global_2_so_a_dropped_division_doubles_every_weight,
        );
        let host_bf: Vec<half::bf16> = host_w.iter().map(|v| half::bf16::from_f32(*v)).collect();
        let dense = nv_layers::linear::Linear::new(
            Tensor::from_vec(host_bf, (n, k), &device).expect("host weight"),
            None,
        )
        .expect("dense reference linear");
        let rows_x = 4usize;
        let xv: Vec<f32> = (0..rows_x * k).map(|_| rng.next_f32()).collect();
        let x = Tensor::from_vec(xv, (rows_x, k), &device)
            .expect("x")
            .to_dtype(DType::BF16)
            .expect("x bf16");
        let y_native = host_f32(&native.forward(&x).expect("native forward"));
        let y_dense = host_f32(&dense.forward(&x).expect("dense forward"));
        let rms = |v: &[f32]| {
            (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
        };
        let dot: f64 = y_native
            .iter()
            .zip(&y_dense)
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum();
        let rms_n = rms(&y_native);
        let rms_d = rms(&y_dense);
        let cosine = dot / (rms_n * rms_d * y_native.len() as f64).max(1e-30);
        let ratio = rms_n / rms_d.max(1e-30);
        eprintln!(
            "[q38-tiny-quant] nvfp4 arm n={n} k={k} ratio={ratio:.4} cosine={cosine:.5} basis=synthetic seed=0x93804bed0001"
        );
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "native nvfp4 magnitude is {ratio:.3e}x the host dequant; the stored global of 2 \
             makes a dropped or inverted division show as an exact factor of 4"
        );
        assert!(
            cosine > 0.98,
            "native nvfp4 direction diverges from host dequant (cosine {cosine:.4}); magnitude \
             is fine so suspect the block-scale swizzle or packing order"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(feature = "cuda")]
mod cuda_real_load {
    use super::hub_snapshot;
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use nv_models::qwen3_5_moe::Qwen3Moe;
    use nv_weights::{QuantizationConfig, WeightLoader};

    #[test]
    #[ignore = "loads the real ~22.6 GB unsloth/Qwen3.8-27B-NVFP4 on the cuda dense arm; set NV_Q38_TEST=1"]
    fn qwen3_8_27b_real_checkpoint_loads_on_the_cuda_dense_arm_or_reports_the_nvfp4_format_gap() {
        if std::env::var("NV_Q38_TEST").as_deref() != Ok("1") {
            panic!(
                "set NV_Q38_TEST=1 to run the real-checkpoint cuda load (an #[ignore] suite that \
                 also silently skipped when invoked explicitly would hide the format gap)"
            );
        }
        let dir = std::env::var("NV_QWEN38_DIR")
            .map(std::path::PathBuf::from)
            .ok()
            .or_else(|| {
                hub_snapshot::snapshot_of(
                    "unsloth/Qwen3.8-27B-NVFP4",
                    &["config.json", "tokenizer.json", "*.safetensors"],
                )
            })
            .expect("no hydrated unsloth/Qwen3.8-27B-NVFP4 snapshot; set NV_QWEN38_DIR");
        eprintln!("[q38-cuda-real] checkpoint={}", dir.display());
        let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
        let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("parse dense config");
        let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("parse quant config");
        let device = Device::new_cuda_with_stream(0).expect("cuda");
        let weights = WeightLoader::open_dir(&dir, &device).expect("open weights");
        let t0 = std::time::Instant::now();
        let model = Qwen3Moe::from_loader_dense_quantized(cfg.clone(), &weights, &qcfg, &device)
            .expect(
                "build Qwen3.8-27B on the cuda dense arm; this checkpoint is compressed-tensors \
                 MIXED fp8 (attn + last-8-layer mlp + lm_head) plus nvfp4 (remaining mlp \
                 weight_packed/weight_scale/weight_global_scale), so a trip here is the track-1 \
                 format gap: the dense loader resolved only a single collapsed scheme",
            );
        drop(weights);
        eprintln!("[q38-cuda-real] loaded in {:.1}s", t0.elapsed().as_secs_f64());
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .expect("tokenizer.json in the snapshot");
        let question = std::env::var("NV_QWEN38_PROMPT")
            .unwrap_or_else(|_| "Explain, in a few sentences, why the sky appears blue.".into());
        let text = format!(
            "<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let prompt: Vec<u32> = tok
            .encode(text.as_str(), false)
            .expect("encode chat prompt")
            .get_ids()
            .to_vec();
        assert!(
            prompt.len() > 8,
            "chat prompt tokenized to only {} ids; a near-empty prompt makes greedy decode loop \
             whitespace on a healthy model and the degeneracy assert below reads that as a \
             dequant defect",
            prompt.len()
        );
        let new_tokens = 48usize;
        let mut cache = model
            .new_kv_cache(prompt.len() + new_tokens + 16)
            .expect("kv cache");
        let p = prompt.len();
        let tokens = Tensor::from_vec(prompt.clone(), (1usize, p), &device).expect("prompt");
        let positions =
            Tensor::from_vec((0..p as i32).collect::<Vec<_>>(), p, &device).expect("positions");
        let logits = model
            .forward_with_cache(&tokens, &positions, &mut cache)
            .expect("chunk prefill");
        let mut last_row = logits
            .to_dtype(DType::F32)
            .expect("f32")
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("host");
        let mut cur = super::argmax(&last_row[last_row.len() - cfg.vocab_size..]);
        let mut out: Vec<u32> = vec![cur];
        let mut pos = p;
        for _ in 1..new_tokens {
            let tokens = Tensor::from_vec(vec![cur], (1usize, 1usize), &device).expect("token");
            let positions = Tensor::from_vec(vec![pos as i32], 1usize, &device).expect("pos");
            let logits = model
                .forward_with_cache(&tokens, &positions, &mut cache)
                .expect("decode step");
            last_row = logits
                .to_dtype(DType::F32)
                .expect("f32")
                .flatten_all()
                .expect("flatten")
                .to_vec1::<f32>()
                .expect("host");
            let n_nan = last_row.iter().filter(|v| !v.is_finite()).count();
            assert_eq!(n_nan, 0, "non-finite logits at pos {pos}");
            cur = super::argmax(&last_row[last_row.len() - cfg.vocab_size..]);
            out.push(cur);
            pos += 1;
        }
        assert_eq!(last_row.len() % cfg.vocab_size, 0);
        let decoded = tok.decode(&out, false).unwrap_or_default();
        eprintln!(
            "[q38-cuda-real] basis: checkpoint={} backend=cuda batch=1 prompt_tokens={} greedy_tokens={:?} text={:?}",
            dir.display(),
            p,
            out,
            decoded
        );
        let distinct: std::collections::HashSet<u32> = out.iter().copied().collect();
        assert!(
            distinct.len() > 4,
            "{new_tokens} greedy tokens from the chat prompt collapsed to {} distinct ids {:?}; \
             the prompt is a real chat-template prompt, so degeneracy here is a genuine \
             decode/dequant defect (a bare-BOS prompt loops \\n\\n on a healthy checkpoint and \
             must not be used as this oracle)",
            distinct.len(),
            out.first()
        );
    }
}
