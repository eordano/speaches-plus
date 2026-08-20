use candle_core::{DType, Device, Tensor};
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_models::qwen3_mm_splice::{
    build_mrope_positions_matching_hf_get_rope_index,
    interleaved_axis_of_half_freq_matching_hf_overwrite_semantics, mrope_cos_sin_rows,
    mrope_section_from_hf_json_str, splice_image_rows_into_embedded, Qwen3ImageRowSplice,
    QWEN3_5_IMAGE_TOKEN_ID_FROM_THE_RELEASE_CONFIG, QWEN3_5_MROPE_SECTION_FROM_THE_RELEASE_CONFIG,
};

const QWEN38_27B_CONFIG_JSON: &str = include_str!("qwen3_8_27b_config.json");

fn release_section() -> [usize; 3] {
    let s = mrope_section_from_hf_json_str(QWEN38_27B_CONFIG_JSON)
        .expect("release config carries mrope_interleaved sections");
    assert_eq!(s, QWEN3_5_MROPE_SECTION_FROM_THE_RELEASE_CONFIG);
    s
}

#[test]
fn release_image_token_id_matches_the_config_fixture() {
    let v: serde_json::Value = serde_json::from_str(QWEN38_27B_CONFIG_JSON).unwrap();
    assert_eq!(
        v["image_token_id"].as_u64().unwrap() as u32,
        QWEN3_5_IMAGE_TOKEN_ID_FROM_THE_RELEASE_CONFIG
    );
}

#[test]
fn interleave_axis_map_consumes_exactly_11_11_10_with_hf_overwrite_semantics() {
    let section = release_section();
    let half: usize = section.iter().sum();
    assert_eq!(half, 32, "rotary_dim 64 has 32 half-frequencies");
    let mut consumed = [0usize; 3];
    for j in 0..half {
        let axis = interleaved_axis_of_half_freq_matching_hf_overwrite_semantics(j, section);
        let want = match j % 3 {
            1 if j < 3 * section[1] => 1,
            2 if j < 3 * section[2] => 2,
            _ => 0,
        };
        assert_eq!(axis, want, "freq {j}");
        consumed[axis] += 1;
    }
    assert_eq!(consumed.to_vec(), section.to_vec());
    assert_eq!(
        interleaved_axis_of_half_freq_matching_hf_overwrite_semantics(31, section),
        1,
        "index 31 stays h under overwrite semantics (31 < 3*11) even though the omni \
         tail-to-t rule would claim it for t"
    );
    assert_eq!(
        interleaved_axis_of_half_freq_matching_hf_overwrite_semantics(30, section),
        0
    );
    assert_eq!(
        interleaved_axis_of_half_freq_matching_hf_overwrite_semantics(29, section),
        2
    );
}

const IMG: u32 = 900;

#[test]
fn mrope_positions_match_hf_get_rope_index_on_a_text_image_text_prompt() {
    let tokens = [5u32, 7, IMG, IMG, IMG, IMG, IMG, IMG, 9];
    let pos = build_mrope_positions_matching_hf_get_rope_index(&tokens, IMG, &[(1, 2, 3)])
        .expect("positions");
    assert_eq!(pos.t, vec![0, 1, 2, 2, 2, 2, 2, 2, 5]);
    assert_eq!(pos.h, vec![0, 1, 2, 2, 2, 3, 3, 3, 5]);
    assert_eq!(pos.w, vec![0, 1, 2, 3, 4, 2, 3, 4, 5]);
    assert_eq!(
        pos.delta_added_to_token_index_for_every_position_after_this_prefill,
        -3,
        "max position 5 over 9 tokens: the next decode position is token_index - 3"
    );
    assert_eq!(pos.decode_position(9), 6);
    assert!(!pos.is_text_degenerate());
}

#[test]
fn mrope_positions_without_images_are_the_identity_with_zero_delta() {
    let tokens = [4u32, 8, 15, 16, 23, 42];
    let pos = build_mrope_positions_matching_hf_get_rope_index(&tokens, IMG, &[]).expect("text");
    assert!(pos.is_text_degenerate());
    assert_eq!(pos.t, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(
        pos.delta_added_to_token_index_for_every_position_after_this_prefill,
        0
    );
}

#[test]
fn mrope_positions_reject_grid_and_run_mismatches() {
    let tokens = [1u32, IMG, IMG, 2];
    assert!(
        build_mrope_positions_matching_hf_get_rope_index(&tokens, IMG, &[(1, 1, 3)]).is_err(),
        "grid covering 3 tokens against a 2-token run must fail"
    );
    assert!(
        build_mrope_positions_matching_hf_get_rope_index(&tokens, IMG, &[]).is_err(),
        "an image run with no grid must fail"
    );
    assert!(
        build_mrope_positions_matching_hf_get_rope_index(&tokens, IMG, &[(1, 1, 2), (1, 1, 2)])
            .is_err(),
        "an unconsumed grid must fail"
    );
}

fn cpu_rope(head_dim: usize, max_seq: usize, base: f32) -> Rope {
    Rope::new(
        RopeConfig {
            head_dim,
            max_seq_len: max_seq,
            base,
            kind: RopeKind::Standard,
        },
        &Device::Cpu,
    )
    .expect("cpu rope")
}

fn rows_of(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

#[test]
fn degenerate_positions_reproduce_the_1d_table_rows_bitwise() {
    let rope = cpu_rope(16, 32, 10_000_000.0);
    let tokens = [3u32, 1, 4, 1, 5, 9, 2, 6];
    let pos = build_mrope_positions_matching_hf_get_rope_index(&tokens, IMG, &[]).unwrap();
    let (cos, sin) = mrope_cos_sin_rows(&rope, &pos, [3, 3, 2], &Device::Cpu).unwrap();
    let want_cos = rows_of(&rope.cos().narrow(0, 0, tokens.len()).unwrap());
    let want_sin = rows_of(&rope.sin().narrow(0, 0, tokens.len()).unwrap());
    let got_cos = rows_of(&cos);
    let got_sin = rows_of(&sin);
    for (i, (g, w)) in got_cos.iter().zip(want_cos.iter()).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "cos element {i}");
    }
    for (i, (g, w)) in got_sin.iter().zip(want_sin.iter()).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "sin element {i}");
    }
}

#[test]
fn divergent_positions_gather_each_frequency_from_its_axis_row_bitwise() {
    let rope = cpu_rope(16, 32, 10_000_000.0);
    let section = [3usize, 3, 2];
    let half = 8usize;
    let tokens = [5u32, 7, IMG, IMG, IMG, IMG, IMG, IMG, 9];
    let pos = build_mrope_positions_matching_hf_get_rope_index(&tokens, IMG, &[(1, 2, 3)]).unwrap();
    let (cos, _sin) = mrope_cos_sin_rows(&rope, &pos, section, &Device::Cpu).unwrap();
    let got = rows_of(&cos);
    let table = rows_of(rope.cos());
    for i in 0..tokens.len() {
        for j in 0..half {
            let axis = interleaved_axis_of_half_freq_matching_hf_overwrite_semantics(j, section);
            let p = [&pos.t, &pos.h, &pos.w][axis][i] as usize;
            assert_eq!(
                got[i * half + j].to_bits(),
                table[p * half + j].to_bits(),
                "token {i} freq {j} must read the axis-{axis} row of the 1D table"
            );
        }
    }
}

#[test]
fn synthetic_row_table_rope_matches_the_real_rope_on_cpu_bitwise() {
    use nv_models::qwen3_mm_splice::{mrope_rope_one_row_per_token, Qwen3MropePositions};
    let device = Device::Cpu;
    let real = cpu_rope(16, 32, 10_000_000.0);
    let p: Vec<u32> = vec![3, 7, 1, 0, 9, 4, 2, 8];
    let mrope = Qwen3MropePositions {
        t: p.clone(),
        h: p.clone(),
        w: p.clone(),
        delta_added_to_token_index_for_every_position_after_this_prefill: 0,
    };
    let synth = mrope_rope_one_row_per_token(&real, &mrope, [3, 3, 2], &device).expect("synth");
    let vals: Vec<f32> = (0..8 * 2 * 16).map(|i| ((i as f32) * 0.13).sin()).collect();
    let q = Tensor::from_vec(vals.clone(), (1usize, 8, 2, 16), &device).unwrap();
    let k = Tensor::from_vec(vals, (1usize, 8, 2, 16), &device).unwrap();
    let pos_real = Tensor::from_vec(p, 8usize, &device).unwrap();
    let pos_iota = Tensor::from_vec((0u32..8).collect::<Vec<_>>(), 8usize, &device).unwrap();
    let (qa, _) = real.apply(&q, &k, &pos_real).expect("real apply");
    let (qb, _) = synth.apply(&q, &k, &pos_iota).expect("synth apply");
    let va = rows_of(&qa);
    let vb = rows_of(&qb);
    let diff = va
        .iter()
        .zip(vb.iter())
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    assert_eq!(diff, 0, "cpu: {diff}/{} rope outputs differ", va.len());
}

#[test]
fn splice_replaces_exactly_the_named_rows_and_rejects_misuse() {
    let dev = Device::Cpu;
    let hidden = 4usize;
    let x_vals: Vec<f32> = (0..6 * hidden).map(|i| i as f32).collect();
    let x = Tensor::from_vec(x_vals.clone(), (1, 6, hidden), &dev).unwrap();
    let rows = Tensor::from_vec(vec![-1f32; 2 * hidden], (2, hidden), &dev).unwrap();
    let one = Tensor::from_vec(vec![-2f32; hidden], (1, hidden), &dev).unwrap();
    let out = splice_image_rows_into_embedded(
        &x,
        &[
            Qwen3ImageRowSplice {
                position: 1,
                rows: rows.clone(),
            },
            Qwen3ImageRowSplice {
                position: 4,
                rows: one.clone(),
            },
        ],
    )
    .unwrap();
    let got = rows_of(&out);
    for (r, want_row) in [
        (0usize, vec![0f32, 1., 2., 3.]),
        (1, vec![-1f32; 4]),
        (2, vec![-1f32; 4]),
        (3, vec![12f32, 13., 14., 15.]),
        (4, vec![-2f32; 4]),
        (5, vec![20f32, 21., 22., 23.]),
    ] {
        assert_eq!(&got[r * hidden..(r + 1) * hidden], &want_row[..], "row {r}");
    }
    assert!(
        splice_image_rows_into_embedded(
            &x,
            &[
                Qwen3ImageRowSplice {
                    position: 1,
                    rows: rows.clone()
                },
                Qwen3ImageRowSplice {
                    position: 2,
                    rows: one.clone()
                },
            ],
        )
        .is_err(),
        "overlapping splices must fail"
    );
    assert!(
        splice_image_rows_into_embedded(
            &x,
            &[Qwen3ImageRowSplice {
                position: 5,
                rows: rows
            }],
        )
        .is_err(),
        "overrunning splice must fail"
    );
}

#[cfg(feature = "cuda")]
mod ctx_timing_common;

#[cfg(feature = "cuda")]
mod cuda_tiny_prefill {
    use super::*;
    use nv_models::qwen3_5_moe::{LayerType, Qwen3Moe, Qwen3_5DenseConfig};
    use std::collections::HashMap;
    use std::path::PathBuf;

    const TINY_SECTION_3_3_2_TILES_THE_8_HALF_FREQS_OF_ROTARY_DIM_16: [usize; 3] = [3, 3, 2];

    const DELTA_A_LOG_BIAS_KEEPS_RECURRENT_STATE_OBSERVABLE_PAST_A_16_TOKEN_PROMPT: f32 = -4.0;

    const ATTN_WEIGHT_SCALE_KEEPS_THE_ATTENTION_BRANCH_ABOVE_BF16_RESIDUAL_RESOLUTION: f32 = 0.12;

    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32 - 1.0
        }
        fn vec(&mut self, n: usize, s: f32) -> Vec<f32> {
            (0..n).map(|_| self.next_f32() * s).collect()
        }
        fn norm_vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| 0.1 * self.next_f32()).collect()
        }
    }

    fn tiny_cfg() -> Qwen3_5DenseConfig {
        let mut cfg = Qwen3_5DenseConfig::from_hf_json_str(QWEN38_27B_CONFIG_JSON)
            .expect("release config parses");
        cfg.layer_types.truncate(8);
        cfg.num_hidden_layers = 8;
        cfg.hidden_size = 128;
        cfg.head_dim = 64;
        cfg.num_attention_heads = 12;
        cfg.num_key_value_heads = 2;
        cfg.linear_num_value_heads = 6;
        cfg.linear_num_key_heads = 2;
        cfg.linear_key_head_dim = 16;
        cfg.linear_value_head_dim = 16;
        cfg.intermediate_size = 192;
        cfg.vocab_size = 64;
        cfg.max_position_embeddings = 64;
        cfg.bos_token_id = None;
        cfg.eos_token_id = 1;
        assert_eq!(cfg.rotary_dim(), 16);
        cfg
    }

    fn bf16_tensor(vals: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(vals.to_vec(), shape, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
    }

    fn write_tiny_dense_checkpoint(cfg: &Qwen3_5DenseConfig, seed: u64) -> PathBuf {
        let mut r = Lcg::new(seed);
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
            bf16_tensor(&r.vec(cfg.vocab_size * hidden, 0.6), &[cfg.vocab_size, hidden]),
        );
        t.insert(
            "model.language_model.norm.weight".into(),
            bf16_tensor(&r.norm_vec(hidden), &[hidden]),
        );
        t.insert(
            "lm_head.weight".into(),
            bf16_tensor(&r.vec(cfg.vocab_size * hidden, 0.2), &[cfg.vocab_size, hidden]),
        );
        for (i, lt) in cfg.layer_types.iter().enumerate() {
            let p = format!("model.language_model.layers.{i}");
            t.insert(
                format!("{p}.input_layernorm.weight"),
                bf16_tensor(&r.norm_vec(hidden), &[hidden]),
            );
            t.insert(
                format!("{p}.post_attention_layernorm.weight"),
                bf16_tensor(&r.norm_vec(hidden), &[hidden]),
            );
            match lt {
                LayerType::LinearAttention => {
                    let q = format!("{p}.linear_attn");
                    t.insert(
                        format!("{q}.in_proj_qkv.weight"),
                        bf16_tensor(&r.vec(conv_dim * hidden, 0.12), &[conv_dim, hidden]),
                    );
                    t.insert(
                        format!("{q}.in_proj_z.weight"),
                        bf16_tensor(&r.vec(value_dim * hidden, 0.12), &[value_dim, hidden]),
                    );
                    t.insert(
                        format!("{q}.in_proj_a.weight"),
                        bf16_tensor(&r.vec(n_v * hidden, 0.12), &[n_v, hidden]),
                    );
                    t.insert(
                        format!("{q}.in_proj_b.weight"),
                        bf16_tensor(&r.vec(n_v * hidden, 0.12), &[n_v, hidden]),
                    );
                    t.insert(
                        format!("{q}.conv1d.weight"),
                        bf16_tensor(&r.vec(conv_dim * ks, 0.4), &[conv_dim, 1, ks]),
                    );
                    t.insert(
                        format!("{q}.A_log"),
                        bf16_tensor(
                            &r.vec(n_v, 0.3)
                                .iter()
                                .map(|v| {
                                    v + DELTA_A_LOG_BIAS_KEEPS_RECURRENT_STATE_OBSERVABLE_PAST_A_16_TOKEN_PROMPT
                                })
                                .collect::<Vec<_>>(),
                            &[n_v],
                        ),
                    );
                    t.insert(
                        format!("{q}.dt_bias"),
                        bf16_tensor(&r.vec(n_v, 0.5), &[n_v]),
                    );
                    t.insert(
                        format!("{q}.norm.weight"),
                        bf16_tensor(
                            &r.norm_vec(cfg.linear_value_head_dim),
                            &[cfg.linear_value_head_dim],
                        ),
                    );
                    t.insert(
                        format!("{q}.out_proj.weight"),
                        bf16_tensor(&r.vec(hidden * value_dim, 0.12), &[hidden, value_dim]),
                    );
                }
                LayerType::FullAttention => {
                    let a = format!("{p}.self_attn");
                    let q_out = cfg.num_attention_heads * hd * 2;
                    let kv_out = cfg.num_key_value_heads * hd;
                    t.insert(
                        format!("{a}.q_proj.weight"),
                        bf16_tensor(
                            &r.vec(q_out * hidden, ATTN_WEIGHT_SCALE_KEEPS_THE_ATTENTION_BRANCH_ABOVE_BF16_RESIDUAL_RESOLUTION),
                            &[q_out, hidden],
                        ),
                    );
                    t.insert(
                        format!("{a}.k_proj.weight"),
                        bf16_tensor(
                            &r.vec(kv_out * hidden, ATTN_WEIGHT_SCALE_KEEPS_THE_ATTENTION_BRANCH_ABOVE_BF16_RESIDUAL_RESOLUTION),
                            &[kv_out, hidden],
                        ),
                    );
                    t.insert(
                        format!("{a}.v_proj.weight"),
                        bf16_tensor(
                            &r.vec(kv_out * hidden, ATTN_WEIGHT_SCALE_KEEPS_THE_ATTENTION_BRANCH_ABOVE_BF16_RESIDUAL_RESOLUTION),
                            &[kv_out, hidden],
                        ),
                    );
                    t.insert(
                        format!("{a}.o_proj.weight"),
                        bf16_tensor(
                            &r.vec(
                                hidden * cfg.num_attention_heads * hd,
                                ATTN_WEIGHT_SCALE_KEEPS_THE_ATTENTION_BRANCH_ABOVE_BF16_RESIDUAL_RESOLUTION,
                            ),
                            &[hidden, cfg.num_attention_heads * hd],
                        ),
                    );
                    t.insert(
                        format!("{a}.q_norm.weight"),
                        bf16_tensor(&r.norm_vec(hd), &[hd]),
                    );
                    t.insert(
                        format!("{a}.k_norm.weight"),
                        bf16_tensor(&r.norm_vec(hd), &[hd]),
                    );
                }
            }
            t.insert(
                format!("{p}.mlp.gate_proj.weight"),
                bf16_tensor(&r.vec(inter * hidden, 0.15), &[inter, hidden]),
            );
            t.insert(
                format!("{p}.mlp.up_proj.weight"),
                bf16_tensor(&r.vec(inter * hidden, 0.15), &[inter, hidden]),
            );
            t.insert(
                format!("{p}.mlp.down_proj.weight"),
                bf16_tensor(&r.vec(hidden * inter, 0.15), &[hidden, inter]),
            );
        }
        let dir = std::env::temp_dir().join(format!("q38-mm-splice-tiny-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mk temp safetensors dir");
        candle_core::safetensors::save(&t, dir.join("model.safetensors")).expect("save tiny model");
        dir
    }

    fn embed_rows_for(model: &Qwen3Moe, ids: &[u32], device: &Device) -> Tensor {
        let idx = Tensor::from_vec(ids.to_vec(), ids.len(), device).unwrap();
        model.embed_weight().index_select(&idx, 0).unwrap()
    }

    fn last_row_bits(logits: &Tensor) -> Vec<u32> {
        logits
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    const TINY_IMG_TOKEN: u32 = 60;

    #[test]
    fn synthetic_row_table_rope_matches_the_real_rope_on_cuda_bitwise() {
        let _one_gpu_test_at_a_time = super::ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("[skip] no cuda device");
            return;
        };
        let real = Rope::new(
            RopeConfig {
                head_dim: 16,
                max_seq_len: 32,
                base: 10_000_000.0,
                kind: RopeKind::Standard,
            },
            &device,
        )
        .expect("cuda rope");
        let p: Vec<u32> = vec![3, 7, 1, 0, 9, 4, 2, 8];
        let mrope = nv_models::qwen3_mm_splice::Qwen3MropePositions {
            t: p.clone(),
            h: p.clone(),
            w: p.clone(),
            delta_added_to_token_index_for_every_position_after_this_prefill: 0,
        };
        let synth = nv_models::qwen3_mm_splice::mrope_rope_one_row_per_token(
            &real,
            &mrope,
            TINY_SECTION_3_3_2_TILES_THE_8_HALF_FREQS_OF_ROTARY_DIM_16,
            &device,
        )
        .expect("synth rope");
        let mut r = Lcg::new(0x5eed);
        let q = Tensor::from_vec(r.vec(8 * 2 * 16, 0.7), (1usize, 8, 2, 16), &device).unwrap();
        let k = Tensor::from_vec(r.vec(8 * 2 * 16, 0.7), (1usize, 8, 2, 16), &device).unwrap();
        let pos_real = Tensor::from_vec(p.clone(), 8usize, &device).unwrap();
        let pos_iota = Tensor::from_vec((0u32..8).collect::<Vec<_>>(), 8usize, &device).unwrap();
        let (qa, ka) = real.apply(&q, &k, &pos_real).expect("real apply");
        let (qb, kb) = synth.apply(&q, &k, &pos_iota).expect("synth apply");
        for (name, a, b) in [("q", &qa, &qb), ("k", &ka, &kb)] {
            let va: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
            let vb: Vec<f32> = b.flatten_all().unwrap().to_vec1().unwrap();
            let diff = va
                .iter()
                .zip(vb.iter())
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            assert_eq!(
                diff, 0,
                "{name}: synthetic per-token table + iota must equal the real table + true \
                 positions bitwise ({diff}/{} differ)",
                va.len()
            );
        }
        let p2: Vec<u32> = p.iter().map(|v| v + 2).collect();
        let mrope2 = nv_models::qwen3_mm_splice::Qwen3MropePositions {
            t: p2.clone(),
            h: p2.clone(),
            w: p2,
            delta_added_to_token_index_for_every_position_after_this_prefill: 0,
        };
        let synth2 = nv_models::qwen3_mm_splice::mrope_rope_one_row_per_token(
            &real,
            &mrope2,
            TINY_SECTION_3_3_2_TILES_THE_8_HALF_FREQS_OF_ROTARY_DIM_16,
            &device,
        )
        .expect("shifted synth rope");
        let (qc, _kc) = synth2.apply(&q, &k, &pos_iota).expect("shifted apply");
        let va: Vec<f32> = qa.flatten_all().unwrap().to_vec1().unwrap();
        let vc: Vec<f32> = qc.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            va.iter().zip(vc.iter()).any(|(x, y)| x.to_bits() != y.to_bits()),
            "shifted positions must change the rotation, or this parity proves nothing"
        );
    }

    #[test]
    fn text_path_is_sensitive_to_a_mid_sequence_token_change_or_every_splice_assert_is_vacuous() {
        let _one_gpu_test_at_a_time = super::ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("[skip] no cuda device");
            return;
        };
        let cfg = tiny_cfg();
        let dir = write_tiny_dense_checkpoint(&cfg, 0x9380_27b0_5111);
        let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("open tiny dir");
        let model = Qwen3Moe::from_loader_dense(cfg.clone(), &weights, &device).expect("build");
        drop(weights);
        let tokens: Vec<u32> = (0..16u32).map(|i| (i * 7 + 3) % 59).collect();
        let positions = Tensor::from_vec(
            (0..tokens.len() as u32).collect::<Vec<_>>(),
            tokens.len(),
            &device,
        )
        .unwrap();
        let run = |toks: &[u32]| -> Vec<u32> {
            let mut cache = model.new_kv_cache(48).unwrap();
            let t = Tensor::from_vec(toks.to_vec(), (1usize, toks.len()), &device).unwrap();
            let logits = model
                .forward_with_cache_serving_prefill_last_row_logits_because_chat_prefill_samples_only_position_seq_minus_1(
                    &t, &positions, &mut cache,
                )
                .expect("prefill");
            last_row_bits(&logits)
        };
        let base = run(&tokens);
        let mut changed = tokens.clone();
        changed[6] = (changed[6] + 17) % 59;
        let moved = run(&changed);
        let diff = base.iter().zip(moved.iter()).filter(|(a, b)| a != b).count();
        eprintln!("[q38-mm-tiny] token[6] change moves {diff}/{} logits", base.len());
        assert!(
            diff > 0,
            "a mid-sequence token change must reach the last-row logits; if it cannot, \
             the splice asserts in this suite prove nothing"
        );
    }

    #[test]
    fn spliced_prefill_with_degenerate_mrope_is_bit_identical_to_the_text_path() {
        let _one_gpu_test_at_a_time = super::ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("[skip] no cuda device for the tiny mm-splice prefill parity");
            return;
        };
        let cfg = tiny_cfg();
        let dir = write_tiny_dense_checkpoint(&cfg, 0x9380_27b0_5111);
        let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("open tiny dir");
        let model = Qwen3Moe::from_loader_dense(cfg.clone(), &weights, &device).expect("build");
        drop(weights);

        let tokens: Vec<u32> = (0..16u32).map(|i| (i * 7 + 3) % 59).collect();
        let t_dev = Tensor::from_vec(tokens.clone(), (1usize, tokens.len()), &device).unwrap();
        let positions = Tensor::from_vec(
            (0..tokens.len() as u32).collect::<Vec<_>>(),
            tokens.len(),
            &device,
        )
        .unwrap();

        let mut cache_a = model.new_kv_cache(48).unwrap();
        let logits_a = model
            .forward_with_cache_serving_prefill_last_row_logits_because_chat_prefill_samples_only_position_seq_minus_1(
                &t_dev, &positions, &mut cache_a,
            )
            .expect("text prefill");
        let bits_a = last_row_bits(&logits_a);

        let mrope =
            build_mrope_positions_matching_hf_get_rope_index(&tokens, TINY_IMG_TOKEN, &[]).unwrap();
        assert!(mrope.is_text_degenerate());
        let splice_rows = embed_rows_for(&model, &tokens[5..8], &device);
        let splices = [Qwen3ImageRowSplice {
            position: 5,
            rows: splice_rows.clone(),
        }];
        let mut cache_b = model.new_kv_cache(48).unwrap();
        let logits_b = model
            .forward_with_cache_prefill_image_rows_last_row_logits(
                &t_dev,
                &splices,
                &mrope,
                TINY_SECTION_3_3_2_TILES_THE_8_HALF_FREQS_OF_ROTARY_DIM_16,
                &mut cache_b,
            )
            .expect("spliced prefill");
        let bits_b = last_row_bits(&logits_b);
        assert_eq!(bits_a.len(), bits_b.len());
        for (i, (a, b)) in bits_a.iter().zip(bits_b.iter()).enumerate() {
            assert_eq!(
                a, b,
                "logit {i}: identity splice rows + degenerate mrope must be bit-identical \
                 to the text prefill"
            );
        }

        let perturbed = splice_rows
            .to_dtype(DType::F32)
            .unwrap()
            .affine(1.0, 0.5)
            .unwrap()
            .to_dtype(splice_rows.dtype())
            .unwrap();
        let mut cache_c = model.new_kv_cache(48).unwrap();
        let logits_c = model
            .forward_with_cache_prefill_image_rows_last_row_logits(
                &t_dev,
                &[Qwen3ImageRowSplice {
                    position: 5,
                    rows: perturbed,
                }],
                &mrope,
                TINY_SECTION_3_3_2_TILES_THE_8_HALF_FREQS_OF_ROTARY_DIM_16,
                &mut cache_c,
            )
            .expect("perturbed spliced prefill");
        let bits_c = last_row_bits(&logits_c);
        assert!(
            bits_a.iter().zip(bits_c.iter()).any(|(a, c)| a != c),
            "perturbed splice rows must change the logits"
        );
    }

    #[test]
    fn a_real_image_grid_moves_the_logits_through_rope_alone() {
        let _one_gpu_test_at_a_time = super::ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
        let Ok(device) = Device::new_cuda(0) else {
            eprintln!("[skip] no cuda device for the tiny mm mrope divergence test");
            return;
        };
        let cfg = tiny_cfg();
        let dir = write_tiny_dense_checkpoint(&cfg, 0x9380_27b0_5222);
        let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("open tiny dir");
        let model = Qwen3Moe::from_loader_dense(cfg.clone(), &weights, &device).expect("build");
        drop(weights);

        let mut tokens: Vec<u32> = (0..16u32).map(|i| (i * 5 + 2) % 59).collect();
        for slot in tokens.iter_mut().take(11).skip(5) {
            *slot = TINY_IMG_TOKEN;
        }
        let t_dev = Tensor::from_vec(tokens.clone(), (1usize, tokens.len()), &device).unwrap();
        let splice_rows = embed_rows_for(&model, &tokens[5..11], &device);
        let splices = [Qwen3ImageRowSplice {
            position: 5,
            rows: splice_rows,
        }];

        let degenerate =
            build_mrope_positions_matching_hf_get_rope_index(&tokens, TINY_IMG_TOKEN + 1, &[])
                .unwrap();
        let grid = build_mrope_positions_matching_hf_get_rope_index(
            &tokens,
            TINY_IMG_TOKEN,
            &[(1, 2, 3)],
        )
        .unwrap();
        assert!(!grid.is_text_degenerate());
        assert_eq!(
            grid.delta_added_to_token_index_for_every_position_after_this_prefill,
            -3
        );

        let full_attn_layer = cfg
            .layer_types
            .iter()
            .position(|t| matches!(t, LayerType::FullAttention))
            .expect("tiny config keeps a full-attention layer");
        let run = |mrope: &nv_models::qwen3_mm_splice::Qwen3MropePositions| -> (Vec<u16>, Vec<u16>) {
            let mut cache = model.new_kv_cache(48).unwrap();
            model
                .forward_with_cache_prefill_image_rows(
                    &t_dev,
                    &splices,
                    mrope,
                    TINY_SECTION_3_3_2_TILES_THE_8_HALF_FREQS_OF_ROTARY_DIM_16,
                    &mut cache,
                    Some(1),
                )
                .expect("spliced prefill");
            let slot = cache.full_slot_for_layer(full_attn_layer).expect("slot");
            let (k, v) = cache.view(slot, tokens.len()).expect("kv view");
            let flat = |t: &Tensor| -> Vec<u16> {
                t.to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .map(|x| half::bf16::from_f32(*x).to_bits())
                    .collect()
            };
            (flat(&k), flat(&v))
        };
        let (k_deg, v_deg) = run(&degenerate);
        let (k_grid, v_grid) = run(&grid);
        assert_eq!(v_deg, v_grid, "V carries no rope; the values path must be untouched");
        let per_row = k_deg.len() / tokens.len();
        let differing_rows: Vec<usize> = (0..tokens.len())
            .filter(|r| {
                k_deg[r * per_row..(r + 1) * per_row] != k_grid[r * per_row..(r + 1) * per_row]
            })
            .collect();
        eprintln!(
            "[q38-mm-tiny] grid-vs-degenerate differing roped-K cache rows: {differing_rows:?}"
        );
        assert!(
            differing_rows.iter().all(|r| *r >= 5),
            "K rows before the image see identical positions and must be identical, got \
             {differing_rows:?}"
        );
        assert!(
            differing_rows.len() >= 4,
            "the 3D grid rotates the image-run keys away from the text-degenerate keys; \
             an empty diff means the interleaved mrope never reached the attention layers"
        );
    }
}
