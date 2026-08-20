use candle_core::{Device, Tensor};
use nv_models::gemma4_audio::{
    attended_key_range, chunked_local_valid_mask, Gemma4AudioAttention, Gemma4AudioConfig,
    Gemma4AudioEmbedder, Gemma4AudioEncoder, Gemma4MmAudioSection, GEMMA4_AUDIO_FRAME_LENGTH,
    GEMMA4_AUDIO_HOP_LENGTH, GEMMA4_AUDIO_MEL_BINS, GEMMA4_AUDIO_MS_PER_TOKEN,
    GEMMA4_AUDIO_SAMPLE_RATE, GEMMA4_AUDIO_SEQ_LENGTH,
};

fn tiny_cfg() -> Gemma4AudioConfig {
    Gemma4AudioConfig {
        model_type: None,
        attention_chunk_size: 4,
        attention_context_left: 3,
        attention_context_right: 1,
        attention_invalid_logits_value: -1.0e9,
        attention_logit_cap: 50.0,
        conv_kernel_size: 3,
        hidden_size: 16,
        num_attention_heads: 2,
        num_hidden_layers: 1,
        output_proj_dims: 12,
        residual_weight: 0.5,
        rms_norm_eps: 1e-6,
        subsampling_conv_channels: vec![2, 3],
        hidden_act: "silu".to_string(),
        use_clipped_linears: false,
    }
}

fn full_json_with_audio_config() -> &'static str {
    r#"{
        "audio_config": {
            "attention_chunk_size": 12,
            "attention_context_left": 13,
            "conv_kernel_size": 5,
            "hidden_size": 1536,
            "num_attention_heads": 8,
            "num_hidden_layers": 12,
            "output_proj_dims": 2048,
            "rms_norm_eps": 1e-6,
            "subsampling_conv_channels": [128, 32]
        },
        "audio_token_id": 262273,
        "boa_token_id": 256000,
        "eoa_token_id": 256001
    }"#
}

fn det_input(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.7311).sin() + (i as f32 * 0.1913).cos()) * scale)
        .collect()
}

fn max_abs_row_diff(a: &[Vec<f32>], b: &[Vec<f32>], rows: std::ops::Range<usize>) -> f32 {
    let mut m = 0.0f32;
    for r in rows {
        for (x, y) in a[r].iter().zip(b[r].iter()) {
            m = m.max((x - y).abs());
        }
    }
    m
}

#[test]
fn mm_section_parses_tower_and_all_three_token_ids_from_full_hf_json() {
    let s = Gemma4MmAudioSection::from_full_hf_json_str(full_json_with_audio_config()).unwrap();
    let cfg = s.tower.expect("audio_config present must yield Some tower");
    assert_eq!(cfg.attention_chunk_size, 12);
    assert_eq!(cfg.attention_context_left, 13);
    assert_eq!(cfg.hidden_size, 1536);
    assert_eq!(cfg.subsampling_conv_channels, vec![128, 32]);
    assert_eq!(s.audio_token_id, Some(262273));
    assert_eq!(s.boa_token_id, Some(256000));
    assert_eq!(s.eoa_token_id, Some(256001));
}

#[test]
fn serde_defaults_pin_neg1e9_invalid_logit_cap50_residual_half_silu_right0_clip_off() {
    let s = Gemma4MmAudioSection::from_full_hf_json_str(full_json_with_audio_config()).unwrap();
    let cfg = s.tower.unwrap();
    assert_eq!(
        cfg.attention_invalid_logits_value, -1.0e9,
        "default invalid-logit must stay -1e9 or masked softmax stops underflowing to zero"
    );
    assert_eq!(cfg.attention_logit_cap, 50.0);
    assert_eq!(cfg.residual_weight, 0.5);
    assert_eq!(cfg.hidden_act, "silu");
    assert_eq!(cfg.attention_context_right, 0);
    assert!(!cfg.use_clipped_linears);
    assert!(cfg.model_type.is_none());
}

#[test]
fn mm_section_absent_or_null_audio_config_yields_none_tower_but_ids_still_parse() {
    let s = Gemma4MmAudioSection::from_full_hf_json_str(r#"{"audio_token_id": 7}"#).unwrap();
    assert!(s.tower.is_none());
    assert_eq!(s.audio_token_id, Some(7));
    let s2 =
        Gemma4MmAudioSection::from_full_hf_json_str(r#"{"audio_config": null, "boa_token_id": 9}"#)
            .unwrap();
    assert!(s2.tower.is_none());
    assert_eq!(s2.boa_token_id, Some(9));
    assert!(s2.audio_token_id.is_none());
}

#[test]
fn mm_section_untrusted_json_errors_not_panics_on_malformed_nonobject_and_missing_fields() {
    assert!(Gemma4MmAudioSection::from_full_hf_json_str("{").is_err());
    assert!(
        Gemma4MmAudioSection::from_full_hf_json_str("[1,2]").is_err(),
        "non-object root must be a hard error, not a silent empty section"
    );
    assert!(
        Gemma4MmAudioSection::from_full_hf_json_str(r#"{"audio_config": {"hidden_size": 4}}"#)
            .is_err(),
        "audio_config missing required fields must fail deserialization"
    );
}

#[test]
fn mm_section_non_numeric_token_id_degrades_to_none_instead_of_erroring() {
    let s =
        Gemma4MmAudioSection::from_full_hf_json_str(r#"{"audio_token_id": "oops"}"#).unwrap();
    assert!(s.audio_token_id.is_none());
}

#[test]
fn derived_dims_head192_context24_stride4_freq32_inputdim_last_channel_times_32() {
    let s = Gemma4MmAudioSection::from_full_hf_json_str(full_json_with_audio_config()).unwrap();
    let cfg = s.tower.unwrap();
    assert_eq!(cfg.head_dim(), 192);
    assert_eq!(cfg.past_horizon(), 12);
    assert_eq!(cfg.context_size(), 24);
    assert_eq!(cfg.time_stride(), 4);
    assert_eq!(
        cfg.subsampled_freq(),
        32,
        "two valid-pad k3 s2 convs must take 128 mel bins to 64 then 32"
    );
    assert_eq!(cfg.subsample_input_dim(), 32 * 32);
}

#[test]
fn past_horizon_saturates_to_zero_when_context_left_is_zero_instead_of_underflowing() {
    let mut cfg = tiny_cfg();
    cfg.attention_context_left = 0;
    assert_eq!(cfg.past_horizon(), 0);
    assert_eq!(cfg.context_size(), cfg.attention_chunk_size + cfg.attention_context_right);
}

#[test]
fn subsampled_seq_len_matches_padded_k3_s2_twice_with_zero_guard() {
    let cfg = tiny_cfg();
    for (mel, expect) in [(0usize, 0usize), (1, 1), (2, 1), (9, 3), (10, 3), (750, 188)] {
        assert_eq!(
            cfg.subsampled_seq_len(mel),
            expect,
            "mel_frames {mel} must subsample to {expect}"
        );
    }
}

#[test]
fn frame_constants_cross_check_frame_is_two_hops_hop_is_10ms_token_is_40ms_seq_caps_30s() {
    assert_eq!(GEMMA4_AUDIO_FRAME_LENGTH, 2 * GEMMA4_AUDIO_HOP_LENGTH);
    assert_eq!(
        GEMMA4_AUDIO_HOP_LENGTH * 1000 / GEMMA4_AUDIO_SAMPLE_RATE,
        10,
        "hop must stay 10ms so time_stride x hop gives the documented ms-per-token"
    );
    assert_eq!(
        tiny_cfg().time_stride() * (GEMMA4_AUDIO_HOP_LENGTH * 1000 / GEMMA4_AUDIO_SAMPLE_RATE),
        GEMMA4_AUDIO_MS_PER_TOKEN
    );
    assert_eq!(GEMMA4_AUDIO_SEQ_LENGTH * GEMMA4_AUDIO_MS_PER_TOKEN, 30_000);
    assert_eq!(GEMMA4_AUDIO_MEL_BINS, 128);
}

#[test]
fn chunked_mask_agrees_with_attended_key_range_mapped_into_window_coordinates() {
    for (chunk, left, right) in [(4usize, 3usize, 1usize), (4, 1, 0), (3, 5, 2), (2, 2, 0)] {
        let past = left.saturating_sub(1);
        let mask = chunked_local_valid_mask(chunk, left, right);
        assert_eq!(mask.len(), chunk);
        let window_start = chunk * (past + 1);
        for (w, row) in mask.iter().enumerate() {
            assert_eq!(row.len(), chunk + past + right);
            let q_abs = window_start + w;
            let (lo, hi) = attended_key_range(q_abs, left, right);
            for (c, &bit) in row.iter().enumerate() {
                let key_abs = window_start + c - past;
                let expect = key_abs >= lo && key_abs <= hi;
                assert_eq!(
                    bit, expect,
                    "chunk={chunk} left={left} right={right} w={w} c={c}: window mask and attended_key_range must describe the same locality"
                );
            }
        }
    }
}

#[test]
fn chunked_mask_negative_control_left1_right0_is_exact_diagonal_not_all_true() {
    let mask = chunked_local_valid_mask(3, 1, 0);
    let mut trues = 0;
    let mut falses = 0;
    for (w, row) in mask.iter().enumerate() {
        for (c, &bit) in row.iter().enumerate() {
            assert_eq!(bit, c == w, "left=1 right=0 must attend self only");
            if bit {
                trues += 1;
            } else {
                falses += 1;
            }
        }
    }
    assert!(trues > 0 && falses > 0, "mask gate is vacuous if it never says false");
}

#[test]
fn attended_key_range_saturates_at_query_zero_and_handles_left_zero() {
    assert_eq!(attended_key_range(0, 5, 0), (0, 0));
    assert_eq!(attended_key_range(2, 5, 1), (0, 3));
    assert_eq!(attended_key_range(10, 3, 0), (8, 10));
    assert_eq!(attended_key_range(2, 0, 0), (2, 2));
}

#[test]
fn encoder_forward_shape_is_b_tsub_outdims_and_sub_lens_match_helper_and_hand_math() {
    let cfg = tiny_cfg();
    let dev = Device::Cpu;
    let enc = Gemma4AudioEncoder::synthetic(&cfg, &dev).unwrap();
    let (b, t_mel) = (2usize, 9usize);
    let mel = Tensor::from_vec(
        det_input(b * t_mel * GEMMA4_AUDIO_MEL_BINS, 0.5),
        (b, t_mel, GEMMA4_AUDIO_MEL_BINS),
        &dev,
    )
    .unwrap();
    let (out, sub_lens) = enc.forward(&mel, &[9, 4]).unwrap();
    assert_eq!(out.dims(), &[b, 3, cfg.output_proj_dims]);
    assert_eq!(sub_lens, vec![3, 1], "stride-4 mapping of valid lens 9 and 4 over t_sub 3");
    assert_eq!(sub_lens, enc.subsampled_valid_lens(t_mel, &[9, 4]));
}

#[test]
fn encoder_forward_zeroes_rows_at_and_beyond_sub_len_with_nonzero_valid_row_control() {
    let cfg = tiny_cfg();
    let dev = Device::Cpu;
    let enc = Gemma4AudioEncoder::synthetic(&cfg, &dev).unwrap();
    let (b, t_mel) = (2usize, 9usize);
    let mel = Tensor::from_vec(
        det_input(b * t_mel * GEMMA4_AUDIO_MEL_BINS, 0.5),
        (b, t_mel, GEMMA4_AUDIO_MEL_BINS),
        &dev,
    )
    .unwrap();
    let (out, sub_lens) = enc.forward(&mel, &[9, 4]).unwrap();
    let v = out.to_vec3::<f32>().unwrap();
    assert_eq!(sub_lens[1], 1);
    for t in 1..3 {
        for &x in &v[1][t] {
            assert_eq!(x, 0.0, "row {t} past sub_len must be zeroed by the valid-column mask");
        }
    }
    let valid_energy: f32 = v[1][0].iter().map(|x| x.abs()).sum();
    assert!(
        valid_energy > 1e-6,
        "negative control: the zeroing gate is vacuous if valid rows are zero too"
    );
}

#[test]
fn encoder_forward_untrusted_shapes_error_not_panic_on_mel_bins_and_batch_mismatch() {
    let cfg = tiny_cfg();
    let dev = Device::Cpu;
    let enc = Gemma4AudioEncoder::synthetic(&cfg, &dev).unwrap();
    let bad_bins = Tensor::zeros((1usize, 9usize, 64usize), candle_core::DType::F32, &dev).unwrap();
    assert!(enc.forward(&bad_bins, &[9]).is_err(), "64 mel bins must be rejected");
    let ok_bins =
        Tensor::zeros((2usize, 9usize, GEMMA4_AUDIO_MEL_BINS), candle_core::DType::F32, &dev)
            .unwrap();
    assert!(
        enc.forward(&ok_bins, &[9]).is_err(),
        "valid_lens shorter than batch must be rejected"
    );
}

#[test]
fn encoder_synthetic_rejects_config_without_exactly_two_subsampling_channels() {
    let mut cfg = tiny_cfg();
    cfg.subsampling_conv_channels = vec![4];
    assert!(Gemma4AudioEncoder::synthetic(&cfg, &Device::Cpu).is_err());
}

#[test]
fn subsampled_valid_lens_zero_gives_zero_and_full_gives_t_sub() {
    let cfg = tiny_cfg();
    let dev = Device::Cpu;
    let enc = Gemma4AudioEncoder::synthetic(&cfg, &dev).unwrap();
    assert_eq!(enc.subsampled_valid_lens(9, &[0]), vec![0]);
    assert_eq!(enc.subsampled_valid_lens(9, &[9]), vec![cfg.subsampled_seq_len(9)]);
    assert_eq!(enc.subsampled_valid_lens(10, &[10, 0, 5]), vec![3, 0, 2]);
}

#[test]
fn attention_valid_queries_are_bit_isolated_from_positions_masked_invalid() {
    let cfg = tiny_cfg();
    let dev = Device::Cpu;
    let mut seed = 42u64;
    let attn = Gemma4AudioAttention::synthetic(&cfg, &mut seed, &dev).unwrap();
    let t = 8usize;
    let base = det_input(t * cfg.hidden_size, 0.4);
    let x = Tensor::from_vec(base.clone(), (1, t, cfg.hidden_size), &dev).unwrap();
    let mut perturbed = base.clone();
    for f in 0..cfg.hidden_size {
        perturbed[6 * cfg.hidden_size + f] += 3.0;
    }
    let x2 = Tensor::from_vec(perturbed, (1, t, cfg.hidden_size), &dev).unwrap();
    let valid_data: Vec<f32> = (0..t).map(|i| if i < 5 { 1.0 } else { 0.0 }).collect();
    let valid = Tensor::from_vec(valid_data, (1, t), &dev).unwrap();
    let y1 = attn.forward(&x, &valid).unwrap().to_vec3::<f32>().unwrap();
    let y2 = attn.forward(&x2, &valid).unwrap().to_vec3::<f32>().unwrap();
    assert!(
        max_abs_row_diff(&y1[0], &y2[0], 0..5) < 1e-5,
        "perturbing an invalid position leaked into valid queries: invalid-logit masking must drive softmax weight to exactly zero"
    );
    let mut perturbed_valid = base;
    for f in 0..cfg.hidden_size {
        perturbed_valid[4 * cfg.hidden_size + f] += 3.0;
    }
    let x3 = Tensor::from_vec(perturbed_valid, (1, t, cfg.hidden_size), &dev).unwrap();
    let y3 = attn.forward(&x3, &valid).unwrap().to_vec3::<f32>().unwrap();
    assert!(
        max_abs_row_diff(&y1[0], &y3[0], 3..4) > 1e-4,
        "negative control: perturbing an attended valid key must change its neighbors or the isolation gate is vacuous"
    );
}

#[test]
fn attention_local_window_no_leak_from_keys_beyond_context_right() {
    let cfg = tiny_cfg();
    let dev = Device::Cpu;
    let mut seed = 7u64;
    let attn = Gemma4AudioAttention::synthetic(&cfg, &mut seed, &dev).unwrap();
    let t = 8usize;
    let base = det_input(t * cfg.hidden_size, 0.4);
    let x = Tensor::from_vec(base.clone(), (1, t, cfg.hidden_size), &dev).unwrap();
    let valid = Tensor::ones((1, t), candle_core::DType::F32, &dev).unwrap();
    let y1 = attn.forward(&x, &valid).unwrap().to_vec3::<f32>().unwrap();
    let mut far = base.clone();
    for f in 0..cfg.hidden_size {
        far[2 * cfg.hidden_size + f] += 3.0;
    }
    let x_far = Tensor::from_vec(far, (1, t, cfg.hidden_size), &dev).unwrap();
    let y_far = attn.forward(&x_far, &valid).unwrap().to_vec3::<f32>().unwrap();
    assert!(
        max_abs_row_diff(&y1[0], &y_far[0], 0..1) < 1e-5,
        "query 0 with context_right 1 attends keys 0..=1 only; key 2 leaking means the window mask upper bound broke"
    );
    let mut near = base;
    for f in 0..cfg.hidden_size {
        near[cfg.hidden_size + f] += 3.0;
    }
    let x_near = Tensor::from_vec(near, (1, t, cfg.hidden_size), &dev).unwrap();
    let y_near = attn.forward(&x_near, &valid).unwrap().to_vec3::<f32>().unwrap();
    assert!(
        max_abs_row_diff(&y1[0], &y_near[0], 0..1) > 1e-4,
        "negative control: key 1 is inside query 0 future horizon and must influence it"
    );
}

#[test]
fn embedder_maps_last_dim_to_text_hidden_preserving_leading_dims() {
    let dev = Device::Cpu;
    let emb = Gemma4AudioEmbedder::synthetic(12, 20, 1e-6, &dev).unwrap();
    let x = Tensor::from_vec(det_input(2 * 3 * 12, 1.0), (2, 3, 12), &dev).unwrap();
    let y = emb.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 3, 20]);
    assert_eq!(emb.text_hidden, 20);
}

#[test]
fn embedder_rms_norm_makes_projection_scale_invariant_within_eps_tolerance() {
    let dev = Device::Cpu;
    let emb = Gemma4AudioEmbedder::synthetic(12, 20, 1e-6, &dev).unwrap();
    let base = det_input(12, 1.0);
    let x = Tensor::from_vec(base.clone(), (1, 12), &dev).unwrap();
    let scaled: Vec<f32> = base.iter().map(|v| v * 10.0).collect();
    let xs = Tensor::from_vec(scaled, (1, 12), &dev).unwrap();
    let y = emb.forward(&x).unwrap().to_vec2::<f32>().unwrap();
    let ys = emb.forward(&xs).unwrap().to_vec2::<f32>().unwrap();
    for (a, b) in y[0].iter().zip(ys[0].iter()) {
        assert!(
            (a - b).abs() < 1e-3,
            "rms pre-norm must make the audio embedding invariant to input gain: {a} vs {b}"
        );
    }
    let mut other_dir: Vec<f32> = base.iter().rev().cloned().collect();
    other_dir[0] += 1.5;
    let other = Tensor::from_vec(other_dir, (1, 12), &dev).unwrap();
    let yo = emb.forward(&other).unwrap().to_vec2::<f32>().unwrap();
    let diff: f32 = y[0].iter().zip(yo[0].iter()).map(|(a, b)| (a - b).abs()).sum();
    assert!(diff > 1e-3, "negative control: different directions must embed differently");
}

#[test]
fn embedder_rejects_wrong_last_dim_with_error_not_panic() {
    let dev = Device::Cpu;
    let emb = Gemma4AudioEmbedder::synthetic(12, 20, 1e-6, &dev).unwrap();
    let bad = Tensor::zeros((1usize, 11usize), candle_core::DType::F32, &dev).unwrap();
    assert!(emb.forward(&bad).is_err());
}
