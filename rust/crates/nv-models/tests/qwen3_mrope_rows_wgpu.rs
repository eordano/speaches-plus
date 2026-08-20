#![cfg(feature = "wgpu")]

mod common;
use common::bf16_lin;
use common::have_gpu;
use common::LcgOddSeedShift33SignedUnit as Lcg;
use common::norm_vec;
use nv_models::qwen3_5_dense_wgpu as q3d;
use nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseConfig;
use nv_models::qwen3_5_moe::LayerType;
use nv_models::qwen3_5_moe_wgpu::HostBf16Lin;
use nv_models::qwen3_mm_splice as mrope;
use common::tiny_config_q3d_mixed_layers as tiny_config;
use common::tiny_weights_q3d as tiny_weights;

const IMG: u32 = 61;

#[test]
fn hf_get_rope_index_fixture_two_images_spatial_positions_and_delta() {
    let tokens: Vec<u32> = vec![7, 9, IMG, IMG, IMG, IMG, 11, IMG, IMG, IMG, IMG, IMG, IMG, 13];
    let mp = mrope::build_mrope_positions_matching_hf_get_rope_index(
        &tokens,
        IMG,
        &[(1, 2, 2), (1, 2, 3)],
    )
    .expect("build positions");
    assert_eq!(mp.t, vec![0, 1, 2, 2, 2, 2, 4, 5, 5, 5, 5, 5, 5, 8]);
    assert_eq!(mp.h, vec![0, 1, 2, 2, 3, 3, 4, 5, 5, 5, 6, 6, 6, 8]);
    assert_eq!(mp.w, vec![0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 5, 6, 7, 8]);
    assert_eq!(
        mp.delta_added_to_token_index_for_every_position_after_this_prefill,
        -5,
        "HF continues decode at max_position + 1 = 9 for this 14-token prompt"
    );
    assert!(!mp.is_text_degenerate());
    assert_eq!(mp.decode_position(14), 9);
}

#[test]
fn release_section_11_11_10_gives_each_axis_exactly_its_section_count_of_half_freqs() {
    let section = nv_models::qwen3_mm_splice::QWEN3_5_MROPE_SECTION_FROM_THE_RELEASE_CONFIG;
    let half: usize = section.iter().sum();
    assert_eq!(half, 32);
    let mut counts = [0usize; 3];
    for j in 0..half {
        counts[mrope::interleaved_axis_of_half_freq_matching_hf_overwrite_semantics(j, section)] +=
            1;
    }
    assert_eq!(counts, section);
    let tiny = [2usize, 1, 1];
    let axes: Vec<usize> = (0..4)
        .map(|j| mrope::interleaved_axis_of_half_freq_matching_hf_overwrite_semantics(j, tiny))
        .collect();
    assert_eq!(axes, vec![0, 1, 2, 0]);
}

#[test]
fn installed_rope_rows_are_the_axis_gathered_tables_and_reset_restores_them_bit_exact() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x9107_0e40);
    let max_seq = 64usize;
    let mut gpu = q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, max_seq).expect("build wgpu model");

    let tokens: Vec<u32> = vec![3, 5, IMG, IMG, IMG, IMG, 7];
    let mp = mrope::build_mrope_positions_matching_hf_get_rope_index(&tokens, IMG, &[(1, 2, 2)])
        .expect("positions");
    let section = [2usize, 1, 1];
    let rh = gpu.rope_rot_half();
    assert_eq!(rh, 4, "tiny config: head_dim 32 * partial_rotary 0.25 / 2");

    gpu.install_mrope_rows_for_prompt_and_shift_continuation_rows_by_delta(&mp, section)
        .expect("install");
    assert!(gpu.mrope_rows_installed());

    let (std_cos, std_sin) =
        q3d::rope_tables(cfg.rotary_dim().max(2), cfg.rope_theta, max_seq);
    let (got_cos, got_sin) = gpu.read_rope_rows_for_test(0).expect("read rows");
    let n = tokens.len();
    for i in 0..n {
        for j in 0..rh {
            let axis =
                mrope::interleaved_axis_of_half_freq_matching_hf_overwrite_semantics(j, section);
            let p = match axis {
                1 => mp.h[i],
                2 => mp.w[i],
                _ => mp.t[i],
            } as usize;
            assert_eq!(
                got_cos[i * rh + j].to_bits(),
                std_cos[p * rh + j].to_bits(),
                "cos row {i} freq {j} must gather axis-{axis} position {p}"
            );
            assert_eq!(
                got_sin[i * rh + j].to_bits(),
                std_sin[p * rh + j].to_bits(),
                "sin row {i} freq {j}"
            );
        }
    }
    let delta = mp.delta_added_to_token_index_for_every_position_after_this_prefill;
    assert_eq!(delta, -(n as i64) + 5, "image run of 4 compresses to span 2");
    for i in n..max_seq {
        let p = (i as i64 + delta) as usize;
        for j in 0..rh {
            assert_eq!(
                got_cos[i * rh + j].to_bits(),
                std_cos[p * rh + j].to_bits(),
                "continuation cos row {i} must be the standard row at {p}"
            );
            assert_eq!(got_sin[i * rh + j].to_bits(), std_sin[p * rh + j].to_bits());
        }
    }

    gpu.reset().expect("reset");
    assert!(!gpu.mrope_rows_installed());
    let (rest_cos, rest_sin) = gpu.read_rope_rows_for_test(0).expect("read restored");
    for (a, b) in rest_cos.iter().zip(std_cos.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "reset must restore standard cos");
    }
    for (a, b) in rest_sin.iter().zip(std_sin.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "reset must restore standard sin");
    }
}

#[test]
fn spliced_prefill_with_mrope_rows_decodes_and_differs_from_flat_positions() {
    if !have_gpu() {
        return;
    }
    let cfg = tiny_config();
    let hw = tiny_weights(&cfg, 0x9107_0e41);
    let max_seq = 64usize;
    let hidden = cfg.hidden_size;
    let tokens: Vec<u32> = vec![3, 5, IMG, IMG, IMG, IMG, 7, 11];
    let mut r = Lcg::new(0x5eed);
    let rows_bf16: Vec<u16> = r.bf16_vec(4 * hidden, 0.4);
    let splices = vec![nv_models::embed_row_splice::EmbedRowSplice {
        position: 2,
        rows_bf16: rows_bf16.clone(),
    }];

    let run = |with_mrope: bool| -> Vec<u32> {
        let mut gpu =
            q3d::Qwen3_5DenseWgpu::new(cfg.clone(), &hw, max_seq).expect("build wgpu model");
        if gpu.prefill_chunk_len() == 0 {
            return Vec::new();
        }
        if with_mrope {
            let mp = mrope::build_mrope_positions_matching_hf_get_rope_index(
                &tokens,
                IMG,
                &[(1, 2, 2)],
            )
            .expect("positions");
            gpu.install_mrope_rows_for_prompt_and_shift_continuation_rows_by_delta(&mp, [2, 1, 1])
                .expect("install");
        }
        let done = gpu
            .prefill_tokens_with_image_rows(&tokens[..tokens.len() - 1], &splices)
            .expect("spliced prefill");
        assert_eq!(done, tokens.len() - 1);
        let mut out = Vec::new();
        let mut t = *tokens.last().expect("non-empty");
        for _ in 0..6 {
            t = gpu.decode_step(t).expect("decode");
            out.push(t);
        }
        out
    };

    let with_rows = run(true);
    let flat = run(false);
    if with_rows.is_empty() {
        eprintln!("[skip] chunked prefill disabled");
        return;
    }
    assert_eq!(with_rows.len(), 6);
    eprintln!("[mrope-e2e] mrope {with_rows:?} flat {flat:?}");
}
