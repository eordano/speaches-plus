use candle_core::{DType, Device, Tensor};
use nv_models::gemma4_vision::{Gemma4VisionConfig, Gemma4VisionTower, VisionRopeParams};

const VALID_NESTED_JSON: &str = r#"{
  "vision_config": {
    "model_type": "gemma4_vision",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 8,
    "patch_size": 4,
    "pooling_kernel_size": 2,
    "position_embedding_size": 16,
    "default_output_length": 16,
    "rms_norm_eps": 1e-6,
    "hidden_activation": "gelu_pytorch_tanh"
  },
  "vision_soft_tokens_per_image": 16
}"#;

fn nested_with(patch: &[(&str, serde_json::Value)]) -> String {
    let mut v: serde_json::Value = serde_json::from_str(VALID_NESTED_JSON).unwrap();
    let obj = v["vision_config"].as_object_mut().unwrap();
    for (k, val) in patch {
        obj.insert((*k).to_string(), val.clone());
    }
    v.to_string()
}

fn parsed() -> Gemma4VisionConfig {
    Gemma4VisionConfig::from_hf_json_str(VALID_NESTED_JSON).unwrap()
}

#[test]
fn nested_vision_config_parses_and_root_level_soft_token_budget_is_hoisted_into_cfg() {
    let cfg = parsed();
    assert_eq!(cfg.hidden_size, 32, "hidden_size not read from nested vision_config");
    assert_eq!(
        cfg.vision_soft_tokens_per_image,
        Some(16),
        "vision_soft_tokens_per_image lives at the ROOT of hf json and must be hoisted into the nested vision_config before deserialize"
    );
}

#[test]
fn flat_config_without_vision_config_wrapper_also_parses() {
    let v: serde_json::Value = serde_json::from_str(VALID_NESTED_JSON).unwrap();
    let flat = v["vision_config"].to_string();
    let cfg = Gemma4VisionConfig::from_hf_json_str(&flat).unwrap();
    assert_eq!(cfg.patch_size, 4, "flat (non-wrapped) config path lost fields");
    assert_eq!(
        cfg.vision_soft_tokens_per_image, None,
        "flat path has no root-level soft-token field to hoist"
    );
}

#[test]
fn rope_theta_defaults_to_100_when_absent_and_explicit_value_wins() {
    assert_eq!(
        parsed().rope_theta(),
        100.0,
        "missing rope_parameters must default to theta=100, the gemma4 vision constant"
    );
    let s = nested_with(&[(
        "rope_parameters",
        serde_json::json!({"rope_theta": 250.0}),
    )]);
    let cfg = Gemma4VisionConfig::from_hf_json_str(&s).unwrap();
    assert_eq!(cfg.rope_theta(), 250.0, "explicit rope_theta ignored");
}

#[test]
fn config_refusals_are_errors_not_panics_for_each_malformed_input() {
    let e = Gemma4VisionConfig::from_hf_json_str("not json at all").unwrap_err();
    assert!(format!("{e:#}").contains("parse"), "garbage input: {e:#}");

    let e = Gemma4VisionConfig::from_hf_json_str("[1,2]").unwrap_err();
    assert!(format!("{e:#}").contains("not an object"), "non-object root: {e:#}");

    let e = Gemma4VisionConfig::from_hf_json_str(r#"{"vision_config": 5}"#).unwrap_err();
    assert!(
        format!("{e:#}").contains("must be an object"),
        "non-object vision_config: {e:#}"
    );

    let s = nested_with(&[("model_type", serde_json::json!("clip_vision"))]);
    let e = Gemma4VisionConfig::from_hf_json_str(&s).unwrap_err();
    assert!(
        format!("{e:#}").contains("gemma4_vision"),
        "wrong model_type must name the expected type: {e:#}"
    );

    let s = nested_with(&[("hidden_size", serde_json::json!(33))]);
    let e = Gemma4VisionConfig::from_hf_json_str(&s).unwrap_err();
    assert!(
        format!("{e:#}").contains("!="),
        "heads*head_dim != hidden must be refused: {e:#}"
    );

    let s = nested_with(&[
        ("head_dim", serde_json::json!(6)),
        ("hidden_size", serde_json::json!(24)),
    ]);
    let e = Gemma4VisionConfig::from_hf_json_str(&s).unwrap_err();
    assert!(
        format!("{e:#}").contains("divisible by 4"),
        "head_dim%4!=0 breaks 2d rope axis split and must be refused: {e:#}"
    );

    let mut v: serde_json::Value = serde_json::from_str(VALID_NESTED_JSON).unwrap();
    v["vision_config"].as_object_mut().unwrap().remove("patch_size");
    assert!(
        Gemma4VisionConfig::from_hf_json_str(&v.to_string()).is_err(),
        "missing required field patch_size must be a deserialize error"
    );
}

#[test]
fn negative_control_valid_config_with_correct_model_type_passes_every_gate() {
    let cfg = parsed();
    assert_eq!(cfg.model_type.as_deref(), Some("gemma4_vision"));
    assert_eq!(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size);
}

#[test]
fn patch_pixels_is_patchsq_times_rgb_and_unit_is_patch_times_pooling_kernel() {
    let cfg = parsed();
    assert_eq!(cfg.patch_pixels(), 4 * 4 * 3, "patch_pixels = patch^2 * 3 channels");
    assert_eq!(cfg.unit(), 4 * 2, "unit = patch_size * pooling_kernel_size");
}

#[test]
fn image_already_at_exact_patch_budget_keeps_its_resolution_and_fills_the_budget() {
    let cfg = parsed();
    assert_eq!(
        cfg.target_resolution(32, 32, None),
        (32, 32),
        "32x32 at patch=4 is exactly default_output_length*pk^2 patches, no resize needed"
    );
    assert_eq!(cfg.compute_num_soft_tokens(32, 32, None), 16);
}

#[test]
fn oversized_image_is_scaled_down_and_wider_input_yields_wider_target() {
    let cfg = parsed();
    assert_eq!(
        cfg.target_resolution(64, 64, None),
        (32, 32),
        "4x the pixel budget must scale by sqrt back into budget"
    );
    let (tw, th) = cfg.target_resolution(48, 32, None);
    assert_eq!((tw, th), (32, 24), "aspect ratio must survive: landscape stays tw>th");
    assert_eq!(cfg.compute_num_soft_tokens(48, 32, None), 12);
}

#[test]
fn degenerate_aspect_ratio_clamps_short_side_up_to_one_unit_never_zero() {
    let cfg = parsed();
    let (tw, th) = cfg.target_resolution(100, 4, Some(1));
    assert_eq!(
        th,
        cfg.unit(),
        "floor() would give height 0 for extreme aspect; the .max(unit) clamp must hold it at one unit"
    );
    assert_eq!(
        tw,
        cfg.unit(),
        "when the short-side clamp fires the long side must give way so the patch budget holds; \
         a 1-token budget is 4 patches = one 8x8 unit square"
    );
}

#[test]
fn unit_clamp_overshoot_shrinks_the_long_side_so_tower_rows_match_the_reserved_tokens() {
    let cfg = parsed();
    let (tw, th) = cfg.target_resolution(800, 8, Some(1));
    let patches = (tw / cfg.patch_size) * (th / cfg.patch_size);
    assert!(
        patches <= cfg.pooling_kernel_size * cfg.pooling_kernel_size,
        "800x8 with budget 1 must stay within 1 pooled token of patches ({tw}x{th} = {patches}); \
         otherwise the tower emits more soft tokens than the prompt reserves and the splice bails"
    );
    assert_eq!(cfg.compute_num_soft_tokens(800, 8, Some(1)), 1);
}

#[test]
fn target_resolution_sweep_always_lands_on_unit_grid_and_soft_tokens_stay_in_1_to_budget() {
    let cfg = parsed();
    let unit = cfg.unit();
    for &(w, h) in &[
        (1usize, 1usize),
        (7, 13),
        (640, 480),
        (3, 999),
        (4096, 17),
        (8, 8),
        (1, 150_000),
        (150_000, 1),
        (30_000, 200),
    ] {
        for &budget in &[None, Some(2usize), Some(16)] {
            let (tw, th) = cfg.target_resolution(w, h, budget);
            assert_eq!(tw % unit, 0, "tw {tw} for {w}x{h} not a unit multiple");
            assert_eq!(th % unit, 0, "th {th} for {w}x{h} not a unit multiple");
            assert!(tw >= unit && th >= unit, "target below one unit for {w}x{h}");
            let cap = budget.unwrap_or(cfg.default_output_length);
            let pk2 = cfg.pooling_kernel_size * cfg.pooling_kernel_size;
            let patches = (tw / cfg.patch_size) * (th / cfg.patch_size);
            assert!(
                patches <= cap * pk2,
                "{w}x{h} budget {cap}: {tw}x{th} carries {patches} patches, over the {} the \
                 prompt can reserve",
                cap * pk2
            );
            let n = cfg.compute_num_soft_tokens(w, h, budget);
            assert!(
                (1..=cap).contains(&n),
                "soft tokens {n} outside 1..={cap} for {w}x{h}"
            );
        }
    }
}

fn tiny_tower_cfg(layers: usize, pooling_kernel_size: usize, pes: usize) -> Gemma4VisionConfig {
    Gemma4VisionConfig {
        model_type: Some("gemma4_vision".into()),
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: layers,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 4,
        patch_size: 2,
        pooling_kernel_size,
        position_embedding_size: pes,
        default_output_length: 4,
        rms_norm_eps: 1e-6,
        attention_bias: false,
        hidden_activation: "gelu_pytorch_tanh".into(),
        use_clipped_linears: false,
        standardize: false,
        rope_parameters: VisionRopeParams { rope_theta: 100.0, rope_type: None },
        vision_soft_tokens_per_image: None,
    }
}

const TEXT_HIDDEN: usize = 6;

fn tower(layers: usize, pk: usize, pes: usize) -> Gemma4VisionTower {
    Gemma4VisionTower::new_synthetic(
        tiny_tower_cfg(layers, pk, pes),
        TEXT_HIDDEN,
        &Device::Cpu,
        DType::F32,
    )
    .unwrap()
}

fn pixel_rows(fills: &[f32]) -> Tensor {
    let pp = 2 * 2 * 3;
    let mut v = Vec::with_capacity(fills.len() * pp);
    for (i, &f) in fills.iter().enumerate() {
        for j in 0..pp {
            v.push(f + (i * pp + j) as f32 * 1e-3);
        }
    }
    Tensor::from_vec(v, (fills.len(), pp), &Device::Cpu).unwrap()
}

fn pos_tensor(xy: &[(i64, i64)]) -> Tensor {
    let mut v = Vec::with_capacity(xy.len() * 2);
    for &(x, y) in xy {
        v.push(x);
        v.push(y);
    }
    Tensor::from_vec(v, (xy.len(), 2), &Device::Cpu).unwrap()
}

#[test]
fn synthetic_tower_refuses_gqa_config_but_accepts_mha_negative_control() {
    let mut cfg = tiny_tower_cfg(0, 2, 4);
    cfg.num_key_value_heads = 1;
    let e = Gemma4VisionTower::new_synthetic(cfg, TEXT_HIDDEN, &Device::Cpu, DType::F32)
        .err()
        .expect("gqa config must not build a tower");
    assert!(
        format!("{e:#}").contains("MHA"),
        "kv_heads != heads must be refused before any weight is built: {e:#}"
    );
    let t = tower(0, 2, 4);
    assert_eq!(t.num_layers(), 0);
    assert_eq!(t.text_hidden_size(), TEXT_HIDDEN);
}

#[test]
fn forward_refuses_wrong_patch_dim_and_wrong_position_shape_as_errors_not_panics() {
    let t = tower(0, 2, 4);
    let bad_pp = Tensor::zeros((4, 11), DType::F32, &Device::Cpu).unwrap();
    let pos = pos_tensor(&[(0, 0), (1, 0), (0, 1), (1, 1)]);
    let e = t.forward(&bad_pp, &pos).unwrap_err();
    assert!(
        format!("{e:#}").contains("patch_pixels"),
        "pixel row width 11 != patch_pixels 12 must be refused: {e:#}"
    );

    let px = pixel_rows(&[0.1, 0.2, 0.3, 0.4]);
    let bad_pos = Tensor::zeros((4, 3), DType::I64, &Device::Cpu).unwrap();
    let e = t.forward(&px, &bad_pos).unwrap_err();
    assert!(
        format!("{e:#}").contains("pixel_position_ids"),
        "position ids must be [n,2]: {e:#}"
    );
}

#[test]
fn full_4x4_grid_through_one_attention_layer_pools_to_pk2_cells_of_text_hidden_width() {
    let t = tower(1, 2, 4);
    let mut pos = Vec::new();
    let mut fills = Vec::new();
    for y in 0..4i64 {
        for x in 0..4i64 {
            pos.push((x, y));
            fills.push(0.05 * (y * 4 + x) as f32);
        }
    }
    let out = t.forward(&pixel_rows(&fills), &pos_tensor(&pos)).unwrap();
    assert_eq!(
        out.dims(),
        &[4, TEXT_HIDDEN],
        "16 patches with pooling_kernel 2 must pool to a 2x2=4 cell grid projected to text width"
    );
    let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(vals.iter().all(|v| v.is_finite()), "non-finite values out of tiny tower");
}

#[test]
fn patches_marked_minus1_minus1_are_excluded_so_their_pooled_cell_disappears() {
    let t = tower(1, 2, 4);
    let mut pos = Vec::new();
    let mut fills = Vec::new();
    for y in 0..4i64 {
        for x in 0..4i64 {
            if x >= 2 && y >= 2 {
                pos.push((-1, -1));
            } else {
                pos.push((x, y));
            }
            fills.push(0.05 * (y * 4 + x) as f32);
        }
    }
    let out = t.forward(&pixel_rows(&fills), &pos_tensor(&pos)).unwrap();
    assert_eq!(
        out.dims(),
        &[3, TEXT_HIDDEN],
        "invalidating every patch of cell (1,1) must drop exactly that pooled row, not zero it"
    );
}

#[test]
fn all_patches_invalid_is_an_error_not_a_panic_or_empty_tensor() {
    let t = tower(0, 2, 4);
    let e = t
        .forward(&pixel_rows(&[0.1, 0.2]), &pos_tensor(&[(-1, -1), (-1, -1)]))
        .unwrap_err();
    assert!(
        format!("{e:#}").contains("no valid patches"),
        "all-invalid input must fail loudly: {e:#}"
    );
}

#[test]
fn pooling_is_a_mean_two_identical_patches_in_one_cell_equal_one_patch_bitwise() {
    let t = tower(0, 2, 1);
    let one = t
        .forward(&pixel_rows(&[0.25]), &pos_tensor(&[(2, 2)]))
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let pp = 12;
    let row: Vec<f32> = (0..pp).map(|j| 0.25 + j as f32 * 1e-3).collect();
    let mut two_rows = row.clone();
    two_rows.extend_from_slice(&row);
    let px2 = Tensor::from_vec(two_rows, (2, pp), &Device::Cpu).unwrap();
    let two = t
        .forward(&px2, &pos_tensor(&[(2, 2), (3, 3)]))
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(
        one, two,
        "pes=1 clamps both positions to the same embedding, so with mean pooling the duplicated patch changes nothing; a sum would double the cell"
    );
}

#[test]
fn pooled_rows_come_out_in_row_major_cell_order_by_position_not_by_input_order() {
    let t = tower(0, 1, 4);
    let ordered_pos = [(0i64, 0i64), (1, 0), (0, 1), (1, 1)];
    let fills = [0.1f32, 0.2, 0.3, 0.4];
    let baseline = t
        .forward(&pixel_rows(&fills), &pos_tensor(&ordered_pos))
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    assert_eq!(baseline.len(), 4, "pk=1 makes every patch its own pooled cell");

    let perm = [3usize, 1, 0, 2];
    let pp = 12;
    let base_px = pixel_rows(&fills).to_vec2::<f32>().unwrap();
    let mut scrambled_px = Vec::with_capacity(4 * pp);
    let mut scrambled_pos = Vec::with_capacity(4);
    for &i in &perm {
        scrambled_px.extend_from_slice(&base_px[i]);
        scrambled_pos.push(ordered_pos[i]);
    }
    let px = Tensor::from_vec(scrambled_px, (4, pp), &Device::Cpu).unwrap();
    let scrambled = t
        .forward(&px, &pos_tensor(&scrambled_pos))
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    assert_eq!(
        baseline, scrambled,
        "feeding the same patches in a different order must give bitwise-identical output: pooling keys on (y,x) cell, never on input index"
    );

    let swapped_pos = [(1i64, 1i64), (1, 0), (0, 1), (0, 0)];
    let swapped = t
        .forward(&pixel_rows(&fills), &pos_tensor(&swapped_pos))
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();
    assert_ne!(
        baseline, swapped,
        "negative control: moving patches to different positions must change the output, otherwise the ordering assertion above is vacuous"
    );
}

#[test]
fn synthetic_tower_builds_layers_with_clips_disabled() {
    let t = tower(2, 2, 4);
    assert_eq!(t.num_layers(), 2);
    for idx in 0..2 {
        assert!(
            t.layer_clips(idx).iter().all(|c| c.is_none()),
            "new_synthetic must produce plain (unclipped) linears"
        );
    }
    assert_eq!(t.config().patch_pixels(), 12);
}

#[test]
fn full_grid_fast_path_matches_the_irregular_path_on_the_same_patches() {
    let t = tower(2, 2, 6);
    let gw = 4usize;
    let gh = 6usize;
    let mut pos = Vec::new();
    let mut fills = Vec::new();
    for y in 0..gh as i64 {
        for x in 0..gw as i64 {
            pos.push((x, y));
            fills.push(0.03 * (y * gw as i64 + x) as f32 - 0.2);
        }
    }
    let fast = t
        .forward(&pixel_rows(&fills), &pos_tensor(&pos))
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    let mut order: Vec<usize> = (0..pos.len()).collect();
    order.reverse();
    order.swap(0, 7);
    let shuffled_pos: Vec<(i64, i64)> = order.iter().map(|&i| pos[i]).collect();
    let shuffled_fills: Vec<f32> = order.iter().map(|&i| fills[i]).collect();
    let slow = t
        .forward(&pixel_rows_at(&shuffled_fills, &order), &pos_tensor(&shuffled_pos))
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    assert_eq!(fast.len(), slow.len());
    let mut max_diff = 0f32;
    for (rf, rs) in fast.iter().zip(slow.iter()) {
        for (a, b) in rf.iter().zip(rs.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
    }
    assert!(
        max_diff < 5e-5,
        "full-grid fast path (router attention + reshape pool) drifted {max_diff} from the \
         position-keyed irregular path on identical patches"
    );
}

fn pixel_rows_at(fills: &[f32], order: &[usize]) -> Tensor {
    let pp = 2 * 2 * 3;
    let mut v = Vec::with_capacity(fills.len() * pp);
    for (slot, &f) in fills.iter().enumerate() {
        let i = order[slot];
        for j in 0..pp {
            v.push(f + (i * pp + j) as f32 * 1e-3);
        }
    }
    Tensor::from_vec(v, (fills.len(), pp), &Device::Cpu).unwrap()
}
