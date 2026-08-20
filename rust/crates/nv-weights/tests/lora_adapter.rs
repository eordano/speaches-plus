use half::bf16;
use nv_weights::lora_adapter::{
    parse_fine_tuned_lora_name, LoraAdapter, PeftConfig, TargetModules,
};
use nv_weights::{DType, Device};
use safetensors::tensor::{Dtype as StDtype, TensorView};
use std::path::PathBuf;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn make_adapter_dir(
    tag: &str,
    config_json: &str,
    tensors: &[(&str, Vec<usize>, Vec<f32>)],
) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("nv-weights-lora-test-{}-{tag}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("adapter_config.json"), config_json).unwrap();

    let byte_bufs: Vec<Vec<u8>> = tensors.iter().map(|(_, _, v)| f32_bytes(v)).collect();
    let views: Vec<(String, TensorView)> = tensors
        .iter()
        .zip(byte_bufs.iter())
        .map(|((name, shape, _), bytes)| {
            (
                name.to_string(),
                TensorView::new(StDtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, None, &dir.join("adapter_model.safetensors")).unwrap();
    dir
}

fn tensor_to_f32(t: &nv_weights::Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn exactly_representable_bf16(x: f32) -> bool {
    bf16::from_f32(x).to_f32() == x
}

#[test]
fn scaling_folded_into_b_when_alpha_ne_r() {
    let a_vals: Vec<f32> = vec![1.0, -0.5, 0.25, 2.0, -4.0, 0.125, 8.0, -1.0];
    let b_vals: Vec<f32> = vec![0.5, 1.0, -2.0, 0.25, -0.125, 4.0];
    for v in a_vals.iter().chain(b_vals.iter()) {
        assert!(exactly_representable_bf16(*v));
    }
    let config = r#"{
        "r": 4,
        "lora_alpha": 8,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "use_rslora": false,
        "use_dora": false,
        "peft_type": "LORA",
        "rank_pattern": {},
        "alpha_pattern": {},
        "some_future_key": 123
    }"#;
    let dir = make_adapter_dir(
        "fold",
        config,
        &[
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                vec![4, 2],
                a_vals.clone(),
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                vec![3, 2],
                b_vals.clone(),
            ),
        ],
    );

    let adapter = LoraAdapter::load(&dir, &Device::Cpu).unwrap();
    assert_eq!(adapter.config.r, 4);
    assert_eq!(adapter.config.lora_alpha, 8.0);
    assert_eq!(adapter.config.scaling, 2.0);
    assert_eq!(
        adapter.config.target_modules,
        TargetModules::List(vec!["q_proj".to_string(), "v_proj".to_string()])
    );

    let w = adapter
        .loras
        .get("model.language_model.layers.0.self_attn.q_proj")
        .unwrap();
    assert_eq!(w.scaling, 1.0);
    assert_eq!(w.rank, 4);
    assert!(!w.is_embedding);
    assert_eq!(w.lora_a.dtype(), DType::BF16);
    assert_eq!(w.lora_b.dtype(), DType::BF16);
    assert_eq!(w.lora_a.dims(), &[4, 2]);
    assert_eq!(w.lora_b.dims(), &[3, 2]);

    let got_a = tensor_to_f32(&w.lora_a);
    assert_eq!(got_a, a_vals);
    let got_b = tensor_to_f32(&w.lora_b);
    let expected_b: Vec<f32> = b_vals.iter().map(|x| x * 2.0).collect();
    assert_eq!(got_b, expected_b);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn rslora_uses_sqrt_rank() {
    let a_vals: Vec<f32> = vec![1.0, 2.0, -1.0, 0.5, 0.25, -8.0, 4.0, -0.5];
    let b_vals: Vec<f32> = vec![1.0, -0.5, 0.25, 2.0];
    let config = r#"{
        "r": 4,
        "lora_alpha": 6,
        "target_modules": ["gate_proj"],
        "use_rslora": true
    }"#;
    let dir = make_adapter_dir(
        "rslora",
        config,
        &[
            (
                "base_model.model.model.layers.1.mlp.gate_proj.lora_A.weight",
                vec![4, 2],
                a_vals.clone(),
            ),
            (
                "base_model.model.model.layers.1.mlp.gate_proj.lora_B.weight",
                vec![2, 2],
                b_vals.clone(),
            ),
        ],
    );

    let adapter = LoraAdapter::load(&dir, &Device::Cpu).unwrap();
    assert_eq!(adapter.config.scaling, 3.0);
    assert!(adapter.config.use_rslora);

    let w = adapter
        .loras
        .get("model.language_model.layers.1.mlp.gate_proj")
        .unwrap();
    assert_eq!(w.scaling, 1.0);
    let got_b = tensor_to_f32(&w.lora_b);
    let expected_b: Vec<f32> = b_vals.iter().map(|x| x * 3.0).collect();
    assert_eq!(got_b, expected_b);
    assert_eq!(tensor_to_f32(&w.lora_a), a_vals);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn two_adapters_different_alpha_r_both_end_at_scaling_one() {
    let a_vals: Vec<f32> = vec![1.0, -1.0, 0.5, 2.0];
    let b1_vals: Vec<f32> = vec![0.5, -2.0, 1.0, 0.25];
    let b2_vals: Vec<f32> = vec![4.0, -0.25, 0.125, 1.0];

    let dir1 = make_adapter_dir(
        "multi1",
        r#"{"r": 2, "lora_alpha": 16, "target_modules": ["q_proj"]}"#,
        &[
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                vec![2, 2],
                a_vals.clone(),
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                vec![2, 2],
                b1_vals.clone(),
            ),
        ],
    );
    let dir2 = make_adapter_dir(
        "multi2",
        r#"{"r": 8, "lora_alpha": 4, "target_modules": ["q_proj"]}"#,
        &[
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                vec![8, 2],
                vec![1.0; 16],
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                vec![2, 2],
                b2_vals.clone(),
            ),
        ],
    );

    let ad1 = LoraAdapter::load(&dir1, &Device::Cpu).unwrap();
    let ad2 = LoraAdapter::load(&dir2, &Device::Cpu).unwrap();

    assert_eq!(ad1.config.scaling, 8.0);
    assert_eq!(ad2.config.scaling, 0.5);

    let w1 = ad1
        .loras
        .get("model.language_model.layers.0.self_attn.q_proj")
        .unwrap();
    let w2 = ad2
        .loras
        .get("model.language_model.layers.0.self_attn.q_proj")
        .unwrap();
    assert_eq!(w1.scaling, 1.0);
    assert_eq!(w2.scaling, 1.0);

    let got1 = tensor_to_f32(&w1.lora_b);
    let expected1: Vec<f32> = b1_vals.iter().map(|x| x * 8.0).collect();
    assert_eq!(got1, expected1);

    let got2 = tensor_to_f32(&w2.lora_b);
    let expected2: Vec<f32> = b2_vals.iter().map(|x| x * 0.5).collect();
    assert_eq!(got2, expected2);

    std::fs::remove_dir_all(&dir1).ok();
    std::fs::remove_dir_all(&dir2).ok();
}

#[test]
fn scaling_one_short_circuits() {
    let b_vals: Vec<f32> = vec![0.5, -2.0, 1.0, 0.25];
    let dir = make_adapter_dir(
        "noop",
        r#"{"r": 8, "lora_alpha": 8, "target_modules": ["v_proj"]}"#,
        &[
            (
                "base_model.model.model.layers.2.self_attn.v_proj.lora_A.weight",
                vec![8, 2],
                vec![0.5; 16],
            ),
            (
                "base_model.model.model.layers.2.self_attn.v_proj.lora_B.weight",
                vec![2, 2],
                b_vals.clone(),
            ),
        ],
    );
    let adapter = LoraAdapter::load(&dir, &Device::Cpu).unwrap();
    assert_eq!(adapter.config.scaling, 1.0);
    let w = adapter
        .loras
        .get("model.language_model.layers.2.self_attn.v_proj")
        .unwrap();
    assert_eq!(w.scaling, 1.0);
    assert_eq!(tensor_to_f32(&w.lora_b), b_vals);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn embedding_tensor_names_accepted() {
    let dir = make_adapter_dir(
        "embed",
        r#"{"r": 2, "lora_alpha": 4, "target_modules": ["embed_tokens"]}"#,
        &[
            (
                "base_model.model.model.embed_tokens.lora_embedding_A",
                vec![2, 4],
                vec![1.0; 8],
            ),
            (
                "base_model.model.model.embed_tokens.lora_embedding_B",
                vec![4, 2],
                vec![0.5; 8],
            ),
        ],
    );
    let adapter = LoraAdapter::load(&dir, &Device::Cpu).unwrap();
    let w = adapter
        .loras
        .get("model.language_model.embed_tokens")
        .unwrap();
    assert!(w.is_embedding);
    assert_eq!(w.scaling, 1.0);
    assert_eq!(tensor_to_f32(&w.lora_b), vec![1.0; 8]);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn rejects_dora_bias_modules_to_save() {
    let err = PeftConfig::from_json_str(
        r#"{"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"], "use_dora": true}"#,
    )
    .unwrap_err();
    assert!(err.to_string().contains("does not yet support DoRA"));

    let err = PeftConfig::from_json_str(
        r#"{"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"], "bias": "all"}"#,
    )
    .unwrap_err();
    assert!(err.to_string().contains("Adapter bias is not supported"));

    let err = PeftConfig::from_json_str(
        r#"{"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"], "modules_to_save": ["lm_head"]}"#,
    )
    .unwrap_err();
    assert!(err
        .to_string()
        .contains("only supports modules_to_save being None"));

    let ok = PeftConfig::from_json_str(
        r#"{"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"], "modules_to_save": null, "bias": "none"}"#,
    );
    assert!(ok.is_ok());
    let ok = PeftConfig::from_json_str(
        r#"{"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"], "modules_to_save": []}"#,
    );
    assert!(ok.is_ok());
}

#[test]
fn rejects_missing_and_invalid_required_fields() {
    let err = PeftConfig::from_json_str(r#"{"lora_alpha": 8, "target_modules": ["q_proj"]}"#)
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("Missing required configuration fields"));
    assert!(err.to_string().contains("r"));

    let err = PeftConfig::from_json_str(r#"{"r": 4, "lora_alpha": 8}"#).unwrap_err();
    assert!(err.to_string().contains("target_modules"));

    let err =
        PeftConfig::from_json_str(r#"{"r": 0, "lora_alpha": 8, "target_modules": ["q_proj"]}"#)
            .unwrap_err();
    assert!(err.to_string().contains("must be a positive integer"));

    let err =
        PeftConfig::from_json_str(r#"{"r": -2, "lora_alpha": 8, "target_modules": ["q_proj"]}"#)
            .unwrap_err();
    assert!(err.to_string().contains("must be a positive integer"));
}

#[test]
fn target_modules_string_form_accepted() {
    let cfg =
        PeftConfig::from_json_str(r#"{"r": 4, "lora_alpha": 8, "target_modules": ".*(q|v)_proj"}"#)
            .unwrap();
    assert_eq!(
        cfg.target_modules,
        TargetModules::Pattern(".*(q|v)_proj".to_string())
    );
}

#[test]
fn parse_names() {
    assert_eq!(
        parse_fine_tuned_lora_name(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        )
        .unwrap(),
        ("model.layers.0.self_attn.q_proj".to_string(), true, false)
    );
    assert_eq!(
        parse_fine_tuned_lora_name("model.layers.0.self_attn.q_proj.lora_B.weight").unwrap(),
        ("model.layers.0.self_attn.q_proj".to_string(), false, false)
    );
    assert_eq!(
        parse_fine_tuned_lora_name("base_model.model.model.embed_tokens.lora_embedding_B").unwrap(),
        ("model.embed_tokens".to_string(), false, true)
    );
    assert!(
        parse_fine_tuned_lora_name("base_model.model.model.layers.0.mlp.up_proj.weight").is_err()
    );
    assert!(parse_fine_tuned_lora_name("lora_A.weight.extra").is_err());
}

#[test]
fn unparseable_tensor_name_in_file_is_error() {
    let dir = make_adapter_dir(
        "badname",
        r#"{"r": 2, "lora_alpha": 2, "target_modules": ["q_proj"]}"#,
        &[
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                vec![2, 2],
                vec![1.0; 4],
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                vec![2, 2],
                vec![1.0; 4],
            ),
            (
                "base_model.model.model.layers.0.self_attn.q_proj.dora_magnitude",
                vec![2],
                vec![1.0; 2],
            ),
        ],
    );
    let err = LoraAdapter::load(&dir, &Device::Cpu).unwrap_err();
    assert!(err.to_string().contains("unsupported LoRA weight"));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn missing_pair_half_is_error() {
    let dir = make_adapter_dir(
        "halfpair",
        r#"{"r": 2, "lora_alpha": 2, "target_modules": ["q_proj"]}"#,
        &[(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            vec![2, 2],
            vec![1.0; 4],
        )],
    );
    let err = LoraAdapter::load(&dir, &Device::Cpu).unwrap_err();
    assert!(err.to_string().contains("missing lora_B"));
    std::fs::remove_dir_all(&dir).ok();
}
