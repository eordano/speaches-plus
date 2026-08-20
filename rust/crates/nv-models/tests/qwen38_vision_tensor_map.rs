use nv_omni::Qwen3VisionConfig;
use std::collections::BTreeMap;

mod hub_snapshot;

const QWEN38_27B_CONFIG_JSON: &str = include_str!("qwen3_8_27b_config.json");

fn release_vision_config() -> Qwen3VisionConfig {
    let v: serde_json::Value = serde_json::from_str(QWEN38_27B_CONFIG_JSON).unwrap();
    Qwen3VisionConfig::from_hf_value(&v["vision_config"])
        .expect("release vision_config parses as qwen3_5_vision")
}

#[test]
fn release_vision_config_states_the_qwen3_vl_style_non_windowed_tower() {
    let v: serde_json::Value = serde_json::from_str(QWEN38_27B_CONFIG_JSON).unwrap();
    let vc = &v["vision_config"];
    assert_eq!(vc["model_type"], "qwen3_5_vision");
    assert!(
        vc.get("window_size").is_none() && vc.get("fullatt_block_indexes").is_none(),
        "no windowed-attention keys: every block is full attention, unlike qwen2.5-VL"
    );
    assert_eq!(
        vc["deepstack_visual_indexes"]
            .as_array()
            .expect("deepstack key present")
            .len(),
        0,
        "the 27B release ships no deepstack taps"
    );
    assert_eq!(vc["hidden_act"], "gelu_pytorch_tanh");

    let cfg = release_vision_config();
    assert_eq!(cfg.depth, 27);
    assert_eq!(cfg.hidden_size, 1152);
    assert_eq!(cfg.num_heads, 16);
    assert_eq!(cfg.head_dim(), 72);
    assert_eq!(cfg.intermediate_size, 4304);
    assert_eq!(cfg.patch_size, 16);
    assert_eq!(cfg.spatial_merge_size, 2);
    assert_eq!(cfg.temporal_patch_size, 2);
    assert_eq!(cfg.num_position_embeddings, 2304, "48x48 learned grid");
    assert_eq!(cfg.out_hidden_size, 5120, "tower projects into the 27B trunk width");
    assert_eq!(cfg.merger_hidden(), 4608);
}

#[test]
fn expected_tensor_tree_matches_the_release_index_facts() {
    let cfg = release_vision_config();
    let names = cfg.expected_checkpoint_tensor_names_with_shapes();
    assert_eq!(names.len(), 333, "9 singletons + 27 blocks x 12 tensors");

    let map: BTreeMap<String, Vec<usize>> = names.into_iter().collect();
    assert_eq!(map.len(), 333, "no duplicate names");
    assert_eq!(
        map["model.visual.patch_embed.proj.weight"],
        vec![1152, 3, 2, 16, 16]
    );
    assert_eq!(map["model.visual.pos_embed.weight"], vec![2304, 1152]);
    assert_eq!(map["model.visual.blocks.0.attn.qkv.weight"], vec![3456, 1152]);
    assert_eq!(map["model.visual.blocks.26.attn.qkv.weight"], vec![3456, 1152]);
    assert_eq!(
        map["model.visual.blocks.13.mlp.linear_fc1.weight"],
        vec![4304, 1152]
    );
    assert_eq!(
        map["model.visual.merger.linear_fc1.weight"],
        vec![4608, 4608]
    );
    assert_eq!(
        map["model.visual.merger.linear_fc2.weight"],
        vec![5120, 4608]
    );
    for suffix in [
        "norm1.weight",
        "norm1.bias",
        "norm2.weight",
        "norm2.bias",
        "attn.qkv.weight",
        "attn.qkv.bias",
        "attn.proj.weight",
        "attn.proj.bias",
        "mlp.linear_fc1.weight",
        "mlp.linear_fc1.bias",
        "mlp.linear_fc2.weight",
        "mlp.linear_fc2.bias",
    ] {
        for i in 0..27 {
            assert!(
                map.contains_key(&format!("model.visual.blocks.{i}.{suffix}")),
                "block {i} missing {suffix}"
            );
        }
    }
    assert!(
        !map.keys().any(|k| k.contains("deepstack")),
        "no deepstack tensors expected for the 27B release"
    );
}

fn read_safetensors_header(path: &std::path::Path) -> serde_json::Value {
    use std::io::Read;
    let mut f = std::fs::File::open(path).expect("open safetensors");
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8).expect("header length");
    let n = u64::from_le_bytes(len8) as usize;
    let mut hdr = vec![0u8; n];
    f.read_exact(&mut hdr).expect("header json");
    serde_json::from_slice(&hdr).expect("header parses")
}

#[test]
fn real_checkpoint_visual_tree_is_exactly_the_expected_names_and_all_bf16() {
    let Some(dir) = std::env::var("NV_QWEN38_DIR")
        .ok()
        .map(std::path::PathBuf::from)
        .filter(|p| p.is_dir())
        .or_else(|| {
            hub_snapshot::snapshot_of("unsloth/Qwen3.8-27B-NVFP4", &["config.json", "*.safetensors"])
        })
    else {
        eprintln!(
            "[skip] no unsloth/Qwen3.8-27B-NVFP4 snapshot under {:?}; the header-only \
             tensor-map check needs the checkpoint files (weights are not loaded)",
            hub_snapshot::hub_roots()
        );
        return;
    };
    let cfg = Qwen3VisionConfig::from_hf_config_json(dir.join("config.json"))
        .expect("real config vision_config");
    let hdr = read_safetensors_header(&dir.join("model.safetensors"));
    let obj = hdr.as_object().expect("header object");
    let mut real: BTreeMap<String, (String, Vec<usize>)> = BTreeMap::new();
    for (k, v) in obj {
        if !k.contains("visual") {
            continue;
        }
        let dtype = v["dtype"].as_str().expect("dtype").to_string();
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .expect("shape")
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        real.insert(k.clone(), (dtype, shape));
    }
    let expected: BTreeMap<String, Vec<usize>> = cfg
        .expected_checkpoint_tensor_names_with_shapes()
        .into_iter()
        .collect();
    let real_names: Vec<&String> = real.keys().collect();
    let expected_names: Vec<&String> = expected.keys().collect();
    assert_eq!(
        real_names, expected_names,
        "checkpoint visual.* tree must be exactly the loader's expected tree"
    );
    for (name, (dtype, shape)) in &real {
        assert_eq!(
            dtype, "BF16",
            "{name}: the quantization_config targets never match visual.*, so every \
             vision tensor ships bf16 and the tower loads without dequant"
        );
        assert_eq!(&expected[name], shape, "{name} shape");
    }
    eprintln!(
        "[q38-vision-map] {} visual tensors verified BF16 with expected shapes",
        real.len()
    );
}
