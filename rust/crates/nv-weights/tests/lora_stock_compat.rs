use nv_weights::lora_adapter::{normalize_module_name, LoraAdapter, TargetModules};
use nv_weights::{DType, Device, Tensor};
use safetensors::tensor::{Dtype as StDtype, TensorView};
use std::path::PathBuf;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn write_adapter(
    tag: &str,
    config_json: &str,
    tensors: &[(String, Vec<usize>, Vec<f32>)],
) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "nv-weights-stock-lora-{}-{tag}",
        std::process::id()
    ));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("adapter_config.json"), config_json).unwrap();
    let byte_bufs: Vec<Vec<u8>> = tensors.iter().map(|(_, _, v)| f32_bytes(v)).collect();
    let views: Vec<(String, TensorView)> = tensors
        .iter()
        .zip(byte_bufs.iter())
        .map(|((name, shape, _), bytes)| {
            (
                name.clone(),
                TensorView::new(StDtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, None, &dir.join("adapter_model.safetensors")).unwrap();
    dir
}

fn tensor_to_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn ramp(n: usize, base: f32) -> Vec<f32> {
    (0..n)
        .map(|i| base * (if i % 2 == 0 { 1.0 } else { -0.5 }) * (1.0 + (i % 4) as f32))
        .collect()
}

const HIDDEN: usize = 8;
const RANK: usize = 4;

const SUBS: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

#[allow(clippy::type_complexity)]
fn stock_tensors(
    layers: usize,
) -> (
    Vec<(String, Vec<usize>, Vec<f32>)>,
    Vec<((usize, &'static str), (Vec<f32>, Vec<f32>))>,
) {
    let mut tensors = Vec::new();
    let mut vals = Vec::new();
    for li in 0..layers {
        for (si, sub) in SUBS.iter().enumerate() {
            let a = ramp(RANK * HIDDEN, 0.25 + (li as f32) + (si as f32) * 0.125);
            let b = ramp(
                HIDDEN * RANK,
                0.5 + (li as f32) * 0.0625 + (si as f32) * 0.25,
            );
            let base = format!("base_model.model.model.layers.{li}.{sub}");
            tensors.push((
                format!("{base}.lora_A.weight"),
                vec![RANK, HIDDEN],
                a.clone(),
            ));
            tensors.push((
                format!("{base}.lora_B.weight"),
                vec![HIDDEN, RANK],
                b.clone(),
            ));
            vals.push(((li, *sub), (a, b)));
        }
    }
    (tensors, vals)
}

fn stock_config() -> String {
    r#"{
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "google/gemma-3-4b-it",
        "r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_rslora": false,
        "use_dora": false,
        "fan_in_fan_out": false,
        "inference_mode": true,
        "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    }"#
    .to_string()
}

#[test]
fn stock_adapter_loads_and_routes() {
    let layers = 3;
    let (tensors, vals) = stock_tensors(layers);
    let dir = write_adapter("route", &stock_config(), &tensors);
    let adapter = LoraAdapter::load(&dir, &Device::Cpu).unwrap();

    assert_eq!(adapter.config.r, 4);
    assert_eq!(adapter.config.lora_alpha, 8.0);
    assert_eq!(adapter.config.scaling, 2.0);
    assert!(!adapter.config.use_rslora);
    assert_eq!(
        adapter.config.target_modules,
        TargetModules::List(
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]
            .iter()
            .map(|s| s.to_string())
            .collect()
        )
    );

    let mut want: Vec<String> = Vec::new();
    for li in 0..layers {
        for sub in SUBS {
            want.push(format!("model.language_model.layers.{li}.{sub}"));
        }
    }
    want.sort();
    let mut got: Vec<String> = adapter.loras.keys().cloned().collect();
    got.sort();
    assert_eq!(
        got, want,
        "stock modules must route to language_model serving names"
    );

    for ((li, sub), (a, b)) in &vals {
        let key = format!("model.language_model.layers.{li}.{sub}");
        let w = adapter
            .loras
            .get(&key)
            .unwrap_or_else(|| panic!("missing {key}"));
        assert_eq!(w.rank, RANK);
        assert!(!w.is_embedding);
        assert_eq!(w.lora_a.dims(), &[RANK, HIDDEN]);
        assert_eq!(w.lora_b.dims(), &[HIDDEN, RANK]);
        assert_eq!(
            &tensor_to_f32(&w.lora_a),
            a,
            "{key} A must be byte-identical"
        );
        let exp_b: Vec<f32> = b.iter().map(|x| x * 2.0).collect();
        assert_eq!(
            tensor_to_f32(&w.lora_b),
            exp_b,
            "{key} B must be scaled by alpha/r"
        );
    }

    eprintln!(
        "STOCK_LOAD ok: r={} alpha={} scaling={} modules={} (routed to model.language_model.layers.*)",
        adapter.config.r,
        adapter.config.lora_alpha,
        adapter.config.scaling,
        adapter.loras.len()
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn our_vs_stock_namespace_equivalence() {
    let layers = 3;
    let (stock, _vals) = stock_tensors(layers);

    let ours: Vec<(String, Vec<usize>, Vec<f32>)> = stock
        .iter()
        .map(|(name, shape, v)| {
            let module = name
                .strip_prefix("base_model.model.model.layers.")
                .expect("stock key prefix");
            (
                format!("base_model.model.model.language_model.layers.{module}"),
                shape.clone(),
                v.clone(),
            )
        })
        .collect();

    let dir_stock = write_adapter("equiv-stock", &stock_config(), &stock);
    let dir_ours = write_adapter("equiv-ours", &stock_config(), &ours);
    let a_stock = LoraAdapter::load(&dir_stock, &Device::Cpu).unwrap();
    let a_ours = LoraAdapter::load(&dir_ours, &Device::Cpu).unwrap();

    let ks: Vec<String> = {
        let mut k: Vec<_> = a_stock.loras.keys().cloned().collect();
        k.sort();
        k
    };
    let ko: Vec<String> = {
        let mut k: Vec<_> = a_ours.loras.keys().cloned().collect();
        k.sort();
        k
    };
    assert_eq!(
        ks, ko,
        "our-namespace and stock-namespace keys must coincide after the shim"
    );

    let mut max_abs = 0.0f32;
    for key in &ks {
        let ws = a_stock.loras.get(key).unwrap();
        let wo = a_ours.loras.get(key).unwrap();
        let da = tensor_to_f32(&ws.lora_a);
        let db = tensor_to_f32(&wo.lora_a);
        for (x, y) in da.iter().zip(db.iter()) {
            max_abs = max_abs.max((x - y).abs());
        }
        let sb = tensor_to_f32(&ws.lora_b);
        let ob = tensor_to_f32(&wo.lora_b);
        for (x, y) in sb.iter().zip(ob.iter()) {
            max_abs = max_abs.max((x - y).abs());
        }
    }
    assert_eq!(max_abs, 0.0, "namespace shim must be a pure rename");
    eprintln!(
        "NAMESPACE_EQUIV_MAXABS {:.3e}  (our vs stock, {} modules)",
        max_abs,
        a_stock.loras.len()
    );
    std::fs::remove_dir_all(&dir_stock).ok();
    std::fs::remove_dir_all(&dir_ours).ok();
}

#[test]
fn our_adapter_regression_unchanged() {
    let a = ramp(RANK * HIDDEN, 0.75);
    let b = ramp(HIDDEN * RANK, 1.25);
    let tensors = vec![
        (
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
                .to_string(),
            vec![RANK, HIDDEN],
            a.clone(),
        ),
        (
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight"
                .to_string(),
            vec![HIDDEN, RANK],
            b.clone(),
        ),
    ];
    let dir = write_adapter("regress", &stock_config(), &tensors);
    let adapter = LoraAdapter::load(&dir, &Device::Cpu).unwrap();

    let key = "model.language_model.layers.0.self_attn.q_proj";
    assert_eq!(
        normalize_module_name(key),
        key,
        "our namespace must be a no-op"
    );
    let w = adapter.loras.get(key).unwrap();
    assert_eq!(&tensor_to_f32(&w.lora_a), &a);
    let exp_b: Vec<f32> = b.iter().map(|x| x * 2.0).collect();
    assert_eq!(tensor_to_f32(&w.lora_b), exp_b);
    assert_eq!(adapter.loras.len(), 1);
    eprintln!("OUR_ADAPTER_REGRESSION ok: key unchanged `{key}`, A byte-identical");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn dora_fails_clearly() {
    let cfg_dora = r#"{
        "peft_type": "LORA", "r": 4, "lora_alpha": 8, "use_dora": true,
        "target_modules": ["q_proj"]
    }"#;
    let a = ramp(RANK * HIDDEN, 0.5);
    let b = ramp(HIDDEN * RANK, 0.5);
    let ok_tensors = vec![
        (
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
            vec![RANK, HIDDEN],
            a.clone(),
        ),
        (
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_string(),
            vec![HIDDEN, RANK],
            b.clone(),
        ),
    ];
    let dir = write_adapter("dora-cfg", cfg_dora, &ok_tensors);
    let err = LoraAdapter::load(&dir, &Device::Cpu)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("DoRA"),
        "config DoRA must be rejected, got: {err}"
    );
    eprintln!("DORA_CONFIG_REJECT: {err}");
    std::fs::remove_dir_all(&dir).ok();

    let mut dora_tensors = ok_tensors.clone();
    dora_tensors.push((
        "base_model.model.model.layers.0.self_attn.q_proj.lora_magnitude_vector".to_string(),
        vec![HIDDEN],
        ramp(HIDDEN, 1.0),
    ));
    let dir2 = write_adapter("dora-vec", &stock_config(), &dora_tensors);
    let err2 = LoraAdapter::load(&dir2, &Device::Cpu)
        .unwrap_err()
        .to_string();
    assert!(
        err2.contains("DoRA") && err2.contains("lora_magnitude_vector"),
        "tensor DoRA must be rejected clearly, got: {err2}"
    );
    eprintln!("DORA_TENSOR_REJECT: {err2}");
    std::fs::remove_dir_all(&dir2).ok();
}
