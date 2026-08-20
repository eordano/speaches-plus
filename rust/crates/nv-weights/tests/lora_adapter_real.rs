use nv_weights::lora_adapter::LoraAdapter;
use nv_weights::{DType, Device};

fn adapter_dir() -> Option<String> {
    std::env::var("NV_LORA_REAL_ADAPTER_DIR").ok()
}

#[test]
#[ignore]
fn real_peft_adapter_loads_and_folds_scaling() {
    let Some(dir) = adapter_dir() else {
        panic!("set NV_LORA_REAL_ADAPTER_DIR to a PEFT adapter directory");
    };
    let adapter = LoraAdapter::load(&dir, &Device::Cpu).unwrap();
    assert!(adapter.config.r > 0);
    assert!(!adapter.loras.is_empty());

    let expected_scaling = if adapter.config.use_rslora {
        adapter.config.lora_alpha / (adapter.config.r as f64).sqrt()
    } else {
        adapter.config.lora_alpha / adapter.config.r as f64
    };
    assert_eq!(adapter.config.scaling, expected_scaling);

    for (name, w) in &adapter.loras {
        assert_eq!(w.scaling, 1.0, "{name} scaling not folded");
        assert_eq!(w.lora_a.dtype(), DType::BF16, "{name} lora_a dtype");
        assert_eq!(w.lora_b.dtype(), DType::BF16, "{name} lora_b dtype");
        let a = w.lora_a.dims2().unwrap();
        let b = w.lora_b.dims2().unwrap();
        assert_eq!(a.0, b.1, "{name} rank mismatch a={a:?} b={b:?}");
        assert!(a.0 <= adapter.config.r, "{name} rank {} > r", a.0);
    }

    if let Ok(want) = std::env::var("NV_LORA_REAL_ADAPTER_MODULE") {
        let w = adapter.loras.get(&want).unwrap_or_else(|| {
            panic!("module {want} absent; have {} modules", adapter.loras.len())
        });
        assert!(!w.is_embedding);
    }

    eprintln!(
        "loaded r={} alpha={} rslora={} scaling={} modules={}",
        adapter.config.r,
        adapter.config.lora_alpha,
        adapter.config.use_rslora,
        adapter.config.scaling,
        adapter.loras.len()
    );
}
