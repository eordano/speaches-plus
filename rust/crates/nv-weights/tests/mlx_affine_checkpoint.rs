use nv_weights::{QuantScheme, QuantizationConfig, WeightLoader};
use std::path::PathBuf;

#[test]
#[ignore = "reads a real mlx-community 4bit snapshot; set NV_MLXQ_DIR to its path -- detects the {quantization: {mode: affine, bits: 4}} config and round-trips a packed linear through load_quantized_weight"]
fn detects_and_loads_the_mlx_affine_checkpoint() {
    let dir = PathBuf::from(std::env::var("NV_MLXQ_DIR").expect(
        "set NV_MLXQ_DIR to an mlx-community 4bit snapshot; this gate must never silently skip",
    ));
    let cfg = QuantizationConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
    assert_eq!(cfg.scheme, QuantScheme::MlxAffineW4, "scheme detection");
    assert_eq!(cfg.group_size, Some(64), "group size from config");

    let w = WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("open");
    let name = "language_model.model.layers.0.mlp.gate_proj.weight";
    let qw = w
        .load_quantized_weight(name, QuantScheme::MlxAffineW4)
        .expect("load affine linear");
    let n = qw.shape[0];
    let words = qw.shape[1];
    assert_eq!(qw.packed_bytes.len(), n * words * 4, "u32 words payload");
    let groups = words * 8 / 64;
    let scales = qw.weight_scale_bytes.as_ref().expect("scales");
    let biases = qw.weight_bias_bytes.as_ref().expect("biases");
    assert_eq!(scales.len(), n * groups * 2, "bf16 scales payload");
    assert_eq!(biases.len(), n * groups * 2, "bf16 biases payload");
    eprintln!("MLX-AFFINE-LOAD {name} n{n} k{} groups/row {groups}", words * 8);
}
