use half::bf16;
use nv_weights::lora_merge::{
    list_mergeable_modules, merge_lora_adapter_into_quantized_snapshot, MODEL_FILE,
};
use safetensors::tensor::{Dtype, TensorView};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const R: usize = 2;
const NV_OUT: usize = 4;
const NV_IN: usize = 32;
const FP8_OUT: usize = 4;
const FP8_IN: usize = 8;
const BF_OUT: usize = 3;
const BF_IN: usize = 8;

const NV_MOD: &str = "model.language_model.layers.0.mlp.gate_proj";
const FP8_MOD: &str = "model.language_model.layers.0.self_attn.q_proj";
const BF_MOD: &str = "model.language_model.layers.0.linear_attn.in_proj_a";
const UNTOUCHED_MOD: &str = "model.language_model.layers.1.mlp.gate_proj";

struct Lcg(u64);
impl Lcg {
    fn f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as u32 as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn bf16_le(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
        .collect()
}

fn f32_le(vals: &[f32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

struct Owned {
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn write_st(path: &Path, tensors: &BTreeMap<String, Owned>) {
    let views: Vec<(String, TensorView)> = tensors
        .iter()
        .map(|(k, o)| {
            (
                k.clone(),
                TensorView::new(o.dtype, o.shape.clone(), &o.bytes).unwrap(),
            )
        })
        .collect();
    let data = safetensors::serialize(views, None).unwrap();
    std::fs::write(path, data).unwrap();
}

fn fake_base(dir: &Path, seed: u64) -> BTreeMap<String, Vec<f32>> {
    std::fs::create_dir_all(dir).unwrap();
    let mut rng = Lcg(seed);
    let mut tensors: BTreeMap<String, Owned> = BTreeMap::new();
    let mut dequants: BTreeMap<String, Vec<f32>> = BTreeMap::new();

    for m in [NV_MOD, UNTOUCHED_MOD] {
        let vals: Vec<Vec<f32>> = (0..NV_OUT)
            .map(|_| (0..NV_IN).map(|_| rng.f32()).collect())
            .collect();
        let stored_global = 2.0f32;
        let q = nv_quant::nvfp4::Nvfp4Tensor::quantize_rows_with_global(&vals, stored_global);
        let deq: Vec<f32> = q
            .dequantize_scaled(1.0 / stored_global)
            .into_iter()
            .flatten()
            .collect();
        dequants.insert(m.to_string(), deq);
        tensors.insert(
            format!("{m}.weight_packed"),
            Owned {
                dtype: Dtype::U8,
                shape: vec![NV_OUT, NV_IN / 2],
                bytes: q.data.clone(),
            },
        );
        tensors.insert(
            format!("{m}.weight_scale"),
            Owned {
                dtype: Dtype::F8_E4M3,
                shape: vec![NV_OUT, NV_IN / 16],
                bytes: q.scales.clone(),
            },
        );
        tensors.insert(
            format!("{m}.weight_global_scale"),
            Owned {
                dtype: Dtype::F32,
                shape: vec![1],
                bytes: f32_le(&[stored_global]),
            },
        );
        tensors.insert(
            format!("{m}.input_global_scale"),
            Owned {
                dtype: Dtype::F32,
                shape: vec![1],
                bytes: f32_le(&[1.0]),
            },
        );
    }

    let fp8_vals: Vec<bf16> = (0..FP8_OUT * FP8_IN)
        .map(|_| bf16::from_f32(rng.f32()))
        .collect();
    let (fp8_bytes, fp8_scales) =
        nv_quant::fp8::quantize_e4m3_per_row(&fp8_vals, FP8_OUT, FP8_IN).unwrap();
    let deq = nv_quant::fp8::dequantize_e4m3_per_row(&fp8_bytes, FP8_OUT, FP8_IN, &fp8_scales)
        .unwrap();
    dequants.insert(FP8_MOD.to_string(), deq);
    tensors.insert(
        format!("{FP8_MOD}.weight"),
        Owned {
            dtype: Dtype::F8_E4M3,
            shape: vec![FP8_OUT, FP8_IN],
            bytes: fp8_bytes,
        },
    );
    tensors.insert(
        format!("{FP8_MOD}.weight_scale"),
        Owned {
            dtype: Dtype::BF16,
            shape: vec![FP8_OUT, 1],
            bytes: bf16_le(&fp8_scales),
        },
    );

    let bf_vals: Vec<f32> = (0..BF_OUT * BF_IN)
        .map(|_| bf16::from_f32(rng.f32()).to_f32())
        .collect();
    dequants.insert(BF_MOD.to_string(), bf_vals.clone());
    tensors.insert(
        format!("{BF_MOD}.weight"),
        Owned {
            dtype: Dtype::BF16,
            shape: vec![BF_OUT, BF_IN],
            bytes: bf16_le(&bf_vals),
        },
    );

    write_st(&dir.join(MODEL_FILE), &tensors);
    std::fs::write(dir.join("config.json"), "{}").unwrap();
    dequants
}

fn fake_adapter(dir: &Path, seed: u64) -> BTreeMap<String, Vec<f32>> {
    std::fs::create_dir_all(dir).unwrap();
    let mut rng = Lcg(seed ^ 0xADA9);
    let alpha = 2.0 * R as f64;
    let scaling = (alpha / R as f64) as f32;
    let mut tensors: BTreeMap<String, Owned> = BTreeMap::new();
    let mut deltas: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for (m, out_f, in_f) in [
        (NV_MOD, NV_OUT, NV_IN),
        (FP8_MOD, FP8_OUT, FP8_IN),
        (BF_MOD, BF_OUT, BF_IN),
    ] {
        let a: Vec<f32> = (0..R * in_f)
            .map(|_| bf16::from_f32(rng.f32()).to_f32())
            .collect();
        let b: Vec<f32> = (0..out_f * R)
            .map(|_| bf16::from_f32(rng.f32()).to_f32())
            .collect();
        let mut delta = vec![0f32; out_f * in_f];
        for r in 0..out_f {
            for c in 0..in_f {
                let mut acc = 0f32;
                for k in 0..R {
                    acc += b[r * R + k] * a[k * in_f + c];
                }
                delta[r * in_f + c] = acc * scaling;
            }
        }
        deltas.insert(m.to_string(), delta);
        tensors.insert(
            format!("base_model.model.{m}.lora_A.weight"),
            Owned {
                dtype: Dtype::BF16,
                shape: vec![R, in_f],
                bytes: bf16_le(&a),
            },
        );
        tensors.insert(
            format!("base_model.model.{m}.lora_B.weight"),
            Owned {
                dtype: Dtype::BF16,
                shape: vec![out_f, R],
                bytes: bf16_le(&b),
            },
        );
    }
    write_st(&dir.join("adapter_model.safetensors"), &tensors);
    let cfg = serde_json::json!({
        "peft_type": "LORA",
        "r": R,
        "lora_alpha": alpha,
        "target_modules": ["gate_proj", "q_proj", "in_proj_a"],
        "bias": "none",
    });
    std::fs::write(
        dir.join("adapter_config.json"),
        serde_json::to_string_pretty(&cfg).unwrap(),
    )
    .unwrap();
    deltas
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("nvw-lora-merge-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn dequant_module(dir: &Path, module: &str, kind: &str, out_f: usize, in_f: usize) -> Vec<f32> {
    let device = candle_core::Device::Cpu;
    let loader = nv_weights::WeightLoader::open_file(dir.join(MODEL_FILE), &device).unwrap();
    match kind {
        "nvfp4_packed" => {
            let packed = loader
                .raw_bytes(&format!("{module}.weight_packed"))
                .unwrap()
                .to_vec();
            let scales = loader
                .raw_bytes(&format!("{module}.weight_scale"))
                .unwrap()
                .to_vec();
            let g: Vec<f32> = loader
                .get(&format!("{module}.weight_global_scale"), nv_weights::DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            nv_quant::nvfp4::dequantize_packed_linear(&packed, &scales, out_f, in_f, 1.0 / g[0])
        }
        "fp8_rowscale" => {
            let bytes = loader
                .raw_bytes(&format!("{module}.weight"))
                .unwrap()
                .to_vec();
            let scales: Vec<f32> = loader
                .get(&format!("{module}.weight_scale"), nv_weights::DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            nv_quant::fp8::dequantize_e4m3_per_row(&bytes, out_f, in_f, &scales).unwrap()
        }
        "bf16" => loader
            .get(&format!("{module}.weight"), nv_weights::DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap(),
        other => panic!("unknown kind {other}"),
    }
}

#[test]
fn merged_snapshot_dequantizes_to_base_plus_delta_within_the_storage_quant_step() {
    let base_dir = tmp("base");
    let adapter_dir = tmp("adapter");
    let out_dir = tmp("out");
    let base_deq = fake_base(&base_dir, 11);
    let deltas = fake_adapter(&adapter_dir, 11);

    let report =
        merge_lora_adapter_into_quantized_snapshot(&base_dir, &adapter_dir, &out_dir).unwrap();
    assert_eq!(report.modules.len(), 3, "three adapter modules must merge");
    assert_eq!(report.count("nvfp4_packed"), 1);
    assert_eq!(report.count("fp8_rowscale"), 1);
    assert_eq!(report.count("bf16"), 1);

    for (m, kind, out_f, in_f, tol) in [
        (NV_MOD, "nvfp4_packed", NV_OUT, NV_IN, 0.40f32),
        (FP8_MOD, "fp8_rowscale", FP8_OUT, FP8_IN, 0.10f32),
        (BF_MOD, "bf16", BF_OUT, BF_IN, 0.02f32),
    ] {
        let got = dequant_module(&out_dir, m, kind, out_f, in_f);
        let want: Vec<f32> = base_deq[m]
            .iter()
            .zip(deltas[m].iter())
            .map(|(w, d)| w + d)
            .collect();
        let amax = want.iter().fold(0f32, |a, v| a.max(v.abs()));
        let worst = got
            .iter()
            .zip(want.iter())
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        assert!(
            worst / amax <= tol,
            "{m} ({kind}): merged dequant drifted {worst:.4} (amax {amax:.4}, rel {:.4}) from \
             base+delta; the requant step for this storage should stay under {tol}",
            worst / amax
        );
        let base_worst = got
            .iter()
            .zip(base_deq[m].iter())
            .map(|(g, w)| (g - w).abs())
            .fold(0f32, f32::max);
        let delta_amax = deltas[m].iter().fold(0f32, |a, v| a.max(v.abs()));
        assert!(
            base_worst > delta_amax * 0.25,
            "{m} ({kind}): merged bytes barely moved off the base (moved {base_worst:.5}, delta \
             amax {delta_amax:.5}) -- the delta was not applied"
        );
    }
}

#[test]
fn modules_absent_from_the_adapter_stay_byte_identical_and_the_header_layout_is_unchanged() {
    let base_dir = tmp("base2");
    let adapter_dir = tmp("adapter2");
    let out_dir = tmp("out2");
    fake_base(&base_dir, 23);
    fake_adapter(&adapter_dir, 23);
    merge_lora_adapter_into_quantized_snapshot(&base_dir, &adapter_dir, &out_dir).unwrap();

    let base_bytes = std::fs::read(base_dir.join(MODEL_FILE)).unwrap();
    let out_bytes = std::fs::read(out_dir.join(MODEL_FILE)).unwrap();
    assert_eq!(
        base_bytes.len(),
        out_bytes.len(),
        "in-place merge must never change the file size"
    );
    let device = candle_core::Device::Cpu;
    let base_loader = nv_weights::WeightLoader::open_file(base_dir.join(MODEL_FILE), &device).unwrap();
    let out_loader = nv_weights::WeightLoader::open_file(out_dir.join(MODEL_FILE), &device).unwrap();
    for suffix in ["weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"] {
        let name = format!("{UNTOUCHED_MOD}.{suffix}");
        assert_eq!(
            base_loader.raw_bytes(&name).unwrap(),
            out_loader.raw_bytes(&name).unwrap(),
            "{name} is not in the adapter and must be byte-identical after the merge"
        );
    }
}

#[test]
fn the_module_lister_reports_every_quantized_linear_with_its_true_geometry() {
    let base_dir = tmp("base3");
    fake_base(&base_dir, 31);
    let mods = list_mergeable_modules(&base_dir).unwrap();
    let find = |m: &str| mods.iter().find(|x| x.module == m).unwrap();
    let nv = find(NV_MOD);
    assert_eq!((nv.kind, nv.out_features, nv.in_features), ("nvfp4_packed", NV_OUT, NV_IN));
    let f8 = find(FP8_MOD);
    assert_eq!((f8.kind, f8.out_features, f8.in_features), ("fp8_rowscale", FP8_OUT, FP8_IN));
    let bf = find(BF_MOD);
    assert_eq!((bf.kind, bf.out_features, bf.in_features), ("bf16", BF_OUT, BF_IN));
}
