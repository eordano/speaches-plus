#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

mod hub_snapshot;

fn require_probe_env() {
    if std::env::var("NV_Q38_TEST").as_deref() != Ok("1") {
        panic!("set NV_Q38_TEST=1 to run the real-checkpoint quant-arm probes");
    }
}

fn snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN38_DIR") {
        return PathBuf::from(d);
    }
    hub_snapshot::snapshot_of(
        "unsloth/Qwen3.8-27B-NVFP4",
        &["config.json", "*.safetensors"],
    )
    .expect("no hydrated unsloth/Qwen3.8-27B-NVFP4 snapshot; set NV_QWEN38_DIR")
}

fn shape2_of(weights: &WeightLoader, name: &str) -> (usize, usize) {
    let s = weights
        .shape_of(name)
        .unwrap_or_else(|| panic!("missing shape for {name}"));
    assert_eq!(s.len(), 2, "{name}: expected 2-D, got {s:?}");
    (s[0], s[1])
}

fn host_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("to host")
}

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32) as f32 / (1u64 << 32) as f32 - 0.5
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

#[test]
#[ignore = "reads two real fp8 tensors from the 22.6 GB checkpoint; set NV_Q38_TEST=1"]
fn probe_a_fp8_rowscale_dequant_linear_is_bitwise_the_host_reference_on_real_tensors() {
    require_probe_env();
    let dir = snapshot_dir();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    for module in [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.3.self_attn.q_proj",
    ] {
        let wname = format!("{module}.weight");
        let sname = format!("{module}.weight_scale");
        let (n, k) = shape2_of(&weights, &wname);
        let lin = nv_layers::linear::fp8_e4m3_rowscale_checkpoint_dequant_linear(
            &weights,
            module,
            n,
            k,
            DType::BF16,
        )
        .expect("fp8 dequant linear");
        let got = host_f32(lin.weight().expect("bf16 storage"));
        let bytes = weights.raw_bytes(&wname).expect("raw fp8 bytes");
        let scale_t = weights.get(&sname, DType::F32).expect("scale tensor");
        let scale_dims = scale_t.dims().to_vec();
        let scale_vals = host_f32(&scale_t);
        let rows = nv_weights::fp8_row_scales_from(&scale_dims, &scale_vals, n).expect("rows");
        let reference = nv_quant::fp8::dequantize_e4m3_per_row(bytes, n, k, &rows).expect("ref");
        assert_eq!(got.len(), reference.len(), "{module}: element count");
        let mut mismatches = 0usize;
        let mut max_abs = 0f32;
        for (g, r) in got.iter().zip(&reference) {
            let r_bf = half::bf16::from_f32(*r).to_f32();
            let d = (g - r_bf).abs();
            if d != 0.0 {
                mismatches += 1;
                if d > max_abs {
                    max_abs = d;
                }
            }
        }
        eprintln!(
            "[q38-probe-a] basis: checkpoint={} module={module} n={n} k={k} mismatches={mismatches} max_abs_diff={max_abs:.3e}",
            dir.display()
        );
        assert_eq!(
            mismatches, 0,
            "{module}: loader fp8 dequant deviates from host dequantize_e4m3_per_row"
        );
    }
}

#[test]
#[ignore = "runs the native nvfp4 gemm against a host-dequant bf16 gemm on one real mlp tensor; set NV_Q38_TEST=1"]
fn probe_b_native_nvfp4_gemm_forward_matches_host_dequant_bf16_forward_on_real_gate_proj() {
    require_probe_env();
    let dir = snapshot_dir();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let module = "model.language_model.layers.0.mlp.gate_proj";
    let (n, k) = shape2_of(&weights, &format!("{module}.weight_packed"));
    let k = k * 2;
    let dev = match &device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let runner = Arc::new(Mutex::new(
        nv_quant::nvfp4::Nvfp4GemmRunner::new(dev.cuda_stream()).expect("nvfp4 runner"),
    ));
    let native =
        nv_layers::moe::nvfp4_linear_from_disk_pub(&weights, module, n, k, runner, &device)
            .expect("native nvfp4 linear");

    let packed = weights
        .raw_bytes(&format!("{module}.weight_packed"))
        .expect("packed");
    let scales = weights
        .raw_bytes(&format!("{module}.weight_scale"))
        .expect("scales");
    let g = host_f32(
        &weights
            .get(&format!("{module}.weight_global_scale"), DType::F32)
            .expect("global scale"),
    )[0];
    let host_w = nv_quant::nvfp4::dequantize_packed_linear(packed, scales, n, k, 1.0 / g);
    let host_bf: Vec<half::bf16> = host_w.iter().map(|v| half::bf16::from_f32(*v)).collect();
    let dense = nv_layers::linear::Linear::new(
        Tensor::from_vec(host_bf, (n, k), &device).expect("host weight"),
        None,
    )
    .expect("dense linear");

    let rows = 4usize;
    let mut lcg = Lcg(0x9380_27b0_b001);
    let xv: Vec<f32> = (0..rows * k).map(|_| lcg.next_f32()).collect();
    let x = Tensor::from_vec(xv, (rows, k), &device)
        .expect("x")
        .to_dtype(DType::BF16)
        .expect("x bf16");
    let y_native = host_f32(&native.forward(&x).expect("native forward"));
    let y_dense = host_f32(&dense.forward(&x).expect("dense forward"));
    let rms = |v: &[f32]| (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    let dot: f64 = y_native
        .iter()
        .zip(&y_dense)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let rms_n = rms(&y_native);
    let rms_d = rms(&y_dense);
    let cosine = dot / (rms_n * rms_d * y_native.len() as f64).max(1e-30);
    let ratio = rms_n / rms_d.max(1e-30);
    eprintln!(
        "[q38-probe-b] basis: checkpoint={} module={module} n={n} k={k} weight_global_scale={g:.4} rows={rows} rms_native={rms_n:.5e} rms_host_dequant={rms_d:.5e} ratio={ratio:.4} cosine={cosine:.5}",
        dir.display()
    );
    assert!(
        ratio > 0.5 && ratio < 2.0,
        "native nvfp4 output magnitude is {ratio:.3e}x the host-dequant magnitude; a large factor \
         here is the global-scale-semantics defect on this checkpoint"
    );
    assert!(
        cosine > 0.98,
        "native nvfp4 output direction diverges from host dequant (cosine {cosine:.4}); magnitude \
         is fine so suspect the block-scale swizzle or packing order"
    );
}

#[test]
#[ignore = "loads the full 27B once and compares cacheless forward vs stepwise cached decode; set NV_Q38_TEST=1"]
fn probe_d_cacheless_forward_and_stepwise_cached_decode_agree_on_the_final_position() {
    require_probe_env();
    let dir = snapshot_dir();
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw).expect("dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("quant config");
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg.clone(), &weights, &qcfg, &device)
        .expect("build q38 dense");
    drop(weights);

    let bos = cfg.bos_token_id.unwrap_or(cfg.eos_token_id);
    let toks: Vec<u32> = vec![bos, 3, 11, 5, 40, 2, 19, 7];
    let t = toks.len();
    let tokens = Tensor::from_vec(toks.clone(), (1usize, t), &device).expect("tokens");
    let positions =
        Tensor::from_vec((0..t as i32).collect::<Vec<_>>(), t, &device).expect("positions");
    let full = model.forward(&tokens, &positions).expect("cacheless forward");
    let full_rows = host_f32(&full);
    let v = cfg.vocab_size;
    let full_last = &full_rows[full_rows.len() - v..];

    let mut cache = model.new_kv_cache(64).expect("kv cache");
    let mut step_last: Vec<f32> = Vec::new();
    for (i, tk) in toks.iter().enumerate() {
        let tokens = Tensor::from_vec(vec![*tk], (1usize, 1usize), &device).expect("token");
        let positions = Tensor::from_vec(vec![i as i32], 1usize, &device).expect("pos");
        let logits = model
            .forward_with_cache(&tokens, &positions, &mut cache)
            .expect("cached step");
        step_last = host_f32(&logits);
    }
    let step_last = &step_last[step_last.len() - v..];

    let mut worst = 0f32;
    let denom = full_last
        .iter()
        .fold(0f32, |m, x| m.max(x.abs()))
        .max(1e-6);
    for (a, b) in full_last.iter().zip(step_last) {
        worst = worst.max((a - b).abs() / denom);
    }
    let top = |row: &[f32]| {
        let mut idx: Vec<usize> = (0..row.len()).collect();
        idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal));
        idx.truncate(5);
        let vals: Vec<(usize, f32)> = idx.iter().map(|&i| (i, row[i])).collect();
        vals
    };
    eprintln!(
        "[q38-probe-d] basis: checkpoint={} backend=cuda prompt_len={t} worst_rel_vs_absmax={worst:.3e} argmax_full={} argmax_step={} top5_full={:?} top5_step={:?}",
        dir.display(),
        argmax(full_last),
        argmax(step_last),
        top(full_last),
        top(step_last)
    );
    assert!(
        full_last.iter().all(|x| x.is_finite()) && step_last.iter().all(|x| x.is_finite()),
        "non-finite logits"
    );
    assert!(
        worst < 0.3,
        "stepwise cached decode diverges {worst:.3e} from the cacheless forward at the same \
         position; the weights and gemms are shared between both paths, so this isolates the \
         defect to the decode-time cache/kernel path (fp8 kv attention decode or fused gdn state)"
    );
    assert_eq!(
        argmax(full_last),
        argmax(step_last),
        "argmax disagreement between cacheless forward and cached decode"
    );
}
