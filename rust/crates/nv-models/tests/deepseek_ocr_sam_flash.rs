#![cfg(feature = "cuda")]

mod hub_snapshot;

use candle_core::{DType, Device, Tensor};
use nv_models::deepseek_ocr::sam::{sam_prof_report, sam_prof_reset, SamConfig, SamEncoder};
use nv_weights::WeightLoader;
use std::collections::HashMap;

struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as f32 / (1u64 << 31) as f32 - 0.5
    }

    fn tensor(&mut self, shape: &[usize], amp: f32, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| self.next_f32() * amp).collect();
        Tensor::from_vec(v, shape, dev).unwrap()
    }

    fn ones_ish(&mut self, n: usize, dev: &Device) -> Tensor {
        let v: Vec<f32> = (0..n).map(|_| 1.0 + self.next_f32() * 0.1).collect();
        Tensor::from_vec(v, n, dev).unwrap()
    }
}

fn cfg() -> SamConfig {
    SamConfig {
        embed_dim: 128,
        depth: 2,
        num_heads: 2,
        mlp_ratio: 2,
        patch_size: 16,
        window_size: 14,
        global_attn_indexes: vec![1],
        pos_grid: 64,
        ln_eps: 1e-6,
    }
}

fn synth_checkpoint(path: &std::path::Path) {
    let dev = Device::Cpu;
    let mut rng = Rng(0xf1a5_4102);
    let mut t: HashMap<String, Tensor> = HashMap::new();
    let p = "sam.";
    t.insert(
        format!("{p}patch_embed.proj.weight"),
        rng.tensor(&[128, 3, 16, 16], 0.1, &dev),
    );
    t.insert(
        format!("{p}patch_embed.proj.bias"),
        rng.tensor(&[128], 0.1, &dev),
    );
    t.insert(
        format!("{p}pos_embed"),
        rng.tensor(&[1, 64, 64, 128], 0.1, &dev),
    );
    for i in 0..2 {
        let b = format!("{p}blocks.{i}.");
        for nm in ["norm1", "norm2"] {
            t.insert(format!("{b}{nm}.weight"), rng.ones_ish(128, &dev));
            t.insert(format!("{b}{nm}.bias"), rng.tensor(&[128], 0.1, &dev));
        }
        t.insert(
            format!("{b}attn.qkv.weight"),
            rng.tensor(&[384, 128], 0.1, &dev),
        );
        t.insert(format!("{b}attn.qkv.bias"), rng.tensor(&[384], 0.1, &dev));
        t.insert(
            format!("{b}attn.proj.weight"),
            rng.tensor(&[128, 128], 0.1, &dev),
        );
        t.insert(format!("{b}attn.proj.bias"), rng.tensor(&[128], 0.1, &dev));
        t.insert(
            format!("{b}attn.rel_pos_h"),
            rng.tensor(&[127, 64], 1.0, &dev),
        );
        t.insert(
            format!("{b}attn.rel_pos_w"),
            rng.tensor(&[127, 64], 1.0, &dev),
        );
        t.insert(
            format!("{b}mlp.lin1.weight"),
            rng.tensor(&[256, 128], 0.1, &dev),
        );
        t.insert(format!("{b}mlp.lin1.bias"), rng.tensor(&[256], 0.1, &dev));
        t.insert(
            format!("{b}mlp.lin2.weight"),
            rng.tensor(&[128, 256], 0.1, &dev),
        );
        t.insert(format!("{b}mlp.lin2.bias"), rng.tensor(&[128], 0.1, &dev));
    }
    candle_core::safetensors::save(&t, path).unwrap();
}

fn host_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
#[ignore]
fn sam_flash_parity_synthetic() {
    if std::env::var("NV_SAM_FLASH_TEST").map(|v| v == "1") != Ok(true) {
        hub_snapshot::precondition_absent(
            "sam_flash rel-pos parity",
            "NV_SAM_FLASH_TEST != 1",
            "set NV_SAM_FLASH_TEST=1; the DeepSeek-OCR-2 checkpoint IS cached, so this is an opt-in knob, not a missing artifact",
        );
        return;
    }
    std::env::set_var("NV_SAM_PROF", "1");
    std::env::remove_var("NV_SAM_FLASH");
    std::env::remove_var("NV_SAM_FUSED");
    let dev = Device::new_cuda(0).expect("cuda device");
    let dir = std::env::temp_dir().join(format!("dsocr2-sam-flash-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ckpt = dir.join("sam-flash.safetensors");
    synth_checkpoint(&ckpt);
    let weights = WeightLoader::open_file(&ckpt, &dev).unwrap();
    let enc = SamEncoder::from_loader(&weights, "sam.", cfg(), DType::BF16).unwrap();

    let mut rng = Rng(0x0bad_cafe);
    for size in [224usize, 512, 1024] {
        let px = rng
            .tensor(&[1, 3, size, size], 1.0, &Device::Cpu)
            .to_device(&dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let base = host_f32(&enc.forward(&px).unwrap());
        assert!(base.iter().all(|x| x.is_finite()), "base non-finite");

        std::env::set_var("NV_SAM_FUSED", "0");
        let eager = host_f32(&enc.forward(&px).unwrap());
        std::env::remove_var("NV_SAM_FUSED");

        sam_prof_reset();
        std::env::set_var("NV_SAM_FLASH", "1");
        let flash = host_f32(&enc.forward(&px).unwrap());
        std::env::remove_var("NV_SAM_FLASH");
        let prof = sam_prof_report(1.0);
        assert!(
            prof.contains("attn.flash"),
            "flash path did not execute at size {size}; prof:\n{prof}"
        );
        assert!(flash.iter().all(|x| x.is_finite()), "flash non-finite");

        let base2 = host_f32(&enc.forward(&px).unwrap());
        assert_eq!(
            base, base2,
            "gate-off output changed after flash run at size {size}"
        );

        let d_base = max_abs_diff(&base, &eager);
        let d_flash = max_abs_diff(&base, &flash);
        println!(
            "size={size} grid={} max_abs_diff flash_vs_default={d_flash:.6e} eager_vs_default={d_base:.6e}",
            size / 16
        );
        let tol = d_base.max(2e-3) * 8.0;
        assert!(
            d_flash <= tol,
            "size {size}: flash max_abs_diff {d_flash:.6e} exceeds tol {tol:.6e} (baseline fused-vs-eager {d_base:.6e})"
        );
    }
}

#[test]
#[ignore]
fn sam_flash_parity_real_weights() {
    if std::env::var("NV_SAM_FLASH_TEST").map(|v| v == "1") != Ok(true) {
        hub_snapshot::precondition_absent(
            "sam_flash (second arm)",
            "NV_SAM_FLASH_TEST != 1",
            "set NV_SAM_FLASH_TEST=1; the DeepSeek-OCR-2 checkpoint IS cached, so this is an opt-in knob, not a missing artifact",
        );
        return;
    }
    let snap = std::env::var("NV_DSOCR_DIR").unwrap_or_else(|_| {
        format!(
            "{}/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR-2/snapshots/aaa02f3811945a91062062994c5c4a3f4c0af2b0",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    if !std::path::Path::new(&snap)
        .join("model-00001-of-000001.safetensors")
        .exists()
    {
        eprintln!("SKIP: no DeepSeek-OCR-2 snapshot at {snap}");
        return;
    }
    std::env::set_var("NV_SAM_PROF", "1");
    std::env::remove_var("NV_SAM_FLASH");
    std::env::remove_var("NV_SAM_FUSED");
    let dev = Device::new_cuda(0).expect("cuda device");
    let weights = WeightLoader::open_dir(std::path::Path::new(&snap), &dev).expect("weights");
    let enc = SamEncoder::from_loader(
        &weights,
        "model.sam_model.",
        SamConfig::vit_b(),
        DType::BF16,
    )
    .expect("real SAM encoder");

    let mut rng = Rng(0x5eed_0c12);
    for size in [512usize, 1024] {
        let px = rng
            .tensor(&[1, 3, size, size], 1.0, &Device::Cpu)
            .to_device(&dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let base = host_f32(&enc.forward(&px).unwrap());
        assert!(base.iter().all(|x| x.is_finite()), "base non-finite");

        std::env::set_var("NV_SAM_FUSED", "0");
        let eager = host_f32(&enc.forward(&px).unwrap());
        std::env::remove_var("NV_SAM_FUSED");

        sam_prof_reset();
        std::env::set_var("NV_SAM_FLASH", "1");
        let flash = host_f32(&enc.forward(&px).unwrap());
        std::env::remove_var("NV_SAM_FLASH");
        let prof = sam_prof_report(1.0);
        assert!(
            prof.contains("attn.flash"),
            "flash path did not execute at size {size}; prof:\n{prof}"
        );
        assert!(flash.iter().all(|x| x.is_finite()), "flash non-finite");

        let d_base = max_abs_diff(&base, &eager);
        let d_flash = max_abs_diff(&base, &flash);
        println!(
            "REAL size={size} grid={} max_abs_diff flash_vs_default={d_flash:.6e} eager_vs_default={d_base:.6e}",
            size / 16
        );
        let tol = d_base.max(2e-3) * 8.0;
        assert!(
            d_flash <= tol,
            "REAL size {size}: flash {d_flash:.6e} exceeds tol {tol:.6e} (eager baseline {d_base:.6e})"
        );
    }
}
