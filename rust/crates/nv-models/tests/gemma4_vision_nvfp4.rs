#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::gemma4_vision::{Gemma4VisionConfig, Gemma4VisionTower, VisionRopeParams};
use std::path::PathBuf;
use std::time::Instant;

const TEXT_HIDDEN: usize = 256;
const PROJECTIONS_PER_LAYER: usize = 7;
const SYNTHETIC_NVFP4_VS_BF16_COSINE_FLOOR: f32 = 0.995;
const REAL_TOWER_NVFP4_VS_BF16_COSINE_GUARD_MEASURED_0_967_ON_27_LAYERS_TOO_LOW_TO_FLIP_DEFAULT:
    f32 = 0.96;
const TIMED_RUNS_AFTER_ONE_WARMUP: usize = 5;

fn synth_cfg() -> Gemma4VisionConfig {
    Gemma4VisionConfig {
        model_type: Some("gemma4_vision".into()),
        hidden_size: 256,
        intermediate_size: 512,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        head_dim: 64,
        patch_size: 4,
        pooling_kernel_size: 2,
        position_embedding_size: 64,
        default_output_length: 64,
        rms_norm_eps: 1e-6,
        attention_bias: false,
        hidden_activation: "gelu_pytorch_tanh".into(),
        use_clipped_linears: false,
        standardize: false,
        rope_parameters: VisionRopeParams {
            rope_theta: 100.0,
            rope_type: None,
        },
        vision_soft_tokens_per_image: None,
    }
}

fn grid_pixels(grid_w: usize, grid_h: usize, pp: usize, seed: u64) -> Vec<f32> {
    let n = grid_w * grid_h * pp;
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        v.push(((s >> 40) & 0xFFFF) as f32 / 65535.0);
    }
    v
}

fn grid_pos_tensor(grid_w: usize, grid_h: usize, device: &Device) -> Tensor {
    let mut v = Vec::with_capacity(grid_w * grid_h * 2);
    for y in 0..grid_h as i64 {
        for x in 0..grid_w as i64 {
            v.push(x);
            v.push(y);
        }
    }
    Tensor::from_vec(v, (grid_w * grid_h, 2), device).unwrap()
}

fn to_f32_vec(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn cosine_and_rel_l2(a: &Tensor, b: &Tensor) -> (f32, f32) {
    let av = to_f32_vec(a);
    let bv = to_f32_vec(b);
    assert_eq!(av.len(), bv.len(), "output shape drifted between arms");
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut diff = 0f64;
    for (x, y) in av.iter().zip(bv.iter()) {
        assert!(x.is_finite() && y.is_finite(), "non-finite embedding value");
        dot += (x * y) as f64;
        na += (x * x) as f64;
        nb += (y * y) as f64;
        diff += ((x - y) * (x - y)) as f64;
    }
    (
        (dot / (na.sqrt() * nb.sqrt()).max(1e-12)) as f32,
        (diff.sqrt() / na.sqrt().max(1e-12)) as f32,
    )
}

#[test]
fn synthetic_nvfp4_quantize_counts_and_tracks_bf16() {
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let cfg = synth_cfg();
    let pp = cfg.patch_pixels();
    let layers = cfg.num_hidden_layers;
    let mut tower = Gemma4VisionTower::new_synthetic(cfg, TEXT_HIDDEN, &device, DType::BF16)
        .expect("synthetic tower");
    let (gw, gh) = (8usize, 8usize);
    let pixels = grid_pixels(gw, gh, pp, 5);
    let pv = Tensor::from_slice(&pixels, (gw * gh, pp), &device).unwrap();
    let pos = grid_pos_tensor(gw, gh, &device);
    let dense_out = tower.forward(&pv, &pos).expect("bf16 dense forward");

    let quantized = tower.quantize_linears_nvfp4().expect("nvfp4 quantize");
    let input_proj_stays_dense_below_min_tile = 1usize;
    let embedding_projection = 1usize;
    assert_eq!(
        quantized,
        layers * PROJECTIONS_PER_LAYER + embedding_projection,
        "expected every layer projection plus the embedding projection quantized, with the \
         {input_proj_stays_dense_below_min_tile} sub-MIN_TILE input_proj left dense"
    );

    let quant_out = tower.forward(&pv, &pos).expect("nvfp4 forward");
    let (cos, rel_l2) = cosine_and_rel_l2(&dense_out, &quant_out);
    eprintln!("[gemma4-vision-nvfp4 synthetic] cosine={cos} rel_l2={rel_l2}");
    assert!(
        cos >= SYNTHETIC_NVFP4_VS_BF16_COSINE_FLOOR,
        "nvfp4 vision output drifted from bf16: cosine {cos} < {SYNTHETIC_NVFP4_VS_BF16_COSINE_FLOOR} (rel_l2 {rel_l2})"
    );
}

#[test]
fn synthetic_ragged_intermediate_pads_k_and_still_tracks_bf16() {
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let mut cfg = synth_cfg();
    cfg.intermediate_size = 272;
    let pp = cfg.patch_pixels();
    let layers = cfg.num_hidden_layers;
    let mut tower = Gemma4VisionTower::new_synthetic(cfg, TEXT_HIDDEN, &device, DType::BF16)
        .expect("synthetic tower");
    let (gw, gh) = (8usize, 8usize);
    let pixels = grid_pixels(gw, gh, pp, 17);
    let pv = Tensor::from_slice(&pixels, (gw * gh, pp), &device).unwrap();
    let pos = grid_pos_tensor(gw, gh, &device);
    let dense_out = tower.forward(&pv, &pos).expect("bf16 dense forward");

    let quantized = tower.quantize_linears_nvfp4().expect("nvfp4 quantize");
    let embedding_projection = 1usize;
    assert_eq!(
        quantized,
        layers * PROJECTIONS_PER_LAYER + embedding_projection,
        "the 272-wide down_proj must quantize via k-padding, not fall back to dense"
    );

    let quant_out = tower.forward(&pv, &pos).expect("nvfp4 k-padded forward");
    let (cos, rel_l2) = cosine_and_rel_l2(&dense_out, &quant_out);
    eprintln!("[gemma4-vision-nvfp4 synthetic-ragged] cosine={cos} rel_l2={rel_l2}");
    assert!(
        cos >= SYNTHETIC_NVFP4_VS_BF16_COSINE_FLOOR,
        "k-padded nvfp4 output drifted from bf16: cosine {cos} < \
         {SYNTHETIC_NVFP4_VS_BF16_COSINE_FLOOR} (rel_l2 {rel_l2})"
    );
}

#[test]
fn quantize_refuses_f32_activation_tower() {
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let mut tower =
        Gemma4VisionTower::new_synthetic(synth_cfg(), TEXT_HIDDEN, &device, DType::F32)
            .expect("synthetic f32 tower");
    let e = tower
        .quantize_linears_nvfp4()
        .expect_err("f32 tower must be refused");
    assert!(
        e.to_string().contains("bf16/f16"),
        "refusal must name the bf16/f16 requirement, got: {e:#}"
    );
}

fn gemma4_mm_dir() -> PathBuf {
    for key in ["NV_G4_MM_DIR", "NV_CHAT_MODEL_DIR"] {
        if let Ok(d) = std::env::var(key) {
            if d.is_empty() {
                continue;
            }
            let p = PathBuf::from(&d);
            let cfg = p.join("config.json");
            let has_vision = std::fs::read_to_string(&cfg)
                .map(|s| s.contains("\"vision_config\""))
                .unwrap_or(false);
            if has_vision {
                return p;
            }
        }
    }
    let root = PathBuf::from(std::env::var("HOME").expect("HOME"))
        .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
    std::fs::read_dir(&root)
        .ok()
        .and_then(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .find(|p| p.join("config.json").is_file())
        })
        .expect(
            "no gemma4 mm checkpoint: NV_G4_MM_DIR / NV_CHAT_MODEL_DIR lack a vision_config and \
             the hub cache has no nvidia/Gemma-4-31B-IT-NVFP4 snapshot",
        )
}

fn checkpoint_label(dir: &PathBuf) -> String {
    let rev = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into());
    let repo = dir
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .find(|c| c.starts_with("models--"))
        .map(|c| c.trim_start_matches("models--").replace("--", "/"))
        .unwrap_or_else(|| "local/gemma4-mm".into());
    format!("{repo}@{rev}")
}

fn median_ms(runs: &mut [f64]) -> f64 {
    runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    runs[runs.len() / 2]
}

fn timed_forward_ms(tower: &Gemma4VisionTower, pv: &Tensor, pos: &Tensor) -> (f64, Tensor) {
    let mut out = tower.forward(pv, pos).expect("warmup forward");
    let mut runs = Vec::with_capacity(TIMED_RUNS_AFTER_ONE_WARMUP);
    for _ in 0..TIMED_RUNS_AFTER_ONE_WARMUP {
        let t0 = Instant::now();
        out = tower.forward(pv, pos).expect("timed forward");
        let _sync = out
            .to_dtype(DType::F32)
            .and_then(|x| x.sum_all())
            .and_then(|x| x.to_scalar::<f32>())
            .expect("device sync via scalar readback");
        runs.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    (median_ms(&mut runs), out)
}

#[test]
#[ignore = "loads the real gemma4 mm vision tower; NV_GEMMA4_VISION_NVFP4_TEST=1 opts in; \
            NV_MM_VISION_DTYPE picks the bf16 or nvfp4 arm"]
fn real_tower_nvfp4_ab_encode_and_quality() {
    if std::env::var("NV_GEMMA4_VISION_NVFP4_TEST").ok().as_deref() != Some("1") {
        panic!(
            "real_tower_nvfp4_ab_encode_and_quality is #[ignore]d and was asked for by name, \
             but NV_GEMMA4_VISION_NVFP4_TEST=1 is not set. This is a SKIP, not a pass."
        );
    }
    let arm = std::env::var("NV_MM_VISION_DTYPE").unwrap_or_else(|_| "bf16".into());
    assert!(
        arm == "bf16" || arm == "nvfp4",
        "NV_MM_VISION_DTYPE={arm}: this A/B test only measures the bf16 and nvfp4 arms"
    );
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let dir = gemma4_mm_dir();
    let label = checkpoint_label(&dir);
    let cfg =
        Gemma4VisionConfig::from_hf_json_file(&dir.join("config.json")).expect("vision cfg");
    let pp = cfg.patch_pixels();
    let pk = cfg.pooling_kernel_size;
    let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("open weights");
    let mut tower =
        Gemma4VisionTower::load(cfg, &weights, &device, DType::BF16).expect("load tower bf16");

    let side = 16 * pk;
    let (gw, gh) = (side, side);
    let n = gw * gh;
    let pixels = grid_pixels(gw, gh, pp, 9);
    let pv = Tensor::from_slice(&pixels, (n, pp), &device).unwrap();
    let pos = grid_pos_tensor(gw, gh, &device);

    if arm == "nvfp4" {
        let dense_out = tower.forward(&pv, &pos).expect("bf16 reference forward");
        let quantized = tower.quantize_linears_nvfp4().expect("nvfp4 quantize");
        let input_proj_and_embedding_projection = 2usize;
        assert_eq!(
            quantized,
            tower.num_layers() * PROJECTIONS_PER_LAYER + input_proj_and_embedding_projection,
            "a projection fell back to dense on this box; that is a perf regression, not a pass"
        );
        let (ms, quant_out) = timed_forward_ms(&tower, &pv, &pos);
        let (cos, rel_l2) = cosine_and_rel_l2(&dense_out, &quant_out);
        eprintln!(
            "[gemma4-vision-nvfp4 real-ab] quantized={quantized} cosine={cos} rel_l2={rel_l2}"
        );
        println!(
            "CTX-PREFILL depth={n} prefill_s={:.6} prefill_tok_s={:.1} checkpoint={label}",
            ms / 1e3,
            n as f64 / (ms / 1e3)
        );
        let guard =
            REAL_TOWER_NVFP4_VS_BF16_COSINE_GUARD_MEASURED_0_967_ON_27_LAYERS_TOO_LOW_TO_FLIP_DEFAULT;
        assert!(
            cos >= guard,
            "real tower nvfp4 drifted below its recorded band: cosine {cos} < {guard} \
             (rel_l2 {rel_l2}); the opt-in path regressed, it was never flip-quality"
        );
    } else {
        let (ms, out) = timed_forward_ms(&tower, &pv, &pos);
        let sum = out
            .to_dtype(DType::F32)
            .and_then(|x| x.sum_all())
            .and_then(|x| x.to_scalar::<f32>())
            .expect("readback");
        assert!(sum.is_finite(), "bf16 tower output must stay finite");
        println!(
            "CTX-PREFILL depth={n} prefill_s={:.6} prefill_tok_s={:.1} checkpoint={label}",
            ms / 1e3,
            n as f64 / (ms / 1e3)
        );
    }
}
