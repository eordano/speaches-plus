#![cfg(feature = "wgpu")]

mod hub_snapshot;

use candle_core::{DType, Device, Tensor};
use nv_omni::{Qwen3VisionConfig, Qwen3VisionTower};
use nv_weights::WeightLoader;
use std::path::PathBuf;

const REPO: &str = "unsloth/Qwen3.8-27B-NVFP4";
const VISION_GATE_ENV: &str = "NV_QWEN38_VISION_TEST";

fn require_vision_gate() {
    if std::env::var(VISION_GATE_ENV).as_deref() != Ok("1") {
        panic!(
            "set {VISION_GATE_ENV}=1 to run the real-weights Qwen3.8-27B vision tower suite \
             (it must never silently skip)"
        );
    }
}

fn snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN38_DIR") {
        if !d.is_empty() {
            let p = PathBuf::from(&d);
            assert!(p.is_dir(), "NV_QWEN38_DIR={d} is not a directory");
            return p;
        }
    }
    hub_snapshot::snapshot_of(REPO, &["config.json", "*.safetensors"]).unwrap_or_else(|| {
        panic!(
            "no complete {REPO} snapshot under {:?}; set NV_QWEN38_DIR",
            hub_snapshot::hub_roots()
        )
    })
}

fn tower_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::new_cuda(0).expect("cuda device for the vision tower")
    }
    #[cfg(not(feature = "cuda"))]
    {
        Device::Cpu
    }
}

fn synthetic_photo(h: usize, w: usize, device: &Device) -> Tensor {
    let mut pixels = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let fr = (x as f32 / w as f32 + y as f32 / h as f32) * 0.5;
            let fg = ((x * 3 + y) % 251) as f32 / 251.0;
            let fb = ((x + y * 2) % 241) as f32 / 241.0;
            let ch = [fr, fg, fb];
            for (c, v) in ch.iter().enumerate() {
                pixels[c * h * w + y * w + x] = (v - 0.5) / 0.5;
            }
        }
    }
    Tensor::from_vec(pixels, (1, 3, h, w), device).expect("photo tensor")
}

fn load_tower(dir: &PathBuf, device: &Device) -> Qwen3VisionTower {
    let cfg = Qwen3VisionConfig::from_hf_config_json(dir.join("config.json"))
        .expect("parse vision_config from real config.json");
    assert_eq!(cfg.out_hidden_size, 5120, "27B tower out_hidden_size");
    let weights = WeightLoader::open_dir(dir, device).expect("open snapshot weights");
    let mut tower = Qwen3VisionTower::new_empty(cfg, device).expect("new_empty");
    tower
        .load_weights(&weights)
        .expect("load_weights: every model.visual.* tensor must resolve");
    tower
}

fn assert_non_garbage(out: &Tensor, rows: usize) {
    let d = out.dims();
    assert_eq!(d, &[rows, 5120], "merged token grid");
    let v: Vec<f32> = out
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(v.iter().all(|x| x.is_finite()), "all finite");
    for r in 0..rows {
        let row = &v[r * 5120..(r + 1) * 5120];
        let mean = row.iter().sum::<f32>() / 5120.0;
        let var = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 5120.0;
        assert!(var.sqrt() > 1e-3, "row {r} std must exceed 1e-3");
    }
    for k in 0..32 {
        let a = (k * 7) % rows;
        let b = (k * 13 + 1) % rows;
        if a == b {
            continue;
        }
        let ra = &v[a * 5120..a * 5120 + 5120];
        let rb = &v[b * 5120..b * 5120 + 5120];
        let diff = ra
            .iter()
            .zip(rb.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(diff > 1e-3, "rows {a},{b} must differ");
    }
}

#[test]
#[ignore]
fn qwen38_vision_tower_real_weights_forward() {
    require_vision_gate();
    let dir = snapshot_dir();
    let device = tower_device();
    let tower = load_tower(&dir, &device);

    let square = synthetic_photo(448, 448, &device);
    let out = tower.forward(&square).expect("448x448 forward");
    assert_non_garbage(&out, 196);

    let wide = synthetic_photo(448, 672, &device);
    let out_wide = tower.forward(&wide).expect("448x672 forward");
    assert_non_garbage(&out_wide, 294);
    eprintln!("[qwen38-vision] 448x448 -> [196,5120], 448x672 -> [294,5120] OK");
}

mod splice {
    use nv_models::qwen3_5_dense_wgpu as q3d;
    use nv_models::qwen3_5_dense_wgpu::{ImageRowSplice, Qwen3_5DenseConfig, Qwen3_5DenseWgpu};
    use nv_models::qwen3_5_moe::LayerType;
    use nv_models::qwen3_5_moe_wgpu::HostBf16Lin;

    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as u32) as f32 / (1u64 << 31) as f32 - 1.0
        }
        fn bf16_vec(&mut self, n: usize, s: f32) -> Vec<u16> {
            (0..n)
                .map(|_| half::bf16::from_f32(self.next_f32() * s).to_bits())
                .collect()
        }
        fn f32_vec(&mut self, n: usize, s: f32) -> Vec<f32> {
            (0..n)
                .map(|_| half::bf16::from_f32(self.next_f32() * s).to_f32())
                .collect()
        }
    }

    fn cfg() -> Qwen3_5DenseConfig {
        Qwen3_5DenseConfig {
            hidden_size: 128,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 32,
            intermediate_size: 96,
            vocab_size: 64,
            max_position_embeddings: 128,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            partial_rotary_factor: 0.25,
            bos_token_id: None,
            eos_token_id: 1,
            layer_types: vec![
                LayerType::LinearAttention,
                LayerType::LinearAttention,
                LayerType::LinearAttention,
                LayerType::FullAttention,
            ],
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            linear_key_head_dim: 16,
            linear_value_head_dim: 16,
            linear_conv_kernel_dim: 4,
            attn_output_gate: true,
            tie_word_embeddings: false,
        }
    }

    fn bf16_lin(r: &mut Lcg, n: usize, k: usize, s: f32) -> HostBf16Lin {
        HostBf16Lin {
            w: r.bf16_vec(n * k, s),
            n,
            k,
        }
    }
    fn norm_vec(r: &mut Lcg, n: usize) -> Vec<u16> {
        (0..n)
            .map(|_| half::bf16::from_f32(1.0 + 0.1 * r.next_f32()).to_bits())
            .collect()
    }

    fn weights(c: &Qwen3_5DenseConfig, seed: u64) -> q3d::HostDenseWeights {
        let mut r = Lcg::new(seed);
        let hidden = c.hidden_size;
        let inter = c.intermediate_size;
        let hd = c.head_dim;
        let n_k = c.linear_num_key_heads;
        let n_v = c.linear_num_value_heads;
        let d_k = c.linear_key_head_dim;
        let d_v = c.linear_value_head_dim;
        let key_dim = n_k * d_k;
        let value_dim = n_v * d_v;
        let conv_dim = 2 * key_dim + value_dim;
        let ks = c.linear_conv_kernel_dim;
        let mut layers = Vec::new();
        for li in 0..c.num_hidden_layers {
            let mixer = match c.layer_types[li] {
                LayerType::LinearAttention => {
                    q3d::HostDenseMixer::Delta(Box::new(nv_models::qwen3_5_moe_wgpu::HostDeltaNet {
                        in_proj_qkv: bf16_lin(&mut r, conv_dim, hidden, 0.12),
                        in_proj_z: bf16_lin(&mut r, value_dim, hidden, 0.12),
                        in_proj_ab: bf16_lin(&mut r, 2 * n_v, hidden, 0.12),
                        conv1d: r.f32_vec(conv_dim * ks, 0.4),
                        a_log: r.f32_vec(n_v, 0.5),
                        dt_bias: r.f32_vec(n_v, 0.5),
                        norm_w: norm_vec(&mut r, d_v),
                        out_proj: bf16_lin(&mut r, hidden, value_dim, 0.12),
                    }))
                }
                LayerType::FullAttention => {
                    let q_out = c.num_attention_heads * hd * 2;
                    let kv_out = c.num_key_value_heads * hd;
                    q3d::HostDenseMixer::Attn(Box::new(q3d::HostDenseAttention {
                        q: bf16_lin(&mut r, q_out, hidden, 0.12).into(),
                        k: bf16_lin(&mut r, kv_out, hidden, 0.12).into(),
                        v: bf16_lin(&mut r, kv_out, hidden, 0.12).into(),
                        o: bf16_lin(&mut r, hidden, c.num_attention_heads * hd, 0.12).into(),
                        q_norm: norm_vec(&mut r, hd),
                        k_norm: norm_vec(&mut r, hd),
                    }))
                }
            };
            layers.push(q3d::HostDenseLayer {
                input_ln: norm_vec(&mut r, hidden),
                post_attn_ln: norm_vec(&mut r, hidden),
                mixer,
                mlp: q3d::HostDenseMlp {
                    gate: bf16_lin(&mut r, inter, hidden, 0.15).into(),
                    up: bf16_lin(&mut r, inter, hidden, 0.15).into(),
                    down: bf16_lin(&mut r, hidden, inter, 0.15).into(),
                },
                delta_fp8: q3d::DeltaFp8 {
                    qkv: None,
                    z: None,
                    out: None,
                },
            });
        }
        q3d::HostDenseWeights {
            embed: r.bf16_vec(c.vocab_size * hidden, 0.6),
            final_norm: norm_vec(&mut r, hidden),
            lm_head: r.bf16_vec(c.vocab_size * hidden, 0.2),
            layers,
        }
    }

    fn have_gpu() -> bool {
        nv_kernels::wgpu_backend::WgpuContext::shared().is_ok()
    }

    fn embed_row(w: &q3d::HostDenseWeights, tok: u32, hidden: usize) -> Vec<u16> {
        let s = tok as usize * hidden;
        w.embed[s..s + hidden].to_vec()
    }

    #[test]
    fn splice_prefill_is_bit_identical_to_text_gather() {
        if !have_gpu() {
            eprintln!("[skip] no wgpu adapter");
            return;
        }
        let c = cfg();
        let hidden = c.hidden_size;
        let w = weights(&c, 7);
        let mut gpu = Qwen3_5DenseWgpu::new(c.clone(), &w, 128).expect("build");
        let chunk = gpu.prefill_chunk_len();
        assert!(chunk >= 2, "need chunked prefill graph for the splice oracle");
        let t: Vec<u32> = (0..(2 * chunk + 3)).map(|i| (i as u32 * 5 + 3) % 60).collect();
        let last = *t.last().unwrap();
        let rest = &t[..t.len() - 1];

        gpu.reset().unwrap();
        let done_a = gpu.prefill_tokens(rest).unwrap();
        for tk in &rest[done_a..] {
            gpu.prefill_step(*tk).unwrap();
        }
        let (_, logits_a) = gpu.decode_step_logits(last).unwrap();

        let positions = [rest.len() - 3, rest.len() - 2, rest.len() - 1];
        let mut sp = Vec::new();
        for &p in &positions {
            sp.push(ImageRowSplice {
                position: p,
                rows_bf16: embed_row(&w, rest[p], hidden),
            });
        }
        gpu.reset().unwrap();
        let done_b = gpu.prefill_tokens_with_image_rows(rest, &sp).unwrap();
        assert_eq!(done_b, rest.len(), "splice prefill consumes all tokens");
        let (_, logits_b) = gpu.decode_step_logits(last).unwrap();
        for (i, (x, y)) in logits_a.iter().zip(logits_b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "logit {i} must be bit-identical when splice rows equal the gathered embeddings"
            );
        }

        let mut perturbed = sp;
        for row in &mut perturbed {
            for wd in &mut row.rows_bf16 {
                *wd = half::bf16::from_f32(half::bf16::from_bits(*wd).to_f32() + 0.5).to_bits();
            }
        }
        gpu.reset().unwrap();
        gpu.prefill_tokens_with_image_rows(rest, &perturbed).unwrap();
        let (_, logits_c) = gpu.decode_step_logits(last).unwrap();
        let differ = logits_a
            .iter()
            .zip(logits_c.iter())
            .any(|(x, y)| x.to_bits() != y.to_bits());
        assert!(differ, "perturbed splice rows must change the logits");
    }
}
