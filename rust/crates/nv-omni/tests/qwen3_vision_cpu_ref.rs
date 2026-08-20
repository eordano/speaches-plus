use candle_core::{DType, Device, Tensor};
use nv_omni::{Qwen3VisionConfig, Qwen3VisionTower};
use std::collections::HashMap;

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
    fn vec(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * s).collect()
    }
}

fn tiny_cfg() -> Qwen3VisionConfig {
    Qwen3VisionConfig {
        depth: 1,
        hidden_size: 8,
        num_heads: 2,
        intermediate_size: 12,
        in_channels: 3,
        patch_size: 2,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        num_position_embeddings: 4,
        out_hidden_size: 6,
        layer_norm_eps: 1e-6,
        dtype: DType::F32,
    }
}

fn write_tiny_visual_checkpoint(cfg: &Qwen3VisionConfig, seed: u64) -> (std::path::PathBuf, HashMap<String, Vec<f32>>) {
    let mut r = Lcg::new(seed);
    let mut host: HashMap<String, Vec<f32>> = HashMap::new();
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (name, shape) in cfg.expected_checkpoint_tensor_names_with_shapes() {
        let n: usize = shape.iter().product();
        let scale = if name.ends_with("norm1.weight")
            || name.ends_with("norm2.weight")
            || name.ends_with("norm.weight")
        {
            0.2
        } else {
            0.35
        };
        let mut vals = r.vec(n, scale);
        if name.ends_with("norm1.weight")
            || name.ends_with("norm2.weight")
            || name.ends_with("norm.weight")
        {
            for v in &mut vals {
                *v += 1.0;
            }
        }
        tensors.insert(
            name.clone(),
            Tensor::from_vec(vals.clone(), shape.as_slice(), &Device::Cpu).unwrap(),
        );
        host.insert(name, vals);
    }
    let dir = std::env::temp_dir().join(format!("q38-vision-cpu-ref-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
    (dir, host)
}

fn layernorm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let denom = (var + eps).sqrt();
    x.iter()
        .zip(w.iter().zip(b.iter()))
        .map(|(v, (wi, bi))| (v - mean) / denom * wi + bi)
        .collect()
}

fn matvec(w: &[f32], bias: Option<&[f32]>, x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| {
            let dot: f32 = (0..cols).map(|c| w[r * cols + c] * x[c]).sum();
            dot + bias.map(|b| b[r]).unwrap_or(0.0)
        })
        .collect()
}

fn gelu_tanh(x: f32) -> f32 {
    let k = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (k * (x + 0.044715 * x * x * x)).tanh())
}

const TINY_IMAGE_SIDE_4_GIVES_A_2X2_GRID_MATCHING_THE_POS_EMBED_SIDE_SO_INTERPOLATION_IS_IDENTITY:
    usize = 4;

#[test]
fn tiny_tower_forward_matches_a_naive_cpu_reference_including_the_2d_rope() {
    let cfg = tiny_cfg();
    let (dir, host) = write_tiny_visual_checkpoint(&cfg, 0x51_0e_11);
    let device = Device::Cpu;
    let weights = nv_weights::WeightLoader::open_dir(&dir, &device).expect("open tiny visual dir");
    let mut tower = Qwen3VisionTower::new_empty(cfg.clone(), &device).expect("new_empty");
    tower
        .load_weights(&weights)
        .expect("load exactly the expected tensor tree");

    let side = TINY_IMAGE_SIDE_4_GIVES_A_2X2_GRID_MATCHING_THE_POS_EMBED_SIDE_SO_INTERPOLATION_IS_IDENTITY;
    let mut r = Lcg::new(0x9121);
    let px = r.vec(3 * side * side, 0.8);
    let img = Tensor::from_vec(px.clone(), (1, 3, side, side), &device).unwrap();
    let got: Vec<f32> = tower
        .forward(&img)
        .expect("tiny forward")
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let h = cfg.hidden_size;
    let hd = cfg.head_dim();
    let nh = cfg.num_heads;
    let gh = side / cfg.patch_size;
    let gw = side / cfg.patch_size;
    let n_patches = gh * gw;
    let g = |k: &str| -> &Vec<f32> { &host[&format!("model.visual.{k}")] };

    let pe_w = g("patch_embed.proj.weight");
    let pe_b = g("patch_embed.proj.bias");
    let p = cfg.patch_size;
    let tp = cfg.temporal_patch_size;
    let mut x: Vec<Vec<f32>> = Vec::new();
    for pr in 0..gh {
        for pc in 0..gw {
            let mut row = vec![0f32; h];
            for (oc, slot) in row.iter_mut().enumerate() {
                let mut acc = pe_b[oc];
                for ci in 0..3 {
                    for ti in 0..tp {
                        for yy in 0..p {
                            for xx in 0..p {
                                let wi = ((((oc * 3 + ci) * tp + ti) * p + yy) * p) + xx;
                                let pxv = px[ci * side * side + (pr * p + yy) * side + (pc * p + xx)];
                                acc += pe_w[wi] * pxv;
                            }
                        }
                    }
                }
                *slot = acc;
            }
            x.push(row);
        }
    }
    let pos = g("pos_embed.weight");
    for (pi, row) in x.iter_mut().enumerate() {
        for (oc, v) in row.iter_mut().enumerate() {
            *v += pos[pi * h + oc];
        }
    }

    let half = hd / 2;
    let quarter = hd / 4;
    let theta = 10_000f32;
    let inv: Vec<f32> = (0..quarter)
        .map(|j| 1.0 / theta.powf((2 * j) as f32 / half as f32))
        .collect();
    let mut cos_rows = vec![vec![0f32; half]; n_patches];
    let mut sin_rows = vec![vec![0f32; half]; n_patches];
    for rr in 0..gh {
        for cc in 0..gw {
            let pi = rr * gw + cc;
            for j in 0..quarter {
                cos_rows[pi][j] = (rr as f32 * inv[j]).cos();
                sin_rows[pi][j] = (rr as f32 * inv[j]).sin();
                cos_rows[pi][quarter + j] = (cc as f32 * inv[j]).cos();
                sin_rows[pi][quarter + j] = (cc as f32 * inv[j]).sin();
            }
        }
    }

    let eps = cfg.layer_norm_eps as f32;
    let n1w = g("blocks.0.norm1.weight");
    let n1b = g("blocks.0.norm1.bias");
    let qkv_w = g("blocks.0.attn.qkv.weight");
    let qkv_b = g("blocks.0.attn.qkv.bias");
    let mut q_all = vec![vec![0f32; nh * hd]; n_patches];
    let mut k_all = vec![vec![0f32; nh * hd]; n_patches];
    let mut v_all = vec![vec![0f32; nh * hd]; n_patches];
    for (pi, row) in x.iter().enumerate() {
        let n1 = layernorm(row, n1w, n1b, eps);
        let qkv = matvec(qkv_w, Some(qkv_b), &n1, 3 * h, h);
        for which in 0..3 {
            for head in 0..nh {
                for d in 0..hd {
                    let val = qkv[which * h + head * hd + d];
                    match which {
                        0 => q_all[pi][head * hd + d] = val,
                        1 => k_all[pi][head * hd + d] = val,
                        _ => v_all[pi][head * hd + d] = val,
                    }
                }
            }
        }
    }
    let roped = |all: &Vec<Vec<f32>>| -> Vec<Vec<f32>> {
        let mut out = all.clone();
        for pi in 0..n_patches {
            for head in 0..nh {
                for j in 0..half {
                    let lo = all[pi][head * hd + j];
                    let hi = all[pi][head * hd + half + j];
                    let (c, s) = (cos_rows[pi][j], sin_rows[pi][j]);
                    out[pi][head * hd + j] = lo * c - hi * s;
                    out[pi][head * hd + half + j] = lo * s + hi * c;
                }
            }
        }
        out
    };
    let q_all = roped(&q_all);
    let k_all = roped(&k_all);

    let scale = 1.0 / (hd as f32).sqrt();
    let mut attn_ctx = vec![vec![0f32; nh * hd]; n_patches];
    for head in 0..nh {
        for qi in 0..n_patches {
            let mut scores: Vec<f32> = (0..n_patches)
                .map(|ki| {
                    (0..hd)
                        .map(|d| q_all[qi][head * hd + d] * k_all[ki][head * hd + d])
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0f32;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                denom += *s;
            }
            for d in 0..hd {
                attn_ctx[qi][head * hd + d] = (0..n_patches)
                    .map(|ki| scores[ki] / denom * v_all[ki][head * hd + d])
                    .sum();
            }
        }
    }
    let pj_w = g("blocks.0.attn.proj.weight");
    let pj_b = g("blocks.0.attn.proj.bias");
    for (pi, row) in x.iter_mut().enumerate() {
        let o = matvec(pj_w, Some(pj_b), &attn_ctx[pi], h, h);
        for (a, b) in row.iter_mut().zip(o.iter()) {
            *a += b;
        }
    }

    let n2w = g("blocks.0.norm2.weight");
    let n2b = g("blocks.0.norm2.bias");
    let f1w = g("blocks.0.mlp.linear_fc1.weight");
    let f1b = g("blocks.0.mlp.linear_fc1.bias");
    let f2w = g("blocks.0.mlp.linear_fc2.weight");
    let f2b = g("blocks.0.mlp.linear_fc2.bias");
    for row in x.iter_mut() {
        let n2 = layernorm(row, n2w, n2b, eps);
        let mut h1 = matvec(f1w, Some(f1b), &n2, cfg.intermediate_size, h);
        for v in &mut h1 {
            *v = gelu_tanh(*v);
        }
        let h2 = matvec(f2w, Some(f2b), &h1, h, cfg.intermediate_size);
        for (a, b) in row.iter_mut().zip(h2.iter()) {
            *a += b;
        }
    }

    let mnw = g("merger.norm.weight");
    let mnb = g("merger.norm.bias");
    let group = cfg.spatial_merge_size * cfg.spatial_merge_size;
    assert_eq!(n_patches, group, "one merged token for the 2x2 tiny grid");
    let mut flat = Vec::with_capacity(group * h);
    for row in &x {
        flat.extend(layernorm(row, mnw, mnb, eps));
    }
    let m1w = g("merger.linear_fc1.weight");
    let m1b = g("merger.linear_fc1.bias");
    let m2w = g("merger.linear_fc2.weight");
    let m2b = g("merger.linear_fc2.bias");
    let merged_hidden = cfg.merger_hidden();
    let mut y = matvec(m1w, Some(m1b), &flat, merged_hidden, merged_hidden);
    for v in &mut y {
        *v = gelu_tanh(*v);
    }
    let want = matvec(m2w, Some(m2b), &y, cfg.out_hidden_size, merged_hidden);

    assert_eq!(got.len(), want.len(), "one [out_hidden] merged token");
    let mut worst = 0f32;
    for (i, (gv, wv)) in got.iter().zip(want.iter()).enumerate() {
        let rel = (gv - wv).abs() / wv.abs().max(1e-3);
        worst = worst.max(rel);
        assert!(
            rel < 1e-3,
            "element {i}: tower {gv} vs reference {wv} (rel {rel:.3e})"
        );
    }
    eprintln!("[q38-vision-cpu-ref] worst_rel={worst:.3e} over {} outputs", want.len());

    let mut rot_px = px.clone();
    rot_px.swap(0, 3 * side * side - 1);
    let img2 = Tensor::from_vec(rot_px, (1, 3, side, side), &device).unwrap();
    let got2: Vec<f32> = tower
        .forward(&img2)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(
        got.iter().zip(got2.iter()).any(|(a, b)| (a - b).abs() > 1e-6),
        "moving a pixel must move the embedding"
    );
}
