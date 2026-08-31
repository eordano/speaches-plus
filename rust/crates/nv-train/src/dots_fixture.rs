use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

use crate::dots::{
    dots_layer_prefix, write_dump_sample, DOTS_AFFINE_GROUP_SIZE, DOTS_EMBED_MODULE,
    DOTS_FINAL_NORM, DOTS_LM_HEAD_MODULE,
};

pub const HIDDEN: usize = 64;
pub const LAYERS: usize = 2;
pub const HEADS: usize = 4;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = 16;
pub const INTER: usize = 128;
pub const VOCAB: usize = 96;

pub struct Rng(pub u64);

impl Rng {
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

pub fn quantize_affine_u4(
    w: &[f32],
    n: usize,
    k: usize,
) -> (Vec<u32>, Vec<half::bf16>, Vec<half::bf16>) {
    let gs = DOTS_AFFINE_GROUP_SIZE;
    let groups = k / gs;
    let mut words = vec![0u32; n * k / 8];
    let mut scales = Vec::with_capacity(n * groups);
    let mut biases = Vec::with_capacity(n * groups);
    for r in 0..n {
        for g in 0..groups {
            let seg = &w[r * k + g * gs..r * k + (g + 1) * gs];
            let mn = seg.iter().cloned().fold(f32::INFINITY, f32::min);
            let mx = seg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let scale_f = ((mx - mn) / 15.0).max(1e-6);
            let scale = half::bf16::from_f32(scale_f);
            let bias = half::bf16::from_f32(mn);
            let (sf, bf) = (scale.to_f32(), bias.to_f32());
            scales.push(scale);
            biases.push(bias);
            for (j, &x) in seg.iter().enumerate() {
                let q = (((x - bf) / sf).round().clamp(0.0, 15.0)) as u32;
                let e = g * gs + j;
                words[r * (k / 8) + e / 8] |= q << (4 * (e % 8));
            }
        }
    }
    (words, scales, biases)
}

fn push_affine(
    tensors: &mut HashMap<String, Tensor>,
    module: &str,
    n: usize,
    k: usize,
    rng: &mut Rng,
) -> Result<()> {
    let dev = Device::Cpu;
    let w: Vec<f32> = (0..n * k).map(|_| rng.next_f32() * 0.2).collect();
    let (words, scales, biases) = quantize_affine_u4(&w, n, k);
    tensors.insert(
        format!("{module}.weight"),
        Tensor::from_vec(words, (n, k / 8), &dev)?,
    );
    tensors.insert(
        format!("{module}.scales"),
        Tensor::from_vec(scales, (n, k / DOTS_AFFINE_GROUP_SIZE), &dev)?.to_dtype(DType::BF16)?,
    );
    tensors.insert(
        format!("{module}.biases"),
        Tensor::from_vec(biases, (n, k / DOTS_AFFINE_GROUP_SIZE), &dev)?.to_dtype(DType::BF16)?,
    );
    Ok(())
}

fn push_bf16_vec(tensors: &mut HashMap<String, Tensor>, name: &str, v: Vec<f32>) -> Result<()> {
    let n = v.len();
    tensors.insert(
        name.to_string(),
        Tensor::from_vec(v, n, &Device::Cpu)?.to_dtype(DType::BF16)?,
    );
    Ok(())
}

pub fn write_tiny_checkpoint(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut rng = Rng(7);
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let q_out = HEADS * HEAD_DIM;
    let kv_out = KV_HEADS * HEAD_DIM;
    for li in 0..LAYERS {
        let p = dots_layer_prefix(li);
        push_affine(
            &mut tensors,
            &format!("{p}.self_attn.q_proj"),
            q_out,
            HIDDEN,
            &mut rng,
        )?;
        push_affine(
            &mut tensors,
            &format!("{p}.self_attn.k_proj"),
            kv_out,
            HIDDEN,
            &mut rng,
        )?;
        push_affine(
            &mut tensors,
            &format!("{p}.self_attn.v_proj"),
            kv_out,
            HIDDEN,
            &mut rng,
        )?;
        push_affine(
            &mut tensors,
            &format!("{p}.self_attn.o_proj"),
            HIDDEN,
            q_out,
            &mut rng,
        )?;
        push_affine(
            &mut tensors,
            &format!("{p}.mlp.gate_proj"),
            INTER,
            HIDDEN,
            &mut rng,
        )?;
        push_affine(
            &mut tensors,
            &format!("{p}.mlp.up_proj"),
            INTER,
            HIDDEN,
            &mut rng,
        )?;
        push_affine(
            &mut tensors,
            &format!("{p}.mlp.down_proj"),
            HIDDEN,
            INTER,
            &mut rng,
        )?;
        push_bf16_vec(
            &mut tensors,
            &format!("{p}.self_attn.q_proj.bias"),
            (0..q_out).map(|_| rng.next_f32() * 0.05).collect(),
        )?;
        push_bf16_vec(
            &mut tensors,
            &format!("{p}.self_attn.k_proj.bias"),
            (0..kv_out).map(|_| rng.next_f32() * 0.05).collect(),
        )?;
        push_bf16_vec(
            &mut tensors,
            &format!("{p}.self_attn.v_proj.bias"),
            (0..kv_out).map(|_| rng.next_f32() * 0.05).collect(),
        )?;
        push_bf16_vec(
            &mut tensors,
            &format!("{p}.input_layernorm.weight"),
            vec![1.0; HIDDEN],
        )?;
        push_bf16_vec(
            &mut tensors,
            &format!("{p}.post_attention_layernorm.weight"),
            vec![1.0; HIDDEN],
        )?;
    }
    push_affine(&mut tensors, DOTS_EMBED_MODULE, VOCAB, HIDDEN, &mut rng)?;
    push_affine(&mut tensors, DOTS_LM_HEAD_MODULE, VOCAB, HIDDEN, &mut rng)?;
    push_bf16_vec(&mut tensors, DOTS_FINAL_NORM, vec![1.0; HIDDEN])?;
    candle_core::safetensors::save(&tensors, dir.join("model.safetensors"))?;
    std::fs::write(
        dir.join("config.json"),
        serde_json::json!({
            "model_type": "dots_ocr",
            "hidden_size": HIDDEN,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "intermediate_size": INTER,
            "vocab_size": VOCAB,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "quantization": {"group_size": DOTS_AFFINE_GROUP_SIZE, "bits": 4, "mode": "affine"},
        })
        .to_string(),
    )?;
    Ok(())
}

pub fn write_dump_dir(dir: &Path, n: usize, seed: u64) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut rng = Rng(seed);
    let mut index = String::new();
    for i in 0..n {
        let n_vis = 4;
        let vision_start = 2;
        let prompt_tail = 3;
        let completion = 6;
        let total = vision_start + n_vis + prompt_tail + completion;
        let ids: Vec<u32> = (0..total)
            .map(|j| ((j as u64 * 13 + seed + i as u64) % VOCAB as u64) as u32)
            .collect();
        let rows: Vec<f32> = (0..n_vis * HIDDEN).map(|_| rng.next_f32()).collect();
        let leaf = format!("{i:03}.safetensors");
        write_dump_sample(
            &dir.join(&leaf),
            &ids,
            &rows,
            HIDDEN,
            vision_start,
            vision_start + n_vis + prompt_tail,
        )?;
        index.push_str(&serde_json::json!({"file": leaf, "len": total}).to_string());
        index.push('\n');
    }
    std::fs::write(dir.join("index.jsonl"), index)?;
    Ok(())
}
