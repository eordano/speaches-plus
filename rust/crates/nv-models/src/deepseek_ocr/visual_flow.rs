use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D};
use nv_layers::norm::RmsNorm;
use nv_weights::WeightLoader;

use super::linear;

#[derive(Clone, Debug)]
pub struct FlowConfig {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub query_lengths: Vec<usize>,
}

impl FlowConfig {
    pub fn deepseek_ocr2() -> Self {
        Self {
            num_layers: 24,
            hidden_size: 896,
            num_heads: 14,
            num_kv_heads: 2,
            head_dim: 64,
            intermediate_size: 4864,
            rms_norm_eps: 1e-6,
            rope_theta: 1e6,
            query_lengths: vec![144, 256],
        }
    }
}

pub const NEG_INF: f32 = -1e30;

pub fn flow_mask(n: usize, device: &Device) -> Result<Tensor> {
    let s = 2 * n;
    let mut m = vec![NEG_INF; s * s];
    for i in 0..s {
        for j in 0..s {
            let allowed = if i < n { j < n } else { j < n || j <= i };
            if allowed {
                m[i * s + j] = 0.0;
            }
        }
    }
    Ok(Tensor::from_vec(m, (1, 1, s, s), &Device::Cpu)?.to_device(device)?)
}

struct RopeTables {
    cos: Tensor,
    sin: Tensor,
}

fn rope_tables(seq_len: usize, head_dim: usize, theta: f64, device: &Device) -> Result<RopeTables> {
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(seq_len * head_dim);
    let mut sin = Vec::with_capacity(seq_len * head_dim);
    for pos in 0..seq_len {
        let mut c_half = Vec::with_capacity(half);
        let mut s_half = Vec::with_capacity(half);
        for i in 0..half {
            let inv_freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
            let angle = pos as f64 * inv_freq;
            c_half.push(angle.cos() as f32);
            s_half.push(angle.sin() as f32);
        }
        cos.extend_from_slice(&c_half);
        cos.extend_from_slice(&c_half);
        sin.extend_from_slice(&s_half);
        sin.extend_from_slice(&s_half);
    }
    Ok(RopeTables {
        cos: Tensor::from_vec(cos, (seq_len, head_dim), &Device::Cpu)?.to_device(device)?,
        sin: Tensor::from_vec(sin, (seq_len, head_dim), &Device::Cpu)?.to_device(device)?,
    })
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let hd = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, hd / 2)?;
    let x2 = x.narrow(D::Minus1, hd / 2, hd / 2)?;
    Ok(Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?)
}

fn apply_rope(x: &Tensor, rope: &RopeTables) -> Result<Tensor> {
    let cos = rope.cos.to_dtype(x.dtype())?;
    let sin = rope.sin.to_dtype(x.dtype())?;
    Ok((x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?)?)
}

fn repeat_kv(x: &Tensor, rep: usize) -> Result<Tensor> {
    if rep == 1 {
        return Ok(x.clone());
    }
    let (b, kvh, s, hd) = x.dims4()?;
    Ok(x.unsqueeze(2)?
        .broadcast_as((b, kvh, rep, s, hd))?
        .contiguous()?
        .reshape((b, kvh * rep, s, hd))?)
}

struct FlowLayer {
    input_norm: RmsNorm,
    post_norm: RmsNorm,
    q_w: Tensor,
    q_b: Tensor,
    k_w: Tensor,
    k_b: Tensor,
    v_w: Tensor,
    v_b: Tensor,
    o_w: Tensor,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
}

impl FlowLayer {
    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        rope: &RopeTables,
        cfg: &FlowConfig,
    ) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let nh = cfg.num_heads;
        let kvh = cfg.num_kv_heads;
        let hd = cfg.head_dim;
        let normed = self.input_norm.forward(x)?;
        let q = linear(&normed, &self.q_w, Some(&self.q_b))?
            .reshape((b, s, nh, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = linear(&normed, &self.k_w, Some(&self.k_b))?
            .reshape((b, s, kvh, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = linear(&normed, &self.v_w, Some(&self.v_b))?
            .reshape((b, s, kvh, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let q = apply_rope(&q, rope)?;
        let k = apply_rope(&k, rope)?;
        let k = repeat_kv(&k, nh / kvh)?;
        let v = repeat_kv(&v, nh / kvh)?;
        let scale = (hd as f64).powf(-0.5);
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let scores = scores
            .to_dtype(DType::F32)?
            .broadcast_add(&mask.to_dtype(DType::F32)?)?;
        let probs = if scores.device().is_cpu() {
            candle_nn::ops::softmax(&scores, D::Minus1)?
        } else {
            candle_nn::ops::softmax_last_dim(&scores.contiguous()?)?
        }
        .to_dtype(x.dtype())?;
        let attn = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, nh * hd))?;
        let attn = linear(&attn, &self.o_w, None)?;
        let x = (x + attn)?;
        let normed = self.post_norm.forward(&x)?;
        let mlp =
            (linear(&normed, &self.gate_w, None)?.silu()? * linear(&normed, &self.up_w, None)?)?;
        let mlp = linear(&mlp, &self.down_w, None)?;
        Ok((x + mlp)?)
    }
}

struct FlowConst {
    mask: Tensor,
    rope: RopeTables,
}

pub struct VisualFlow {
    cfg: FlowConfig,
    layers: Vec<FlowLayer>,
    final_norm: RmsNorm,
    queries: Vec<(usize, Tensor)>,
    consts: Mutex<HashMap<(usize, &'static str), Arc<FlowConst>>>,
}

const FLOW_CONST_CACHE_MAX_ENTRIES_8_COVERS_EVERY_QUERY_LENGTH_TIMES_DTYPE_AND_CLEAR_ON_OVERFLOW_IS_SAFE_BECAUSE_FLOW_RUNS_EAGER_AND_ARC_CLONES_KEEP_INFLIGHT_USERS_ALIVE:
    usize = 8;

impl VisualFlow {
    pub fn from_loader(
        weights: &WeightLoader,
        prefix: &str,
        cfg: FlowConfig,
        dtype: DType,
    ) -> Result<Self> {
        let g = |name: &str| -> Result<Tensor> {
            weights
                .get(&format!("{prefix}{name}"), dtype)
                .with_context(|| format!("load {prefix}{name}"))
        };
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let p = format!("model.model.layers.{i}.");
            layers.push(FlowLayer {
                input_norm: RmsNorm::new(
                    g(&format!("{p}input_layernorm.weight"))?,
                    cfg.rms_norm_eps,
                ),
                post_norm: RmsNorm::new(
                    g(&format!("{p}post_attention_layernorm.weight"))?,
                    cfg.rms_norm_eps,
                ),
                q_w: g(&format!("{p}self_attn.q_proj.weight"))?,
                q_b: g(&format!("{p}self_attn.q_proj.bias"))?,
                k_w: g(&format!("{p}self_attn.k_proj.weight"))?,
                k_b: g(&format!("{p}self_attn.k_proj.bias"))?,
                v_w: g(&format!("{p}self_attn.v_proj.weight"))?,
                v_b: g(&format!("{p}self_attn.v_proj.bias"))?,
                o_w: g(&format!("{p}self_attn.o_proj.weight"))?,
                gate_w: g(&format!("{p}mlp.gate_proj.weight"))?,
                up_w: g(&format!("{p}mlp.up_proj.weight"))?,
                down_w: g(&format!("{p}mlp.down_proj.weight"))?,
            });
        }
        let final_norm = RmsNorm::new(g("model.model.norm.weight")?, cfg.rms_norm_eps);
        let mut queries = Vec::new();
        for &n in &cfg.query_lengths {
            let name = format!("query_{}", query_table_name(n));
            queries.push((n, g(&format!("{name}.weight"))?));
        }
        Ok(Self {
            cfg,
            layers,
            final_norm,
            queries,
            consts: Mutex::new(HashMap::new()),
        })
    }

    fn consts(&self, n: usize, dtype: DType, device: &Device) -> Result<Arc<FlowConst>> {
        let key = (n, dtype.as_str());
        let mut cache = self
            .consts
            .lock()
            .map_err(|e| anyhow::anyhow!("visual-flow const cache poisoned: {e}"))?;
        if let Some(c) = cache.get(&key) {
            return Ok(c.clone());
        }
        let s = 2 * n;
        let rope = rope_tables(s, self.cfg.head_dim, self.cfg.rope_theta, device)?;
        let c = Arc::new(FlowConst {
            mask: flow_mask(n, device)?,
            rope: RopeTables {
                cos: rope.cos.to_dtype(dtype)?,
                sin: rope.sin.to_dtype(dtype)?,
            },
        });
        if cache.len()
            >= FLOW_CONST_CACHE_MAX_ENTRIES_8_COVERS_EVERY_QUERY_LENGTH_TIMES_DTYPE_AND_CLEAR_ON_OVERFLOW_IS_SAFE_BECAUSE_FLOW_RUNS_EAGER_AND_ARC_CLONES_KEEP_INFLIGHT_USERS_ALIVE
        {
            cache.clear();
        }
        cache.insert(key, c.clone());
        Ok(c)
    }

    pub fn forward(&self, feat: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = feat.dims4()?;
        let n = h * w;
        let x = feat.reshape((b, c, n))?.transpose(1, 2)?.contiguous()?;
        let table = self
            .queries
            .iter()
            .find(|(len, _)| *len == n)
            .map(|(_, t)| t)
            .ok_or_else(|| anyhow::anyhow!("no query table for n={n} tokens"))?;
        let q = table.unsqueeze(0)?.broadcast_as((b, n, c))?;
        let mut x = Tensor::cat(&[&x, &q], 1)?.contiguous()?;
        let consts = self.consts(n, x.dtype(), x.device())?;
        let (mask, rope) = (&consts.mask, &consts.rope);
        for layer in &self.layers {
            x = layer.forward(&x, mask, rope, &self.cfg)?;
        }
        let x = self.final_norm.forward(&x)?;
        Ok(x.narrow(1, n, n)?.contiguous()?)
    }

    pub fn config(&self) -> &FlowConfig {
        &self.cfg
    }
}

fn query_table_name(n: usize) -> usize {
    match n {
        144 => 768,
        256 => 1024,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_mask(n: usize) -> Vec<f32> {
        let s = 2 * n;
        let mut mask = vec![NEG_INF; s * s];
        let image: Vec<usize> = (0..n).collect();
        let text: Vec<usize> = (n..s).collect();
        for &i in &image {
            for &j in &image {
                mask[i * s + j] = 0.0;
            }
        }
        for (idx, &tp) in text.iter().enumerate() {
            for &j in &image {
                mask[tp * s + j] = 0.0;
            }
            for &j in &text[..idx + 1] {
                mask[tp * s + j] = 0.0;
            }
        }
        mask
    }

    #[test]
    fn analytic_mask_matches_reference_loops() {
        for n in [1, 3, 7] {
            let got: Vec<f32> = flow_mask(n, &Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(got, reference_mask(n), "n={n}");
        }
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let dev = Device::Cpu;
        let rope = rope_tables(4, 8, 1e6, &dev).unwrap();
        let x = Tensor::from_vec(
            (0..8).map(|i| i as f32 + 1.0).collect::<Vec<_>>(),
            (1, 1, 1, 8),
            &dev,
        )
        .unwrap();
        let rope0 = RopeTables {
            cos: rope.cos.narrow(0, 0, 1).unwrap(),
            sin: rope.sin.narrow(0, 0, 1).unwrap(),
        };
        let y = apply_rope(&x, &rope0).unwrap();
        let v: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        let expect: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        for (a, b) in v.iter().zip(&expect) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn repeat_kv_interleaves_heads() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 2, 1, 2), &dev).unwrap();
        let y = repeat_kv(&x, 3).unwrap();
        assert_eq!(y.dims(), &[1, 6, 1, 2]);
        let v: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(
            v,
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
        );
    }
}
