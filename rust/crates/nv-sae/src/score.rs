use anyhow::Result;
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::train::Sae;

pub const SCALE_PERCENTILE: f64 = 0.995;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Scored {
    pub width: usize,
    pub top_k: usize,
    pub scale: Vec<f32>,
    pub indices: Vec<Vec<u32>>,
    pub values: Vec<Vec<f32>>,
}

impl Scored {
    pub fn dense_row(&self, i: usize) -> Vec<f32> {
        let mut v = vec![0f32; self.width];
        for (f, x) in self.indices[i].iter().zip(&self.values[i]) {
            v[*f as usize] = *x;
        }
        v
    }
}

pub fn percentile_scale(h: &[Vec<f32>], width: usize) -> Vec<f32> {
    let mut scale = vec![1f32; width];
    let n = h.len();
    if n == 0 {
        return scale;
    }
    let at = ((n as f64 - 1.0) * SCALE_PERCENTILE).round() as usize;
    for (f, s) in scale.iter_mut().enumerate() {
        let mut col: Vec<f32> = h.iter().map(|r| r[f]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p = col[at];
        *s = if p > 0.0 { p } else { col[n - 1].max(1e-6) };
    }
    scale
}

pub fn score(sae: &Sae, x: &Tensor, batch: usize) -> Result<Scored> {
    let n = x.dims()[0];
    let width = sae.width();
    let mut indices = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    let mut dense: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut start = 0;
    while start < n {
        let len = batch.min(n - start);
        let z = sae.normalize(&crate::train::rows_to_device(x, start, len, sae.mu.device())?)?;
        let h = sae.encode(&z)?.to_vec2::<f32>()?;
        for row in h {
            let mut on: Vec<(u32, f32)> = row
                .iter()
                .enumerate()
                .filter(|(_, v)| **v > 0.0)
                .map(|(f, v)| (f as u32, *v))
                .collect();
            on.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indices.push(on.iter().map(|x| x.0).collect());
            values.push(on.iter().map(|x| x.1).collect());
            dense.push(row);
        }
        start += len;
    }
    let scale = percentile_scale(&dense, width);
    Ok(Scored {
        width,
        top_k: sae.top_k,
        scale,
        indices,
        values,
    })
}

pub fn load_matrix(path: &Path, key: &str) -> Result<Tensor> {
    let m = candle_core::safetensors::load(path, &Device::Cpu)?;
    m.get(key)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("missing tensor {key} in {}", path.display()))
}

pub fn save_scored(path: &Path, s: &Scored) -> Result<()> {
    let n = s.indices.len();
    let k = s.top_k;
    let mut idx = vec![0i64; n * k];
    let mut val = vec![0f32; n * k];
    for i in 0..n {
        for (j, (f, v)) in s.indices[i].iter().zip(&s.values[i]).take(k).enumerate() {
            idx[i * k + j] = *f as i64;
            val[i * k + j] = *v;
        }
    }
    let dev = Device::Cpu;
    let mut m = HashMap::new();
    m.insert("indices".to_string(), Tensor::from_vec(idx, (n, k), &dev)?);
    m.insert("values".to_string(), Tensor::from_vec(val, (n, k), &dev)?);
    m.insert("scale".to_string(), Tensor::new(s.scale.as_slice(), &dev)?);
    candle_core::safetensors::save(&m, path)?;
    Ok(())
}

pub fn load_scored(path: &Path, width: usize) -> Result<Scored> {
    let m = candle_core::safetensors::load(path, &Device::Cpu)?;
    let idx = m["indices"].to_vec2::<i64>()?;
    let val = m["values"].to_vec2::<f32>()?;
    let scale = m["scale"].to_vec1::<f32>()?;
    let k = idx.first().map(|r| r.len()).unwrap_or(0);
    let mut indices = Vec::with_capacity(idx.len());
    let mut values = Vec::with_capacity(idx.len());
    for (ir, vr) in idx.iter().zip(&val) {
        let mut a = Vec::new();
        let mut b = Vec::new();
        for (f, v) in ir.iter().zip(vr) {
            if *v > 0.0 {
                a.push(*f as u32);
                b.push(*v);
            }
        }
        indices.push(a);
        values.push(b);
    }
    Ok(Scored {
        width,
        top_k: k,
        scale,
        indices,
        values,
    })
}
