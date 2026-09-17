use std::path::Path;

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};

use super::fbank::Fbank;
use super::onnx_exec::OnnxGraph;

pub struct WeSpeaker {
    pub fbank: Fbank,
    graph: OnnxGraph,
}

impl WeSpeaker {
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        Self::from_graph(OnnxGraph::load(dir, device)?, device)
    }

    pub fn from_bytes(graph_json: &str, weights: &[u8], device: &Device) -> Result<Self> {
        let weights = candle_core::safetensors::load_buffer(weights, device)?;
        Self::from_graph(OnnxGraph::from_parts(graph_json, weights)?, device)
    }

    pub fn from_graph(graph: OnnxGraph, device: &Device) -> Result<Self> {
        if graph.input_names.len() != 1 {
            return Err(anyhow!(
                "wespeaker graph: expected one input, got {:?}",
                graph.input_names
            ));
        }
        Ok(Self {
            fbank: Fbank::new(device)?,
            graph,
        })
    }

    pub fn embed_feats(&self, feats: &Tensor) -> Result<Tensor> {
        let name = self.graph.input_names[0].as_str();
        let mut outs = self.graph.run(&[(name, feats.unsqueeze(0)?)])?;
        let out = outs
            .remove(&self.graph.output_names[0])
            .ok_or_else(|| anyhow!("no output"))?;
        Ok(out.flatten_all()?)
    }

    pub fn embed_wav16k(&self, wav: &Tensor) -> Result<Tensor> {
        self.embed_feats(&self.fbank.forward(wav)?)
    }
}

pub fn resample_24k_to_16k(wav: &Tensor) -> Result<Tensor> {
    let l = wav.dim(0)?;
    let n = l * 2 / 3;
    let step = (l as f64 - 1.0) / (n.max(2) as f64 - 1.0);
    let mut i0 = Vec::with_capacity(n);
    let mut i1 = Vec::with_capacity(n);
    let mut t = Vec::with_capacity(n);
    for i in 0..n {
        let pos = i as f64 * step;
        let j = (pos.floor() as usize).min(l - 1);
        i0.push(j as u32);
        i1.push((j + 1).min(l - 1) as u32);
        t.push((pos - j as f64) as f32);
    }
    let dev = wav.device();
    let a = wav.index_select(&Tensor::new(i0, dev)?, 0)?;
    let b = wav.index_select(&Tensor::new(i1, dev)?, 0)?;
    let t = Tensor::new(t, dev)?;
    Ok((&a + ((b - &a)? * t)?)?)
}
