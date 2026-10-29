use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Node {
    op: String,
    #[serde(default)]
    name: Option<String>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    #[serde(default)]
    attrs: HashMap<String, (String, serde_json::Value)>,
}

#[derive(Debug, Deserialize)]
struct GraphFile {
    inputs: Vec<GraphIo>,
    outputs: Vec<GraphIo>,
    nodes: Vec<Node>,
}

#[derive(Debug, Deserialize)]
struct GraphIo {
    name: String,
}

pub const ACTIVATIONS_LEAVE_THE_ENV_AT_THEIR_LAST_CONSUMER_SO_THE_LIVE_SET_NOT_THE_GRAPH_BOUNDS_MEMORY: &str =
    "ResNet293 has 710 nodes whose early activations run 32 channels wide over every fbank \
     frame; keeping all of them until the run ends costs about a gigabyte per ten seconds of \
     audio, which is what wasm32's 4 GiB heap ran out of, so each input tensor is dropped after \
     the last node that reads it and weights are looked up in place instead of cloned into the \
     environment";

pub const A_STRIDED_CONV_BACKPROPAGATES_ONLY_WHEN_THE_TRANSPOSE_REPRODUCES_THE_INPUT_EXTENT: &str =
    "candle's conv2d backward hands the input gradient back through conv_transpose2d, whose \
     extent is (out-1)*stride + kernel - 2*pad and falls short by the rows a strided forward \
     discards; zero-padding the input up to the next stride multiple before the strided conv \
     and narrowing the output to the ONNX size keeps every activation at its strided \
     resolution, where the old stride-1-then-subsample route held ResNet293's stages at full \
     resolution and cost 1.7 GiB of tape for two seconds of audio";

pub struct OnnxGraph {
    nodes: Vec<Node>,
    weights: HashMap<String, Tensor>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

impl Node {
    fn int(&self, k: &str, default: i64) -> i64 {
        match self.attrs.get(k) {
            Some((t, v)) if t == "i" => v.as_i64().unwrap_or(default),
            _ => default,
        }
    }
    fn ints(&self, k: &str) -> Option<Vec<i64>> {
        match self.attrs.get(k) {
            Some((t, v)) if t == "ints" => v
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_i64()).collect()),
            _ => None,
        }
    }
    fn float(&self, k: &str, default: f64) -> f64 {
        match self.attrs.get(k) {
            Some((t, v)) if t == "f" => v.as_f64().unwrap_or(default),
            _ => default,
        }
    }
    fn tensor_attr(&self, k: &str) -> Option<String> {
        match self.attrs.get(k) {
            Some((t, v)) if t == "tensor" => v.as_str().map(|s| s.to_string()),
            _ => None,
        }
    }
}

fn norm_axis(axis: i64, rank: usize) -> usize {
    if axis < 0 {
        (rank as i64 + axis) as usize
    } else {
        axis as usize
    }
}

impl OnnxGraph {
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let graph_json = std::fs::read_to_string(dir.join("graph.json"))?;
        let weights = candle_core::safetensors::load(dir.join("weights.safetensors"), device)?;
        Self::from_parts(&graph_json, weights)
    }

    pub fn from_parts(graph_json: &str, weights: HashMap<String, Tensor>) -> Result<Self> {
        let g: GraphFile = serde_json::from_str(graph_json)?;
        Ok(Self {
            nodes: g.nodes,
            weights,
            input_names: g.inputs.into_iter().map(|i| i.name).collect(),
            output_names: g.outputs.into_iter().map(|o| o.name).collect(),
        })
    }

    pub fn run(&self, inputs: &[(&str, Tensor)]) -> Result<HashMap<String, Tensor>> {
        let _ = ACTIVATIONS_LEAVE_THE_ENV_AT_THEIR_LAST_CONSUMER_SO_THE_LIVE_SET_NOT_THE_GRAPH_BOUNDS_MEMORY;
        let mut last_use: HashMap<&str, usize> = HashMap::with_capacity(self.nodes.len() * 2);
        for (i, n) in self.nodes.iter().enumerate() {
            for k in &n.inputs {
                last_use.insert(k.as_str(), i);
            }
        }
        let mut env: HashMap<String, Tensor> = HashMap::with_capacity(self.nodes.len());
        for (k, v) in inputs {
            env.insert(k.to_string(), v.clone());
        }
        for (ni, n) in self.nodes.iter().enumerate() {
            let name = n.name.clone().unwrap_or_default();
            let look = |k: &str| env.get(k).or_else(|| self.weights.get(k));
            let get = |i: usize| -> Result<&Tensor> {
                let key = n
                    .inputs
                    .get(i)
                    .ok_or_else(|| anyhow!("{name}: missing input {i}"))?;
                look(key).ok_or_else(|| anyhow!("{name}: input {key:?} not computed"))
            };
            let in_shapes: Vec<String> = n
                .inputs
                .iter()
                .map(|k| {
                    look(k)
                        .map(|t| format!("{:?}", t.dims()))
                        .unwrap_or_else(|| "?".into())
                })
                .collect();
            let out: Tensor = (|| -> Result<Tensor> {
                Ok(match n.op.as_str() {
                    "Constant" => {
                        let key = n
                            .tensor_attr("value")
                            .ok_or_else(|| anyhow!("{name}: Constant without tensor"))?;
                        look(&key)
                            .cloned()
                            .ok_or_else(|| anyhow!("{name}: constant {key} missing"))?
                    }
                    "Transpose" => {
                        let perm: Vec<usize> = n
                            .ints("perm")
                            .ok_or_else(|| anyhow!("{name}: perm"))?
                            .iter()
                            .map(|x| *x as usize)
                            .collect();
                        get(0)?.permute(perm.as_slice())?.contiguous()?
                    }
                    "Unsqueeze" => {
                        let axes: Vec<i64> = match n.ints("axes") {
                            Some(a) => a,
                            None => get(1)?.to_dtype(DType::I64)?.flatten_all()?.to_vec1()?,
                        };
                        let mut t = get(0)?.clone();
                        for a in axes {
                            let ax = norm_axis(a, t.rank() + 1);
                            t = t.unsqueeze(ax)?;
                        }
                        t
                    }
                    "Conv" => {
                        let x = get(0)?;
                        let w = get(1)?;
                        let pads = n.ints("pads").unwrap_or_else(|| vec![0; 4]);
                        let strides = n.ints("strides").unwrap_or_else(|| vec![1; 2]);
                        let dil = n.ints("dilations").unwrap_or_else(|| vec![1; 2]);
                        let groups = n.int("group", 1) as usize;
                        if pads.len() == 4
                            && (pads[0] != pads[2] || pads[1] != pads[3] || pads[0] != pads[1])
                        {
                            return Err(anyhow!("{name}: asymmetric pads {pads:?} not supported"));
                        }
                        if strides[0] != strides[1] || dil[0] != dil[1] {
                            return Err(anyhow!(
                                "{name}: anisotropic stride/dilation not supported"
                            ));
                        }
                        let stride = strides[0] as usize;

                        let y = if stride > 1 && super::layers::differentiable() {
                            let _ = A_STRIDED_CONV_BACKPROPAGATES_ONLY_WHEN_THE_TRANSPOSE_REPRODUCES_THE_INPUT_EXTENT;
                            let (p, d) = (pads[0] as usize, dil[0] as usize);
                            let eff = d * (w.dim(2)? - 1) + 1;
                            let (h, wd) = (x.dim(2)?, x.dim(3)?);
                            let extra = |n: usize| (stride - (n + 2 * p - eff) % stride) % stride;
                            let (eh, ew) = (extra(h), extra(wd));
                            let xp = if eh > 0 || ew > 0 {
                                x.pad_with_zeros(2, 0, eh)?.pad_with_zeros(3, 0, ew)?
                            } else {
                                x.clone()
                            };
                            let y = xp.conv2d(w, p, stride, d, groups)?;
                            let (oh, ow) = ((h + 2 * p - eff) / stride + 1, (wd + 2 * p - eff) / stride + 1);
                            y.narrow(2, 0, oh)?.narrow(3, 0, ow)?
                        } else {
                            x.conv2d(w, pads[0] as usize, stride, dil[0] as usize, groups)?
                        };
                        match n.inputs.get(2) {
                            Some(b) if !b.is_empty() => {
                                let bias = look(b).ok_or_else(|| anyhow!("{name}: bias {b}"))?;
                                y.broadcast_add(&bias.reshape((1, bias.dim(0)?, 1, 1))?)?
                            }
                            _ => y,
                        }
                    }
                    "Relu" => get(0)?.relu()?,
                    "Add" => get(0)?.broadcast_add(get(1)?)?,
                    "Sub" => get(0)?.broadcast_sub(get(1)?)?,
                    "Mul" => get(0)?.broadcast_mul(get(1)?)?,
                    "Div" => get(0)?.broadcast_div(get(1)?)?,
                    "Sqrt" => get(0)?.sqrt()?,
                    "ReduceMean" => {
                        let x = get(0)?;
                        let axes: Vec<usize> = n
                            .ints("axes")
                            .unwrap_or_else(|| (0..x.rank() as i64).collect())
                            .iter()
                            .map(|a| norm_axis(*a, x.rank()))
                            .collect();
                        let keep = n.int("keepdims", 1) == 1;
                        let mut t = x.clone();
                        let mut sorted = axes.clone();
                        sorted.sort_unstable_by(|a, b| b.cmp(a));
                        for a in sorted {
                            t = if keep { t.mean_keepdim(a)? } else { t.mean(a)? };
                        }
                        t
                    }
                    "ReduceProd" => {
                        let v: Vec<i64> = get(0)?.to_dtype(DType::I64)?.flatten_all()?.to_vec1()?;
                        let p: i64 = v.iter().product();
                        if n.int("keepdims", 1) == 1 {
                            Tensor::new(&[p], get(0)?.device())?
                        } else {
                            Tensor::new(p, get(0)?.device())?
                        }
                    }
                    "Shape" => {
                        let dims: Vec<i64> = get(0)?.dims().iter().map(|d| *d as i64).collect();
                        Tensor::new(dims.as_slice(), get(0)?.device())?
                    }
                    "Gather" => {
                        let x = get(0)?;
                        let idx = get(1)?;
                        let axis = norm_axis(n.int("axis", 0), x.rank());
                        let ids: Vec<i64> = idx.to_dtype(DType::I64)?.flatten_all()?.to_vec1()?;
                        let n_axis = x.dim(axis)?;
                        let ids: Vec<u32> =
                            ids.iter().map(|i| norm_axis(*i, n_axis) as u32).collect();
                        let sel =
                            x.index_select(&Tensor::new(ids.as_slice(), x.device())?, axis)?;
                        if idx.rank() == 0 {
                            sel.squeeze(axis)?
                        } else {
                            sel
                        }
                    }
                    "Cast" => {
                        let to = n.int("to", 1);
                        let dt = match to {
                            1 => DType::F32,
                            7 => DType::I64,
                            6 => DType::I64,
                            11 => DType::F64,
                            _ => return Err(anyhow!("{name}: Cast to {to} unsupported")),
                        };
                        get(0)?.to_dtype(dt)?
                    }
                    "Flatten" => {
                        let x = get(0)?;
                        let axis = norm_axis(n.int("axis", 1), x.rank());
                        let outer: usize = x.dims()[..axis].iter().product();
                        let inner: usize = x.dims()[axis..].iter().product();
                        x.reshape((outer.max(1), inner))?
                    }
                    "Concat" => {
                        let ts: Vec<&Tensor> =
                            (0..n.inputs.len()).map(&get).collect::<Result<_>>()?;
                        let axis = norm_axis(n.int("axis", 0), ts[0].rank());
                        Tensor::cat(&ts, axis)?
                    }
                    "Gemm" => {
                        let a = get(0)?;
                        let b = get(1)?;
                        let alpha = n.float("alpha", 1.0);
                        let beta = n.float("beta", 1.0);
                        let a = if n.int("transA", 0) == 1 {
                            a.t()?
                        } else {
                            a.clone()
                        };
                        let b = if n.int("transB", 0) == 1 {
                            b.t()?.contiguous()?
                        } else {
                            b.clone()
                        };
                        let mut y = (a.matmul(&b)? * alpha)?;
                        if let Some(c) = n.inputs.get(2).and_then(|c| look(c)) {
                            y = y.broadcast_add(&(c * beta)?)?;
                        }
                        y
                    }
                    other => {
                        return Err(anyhow!(
                            "{name}: op {other} not supported by the candle executor"
                        ))
                    }
                })
            })()
            .with_context(|| format!("{name} ({}) inputs {}", n.op, in_shapes.join(" ")))?;
            let out_name = n
                .outputs
                .first()
                .ok_or_else(|| anyhow!("{name}: no output"))?;
            env.insert(out_name.clone(), out);
            for k in &n.inputs {
                if last_use.get(k.as_str()) == Some(&ni) && !self.output_names.contains(k) {
                    env.remove(k);
                }
            }
        }
        let mut outs = HashMap::new();
        for o in &self.output_names {
            outs.insert(
                o.clone(),
                env.remove(o)
                    .ok_or_else(|| anyhow!("output {o} not produced"))?,
            );
        }
        let _ = D::Minus1;
        Ok(outs)
    }
}
