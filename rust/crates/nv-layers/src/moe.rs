use anyhow::{Context, Result};
use candle_core::{DType, Tensor};

#[cfg(feature = "cuda")]
use candle_core::Device;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

use crate::linear::Linear;
use crate::mlp::Mlp;

#[derive(Clone, Copy, Debug)]
pub struct MoeConfig {
    pub hidden_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
}

pub struct MoeBlock {
    cfg: MoeConfig,
    gate: Linear,
    experts: Vec<Mlp>,
    shared_expert: Mlp,
    shared_expert_gate: Linear,

    #[cfg(feature = "cuda")]
    grouped: std::sync::Mutex<Option<std::sync::Arc<crate::moe_grouped::MoeGroupedWeights>>>,
}

impl MoeBlock {
    pub fn new(
        cfg: MoeConfig,
        gate: Linear,
        experts: Vec<Mlp>,
        shared_expert: Mlp,
        shared_expert_gate: Linear,
    ) -> Result<Self> {
        if experts.len() != cfg.num_experts {
            anyhow::bail!(
                "MoeBlock: experts.len()={} != cfg.num_experts={}",
                experts.len(),
                cfg.num_experts
            );
        }
        if cfg.num_experts_per_tok == 0 || cfg.num_experts_per_tok > cfg.num_experts {
            anyhow::bail!(
                "MoeBlock: num_experts_per_tok={} invalid for num_experts={}",
                cfg.num_experts_per_tok,
                cfg.num_experts
            );
        }
        if gate.in_features() != cfg.hidden_size || gate.out_features() != cfg.num_experts {
            anyhow::bail!(
                "MoeBlock: gate shape mismatch -- expected [{}, {}], got [{}, {}]",
                cfg.num_experts,
                cfg.hidden_size,
                gate.out_features(),
                gate.in_features()
            );
        }
        if shared_expert_gate.in_features() != cfg.hidden_size
            || shared_expert_gate.out_features() != 1
        {
            anyhow::bail!(
                "MoeBlock: shared_expert_gate shape mismatch -- expected [1, {}], got [{}, {}]",
                cfg.hidden_size,
                shared_expert_gate.out_features(),
                shared_expert_gate.in_features()
            );
        }
        for (i, e) in experts.iter().enumerate() {
            if e.gate_proj().in_features() != cfg.hidden_size
                || e.gate_proj().out_features() != cfg.moe_intermediate_size
                || e.down_proj().out_features() != cfg.hidden_size
            {
                anyhow::bail!(
                    "MoeBlock: expert {} shape mismatch (hidden={}, inter={})",
                    i,
                    cfg.hidden_size,
                    cfg.moe_intermediate_size
                );
            }
        }
        if shared_expert.gate_proj().in_features() != cfg.hidden_size
            || shared_expert.gate_proj().out_features() != cfg.shared_expert_intermediate_size
            || shared_expert.down_proj().out_features() != cfg.hidden_size
        {
            anyhow::bail!(
                "MoeBlock: shared_expert shape mismatch (hidden={}, inter={})",
                cfg.hidden_size,
                cfg.shared_expert_intermediate_size
            );
        }
        Ok(Self {
            cfg,
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
            #[cfg(feature = "cuda")]
            grouped: std::sync::Mutex::new(None),
        })
    }

    pub fn from_loader(
        cfg: MoeConfig,
        prefix: &str,
        weights: &nv_weights::WeightLoader,
        dtype: DType,
    ) -> Result<Self> {
        let gate = load_linear(
            weights,
            &format!("{prefix}.gate.weight"),
            cfg.num_experts,
            cfg.hidden_size,
            dtype,
        )?;
        let mut experts = Vec::with_capacity(cfg.num_experts);
        for i in 0..cfg.num_experts {
            let gate_proj = load_linear(
                weights,
                &format!("{prefix}.experts.{i}.gate_proj.weight"),
                cfg.moe_intermediate_size,
                cfg.hidden_size,
                dtype,
            )?;
            let up_proj = load_linear(
                weights,
                &format!("{prefix}.experts.{i}.up_proj.weight"),
                cfg.moe_intermediate_size,
                cfg.hidden_size,
                dtype,
            )?;
            let down_proj = load_linear(
                weights,
                &format!("{prefix}.experts.{i}.down_proj.weight"),
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                dtype,
            )?;
            experts.push(Mlp::new(gate_proj, up_proj, down_proj)?);
        }
        let se_gate = load_linear(
            weights,
            &format!("{prefix}.shared_expert.gate_proj.weight"),
            cfg.shared_expert_intermediate_size,
            cfg.hidden_size,
            dtype,
        )?;
        let se_up = load_linear(
            weights,
            &format!("{prefix}.shared_expert.up_proj.weight"),
            cfg.shared_expert_intermediate_size,
            cfg.hidden_size,
            dtype,
        )?;
        let se_down = load_linear(
            weights,
            &format!("{prefix}.shared_expert.down_proj.weight"),
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
            dtype,
        )?;
        let shared_expert = Mlp::new(se_gate, se_up, se_down)?;
        let shared_expert_gate = load_linear(
            weights,
            &format!("{prefix}.shared_expert_gate.weight"),
            1,
            cfg.hidden_size,
            dtype,
        )?;
        Self::new(cfg, gate, experts, shared_expert, shared_expert_gate)
    }

    #[cfg(feature = "cuda")]
    pub fn from_loader_quantized(
        cfg: MoeConfig,
        prefix: &str,
        weights: &nv_weights::WeightLoader,
        dtype: DType,
        qconfig: &nv_weights::QuantizationConfig,
        nvfp4_runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
        device: &Device,
    ) -> Result<Self> {
        if !matches!(qconfig.scheme, nv_weights::QuantScheme::Nvfp4) {
            return Self::from_loader(cfg, prefix, weights, dtype);
        }
        let gate = load_linear_maybe_quant(
            weights,
            &format!("{prefix}.gate.weight"),
            cfg.num_experts,
            cfg.hidden_size,
            dtype,
            qconfig,
            nvfp4_runner.clone(),
            device,
        )?;
        let mut experts = Vec::with_capacity(cfg.num_experts);
        for i in 0..cfg.num_experts {
            let gate_proj = load_linear_maybe_quant(
                weights,
                &format!("{prefix}.experts.{i}.gate_proj.weight"),
                cfg.moe_intermediate_size,
                cfg.hidden_size,
                dtype,
                qconfig,
                nvfp4_runner.clone(),
                device,
            )?;
            let up_proj = load_linear_maybe_quant(
                weights,
                &format!("{prefix}.experts.{i}.up_proj.weight"),
                cfg.moe_intermediate_size,
                cfg.hidden_size,
                dtype,
                qconfig,
                nvfp4_runner.clone(),
                device,
            )?;
            let down_proj = load_linear_maybe_quant(
                weights,
                &format!("{prefix}.experts.{i}.down_proj.weight"),
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                dtype,
                qconfig,
                nvfp4_runner.clone(),
                device,
            )?;
            experts.push(Mlp::new(gate_proj, up_proj, down_proj)?);
        }
        let se_gate = load_linear_maybe_quant(
            weights,
            &format!("{prefix}.shared_expert.gate_proj.weight"),
            cfg.shared_expert_intermediate_size,
            cfg.hidden_size,
            dtype,
            qconfig,
            nvfp4_runner.clone(),
            device,
        )?;
        let se_up = load_linear_maybe_quant(
            weights,
            &format!("{prefix}.shared_expert.up_proj.weight"),
            cfg.shared_expert_intermediate_size,
            cfg.hidden_size,
            dtype,
            qconfig,
            nvfp4_runner.clone(),
            device,
        )?;
        let se_down = load_linear_maybe_quant(
            weights,
            &format!("{prefix}.shared_expert.down_proj.weight"),
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
            dtype,
            qconfig,
            nvfp4_runner.clone(),
            device,
        )?;
        let shared_expert = Mlp::new(se_gate, se_up, se_down)?;
        let shared_expert_gate = load_linear_maybe_quant(
            weights,
            &format!("{prefix}.shared_expert_gate.weight"),
            1,
            cfg.hidden_size,
            dtype,
            qconfig,
            nvfp4_runner.clone(),
            device,
        )?;
        Self::new(cfg, gate, experts, shared_expert, shared_expert_gate)
    }

    pub fn config(&self) -> &MoeConfig {
        &self.cfg
    }

    pub fn gate(&self) -> &Linear {
        &self.gate
    }

    pub fn expert(&self, i: usize) -> &Mlp {
        &self.experts[i]
    }

    pub fn shared_expert(&self) -> &Mlp {
        &self.shared_expert
    }

    pub fn shared_expert_gate(&self) -> &Linear {
        &self.shared_expert_gate
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dims = x.dims().to_vec();
        if *in_dims.last().unwrap() != self.cfg.hidden_size {
            anyhow::bail!(
                "MoeBlock: last dim {} != hidden_size {}",
                in_dims.last().unwrap(),
                self.cfg.hidden_size
            );
        }
        let n_tokens: usize = in_dims[..in_dims.len() - 1].iter().product();
        let in_dtype = x.dtype();
        let device = x.device().clone();
        let x_flat = x.reshape((n_tokens, self.cfg.hidden_size))?.contiguous()?;

        let logits = self
            .gate
            .forward(&x_flat)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let (sorted_logits, sorted_idx) = logits.sort_last_dim(false)?;
        let k = self.cfg.num_experts_per_tok;
        let top_logits = sorted_logits.narrow(1, 0, k)?.contiguous()?;
        let top_idx = sorted_idx.narrow(1, 0, k)?.contiguous()?;
        let top_weights = candle_nn::ops::softmax_last_dim(&top_logits)?.contiguous()?;

        let top_idx_host: Vec<u32> = top_idx.flatten_all()?.to_vec1::<u32>()?;
        let top_weights_host: Vec<f32> = top_weights.flatten_all()?.to_vec1::<f32>()?;
        if let Some(&bad) = top_idx_host
            .iter()
            .find(|&&e| e as usize >= self.cfg.num_experts)
        {
            anyhow::bail!(
                "MoeBlock: routed expert id {} out of range ({} experts) -- \
                 router output is corrupt (e.g. read during graph capture)",
                bad,
                self.cfg.num_experts
            );
        }

        #[cfg(feature = "cuda")]
        if grouped_enabled() {
            if let Some(out) = self.try_forward_grouped(
                &x_flat,
                &top_idx_host,
                &top_weights_host,
                n_tokens,
                k,
                &device,
            )? {
                let shared_contrib = self.shared_contribution(&x_flat)?;
                let y = out.add(&shared_contrib)?;
                let mut out_dims = in_dims[..in_dims.len() - 1].to_vec();
                out_dims.push(self.cfg.hidden_size);
                return Ok(y.reshape(out_dims)?.to_dtype(in_dtype)?);
            }
        }

        let mut expert_rows: Vec<Vec<u32>> = vec![Vec::new(); self.cfg.num_experts];
        let mut expert_w: Vec<Vec<f32>> = vec![Vec::new(); self.cfg.num_experts];
        for n in 0..n_tokens {
            for j in 0..k {
                let e = top_idx_host[n * k + j] as usize;
                expert_rows[e].push(n as u32);
                expert_w[e].push(top_weights_host[n * k + j]);
            }
        }

        let mut acc = Tensor::zeros((n_tokens, self.cfg.hidden_size), DType::F32, &device)?;
        for e in 0..self.cfg.num_experts {
            let rows = &expert_rows[e];
            if rows.is_empty() {
                continue;
            }
            let m = rows.len();
            let idx_t = Tensor::from_vec(rows.clone(), m, &device)?;
            let gathered = x_flat.index_select(&idx_t, 0)?.contiguous()?;
            let y_e = self.experts[e].forward(&gathered)?.to_dtype(DType::F32)?;
            let w_t = Tensor::from_vec(expert_w[e].clone(), (m, 1), &device)?;
            let weighted = y_e.broadcast_mul(&w_t)?;
            acc = acc.index_add(&idx_t, &weighted, 0)?;
        }

        let shared_contrib = self.shared_contribution(&x_flat)?;
        let y = acc.add(&shared_contrib)?;

        let mut out_dims = in_dims[..in_dims.len() - 1].to_vec();
        out_dims.push(self.cfg.hidden_size);
        let y = y.reshape(out_dims)?.to_dtype(in_dtype)?;
        Ok(y)
    }

    pub fn shared_contribution(&self, x_flat: &Tensor) -> Result<Tensor> {
        let shared_out = self
            .shared_expert
            .forward(x_flat)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let shared_gate_logits = self
            .shared_expert_gate
            .forward(x_flat)?
            .to_dtype(DType::F32)?;
        let shared_gate = candle_nn::ops::sigmoid(&shared_gate_logits)?;
        Ok(shared_gate.broadcast_mul(&shared_out)?)
    }

    #[cfg(feature = "cuda")]
    pub fn shared_contribution_device(&self, x_flat: &Tensor) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use std::ffi::c_void;

        let dev = match x_flat.device() {
            Device::Cuda(d) => d.clone(),
            _ => return self.shared_contribution(x_flat),
        };
        let shared_out = self
            .shared_expert
            .forward(x_flat)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let gate_logits = self
            .shared_expert_gate
            .forward(x_flat)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        let (rows, hidden) = shared_out.dims2()?;
        anyhow::ensure!(
            gate_logits.elem_count() == rows,
            "shared_contribution_device: gate logits {} != rows {}",
            gate_logits.elem_count(),
            rows
        );
        let stream = crate::cuda_stream::current_stream(&dev);
        let mut y: cudarc::driver::CudaSlice<f32> = unsafe {
            stream
                .alloc(rows * hidden)
                .map_err(|e| anyhow::anyhow!("shared gate alloc: {e:?}"))?
        };
        {
            let (xs, xl) = shared_out.storage_and_layout();
            let x_cu = match &*xs {
                candle_core::Storage::Cuda(c) => c,
                _ => anyhow::bail!("shared_out must be CUDA"),
            };
            let x_sl = x_cu.as_cuda_slice::<f32>()?;
            let (xp, _g1) = x_sl.device_ptr(&stream);
            let xp = xp + (xl.start_offset() * std::mem::size_of::<f32>()) as u64;
            let (gs, gl) = gate_logits.storage_and_layout();
            let g_cu = match &*gs {
                candle_core::Storage::Cuda(c) => c,
                _ => anyhow::bail!("gate logits must be CUDA"),
            };
            let g_sl = g_cu.as_cuda_slice::<f32>()?;
            let (gp, _g2) = g_sl.device_ptr(&stream);
            let gp = gp + (gl.start_offset() * std::mem::size_of::<f32>()) as u64;
            let (yp, _g3) = y.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::mul_sigmoid_rowgate_f32(
                    stream.cu_stream() as *mut c_void,
                    xp as *const f32,
                    gp as *const f32,
                    yp as *mut f32,
                    rows as i32,
                    hidden as i32,
                )
            };
            anyhow::ensure!(rc == 0, "mul_sigmoid_rowgate_f32 rc={rc}");
        }
        let storage = candle_core::CudaStorage::wrap_cuda_slice(y, dev);
        Ok(Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (rows, hidden),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    #[cfg(feature = "cuda")]
    pub fn grouped_weights_built(
        &self,
        device: &Device,
    ) -> Result<Arc<crate::moe_grouped::MoeGroupedWeights>> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("MoeBlock: grouped weights require a CUDA device"),
        };
        let mut slot = self
            .grouped
            .lock()
            .map_err(|e| anyhow::anyhow!("MoeBlock.grouped mutex poisoned: {e}"))?;
        if slot.is_none() {
            let stream = dev.cuda_stream();
            let built = crate::moe_grouped::MoeGroupedWeights::build_from_experts(
                &self.experts,
                self.cfg.hidden_size,
                self.cfg.moe_intermediate_size,
                &stream,
            )?;
            *slot = Some(Arc::new(built));
        }
        Ok(slot.as_ref().unwrap().clone())
    }

    #[cfg(feature = "cuda")]
    fn try_forward_grouped(
        &self,
        x_flat: &Tensor,
        topk_ids: &[u32],
        topk_weights: &[f32],
        n_tokens: usize,
        k: usize,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        const MIN_TILE: usize = nv_quant::nvfp4::MIN_TILE;
        let mut counts = vec![0u32; self.cfg.num_experts];
        for &e in topk_ids {
            counts[e as usize] += 1;
        }
        if counts.iter().any(|&c| c as usize > MIN_TILE) {
            return Ok(None);
        }

        if !matches!(device, Device::Cuda(_)) {
            return Ok(None);
        }
        let grouped = match self.grouped_weights_built(device) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[moe] grouped path init failed, falling back: {e}");
                return Ok(None);
            }
        };

        let out = crate::moe_grouped::forward_grouped(
            &grouped,
            &grouped,
            x_flat,
            topk_ids,
            topk_weights,
            n_tokens,
            k,
            device,
        )?;
        Ok(Some(out))
    }
}

#[cfg(feature = "cuda")]
fn grouped_enabled() -> bool {
    match std::env::var("SPEACHES_MOE_GROUPED").ok().as_deref() {
        Some("1") | Some("true") | Some("on") => true,
        _ => false,
    }
}

fn load_linear(
    weights: &nv_weights::WeightLoader,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<Linear> {
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 2 || d[0] != out_features || d[1] != in_features {
        anyhow::bail!(
            "linear {name}: expected [{}, {}], got {:?}",
            out_features,
            in_features,
            d
        );
    }
    Linear::new(w, None)
}

#[cfg(feature = "cuda")]
pub fn load_linear_maybe_quant(
    weights: &nv_weights::WeightLoader,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
    qconfig: &nv_weights::QuantizationConfig,
    runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    device: &Device,
) -> Result<Linear> {
    let module = name.strip_suffix(".weight").unwrap_or(name);
    let is_ignored = qconfig.is_module_ignored(module);
    if is_ignored {
        return load_linear(weights, name, out_features, in_features, dtype);
    }
    if weights.has(name) {
        return load_linear(weights, name, out_features, in_features, dtype);
    }
    let packed_name = format!("{module}.weight_packed");
    if !weights.has(&packed_name) {
        return load_linear(weights, name, out_features, in_features, dtype);
    }
    if in_features < nv_quant::nvfp4::MIN_TILE || out_features < nv_quant::nvfp4::MIN_TILE {
        let _ = runner;
        let deq = dequantize_nvfp4_to_bf16(weights, module, out_features, in_features, device)?;
        return Linear::new(deq, None);
    }
    nvfp4_linear_from_disk(weights, module, out_features, in_features, runner, device)
}

#[derive(Clone, Copy, Debug)]
pub struct Nvfp4Suffixes {
    pub packed: &'static str,
    pub block_scale: &'static str,
    pub weight_global_scale: &'static str,
    pub input_global_scale: &'static str,
    pub global_scale_is_inverse: bool,
}

impl Nvfp4Suffixes {
    pub const QWEN_COMPRESSED_TENSORS: Self = Self {
        packed: "weight_packed",
        block_scale: "weight_scale",
        weight_global_scale: "weight_global_scale",
        input_global_scale: "input_global_scale",
        global_scale_is_inverse: true,
    };
    pub const GEMMA_MODELOPT: Self = Self {
        packed: "weight",
        block_scale: "weight_scale",
        weight_global_scale: "weight_scale_2",
        input_global_scale: "input_scale",
        global_scale_is_inverse: false,
    };
}

#[cfg(feature = "cuda")]
pub fn nvfp4_dequant_bf16_linear_from_disk_because_ablating_the_native_gemm_isolates_decode_defects(
    weights: &nv_weights::WeightLoader,
    module: &str,
    out_features: usize,
    in_features: usize,
    device: &Device,
) -> Result<Linear> {
    let deq = dequantize_nvfp4_to_bf16(weights, module, out_features, in_features, device)?;
    Linear::new(deq, None)
}

#[cfg(feature = "cuda")]
pub fn nvfp4_linear_from_disk_pub(
    weights: &nv_weights::WeightLoader,
    module: &str,
    out_features: usize,
    in_features: usize,
    runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    device: &Device,
) -> Result<Linear> {
    nvfp4_linear_from_disk_with_suffixes(
        weights,
        module,
        out_features,
        in_features,
        runner,
        device,
        Nvfp4Suffixes::QWEN_COMPRESSED_TENSORS,
    )
}

#[cfg(feature = "cuda")]
pub fn nvfp4_linear_from_disk_with_suffixes(
    weights: &nv_weights::WeightLoader,
    module: &str,
    out_features: usize,
    in_features: usize,
    runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    device: &Device,
    sfx: Nvfp4Suffixes,
) -> Result<Linear> {
    nvfp4_linear_from_disk_inner(
        weights,
        module,
        out_features,
        in_features,
        runner,
        device,
        sfx,
    )
}

#[cfg(feature = "cuda")]
fn nvfp4_linear_from_disk(
    weights: &nv_weights::WeightLoader,
    module: &str,
    out_features: usize,
    in_features: usize,
    runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    device: &Device,
) -> Result<Linear> {
    nvfp4_linear_from_disk_inner(
        weights,
        module,
        out_features,
        in_features,
        runner,
        device,
        Nvfp4Suffixes::QWEN_COMPRESSED_TENSORS,
    )
}

#[cfg(feature = "cuda")]
fn nvfp4_linear_from_disk_inner(
    weights: &nv_weights::WeightLoader,
    module: &str,
    out_features: usize,
    in_features: usize,
    runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    device: &Device,
    sfx: Nvfp4Suffixes,
) -> Result<Linear> {
    let dev = match device {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("nvfp4 ingest requires a CUDA device"),
    };
    let packed_name = format!("{module}.{}", sfx.packed);
    let scale_name = format!("{module}.{}", sfx.block_scale);
    let gscale_name = format!("{module}.{}", sfx.weight_global_scale);
    let packed_shape = weights
        .shape_of(&packed_name)
        .ok_or_else(|| anyhow::anyhow!("missing {packed_name}"))?;
    if packed_shape.len() != 2
        || packed_shape[0] != out_features
        || packed_shape[1] != in_features / 2
    {
        anyhow::bail!(
            "nvfp4 {module}: weight_packed shape expected [{}, {}], got {:?}",
            out_features,
            in_features / 2,
            packed_shape
        );
    }
    let packed_bytes = weights
        .raw_bytes(&packed_name)
        .with_context(|| format!("read {packed_name}"))?
        .to_vec();
    if packed_bytes.len() != out_features * in_features / 2 {
        anyhow::bail!(
            "nvfp4 {module}: weight_packed byte length {} != expected {}",
            packed_bytes.len(),
            out_features * in_features / 2
        );
    }
    let scale_shape = weights
        .shape_of(&scale_name)
        .ok_or_else(|| anyhow::anyhow!("missing {scale_name}"))?;
    let scale_bytes_raw = weights
        .raw_bytes(&scale_name)
        .with_context(|| format!("read {scale_name}"))?;
    let expected_scale_len = if scale_shape.is_empty() {
        0
    } else {
        scale_shape.iter().product::<usize>()
    };
    if scale_bytes_raw.len() != expected_scale_len {
        anyhow::bail!(
            "nvfp4 {module}: weight_scale byte length {} != prod(shape)={}",
            scale_bytes_raw.len(),
            expected_scale_len
        );
    }
    let gscale_t = weights
        .get(&gscale_name, DType::F32)
        .with_context(|| format!("read {gscale_name}"))?;
    let gscale_vec = gscale_t.flatten_all()?.to_vec1::<f32>()?;
    let raw_weight_global = *gscale_vec.first().unwrap_or(&1.0f32);
    let input_gscale_name = format!("{module}.{}", sfx.input_global_scale);
    let raw_input_global = if weights.has(&input_gscale_name) {
        let t = weights
            .get(&input_gscale_name, DType::F32)
            .with_context(|| format!("read {input_gscale_name}"))?;
        let v = t.flatten_all()?.to_vec1::<f32>()?;
        *v.first().unwrap_or(&1.0f32)
    } else {
        1.0f32
    };
    let safe_recip = |x: f32| {
        if x == 0.0 || !x.is_finite() {
            1.0
        } else {
            1.0 / x
        }
    };

    let (stored_weight_global, stored_input_global) = if sfx.global_scale_is_inverse {
        (raw_weight_global, raw_input_global)
    } else {
        (safe_recip(raw_weight_global), safe_recip(raw_input_global))
    };
    let weight_alpha = safe_recip(stored_weight_global) * safe_recip(stored_input_global);
    let stream = dev.cuda_stream();
    #[allow(deprecated)]
    let w_dev = stream
        .clone_htod(&packed_bytes)
        .map_err(|e| anyhow::anyhow!(e))?;
    let blocks_per_row = in_features / nv_quant::nvfp4::BLOCK_SIZE;
    let swizzled_scales =
        nv_quant::nvfp4::swizzle_scales(scale_bytes_raw, out_features, blocks_per_row);
    #[allow(deprecated)]
    let s_dev = stream
        .clone_htod(&swizzled_scales)
        .map_err(|e| anyhow::anyhow!(e))?;
    Linear::new_nvfp4_scaled(
        w_dev,
        s_dev,
        in_features,
        out_features,
        None,
        device,
        runner,
        weight_alpha,
        stored_input_global,
    )
}

#[cfg(feature = "cuda")]
pub fn nvfp4_linear_from_disk_fused_pair(
    weights: &nv_weights::WeightLoader,
    module_a: &str,
    module_b: &str,
    out_features_each: usize,
    in_features: usize,
    runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    device: &Device,
    sfx: Nvfp4Suffixes,
) -> Result<Linear> {
    if out_features_each % 128 != 0 {
        anyhow::bail!(
            "fused NVFP4 pair: out_features_each ({}) must be a multiple of 128 \
             so the swizzled scale layout concatenates cleanly",
            out_features_each
        );
    }
    let dev = match device {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("nvfp4 ingest requires a CUDA device"),
    };

    let read_raw = |module: &str| -> Result<(Vec<u8>, Vec<u8>, f32, f32)> {
        let packed_name = format!("{module}.{}", sfx.packed);
        let scale_name = format!("{module}.{}", sfx.block_scale);
        let gscale_name = format!("{module}.{}", sfx.weight_global_scale);
        let input_gscale_name = format!("{module}.{}", sfx.input_global_scale);

        let packed_shape = weights
            .shape_of(&packed_name)
            .ok_or_else(|| anyhow::anyhow!("missing {packed_name}"))?;
        if packed_shape.len() != 2
            || packed_shape[0] != out_features_each
            || packed_shape[1] != in_features / 2
        {
            anyhow::bail!(
                "fused {module}: weight_packed shape expected [{}, {}], got {:?}",
                out_features_each,
                in_features / 2,
                packed_shape
            );
        }
        let packed = weights
            .raw_bytes(&packed_name)
            .with_context(|| format!("read {packed_name}"))?
            .to_vec();
        let scale = weights
            .raw_bytes(&scale_name)
            .with_context(|| format!("read {scale_name}"))?
            .to_vec();
        let gscale_t = weights
            .get(&gscale_name, DType::F32)
            .with_context(|| format!("read {gscale_name}"))?;
        let raw_w = *gscale_t
            .flatten_all()?
            .to_vec1::<f32>()?
            .first()
            .unwrap_or(&1.0);
        let raw_x = if weights.has(&input_gscale_name) {
            let t = weights
                .get(&input_gscale_name, DType::F32)
                .with_context(|| format!("read {input_gscale_name}"))?;
            *t.flatten_all()?.to_vec1::<f32>()?.first().unwrap_or(&1.0)
        } else {
            1.0
        };
        Ok((packed, scale, raw_w, raw_x))
    };

    let (packed_a, scale_a, raw_wa, raw_xa) = read_raw(module_a)?;
    let (packed_b, scale_b, raw_wb, raw_xb) = read_raw(module_b)?;

    if raw_wa != raw_wb || raw_xa != raw_xb {
        anyhow::bail!(
            "fused NVFP4 pair ({} + {}): global scales must match. \
             gate(w={:.6e} x={:.6e}) up(w={:.6e} x={:.6e})",
            module_a,
            module_b,
            raw_wa,
            raw_xa,
            raw_wb,
            raw_xb
        );
    }

    let mut packed_fused = Vec::with_capacity(packed_a.len() + packed_b.len());
    packed_fused.extend_from_slice(&packed_a);
    packed_fused.extend_from_slice(&packed_b);

    let mut scale_raw_fused = Vec::with_capacity(scale_a.len() + scale_b.len());
    scale_raw_fused.extend_from_slice(&scale_a);
    scale_raw_fused.extend_from_slice(&scale_b);

    let out_features = 2 * out_features_each;
    let safe_recip = |x: f32| {
        if x == 0.0 || !x.is_finite() {
            1.0
        } else {
            1.0 / x
        }
    };
    let (stored_weight_global, stored_input_global) = if sfx.global_scale_is_inverse {
        (raw_wa, raw_xa)
    } else {
        (safe_recip(raw_wa), safe_recip(raw_xa))
    };
    let weight_alpha = safe_recip(stored_weight_global) * safe_recip(stored_input_global);

    let stream = dev.cuda_stream();
    #[allow(deprecated)]
    let w_dev = stream
        .clone_htod(&packed_fused)
        .map_err(|e| anyhow::anyhow!(e))?;
    let blocks_per_row = in_features / nv_quant::nvfp4::BLOCK_SIZE;
    let swizzled_scales =
        nv_quant::nvfp4::swizzle_scales(&scale_raw_fused, out_features, blocks_per_row);
    #[allow(deprecated)]
    let s_dev = stream
        .clone_htod(&swizzled_scales)
        .map_err(|e| anyhow::anyhow!(e))?;
    Linear::new_nvfp4_scaled(
        w_dev,
        s_dev,
        in_features,
        out_features,
        None,
        device,
        runner,
        weight_alpha,
        stored_input_global,
    )
}

#[cfg(feature = "cuda")]
fn dequantize_nvfp4_to_bf16(
    weights: &nv_weights::WeightLoader,
    module: &str,
    out_features: usize,
    in_features: usize,
    device: &Device,
) -> Result<Tensor> {
    use nv_quant::nvfp4::{decode_e2m1, decode_ue4m3, unpack_e2m1_pair, BLOCK_SIZE};
    let packed_name = format!("{module}.weight_packed");
    let scale_name = format!("{module}.weight_scale");
    let gscale_name = format!("{module}.weight_global_scale");
    let packed_bytes = weights.raw_bytes(&packed_name)?.to_vec();
    let scale_bytes = weights.raw_bytes(&scale_name)?.to_vec();
    let gscale = weights.get(&gscale_name, DType::F32)?;
    let gvec = gscale.flatten_all()?.to_vec1::<f32>()?;
    let stored_global = *gvec.first().unwrap_or(&1.0f32);
    let inv_global = if stored_global == 0.0 {
        1.0
    } else {
        1.0 / stored_global
    };
    let blocks_per_row = in_features / BLOCK_SIZE;
    let bytes_per_row = in_features / 2;
    let mut out: Vec<half::bf16> = Vec::with_capacity(out_features * in_features);
    for r in 0..out_features {
        for b in 0..blocks_per_row {
            let scale_byte = scale_bytes[r * blocks_per_row + b];
            let block_scale = decode_ue4m3(scale_byte) * inv_global;
            let block_off = r * bytes_per_row + b * (BLOCK_SIZE / 2);
            for i in 0..(BLOCK_SIZE / 2) {
                let (lo, hi) = unpack_e2m1_pair(packed_bytes[block_off + i]);
                let v_lo = decode_e2m1(lo) * block_scale;
                let v_hi = decode_e2m1(hi) * block_scale;
                out.push(half::bf16::from_f32(v_lo));
                out.push(half::bf16::from_f32(v_hi));
            }
        }
    }
    Ok(Tensor::from_vec(out, (out_features, in_features), device)?)
}

#[cfg(test)]
mod paper_validation {

    use super::*;
    use candle_core::Device;

    fn det(vals: usize, seed: f32) -> Vec<f32> {
        (0..vals)
            .map(|i| ((i as f32 + seed) * 0.7311).sin() * 0.9)
            .collect()
    }

    fn linear_from(vals: &[f32], out_f: usize, in_f: usize) -> Linear {
        let t = Tensor::from_vec(vals.to_vec(), (out_f, in_f), &Device::Cpu).unwrap();
        Linear::new(t, None).unwrap()
    }

    fn matvec(w: &[f32], x: &[f32], out_f: usize, in_f: usize) -> Vec<f32> {
        (0..out_f)
            .map(|o| (0..in_f).map(|i| w[o * in_f + i] * x[i]).sum())
            .collect()
    }

    fn silu(v: f32) -> f32 {
        v / (1.0 + (-v).exp())
    }

    fn mlp_ref(
        gate: &[f32],
        up: &[f32],
        down: &[f32],
        x: &[f32],
        hidden: usize,
        inter: usize,
    ) -> Vec<f32> {
        let g = matvec(gate, x, inter, hidden);
        let u = matvec(up, x, inter, hidden);
        let act: Vec<f32> = g.iter().zip(u.iter()).map(|(a, b)| silu(*a) * b).collect();
        matvec(down, &act, hidden, inter)
    }

    #[test]
    fn moe_forward_matches_shazeer_mixture_with_renormalized_topk_softmax() {
        let hidden = 4usize;
        let inter = 6usize;
        let shared_inter = 5usize;
        let n_exp = 3usize;
        let k = 2usize;
        let n_tok = 5usize;

        let wg = det(n_exp * hidden, 1.0);
        let mut e_gate = Vec::new();
        let mut e_up = Vec::new();
        let mut e_down = Vec::new();
        let mut experts = Vec::new();
        for e in 0..n_exp {
            let g = det(inter * hidden, 10.0 + e as f32);
            let u = det(inter * hidden, 20.0 + e as f32);
            let d = det(hidden * inter, 30.0 + e as f32);
            experts.push(
                Mlp::new(
                    linear_from(&g, inter, hidden),
                    linear_from(&u, inter, hidden),
                    linear_from(&d, hidden, inter),
                )
                .unwrap(),
            );
            e_gate.push(g);
            e_up.push(u);
            e_down.push(d);
        }
        let sg = det(shared_inter * hidden, 40.0);
        let su = det(shared_inter * hidden, 41.0);
        let sd = det(hidden * shared_inter, 42.0);
        let wsg = det(hidden, 43.0);

        let cfg = MoeConfig {
            hidden_size: hidden,
            num_experts: n_exp,
            num_experts_per_tok: k,
            moe_intermediate_size: inter,
            shared_expert_intermediate_size: shared_inter,
        };
        let block = MoeBlock::new(
            cfg,
            linear_from(&wg, n_exp, hidden),
            experts,
            Mlp::new(
                linear_from(&sg, shared_inter, hidden),
                linear_from(&su, shared_inter, hidden),
                linear_from(&sd, hidden, shared_inter),
            )
            .unwrap(),
            linear_from(&wsg, 1, hidden),
        )
        .unwrap();

        let x_host = det(n_tok * hidden, 99.0);
        let x = Tensor::from_vec(x_host.clone(), (n_tok, hidden), &Device::Cpu).unwrap();
        let y = block.forward(&x).unwrap();
        let y_host = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        for n in 0..n_tok {
            let xr = &x_host[n * hidden..(n + 1) * hidden];

            let logits = matvec(&wg, xr, n_exp, hidden);
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
            let z: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / z).collect();
            let mut idx: Vec<usize> = (0..n_exp).collect();
            idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            let sel = &idx[..k];
            let sel_sum: f32 = sel.iter().map(|&e| probs[e]).sum();

            let mut want = vec![0f32; hidden];
            for &e in sel {
                let w = probs[e] / sel_sum;
                let ye = mlp_ref(&e_gate[e], &e_up[e], &e_down[e], xr, hidden, inter);
                for i in 0..hidden {
                    want[i] += w * ye[i];
                }
            }
            let ys = mlp_ref(&sg, &su, &sd, xr, hidden, shared_inter);
            let gate_logit: f32 = (0..hidden).map(|i| wsg[i] * xr[i]).sum();
            let gate = 1.0 / (1.0 + (-gate_logit).exp());
            for i in 0..hidden {
                want[i] += gate * ys[i];
            }

            for i in 0..hidden {
                let got = y_host[n * hidden + i];
                assert!(
                    (got - want[i]).abs() <= 1e-4 * want[i].abs().max(1.0),
                    "token {n} dim {i}: got {got}, mixture reference {}",
                    want[i]
                );
            }
        }
    }

    #[test]
    fn moe_topk_weights_sum_to_one() {
        let device = Device::Cpu;
        let n_exp = 7usize;
        let k = 3usize;
        let logits_host = det(4 * n_exp, 5.0);
        let logits = Tensor::from_vec(logits_host.clone(), (4, n_exp), &device).unwrap();
        let (sorted, _) = logits.sort_last_dim(false).unwrap();
        let top = sorted.narrow(1, 0, k).unwrap().contiguous().unwrap();
        let w = candle_nn::ops::softmax_last_dim(&top).unwrap();
        let w_host = w.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for n in 0..4 {
            let s: f32 = w_host[n * k..(n + 1) * k].iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-5,
                "token {n}: top-k weights sum {s} != 1"
            );

            let row = &logits_host[n * n_exp..(n + 1) * n_exp];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|l| (l - max).exp()).collect();
            let z: f32 = exps.iter().sum();
            let mut probs: Vec<f32> = exps.iter().map(|e| e / z).collect();
            probs.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let sel_sum: f32 = probs[..k].iter().sum();
            for j in 0..k {
                let want = probs[j] / sel_sum;
                assert!(
                    (w_host[n * k + j] - want).abs() < 1e-5,
                    "token {n} slot {j}: {} vs renormalized {want}",
                    w_host[n * k + j]
                );
            }
        }
    }
}
