use anyhow::Result;
use candle_core::{CpuStorage, CustomOp3, DType, Layout, Shape, Storage, Tensor, D};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

pub struct GqaSdpaRecomputePerKvHeadGroup {
    pub batch: usize,
    pub n_q: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub seq: usize,
    pub window: Option<usize>,
    pub serial_kv_groups: bool,
}

pub const GQA_SDPA_RECOMPUTE_KEEPS_ONLY_Q_K_V_AND_O_ON_THE_GRAPH_AND_REBUILDS_THE_PROBABILITIES_IN_THE_BACKWARD: &str =
    "the naive training attention held f32 q, K/V expanded to every query head and three T x T score \
     tensors per layer on the autograd graph; this op keeps the four bf16 operands and recomputes the \
     kv-head-grouped softmax inside bwd, so attention memory is linear in tokens";

static CAUSAL_WINDOW_MASKS: OnceLock<Mutex<HashMap<(usize, Option<usize>, usize), Tensor>>> =
    OnceLock::new();

fn device_key(device: &candle_core::Device) -> usize {
    match device.location() {
        candle_core::DeviceLocation::Cpu => usize::MAX,
        candle_core::DeviceLocation::Cuda { gpu_id } => gpu_id,
        candle_core::DeviceLocation::Metal { gpu_id } => gpu_id,
    }
}

fn causal_window_mask(
    seq: usize,
    window: Option<usize>,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let key = (seq, window, device_key(device));
    let cache = CAUSAL_WINDOW_MASKS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(m) = cache.lock().unwrap().get(&key) {
        return Ok(m.clone());
    }
    let mut mask = vec![0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            let visible = j <= i && window.is_none_or(|w| i - j < w);
            if !visible {
                mask[i * seq + j] = f32::NEG_INFINITY;
            }
        }
    }
    let t = Tensor::from_vec(mask, (1usize, 1usize, seq, seq), device)?;
    cache.lock().unwrap().insert(key, t.clone());
    Ok(t)
}

impl GqaSdpaRecomputePerKvHeadGroup {
    fn one_group(&self) -> Self {
        Self {
            batch: 1,
            n_q: self.group(),
            n_kv: 1,
            head_dim: self.head_dim,
            seq: self.seq,
            window: self.window,
            serial_kv_groups: false,
        }
    }

    fn group_slice(&self, x: &Tensor, batch: usize, head: usize, heads: usize) -> Result<Tensor> {
        Ok(x.narrow(0, batch, 1)?
            .narrow(2, head, heads)?
            .contiguous()?)
    }

    fn forward_serial(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let mut op = self.one_group();
        op.n_q = 1;
        let mut batches = Vec::with_capacity(self.batch);
        for b in 0..self.batch {
            let mut heads = Vec::with_capacity(self.n_q);
            for h in 0..self.n_kv {
                let key = self.group_slice(k, b, h, 1)?;
                let value = self.group_slice(v, b, h, 1)?;
                for query in h * self.group()..(h + 1) * self.group() {
                    heads.push(op.forward_detached(
                        &self.group_slice(q, b, query, 1)?,
                        &key,
                        &value,
                    )?);
                }
            }
            batches.push(Tensor::cat(&heads, 2)?);
        }
        Ok(Tensor::cat(&batches, 0)?)
    }

    fn backward_serial(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        out: &Tensor,
        dout: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let mut op = self.one_group();
        op.n_q = 1;
        let mut batches: [Vec<Tensor>; 3] = std::array::from_fn(|_| Vec::with_capacity(self.batch));
        for b in 0..self.batch {
            let mut groups: [Vec<Tensor>; 3] =
                std::array::from_fn(|_| Vec::with_capacity(self.n_kv));
            for h in 0..self.n_kv {
                let key = self.group_slice(k, b, h, 1)?.to_dtype(DType::F32)?;
                let value = self.group_slice(v, b, h, 1)?.to_dtype(DType::F32)?;
                let mut queries = Vec::with_capacity(self.group());
                let mut dk_sum: Option<Tensor> = None;
                let mut dv_sum: Option<Tensor> = None;
                for query in h * self.group()..(h + 1) * self.group() {
                    let (dq, dk, dv) = op.backward_detached(
                        &self.group_slice(q, b, query, 1)?,
                        &key,
                        &value,
                        &self.group_slice(out, b, query, 1)?,
                        &self.group_slice(dout, b, query, 1)?,
                    )?;
                    queries.push(dq);
                    dk_sum = Some(match dk_sum {
                        Some(sum) => sum.add(&dk)?,
                        None => dk,
                    });
                    dv_sum = Some(match dv_sum {
                        Some(sum) => sum.add(&dv)?,
                        None => dv,
                    });
                }
                groups[0].push(Tensor::cat(&queries, 2)?);
                groups[1].push(
                    dk_sum
                        .ok_or_else(|| anyhow::anyhow!("missing key gradient"))?
                        .to_dtype(k.dtype())?,
                );
                groups[2].push(
                    dv_sum
                        .ok_or_else(|| anyhow::anyhow!("missing value gradient"))?
                        .to_dtype(v.dtype())?,
                );
            }
            for (batch, group) in batches.iter_mut().zip(groups) {
                batch.push(Tensor::cat(&group, 2)?);
            }
        }
        Ok((
            Tensor::cat(&batches[0], 0)?,
            Tensor::cat(&batches[1], 0)?,
            Tensor::cat(&batches[2], 0)?,
        ))
    }

    fn group(&self) -> usize {
        self.n_q / self.n_kv
    }

    fn scale(&self) -> f64 {
        1.0 / (self.head_dim as f64).sqrt()
    }

    fn q_grouped_f32(&self, q: &Tensor) -> Result<Tensor> {
        let (b, t, g, d) = (self.batch, self.seq, self.group(), self.head_dim);
        Ok(q.to_dtype(DType::F32)?
            .reshape((b, t, self.n_kv, g, d))?
            .permute((0, 2, 3, 1, 4))?
            .contiguous()?
            .reshape((b * self.n_kv, g * t, d))?)
    }

    fn q_ungrouped(&self, grouped: &Tensor) -> Result<Tensor> {
        let (b, t, g, d) = (self.batch, self.seq, self.group(), self.head_dim);
        Ok(grouped
            .reshape((b, self.n_kv, g, t, d))?
            .permute((0, 3, 1, 2, 4))?
            .contiguous()?
            .reshape((b, t, self.n_q, d))?)
    }

    fn kv_grouped_f32(&self, kv: &Tensor) -> Result<Tensor> {
        let (b, t, d) = (self.batch, self.seq, self.head_dim);
        Ok(kv
            .to_dtype(DType::F32)?
            .reshape((b, t, self.n_kv, d))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b * self.n_kv, t, d))?)
    }

    fn kv_ungrouped(&self, grouped: &Tensor) -> Result<Tensor> {
        let (b, t, d) = (self.batch, self.seq, self.head_dim);
        Ok(grouped
            .reshape((b, self.n_kv, t, d))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.n_kv, d))?)
    }

    fn probabilities(&self, qg: &Tensor, kg: &Tensor) -> Result<Tensor> {
        let (bn, gt, t) = (self.batch * self.n_kv, self.group() * self.seq, self.seq);
        let scores = qg
            .matmul(&kg.t()?)?
            .affine(self.scale(), 0.0)?
            .reshape((bn, self.group(), t, t))?
            .broadcast_add(&causal_window_mask(t, self.window, qg.device())?)?;
        Ok(candle_nn::ops::softmax_last_dim(&scores)?.reshape((bn, gt, t))?)
    }

    #[cfg(feature = "cuda")]
    fn q_grouped_keep_dtype(&self, q: &Tensor) -> Result<Tensor> {
        let (b, t, g, d) = (self.batch, self.seq, self.group(), self.head_dim);
        Ok(q.reshape((b, t, self.n_kv, g, d))?
            .permute((0, 2, 3, 1, 4))?
            .contiguous()?
            .reshape((b * self.n_kv, g * t, d))?)
    }

    #[cfg(feature = "cuda")]
    fn kv_grouped_keep_dtype(&self, kv: &Tensor) -> Result<Tensor> {
        let (b, t, d) = (self.batch, self.seq, self.head_dim);
        Ok(kv
            .reshape((b, t, self.n_kv, d))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b * self.n_kv, t, d))?)
    }

    #[cfg(feature = "cuda")]
    fn fused_softmax_serves(&self, k: &Tensor) -> bool {
        k.dtype() == DType::BF16 && k.device().is_cuda()
    }

    #[cfg(feature = "cuda")]
    fn scores_f32(&self, qg: &Tensor, kg: &Tensor) -> Result<Tensor> {
        Ok(qg.matmul(&kg.t()?)?)
    }

    #[cfg(feature = "cuda")]
    fn softmax_rows_bf16(&self, scores: &Tensor) -> Result<Tensor> {
        use cudarc::driver::DevicePtr;
        let candle_core::Device::Cuda(dev) = scores.device() else {
            anyhow::bail!("fused softmax needs cuda")
        };
        let stream = crate::cuda_stream::current_stream(dev);
        let rows = scores.elem_count() / self.seq;
        let scores = scores.contiguous()?;
        let p = crate::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
            scores.device(),
            DType::BF16,
            &[rows, self.seq],
        )?;
        let rc = {
            let (ss, sl) = scores.storage_and_layout();
            let candle_core::Storage::Cuda(sc) = &*ss else {
                anyhow::bail!("scores not cuda")
            };
            let (sp, _g) = sc.as_cuda_slice::<f32>()?.device_ptr(&stream);
            let sp = sp + (sl.start_offset() * 4) as u64;
            let pp = crate::step_arena::device_ptr_for_a_kernel_to_write(&p)?;
            unsafe {
                nv_kernels::cuda::attn_masked_softmax_rows_bf16(
                    stream.cu_stream() as *mut _,
                    sp as *const f32,
                    pp as *mut u16,
                    rows as i64,
                    self.seq as i32,
                    self.window.map_or(0, |w| w as i32),
                    self.scale() as f32,
                )
            }
        };
        anyhow::ensure!(rc == 0, "attn_masked_softmax_rows_bf16 rc={rc}");
        Ok(p)
    }

    #[cfg(feature = "cuda")]
    fn forward_detached_fused(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (bn, gt) = (self.batch * self.n_kv, self.group() * self.seq);
        let qg = self.q_grouped_f32(q)?;
        let kg = self.kv_grouped_f32(k)?;
        let vg = self.kv_grouped_keep_dtype(v)?;
        let p = self.softmax_rows_bf16(&self.scores_f32(&qg, &kg)?)?;
        let out = p.reshape((bn, gt, self.seq))?.matmul(&vg)?;
        Ok(self.q_ungrouped(&out)?.to_dtype(q.dtype())?)
    }

    #[cfg(feature = "cuda")]
    fn backward_detached_fused(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        out: &Tensor,
        dout: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        use cudarc::driver::DevicePtr;
        let candle_core::Device::Cuda(dev) = q.device() else {
            anyhow::bail!("fused softmax needs cuda")
        };
        let stream = crate::cuda_stream::current_stream(dev);
        let (bn, gt, t) = (self.batch * self.n_kv, self.group() * self.seq, self.seq);
        let qg = self.q_grouped_f32(q)?;
        let kg = self.kv_grouped_f32(k)?;
        let vg = self.kv_grouped_f32(v)?;
        let og = self.q_grouped_f32(out)?;
        let dog = self.q_grouped_f32(dout)?;
        let scores = self.scores_f32(&qg, &kg)?.contiguous()?;
        let dp = dog.matmul(&vg.t()?)?.contiguous()?;
        let delta = (&dog * &og)?.sum(D::Minus1)?.contiguous()?;
        let rows = bn * gt;
        let p = crate::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
            q.device(),
            DType::BF16,
            &[rows, t],
        )?;
        let ds = crate::step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
            q.device(),
            DType::BF16,
            &[rows, t],
        )?;
        let rc = {
            let f32_ptr = |x: &Tensor| -> Result<u64> {
                let (st, l) = x.storage_and_layout();
                let candle_core::Storage::Cuda(c) = &*st else {
                    anyhow::bail!("operand not cuda")
                };
                let (ptr, _g) = c.as_cuda_slice::<f32>()?.device_ptr(&stream);
                Ok(ptr + (l.start_offset() * 4) as u64)
            };
            let sp = f32_ptr(&scores)?;
            let dpp = f32_ptr(&dp)?;
            let dlp = f32_ptr(&delta)?;
            let pp = crate::step_arena::device_ptr_for_a_kernel_to_write(&p)?;
            let dsp = crate::step_arena::device_ptr_for_a_kernel_to_write(&ds)?;
            unsafe {
                nv_kernels::cuda::attn_masked_softmax_bwd_rows_bf16(
                    stream.cu_stream() as *mut _,
                    sp as *const f32,
                    dpp as *const f32,
                    dlp as *const f32,
                    pp as *mut u16,
                    dsp as *mut u16,
                    rows as i64,
                    t as i32,
                    self.window.map_or(0, |w| w as i32),
                    self.scale() as f32,
                )
            }
        };
        anyhow::ensure!(rc == 0, "attn_masked_softmax_bwd_rows_bf16 rc={rc}");
        let p = p.reshape((bn, gt, t))?;
        let ds = ds.reshape((bn, gt, t))?;
        let qb = self.q_grouped_keep_dtype(&q.to_dtype(DType::BF16)?)?;
        let kb = self.kv_grouped_keep_dtype(k)?;
        let dob = self.q_grouped_keep_dtype(&dout.to_dtype(DType::BF16)?)?;
        let dv = p.t()?.matmul(&dob)?;
        let dq = ds.matmul(&kb)?;
        let dk = ds.t()?.matmul(&qb)?;
        Ok((
            self.q_ungrouped(&dq)?.to_dtype(q.dtype())?,
            self.kv_ungrouped(&dk)?.to_dtype(k.dtype())?,
            self.kv_ungrouped(&dv)?.to_dtype(v.dtype())?,
        ))
    }

    pub fn forward_detached(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        if self.serial_kv_groups {
            return self.forward_serial(q, k, v);
        }
        #[cfg(feature = "cuda")]
        let _span = match q.device() {
            candle_core::Device::Cuda(d) => crate::prof_events::span(
                "fwd_gqa_attention_recompute",
                &crate::cuda_stream::current_stream(d),
            ),
            _ => None,
        };
        #[cfg(feature = "cuda")]
        if self.fused_softmax_serves(k) {
            return self.forward_detached_fused(q, k, v);
        }
        let qg = self.q_grouped_f32(q)?;
        let kg = self.kv_grouped_f32(k)?;
        let vg = self.kv_grouped_f32(v)?;
        let out = self.probabilities(&qg, &kg)?.matmul(&vg)?;
        Ok(self.q_ungrouped(&out)?.to_dtype(q.dtype())?)
    }

    pub fn backward_detached(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        out: &Tensor,
        dout: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        if self.serial_kv_groups {
            return self.backward_serial(q, k, v, out, dout);
        }
        #[cfg(feature = "cuda")]
        if self.fused_softmax_serves(k) {
            return self.backward_detached_fused(q, k, v, out, dout);
        }
        let qg = self.q_grouped_f32(q)?;
        let kg = self.kv_grouped_f32(k)?;
        let vg = self.kv_grouped_f32(v)?;
        let og = self.q_grouped_f32(out)?;
        let dog = self.q_grouped_f32(dout)?;
        let p = self.probabilities(&qg, &kg)?;
        let dv = p.t()?.matmul(&dog)?;
        let dp = dog.matmul(&vg.t()?)?;
        let delta = (&dog * &og)?.sum_keepdim(D::Minus1)?;
        let ds = (&p * dp.broadcast_sub(&delta)?)?;
        let dq = ds.matmul(&kg)?.affine(self.scale(), 0.0)?;
        let dk = ds.t()?.matmul(&qg)?.affine(self.scale(), 0.0)?;
        Ok((
            self.q_ungrouped(&dq)?.to_dtype(q.dtype())?,
            self.kv_ungrouped(&dk)?.to_dtype(k.dtype())?,
            self.kv_ungrouped(&dv)?.to_dtype(v.dtype())?,
        ))
    }

    fn q_shape(&self) -> Shape {
        Shape::from((self.batch, self.seq, self.n_q, self.head_dim))
    }

    fn kv_shape(&self) -> Shape {
        Shape::from((self.batch, self.seq, self.n_kv, self.head_dim))
    }
}

fn detached_tensor_of(
    storage: Storage,
    layout: &Layout,
    shape: Shape,
    clone_kept_the_offset: bool,
) -> candle_core::Result<Tensor> {
    let numel = shape.elem_count();
    if !layout.is_contiguous() {
        candle_core::bail!("gqa sdpa recompute: operands must be contiguous, got {layout:?}");
    }
    let offset = if clone_kept_the_offset {
        layout.start_offset()
    } else {
        0
    };
    let flat = Tensor::from_storage(
        storage,
        offset + numel,
        candle_core::op::BackpropOp::none(),
        false,
    );
    flat.narrow(0, offset, numel)?.reshape(shape)
}

fn storage_of(t: &Tensor) -> candle_core::Result<Storage> {
    let t = t.contiguous()?;
    let (st, l) = t.storage_and_layout();
    st.try_clone(l)
}

impl CustomOp3 for GqaSdpaRecomputePerKvHeadGroup {
    fn name(&self) -> &'static str {
        "gqa_sdpa_recompute_per_kv_head_group"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let q = detached_tensor_of(Storage::Cpu(s1.clone()), l1, self.q_shape(), true)?;
        let k = detached_tensor_of(Storage::Cpu(s2.clone()), l2, self.kv_shape(), true)?;
        let v = detached_tensor_of(Storage::Cpu(s3.clone()), l3, self.kv_shape(), true)?;
        let out = self
            .forward_detached(&q, &k, &v)
            .map_err(|e| candle_core::Error::Msg(format!("gqa sdpa recompute fwd: {e:#}")))?;
        match storage_of(&out)? {
            Storage::Cpu(s) => Ok((s, self.q_shape())),
            _ => candle_core::bail!("gqa sdpa recompute: cpu fwd produced a non-cpu storage"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        let q = detached_tensor_of(Storage::Cuda(s1.try_clone(l1)?), l1, self.q_shape(), false)?;
        let k = detached_tensor_of(Storage::Cuda(s2.try_clone(l2)?), l2, self.kv_shape(), false)?;
        let v = detached_tensor_of(Storage::Cuda(s3.try_clone(l3)?), l3, self.kv_shape(), false)?;
        let out = self
            .forward_detached(&q, &k, &v)
            .map_err(|e| candle_core::Error::Msg(format!("gqa sdpa recompute fwd: {e:#}")))?;
        match storage_of(&out)? {
            Storage::Cuda(s) => Ok((s, self.q_shape())),
            _ => candle_core::bail!("gqa sdpa recompute: cuda fwd produced a non-cuda storage"),
        }
    }

    fn bwd(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        res: &Tensor,
        grad_res: &Tensor,
    ) -> candle_core::Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        #[cfg(feature = "cuda")]
        let _span = match q.device() {
            candle_core::Device::Cuda(d) => crate::prof_events::span(
                "bwd_gqa_attention_recompute",
                &crate::cuda_stream::current_stream(d),
            ),
            _ => None,
        };
        let (dq, dk, dv) = self
            .backward_detached(
                &q.detach(),
                &k.detach(),
                &v.detach(),
                &res.detach(),
                &grad_res.detach(),
            )
            .map_err(|e| candle_core::Error::Msg(format!("gqa sdpa recompute bwd: {e:#}")))?;
        Ok((Some(dq), Some(dk), Some(dv)))
    }
}

pub fn gqa_sdpa_recompute(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    batch: usize,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    seq: usize,
    window: Option<usize>,
) -> Result<Tensor> {
    gqa_sdpa_recompute_with_group_serialization(
        q, k, v, batch, n_q, n_kv, head_dim, seq, window, false,
    )
}

pub fn gqa_sdpa_recompute_with_group_serialization(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    batch: usize,
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    seq: usize,
    window: Option<usize>,
    serial_kv_groups: bool,
) -> Result<Tensor> {
    anyhow::ensure!(
        n_kv > 0 && n_q % n_kv == 0,
        "gqa sdpa recompute: {n_q} query heads do not group over {n_kv} kv heads"
    );
    anyhow::ensure!(
        q.dims() == [batch, seq, n_q, head_dim] && k.dims() == [batch, seq, n_kv, head_dim] && v.dims() == k.dims(),
        "gqa sdpa recompute: q {:?} k {:?} v {:?} vs batch {batch} seq {seq} heads {n_q}/{n_kv} dim {head_dim}",
        q.dims(),
        k.dims(),
        v.dims()
    );
    let op = GqaSdpaRecomputePerKvHeadGroup {
        batch,
        n_q,
        n_kv,
        head_dim,
        seq,
        window,
        serial_kv_groups,
    };
    Ok(q.contiguous()?
        .apply_op3(&k.contiguous()?, &v.contiguous()?, op)?)
}
