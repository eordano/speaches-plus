use anyhow::{bail, Result};

pub const DEFAULT_GRAPH_SIZES: &[usize] = &[1, 2, 4, 8];

#[derive(Clone, Debug)]
pub struct BucketPlan {
    sizes: Vec<usize>,
}

impl BucketPlan {
    pub fn new(mut sizes: Vec<usize>) -> Self {
        sizes.retain(|&s| s > 0);
        sizes.sort_unstable();
        sizes.dedup();
        if sizes.is_empty() {
            sizes = DEFAULT_GRAPH_SIZES.to_vec();
        }
        Self { sizes }
    }

    pub fn parse(spec: Option<&str>) -> Self {
        let sizes = match spec {
            Some(s) => s
                .split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .collect(),
            None => Vec::new(),
        };
        Self::new(sizes)
    }

    pub fn from_env() -> Self {
        Self::parse(std::env::var("NV_BATCH_GRAPH_SIZES").ok().as_deref())
    }

    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    pub fn max_bucket(&self) -> usize {
        *self.sizes.last().unwrap()
    }

    pub fn bucket_for(&self, b: usize) -> Option<usize> {
        if b == 0 {
            return None;
        }
        self.sizes.iter().copied().find(|&s| s >= b)
    }
}

pub fn shape_token(b_bucket: usize) -> u64 {
    b_bucket as u64
}

pub fn is_uniform_decode(num_prefill_chunks: usize, num_decodes: usize) -> bool {
    num_prefill_chunks == 0 && num_decodes > 0
}

#[derive(Clone, Debug)]
pub struct SlotUpdate {
    pub token: u32,
    pub pos: i32,
    pub n_total: i32,
    pub block_table: Vec<u32>,

    pub lora_slot: i32,
}

pub const NO_LORA_SLOT: i32 = -1;

pub fn inactive_slot() -> SlotUpdate {
    SlotUpdate {
        token: 0,
        pos: 0,
        n_total: 1,
        block_table: Vec::new(),
        lora_slot: NO_LORA_SLOT,
    }
}

pub fn build_slot_table(seq_table: &[u32], scratch_block: u32, rows: usize) -> Result<Vec<i32>> {
    if seq_table.len() > rows {
        bail!(
            "build_slot_table: sequence table has {} rows, graph holds {rows}",
            seq_table.len()
        );
    }
    let mut out = Vec::with_capacity(rows);
    out.extend(seq_table.iter().map(|&b| b as i32));
    out.resize(rows, scratch_block as i32);
    Ok(out)
}

#[cfg(feature = "cuda")]
#[path = "graph_teardown.rs"]
pub mod graph_teardown;

#[cfg(feature = "cuda")]
#[path = "capture_stream.rs"]
pub mod capture_stream;

#[cfg(feature = "cuda")]
pub use graphed::Gemma4BatchGraphFamily;

#[cfg(feature = "cuda")]
mod graphed {
    use super::capture_stream::CaptureStream;
    use super::*;
    use crate::gemma4::{Gemma4, LayerType};
    use crate::paged_fp8::PagedKvFp8Pool;
    use anyhow::anyhow;
    use candle_core::{DType, Device, Tensor};
    use cudarc::driver::{CudaSlice, CudaStream};
    use half::bf16;
    use nv_kernels::graph::CudaGraphRunner;
    use nv_layers::lora_slots::LoraDispatch;
    use std::collections::{HashMap, HashSet};
    use std::sync::{Arc, Mutex};

    enum LoraMode {
        Uniform(i32),

        PerRow,
    }

    struct LoraArm {
        dispatch: Arc<LoraDispatch>,
        mode: LoraMode,
    }

    struct GraphSlot {
        table_dev: CudaSlice<i32>,
        start_dev: CudaSlice<i32>,
        n_total_dev: CudaSlice<i32>,
        host_table: Box<[i32]>,
        host_start: Box<[i32; 1]>,
        host_n_total: Box<[i32; 1]>,
    }

    pub struct Gemma4BatchGraphFamily {
        model: Arc<Gemma4>,
        pool: Arc<Mutex<PagedKvFp8Pool>>,
        device: Device,
        plan: BucketPlan,
        max_ctx: usize,
        max_rows: usize,
        vocab: usize,
        capture: CaptureStream,
        runners: HashMap<u64, CudaGraphRunner>,
        warmed: HashSet<u64>,
        slots: Vec<GraphSlot>,
        token_buf: CudaSlice<u32>,
        pos_buf: CudaSlice<i32>,
        host_tokens: Box<[u32]>,
        host_pos: Box<[i32]>,
        logits_buf: CudaSlice<bf16>,
        scratch: CudaSlice<f32>,
        fan_in: CudaSlice<u32>,
        scratch_base: u32,
        captures: u64,
        replays: u64,
        lora: Option<LoraArm>,
    }

    struct BodyArgs<'a> {
        model: &'a Gemma4,
        pool: &'a Mutex<PagedKvFp8Pool>,
        device: &'a Device,
        b: usize,
        slots: &'a mut [GraphSlot],
        token_buf: &'a mut CudaSlice<u32>,
        pos_buf: &'a mut CudaSlice<i32>,
        host_tokens: &'a [u32],
        host_pos: &'a [i32],
        logits_buf: &'a mut CudaSlice<bf16>,
        scratch: &'a mut CudaSlice<f32>,
        fan_in: &'a mut CudaSlice<u32>,
    }

    impl Gemma4BatchGraphFamily {
        pub fn new(
            model: Arc<Gemma4>,
            pool: Arc<Mutex<PagedKvFp8Pool>>,
            device: &Device,
            plan: BucketPlan,
            scratch_base: u32,
            max_ctx: usize,
        ) -> Result<Self> {
            if !matches!(device, Device::Cuda(_)) {
                bail!("Gemma4BatchGraphFamily requires a CUDA device");
            }
            if model.dtype() != DType::BF16 {
                bail!(
                    "Gemma4BatchGraphFamily: model dtype must be BF16, got {:?}",
                    model.dtype()
                );
            }
            let cfg = model.config().clone();
            let b_max = plan.max_bucket();
            let (block_size, pool_blocks, lanes) = {
                let p = pool
                    .lock()
                    .map_err(|_| anyhow!("Gemma4BatchGraphFamily: kv pool mutex poisoned"))?;
                let c = p.config();
                (c.block_size, c.num_blocks, c.lanes)
            };
            if lanes > 0 {
                bail!("Gemma4BatchGraphFamily: hybrid KV-ring lanes are not graphable; run eager");
            }
            if (scratch_base as usize) + b_max > pool_blocks {
                bail!(
                    "Gemma4BatchGraphFamily: need {} scratch blocks at base {} but pool has {}",
                    b_max,
                    scratch_base,
                    pool_blocks
                );
            }
            let max_ctx = max_ctx.max(block_size).next_power_of_two();
            if max_ctx % block_size != 0 {
                bail!(
                    "Gemma4BatchGraphFamily: max ctx {max_ctx} not a multiple of block size {block_size}"
                );
            }
            let max_rows = max_ctx.div_ceil(block_size);
            let n_q = cfg.num_attention_heads;
            let hd_max = cfg.head_dim.max(cfg.global_head_dim);
            let vocab = cfg.vocab_size;

            let capture = CaptureStream::for_device(device)?;
            capture.forked_candle_capture_is_an_asserted_coincidence(
                "Gemma4BatchGraphFamily",
                crate::gemma4_graph::GEMMA4_GRAPH_COINCIDENCE_GATE,
            );
            let capture_stream = capture.stream().clone();

            let alloc = |n: usize| -> Result<CudaSlice<bf16>> {
                capture_stream.alloc_zeros::<bf16>(n).map_err(|e| anyhow!(e))
            };
            let mut slots = Vec::with_capacity(b_max);
            for _ in 0..b_max {
                slots.push(GraphSlot {
                    table_dev: capture_stream
                        .alloc_zeros::<i32>(max_rows)
                        .map_err(|e| anyhow!(e))?,
                    start_dev: capture_stream.alloc_zeros::<i32>(1).map_err(|e| anyhow!(e))?,
                    n_total_dev: capture_stream.alloc_zeros::<i32>(1).map_err(|e| anyhow!(e))?,
                    host_table: vec![0i32; max_rows].into_boxed_slice(),
                    host_start: Box::new([0i32; 1]),
                    host_n_total: Box::new([1i32; 1]),
                });
            }
            let token_buf = capture_stream.alloc_zeros::<u32>(b_max).map_err(|e| anyhow!(e))?;
            let pos_buf = capture_stream.alloc_zeros::<i32>(b_max).map_err(|e| anyhow!(e))?;
            let logits_buf = alloc(b_max * vocab)?;
            let scratch_elems = crate::paged_fp8::flash_scratch_elems_for(n_q, hd_max);
            let scratch = capture_stream
                .alloc_zeros::<f32>(scratch_elems)
                .map_err(|e| anyhow!(e))?;
            let fan_in = capture_stream.alloc_zeros::<u32>(n_q).map_err(|e| anyhow!(e))?;
            capture_stream.synchronize().map_err(|e| anyhow!(e))?;

            Ok(Self {
                model,
                pool,
                device: device.clone(),
                plan,
                max_ctx,
                max_rows,
                vocab,
                capture,
                runners: HashMap::new(),
                warmed: HashSet::new(),
                slots,
                token_buf,
                pos_buf,
                host_tokens: vec![0u32; b_max].into_boxed_slice(),
                host_pos: vec![0i32; b_max].into_boxed_slice(),
                logits_buf,
                scratch,
                fan_in,
                scratch_base,
                captures: 0,
                replays: 0,
                lora: None,
            })
        }

        pub fn arm_lora(&mut self, dispatch: Arc<LoraDispatch>, slot: usize) -> Result<()> {
            if slot >= dispatch.max_loras() {
                bail!(
                    "arm_lora: slot {slot} out of range for max_loras {}",
                    dispatch.max_loras()
                );
            }
            if dispatch.max_tokens() < self.plan.max_bucket() {
                bail!(
                    "arm_lora: dispatch max_tokens {} < max graph bucket {}",
                    dispatch.max_tokens(),
                    self.plan.max_bucket()
                );
            }
            self.lora = Some(LoraArm {
                dispatch,
                mode: LoraMode::Uniform(slot as i32),
            });
            Ok(())
        }

        pub fn arm_lora_multi(&mut self, dispatch: Arc<LoraDispatch>) -> Result<()> {
            if dispatch.max_tokens() < self.plan.max_bucket() {
                bail!(
                    "arm_lora_multi: dispatch max_tokens {} < max graph bucket {}",
                    dispatch.max_tokens(),
                    self.plan.max_bucket()
                );
            }
            self.lora = Some(LoraArm {
                dispatch,
                mode: LoraMode::PerRow,
            });
            Ok(())
        }

        pub fn disarm_lora(&mut self) {
            if let Some(arm) = self.lora.take() {
                arm.dispatch.disarm();
            }
        }

        pub fn lora_armed(&self) -> bool {
            self.lora.is_some()
        }

        pub fn plan(&self) -> &BucketPlan {
            &self.plan
        }
        pub fn max_ctx(&self) -> usize {
            self.max_ctx
        }
        pub fn captures(&self) -> u64 {
            self.captures
        }
        pub fn replays(&self) -> u64 {
            self.replays
        }

        pub fn node_count(&self) -> usize {
            self.runners.values().map(|r| r.cached_node_count()).sum()
        }

        pub fn accepts(&self, batch: usize, max_n_total: usize) -> bool {
            self.plan.bucket_for(batch).is_some() && max_n_total <= self.max_ctx
        }

        pub fn step(&mut self, active: &[SlotUpdate]) -> Result<Vec<Vec<f32>>> {
            if active.is_empty() {
                bail!("Gemma4BatchGraphFamily.step: empty batch");
            }
            let b_bucket = self
                .plan
                .bucket_for(active.len())
                .ok_or_else(|| anyhow!("batch {} exceeds max graph bucket", active.len()))?;
            let max_total = active
                .iter()
                .map(|u| u.n_total.max(1) as usize)
                .max()
                .unwrap();
            if max_total > self.max_ctx {
                bail!(
                    "Gemma4BatchGraphFamily.step: n_total {max_total} exceeds max ctx {}",
                    self.max_ctx
                );
            }
            let token = shape_token(b_bucket);

            if let Some(arm) = &self.lora {
                let mapping: Vec<i32> = match arm.mode {
                    LoraMode::Uniform(slot) => vec![slot; b_bucket],
                    LoraMode::PerRow => {
                        let mut m = vec![NO_LORA_SLOT; b_bucket];
                        for (i, u) in active.iter().enumerate() {
                            m[i] = u.lora_slot;
                        }
                        m
                    }
                };
                arm.dispatch
                    .set_mapping(&mapping)
                    .map_err(|e| anyhow!("arm lora mapping: {e}"))?;
            }

            let pad = inactive_slot();
            for i in 0..b_bucket {
                let u = if i < active.len() { &active[i] } else { &pad };
                self.host_tokens[i] = u.token;
                self.host_pos[i] = u.pos;
                let slot = &mut self.slots[i];
                slot.host_start[0] = u.pos;
                slot.host_n_total[0] = u.n_total.max(1);
                let table =
                    build_slot_table(&u.block_table, self.scratch_base + i as u32, self.max_rows)?;
                slot.host_table.copy_from_slice(&table);
            }

            let dev = match &self.device {
                Device::Cuda(d) => d.clone(),
                _ => unreachable!(),
            };
            let ctx = dev.cuda_stream().context().clone();
            if ctx.is_event_tracking() {
                unsafe { ctx.disable_event_tracking() };
            }
            dev.cuda_stream()
                .synchronize()
                .map_err(|e| anyhow!("null-stream sync before graph step: {e:?}"))?;

            let capture_stream = self.capture.stream().clone();
            let need_warm = !self.warmed.contains(&token);
            let Gemma4BatchGraphFamily {
                model,
                pool,
                device,
                slots,
                token_buf,
                pos_buf,
                host_tokens,
                host_pos,
                logits_buf,
                scratch,
                fan_in,
                runners,
                ..
            } = self;
            let mut args = BodyArgs {
                model: model.as_ref(),
                pool: pool.as_ref(),
                device: &*device,
                b: b_bucket,
                slots: &mut slots[..],
                token_buf,
                pos_buf,
                host_tokens: &host_tokens[..],
                host_pos: &host_pos[..],
                logits_buf,
                scratch,
                fan_in,
            };

            if need_warm {
                nv_layers::cuda_stream::with_stream(capture_stream.clone(), || {
                    run_body(&capture_stream, &mut args)
                })?;
                capture_stream.synchronize().map_err(|e| anyhow!(e))?;
                self.warmed.insert(token);
            }

            let runner = runners
                .entry(token)
                .or_insert_with(|| CudaGraphRunner::new(capture_stream.clone()));
            let was_cached = runner.has_cached();
            runner.run(token, |s| {
                nv_layers::cuda_stream::with_stream(s.clone(), || run_body(s, &mut args))
            })?;
            if was_cached {
                self.replays += 1;
            } else {
                self.captures += 1;
            }
            self.capture.stream().synchronize().map_err(|e| anyhow!(e))?;

            let host: Vec<bf16> = self
                .capture
                .stream()
                .clone_dtoh(&self.logits_buf)
                .map_err(|e| anyhow!(e))?;
            let mut out = Vec::with_capacity(active.len());
            for i in 0..active.len() {
                let row = &host[i * self.vocab..(i + 1) * self.vocab];
                out.push(row.iter().map(|x| x.to_f32()).collect());
            }
            Ok(out)
        }
    }

    impl Drop for Gemma4BatchGraphFamily {
        fn drop(&mut self) {
            let td = super::graph_teardown::GraphTeardown::for_capture(&self.capture);
            let runners = &mut self.runners;
            td.run(|| {
                for (_, r) in runners.iter_mut() {
                    r.invalidate();
                }
            });
        }
    }

    fn run_body(s: &Arc<CudaStream>, a: &mut BodyArgs) -> Result<()> {
        let b = a.b;
        let dev = match a.device {
            Device::Cuda(d) => d.clone(),
            _ => bail!("cuda device required"),
        };

        {
            let mut dst = a.token_buf.slice_mut(0..b);
            s.memcpy_htod(&a.host_tokens[..b], &mut dst)
                .map_err(|e| anyhow!("htod tokens: {e:?}"))?;
        }
        {
            let mut dst = a.pos_buf.slice_mut(0..b);
            s.memcpy_htod(&a.host_pos[..b], &mut dst)
                .map_err(|e| anyhow!("htod pos: {e:?}"))?;
        }
        for slot in a.slots[..b].iter_mut() {
            s.memcpy_htod(&slot.host_table[..], &mut slot.table_dev)
                .map_err(|e| anyhow!("htod table: {e:?}"))?;
            s.memcpy_htod(&slot.host_start[..], &mut slot.start_dev)
                .map_err(|e| anyhow!("htod start: {e:?}"))?;
            s.memcpy_htod(&slot.host_n_total[..], &mut slot.n_total_dev)
                .map_err(|e| anyhow!("htod n_total: {e:?}"))?;
        }

        let mut tok_b = unsafe { s.alloc::<u32>(b).map_err(|e| anyhow!(e))? };
        s.memcpy_dtod(&a.token_buf.slice(0..b), &mut tok_b)
            .map_err(|e| anyhow!(e))?;
        let mut pos_b = unsafe { s.alloc::<i32>(b).map_err(|e| anyhow!(e))? };
        s.memcpy_dtod(&a.pos_buf.slice(0..b), &mut pos_b)
            .map_err(|e| anyhow!(e))?;
        let tokens_t = {
            let st = candle_core::CudaStorage::wrap_cuda_slice(tok_b, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(st),
                (b,),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        let positions_t = {
            let st = candle_core::CudaStorage::wrap_cuda_slice(pos_b, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(st),
                (b,),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };

        let model = a.model;
        let cfg = model.config();
        let hidden_size = cfg.hidden_size;
        let x_flat =
            crate::gemma4::embed_lookup_bf16_op(model.embed_weight(), &tokens_t, a.device)?;
        let x = x_flat.reshape((1usize, b, hidden_size))?;
        let mut hidden = crate::gemma4::scale_bf16_op(&x, model.embed_scale(), a.device)?;

        for li in 0..model.layers().len() {
            hidden = layer_body(s, a, &dev, li, &hidden, &positions_t)?;
        }

        let normed = model.final_norm().forward(&hidden)?;
        let logits = model.lm_head().forward(&normed)?;
        let lc = logits.contiguous()?;
        let (ls, ll) = lc.storage_and_layout();
        let lcuda = match &*ls {
            candle_core::Storage::Cuda(c) => c,
            _ => bail!("logits must be on CUDA"),
        };
        let lsl = lcuda.as_cuda_slice::<bf16>()?;
        let n = b * cfg.vocab_size;
        let off = ll.start_offset();
        let src = lsl.slice(off..off + n);
        let mut dst = a.logits_buf.slice_mut(0..n);
        s.memcpy_dtod(&src, &mut dst).map_err(|e| anyhow!(e))?;
        Ok(())
    }

    fn layer_body(
        s: &Arc<CudaStream>,
        a: &mut BodyArgs,
        dev: &candle_core::CudaDevice,
        li: usize,
        x: &Tensor,
        positions_t: &Tensor,
    ) -> Result<Tensor> {
        let model = a.model;
        let cfg = model.config();
        let b = a.b;
        let layer = &model.layers()[li];
        let attn = &layer.self_attn;
        let kind = attn.kind;
        let head_dim = cfg.head_dim_for(kind);
        let n_q = cfg.num_attention_heads;
        let n_kv = cfg.num_kv_heads_for(kind);
        let rope = match kind {
            LayerType::SlidingAttention => model.sliding_rope(),
            LayerType::FullAttention => model.full_rope(),
        };
        let window = match kind {
            LayerType::SlidingAttention => Some(cfg.sliding_window),
            LayerType::FullAttention => None,
        };

        let residual_attn = x.clone();
        let normed_pre_attn = layer.input_layernorm.forward(x)?;
        let (q_raw, k_raw, v_raw) = attn.qkv_forward(&normed_pre_attn)?;
        let q = q_raw.reshape((1usize, b, n_q, head_dim))?;
        let q_normed = attn.q_norm.forward(&q)?;
        let k = k_raw.reshape((1usize, b, n_kv, head_dim))?;
        let k_normed = attn.k_norm.forward(&k)?;
        let v = v_raw.reshape((1usize, b, n_kv, head_dim))?;
        let v_normed = attn.v_norm.forward(&v)?;

        let (q_rot, k_rot) = rope.apply(&q_normed, &k_normed, positions_t)?;
        let (q_rot, k_rot) = crate::hadamard_kv::maybe_rotate_qk(q_rot, k_rot, head_dim)?;
        let q_rot = q_rot.contiguous()?;
        let k_rot = k_rot.contiguous()?;
        let v_for_cache = v_normed.contiguous()?;

        let row_elems = n_q * head_dim;
        let mut out_slice = unsafe { s.alloc::<bf16>(b * row_elems).map_err(|e| anyhow!(e))? };

        let (q_storage, q_layout) = q_rot.storage_and_layout();
        let q_cuda = match &*q_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => bail!("q must be on CUDA"),
        };
        let q_slice = q_cuda.as_cuda_slice::<bf16>()?;
        let q_off = q_layout.start_offset();

        for i in 0..b {
            let k_i = k_rot.narrow(1, i, 1)?;
            let v_i = v_for_cache.narrow(1, i, 1)?;
            let slot = &a.slots[i];
            let mut p = a
                .pool
                .lock()
                .map_err(|_| anyhow!("Gemma4BatchGraphFamily: kv pool mutex poisoned"))?;
            let block_size = p.config().block_size;
            p.append_layer(li, &k_i, &v_i, 1, &slot.start_dev, &slot.table_dev)?;
            p.decode_attention_paged_into(
                li,
                q_slice,
                q_off + i * row_elems,
                &mut out_slice,
                i * row_elems,
                n_q,
                &slot.table_dev,
                block_size,
                &slot.n_total_dev,
                a.scratch,
                a.fan_in,
                window,
                1.0,
            )?;
        }

        let attn_flat = {
            let st = candle_core::CudaStorage::wrap_cuda_slice(out_slice, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(st),
                (1usize, b, row_elems),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        let attn_out = attn.o_proj.forward(&attn_flat)?;
        let attn_post = layer.post_attention_layernorm.forward(&attn_out)?;
        let (normed_pre_mlp, after_attn) = layer
            .pre_feedforward_layernorm
            .forward_residual(&attn_post, &residual_attn)?;
        let residual_mlp = after_attn.clone();
        let mlp_out = crate::gemma4::mlp_forward(&layer.mlp, &normed_pre_mlp)?;
        let mlp_post = layer.post_feedforward_layernorm.forward(&mlp_out)?;
        crate::gemma4::residual_add_scale_bf16_op(
            &residual_mlp,
            &mlp_post,
            layer.layer_scalar_host,
            a.device,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_pads_to_smallest_ge() {
        let plan = BucketPlan::new(vec![1, 2, 4, 8]);
        assert_eq!(plan.bucket_for(1), Some(1));
        assert_eq!(plan.bucket_for(2), Some(2));
        assert_eq!(plan.bucket_for(3), Some(4));
        assert_eq!(plan.bucket_for(5), Some(8));
        assert_eq!(plan.bucket_for(8), Some(8));
        assert_eq!(plan.bucket_for(9), None);
        assert_eq!(plan.bucket_for(0), None);
    }

    #[test]
    fn graph_sizes_env_parsing() {
        assert_eq!(BucketPlan::parse(None).sizes(), DEFAULT_GRAPH_SIZES);
        assert_eq!(BucketPlan::parse(Some("2,6,2")).sizes(), &[2, 6]);
        assert_eq!(BucketPlan::parse(Some("0,-1,x,3")).sizes(), &[3]);
        assert_eq!(BucketPlan::parse(Some("0,x")).sizes(), DEFAULT_GRAPH_SIZES);
        assert_eq!(BucketPlan::parse(Some("")).sizes(), DEFAULT_GRAPH_SIZES);
        assert_eq!(BucketPlan::parse(Some("8,4,2,1")).sizes(), &[1, 2, 4, 8]);
    }

    #[test]
    fn slot_table_padding() {
        assert_eq!(
            build_slot_table(&[5, 9], 42, 4).unwrap(),
            vec![5, 9, 42, 42]
        );
        assert_eq!(build_slot_table(&[], 7, 3).unwrap(), vec![7, 7, 7]);
        assert!(build_slot_table(&[1, 2, 3], 42, 2).is_err());
    }

    #[test]
    fn inactive_slot_never_zero_len() {
        let s = inactive_slot();
        assert_eq!(s.n_total, 1);
        assert_eq!(s.pos, 0);
        assert!(s.block_table.is_empty());
    }

    #[test]
    fn shape_token_is_one_graph_per_batch_size() {
        let mut seen = std::collections::HashSet::new();
        for &b in &[1usize, 2, 4, 8] {
            assert!(seen.insert(shape_token(b)), "batch {b} collided with an earlier graph");
        }
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn uniform_decode_predicate() {
        assert!(is_uniform_decode(0, 1));
        assert!(is_uniform_decode(0, 8));
        assert!(!is_uniform_decode(1, 4));
        assert!(!is_uniform_decode(2, 0));
        assert!(!is_uniform_decode(0, 0));
    }
}
