use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use cudarc::driver::{CudaSlice, CudaStream};
use nv_kernels::graph::CudaGraphRunner;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::gemma4_batch_graph::capture_stream::CaptureStream;
use crate::gemma4_batch_graph::graph_teardown::GraphTeardown;
use crate::gemma4_vision::{
    full_grid_positions, vision_prof_enabled, Gemma4VisionTower,
};

pub const VISION_GRAPH_BUCKET_CAP_BOUNDS_GRAPH_MEMPOOL_ACROSS_RESOLUTIONS: usize = 8;

pub const VISION_GRAPH_BUCKET_BYTE_BUDGET_EVICTS_LARGEST_BECAUSE_A_MISS_ONLY_COSTS_A_RECAPTURE:
    usize = 256 * 1024 * 1024;

const VISION_GRAPH_SHAPE_TOKEN_BASE_KEEPS_DECODE_AND_VERIFY_KEYS_UNTOUCHED: u64 = 0x76000000;

pub(crate) enum BucketAdmission {
    RefusedBecauseOneBucketAloneExceedsTheBudget,
    Admit {
        evict_largest_first: Vec<(usize, usize)>,
    },
}

pub(crate) fn plan_bucket_admission(
    resident: &[((usize, usize), usize)],
    incoming_bytes: usize,
    budget_bytes: usize,
) -> BucketAdmission {
    if incoming_bytes > budget_bytes {
        return BucketAdmission::RefusedBecauseOneBucketAloneExceedsTheBudget;
    }
    let mut total: usize = resident.iter().map(|(_, b)| *b).sum();
    let mut by_size: Vec<((usize, usize), usize)> = resident.to_vec();
    by_size.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut evict_largest_first = Vec::new();
    for (grid, bytes) in by_size {
        if total + incoming_bytes <= budget_bytes {
            break;
        }
        evict_largest_first.push(grid);
        total -= bytes;
    }
    BucketAdmission::Admit {
        evict_largest_first,
    }
}

pub const THE_CAPTURED_BODY_ALLOWS_ONLY_LAYOUT_INFO_FREE_OPS: &str =
    "candle uploads dims/strides for index_select, gather, reduce, and every op on a \
     non-contiguous layout via clone_htod of a TEMPORARY host Vec (SlicePtrOrNull, IndexSelect's \
     `ds`); a captured graph bakes that memcpy node's HOST pointer and every replay re-reads the \
     freed Vec, which compute-sanitizer catches as is_u32_bf16 reading gigabytes past its \
     allocation. candle-flash-attn additionally launches on the legacy NULL stream, which a \
     capture cannot contain. So the captured tower body is rebuilt from ops that carry their \
     layout by value: cublas matmuls on contiguous or plainly-transposed 2-D operands, same-shape \
     elementwise ops, softmax_last_dim, the nv-* RmsNorm and TensorCoreGemm kernels, and \
     plan-precomputed tensors for everything positional (position-embedding sum, full-width rope \
     cos/sin, a rotate-half matrix, a mean-pool matrix). Gathers, permutes, broadcasts and flash \
     stay in the eager path.";

pub fn vision_graph_enabled() -> bool {
    std::env::var("NV_VISION_GRAPH").ok().as_deref() == Some("1")
}

struct CaptureAids {
    pos_sum: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    rope_rot: Tensor,
    pool: Tensor,
}

struct LayerHeads {
    q_t: Vec<Tensor>,
    k_t: Vec<Tensor>,
    v_t: Vec<Tensor>,
    o_t: Vec<Tensor>,
}

struct Bucket {
    key: u64,
    host_pixels: cudarc::driver::PinnedHostSlice<f32>,
    input: CudaSlice<f32>,
    output: CudaSlice<f32>,
    aids: CaptureAids,
    cells: usize,
    resident_bytes: usize,
}

struct GraphState {
    runner: CudaGraphRunner,
    buckets: HashMap<(usize, usize), Bucket>,
    next_key: u64,
    oversize_grids_stay_eager: std::collections::HashSet<(usize, usize)>,
    head_weights: Option<Arc<Vec<LayerHeads>>>,
    capture_failed: bool,
    captures: u64,
    replays: u64,
}

pub struct Gemma4VisionGraph {
    device: candle_core::CudaDevice,
    capture: CaptureStream,
    stream: Arc<CudaStream>,
    bucket_byte_budget: usize,
    state: Mutex<GraphState>,
}

fn bucket_resident_bytes(n: usize, pp: usize, cells: usize, text_hidden: usize, aids: &CaptureAids) -> usize {
    let f32b = std::mem::size_of::<f32>();
    let pinned_host = n * pp * f32b;
    let device_io = n * pp * f32b + cells * text_hidden * f32b;
    let aids_device: usize = [
        &aids.pos_sum,
        &aids.rope_cos,
        &aids.rope_sin,
        &aids.rope_rot,
        &aids.pool,
    ]
    .iter()
    .map(|t| t.elem_count() * t.dtype().size_in_bytes())
    .sum();
    pinned_host + device_io + aids_device
}

fn capture_supported_for(tower: &Gemma4VisionTower) -> Result<()> {
    anyhow::ensure!(
        matches!(tower.dtype, DType::BF16 | DType::F16),
        "vision graph capture needs a bf16/f16 tower (NV_MM_VISION_DTYPE=bf16); an f32 tower's \
         linears route through a per-call transposed-weight copy that is not layout-info-free"
    );
    anyhow::ensure!(
        tower.patch_embedder.input_proj.dense_weight().is_some()
            && tower.embedding_projection.dense_weight().is_some(),
        "vision graph capture needs dense weights; the int8 vision mode dequantizes per call"
    );
    for layer in &tower.layers {
        for lin in [
            &layer.q_proj,
            &layer.k_proj,
            &layer.v_proj,
            &layer.o_proj,
            &layer.gate_proj,
            &layer.up_proj,
            &layer.down_proj,
        ] {
            anyhow::ensure!(
                lin.dense_weight().is_some(),
                "vision graph capture needs dense weights; the int8 vision mode dequantizes per \
                 call"
            );
        }
    }
    Ok(())
}

fn clamp_by_value_in_f32_because_the_shift_by_the_bound_eats_bf16_mantissa(
    x: &Tensor,
    (lo, hi): (f64, f64),
) -> Result<Tensor> {
    let dtype = x.dtype();
    Ok(x
        .to_dtype(DType::F32)?
        .affine(1.0, -lo)?
        .relu()?
        .affine(-1.0, hi - lo)?
        .relu()?
        .affine(-1.0, hi)?
        .to_dtype(dtype)?)
}

fn maybe_clamp(x: &Tensor, clip: Option<(f64, f64)>) -> Result<Tensor> {
    match clip {
        Some(c) => clamp_by_value_in_f32_because_the_shift_by_the_bound_eats_bf16_mantissa(x, c),
        None => Ok(x.clone()),
    }
}

fn clipped_forward_by_value(
    lin: &crate::gemma4_vision::ClippedLinear,
    x: &Tensor,
) -> Result<Tensor> {
    let x = maybe_clamp(x, lin.input_clip())?;
    let y = lin.linear.forward(&x)?;
    maybe_clamp(&y, lin.output_clip())
}

fn build_head_weights(tower: &Gemma4VisionTower) -> Result<Vec<LayerHeads>> {
    let cfg = tower.config();
    let nh = cfg.num_attention_heads;
    let hd = cfg.head_dim;
    let mut out = Vec::with_capacity(tower.layers.len());
    for layer in &tower.layers {
        let q_w = layer.q_proj.dense_weight().context("q weight")?;
        let k_w = layer.k_proj.dense_weight().context("k weight")?;
        let v_w = layer.v_proj.dense_weight().context("v weight")?;
        let o_w = layer.o_proj.dense_weight().context("o weight")?;
        let mut lh = LayerHeads {
            q_t: Vec::with_capacity(nh),
            k_t: Vec::with_capacity(nh),
            v_t: Vec::with_capacity(nh),
            o_t: Vec::with_capacity(nh),
        };
        for h in 0..nh {
            lh.q_t
                .push(q_w.narrow(0, h * hd, hd)?.t()?.contiguous()?);
            lh.k_t
                .push(k_w.narrow(0, h * hd, hd)?.t()?.contiguous()?);
            lh.v_t
                .push(v_w.narrow(0, h * hd, hd)?.t()?.contiguous()?);
            lh.o_t
                .push(o_w.narrow(1, h * hd, hd)?.t()?.contiguous()?);
        }
        out.push(lh);
    }
    Ok(out)
}

fn build_capture_aids(
    tower: &Gemma4VisionTower,
    grid_w: usize,
    grid_h: usize,
    device: &Device,
) -> Result<CaptureAids> {
    let cfg = tower.config();
    let hd = cfg.head_dim;
    let axis = hd / 2;
    let half = axis / 2;
    let pk = cfg.pooling_kernel_size;
    let n = grid_w * grid_h;
    let theta = cfg.rope_theta();

    let positions = full_grid_positions(grid_w, grid_h);
    let pos_sum = tower
        .patch_embedder
        .position_sum_for(&positions, device)?
        .to_dtype(tower.dtype)?;

    let inv_freq: Vec<f32> = (0..half)
        .map(|j| theta.powf(-((2 * j) as f32) / axis as f32))
        .collect();
    let mut cos = vec![0f32; n * hd];
    let mut sin = vec![0f32; n * hd];
    for (i, &(x, y)) in positions.iter().enumerate() {
        for j in 0..half {
            let ax = x.max(0) as f32 * inv_freq[j];
            let ay = y.max(0) as f32 * inv_freq[j];
            for &(base, angle) in &[(0usize, ax), (axis, ay)] {
                cos[i * hd + base + j] = angle.cos();
                cos[i * hd + base + half + j] = angle.cos();
                sin[i * hd + base + j] = angle.sin();
                sin[i * hd + base + half + j] = angle.sin();
            }
        }
    }
    let rope_cos = Tensor::from_vec(cos, (n, hd), device)?;
    let rope_sin = Tensor::from_vec(sin, (n, hd), device)?;

    let mut rot = vec![0f32; hd * hd];
    for base in [0usize, axis] {
        for j in 0..half {
            rot[(base + half + j) * hd + base + j] = -1.0;
            rot[(base + j) * hd + base + half + j] = 1.0;
        }
    }
    let rope_rot = Tensor::from_vec(rot, (hd, hd), device)?;

    let cells_w = grid_w / pk;
    let cells_h = grid_h / pk;
    let cells = cells_w * cells_h;
    let mut pool = vec![0f32; cells * n];
    let inv = 1.0f32 / (pk * pk) as f32;
    for (i, &(x, y)) in positions.iter().enumerate() {
        let c = (y as usize / pk) * cells_w + (x as usize / pk);
        pool[c * n + i] = inv;
    }
    let pool = Tensor::from_vec(pool, (cells, n), device)?.to_dtype(tower.dtype)?;

    Ok(CaptureAids {
        pos_sum,
        rope_cos,
        rope_sin,
        rope_rot,
        pool,
    })
}

fn rope_by_value(t: &Tensor, aids: &CaptureAids) -> Result<Tensor> {
    Ok(t
        .mul(&aids.rope_cos)?
        .add(&t.matmul(&aids.rope_rot)?.mul(&aids.rope_sin)?)?)
}

fn forward_capture_body(
    tower: &Gemma4VisionTower,
    pv: &Tensor,
    aids: &CaptureAids,
    heads: &[LayerHeads],
) -> Result<Tensor> {
    let _ = THE_CAPTURED_BODY_ALLOWS_ONLY_LAYOUT_INFO_FREE_OPS;
    let cfg = tower.config();
    let nh = cfg.num_attention_heads;
    let hd = cfg.head_dim;
    let scale = 1.0 / (hd as f64).sqrt();

    let pv = pv.to_dtype(tower.dtype)?;
    let mut x = tower
        .patch_embedder
        .input_proj
        .forward(&pv)?
        .add(&aids.pos_sum)?;

    for (layer, lh) in tower.layers.iter().zip(heads.iter()) {
        let normed = layer.input_layernorm.forward(&x)?;
        let q_in = maybe_clamp(&normed, layer.q_proj.input_clip())?;
        let k_in = maybe_clamp(&normed, layer.k_proj.input_clip())?;
        let v_in = maybe_clamp(&normed, layer.v_proj.input_clip())?;
        let mut acc: Option<Tensor> = None;
        for h in 0..nh {
            let q = maybe_clamp(&q_in.matmul(&lh.q_t[h])?, layer.q_proj.output_clip())?;
            let k = maybe_clamp(&k_in.matmul(&lh.k_t[h])?, layer.k_proj.output_clip())?;
            let v = maybe_clamp(&v_in.matmul(&lh.v_t[h])?, layer.v_proj.output_clip())?
                .to_dtype(DType::F32)?;
            let q = layer.q_norm.forward(&q)?;
            let k = layer.k_norm.forward(&k)?;
            let q = rope_by_value(&q.to_dtype(DType::F32)?, aids)?;
            let k = rope_by_value(&k.to_dtype(DType::F32)?, aids)?;
            let scores = (q.matmul(&k.t()?)? * scale)?;
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            let out = probs.matmul(&v)?.to_dtype(tower.dtype)?;
            let out = maybe_clamp(&out, layer.o_proj.input_clip())?;
            let part = out.matmul(&lh.o_t[h])?;
            acc = Some(match acc {
                Some(a) => a.add(&part)?,
                None => part,
            });
        }
        let attn = maybe_clamp(
            &acc.context("tower with zero attention heads")?,
            layer.o_proj.output_clip(),
        )?;
        let attn = layer.post_attention_layernorm.forward(&attn)?;
        let x1 = x.add(&attn)?;
        let normed = layer.pre_feedforward_layernorm.forward(&x1)?;
        let gate = clipped_forward_by_value(&layer.gate_proj, &normed)?.gelu()?;
        let up = clipped_forward_by_value(&layer.up_proj, &normed)?;
        let mlp = clipped_forward_by_value(&layer.down_proj, &(gate * up)?)?;
        let mlp = layer.post_feedforward_layernorm.forward(&mlp)?;
        x = x1.add(&mlp)?;
    }

    let pooled = aids.pool.matmul(&x)?;
    let normed = tower.embed_pre_projection_norm.forward(&pooled)?;
    tower.embedding_projection.forward(&normed)
}

impl Gemma4VisionGraph {
    pub fn new(device: &Device) -> Result<Self> {
        Self::with_bucket_byte_budget(
            device,
            VISION_GRAPH_BUCKET_BYTE_BUDGET_EVICTS_LARGEST_BECAUSE_A_MISS_ONLY_COSTS_A_RECAPTURE,
        )
    }

    pub fn with_bucket_byte_budget(device: &Device, bucket_byte_budget: usize) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("Gemma4VisionGraph requires a CUDA device"),
        };
        let capture = CaptureStream::for_device(device)?;
        capture.require_capture_of_a_candle_body("Gemma4VisionGraph")?;
        let stream = capture.stream().clone();
        let runner = CudaGraphRunner::new(stream.clone());
        Ok(Self {
            device: dev,
            capture,
            stream,
            bucket_byte_budget,
            state: Mutex::new(GraphState {
                runner,
                buckets: HashMap::new(),
                next_key: VISION_GRAPH_SHAPE_TOKEN_BASE_KEEPS_DECODE_AND_VERIFY_KEYS_UNTOUCHED,
                oversize_grids_stay_eager: std::collections::HashSet::new(),
                head_weights: None,
                capture_failed: false,
                captures: 0,
                replays: 0,
            }),
        })
    }

    pub fn bucket_count(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .buckets
            .len()
    }

    pub fn resident_bucket_bytes(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .buckets
            .values()
            .map(|b| b.resident_bytes)
            .sum()
    }

    pub fn capture_active(&self) -> bool {
        let s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        s.captures > 0 && !s.capture_failed
    }

    pub fn captures(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .captures
    }

    pub fn replays(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .replays
    }

    fn forward_eager(
        &self,
        tower: &Gemma4VisionTower,
        pixels: &[f32],
        grid_w: usize,
        grid_h: usize,
    ) -> Result<Tensor> {
        let pp = tower.config().patch_pixels();
        let n = grid_w * grid_h;
        let pv = Tensor::from_slice(pixels, (n, pp), &Device::Cuda(self.device.clone()))?;
        let plan = tower.plan_full_grid(grid_w, grid_h)?;
        tower.forward_full_grid(&pv, &plan)
    }

    pub fn forward(
        &self,
        tower: &Gemma4VisionTower,
        pixels: &[f32],
        grid_w: usize,
        grid_h: usize,
    ) -> Result<Tensor> {
        let pp = tower.config().patch_pixels();
        let pk = tower.config().pooling_kernel_size;
        let n = grid_w * grid_h;
        anyhow::ensure!(
            pixels.len() == n * pp,
            "vision graph forward: {} pixels for grid {grid_w}x{grid_h} needing {}",
            pixels.len(),
            n * pp
        );
        if vision_prof_enabled() {
            return self.forward_eager(tower, pixels, grid_w, grid_h);
        }
        let text_hidden = tower.text_hidden_size();
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if !state.capture_failed && state.head_weights.is_none() {
            match capture_supported_for(tower).and_then(|()| build_head_weights(tower)) {
                Ok(hw) => state.head_weights = Some(Arc::new(hw)),
                Err(e) => {
                    state.capture_failed = true;
                    eprintln!("[vision_graph] capture unsupported; encoding eager: {e:#}");
                }
            }
        }
        if state.capture_failed
            || !grid_w.is_multiple_of(pk)
            || !grid_h.is_multiple_of(pk)
            || state.oversize_grids_stay_eager.contains(&(grid_w, grid_h))
            || (!state.buckets.contains_key(&(grid_w, grid_h))
                && state.buckets.len()
                    >= VISION_GRAPH_BUCKET_CAP_BOUNDS_GRAPH_MEMPOOL_ACROSS_RESOLUTIONS)
        {
            drop(state);
            return self.forward_eager(tower, pixels, grid_w, grid_h);
        }
        if !state.buckets.contains_key(&(grid_w, grid_h)) {
            let cells = (grid_h / pk) * (grid_w / pk);
            let aids = build_capture_aids(
                tower,
                grid_w,
                grid_h,
                &Device::Cuda(self.device.clone()),
            )?;
            let incoming = bucket_resident_bytes(n, pp, cells, text_hidden, &aids);
            let resident: Vec<((usize, usize), usize)> = state
                .buckets
                .iter()
                .map(|(g, b)| (*g, b.resident_bytes))
                .collect();
            match plan_bucket_admission(&resident, incoming, self.bucket_byte_budget) {
                BucketAdmission::RefusedBecauseOneBucketAloneExceedsTheBudget => {
                    state.oversize_grids_stay_eager.insert((grid_w, grid_h));
                    eprintln!(
                        "[vision_graph] grid {grid_w}x{grid_h} needs {incoming} resident bytes, \
                         over the {} byte bucket budget; encoding eager",
                        self.bucket_byte_budget
                    );
                    drop(state);
                    return self.forward_eager(tower, pixels, grid_w, grid_h);
                }
                BucketAdmission::Admit {
                    evict_largest_first,
                } => {
                    if !evict_largest_first.is_empty() {
                        self.stream
                            .synchronize()
                            .map_err(|e| anyhow::anyhow!("sync before bucket eviction: {e:?}"))?;
                        for g in &evict_largest_first {
                            if let Some(b) = state.buckets.remove(g) {
                                state.runner.invalidate_token(b.key);
                                eprintln!(
                                    "[vision_graph] evicted bucket {}x{} ({} bytes) to admit \
                                     {grid_w}x{grid_h} ({incoming} bytes) under the {} byte \
                                     budget; that grid re-captures on its next miss",
                                    g.0, g.1, b.resident_bytes, self.bucket_byte_budget
                                );
                            }
                        }
                        self.stream
                            .synchronize()
                            .map_err(|e| anyhow::anyhow!("sync after bucket eviction: {e:?}"))?;
                        nv_kernels::graph::trim_device_graph_mempool_because_graph_destroy_keeps_the_reserved_pages(
                            self.stream.context().ordinal(),
                        );
                    }
                }
            }
            let input = self
                .stream
                .alloc_zeros::<f32>(n * pp)
                .map_err(|e| anyhow::anyhow!("alloc vision input buf: {e:?}"))?;
            let output = self
                .stream
                .alloc_zeros::<f32>(cells * text_hidden)
                .map_err(|e| anyhow::anyhow!("alloc vision output buf: {e:?}"))?;
            let host_pixels = unsafe {
                self.stream
                    .context()
                    .alloc_pinned::<f32>(n * pp)
                    .map_err(|e| anyhow::anyhow!("alloc pinned vision staging: {e:?}"))?
            };
            self.stream
                .synchronize()
                .map_err(|e| anyhow::anyhow!("sync after vision bucket alloc: {e:?}"))?;
            let key = state.next_key;
            state.next_key += 1;
            state.buckets.insert(
                (grid_w, grid_h),
                Bucket {
                    key,
                    host_pixels,
                    input,
                    output,
                    aids,
                    cells,
                    resident_bytes: incoming,
                },
            );
        }

        let dev = self.device.clone();
        let capture_stream = self.stream.clone();
        let heads = state.head_weights.clone().unwrap();
        let GraphState {
            runner, buckets, ..
        } = &mut *state;
        let bucket = buckets.get_mut(&(grid_w, grid_h)).unwrap();
        bucket
            .host_pixels
            .as_mut_slice()
            .map_err(|e| anyhow::anyhow!("pinned staging view: {e:?}"))?
            .copy_from_slice(pixels);
        let key = bucket.key;
        let cells = bucket.cells;
        let need_warm = !runner.has_cached_token(key);

        let step = |s: &Arc<CudaStream>, bucket: &mut Bucket| -> Result<()> {
            let Bucket {
                host_pixels,
                input,
                output,
                aids,
                ..
            } = bucket;
            let pinned_view_skips_the_event_wait_that_invalidates_capture = host_pixels
                .as_slice()
                .map_err(|e| anyhow::anyhow!("pinned staging view: {e:?}"))?;
            s.memcpy_htod(
                pinned_view_skips_the_event_wait_that_invalidates_capture,
                input,
            )
            .map_err(|e| anyhow::anyhow!("htod vision pixels: {e:?}"))?;
            let pv = wrap_slice_f32(input, &dev, (n, pp))?;
            let emb = forward_capture_body(tower, &pv, aids, &heads)?;
            let emb = emb.to_dtype(DType::F32)?.contiguous()?;
            copy_all_f32(&emb, cells * text_hidden, output, &dev)
        };

        let run_result = (|| -> Result<()> {
            if need_warm {
                runner.probe_capture()?;
                nv_layers::cuda_stream::with_stream(capture_stream.clone(), || {
                    step(&capture_stream, bucket)
                })
                .context("warm pass before vision capture")?;
                capture_stream
                    .synchronize()
                    .map_err(|e| anyhow::anyhow!("vision warm sync: {e:?}"))?;
            }
            runner.run(key, |s| {
                nv_layers::cuda_stream::with_stream(s.clone(), || step(s, bucket))
            })
        })();

        match run_result {
            Ok(()) => {
                if need_warm {
                    state.captures += 1;
                } else {
                    state.replays += 1;
                }
                let bucket = state.buckets.get(&(grid_w, grid_h)).unwrap();
                let clone = bucket
                    .output
                    .try_clone()
                    .map_err(|e| anyhow::anyhow!("clone vision output buf: {e:?}"))?;
                let storage = candle_core::CudaStorage::wrap_cuda_slice(clone, self.device.clone());
                Ok(Tensor::from_storage(
                    candle_core::Storage::Cuda(storage),
                    (cells, text_hidden),
                    candle_core::op::BackpropOp::none(),
                    false,
                ))
            }
            Err(e) => {
                state.capture_failed = true;
                state.runner.invalidate();
                let _ = self.stream.synchronize();
                nv_kernels::graph::trim_device_graph_mempool_because_graph_destroy_keeps_the_reserved_pages(
                    self.stream.context().ordinal(),
                );
                eprintln!(
                    "[vision_graph] capture unavailable; vision encode continues eager: {e:#}"
                );
                drop(state);
                self.forward_eager(tower, pixels, grid_w, grid_h)
            }
        }
    }
}

impl Drop for Gemma4VisionGraph {
    fn drop(&mut self) {
        let td = GraphTeardown::for_capture(&self.capture);
        let state = self.state.get_mut().unwrap_or_else(|p| p.into_inner());
        let runner = &mut state.runner;
        td.run(|| runner.invalidate());
    }
}

fn wrap_slice_f32<S: Into<candle_core::Shape>>(
    buf: &CudaSlice<f32>,
    dev: &candle_core::CudaDevice,
    shape: S,
) -> Result<Tensor> {
    let clone = buf
        .try_clone()
        .map_err(|e| anyhow::anyhow!("clone f32 buf: {e:?}"))?;
    let storage = candle_core::CudaStorage::wrap_cuda_slice(clone, dev.clone());
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        shape,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evictions(resident: &[((usize, usize), usize)], incoming: usize, budget: usize) -> Vec<(usize, usize)> {
        match plan_bucket_admission(resident, incoming, budget) {
            BucketAdmission::Admit {
                evict_largest_first,
            } => evict_largest_first,
            BucketAdmission::RefusedBecauseOneBucketAloneExceedsTheBudget => {
                panic!("expected admission, got oversize refusal")
            }
        }
    }

    #[test]
    fn a_bucket_larger_than_the_whole_budget_is_refused_because_no_eviction_can_fit_it() {
        assert!(matches!(
            plan_bucket_admission(&[], 101, 100),
            BucketAdmission::RefusedBecauseOneBucketAloneExceedsTheBudget
        ));
        assert!(matches!(
            plan_bucket_admission(&[((8, 8), 40)], 101, 100),
            BucketAdmission::RefusedBecauseOneBucketAloneExceedsTheBudget
        ));
    }

    #[test]
    fn admission_within_budget_evicts_nothing() {
        assert!(evictions(&[], 100, 100).is_empty());
        assert!(evictions(&[((8, 8), 30), ((12, 8), 40)], 30, 100).is_empty());
    }

    #[test]
    fn admission_over_budget_evicts_the_largest_buckets_first_and_only_as_many_as_needed() {
        let resident = [((8, 8), 20), ((16, 16), 50), ((12, 8), 30)];
        assert_eq!(
            evictions(&resident, 40, 100),
            vec![(16, 16)],
            "evicting the single largest bucket (50) brings 100 resident to 50 and admits 40; \
             evicting anything smaller first would need two evictions"
        );
        assert_eq!(
            evictions(&resident, 80, 100),
            vec![(16, 16), (12, 8)],
            "admitting 80 into a full 100-byte budget must shed largest-first until it fits"
        );
        assert_eq!(
            evictions(&resident, 100, 100),
            vec![(16, 16), (12, 8), (8, 8)],
            "a budget-sized bucket may evict everything, never refuse: the vision tower \
             re-captures evicted grids on their next miss, so eviction is correct, just slow"
        );
    }

    #[test]
    fn eviction_order_is_deterministic_for_equal_sizes() {
        let resident = [((12, 8), 40), ((8, 12), 40), ((8, 8), 40)];
        assert_eq!(
            evictions(&resident, 90, 120),
            vec![(8, 8), (8, 12), (12, 8)],
            "equal-sized buckets tie-break on the grid key so repeated runs evict the same set"
        );
    }
}

fn copy_all_f32(
    src: &Tensor,
    expected: usize,
    dst: &mut CudaSlice<f32>,
    dev: &candle_core::CudaDevice,
) -> Result<()> {
    let contig = src.contiguous()?;
    let (storage, _layout) = contig.storage_and_layout();
    let cuda = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => anyhow::bail!("vision embedding must be on CUDA"),
    };
    let slice = cuda.as_cuda_slice::<f32>()?;
    let len = slice.len();
    anyhow::ensure!(
        len >= expected,
        "vision embedding len {len} < expected {expected}"
    );
    let src = slice.slice(len - expected..);
    let stream = nv_layers::cuda_stream::current_stream(dev);
    stream
        .memcpy_dtod(&src, dst)
        .map_err(|e| anyhow::anyhow!("dtod vision embedding: {e:?}"))?;
    Ok(())
}
