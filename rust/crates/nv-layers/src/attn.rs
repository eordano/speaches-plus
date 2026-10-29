use anyhow::Result;
use candle_core::{DType, Tensor};

pub struct AttnConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub softmax_scale: f32,
    pub causal: bool,
}

pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, cfg: &AttnConfig) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if q.device().is_cuda()
        && matches!(q.dtype(), DType::BF16 | DType::F16)
    {
        return flash_attn(&q.contiguous()?, &k.contiguous()?, &v.contiguous()?, cfg);
    }
    sdpa(q, k, v, cfg)
}

pub const AN_ATTENTION_SINK_IS_ONE_VIRTUAL_LOGIT_PER_HEAD_IN_MAX_AND_DENOMINATOR_ONLY: &str =
    "gpt-oss learns one scalar `sinks[h]` per attention head that enters the softmax as an extra \
     logit which no value row answers: it raises the running max and it adds exp(sink - m) to the \
     denominator, so every real key's probability is scaled down by a learned amount and the head \
     can attend to nothing. nv_models::gpt_oss_wgpu's decode and prefill WGSL fold it as \
     `m = max(red[0], sink); z = red[0] + exp(sink - m)`, and its host reference seeds \
     `let mut m = a.sinks[h]` before the score loop. sdpa_with_sinks reproduces that exactly by \
     appending the per-head sink as one more score column and appending one all-zero value row, \
     which is algebraically identical to the fold and needs no new softmax kernel; \
     candle_flash_attn cannot express it at all, so a sink model must not take the flash_attn \
     branch of `attention`.";

pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, cfg: &AttnConfig) -> Result<Tensor> {
    sdpa_masked(q, k, v, cfg, None, 0)
}

pub fn sdpa_with_sinks(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cfg: &AttnConfig,
    sinks: &Tensor,
    sliding_window: usize,
) -> Result<Tensor> {
    let _ = AN_ATTENTION_SINK_IS_ONE_VIRTUAL_LOGIT_PER_HEAD_IN_MAX_AND_DENOMINATOR_ONLY;
    sdpa_masked(q, k, v, cfg, Some(sinks), sliding_window)
}

pub fn sdpa_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cfg: &AttnConfig,
    sliding_window: usize,
) -> Result<Tensor> {
    sdpa_masked(q, k, v, cfg, None, sliding_window)
}

pub const GQA_GROUPED_BATCHES_THE_KV_GROUPS_INSTEAD_OF_EXPANDING_KV_TO_EVERY_HEAD: &str =
    "the shipped GQA path materializes k and v expanded to all H heads (two rank-5 expand \
     copies forward, and backward pays a rank-5 strided sum-reduce folding dK/dV back to H_kv \
     -- the candle strided-reduce class), then runs B*H batched GEMMs at m=sq. \
     NV_ATTN_GQA_GROUPED=1 instead views q as (B*H_kv, factor*sq, d) -- head h = g*factor + f \
     is exactly the layout the expand construction distributes, so the group rows are already \
     contiguous -- and runs B*H_kv batched GEMMs at m=factor*sq against the unexpanded \
     (B*H_kv, d, sk) k and (B*H_kv, sk, d) v: no expand copies, no expand-backward reduce, dK \
     and dV fall straight out of matmul backward. The causal mask tiles to (factor*sq, sk) with \
     row r masking query position r % sq. Every score and probability is the same dot product \
     over the same operands, but the GEMM tiling moves at m=factor*sq, so the arm is \
     NON-BITWISE BY CONSTRUCTION: it gates on the eval-CE census (28-row base CE deltas, \
     step-19 trajectory band, adapter delta-amax), never on bit-identity, and must not be \
     composed silently with bitwise-claimed arms";

fn attn_gqa_grouped_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NV_ATTN_GQA_GROUPED").as_deref() == Ok("1"))
}

fn sdpa_masked_grouped(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cfg: &AttnConfig,
    sliding_window: usize,
) -> Result<Tensor> {
    let _ = GQA_GROUPED_BATCHES_THE_KV_GROUPS_INSTEAD_OF_EXPANDING_KV_TO_EVERY_HEAD;
    let qd = q.dims();
    let kd = k.dims();
    let (b, sq, h, d) = (qd[0], qd[1], qd[2], qd[3]);
    let sk = kd[1];
    let h_kv = kd[2];
    let factor = h / h_kv;
    let orig_dtype = q.dtype();

    let q = q.to_dtype(DType::F32).map_err(|e| anyhow::anyhow!(e))?;
    let k = k.to_dtype(DType::F32).map_err(|e| anyhow::anyhow!(e))?;
    let v = v.to_dtype(DType::F32).map_err(|e| anyhow::anyhow!(e))?;

    let q_g = q
        .permute((0, 2, 1, 3))
        .map_err(|e| anyhow::anyhow!(e))?
        .contiguous()
        .map_err(|e| anyhow::anyhow!(e))?
        .reshape((b * h_kv, factor * sq, d))
        .map_err(|e| anyhow::anyhow!(e))?;
    let k_t = k
        .permute((0, 2, 1, 3))
        .map_err(|e| anyhow::anyhow!(e))?
        .contiguous()
        .map_err(|e| anyhow::anyhow!(e))?
        .reshape((b * h_kv, sk, d))
        .map_err(|e| anyhow::anyhow!(e))?;
    let v_flat = v
        .permute((0, 2, 1, 3))
        .map_err(|e| anyhow::anyhow!(e))?
        .contiguous()
        .map_err(|e| anyhow::anyhow!(e))?
        .reshape((b * h_kv, sk, d))
        .map_err(|e| anyhow::anyhow!(e))?;
    let k_perm = k_t
        .permute((0, 2, 1))
        .map_err(|e| anyhow::anyhow!(e))?
        .contiguous()
        .map_err(|e| anyhow::anyhow!(e))?;

    let scale = if attn_resident_enabled() {
        softmax_scale_resident(cfg.softmax_scale, q_g.device())?
    } else {
        Tensor::new(cfg.softmax_scale, q_g.device()).map_err(|e| anyhow::anyhow!(e))?
    };
    let mut scores = q_g
        .matmul(&k_perm)
        .map_err(|e| anyhow::anyhow!(e))?
        .broadcast_mul(&scale)
        .map_err(|e| anyhow::anyhow!(e))?;
    if cfg.causal {
        let mask = if attn_resident_enabled() {
            causal_mask_grouped_resident(sq, sk, sliding_window, factor, q_g.device())?
        } else {
            build_causal_mask_grouped(sq, sk, sliding_window, factor, q_g.device())?
        };
        scores = scores
            .broadcast_add(&mask)
            .map_err(|e| anyhow::anyhow!(e))?;
    }
    let probs = softmax_last_dim_that_keeps_q_and_k_on_the_graph(&scores)?;
    let out = probs
        .contiguous()
        .map_err(|e| anyhow::anyhow!(e))?
        .matmul(&v_flat)
        .map_err(|e| anyhow::anyhow!(e))?;
    let out = out
        .reshape((b, h_kv, factor, sq, d))
        .map_err(|e| anyhow::anyhow!(e))?
        .permute((0, 3, 1, 2, 4))
        .map_err(|e| anyhow::anyhow!(e))?
        .contiguous()
        .map_err(|e| anyhow::anyhow!(e))?
        .reshape((b, sq, h, d))
        .map_err(|e| anyhow::anyhow!(e))?;
    out.to_dtype(orig_dtype).map_err(|e| anyhow::anyhow!(e))
}

fn sdpa_masked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cfg: &AttnConfig,
    sinks: Option<&Tensor>,
    sliding_window: usize,
) -> Result<Tensor> {
    let qd = q.dims();
    let kd = k.dims();
    if qd.len() != 4 || kd.len() != 4 {
        anyhow::bail!(
            "sdpa: expected rank-4 q/k tensors (B, S, H, D), got q={:?} k={:?}",
            qd,
            kd
        );
    }
    let (b, sq, h, d) = (qd[0], qd[1], qd[2], qd[3]);
    let sk = kd[1];
    let h_kv = kd[2];
    if h % h_kv != 0 {
        anyhow::bail!("sdpa: H ({h}) not divisible by H_kv ({h_kv})");
    }
    if sliding_window > 0 && !cfg.causal {
        anyhow::bail!("sdpa: a sliding window is only defined on the causal path");
    }
    if attn_gqa_grouped_enabled() && h_kv != h && sinks.is_none() {
        return sdpa_masked_grouped(q, k, v, cfg, sliding_window);
    }
    let orig_dtype = q.dtype();

    let q = q.to_dtype(DType::F32).map_err(|e| anyhow::anyhow!(e))?;
    let k = k.to_dtype(DType::F32).map_err(|e| anyhow::anyhow!(e))?;
    let v = v.to_dtype(DType::F32).map_err(|e| anyhow::anyhow!(e))?;

    let (k, v) = if h_kv == h {
        (k, v)
    } else {
        let factor = h / h_kv;
        let k = k
            .unsqueeze(3)
            .map_err(|e| anyhow::anyhow!(e))?
            .expand((b, sk, h_kv, factor, d))
            .map_err(|e| anyhow::anyhow!(e))?
            .reshape((b, sk, h, d))
            .map_err(|e| anyhow::anyhow!(e))?;
        let v = v
            .unsqueeze(3)
            .map_err(|e| anyhow::anyhow!(e))?
            .expand((b, sk, h_kv, factor, d))
            .map_err(|e| anyhow::anyhow!(e))?
            .reshape((b, sk, h, d))
            .map_err(|e| anyhow::anyhow!(e))?;
        (k, v)
    };

    let q_t = q
        .permute((0, 2, 1, 3))
        .map_err(|e| anyhow::anyhow!(e))?
        .contiguous()
        .map_err(|e| anyhow::anyhow!(e))?;
    let k_t = k
        .permute((0, 2, 1, 3))
        .map_err(|e| anyhow::anyhow!(e))?
        .contiguous()
        .map_err(|e| anyhow::anyhow!(e))?;
    let v_t = v
        .permute((0, 2, 1, 3))
        .map_err(|e| anyhow::anyhow!(e))?
        .contiguous()
        .map_err(|e| anyhow::anyhow!(e))?;

    let q_flat = q_t
        .reshape((b * h, sq, d))
        .map_err(|e| anyhow::anyhow!(e))?;
    let k_flat = k_t
        .reshape((b * h, sk, d))
        .map_err(|e| anyhow::anyhow!(e))?;
    let v_flat = v_t
        .reshape((b * h, sk, d))
        .map_err(|e| anyhow::anyhow!(e))?;

    let k_perm = k_flat
        .permute((0, 2, 1))
        .map_err(|e| anyhow::anyhow!(e))?
        .contiguous()
        .map_err(|e| anyhow::anyhow!(e))?;
    let scale = if attn_resident_enabled() {
        let _ = ATTN_RESIDENT_CACHES_THE_MASK_AND_SCALE_UPLOADS_THAT_SDPA_REBUILT_EVERY_CALL;
        softmax_scale_resident(cfg.softmax_scale, q_flat.device())?
    } else {
        Tensor::new(cfg.softmax_scale, q_flat.device()).map_err(|e| anyhow::anyhow!(e))?
    };
    let mut scores = q_flat
        .matmul(&k_perm)
        .map_err(|e| anyhow::anyhow!(e))?
        .broadcast_mul(&scale)
        .map_err(|e| anyhow::anyhow!(e))?;

    if cfg.causal {
        let mask = if attn_resident_enabled() {
            causal_mask_resident(sq, sk, sliding_window, q_flat.device())?
        } else {
            build_causal_mask_windowed(sq, sk, sliding_window, q_flat.device())?
        };
        scores = scores
            .broadcast_add(&mask)
            .map_err(|e| anyhow::anyhow!(e))?;
    }

    let (scores, v_flat) = match sinks {
        None => (scores, v_flat),
        Some(s) => {
            let sd = s.dims();
            if sd.len() != 1 || sd[0] != h {
                anyhow::bail!("sdpa_with_sinks: sinks must be [{h}], got {sd:?}");
            }
            let col = s
                .to_dtype(DType::F32)
                .map_err(|e| anyhow::anyhow!(e))?
                .reshape((1, h, 1, 1))
                .map_err(|e| anyhow::anyhow!(e))?
                .broadcast_as((b, h, sq, 1))
                .map_err(|e| anyhow::anyhow!(e))?
                .reshape((b * h, sq, 1))
                .map_err(|e| anyhow::anyhow!(e))?
                .contiguous()
                .map_err(|e| anyhow::anyhow!(e))?;
            let zero_row = Tensor::zeros((b * h, 1, d), DType::F32, v_flat.device())
                .map_err(|e| anyhow::anyhow!(e))?;
            (
                Tensor::cat(&[&scores, &col], 2).map_err(|e| anyhow::anyhow!(e))?,
                Tensor::cat(&[&v_flat, &zero_row], 1).map_err(|e| anyhow::anyhow!(e))?,
            )
        }
    };

    let probs = softmax_last_dim_that_keeps_q_and_k_on_the_graph(&scores)?;
    let out = probs
        .contiguous()
        .map_err(|e| anyhow::anyhow!(e))?
        .matmul(&v_flat.contiguous().map_err(|e| anyhow::anyhow!(e))?)
        .map_err(|e| anyhow::anyhow!(e))?;
    let out = out
        .reshape((b, h, sq, d))
        .map_err(|e| anyhow::anyhow!(e))?
        .permute((0, 2, 1, 3))
        .map_err(|e| anyhow::anyhow!(e))?
        .contiguous()
        .map_err(|e| anyhow::anyhow!(e))?;
    out.to_dtype(orig_dtype).map_err(|e| anyhow::anyhow!(e))
}

pub const CANDLE_INSTALLS_SOFTMAX_LAST_DIM_WITH_APPLY_OP1_NO_BWD_SO_THE_PROBABILITIES_ARE_A_GRAPH_LEAF_AND_ONLY_V_WOULD_TRAIN: &str =
    "candle_nn::ops::softmax_last_dim is `xs.apply_op1_no_bwd(&SoftmaxLastDim)`: the fused kernel \
     is fast and exact but its output carries BackpropOp::none, so scores -- and with them q and \
     k -- are unreachable from the loss and receive no gradient entry at all, while v still gets \
     one through the `probs @ v` matmul that comes after. A trainer that scores attention through \
     the fused kernel therefore optimises the value projection alone, does not error, and reports \
     a falling loss the whole time; a LoRA fitted on q_proj/k_proj under it keeps its zero-init B \
     forever and merges as an exact no-op. The composed candle softmax (max_keepdim, \
     broadcast_sub, exp, sum_keepdim, broadcast_div) is the same function built from tracked \
     primitives, and its backward is the softmax Jacobian-vector product dx = p * (dy - sum(p * \
     dy)) -- the max subtraction contributes exactly zero because sum_j p_j (dy_j - sum(p dy)) is \
     identically zero. Serving never takes it: this_call_is_building_an_autograd_graph is the same \
     predicate candle's own BackpropOp::new1 applies, so on an untracked score tensor -- every \
     tensor any decoder ever passes -- the fused kernel is still the only thing that runs and the \
     served probabilities are bit-identical.";

pub fn softmax_last_dim_that_keeps_q_and_k_on_the_graph(scores: &Tensor) -> Result<Tensor> {
    if crate::linear::this_call_is_building_an_autograd_graph(scores) {
        let _ = CANDLE_INSTALLS_SOFTMAX_LAST_DIM_WITH_APPLY_OP1_NO_BWD_SO_THE_PROBABILITIES_ARE_A_GRAPH_LEAF_AND_ONLY_V_WOULD_TRAIN;
        return candle_nn::ops::softmax(scores, candle_core::D::Minus1)
            .map_err(|e| anyhow::anyhow!(e));
    }
    candle_nn::ops::softmax_last_dim(scores).map_err(|e| anyhow::anyhow!(e))
}

pub const ATTN_RESIDENT_CACHES_THE_MASK_AND_SCALE_UPLOADS_THAT_SDPA_REBUILT_EVERY_CALL: &str =
    "sdpa_masked rebuilt the causal mask on the host (an sq*sk f32 fill plus upload) and \
     re-uploaded the softmax scale scalar on every call; both are pure functions of \
     (sq, sk, window, device) and (scale bits, device), so NV_ATTN_RESIDENT=1 serves them from \
     a process cache -- identical bytes reach identical ops, bitwise by construction. The qlora \
     window forward pays the rebuild once per layer per bucket (48 per step at up to 450x450)";

fn attn_resident_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NV_ATTN_RESIDENT").as_deref() == Ok("1"))
}

type MaskKey = (usize, usize, usize, candle_core::DeviceLocation);

fn causal_mask_resident(
    sq: usize,
    sk: usize,
    sliding_window: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    static CACHE: std::sync::Mutex<Vec<(MaskKey, Tensor)>> = std::sync::Mutex::new(Vec::new());
    let key = (sq, sk, sliding_window, device.location());
    let mut cache = CACHE
        .lock()
        .map_err(|e| anyhow::anyhow!("mask cache poisoned: {e}"))?;
    if let Some((_, t)) = cache.iter().find(|(k, _)| *k == key) {
        return Ok(t.clone());
    }
    let t = build_causal_mask_windowed(sq, sk, sliding_window, device)?;
    cache.push((key, t.clone()));
    Ok(t)
}

fn softmax_scale_resident(scale: f32, device: &candle_core::Device) -> Result<Tensor> {
    static CACHE: std::sync::Mutex<Vec<((u32, candle_core::DeviceLocation), Tensor)>> =
        std::sync::Mutex::new(Vec::new());
    let key = (scale.to_bits(), device.location());
    let mut cache = CACHE
        .lock()
        .map_err(|e| anyhow::anyhow!("scale cache poisoned: {e}"))?;
    if let Some((_, t)) = cache.iter().find(|(k, _)| *k == key) {
        return Ok(t.clone());
    }
    let t = Tensor::new(scale, device).map_err(|e| anyhow::anyhow!(e))?;
    cache.push((key, t.clone()));
    Ok(t)
}

fn causal_mask_grouped_resident(
    sq: usize,
    sk: usize,
    sliding_window: usize,
    factor: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    static CACHE: std::sync::Mutex<
        Vec<(
            (usize, usize, usize, usize, candle_core::DeviceLocation),
            Tensor,
        )>,
    > = std::sync::Mutex::new(Vec::new());
    let key = (sq, sk, sliding_window, factor, device.location());
    let mut cache = CACHE
        .lock()
        .map_err(|e| anyhow::anyhow!("grouped mask cache poisoned: {e}"))?;
    if let Some((_, t)) = cache.iter().find(|(k, _)| *k == key) {
        return Ok(t.clone());
    }
    let t = build_causal_mask_grouped(sq, sk, sliding_window, factor, device)?;
    cache.push((key, t.clone()));
    Ok(t)
}

fn build_causal_mask_grouped(
    sq: usize,
    sk: usize,
    sliding_window: usize,
    factor: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let off = sk.saturating_sub(sq);
    let mut mask = vec![0f32; factor * sq * sk];
    for r in 0..factor * sq {
        let i = r % sq;
        let last = i + off;
        let first = if sliding_window > 0 {
            (last + 1).saturating_sub(sliding_window)
        } else {
            0
        };
        for (j, cell) in mask[r * sk..(r + 1) * sk].iter_mut().enumerate() {
            if j > last || j < first {
                *cell = f32::NEG_INFINITY;
            }
        }
    }
    let t = Tensor::from_vec(mask, (1, factor * sq, sk), device).map_err(|e| anyhow::anyhow!(e))?;
    Ok(t)
}

fn build_causal_mask_windowed(
    sq: usize,
    sk: usize,
    sliding_window: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let off = sk.saturating_sub(sq);
    let mut mask = vec![0f32; sq * sk];
    for i in 0..sq {
        let last = i + off;
        let first = if sliding_window > 0 {
            (last + 1).saturating_sub(sliding_window)
        } else {
            0
        };
        for (j, cell) in mask[i * sk..(i + 1) * sk].iter_mut().enumerate() {
            if j > last || j < first {
                *cell = f32::NEG_INFINITY;
            }
        }
    }
    let t = Tensor::from_vec(mask, (1, sq, sk), device).map_err(|e| anyhow::anyhow!(e))?;
    Ok(t)
}

#[cfg(feature = "cuda")]
pub const CANDLE_FLASH_ATTN_LAUNCHES_ON_THE_LEGACY_NULL_STREAM_SO_EVERY_CALL_IS_EVENT_FENCED: &str =
    "candle-flash-attn's flash_api.cu hardcodes cudaStream_t stream = 0, so its kernel \
     runs on the legacy NULL stream while a Device::new_cuda_with_stream device runs every \
     candle op, every nv-* kernel, every cuMemAllocAsync and every CudaSlice drop's \
     cuMemFreeAsync on a CU_STREAM_NON_BLOCKING stream the NULL stream never synchronizes with. \
     Unfenced, the flash kernel can read q/k/v before the device stream has produced them, and \
     it keeps writing dst and its softmax_lse scratch after the device stream has already freed \
     the step's temporaries, so the async mempool recycles pages a queued kernel still touches: \
     CUDA_ERROR_ILLEGAL_ADDRESS surfacing steps after the causal launch, memcheck-clean, and \
     masked by CUDA_LAUNCH_BLOCKING or compute-sanitizer serialization. The pre-fence makes the \
     NULL stream wait on the device stream's queued work; the post-fence makes the device \
     stream (and any nv thread-local override stream) wait on the flash kernel, which also \
     orders every later cuMemFreeAsync and mempool reuse after it. Cost is event records and \
     stream waits only; the host never blocks.";

#[cfg(feature = "cuda")]
thread_local! {
    static FENCE_EVENT_REUSED_BECAUSE_EACH_STREAM_WAIT_SNAPSHOTS_THE_RECORD_IT_FOLLOWS:
        std::cell::RefCell<Option<(usize, cudarc::driver::CudaEvent)>> =
            const { std::cell::RefCell::new(None) };
}

#[cfg(feature = "cuda")]
fn fence_the_legacy_null_stream_around<T>(
    device: &candle_core::Device,
    launch_on_the_null_stream: impl FnOnce() -> candle_core::Result<T>,
) -> Result<T> {
    let candle_core::Device::Cuda(dev) = device else {
        return launch_on_the_null_stream().map_err(|e| anyhow::anyhow!(e));
    };
    let device_stream = dev.cuda_stream();
    if device_stream.cu_stream().is_null() {
        return launch_on_the_null_stream().map_err(|e| anyhow::anyhow!(e));
    }
    let _ = CANDLE_FLASH_ATTN_LAUNCHES_ON_THE_LEGACY_NULL_STREAM_SO_EVERY_CALL_IS_EVENT_FENCED;
    let ctx = device_stream.context().clone();
    let ctx_key = std::sync::Arc::as_ptr(&ctx) as usize;
    let event = FENCE_EVENT_REUSED_BECAUSE_EACH_STREAM_WAIT_SNAPSHOTS_THE_RECORD_IT_FOLLOWS
        .with(|cell| match cell.borrow_mut().take() {
            Some((key, event)) if key == ctx_key => Ok(event),
            _ => ctx.new_event(None),
        })
        .map_err(|e| anyhow::anyhow!(e))?;
    let null_stream = ctx.default_stream();
    let override_stream = crate::cuda_stream::current_stream(dev);
    let override_needs_its_own_fence = override_stream.cu_stream() != device_stream.cu_stream()
        && !override_stream.cu_stream().is_null();
    event
        .record(&device_stream)
        .map_err(|e| anyhow::anyhow!(e))?;
    null_stream.wait(&event).map_err(|e| anyhow::anyhow!(e))?;
    if override_needs_its_own_fence {
        event
            .record(&override_stream)
            .map_err(|e| anyhow::anyhow!(e))?;
        null_stream.wait(&event).map_err(|e| anyhow::anyhow!(e))?;
    }
    let out = launch_on_the_null_stream().map_err(|e| anyhow::anyhow!(e))?;
    event.record(&null_stream).map_err(|e| anyhow::anyhow!(e))?;
    device_stream.wait(&event).map_err(|e| anyhow::anyhow!(e))?;
    if override_needs_its_own_fence {
        override_stream
            .wait(&event)
            .map_err(|e| anyhow::anyhow!(e))?;
    }
    FENCE_EVENT_REUSED_BECAUSE_EACH_STREAM_WAIT_SNAPSHOTS_THE_RECORD_IT_FOLLOWS.with(|cell| {
        *cell.borrow_mut() = Some((ctx_key, event));
    });
    Ok(out)
}

#[cfg(feature = "cuda")]
pub fn flash_attn(q: &Tensor, k: &Tensor, v: &Tensor, cfg: &AttnConfig) -> Result<Tensor> {
    fence_the_legacy_null_stream_around(q.device(), || {
        candle_flash_attn::flash_attn(q, k, v, cfg.softmax_scale, cfg.causal)
    })
}

#[cfg(not(feature = "cuda"))]
pub fn flash_attn(_q: &Tensor, _k: &Tensor, _v: &Tensor, _cfg: &AttnConfig) -> Result<Tensor> {
    anyhow::bail!("flash_attn requires --features cuda")
}

#[cfg(feature = "cuda")]
pub fn flash_attn_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cfg: &AttnConfig,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor> {
    fence_the_legacy_null_stream_around(q.device(), || {
        candle_flash_attn::flash_attn_windowed(
            q,
            k,
            v,
            cfg.softmax_scale,
            window_size_left,
            window_size_right,
        )
    })
}

#[cfg(not(feature = "cuda"))]
pub fn flash_attn_windowed(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _cfg: &AttnConfig,
    _window_size_left: Option<usize>,
    _window_size_right: Option<usize>,
) -> Result<Tensor> {
    anyhow::bail!("flash_attn_windowed requires --features cuda")
}

#[cfg(feature = "cuda")]
pub fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    cfg: &AttnConfig,
) -> Result<Tensor> {
    fence_the_legacy_null_stream_around(q.device(), || {
        candle_flash_attn::flash_attn_varlen(
            q,
            k,
            v,
            seqlens_q,
            seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            cfg.softmax_scale,
            cfg.causal,
        )
    })
}

#[cfg(not(feature = "cuda"))]
pub fn flash_attn_varlen(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _seqlens_q: &Tensor,
    _seqlens_k: &Tensor,
    _max_seqlen_q: usize,
    _max_seqlen_k: usize,
    _cfg: &AttnConfig,
) -> Result<Tensor> {
    anyhow::bail!("flash_attn_varlen requires --features cuda")
}

#[cfg(test)]
mod paper_validation {

    use super::*;
    use candle_core::Device;

    #[test]
    fn sdpa_causal_mask_is_bottom_right_aligned_like_flash_attn() {
        let device = Device::Cpu;
        let (sq, sk) = (3usize, 5usize);
        let d = sk;
        let q = Tensor::zeros((1, sq, 1, d), candle_core::DType::F32, &device).unwrap();
        let k = Tensor::zeros((1, sk, 1, d), candle_core::DType::F32, &device).unwrap();
        let mut v_host = vec![0f32; sk * d];
        for j in 0..sk {
            v_host[j * d + j] = 1.0;
        }
        let v = Tensor::from_vec(v_host, (1, sk, 1, d), &device).unwrap();
        let cfg = AttnConfig {
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: d,
            softmax_scale: 1.0,
            causal: true,
        };
        let out = sdpa(&q, &k, &v, &cfg).unwrap();
        let o = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let off = sk - sq;
        for i in 0..sq {
            let allowed: Vec<usize> = (0..sk).filter(|&j| j <= i + off).collect();
            for j in 0..sk {
                let p = o[i * d + j];
                if allowed.contains(&j) {
                    let want = 1.0 / allowed.len() as f32;
                    assert!(
                        (p - want).abs() < 1e-5,
                        "row {i} key {j}: prob {p}, want uniform {want}"
                    );
                } else {
                    assert!(p.abs() < 1e-6, "row {i} attends future key {j} (p={p})");
                }
            }
        }
    }

    #[test]
    fn sdpa_gqa_expansion_matches_repeated_heads_and_the_grouped_path_matches_the_expand_path() {
        let device = Device::Cpu;
        let (sq, sk, d) = (2usize, 4usize, 8usize);
        let mk = |seed: f32, n: usize| -> Vec<f32> {
            (0..n).map(|i| ((i as f32 + seed) * 0.311).sin()).collect()
        };
        let q = Tensor::from_vec(mk(1.0, sq * 4 * d), (1, sq, 4, d), &device).unwrap();
        let k1 = mk(2.0, sk * 2 * d);
        let v1 = mk(3.0, sk * 2 * d);
        let k = Tensor::from_vec(k1.clone(), (1, sk, 2, d), &device).unwrap();
        let v = Tensor::from_vec(v1.clone(), (1, sk, 2, d), &device).unwrap();
        let cfg = AttnConfig {
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: d,
            softmax_scale: 0.35,
            causal: true,
        };
        let out = sdpa(&q, &k, &v, &cfg).unwrap();

        let mut k_rep = vec![0f32; sk * 4 * d];
        let mut v_rep = vec![0f32; sk * 4 * d];
        for t in 0..sk {
            for h in 0..4 {
                let g = h / 2;
                for x in 0..d {
                    k_rep[(t * 4 + h) * d + x] = k1[(t * 2 + g) * d + x];
                    v_rep[(t * 4 + h) * d + x] = v1[(t * 2 + g) * d + x];
                }
            }
        }
        let k_r = Tensor::from_vec(k_rep, (1, sk, 4, d), &device).unwrap();
        let v_r = Tensor::from_vec(v_rep, (1, sk, 4, d), &device).unwrap();
        let cfg_mha = AttnConfig {
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: d,
            softmax_scale: 0.35,
            causal: true,
        };
        let out_mha = sdpa(&q, &k_r, &v_r, &cfg_mha).unwrap();
        let a = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = out_mha.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < 1e-5, "GQA mismatch at {i}: {x} vs {y}");
        }
        for &(sq, sk, h, h_kv, d, window) in &[
            (5usize, 5usize, 12usize, 2usize, 16usize, 0usize),
            (3, 7, 4, 2, 8, 0),
            (6, 6, 6, 3, 4, 4),
        ] {
            let mk = |seed: f32, dims: (usize, usize, usize, usize)| -> Tensor {
                let n = dims.0 * dims.1 * dims.2 * dims.3;
                let v: Vec<f32> = (0..n).map(|i| ((i as f32 + seed) * 0.173).sin()).collect();
                Tensor::from_vec(v, dims, &device).unwrap()
            };
            let q = mk(1.0, (1, sq, h, d));
            let k = mk(2.0, (1, sk, h_kv, d));
            let v = mk(3.0, (1, sk, h_kv, d));
            let cfg = AttnConfig {
                num_heads: h,
                num_kv_heads: h_kv,
                head_dim: d,
                softmax_scale: 0.31,
                causal: true,
            };
            let shipped = sdpa_masked(&q, &k, &v, &cfg, None, window).unwrap();
            let grouped = sdpa_masked_grouped(&q, &k, &v, &cfg, window).unwrap();
            let a = shipped.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = grouped.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (x - y).abs() < 1e-5,
                    "grouped diverges at sq{sq} sk{sk} h{h}/{h_kv} w{window} idx {i}: {x} vs {y}"
                );
            }
        }
    }

    const WARPS: usize = 8;

    #[derive(Clone)]
    struct Partial {
        m: f32,
        l: f32,
        acc: Vec<f32>,
    }

    fn stream_positions(
        q: &[f32],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        positions: impl Iterator<Item = usize>,
        hd: usize,
    ) -> Partial {
        let mut p = Partial {
            m: f32::NEG_INFINITY,
            l: 0.0,
            acc: vec![0.0; hd],
        };
        for pos in positions {
            let score: f32 = (0..hd).map(|d| q[d] * k[pos][d]).sum();
            let m_new = p.m.max(score);
            let corr = (p.m - m_new).exp();
            let w = (score - m_new).exp();
            p.l = p.l * corr + w;
            for d in 0..hd {
                p.acc[d] = p.acc[d] * corr + w * v[pos][d];
            }
            p.m = m_new;
        }
        p
    }

    fn merge(partials: &[Partial], hd: usize) -> Partial {
        let m_glob = partials.iter().fold(f32::NEG_INFINITY, |a, p| a.max(p.m));
        let mut out = Partial {
            m: m_glob,
            l: 0.0,
            acc: vec![0.0; hd],
        };
        for p in partials {
            if p.m > f32::NEG_INFINITY {
                let sc = (p.m - m_glob).exp();
                out.l += p.l * sc;
                for d in 0..hd {
                    out.acc[d] += p.acc[d] * sc;
                }
            }
        }
        out
    }

    fn flash_decode_model(
        q: &[f32],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        total: usize,
        window: usize,
        splits: usize,
        hd: usize,
    ) -> Vec<f32> {
        let start = if window > 0 && total > window {
            total - window
        } else {
            0
        };
        let mut split_partials = Vec::new();
        for split in 0..splits {
            let mut warp_partials = Vec::new();
            for warp in 0..WARPS {
                let stride = splits * WARPS;
                let first = start + split * WARPS + warp;
                let positions = (first..total).step_by(stride.max(1));
                warp_partials.push(stream_positions(q, k, v, positions, hd));
            }
            split_partials.push(merge(&warp_partials, hd));
        }
        let g = merge(&split_partials, hd);
        let inv_l = if g.l > 0.0 { 1.0 / g.l } else { 0.0 };
        g.acc.iter().map(|a| a * inv_l).collect()
    }

    fn naive_softmax_attention(
        q: &[f32],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        total: usize,
        window: usize,
        hd: usize,
    ) -> Vec<f32> {
        let start = if window > 0 && total > window {
            total - window
        } else {
            0
        };
        let scores: Vec<f64> = (start..total)
            .map(|p| (0..hd).map(|d| (q[d] * k[p][d]) as f64).sum())
            .collect();
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let z: f64 = exps.iter().sum();
        let mut out = vec![0f64; hd];
        for (i, p) in (start..total).enumerate() {
            let w = exps[i] / z;
            for d in 0..hd {
                out[d] += w * v[p][d] as f64;
            }
        }
        out.iter().map(|&x| x as f32).collect()
    }

    #[test]
    fn flash_decode_split_merge_equals_naive_softmax_attention() {
        let hd = 8usize;
        let mut seed = 0x9e3779b9u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / (1u64 << 31) as f64) as f32 * 2.0 - 1.0
        };
        for &total in &[1usize, 2, 7, 8, 9, 100, 127, 128, 129, 300] {
            for &window in &[0usize, 1, 5, 64] {
                for &splits in &[1usize, 16] {
                    let q: Vec<f32> = (0..hd).map(|_| next() * 3.0).collect();
                    let k: Vec<Vec<f32>> = (0..total)
                        .map(|_| (0..hd).map(|_| next() * 3.0).collect())
                        .collect();
                    let v: Vec<Vec<f32>> = (0..total)
                        .map(|_| (0..hd).map(|_| next()).collect())
                        .collect();
                    let got = flash_decode_model(&q, &k, &v, total, window, splits, hd);
                    let want = naive_softmax_attention(&q, &k, &v, total, window, hd);
                    for d in 0..hd {
                        assert!(
                            (got[d] - want[d]).abs() <= 3e-5,
                            "total={total} window={window} splits={splits} dim {d}: \
                             flash {} vs naive {}",
                            got[d],
                            want[d]
                        );
                        assert!(got[d].is_finite(), "NaN/inf leaked through the merge");
                    }
                }
            }
        }
    }
}

#[cfg(feature = "cuda")]
struct E4bFullAttnScratch {
    s: Option<cudarc::driver::CudaSlice<f32>>,
    p: Option<cudarc::driver::CudaSlice<u16>>,
    m: Option<cudarc::driver::CudaSlice<f32>>,
    l: Option<cudarc::driver::CudaSlice<f32>>,
    o: Option<cudarc::driver::CudaSlice<f32>>,
    q8: Option<cudarc::driver::CudaSlice<u8>>,
    k8: Option<cudarc::driver::CudaSlice<u8>>,
    scales_and_amax: Option<cudarc::driver::CudaSlice<f32>>,
}

pub const FP8_SCORE_TILES_NEED_16_ALIGNED_ROWS_AND_KEYS_SO_A_RAGGED_TAIL_TILE_RUNS_THE_BF16_GEMM:
    usize = 16;

#[cfg(feature = "cuda")]
fn grow_u8<'a>(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    slot: &'a mut Option<cudarc::driver::CudaSlice<u8>>,
    n: usize,
) -> Result<&'a mut cudarc::driver::CudaSlice<u8>> {
    if slot.as_ref().is_none_or(|s| s.len() < n) {
        *slot = Some(unsafe { stream.alloc::<u8>(n) }.map_err(|e| anyhow::anyhow!(e))?);
    }
    Ok(slot.as_mut().expect("just filled"))
}

#[cfg(feature = "cuda")]
static E4B_FULL_ATTN_SCRATCH_GROWS_ONCE_PER_STREAM_BECAUSE_A_CUDAMALLOC_PER_CHUNK_LAYER_COST_4_PERCENT_AT_2K:
    std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<usize, E4bFullAttnScratch>>> =
    std::sync::OnceLock::new();

#[cfg(feature = "cuda")]
fn grow_f32<'a>(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    slot: &'a mut Option<cudarc::driver::CudaSlice<f32>>,
    len: usize,
) -> Result<&'a mut cudarc::driver::CudaSlice<f32>> {
    if slot.as_ref().map(|b| b.len() < len).unwrap_or(true) {
        *slot = Some(unsafe { stream.alloc::<f32>(len) }.map_err(|e| anyhow::anyhow!(e))?);
    }
    Ok(slot.as_mut().expect("just filled"))
}

#[cfg(feature = "cuda")]
fn grow_u16<'a>(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    slot: &'a mut Option<cudarc::driver::CudaSlice<u16>>,
    len: usize,
) -> Result<&'a mut cudarc::driver::CudaSlice<u16>> {
    if slot.as_ref().map(|b| b.len() < len).unwrap_or(true) {
        *slot = Some(unsafe { stream.alloc::<u16>(len) }.map_err(|e| anyhow::anyhow!(e))?);
    }
    Ok(slot.as_mut().expect("just filled"))
}

pub const E4B_FULL_ATTN_KEY_TILE_4096_KEEPS_THE_F32_SCORE_TILE_INSIDE_L2_AND_MATCHES_THE_SOFTMAX_KERNELS_REGISTER_BUDGET: usize = 4096;

pub const E4B_FULL_ATTN_KEY_TILE_ENV_4096_OR_8192_THE_8192_ARM_IS_THE_SHAPE_ONLY_PROBE_OF_THE_TILE_TERM: &str =
    "NV_E4B_FULL_ATTN_KEY_TILE";

pub fn e4b_full_attn_key_tile() -> usize {
    match std::env::var(E4B_FULL_ATTN_KEY_TILE_ENV_4096_OR_8192_THE_8192_ARM_IS_THE_SHAPE_ONLY_PROBE_OF_THE_TILE_TERM)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(8192) => 8192,
        _ => E4B_FULL_ATTN_KEY_TILE_4096_KEEPS_THE_F32_SCORE_TILE_INSIDE_L2_AND_MATCHES_THE_SOFTMAX_KERNELS_REGISTER_BUDGET,
    }
}

#[cfg(feature = "cuda")]
pub fn e4b_full_attn_gqa_folded_f32_scores_tiled_online_softmax(
    q_groups_rows_hd_bf16: &Tensor,
    k_groups_len_hd_bf16: &Tensor,
    v_groups_len_hd_bf16: &Tensor,
    n: usize,
    group: usize,
    scale: f32,
) -> Result<Tensor> {
    use candle_core::Device;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dev = match q_groups_rows_hd_bf16.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("e4b full attention gemm arm requires cuda"),
    };
    let (nkv, rows, hd) = q_groups_rows_hd_bf16.dims3()?;
    let (nkv_k, len, hd_k) = k_groups_len_hd_bf16.dims3()?;
    anyhow::ensure!(
        nkv == nkv_k && hd == hd_k && rows == group * n && len >= n,
        "e4b full attention gemm arm: q {:?} k {:?} n {n} group {group}",
        q_groups_rows_hd_bf16.dims(),
        k_groups_len_hd_bf16.dims()
    );
    anyhow::ensure!(
        v_groups_len_hd_bf16.dims() == k_groups_len_hd_bf16.dims(),
        "e4b full attention gemm arm: v {:?} k {:?}",
        v_groups_len_hd_bf16.dims(),
        k_groups_len_hd_bf16.dims()
    );
    let q = q_groups_rows_hd_bf16.contiguous()?;
    let kv_head_stride_elems_lets_a_capacity_sized_twin_serve_its_prefix_without_a_copy =
        |t: &Tensor| -> Result<Option<usize>> {
            let (dims, strides) = (t.dims(), t.stride());
            if strides[2] == 1 && strides[1] == dims[2] && strides[0] % dims[2] == 0 {
                Ok(Some(strides[0]))
            } else {
                Ok(None)
            }
        };
    let (k, k_stride) =
        match kv_head_stride_elems_lets_a_capacity_sized_twin_serve_its_prefix_without_a_copy(
            k_groups_len_hd_bf16,
        )? {
            Some(st) => (k_groups_len_hd_bf16.clone(), st),
            None => {
                let c = k_groups_len_hd_bf16.contiguous()?;
                let st = len * hd;
                (c, st)
            }
        };
    let (v, v_stride) =
        match kv_head_stride_elems_lets_a_capacity_sized_twin_serve_its_prefix_without_a_copy(
            v_groups_len_hd_bf16,
        )? {
            Some(st) => (v_groups_len_hd_bf16.clone(), st),
            None => {
                let c = v_groups_len_hd_bf16.contiguous()?;
                let st = len * hd;
                (c, st)
            }
        };
    let tile = e4b_full_attn_key_tile();
    let stream = crate::cuda_stream::current_stream(&dev);
    let all_rows = nkv * rows;
    let scratch = E4B_FULL_ATTN_SCRATCH_GROWS_ONCE_PER_STREAM_BECAUSE_A_CUDAMALLOC_PER_CHUNK_LAYER_COST_4_PERCENT_AT_2K
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut scratch = scratch
        .lock()
        .map_err(|e| anyhow::anyhow!("e4b full attention scratch poisoned: {e}"))?;
    let key = nv_quant::stream_cache_key(&stream);
    let sc = scratch.entry(key).or_insert_with(|| E4bFullAttnScratch {
        s: None,
        p: None,
        m: None,
        l: None,
        o: None,
        q8: None,
        k8: None,
        scales_and_amax: None,
    });
    anyhow::ensure!(
        nkv == 2,
        "e4b fp8 scores: the two-segment per-tensor quantizer covers exactly two kv heads, got {nkv}"
    );
    let q8_dev = grow_u8(&stream, &mut sc.q8, nkv * rows * hd)?;
    let k8_dev = grow_u8(&stream, &mut sc.k8, nkv * len * hd)?;
    let scales_dev = grow_f32(&stream, &mut sc.scales_and_amax, 4)?;
    let s_dev = grow_f32(&stream, &mut sc.s, all_rows * tile)?;
    let p_dev = grow_u16(&stream, &mut sc.p, all_rows * tile)?;
    let m_dev = grow_f32(&stream, &mut sc.m, all_rows)?;
    let l_dev = grow_f32(&stream, &mut sc.l, all_rows)?;
    let o_dev = grow_f32(&stream, &mut sc.o, all_rows * hd)?;
    let mut out_dev: cudarc::driver::CudaSlice<half::bf16> =
        unsafe { stream.alloc::<half::bf16>(n * nkv * group * hd) }
            .map_err(|e| anyhow::anyhow!(e))?;
    {
        let (qs, _) = q.storage_and_layout();
        let (ks, _) = k.storage_and_layout();
        let (vs, _) = v.storage_and_layout();
        let cuda_of = |st: &candle_core::Storage| -> Result<u64> {
            match st {
                candle_core::Storage::Cuda(c) => {
                    let sl = c.as_cuda_slice::<half::bf16>()?;
                    let (p, _g) = sl.device_ptr(&stream);
                    Ok(p)
                }
                _ => anyhow::bail!("e4b full attention gemm arm: tensor not on cuda"),
            }
        };
        let qp = cuda_of(&qs)?;
        let kp = cuda_of(&ks)?;
        let vp = cuda_of(&vs)?;
        let (sp, _gs) = s_dev.device_ptr_mut(&stream);
        let (pp, _gp) = p_dev.device_ptr_mut(&stream);
        let (mp, _gm) = m_dev.device_ptr_mut(&stream);
        let (lp, _gl) = l_dev.device_ptr_mut(&stream);
        let (op, _go) = o_dev.device_ptr_mut(&stream);
        let (outp, _gout) = out_dev.device_ptr_mut(&stream);
        let q_pos0 = len - n;
        let (q8p, _g8) = q8_dev.device_ptr_mut(&stream);
        let (k8p, _gk8) = k8_dev.device_ptr_mut(&stream);
        let (scp, _gsc) = scales_dev.device_ptr_mut(&stream);
        let q_scale = scp as *mut f32;
        let k_scale = (scp + 4) as *mut f32;
        let amax = (scp + 8) as *mut u32;
        let rc = unsafe {
            nv_kernels::cuda::e4m3_pertensor_quant_two_segments(
                stream.cu_stream() as *mut _,
                qp as *const u16,
                (qp + (rows * hd * 2) as u64) as *const u16,
                (rows * hd) as i64,
                q8p as *mut u8,
                (q8p + (rows * hd) as u64) as *mut u8,
                q_scale,
                amax,
            )
        };
        anyhow::ensure!(rc == 0, "e4b fp8 scores: q quantize rc={rc}");
        let rc = unsafe {
            nv_kernels::cuda::e4m3_pertensor_quant_two_segments(
                stream.cu_stream() as *mut _,
                kp as *const u16,
                (kp + (k_stride * 2) as u64) as *const u16,
                (len * hd) as i64,
                k8p as *mut u8,
                (k8p + (len * hd) as u64) as *mut u8,
                k_scale,
                amax,
            )
        };
        anyhow::ensure!(rc == 0, "e4b fp8 scores: k quantize rc={rc}");
        let mut key_off = 0usize;
        let mut first = true;
        while key_off < len {
            let tile_len = tile.min(len - key_off);
            unsafe {
                let _ = FP8_SCORE_TILES_NEED_16_ALIGNED_ROWS_AND_KEYS_SO_A_RAGGED_TAIL_TILE_RUNS_THE_BF16_GEMM;
                if rows % 16 == 0 && tile_len % 16 == 0 {
                    nv_quant::matmul::e4m3_gemm_f32_out_strided_batched_raw(
                        &stream,
                        q8p as *const std::ffi::c_void,
                        (k8p + (key_off * hd) as u64) as *const std::ffi::c_void,
                        sp as *mut std::ffi::c_void,
                        rows as u64,
                        tile_len as u64,
                        hd as u64,
                        nkv as i32,
                        (rows * hd) as i64,
                        (len * hd) as i64,
                        (rows * tile_len) as i64,
                        q_scale as *const f32,
                        k_scale as *const f32,
                        0.0,
                    )?;
                } else {
                    nv_quant::matmul::bf16_gemm_f32_out_strided_batched_raw(
                        &stream,
                        qp as *const std::ffi::c_void,
                        (kp + (key_off * hd * 2) as u64) as *const std::ffi::c_void,
                        sp as *mut std::ffi::c_void,
                        rows as u64,
                        tile_len as u64,
                        hd as u64,
                        nkv as i32,
                        (rows * hd) as i64,
                        k_stride as i64,
                        (rows * tile_len) as i64,
                        true,
                        0.0,
                    )?;
                }
                let rc = nv_kernels::cuda::e4b_online_softmax_tile(
                    stream.cu_stream() as *mut _,
                    sp as *const f32,
                    pp as *mut u16,
                    mp as *mut f32,
                    lp as *mut f32,
                    op as *mut f32,
                    rows as i32,
                    nkv as i32,
                    n as i32,
                    tile_len as i32,
                    key_off as i32,
                    q_pos0 as i32,
                    hd as i32,
                    scale,
                    first as i32,
                );
                anyhow::ensure!(rc == 0, "e4b_online_softmax_tile returned {rc}");
                nv_quant::matmul::bf16_gemm_f32_out_strided_batched_raw(
                    &stream,
                    pp as *const std::ffi::c_void,
                    (vp + (key_off * hd * 2) as u64) as *const std::ffi::c_void,
                    op as *mut std::ffi::c_void,
                    rows as u64,
                    hd as u64,
                    tile_len as u64,
                    nkv as i32,
                    (rows * tile_len) as i64,
                    v_stride as i64,
                    (rows * hd) as i64,
                    false,
                    if first { 0.0 } else { 1.0 },
                )?;
            }
            first = false;
            key_off += tile_len;
        }
        let rc = unsafe {
            nv_kernels::cuda::e4b_attn_finalize(
                stream.cu_stream() as *mut _,
                op as *const f32,
                lp as *const f32,
                outp as *mut u16,
                n as i32,
                (nkv * group) as i32,
                hd as i32,
            )
        };
        anyhow::ensure!(rc == 0, "e4b_attn_finalize returned {rc}");
    }
    let storage = candle_core::CudaStorage::wrap_cuda_slice(out_dev, dev);
    let storage = candle_core::Storage::Cuda(storage);
    Ok(Tensor::from_storage(
        storage,
        (n, nkv * group * hd),
        candle_core::op::BackpropOp::none(),
        false,
    ))
}
