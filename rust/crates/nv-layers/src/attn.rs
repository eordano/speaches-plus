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
    if matches!(q.device(), candle_core::Device::Cuda(_))
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
    let scale = Tensor::new(cfg.softmax_scale, q_flat.device()).map_err(|e| anyhow::anyhow!(e))?;
    let mut scores = q_flat
        .matmul(&k_perm)
        .map_err(|e| anyhow::anyhow!(e))?
        .broadcast_mul(&scale)
        .map_err(|e| anyhow::anyhow!(e))?;

    if sliding_window > 0 && !cfg.causal {
        anyhow::bail!("sdpa: a sliding window is only defined on the causal path");
    }
    if cfg.causal {
        let mask = build_causal_mask_windowed(sq, sk, sliding_window, q_flat.device())?;
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

    let probs = candle_nn::ops::softmax_last_dim(&scores).map_err(|e| anyhow::anyhow!(e))?;
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
pub const CANDLE_FLASH_ATTN_LAUNCHES_ON_THE_LEGACY_NULL_STREAM_SO_EVERY_CALL_IS_EVENT_FENCED:
    &str = "candle-flash-attn's flash_api.cu hardcodes cudaStream_t stream = 0, so its kernel \
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
#[allow(clippy::too_many_arguments)]
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
    fn sdpa_gqa_expansion_matches_repeated_heads() {
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

    #[test]
    fn flash_decode_merge_needs_the_neg_inf_guard() {
        let hd = 4usize;
        let q = vec![1.0f32; hd];
        let k = vec![vec![0.5f32; hd]; 3];
        let v = vec![vec![1.0f32; hd]; 3];
        let got = flash_decode_model(&q, &k, &v, 3, 0, 16, hd);
        for d in 0..hd {
            assert!(
                (got[d] - 1.0).abs() < 1e-6,
                "uniform V must return exactly V regardless of empty splits, got {}",
                got[d]
            );
        }

        let got = flash_decode_model(&q, &[], &[], 0, 0, 16, hd);
        assert!(
            got.iter().all(|&x| x == 0.0),
            "empty range must produce zeros, not NaN"
        );
    }
}
