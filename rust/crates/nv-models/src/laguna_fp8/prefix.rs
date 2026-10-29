use anyhow::{ensure, Context, Result};
use cudarc::driver::CudaStream;
use half::bf16;
use std::sync::Arc;

use super::{Fp8LayerSlot, LagunaFp8ScalePolicy, LagunaKvCacheFp8};
use crate::laguna::prefix::{
    checked_bytes, prefix_stream, read_device_slice, rows_dtoh, rows_htod, validate_tensor_rows,
    PrefixLayout,
};

pub struct SlidingSnapshot {
    pos: usize,
    bytes: usize,
    layout: PrefixLayout,
    rows: Vec<(Vec<bf16>, Vec<bf16>)>,
}

struct PackedFullRows {
    k: Vec<u8>,
    v: Vec<u8>,
    k_scales: Vec<f32>,
    v_scales: Vec<f32>,
}

pub struct FullSnapshot {
    pos: usize,
    bytes: usize,
    layout: PrefixLayout,
    rows: Vec<PackedFullRows>,
}

impl SlidingSnapshot {
    pub fn pos(&self) -> usize {
        self.pos
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl FullSnapshot {
    pub fn pos(&self) -> usize {
        self.pos
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

fn full_bytes(layout: &PrefixLayout, pos: usize) -> Result<usize> {
    let scales = checked_bytes(pos, layout.n_kv, 8)?
        .checked_mul(layout.windows.iter().filter(|w| w.is_none()).count())
        .context("FP8 prefix scale byte sum overflows")?;
    layout
        .full_bytes(pos, 2)?
        .checked_add(scales)
        .context("FP8 prefix full byte sum overflows")
}

impl LagunaKvCacheFp8 {
    pub fn prefix_supported(&self) -> bool {
        self.prefix_layout().is_ok()
    }

    fn prefix_layout(&self) -> Result<(PrefixLayout, Arc<CudaStream>)> {
        ensure!(
            !nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_RING_ATTN"),
            "FP8 prefix reuse refuses the explicit ring-attention arm"
        );
        ensure!(
            self.scale_policy == LagunaFp8ScalePolicy::FixedOne,
            "FP8 prefix reuse supports only immutable FixedOne storage"
        );
        ensure!(
            !self.layers.is_empty()
                && self.layers.len() == self.layer_windows.len()
                && self.layers.len() == self.layer_stored.len()
                && self.max_seq_len > 0
                && self.max_seq_len <= i32::MAX as usize,
            "FP8 prefix cache metadata mismatch"
        );
        let stream = prefix_stream(&self.device)?;
        ensure!(
            Arc::ptr_eq(self.meta_dev.context(), stream.context())
                && Arc::ptr_eq(self.n_total_dev.context(), stream.context()),
            "FP8 prefix metadata context mismatch"
        );
        let layout = PrefixLayout {
            context: stream.context().clone(),
            capacity: self.max_seq_len,
            n_kv: self.n_kv,
            head_dim: self.head_dim,
            windows: self.layer_windows.clone(),
            caps: self
                .layer_windows
                .iter()
                .map(|w| {
                    if w.is_some() {
                        self.s_cap
                    } else {
                        self.max_seq_len
                    }
                })
                .collect(),
        };
        let re = layout.row_elems()?;
        let mut full_layers = 0;
        for (li, layer) in self.layers.iter().enumerate() {
            match (self.layer_windows[li], layer) {
                (
                    None,
                    Fp8LayerSlot::Full {
                        k_fp8,
                        v_fp8,
                        k_scales,
                        v_scales,
                    },
                ) => {
                    let elements = checked_bytes(self.max_seq_len, re, 1)?;
                    let scales = checked_bytes(self.max_seq_len, self.n_kv, 1)?;
                    ensure!(
                        k_fp8.len() == elements
                            && v_fp8.len() == elements
                            && k_scales.len() == scales
                            && v_scales.len() == scales
                            && Arc::ptr_eq(k_fp8.context(), stream.context())
                            && Arc::ptr_eq(v_fp8.context(), stream.context())
                            && Arc::ptr_eq(k_scales.context(), stream.context())
                            && Arc::ptr_eq(v_scales.context(), stream.context()),
                        "FP8 prefix layer {li} allocation/context mismatch"
                    );
                    full_layers += 1;
                }
                (Some(window), Fp8LayerSlot::Bf16 { k, v }) => {
                    ensure!(
                        window > 0
                            && window == self.s_window
                            && self.s_cap > 0
                            && k.dims() == [1, self.s_cap, self.n_kv, self.head_dim]
                            && v.dims() == k.dims(),
                        "FP8 prefix sliding layer {li} window/capacity mismatch"
                    );
                    validate_tensor_rows(&stream, k, 0, self.s_cap, re)?;
                    validate_tensor_rows(&stream, v, 0, self.s_cap, re)?;
                }
                _ => anyhow::bail!("FP8 prefix refuses mixed/full-BF16 fallback at layer {li}"),
            }
        }
        ensure!(
            full_layers > 0,
            "FP8 prefix requires an FP8 full-attention layer"
        );
        Ok((layout, stream))
    }

    fn prefix_initialized(&self) -> Result<()> {
        ensure!(
            self.current_len > 0
                && self.current_len <= self.max_seq_len
                && self.full_stored == self.current_len
                && self.s_stored <= self.s_cap
                && self.s_stored <= self.current_len,
            "FP8 prefix requires a completed initialized frontier"
        );
        for (li, window) in self.layer_windows.iter().enumerate() {
            let stored = if window.is_some() {
                self.s_stored
            } else {
                self.full_stored
            };
            ensure!(
                self.layer_stored[li] == stored
                    && window.is_none_or(|w| stored >= w.min(self.current_len)),
                "FP8 prefix layer {li} is not fully initialized"
            );
        }
        Ok(())
    }

    pub fn snapshot_sliding_bytes(&self) -> Result<usize> {
        let (layout, _) = self.prefix_layout()?;
        self.prefix_initialized()?;
        layout.sliding_bytes(self.current_len)
    }

    pub fn snapshot_full_bytes(&self, pos: usize) -> Result<usize> {
        let (layout, _) = self.prefix_layout()?;
        ensure!(
            self.holds_full_rows(pos),
            "FP8 prefix no longer holds full rows through {pos}"
        );
        full_bytes(&layout, pos)
    }

    pub fn snapshot_sliding(&self) -> Result<SlidingSnapshot> {
        let bytes = self.snapshot_sliding_bytes()?;
        let (layout, stream) = self.prefix_layout()?;
        let re = layout.row_elems()?;
        let pos = self.current_len;
        stream
            .context()
            .synchronize()
            .map_err(|e| anyhow::anyhow!("FP8 prefix snapshot sync: {e:?}"))?;
        let mut rows = Vec::new();
        for (window, layer) in self.layer_windows.iter().zip(&self.layers) {
            if let (Some(window), Fp8LayerSlot::Bf16 { k, v }) = (window, layer) {
                let need = (*window).min(pos);
                rows.push((
                    rows_dtoh(&stream, k, self.s_stored - need, need, re)?,
                    rows_dtoh(&stream, v, self.s_stored - need, need, re)?,
                ));
            }
        }
        Ok(SlidingSnapshot {
            pos,
            bytes,
            layout,
            rows,
        })
    }

    pub fn snapshot_full(&self, pos: usize) -> Result<FullSnapshot> {
        let bytes = self.snapshot_full_bytes(pos)?;
        let (layout, stream) = self.prefix_layout()?;
        let elements = checked_bytes(pos, layout.row_elems()?, 1)?;
        let scales = checked_bytes(pos, self.n_kv, 1)?;
        stream
            .context()
            .synchronize()
            .map_err(|e| anyhow::anyhow!("FP8 prefix snapshot sync: {e:?}"))?;
        let mut rows = Vec::new();
        for layer in &self.layers {
            if let Fp8LayerSlot::Full {
                k_fp8,
                v_fp8,
                k_scales,
                v_scales,
            } = layer
            {
                let k = read_device_slice(&stream, &k_fp8.slice(..elements))?;
                let v = read_device_slice(&stream, &v_fp8.slice(..elements))?;
                let k_scales = read_device_slice(&stream, &k_scales.slice(..scales))?;
                let v_scales = read_device_slice(&stream, &v_scales.slice(..scales))?;
                ensure!(
                    k_scales
                        .iter()
                        .chain(&v_scales)
                        .all(|value| value.to_bits() == 1.0f32.to_bits()),
                    "FP8 prefix initialized scale metadata violates FixedOne"
                );
                rows.push(PackedFullRows {
                    k,
                    v,
                    k_scales,
                    v_scales,
                });
            }
        }
        Ok(FullSnapshot {
            pos,
            bytes,
            layout,
            rows,
        })
    }

    pub fn holds_full_rows(&self, pos: usize) -> bool {
        pos > 0
            && pos <= self.current_len
            && self.prefix_layout().is_ok()
            && self.prefix_initialized().is_ok()
            && self
                .layer_windows
                .iter()
                .enumerate()
                .filter(|(_, window)| window.is_none())
                .all(|(li, _)| self.layer_stored[li] >= pos)
    }

    pub fn rewind_to(
        &mut self,
        sliding: &SlidingSnapshot,
        full: Option<&FullSnapshot>,
    ) -> Result<()> {
        let (layout, stream) = self.prefix_layout()?;
        let target = sliding.pos;
        let re = layout.row_elems()?;
        ensure!(
            target > 0
                && target <= self.max_seq_len
                && layout.matches(&sliding.layout)
                && sliding.bytes == layout.sliding_bytes(target)?
                && sliding.rows.len() == layout.windows.iter().flatten().count(),
            "FP8 prefix sliding snapshot position/layout/bytes/layer mismatch"
        );
        for ((k, v), window) in sliding.rows.iter().zip(layout.windows.iter().flatten()) {
            let need = checked_bytes((*window).min(target), re, 1)?;
            ensure!(
                k.len() == need && v.len() == need,
                "FP8 prefix sliding source length mismatch"
            );
        }
        if let Some(full) = full {
            ensure!(
                full.pos >= target
                    && full.pos <= self.max_seq_len
                    && layout.matches(&full.layout)
                    && full.bytes == full_bytes(&layout, full.pos)?
                    && full.rows.len() == layout.windows.iter().filter(|w| w.is_none()).count(),
                "FP8 prefix full snapshot position/layout/bytes/layer mismatch"
            );
            let elements = checked_bytes(full.pos, re, 1)?;
            let scales = checked_bytes(full.pos, self.n_kv, 1)?;
            for row in &full.rows {
                ensure!(
                    row.k.len() == elements
                        && row.v.len() == elements
                        && row.k_scales.len() == scales
                        && row.v_scales.len() == scales
                        && row
                            .k_scales
                            .iter()
                            .chain(&row.v_scales)
                            .all(|value| value.to_bits() == 1.0f32.to_bits()),
                    "FP8 prefix full source length/FixedOne scale mismatch"
                );
            }
        } else {
            ensure!(
                self.holds_full_rows(target),
                "FP8 prefix resident full rows are unavailable"
            );
        }
        stream
            .context()
            .synchronize()
            .map_err(|e| anyhow::anyhow!("FP8 prefix restore sync: {e:?}"))?;
        let result = (|| -> Result<()> {
            self.reset();
            if let Some(full) = full {
                let elements = target * re;
                let scales = target * self.n_kv;
                let mut sources = full.rows.iter();
                for layer in &mut self.layers {
                    if let Fp8LayerSlot::Full {
                        k_fp8,
                        v_fp8,
                        k_scales,
                        v_scales,
                    } = layer
                    {
                        let row = sources
                            .next()
                            .context("FP8 prefix full layer count changed after validation")?;
                        for (dst, src) in [(k_fp8, &row.k), (v_fp8, &row.v)] {
                            let copy = stream
                                .memcpy_htod(&src[..elements], &mut dst.slice_mut(..elements));
                            let sync = stream.synchronize();
                            copy?;
                            sync?;
                        }
                        for (dst, src) in [(k_scales, &row.k_scales), (v_scales, &row.v_scales)] {
                            let copy =
                                stream.memcpy_htod(&src[..scales], &mut dst.slice_mut(..scales));
                            let sync = stream.synchronize();
                            copy?;
                            sync?;
                        }
                    }
                }
            }
            let mut sources = sliding.rows.iter();
            for (li, layer) in self.layers.iter().enumerate() {
                match (layout.windows[li], layer) {
                    (Some(window), Fp8LayerSlot::Bf16 { k, v }) => {
                        let (hk, hv) = sources
                            .next()
                            .context("FP8 prefix sliding layer count changed after validation")?;
                        rows_htod(&stream, k, 0, hk, re)?;
                        rows_htod(&stream, v, 0, hv, re)?;
                        self.layer_stored[li] = window.min(target);
                    }
                    (None, Fp8LayerSlot::Full { .. }) => self.layer_stored[li] = target,
                    _ => anyhow::bail!("FP8 prefix layer type changed after validation"),
                }
            }
            let stored = layout
                .windows
                .iter()
                .flatten()
                .map(|w| (*w).min(target))
                .max()
                .unwrap_or(0);
            *self.host_meta = [target as i32, stored as i32, 0, stored as i32];
            self.host_n_total[0] = target as i32;
            let metadata = stream.memcpy_htod(&self.host_meta[..], &mut self.meta_dev);
            let total = stream.memcpy_htod(&self.host_n_total[..], &mut self.n_total_dev);
            let sync = stream.synchronize();
            metadata?;
            total?;
            sync?;
            self.full_stored = target;
            self.s_stored = stored;
            self.current_len = target;
            Ok(())
        })();
        if result.is_err() {
            self.reset();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{gemma4::LayerType, laguna::LagunaConfig};
    use candle_core::{DType, Device, Tensor};
    use cudarc::driver::DevicePtr;
    use nv_kernels::graph::CudaGraphRunner;

    fn code(index: usize, seed: usize) -> u8 {
        ((index * 29 + seed * 17) % 127) as u8 | ((((index / 7 + seed) & 1) as u8) << 7)
    }

    fn decode(code: u8) -> bf16 {
        let magnitude = code & 127;
        let exponent = (magnitude >> 3) as i32;
        let mantissa = (magnitude & 7) as f32;
        let value = if exponent == 0 {
            mantissa / 512.0
        } else {
            (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent - 7)
        };
        bf16::from_f32(if code & 128 != 0 { -value } else { value })
    }

    fn values(start: usize, rows: usize, seed: usize, sliding: bool) -> Vec<bf16> {
        (start * 1024..(start + rows) * 1024)
            .map(|i| {
                if sliding {
                    bf16::from_f32(((i * 13 + seed * 31) % 991) as f32 / 119.0 - 4.0)
                } else {
                    decode(code(i, seed))
                }
            })
            .collect()
    }

    fn append(cache: &mut LagunaKvCacheFp8, end: usize, seed: usize) -> Result<()> {
        while cache.current_len < end {
            let start = cache.current_len;
            let rows = (end - start).min(256);
            cache.prepare_for_decode_dev(start, start + rows)?;
            for layer in 0..2 {
                let k = Tensor::from_vec(
                    values(start, rows, seed + layer * 2, layer == 1),
                    (1, rows, 8, 128),
                    &cache.device,
                )?;
                let v = Tensor::from_vec(
                    values(start, rows, seed + layer * 2 + 1, layer == 1),
                    (1, rows, 8, 128),
                    &cache.device,
                )?;
                cache.write_at_impl(layer, &k, &v)?;
                cache.dev.cuda_stream().synchronize()?;
            }
            cache.advance(rows);
        }
        Ok(())
    }

    fn inspect(cache: &LagunaKvCacheFp8) -> Result<Vec<u32>> {
        let stream = cache.dev.cuda_stream();
        let mut out = vec![
            cache.current_len as u32,
            cache.full_stored as u32,
            cache.s_stored as u32,
        ];
        out.extend(cache.layer_stored.iter().map(|n| *n as u32));
        out.extend(
            stream
                .clone_dtoh(&cache.meta_dev)?
                .iter()
                .map(|n| *n as u32),
        );
        out.extend(
            stream
                .clone_dtoh(&cache.n_total_dev)?
                .iter()
                .map(|n| *n as u32),
        );
        for layer in &cache.layers {
            match layer {
                Fp8LayerSlot::Full {
                    k_fp8,
                    v_fp8,
                    k_scales,
                    v_scales,
                } => {
                    for packed in [k_fp8, v_fp8] {
                        out.extend(
                            stream
                                .clone_dtoh(&packed.slice(..cache.current_len * 1024))?
                                .into_iter()
                                .map(u32::from),
                        );
                    }
                    for scales in [k_scales, v_scales] {
                        out.extend(
                            stream
                                .clone_dtoh(&scales.slice(..cache.current_len * 8))?
                                .into_iter()
                                .map(f32::to_bits),
                        );
                    }
                }
                Fp8LayerSlot::Bf16 { k, v } => {
                    for tensor in [k, v] {
                        out.extend(
                            tensor
                                .narrow(1, 0, cache.s_stored)?
                                .flatten_all()?
                                .to_vec1::<bf16>()?
                                .into_iter()
                                .map(|value| value.to_bits() as u32),
                        );
                    }
                }
            }
        }
        Ok(out)
    }

    fn check(cache: &LagunaKvCacheFp8, seed: usize) -> Result<()> {
        let stream = cache.dev.cuda_stream();
        let pos = cache.current_len;
        let Fp8LayerSlot::Full {
            k_fp8,
            v_fp8,
            k_scales,
            v_scales,
        } = &cache.layers[0]
        else {
            unreachable!()
        };
        for (packed, scales, offset) in [(k_fp8, k_scales, 0), (v_fp8, v_scales, 1)] {
            let actual = stream.clone_dtoh(&packed.slice(..pos * 1024))?;
            assert_eq!(
                actual,
                (0..pos * 1024)
                    .map(|i| code(i, seed + offset))
                    .collect::<Vec<_>>()
            );
            assert!(stream
                .clone_dtoh(&scales.slice(..pos * 8))?
                .iter()
                .all(|value| value.to_bits() == 1.0f32.to_bits()));
        }
        let Fp8LayerSlot::Bf16 { k, v } = &cache.layers[1] else {
            unreachable!()
        };
        for (tensor, offset) in [(k, 2), (v, 3)] {
            let actual = tensor
                .narrow(1, 0, cache.s_stored)?
                .flatten_all()?
                .to_vec1::<bf16>()?;
            let expected = values(pos - cache.s_stored, cache.s_stored, seed + offset, true);
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        assert_eq!(cache.layer_stored, vec![pos, cache.s_stored]);
        Ok(())
    }

    fn pointers(cache: &LagunaKvCacheFp8) -> Vec<u64> {
        let stream = cache.dev.cuda_stream();
        let mut p = vec![
            cache.meta_dev.device_ptr(&stream).0,
            cache.n_total_dev.device_ptr(&stream).0,
        ];
        let (k, v, ks, vs) = cache.layer_fp8_ptrs(0).unwrap();
        p.extend([k, v, ks, vs]);
        let Fp8LayerSlot::Bf16 { k, v } = &cache.layers[1] else {
            unreachable!()
        };
        for tensor in [k, v] {
            let (storage, layout) = tensor.storage_and_layout();
            let candle_core::Storage::Cuda(storage) = &*storage else {
                unreachable!()
            };
            p.push(
                storage
                    .as_cuda_slice::<bf16>()
                    .unwrap()
                    .slice(layout.start_offset()..)
                    .device_ptr(&stream)
                    .0,
            );
        }
        p
    }

    #[test]
    #[ignore]
    fn fixed_one_prefix_cuda_gate() -> Result<()> {
        anyhow::ensure!(std::env::var("NV_LAGUNA_PREFIX_CUDA_TEST").as_deref() == Ok("1")
            && std::env::var("NV_LAGUNA_FP8_UNIT_SCALE").as_deref() == Ok("1")
            && !nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_FP8_BF16_FULL")
            && !nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_RING_ATTN"),
            "set NV_LAGUNA_PREFIX_CUDA_TEST=1, UNIT_SCALE=1 and unset FP8_BF16_FULL/RING_ATTN; no skip-as-pass");
        let config = LagunaConfig::from_hf_json_str(
            r#"{
            "vocab_size":32,"hidden_size":6144,"intermediate_size":64,
            "num_hidden_layers":2,"num_attention_heads":48,"num_key_value_heads":8,
            "head_dim":128,"max_position_embeddings":262144,"rms_norm_eps":0.00001,
            "sliding_window":512,"layer_types":["full_attention","sliding_attention"],
            "num_experts":0,"num_experts_per_tok":0,"moe_intermediate_size":0,
            "shared_expert_intermediate_size":0,
            "rope_parameters":{"full_attention":{"rope_theta":500000},"sliding_attention":{"rope_theta":500000}}
        }"#,
        )?;
        assert!(matches!(config.layer_types[0], LayerType::FullAttention));
        let device = Device::new_cuda(0)?;
        let mut cache = LagunaKvCacheFp8::new(&config, 262144, &device, DType::BF16)?;
        let stream = cache.dev.cuda_stream();
        let original_pointers = pointers(&cache);
        assert!(cache.prefix_supported());
        assert!(cache.snapshot_sliding_bytes().is_err());
        cache.prepare_for_decode_dev(0, 1)?;
        cache.advance(1);
        assert!(cache.snapshot_full_bytes(1).is_err() && cache.snapshot_full(1).is_err());
        cache.reset();
        append(&mut cache, 1024, 11)?;
        let mut a = cache.snapshot_sliding()?;
        assert_eq!(a.bytes(), 2 * 512 * 1024 * 2);
        assert_eq!(a.bytes(), cache.snapshot_sliding_bytes()?);
        append(&mut cache, 1536, 11)?;
        let b = cache.snapshot_sliding()?;
        let mut full = cache.snapshot_full(1536)?;
        assert_eq!(full.bytes(), 1536 * (2048 + 64));
        assert_eq!(full.bytes(), cache.snapshot_full_bytes(1536)?);
        append(&mut cache, 2048, 11)?;
        cache.rewind_to(&a, None)?;
        check(&cache, 11)?;
        assert_eq!(cache.current_len, 1024);
        cache.reset();
        assert!(!cache.holds_full_rows(1024));
        assert!(cache.rewind_to(&a, None).is_err());
        append(&mut cache, 2048, 61)?;
        let before = inspect(&cache)?;
        let removed = a.rows[0].1.pop().unwrap();
        assert!(cache.rewind_to(&a, Some(&full)).is_err());
        a.rows[0].1.push(removed);
        assert_eq!(inspect(&cache)?, before);
        let removed = full.rows[0].k.pop().unwrap();
        assert!(cache.rewind_to(&a, Some(&full)).is_err());
        full.rows[0].k.push(removed);
        assert_eq!(inspect(&cache)?, before);
        full.rows[0].v_scales[0] = 0.0;
        assert!(cache.rewind_to(&a, Some(&full)).is_err());
        full.rows[0].v_scales[0] = 1.0;
        assert_eq!(inspect(&cache)?, before);
        a.layout.capacity -= 1;
        assert!(cache.rewind_to(&a, Some(&full)).is_err());
        a.layout.capacity += 1;
        let row = a.rows.pop().unwrap();
        assert!(cache.rewind_to(&a, Some(&full)).is_err());
        a.rows.push(row);
        assert_eq!(inspect(&cache)?, before);
        cache.scale_policy = LagunaFp8ScalePolicy::PerTokenHead;
        assert!(!cache.prefix_supported() && cache.snapshot_sliding_bytes().is_err());
        assert!(cache.rewind_to(&a, Some(&full)).is_err());
        cache.scale_policy = LagunaFp8ScalePolicy::FixedOne;
        assert_eq!(inspect(&cache)?, before);
        let forked = stream.context().new_stream()?;
        let refused =
            nv_layers::cuda_stream::with_stream(forked, || cache.rewind_to(&a, Some(&full)));
        assert!(refused.is_err());
        assert_eq!(inspect(&cache)?, before);
        cache.rewind_to(&a, Some(&full))?;
        check(&cache, 11)?;
        assert_eq!(cache.current_len, 1024);
        cache.reset();
        cache.rewind_to(&b, Some(&full))?;
        check(&cache, 11)?;
        assert_eq!(cache.current_len, 1536);
        assert_eq!(pointers(&cache), original_pointers);
        for m in [1usize, 5] {
            let input = Tensor::from_vec(values(1024, m, 91, false), (1, m, 8, 128), &device)?;
            device.synchronize()?;
            let ip = {
                let (storage, layout) = input.storage_and_layout();
                let candle_core::Storage::Cuda(storage) = &*storage else {
                    unreachable!()
                };
                let ptr = storage
                    .as_cuda_slice::<bf16>()?
                    .slice(layout.start_offset()..)
                    .device_ptr(&stream)
                    .0;
                ptr
            };
            let graph_stream = stream.context().new_stream()?;
            let mut graph = CudaGraphRunner::new(graph_stream.clone());
            let p = pointers(&cache);
            cache.prepare_for_decode_dev(cache.current_len, cache.current_len + m)?;
            stream.synchronize()?;
            graph.run(m as u64, |s| {
                let rc = unsafe {
                    nv_kernels::cuda::kv_append_fp8_unit_scale(
                        s.cu_stream() as *mut std::ffi::c_void,
                        ip as *const u16,
                        ip as *const u16,
                        p[2] as *mut u8,
                        p[3] as *mut u8,
                        p[4] as *mut f32,
                        p[5] as *mut f32,
                        p[0] as *const i32,
                        m as i32,
                        8,
                        128,
                        0,
                    )
                };
                anyhow::ensure!(rc == 0, "prefix captured FP8 writer rc={rc}");
                for dst in &p[6..8] {
                    let rc = unsafe {
                        nv_kernels::cuda::kv_ring_append_bf16(
                            s.cu_stream() as *mut std::ffi::c_void,
                            ip as *const u16,
                            *dst as *mut u16,
                            (p[0] as *const i32).add(1),
                            m as i32,
                            768,
                            8,
                            128,
                        )
                    };
                    anyhow::ensure!(rc == 0, "prefix captured sliding writer rc={rc}");
                }
                Ok(())
            })?;
            graph_stream.synchronize()?;
            assert!(graph.has_cached_token(m as u64) && graph.cached_node_count() > 0);
            cache.reset();
            cache.rewind_to(&a, Some(&full))?;
            check(&cache, 11)?;
            cache.prepare_for_decode_dev(1024, 1024 + m)?;
            stream.synchronize()?;
            graph.run(m as u64, |_| {
                anyhow::bail!("prefix restore unexpectedly recaptured retained writer")
            })?;
            graph_stream.synchronize()?;
            cache.advance(m);
            cache.note_graph_write();
            let Fp8LayerSlot::Full {
                k_fp8,
                v_fp8,
                k_scales,
                v_scales,
            } = &cache.layers[0]
            else {
                unreachable!()
            };
            for packed in [k_fp8, v_fp8] {
                assert_eq!(
                    stream.clone_dtoh(&packed.slice(1024 * 1024..(1024 + m) * 1024))?,
                    (1024 * 1024..(1024 + m) * 1024)
                        .map(|i| code(i, 91))
                        .collect::<Vec<_>>()
                );
            }
            for scales in [k_scales, v_scales] {
                assert!(stream
                    .clone_dtoh(&scales.slice(1024 * 8..(1024 + m) * 8))?
                    .iter()
                    .all(|v| v.to_bits() == 1.0f32.to_bits()));
            }
            let Fp8LayerSlot::Bf16 { k, v } = &cache.layers[1] else {
                unreachable!()
            };
            for tensor in [k, v] {
                assert_eq!(
                    tensor
                        .narrow(1, 512, m)?
                        .flatten_all()?
                        .to_vec1::<bf16>()?
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    values(1024, m, 91, false)
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
            assert_eq!(pointers(&cache), original_pointers);
            graph.invalidate();
            let _no_capture_in_flight = nv_kernels::capture_exclusion::no_capture_in_flight();
            stream.context().synchronize()?;
        }
        cache.reset();
        cache.rewind_to(&b, Some(&full))?;
        let slot = crate::laguna_step_graph::StepCache::Fp8(cache);
        assert!(slot.prefix_supported() && slot.prefix_holds_full_rows(1024));
        assert_eq!(slot.prefix_snapshot_sliding()?.bytes(), b.bytes());
        assert_eq!(slot.prefix_snapshot_full(1536)?.pos(), 1536);
        eprintln!("LAGUNA-PREFIX-CUDA fixed_one=true capacity=262144 checkpoints=1024,1536 covering_restore=true raw_kv=exact scales=exact_one sliding=bf16_exact refusal_before_mutation=true reset_reuse=true captured_writer_m1_m5=true graph_pointers=stable generated_tokens=0 model_quality=false timing_measured=false");
        Ok(())
    }
}
