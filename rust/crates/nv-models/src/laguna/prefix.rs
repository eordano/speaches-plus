use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use cudarc::driver::{CudaContext, CudaStream, DevicePtr, DeviceRepr};
use half::bf16;
use std::sync::Arc;

use super::LagunaKvCache;

#[derive(Clone)]
pub(crate) struct PrefixLayout {
    pub context: Arc<CudaContext>,
    pub capacity: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub caps: Vec<usize>,
    pub windows: Vec<Option<usize>>,
}

impl PrefixLayout {
    pub fn matches(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.context, &other.context)
            && self.capacity == other.capacity
            && self.n_kv == other.n_kv
            && self.head_dim == other.head_dim
            && self.caps == other.caps
            && self.windows == other.windows
    }

    pub fn row_elems(&self) -> Result<usize> {
        self.n_kv
            .checked_mul(self.head_dim)
            .filter(|n| *n > 0)
            .context("prefix KV row size is zero or overflows")
    }

    pub fn sliding_bytes(&self, pos: usize) -> Result<usize> {
        let re = self.row_elems()?;
        self.windows
            .iter()
            .flatten()
            .try_fold(0usize, |sum, window| {
                sum.checked_add(checked_bytes((*window).min(pos), re, 4)?)
                    .context("prefix byte sum overflows")
            })
    }

    pub fn full_bytes(&self, pos: usize, bytes_per_pair_element: usize) -> Result<usize> {
        let per_layer = checked_bytes(pos, self.row_elems()?, bytes_per_pair_element)?;
        self.windows
            .iter()
            .filter(|w| w.is_none())
            .count()
            .checked_mul(per_layer)
            .context("prefix full byte sum overflows")
    }
}

pub(crate) fn checked_bytes(rows: usize, row_elems: usize, width: usize) -> Result<usize> {
    rows.checked_mul(row_elems)
        .and_then(|n| n.checked_mul(width))
        .context("prefix byte size overflows")
}

pub(crate) fn read_device_slice<T: DeviceRepr + Clone + Default, S: DevicePtr<T>>(
    stream: &Arc<CudaStream>,
    source: &S,
) -> Result<Vec<T>> {
    checked_bytes(source.len(), 1, std::mem::size_of::<T>())?;
    let mut host = Vec::new();
    host.try_reserve_exact(source.len())
        .map_err(|error| anyhow::anyhow!("prefix snapshot allocation: {error}"))?;
    host.resize(source.len(), T::default());
    if host.is_empty() {
        return Ok(host);
    }
    let copy = stream.memcpy_dtoh(source, &mut host);
    let sync = stream.synchronize();
    copy.map_err(|e| anyhow::anyhow!("prefix D2H: {e:?}"))?;
    sync.map_err(|e| anyhow::anyhow!("prefix D2H sync: {e:?}"))?;
    Ok(host)
}

pub(crate) fn prefix_stream(device: &Device) -> Result<Arc<CudaStream>> {
    let Device::Cuda(dev) = device else {
        bail!("prefix snapshots require CUDA");
    };
    let stream = nv_layers::cuda_stream::current_stream(dev);
    ensure!(
        stream.cu_stream() == dev.cuda_stream().cu_stream()
            && Arc::ptr_eq(stream.context(), dev.cuda_stream().context()),
        "prefix snapshots require the cache legacy stream"
    );
    ensure!(
        stream.capture_status()?
            == cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
        "prefix snapshots cannot copy during CUDA capture"
    );
    Ok(stream)
}

pub(crate) fn validate_tensor_rows(
    stream: &Arc<CudaStream>,
    tensor: &Tensor,
    row0: usize,
    rows: usize,
    row_elems: usize,
) -> Result<()> {
    let dims = tensor.dims();
    let end = row0
        .checked_add(rows)
        .context("prefix row range overflows")?;
    ensure!(
        dims.len() == 4
            && dims[0] == 1
            && row_elems > 0
            && dims[2].checked_mul(dims[3]) == Some(row_elems)
            && end <= dims[1]
            && tensor.dtype() == DType::BF16
            && tensor.is_contiguous(),
        "prefix tensor shape/dtype/row range mismatch"
    );
    let (storage, layout) = tensor.storage_and_layout();
    let candle_core::Storage::Cuda(storage) = &*storage else {
        bail!("prefix tensor is not CUDA");
    };
    let slice = storage.as_cuda_slice::<bf16>()?;
    let end = end
        .checked_mul(row_elems)
        .and_then(|n| n.checked_add(layout.start_offset()))
        .context("prefix storage range overflows")?;
    ensure!(
        end <= slice.len() && Arc::ptr_eq(slice.context(), stream.context()),
        "prefix tensor storage/context mismatch"
    );
    Ok(())
}

pub(crate) fn rows_dtoh(
    stream: &Arc<CudaStream>,
    src: &Tensor,
    row0: usize,
    n_rows: usize,
    row_elems: usize,
) -> Result<Vec<bf16>> {
    validate_tensor_rows(stream, src, row0, n_rows, row_elems)?;
    let (storage, layout) = src.storage_and_layout();
    let candle_core::Storage::Cuda(storage) = &*storage else {
        bail!("prefix source is not CUDA");
    };
    let slice = storage.as_cuda_slice::<bf16>()?.slice(
        layout.start_offset() + row0 * row_elems
            ..layout.start_offset() + (row0 + n_rows) * row_elems,
    );
    read_device_slice(stream, &slice)
}

pub(crate) fn rows_htod(
    stream: &Arc<CudaStream>,
    dst: &Tensor,
    row0: usize,
    host: &[bf16],
    row_elems: usize,
) -> Result<()> {
    ensure!(
        row_elems > 0 && host.len().is_multiple_of(row_elems),
        "prefix host buffer is ragged"
    );
    let rows = host.len() / row_elems;
    validate_tensor_rows(stream, dst, row0, rows, row_elems)?;
    if host.is_empty() {
        return Ok(());
    }
    let (storage, layout) = dst.storage_and_layout();
    let candle_core::Storage::Cuda(storage) = &*storage else {
        bail!("prefix destination is not CUDA");
    };
    let slice = storage.as_cuda_slice::<bf16>()?.slice(
        layout.start_offset() + row0 * row_elems..layout.start_offset() + (row0 + rows) * row_elems,
    );
    let (ptr, _guard) = slice.device_ptr(stream);
    let copy = unsafe { cudarc::driver::result::memcpy_htod_async(ptr, host, stream.cu_stream()) };
    let sync = stream.synchronize();
    copy.map_err(|e| anyhow::anyhow!("prefix H2D: {e:?}"))?;
    sync.map_err(|e| anyhow::anyhow!("prefix H2D sync: {e:?}"))
}

pub struct SlidingSnapshot {
    pub pos: usize,
    pub n_rows: usize,
    pub bytes: usize,
    layout: PrefixLayout,
    rows: Vec<(Vec<bf16>, Vec<bf16>)>,
}

pub struct FullSnapshot {
    pub pos: usize,
    pub bytes: usize,
    layout: PrefixLayout,
    rows: Vec<(Vec<bf16>, Vec<bf16>)>,
}

impl LagunaKvCache {
    pub fn prefix_supported(&self) -> bool {
        self.prefix_layout().is_ok()
    }

    fn prefix_layout(&self) -> Result<(PrefixLayout, Arc<CudaStream>)> {
        let ring = self
            .ring
            .as_ref()
            .context("BF16 prefix reuse refuses legacy KV storage")?;
        ensure!(
            !ring.ring_attn,
            "BF16 prefix reuse refuses ring-attention mode"
        );
        ensure!(
            !self.layers.is_empty()
                && self.layers.len() == self.layer_windows.len()
                && self.layers.len() == self.layer_caps.len()
                && self.layers.len() == self.layer_stored.len()
                && self.max_seq_len > 0
                && self.max_seq_len <= i32::MAX as usize,
            "BF16 prefix cache metadata mismatch"
        );
        let stream = prefix_stream(&Device::Cuda(ring.dev.clone()))?;
        let layout = PrefixLayout {
            context: stream.context().clone(),
            capacity: self.max_seq_len,
            n_kv: self.n_kv,
            head_dim: self.head_dim,
            caps: self.layer_caps.clone(),
            windows: self.layer_windows.clone(),
        };
        let re = layout.row_elems()?;
        for (li, (k, v)) in self.layers.iter().enumerate() {
            let cap = if let Some(window) = self.layer_windows[li] {
                ensure!(
                    window > 0 && window == ring.s_window,
                    "BF16 prefix ring requires uniform sliding windows"
                );
                ring.s_cap
            } else {
                self.max_seq_len
            };
            ensure!(
                self.layer_caps[li] == cap
                    && cap > 0
                    && k.dims() == [1, cap, self.n_kv, self.head_dim]
                    && v.dims() == k.dims(),
                "BF16 prefix layer {li} capacity mismatch"
            );
            validate_tensor_rows(&stream, k, 0, cap, re)?;
            validate_tensor_rows(&stream, v, 0, cap, re)?;
        }
        Ok((layout, stream))
    }

    fn prefix_initialized(&self) -> Result<()> {
        let ring = self
            .ring
            .as_ref()
            .context("BF16 prefix requires ring metadata")?;
        ensure!(
            self.current_len > 0
                && self.current_len <= self.max_seq_len
                && ring.full_stored == self.current_len
                && ring.s_stored <= ring.s_cap
                && ring.s_stored <= self.current_len,
            "BF16 prefix requires a completed initialized frontier"
        );
        for (li, window) in self.layer_windows.iter().enumerate() {
            let stored = if window.is_some() {
                ring.s_stored
            } else {
                ring.full_stored
            };
            ensure!(
                self.layer_stored[li] == stored
                    && window.is_none_or(|window| stored >= window.min(self.current_len)),
                "BF16 prefix layer {li} is not fully initialized"
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
            "BF16 prefix no longer holds full rows through {pos}"
        );
        layout.full_bytes(pos, 4)
    }

    pub fn snapshot_sliding(&self) -> Result<SlidingSnapshot> {
        let bytes = self.snapshot_sliding_bytes()?;
        let (layout, stream) = self.prefix_layout()?;
        let pos = self.current_len;
        let re = layout.row_elems()?;
        let ring = self
            .ring
            .as_ref()
            .context("BF16 prefix requires ring metadata")?;
        stream
            .context()
            .synchronize()
            .map_err(|e| anyhow::anyhow!("prefix snapshot sync: {e:?}"))?;
        let mut rows = Vec::new();
        let mut n_rows = 0;
        for (li, window) in layout.windows.iter().enumerate() {
            if let Some(window) = window {
                let need = (*window).min(pos);
                let (k, v) = &self.layers[li];
                rows.push((
                    rows_dtoh(&stream, k, ring.s_stored - need, need, re)?,
                    rows_dtoh(&stream, v, ring.s_stored - need, need, re)?,
                ));
                n_rows = n_rows.max(need);
            }
        }
        Ok(SlidingSnapshot {
            pos,
            n_rows,
            bytes,
            layout,
            rows,
        })
    }

    pub fn snapshot_full(&self, pos: usize) -> Result<FullSnapshot> {
        let bytes = self.snapshot_full_bytes(pos)?;
        let (layout, stream) = self.prefix_layout()?;
        let re = layout.row_elems()?;
        stream
            .context()
            .synchronize()
            .map_err(|e| anyhow::anyhow!("prefix snapshot sync: {e:?}"))?;
        let mut rows = Vec::new();
        for (li, window) in layout.windows.iter().enumerate() {
            if window.is_none() {
                let (k, v) = &self.layers[li];
                rows.push((
                    rows_dtoh(&stream, k, 0, pos, re)?,
                    rows_dtoh(&stream, v, 0, pos, re)?,
                ));
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
                && sliding.rows.len() == layout.windows.iter().flatten().count()
                && sliding.n_rows
                    == layout
                        .windows
                        .iter()
                        .flatten()
                        .map(|w| (*w).min(target))
                        .max()
                        .unwrap_or(0),
            "BF16 prefix sliding snapshot position/layout/bytes/layer mismatch"
        );
        for ((k, v), window) in sliding.rows.iter().zip(layout.windows.iter().flatten()) {
            let need = checked_bytes((*window).min(target), re, 1)?;
            ensure!(
                k.len() == need && v.len() == need,
                "BF16 prefix sliding source length mismatch"
            );
        }
        if let Some(full) = full {
            ensure!(
                full.pos >= target
                    && full.pos <= self.max_seq_len
                    && layout.matches(&full.layout)
                    && full.bytes == layout.full_bytes(full.pos, 4)?
                    && full.rows.len() == layout.windows.iter().filter(|w| w.is_none()).count(),
                "BF16 prefix full snapshot position/layout/bytes/layer mismatch"
            );
            let need = checked_bytes(full.pos, re, 1)?;
            ensure!(
                full.rows
                    .iter()
                    .all(|(k, v)| k.len() == need && v.len() == need),
                "BF16 prefix full source length mismatch"
            );
        } else {
            ensure!(
                self.holds_full_rows(target),
                "BF16 prefix resident full rows are unavailable"
            );
        }
        stream
            .context()
            .synchronize()
            .map_err(|e| anyhow::anyhow!("prefix restore sync: {e:?}"))?;
        let result = (|| -> Result<()> {
            self.reset();
            if let Some(full) = full {
                for ((li, _), (k, v)) in layout
                    .windows
                    .iter()
                    .enumerate()
                    .filter(|(_, w)| w.is_none())
                    .zip(&full.rows)
                {
                    rows_htod(&stream, &self.layers[li].0, 0, &k[..target * re], re)?;
                    rows_htod(&stream, &self.layers[li].1, 0, &v[..target * re], re)?;
                }
            }
            for ((li, window), (k, v)) in layout
                .windows
                .iter()
                .enumerate()
                .filter_map(|(li, w)| w.map(|w| (li, w)))
                .zip(&sliding.rows)
            {
                rows_htod(&stream, &self.layers[li].0, 0, k, re)?;
                rows_htod(&stream, &self.layers[li].1, 0, v, re)?;
                self.layer_stored[li] = window.min(target);
            }
            for (li, window) in layout.windows.iter().enumerate() {
                if window.is_none() {
                    self.layer_stored[li] = target;
                }
            }
            let ring = self
                .ring
                .as_mut()
                .context("BF16 prefix requires ring metadata")?;
            *ring.host_meta = [
                target as i32,
                sliding.n_rows as i32,
                0,
                sliding.n_rows as i32,
            ];
            let copy = stream.memcpy_htod(&ring.host_meta[..], &mut ring.meta_dev);
            let sync = stream.synchronize();
            copy?;
            sync?;
            ring.full_stored = target;
            ring.s_stored = sliding.n_rows;
            self.current_len = target;
            self.pending_write_pos = target;
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
    use crate::{gemma4::Gemma4Cache, laguna::LagunaKvRing};

    fn values(start: usize, rows: usize, seed: usize) -> Vec<bf16> {
        (start * 1024..(start + rows) * 1024)
            .map(|i| bf16::from_f32(((i * 29 + seed * 17) % 991) as f32 / 113.0 - 4.0))
            .collect()
    }

    fn append(cache: &mut LagunaKvCache, device: &Device, end: usize, seed: usize) -> Result<()> {
        while cache.current_len < end {
            let start = cache.current_len;
            let rows = (end - start).min(64);
            cache.prepare_for_decode(start, start + rows)?;
            for li in 0..2 {
                let k = Tensor::from_vec(
                    values(start, rows, seed + li * 2),
                    (1, rows, 8, 128),
                    device,
                )?;
                let v = Tensor::from_vec(
                    values(start, rows, seed + li * 2 + 1),
                    (1, rows, 8, 128),
                    device,
                )?;
                cache.write_at(li, &k, &v)?;
                device.synchronize()?;
            }
            cache.advance(rows);
        }
        Ok(())
    }

    fn inspect(cache: &LagunaKvCache) -> Result<Vec<u32>> {
        let ring = cache.ring.as_ref().unwrap();
        let stream = ring.dev.cuda_stream();
        let mut out = vec![
            cache.current_len as u32,
            ring.full_stored as u32,
            ring.s_stored as u32,
        ];
        out.extend(
            stream
                .clone_dtoh(&ring.meta_dev)?
                .into_iter()
                .map(|v| v as u32),
        );
        for (li, (k, v)) in cache.layers.iter().enumerate() {
            let stored = cache.layer_stored[li];
            for tensor in [k, v] {
                out.extend(
                    tensor
                        .narrow(1, 0, stored)?
                        .flatten_all()?
                        .to_vec1::<bf16>()?
                        .into_iter()
                        .map(|value| value.to_bits() as u32),
                );
            }
        }
        Ok(out)
    }

    fn check(cache: &LagunaKvCache, seed: usize) -> Result<()> {
        let pos = cache.current_len;
        for (li, (k, v)) in cache.layers.iter().enumerate() {
            let stored = cache.layer_stored[li];
            for (tensor, offset) in [(k, 0), (v, 1)] {
                let actual = tensor
                    .narrow(1, 0, stored)?
                    .flatten_all()?
                    .to_vec1::<bf16>()?;
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    values(pos - stored, stored, seed + li * 2 + offset)
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore]
    fn bf16_prefix_cuda_gate() -> Result<()> {
        anyhow::ensure!(
            std::env::var("NV_LAGUNA_PREFIX_CUDA_TEST").as_deref() == Ok("1"),
            "set NV_LAGUNA_PREFIX_CUDA_TEST=1; no CUDA skip-as-pass"
        );
        let device = Device::new_cuda(0)?;
        let Device::Cuda(dev) = &device else {
            unreachable!()
        };
        let stream = dev.cuda_stream();
        let mut cache = LagunaKvCache {
            layers: [2048, 264]
                .into_iter()
                .map(|cap| {
                    Ok((
                        Tensor::zeros((1, cap, 8, 128), DType::BF16, &device)?,
                        Tensor::zeros((1, cap, 8, 128), DType::BF16, &device)?,
                    ))
                })
                .collect::<Result<_>>()?,
            layer_caps: vec![2048, 264],
            layer_windows: vec![None, Some(8)],
            layer_stored: vec![0, 0],
            n_kv: 8,
            head_dim: 128,
            current_len: 0,
            pending_write_pos: 0,
            max_seq_len: 2048,
            ring: Some(LagunaKvRing {
                dev: dev.clone(),
                meta_dev: stream.alloc_zeros::<i32>(4)?,
                host_meta: Box::new([0; 4]),
                s_stored: 0,
                full_stored: 0,
                s_cap: 264,
                s_window: 8,
                ring_attn: false,
            }),
            gqa_bufs: None,
            prefill_attn: None,
        };
        append(&mut cache, &device, 300, 11)?;
        let mut a = cache.snapshot_sliding()?;
        assert_eq!(a.bytes, cache.snapshot_sliding_bytes()?);
        assert_eq!(a.bytes, 8 * 1024 * 4);
        append(&mut cache, &device, 600, 11)?;
        let mut full = cache.snapshot_full(600)?;
        assert_eq!(full.bytes, 600 * 1024 * 4);
        cache.rewind_to(&a, None)?;
        check(&cache, 11)?;
        cache.reset();
        append(&mut cache, &device, 600, 43)?;
        let before = inspect(&cache)?;
        let value = a.rows[0].1.pop().unwrap();
        assert!(cache.rewind_to(&a, Some(&full)).is_err());
        a.rows[0].1.push(value);
        assert_eq!(inspect(&cache)?, before);
        let value = full.rows[0].1.pop().unwrap();
        assert!(cache.rewind_to(&a, Some(&full)).is_err());
        full.rows[0].1.push(value);
        a.n_rows += 1;
        assert!(cache.rewind_to(&a, Some(&full)).is_err());
        a.n_rows -= 1;
        a.layout.windows[1] = Some(7);
        assert!(cache.rewind_to(&a, Some(&full)).is_err());
        a.layout.windows[1] = Some(8);
        cache.ring.as_mut().unwrap().ring_attn = true;
        assert!(!cache.prefix_supported() && cache.rewind_to(&a, Some(&full)).is_err());
        cache.ring.as_mut().unwrap().ring_attn = false;
        assert_eq!(inspect(&cache)?, before);
        cache.rewind_to(&a, Some(&full))?;
        check(&cache, 11)?;
        assert_eq!(cache.current_len, 300);
        cache.reset();
        cache.rewind_to(&a, Some(&full))?;
        check(&cache, 11)?;
        eprintln!("BF16-PREFIX-CUDA synthetic_cache_fixture=true actual_cache_writer=true checkpoints=300,600 covering_restore=true full_and_sliding_bf16=exact malformed_and_ring_mode=refused_before_mutation reset_restore=true model_quality=false generated_tokens=0");
        Ok(())
    }
}
