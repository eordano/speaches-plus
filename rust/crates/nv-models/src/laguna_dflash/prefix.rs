use anyhow::{Context, Result};
use half::bf16;

use super::DflashCtxCache;
use crate::laguna::prefix::{
    checked_bytes, prefix_stream, rows_dtoh, rows_htod, validate_tensor_rows, PrefixLayout,
};

pub struct CtxSnapshot {
    pub pos: usize,
    pub n_rows: usize,
    pub bytes: usize,
    layout: PrefixLayout,
    rows: Vec<(Vec<bf16>, Vec<bf16>)>,
}

impl DflashCtxCache {
    fn prefix_layout(&self) -> Result<(PrefixLayout, std::sync::Arc<cudarc::driver::CudaStream>)> {
        let ring = self
            .ring
            .as_ref()
            .context("DFlash prefix refuses legacy context storage")?;
        anyhow::ensure!(
            !ring.k.is_empty()
                && ring.k.len() == ring.v.len()
                && ring.k.len() == self.layers.len()
                && ring.window > 0
                && ring.window <= ring.cap
                && ring.cap <= i32::MAX as usize,
            "DFlash prefix ring layer/window/capacity mismatch"
        );
        let dims = ring.scratch.dims();
        anyhow::ensure!(
            dims.len() == 4 && dims[0] == 1 && dims[1] == ring.window,
            "DFlash prefix scratch layout mismatch"
        );
        let stream = prefix_stream(ring.scratch.device())?;
        let layout = PrefixLayout {
            context: stream.context().clone(),
            capacity: ring.cap,
            n_kv: dims[2],
            head_dim: dims[3],
            caps: vec![ring.cap; ring.k.len()],
            windows: vec![Some(ring.window); ring.k.len()],
        };
        let re = layout.row_elems()?;
        anyhow::ensure!(re == ring.row_elems, "DFlash prefix ring row size mismatch");
        validate_tensor_rows(&stream, &ring.scratch, 0, ring.window, re)?;
        for (k, v) in ring.k.iter().zip(&ring.v) {
            anyhow::ensure!(
                k.dims() == [1, ring.cap, layout.n_kv, layout.head_dim] && v.dims() == k.dims(),
                "DFlash prefix KV capacity mismatch"
            );
            validate_tensor_rows(&stream, k, 0, ring.cap, re)?;
            validate_tensor_rows(&stream, v, 0, ring.cap, re)?;
        }
        Ok((layout, stream))
    }

    pub fn snapshot_bytes(&self) -> Result<usize> {
        let (layout, _) = self.prefix_layout()?;
        let ring = self
            .ring
            .as_ref()
            .context("DFlash prefix requires a context ring")?;
        anyhow::ensure!(
            self.len > 0
                && self.len <= i32::MAX as usize
                && ring.stored <= ring.cap
                && ring.stored <= self.len
                && ring.stored >= ring.window.min(self.len),
            "DFlash prefix requires an initialized context window"
        );
        layout.sliding_bytes(self.len)
    }

    pub fn snapshot(&self) -> Result<CtxSnapshot> {
        let bytes = self.snapshot_bytes()?;
        let (layout, stream) = self.prefix_layout()?;
        let ring = self
            .ring
            .as_ref()
            .context("DFlash prefix requires a context ring")?;
        let pos = self.len;
        let need = ring.window.min(pos);
        stream
            .context()
            .synchronize()
            .map_err(|e| anyhow::anyhow!("DFlash prefix snapshot sync: {e:?}"))?;
        let mut rows = Vec::with_capacity(ring.k.len());
        for (k, v) in ring.k.iter().zip(&ring.v) {
            rows.push((
                rows_dtoh(&stream, k, ring.stored - need, need, ring.row_elems)?,
                rows_dtoh(&stream, v, ring.stored - need, need, ring.row_elems)?,
            ));
        }
        Ok(CtxSnapshot {
            pos,
            n_rows: need,
            bytes,
            layout,
            rows,
        })
    }

    pub fn validate_snapshot(&self, snapshot: &CtxSnapshot) -> Result<()> {
        let (layout, _) = self.prefix_layout()?;
        let ring = self
            .ring
            .as_ref()
            .context("DFlash prefix requires a context ring")?;
        anyhow::ensure!(
            snapshot.pos > 0
                && snapshot.pos <= i32::MAX as usize
                && layout.matches(&snapshot.layout)
                && snapshot.n_rows == ring.window.min(snapshot.pos)
                && snapshot.rows.len() == ring.k.len()
                && snapshot.bytes == layout.sliding_bytes(snapshot.pos)?,
            "DFlash prefix snapshot position/layout/bytes/layers mismatch"
        );
        let need = checked_bytes(snapshot.n_rows, ring.row_elems, 1)?;
        anyhow::ensure!(
            snapshot
                .rows
                .iter()
                .all(|(k, v)| k.len() == need && v.len() == need),
            "DFlash prefix snapshot source length mismatch"
        );
        Ok(())
    }

    pub fn restore(&mut self, snapshot: &CtxSnapshot) -> Result<()> {
        self.validate_snapshot(snapshot)?;
        let (_, stream) = self.prefix_layout()?;
        stream
            .context()
            .synchronize()
            .map_err(|e| anyhow::anyhow!("DFlash prefix restore sync: {e:?}"))?;
        let result = (|| -> Result<()> {
            self.reset();
            let ring = self
                .ring
                .as_mut()
                .context("DFlash prefix requires a context ring")?;
            for ((k, v), (hk, hv)) in ring.k.iter().zip(&ring.v).zip(&snapshot.rows) {
                rows_htod(&stream, k, 0, hk, ring.row_elems)?;
                rows_htod(&stream, v, 0, hv, ring.row_elems)?;
            }
            ring.stored = snapshot.n_rows;
            self.len = snapshot.pos;
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
    use crate::laguna_dflash::{copy_rows_bf16, DflashCtxRing};
    use candle_core::{DType, Device, Tensor};

    fn values(pos: usize, seed: usize) -> Vec<bf16> {
        ((pos - 16) * 1024..pos * 1024)
            .map(|i| bf16::from_f32(((i * 13 + seed * 17) % 997) as f32 / 117.0 - 4.0))
            .collect()
    }

    fn fill(cache: &mut DflashCtxCache, pos: usize, seed: usize) -> Result<()> {
        let ring = cache.ring.as_mut().unwrap();
        let stream = prefix_stream(ring.scratch.device())?;
        for layer in 0..2 {
            for (dst, offset) in [(&ring.k[layer], 0), (&ring.v[layer], 1)] {
                let source = Tensor::from_vec(
                    values(pos, seed + layer * 2 + offset),
                    (1, 16, 8, 128),
                    dst.device(),
                )?;
                copy_rows_bf16(&stream, &source, 0, dst, 0, 16, 1024)?;
                stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
            }
        }
        ring.stored = 16;
        cache.len = pos;
        Ok(())
    }

    fn inspect(cache: &DflashCtxCache) -> Result<Vec<u16>> {
        let ring = cache.ring.as_ref().unwrap();
        let mut data = vec![cache.len as u16, ring.stored as u16];
        for tensor in ring.k.iter().chain(&ring.v) {
            data.extend(
                tensor
                    .narrow(1, 0, 16)?
                    .flatten_all()?
                    .to_vec1::<bf16>()?
                    .into_iter()
                    .map(bf16::to_bits),
            );
        }
        Ok(data)
    }

    fn check(cache: &DflashCtxCache, pos: usize, seed: usize) -> Result<()> {
        let ring = cache.ring.as_ref().unwrap();
        assert_eq!(cache.len, pos);
        assert_eq!(ring.stored, 16);
        for layer in 0..2 {
            for (tensor, offset) in [(&ring.k[layer], 0), (&ring.v[layer], 1)] {
                assert_eq!(
                    tensor
                        .narrow(1, 0, 16)?
                        .flatten_all()?
                        .to_vec1::<bf16>()?
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    values(pos, seed + layer * 2 + offset)
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
    fn dflash_prefix_cuda_gate() -> Result<()> {
        anyhow::ensure!(
            std::env::var("NV_LAGUNA_PREFIX_CUDA_TEST").as_deref() == Ok("1"),
            "set NV_LAGUNA_PREFIX_CUDA_TEST=1; no CUDA skip-as-pass"
        );
        let device = Device::new_cuda(0)?;
        let mut cache = DflashCtxCache::new(2);
        assert!(cache.snapshot_bytes().is_err());
        cache.ring = Some(DflashCtxRing {
            k: (0..2)
                .map(|_| Tensor::zeros((1, 21, 8, 128), DType::BF16, &device))
                .collect::<candle_core::Result<_>>()?,
            v: (0..2)
                .map(|_| Tensor::zeros((1, 21, 8, 128), DType::BF16, &device))
                .collect::<candle_core::Result<_>>()?,
            scratch: Tensor::zeros((1, 16, 8, 128), DType::BF16, &device)?,
            stored: 0,
            cap: 21,
            window: 16,
            row_elems: 1024,
        });
        assert!(cache.snapshot_bytes().is_err());
        fill(&mut cache, 1001, 11)?;
        let mut a = cache.snapshot()?;
        assert_eq!(cache.snapshot_bytes()?, 2 * 2 * 16 * 1024 * 2);
        assert_eq!(cache.snapshot_bytes()?, a.bytes);
        fill(&mut cache, 1041, 37)?;
        let b = cache.snapshot()?;
        let before = inspect(&cache)?;
        let removed = a.rows[1].1.pop().unwrap();
        assert!(cache.restore(&a).is_err());
        a.rows[1].1.push(removed);
        assert_eq!(inspect(&cache)?, before);
        a.n_rows -= 1;
        assert!(cache.restore(&a).is_err());
        a.n_rows += 1;
        a.layout.caps[0] -= 1;
        assert!(cache.restore(&a).is_err());
        a.layout.caps[0] += 1;
        assert_eq!(inspect(&cache)?, before);
        cache.restore(&a)?;
        check(&cache, 1001, 11)?;
        cache.reset();
        cache.restore(&b)?;
        check(&cache, 1041, 37)?;
        eprintln!("DFLASH-PREFIX-CUDA context_window_bf16=exact checkpoints=1001,1041 later_layer_malformed=refused_before_mutation reset_restore=true draft_weights=not_loaded generated_tokens=0 model_quality=false");
        Ok(())
    }
}
